import torch
from typing import Dict
import lightning

import segmentation_models_pytorch as smp
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from . import constants as C
from monai.metrics import DiceMetric

import logging

ch = logging.StreamHandler()
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        ch,
                    ])

def dice_loss_multiclass(pred, target, num_classes, epsilon=1e-6):
    """
    Computes the Dice loss for multi-class segmentation tasks.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape [batch_size, num_classes, H, W].
        target (torch.Tensor): Ground truth tensor of shape [batch_size, num_classes, H, W].
        num_classes (int): Number of classes (channels in the segmentation map).
        epsilon (float): Small value to avoid division by zero.
        
    Returns:
        torch.Tensor: The average Dice loss over all classes.
    """
    dice_losses = []
    
    # Loop over each class
    for i in range(1, num_classes):
        pred_class = pred[:, i, :, :]  # Prediction for class i
        target_class = target[:, i, :, :]  # Ground truth for class i
        
        # Compute intersection and union for class i
        intersection = torch.sum(pred_class * target_class, axis=(1,2))
        union = torch.sum(pred_class, axis=(1,2)) + torch.sum(target_class, axis=(1,2))
        
        # Dice score for class i
        dice_score = (2. * intersection + epsilon) / (union + epsilon)
        
        # Dice loss for class i
        dice_loss_value = 1 - dice_score
        dice_losses.append(dice_loss_value)

    
    # Return average Dice loss over all classes
    # Neglect the background class
    return torch.mean(torch.stack(dice_losses))

class SegModel(lightning.LightningModule):

    def __init__(self, **config: Dict):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.save_hyperparameters()
        
        logging.info(f"Model config: {config}")
        self.num_classes = config.get("num_classes", 4)

        model_name = config.get('model_name', C.MODEL_UNETPLUSPLUS)
        
        if model_name == C.MODEL_UNETPLUSPLUS:
            self.network = smp.UnetPlusPlus(
                encoder_name='resnet18',
                encoder_weights="imagenet",
                in_channels=1,
                classes=self.num_classes
            )
        elif model_name == C.MODEL_UNET:
            self.network = smp.Unet(
                encoder_name='resnet18',
                encoder_weights=None,
                in_channels=1,
                classes=self.num_classes
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

        self.validation_metrics = {
            "dice0": [],
            "dice1": [],
            "dice2": [],
            "dice": [],
            "dice_loss": [],
            "ce": []
        }
        self.validation_step_outputs = []
        self.validation_step_segmentation_outputs = []

        self.dice_loss = lambda y_hat, y: dice_loss_multiclass(torch.nn.functional.softmax(y_hat, dim=1), y, self.num_classes)
        self.dice_metric = DiceMetric(include_background=False, num_classes=self.num_classes, ignore_empty=False)
        self.bce = torch.nn.CrossEntropyLoss()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                eta_min=1e-6,
                T_max=self.config.num_iter
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        return self.network.predict(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.MASK]
        y_hat = self.network(x)

        loss = self.dice_loss(y_hat, y) + self.bce(y_hat, y.float())
        self.log("loss/train", loss.detach())

        step = self.global_step
        with torch.no_grad():
            if step % self.config.num_log == 0 or step == self.config.num_iter - 1:
                self.logger.experiment.add_image(
                    f'train-OCT',
                    make_grid(
                        torch.concat([
                        x[:, 0:1,  :, :].detach(),
                        torch.nn.functional.softmax(y_hat, dim=1)[:, 1:2].detach(),
                        y[:, 1:2,  :, :].detach()
                        ], dim=-1),
                        nrow=4, normalize=True
                    ),
                    global_step=step
                )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[C.IMAGE], batch[C.MASK]


        with torch.no_grad():
            if len(x.shape) == 5:   # slice wise computing
                x_perm = x.permute(0, 2, 1, 3, 4)  
                y_perm = y.permute(0, 2, 1, 3, 4)  

                A, _C, B, D, E = x_perm.shape
                x = x_perm.reshape(A * _C, B, D, E)
                A, _C, B, D, E = y_perm.shape
                y = y_perm.reshape(A * _C, B, D, E)

            y_hat = self.network(x)
            pred = torch.argmax(y_hat, dim=1, keepdim=True)

            pred_one_hot = torch.nn.functional.one_hot(pred[:,0], self.num_classes).permute((0,3,1,2))

            dice = self.dice_metric(pred_one_hot[None].permute(0,2,1,3,4), y[None].permute(0,2,1,3,4)).mean(axis=0)
            dice_loss = self.dice_loss(y_hat, y)
            bce_loss = self.bce(y_hat, y.float())
            for i in range(len(dice)):
                self.validation_metrics[f"dice{i}"].append(
                    dice[i]
                    )
            self.validation_metrics[f"dice"].append(
                dice.mean()
                )
            self.validation_metrics[f"dice_loss"].append(
                dice_loss
                )
            self.validation_metrics["ce"].append(
                    bce_loss
                )
            if batch_idx < 15:
                self.validation_step_outputs.append(
                    torch.concat([
                        x[x.shape[0] // 2, 0:1,  :, :],
                        pred[x.shape[0] // 2, 0:1],
                        torch.argmax(y[x.shape[0] // 2, :, :],dim=0, keepdim=True)
                    ], dim=-1)
                )

    def on_validation_epoch_end(self):
        for k, v in self.validation_metrics.items():
            if len(v) > 0:
                self.log(f"{k}/val", sum(v) / len(v), sync_dist=True)
            v.clear()

        diffs = torch.stack(self.validation_step_outputs, dim=0)
        self.logger.experiment.add_image(
            'val-OCT',
            make_grid(diffs, nrow=2, normalize=True),
            global_step=self.global_step
        )
        self.validation_step_outputs.clear()
