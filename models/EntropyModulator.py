import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .NormModulator import NormModulator


class TentNormModulator(nn.Module):
    """
    Test-time adaptation using TENT on top of a frozen base model,
    where only normalization modulation parameters are updated.
    """
    def __init__(self, base_model, embedding_dim, num_classes, lr=1e-3):
        super().__init__()
        self.base_model = base_model.eval()  # freeze base model weights
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Set up normalization modulator
        self.norm_modulator = NormModulator(base_model, embedding_dim)

        # Simple classifier head (assumed part of model)
        self.num_classes = num_classes

        # Collect parameters to optimize (only modulator)
        self.optimizer = optim.Adam(self.norm_modulator.parameters(), lr=lr)

    def forward(self, x, embedding):
        return self.norm_modulator(x, embedding)

    def forward_and_adapt_list(self, x_list, embedding_list, only_last=False):
        """
        Forward pass + TENT adaptation
        """
        self.train()  # Enable gradients for modulator
        self.base_model = self.base_model.eval()
        entropy = 0
        outputs = 0

        if only_last:
            _x = self.forward(x_list[-1], embedding_list[-1:])
    
            # Entropy loss
            entropy -= (_x.softmax(1) * _x.log_softmax(1)).sum()
            outputs += _x.detach().softmax(1)
        else:
            for i, x in enumerate(x_list):
                _x = self.forward(x, embedding_list[i:i+1])
        
                # Entropy loss
                entropy -= (_x.softmax(1) * _x.log_softmax(1)).sum()
                outputs += _x.detach().softmax(1)

        # Optimize modulator params only
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()

        # Return adapted outputs
        return (outputs / len(x_list))