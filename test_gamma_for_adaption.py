import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.EntropyModulator import TentNormModulator
from models.segmodel import SegModel
import imageio
import os
from pathlib import Path
from monai.metrics import DiceMetric

from functions.evaluation import extract_last_ckpt, get_patients_from_df, patients_to_volumes, extract_dice

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.diffusion import Model, get_timestep_embedding
import sys
import cv2
import os
from datasets import get_dataset, data_transform, inverse_data_transform
from runners.diffusion import Diffusion
import imageio
from tqdm.notebook import tqdm
from pathlib import Path
import yaml
from models.diffusion import Model
from torchmetrics.classification import MulticlassCalibrationError

import argparse




# Function to generate a uniform gray image
def generate_grey_image(dim, gray_value=127, device='cpu'):
    gray_image = torch.ones(1, 3, dim, dim) * (gray_value / 255.0)
    return gray_image.to(device)

# Function to compute alpha for given beta and timestep t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def gamma_steps(x, pipi, seq, model, b, theta_0=1, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
	
    nlmed = cv2.fastNlMeansDenoising(((x.cpu().numpy()[0][0] + 1) / 2 * 255).astype(np.uint8), None, h=10, templateWindowSize=7, searchWindowSize=21)
    nlmed = np.expand_dims(nlmed, 0)
    nlmed = np.expand_dims(nlmed, 0).astype(np.float32)
    nlmed = (nlmed / 127.5) - 1
    
    x_svdn = torch.from_numpy(nlmed).to(x.device)
    #x_svdn = x
    x0_preds = []
    betas = b
    a = (1 - b).cumprod(dim=0)
    k = (b / a)/theta_0**2
    theta = (a.sqrt()*theta_0)
    #print(theta_0)
    k_bar = k.cumsum(dim=0)
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long()).to(x.device)
            atm1 = compute_alpha(betas, next_t.long()).to(x.device)
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)
            #print(t)
            
            output = model(x, t.float())
            #output = compiled_model({0: x.numpy(), 1: t.float().numpy()})
            #output = output[0]
            e = output
            x0_from_e = (1.0 / at).sqrt() * (x - (1.0 / at - 1).sqrt() * e)
            #x0_from_e = (1.0 / at).sqrt() * (x - (1 - at) / (1 - at).sqrt() * e)
            x0_from_e_unclamped = x0_from_e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
    
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)
    
            mean = mean_eps
            sample = mean
            gprev = ( sample + x_svdn) / 2
            g = gprev
            start = True
            ll = 10
            itercount = 0
            while (((g - gprev).abs().sum() > 0.001) or start) and itercount < 100:
                itercount += 1
				
                gprev = g
                gprev_svdn = gprev#svdn_transform(gprev, 40)
                start = False
                g = gprev - ( 1 - (x_svdn - gprev_svdn).exp() +  ll * (gprev - sample) ) / ((x_svdn - gprev_svdn).exp() + ll)
            xs.append(g)
            
    return xs, x0_preds


def sample_gamma(model, device, dim, gray_image, pipi, samples = 40, theta_0=0.001):
    seq = [i for i in range(1, samples, 10)]
    b = torch.linspace(0.0001, 0.02, steps=1000)[:samples]
    x = gray_image.to(device)
    xs, x_0 = gamma_steps(x,pipi, seq, model, b.to(device), theta_0=theta_0)
    img = x_0[-1][0]
    return x, (( img) ).permute(1, 2, 0).to('cpu')

def normalize_img(img):
    mmin = img.min()
    mmax = img.max()
    return (img - mmin) / (mmax - mmin)

class Namespace:
    def __init__(self, dictionary):
        self.dict = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = dict2namespace(value)  # Convert nested dict to Namespace
            setattr(self, key, value)
    def to_dict(self):
        return self.dict
def dict2namespace(config_dict):
    return Namespace(config_dict)



def main(samples:int=100, num_iter:int=100, emb_dim:int=16):
    csv_path = Path.home() / "repos/eye-screen/data_splits/20250109/Cirrus_dataset.csv"
    patients = get_patients_from_df(pd.read_csv(csv_path))
    X,y = patients_to_volumes(patients)

    with open(os.path.join("configs", "highres_gamma-32.yml"), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # Initialize model using the loaded configuration
    model = Model(new_config)

    checkpoint_path = "ckpt.pth"
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint[0]
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model.load_state_dict(new_state_dict)

    theta_0=0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_diff = []
    for patient in X:
        tmp = []
        for test_img in patient:
            gray_image = np.expand_dims(test_img / 255., 0)
            gray_image = np.expand_dims(gray_image, 0)
            gray_image = normalize_img(gray_image)

            gray_image = torch.tensor(gray_image*2 - 1, dtype=torch.float32)

            seq = [i for i in range(1, samples, 10)]
            b = torch.linspace(0.0001, 0.02, steps=1000)[:samples]
            x = gray_image.to(device)
            xs, x_0 = gamma_steps(x,None, seq, model, b.to(device), theta_0=theta_0)

            tmp.append([_x[0,0].to('cpu').numpy() for _x in x_0])
        X_diff.append(np.array(tmp))

    ckpt_dir = Path(Path.home() / "data/results/iscreen/da/2025-05-22_13-17-45_ms_unet_high_res_trained_on_spectralis")
    seq = [i for i in range(1, samples, 10)]
    dice = DiceMetric(include_background=False, num_classes=4, ignore_empty=False)
    mce = MulticlassCalibrationError(4)

    model = SegModel.load_from_checkpoint(extract_last_ckpt(ckpt_dir), map_location="cpu").eval()
    modulator = TentNormModulator(model, emb_dim, 4, lr=1e-5).to(device)
    seg_embeddings = get_timestep_embedding(torch.from_numpy(np.array(seq)), emb_dim).to(device)

    for iter in tqdm(range(num_iter)):
        pat_idx = np.random.choice(len(X_diff))
        patient = X_diff[pat_idx]
        slidx = np.random.choice(patient.shape[0], 4)

        x_inp = torch.from_numpy(patient[slidx][:, np.newaxis].transpose((2,0,1,3,4)) * 0.5 + 0.5).to(device)
        
        _ = modulator.forward_and_adapt_list(x_inp, seg_embeddings)

    dices = []

    present = []
    mces = []

    for patient in tqdm(range(0, len(X_diff))):
        ys = []
        yhats = []
        with torch.no_grad():
            y_hat = []
            for _slice in range(X_diff[patient].shape[0]):
                tmp = 0
                for i, instance in enumerate(range(X_diff[patient].shape[1])):
                    tmp += torch.softmax(modulator.forward(torch.from_numpy(X_diff[patient][_slice:_slice+1,instance:instance + 1].astype(np.float32) * 0.5 + 0.5).to(device),
                                                        seg_embeddings[i:i+1]), dim=1)[0].detach().cpu().numpy()
                y_hat.append(tmp / X_diff[patient].shape[1])
        yhats = np.array(y_hat)
        ys = y[patient][np.newaxis, np.newaxis]
        present.append([(ys == i).sum() > 0. for i in range(1,4)])

        yhats = torch.from_numpy(yhats)
        ys = torch.from_numpy(ys)

        pred_one_hot = torch.nn.functional.one_hot(torch.argmax(yhats, dim=1), num_classes=4)[None]
        test = pred_one_hot.permute((0,4,1,2,3))

        dices.append(dice(test, ys).detach().cpu().numpy()[0])
        mces.append(mce(yhats.permute((1,0,2,3))[None], ys[0]))
        print(dices[-1])

    print(extract_dice(np.array(dices), np.array(present), return_std=True), np.mean(extract_dice(np.array(dices), np.array(present))))
    print("Finished run!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--samples', type=int, default=100, help='Number of samples for gamma steps')
    argparser.add_argument('--num_iter', type=int, default=100, help='Number of adaptation iterations')
    argparser.add_argument('--emb_dim', type=int, default=16, help='Embedding dimension for modulator')
    args = argparser.parse_args()
    main(args.samples, args.num_iter, args.emb_dim)