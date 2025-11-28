import imageio
import numpy as np
import os
import torch
import logging

from pathlib import Path
from monai.metrics import DiceMetric


def get_patients_from_folder(path):
    patients = {}
    for _file in (path / "Input").iterdir():
        pat_id = int(Path(_file.name).name.split("_")[1])
        if pat_id not in patients:
            patients[pat_id] = [(_file, path / "Label" / _file.name)]
        else:
            patients[pat_id].append((_file, path / "Label" / _file.name))

    return patients

def get_patients_from_df(df):
    patients = {}
    for key, row in df.iterrows():
        pat_id = int(Path(row.Label).name.split("_")[1])
        if pat_id not in patients:
            patients[pat_id] = [(row.Input, row.Label)]
        else:
            patients[pat_id].append((row.Input, row.Label))

    return patients


def patients_to_volumes(patients):
    sorting_func = lambda x: int(Path(x[0]).name.split(".")[0].split("_")[-1])
    X = []
    y = []
    for pat_id, pat_list in patients.items():
        X.append(np.array([imageio.imread(img_path) for (img_path, lbl_path) in sorted(pat_list, key=sorting_func)]))
        y.append(np.array([imageio.imread(lbl_path) for (img_path, lbl_path) in sorted(pat_list, key=sorting_func)]))
    return X, y


def predict_patients(model, X, y, device="cuda", num_classes=4):
    model = model.to(device).eval()
    dices = []
    dice = DiceMetric(include_background=False, num_classes=num_classes, ignore_empty=False)
    predictions = []
    present = []

    for patient in range(0,len(X)):
        xs = []
        ys = []
        with torch.no_grad():
            y_hat = []
            for _slice in range(len(X[patient])):
                y_hat.append(model(torch.from_numpy(X[patient][_slice:_slice+1,np.newaxis].astype(np.float32) / 255.).to(device))[0].detach().cpu().numpy())

        yhats = np.array(y_hat)

        ys = y[patient][np.newaxis, np.newaxis]
        present.append([(ys == i).sum() > 0. for i in range(1,4)])
    
        yhats = torch.from_numpy(yhats)
        ys = torch.from_numpy(ys)
    
        pred_one_hot = torch.nn.functional.one_hot(torch.argmax(yhats, dim=1), num_classes=num_classes)[None]
        test = pred_one_hot.permute((0,4,1,2,3))

        dices.append(dice(test, ys).detach().cpu().numpy()[0])
        predictions.append(test.detach().cpu().numpy()[0])
        
    return np.array(dices), predictions, np.array(present)


def extract_dice(dice, avail, return_std=False):
    cl_dices = []
    cl_dices_std = []
    for i in range(dice.shape[1]):
        cl_dices.append((dice[:,i][avail[:,i]]).mean())
        if return_std:
            cl_dices_std.append((dice[:,i][avail[:,i]]).std(ddof=1))
    if return_std:
        return cl_dices, cl_dices_std
    return cl_dices