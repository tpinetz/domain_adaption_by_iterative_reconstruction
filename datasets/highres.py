import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class HighRes(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.image_paths = self._get_image_paths()
        self.transform = transform

    def _get_image_paths(self):
        input_dir = Path("./highres_np")
        return [item for item in input_dir.iterdir() if item.is_file()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img = np.load(img_path)
        #img = cv2.bilateralFilter(img, d=0, sigmaColor=14, sigmaSpace=14)
        #min_val = np.min(img)
        #max_val = np.max(img)
        #img = (img - min_val) / (max_val - min_val)

        X = np.expand_dims(img, axis=2)
        if self.transform is not None:
            X = self.transform(X)
        return X, 0



class HighResLog(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.image_paths = self._get_image_paths()
        self.transform = transform

    def _get_image_paths(self):
        input_dir = Path("./highres_np")
        return [item for item in input_dir.iterdir() if item.is_file()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img = np.load(img_path)
        #img = cv2.bilateralFilter(img, d=0, sigmaColor=14, sigmaSpace=14)
        #min_val = np.min(img)
        #max_val = np.max(img)
        #img = (img - min_val) / (max_val - min_val)

        X = np.expand_dims(img, axis=2)
        if self.transform is not None:
            X = self.transform(X)
        return X, 0
        
        

class HighResLog2(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.image_paths = self._get_image_paths()
        self.transform = transform

    def _get_image_paths(self):
        input_dir = Path("./highres_np")
        return [item for item in input_dir.iterdir() if item.is_file()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img = np.load(img_path)
        #img = cv2.bilateralFilter(img, d=0, sigmaColor=14, sigmaSpace=14)
        #min_val = np.min(img)
        #max_val = np.max(img)
        #img = (img - min_val) / (max_val - min_val)

        X = np.expand_dims(img, axis=2) * 255
        eps = 0.00000001
        X = np.log1p(X) / np.log1p(255)
        if self.transform is not None:
            X = self.transform(X)
        return X, 0