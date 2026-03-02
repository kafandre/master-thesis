import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class RealData(Dataset):
    def __init__(self, dataset_name, root_dir='./data', seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.dataset_name = dataset_name.lower()
        
        if self.dataset_name == 'diabetes':
            from sklearn.datasets import load_diabetes
            X_raw, y_raw = load_diabetes(return_X_y=True)
            
        elif self.dataset_name == 'bodyfat':
            path = os.path.join(root_dir, 'body_fat', 'bodyfat.csv')
            df = pd.read_csv(path)
            # Target is 'BodyFat'
            X_raw = df.drop(columns=['BodyFat', 'Density']).values
            y_raw = df['BodyFat'].values

        elif self.dataset_name == 'riboflavin':
            path = os.path.join(root_dir, 'riboflavin', 'riboflavin.csv')
            df = pd.read_csv(path)
            # Target is 'target_y'
            X_raw = df.drop(columns=['target_y']).values
            y_raw = df['target_y'].values

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.x = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.float32)
        
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.true_noise_var = 1.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]