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
        
        # Load Data Logic
        if self.dataset_name == 'diabetes':
            path = os.path.join(root_dir, 'diabetes', 'diabetes.csv')
            df = pd.read_csv(path)
            # Target is 'Outcome'
            X_raw = df.drop(columns=['Outcome']).values
            y_raw = df['Outcome'].values
            
        elif self.dataset_name == 'bodyfat':
            path = os.path.join(root_dir, 'body_fat', 'bodyfat.csv')
            df = pd.read_csv(path)
            # Target is 'BodyFat'
            X_raw = df.drop(columns=['BodyFat']).values
            y_raw = df['BodyFat'].values

        elif self.dataset_name == 'riboflavin':
            path = os.path.join(root_dir, 'riboflavin', 'riboflavin.csv')
            df = pd.read_csv(path)
            # Target is 'target_y'
            X_raw = df.drop(columns=['target_y']).values
            y_raw = df['target_y'].values

        elif self.dataset_name == 'pcr':
            # PCR Data: Space/Tab separated text files
            x_path = os.path.join(root_dir, 'pcr', 'Xgene.txt')
            y_path = os.path.join(root_dir, 'pcr', 'Y3.txt')
            
            # Use 'sep=r"\s+"' to handle variable whitespace
            # added .T since the input matrix is stored transposed
            X_raw = pd.read_csv(x_path, sep=r'\s+', header=None).T.values
            y_raw = pd.read_csv(y_path, sep=r'\s+', header=None).values.flatten()

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Preprocessing: Standardize features (Mean=0, Std=1)
        # This is critical for real data to ensure stable training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Convert to PyTorch Tensors
        self.x = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.float32)
        
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        
        # NOTE: For real data, we don't know the "True Noise Variance".
        # If config.flood_level is None, this default might result in poor auto-calculation.
        # It is recommended to set a fixed flood_level in config.py for real data.
        self.true_noise_var = 1.0 

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]