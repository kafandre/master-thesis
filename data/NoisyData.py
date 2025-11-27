import torch
from torch.utils.data import Dataset
import numpy as np

class NoisyData(Dataset):
    def __init__(self, n_samples=100, dim_mode='low', noise_std=1.0, 
                 seed=None, drift_type='none', drift_magnitude='weak'):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.n_samples = n_samples
        
        # Dimensions
        if dim_mode == 'low':
            self.n_features = 5
        elif dim_mode == 'high':
            self.n_features = 20
        else:
            raise ValueError(f"Unknown dim_mode: {dim_mode}")
            
        self.x = torch.randn(n_samples, self.n_features)
        
        # --- Drift Parameters ---
        self.coef_meaningful = 3.0
        self.coef_interaction = 2.0
        self.noise_mean = 0.0
        
        # Apply Drifts
        if drift_type == 'meaningful':
            # Concept Drift: The rule P(Y|X) changes
            # We change the coefficients of the signal features
            factor = 1.5 if drift_magnitude == 'strong' else 1.2
            self.coef_meaningful *= factor
            self.coef_interaction *= factor
            
        elif drift_type == 'noise':
            # Covariate Shift on Noise: P(X_noise) changes
            # We shift the distribution of the NOISY features
            shift = 5.0 if drift_magnitude == 'strong' else 2.0
            self.noise_mean = shift
            
            # Apply shift to noise features (indices 3 onwards)
            if self.n_features > 3:
                self.x[:, 3:] += self.noise_mean

        # --- Generate Target ---
        # Features 0, 1, 2 are signal. The rest are noise.
        signal = (self.coef_meaningful * self.x[:, 0] + 
                  self.coef_meaningful * self.x[:, 1] - 
                  self.coef_meaningful * self.x[:, 2] +
                  self.coef_interaction * self.x[:, 0] * self.x[:, 1]) 
        
        # Add Noise (controlled by noise_std now)
        epsilon = torch.randn(n_samples) * noise_std
        
        self.y = signal + epsilon
        self.true_noise_var = noise_std ** 2 # For Flooding calculation
        self.len = self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len