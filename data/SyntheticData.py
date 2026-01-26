import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticData(Dataset):
    def __init__(self, n_samples=100, dim_mode=5, noise_std=1.0,
                seed=None, drift_type='none', drift_magnitude='weak',
                signal_type='linear_interaction', feature_dist='normal', noise_dist='normal'):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.n_samples = n_samples
        self.n_features = dim_mode
        
        # 1. Generate Features
        self.x = self._generate_features(feature_dist)
        
        # --- Drift Parameters ---
        self.coef_meaningful = 3.0
        self.coef_interaction = 2.0
        self.noise_mean = 0.0
        
        # Apply Drifts (Modifies coefficients or feature shifts)
        self._apply_drift(drift_type, drift_magnitude)

        # 2. Generate Target Signal
        signal = self._generate_signal(signal_type)
        
        # 3. Add Noise
        if noise_dist == 'student_t':
            # Heavy-tailed noise (df=3)
            # Standard t(3) has variance 3. We scale it by noise_std.
            t_noise = torch.tensor(np.random.standard_t(df=3, size=n_samples), dtype=torch.float32)
            epsilon = t_noise * noise_std
        else:
            # Standard Normal Noise
            epsilon = torch.randn(n_samples) * noise_std
        
        self.y = signal + epsilon
        self.true_noise_var = noise_std ** 2 
        self.len = self.n_samples

    def _generate_features(self, dist_type):
        if dist_type == 'normal':
            return torch.randn(self.n_samples, self.n_features)
        
        elif dist_type == 'student_t':
            # Student-t features (df=3) for Outlier_Features scenario
            return torch.tensor(np.random.standard_t(df=3, size=(self.n_samples, self.n_features)), dtype=torch.float32)
        
        elif dist_type == 'correlated':
            # Generate features with Multicollinearity
            # Create a random correlation matrix
            A = np.random.randn(self.n_features, self.n_features)
            Cov = np.dot(A.T, A)
            
            # Normalize to correlation matrix (diagonal 1) to keep scale similar to normal
            d = np.sqrt(np.diag(Cov))
            Cov = Cov / np.outer(d, d)
            
            # Generate X ~ N(0, Cov)
            X = np.random.multivariate_normal(np.zeros(self.n_features), Cov, self.n_samples)
            return torch.tensor(X, dtype=torch.float32)
            
        else:
            raise ValueError(f"Unknown feature_dist: {dist_type}")

    def _generate_signal(self, signal_type):
        if signal_type == 'linear_interaction':
            # Original signal: y = 3x0 + 3x1 - 3x2 + 2x0x1
            return (self.coef_meaningful * self.x[:, 0] + 
                    self.coef_meaningful * self.x[:, 1] - 
                    self.coef_meaningful * self.x[:, 2] +
                    self.coef_interaction * self.x[:, 0] * self.x[:, 1])
        
        elif signal_type == 'friedman':
            # Friedman #1: y = 10sin(pi*x0*x1) + 20(x2 - 0.5)^2 + 10x3 + 5x4
            # We apply this to the input features (usually N(0,1))
            return (10 * torch.sin(np.pi * self.x[:, 0] * self.x[:, 1]) +
                    20 * (self.x[:, 2] - 0.5)**2 +
                    10 * self.x[:, 3] +
                    5 * self.x[:, 4])
        
        elif signal_type == 'step':
            # Discontinuous step function
            # 5 if x > 0 else 0 (Centered)
            return (5.0 * (self.x[:, 0] > 0).float() + 
                    5.0 * (self.x[:, 1] > 0).float() - 
                    5.0 * (self.x[:, 2] > 0).float() +
                    3.0 * (self.x[:, 0] > 0).float() * (self.x[:, 1] > 0).float()) # Interaction step
            
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

    def _apply_drift(self, drift_type, drift_magnitude):
        if drift_type == 'meaningful':
            # Concept Drift: The rule P(Y|X) changes
            factor = 1.5 if drift_magnitude == 'strong' else 1.2
            self.coef_meaningful *= factor
            self.coef_interaction *= factor
            
        elif drift_type == 'noise':
            # Covariate Shift on Noise: P(X_noise) changes
            shift = 5.0 if drift_magnitude == 'strong' else 2.0
            self.noise_mean = shift
            
            # Apply shift to noise features (indices 3 onwards)
            if self.n_features > 3:
                self.x[:, 3:] += self.noise_mean

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len