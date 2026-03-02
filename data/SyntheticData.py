import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticData(Dataset):
    def __init__(self, n_samples=100, dim_mode=5, noise_std=1.0,
                seed=None, drift_type='none', drift_magnitude='weak',
                signal_type='simple_additive', feature_dist='normal',
                rho1=0.7, rho2=0.5, rho3=0.7):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.n_samples = n_samples
        self.n_features = dim_mode
        
        # Store correlation params
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3

        # Generate Features
        self.signal_type = signal_type
        self.x = self._generate_features(feature_dist)

        # Drift parameters
        self.coef_meaningful_1 = 3.0 
        self.coef_meaningful_2 = 2.0 
        self.noise_mean = 0.0
        
        # Apply Drifts
        self._apply_drift(drift_type, drift_magnitude)

        # Generate Target Signal
        signal = self._generate_signal(signal_type)
        
        # Add Noise
        epsilon = torch.randn(n_samples) * noise_std
        
        self.y = signal + epsilon
        self.true_noise_var = noise_std ** 2 
        self.len = self.n_samples

    def _generate_features(self, dist_type):
        if dist_type == 'normal':
            return torch.randn(self.n_samples, self.n_features)
        
        elif dist_type == 'correlated':
            mean = np.zeros(self.n_features)
            cov = np.eye(self.n_features)

            # Use dynamic rhos passed from config
            r1, r2, r3 = self.rho1, self.rho2, self.rho3
            
            signal_type = self.signal_type

            if signal_type != 'mixed':
                cov[0:3, 0:3] = r1
                if self.n_features >= 5:
                    cov[3:5, 3:5] = r3
                    cov[0:3, 3:5] = r2
                    cov[3:5, 0:3] = r2
            else:
                cov[0:4, 0:4] = r1
                if self.n_features >= 5:
                    cov[4:6, 4:6] = r3
                    cov[0:4, 4:6] = r2
                    cov[4:6, 0:4] = r2

            # Reset diagonal to variance
            np.fill_diagonal(cov, 1.0)
            
            # check for positive semi definiteness
            try:
                X = np.random.multivariate_normal(mean, cov, self.n_samples)
            except np.linalg.LinAlgError:
                print("Warning: Matrix not positive definite, adding jitter.")
                cov += np.eye(self.n_features) * 1e-4
                X = np.random.multivariate_normal(mean, cov, self.n_samples)
                
            return torch.tensor(X, dtype=torch.float32)         
        else:
            raise ValueError(f"Unknown feature_dist: {dist_type}")

    def _generate_signal(self, signal_type):
        if signal_type == 'simple_additive':
            # Linear
            return (self.coef_meaningful_1 * self.x[:, 0] + 
                    self.coef_meaningful_2 * self.x[:, 1] - 
                    self.coef_meaningful_1 * self.x[:, 2])
        
        elif signal_type == 'smooth_qubic':
            # Polynomial
            return (self.coef_meaningful_1 * self.x[:, 0]**2 + 
                    self.coef_meaningful_2 * 0.5 * self.x[:, 1]**3 - 
                    self.coef_meaningful_1 * self.x[:, 2]**2)

        elif signal_type == 'high_freq':
            # Sine
            amp_sine = 0.5 * (self.coef_meaningful_1 + self.coef_meaningful_2)
            amp_linear = (self.coef_meaningful_1 + self.coef_meaningful_2)
            return (amp_sine * torch.sin(self.x[:, 0]) + 
                    amp_linear * self.x[:, 1] + 
                    amp_sine * torch.cos(self.x[:, 2]))
        
        elif signal_type == 'step':
            # Discontinuous
            amp = (self.coef_meaningful_1 + self.coef_meaningful_2)
            return (amp * torch.sign(torch.sin(2.5 * self.x[:, 0])) + 
                    amp * torch.sign(torch.sin(2.5 * self.x[:, 1])) - 
                    amp * torch.sign(torch.sin(2.5 * self.x[:, 2])))
        
        elif signal_type == 'mixed':
            return (self.coef_meaningful_1 * self.x[:, 0] + 
                    self.coef_meaningful_2 * self.x[:, 1]**2 + 
                    self.coef_meaningful_1 * torch.sin(5 * self.x[:, 2]) +
                    self.coef_meaningful_2 * torch.sign(torch.sin(self.x[:, 3])))
        
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

    def _apply_drift(self, drift_type, drift_magnitude):
        if drift_type == 'meaningful':
            factor = 1.5 if drift_magnitude == 'strong' else 1.2
            self.coef_meaningful_1 *= factor
            self.coef_meaningful_2 *= factor
            
        elif drift_type == 'noise':
            shift = 5.0 if drift_magnitude == 'strong' else 2.0
            self.noise_mean = shift
            if self.n_features > 3:
                self.x[:, 3:] += self.noise_mean

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len