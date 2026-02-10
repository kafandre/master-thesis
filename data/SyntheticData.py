import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticData(Dataset):
    def __init__(self, n_samples=100, dim_mode=5, signal_scale=1.0, noise_std=1.0,
                seed=None, drift_type='none', drift_magnitude='weak',
                signal_type='simple_additive', feature_dist='normal', noise_dist='normal'):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.n_samples = n_samples
        self.n_features = dim_mode
        self.signal_scale = signal_scale

        # 1. Generate Features
        self.x = self._generate_features(feature_dist)
        
        # --- Drift Parameters ---
        self.coef_meaningful_1 = 3.0 * signal_scale
        self.coef_meaningful_2 = 2.0 * signal_scale
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
        
        elif dist_type == 'correlated':
            # Create correlation between x0 and x2 to trigger "Cancellation"
            mean = np.zeros(self.n_features)
            cov = np.eye(self.n_features)

            rho1=0.95
            rho2=0.8
            rho3=0.95

            # Signal Block (Features 0, 1, 2)
            cov[0:3, 0:3] = rho1
            
            # Noise Block (Features 3, 4)
            if self.n_features >= 5:
                cov[3:5, 3:5] = rho3
                
                # Cross-Block Correlation (Signal vs Noise)
                cov[0:3, 3:5] = rho2
                cov[3:5, 0:3] = rho2
            
            # Reset diagonal to 1.0 (Variance)
            np.fill_diagonal(cov, 1.0)
            
            # Check for Positive Semi-Definiteness (Numerical stability)
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
            # The New Baseline
            # y = 3x0 + 2x1^2 - 3x2
            # Perfect for CWB. Tests basic additive fit.
            return (self.coef_meaningful_1 * self.x[:, 0] + 
                    self.coef_meaningful_2 * self.x[:, 1] - 
                    self.coef_meaningful_1 * self.x[:, 2])
        
        elif signal_type == 'baseline_composite':
            # Linear + Quadratic + Sine
            # x0: Linear (Easy)
            # x1: Quadratic (Poly Oracle)
            # x2: Sine (Hard for Poly, requires Spline/Tree)
            return (self.coef_meaningful_1 * self.x[:, 0] + 
                    self.coef_meaningful_2 * self.x[:, 1]**2 + 
                    self.coef_meaningful_1 * torch.sin(3.0 * self.x[:, 2]))
        # --- 2. FAVOR POLY: Smooth Quadratic ---
        # Polynomial (deg=2) fits this perfectly. 

        elif signal_type == 'smooth_qubic':
            return (self.coef_meaningful_1 * self.x[:, 0]**2 + 
                    self.coef_meaningful_2 * 0.5 * self.x[:, 1]**3 - 
                    self.coef_meaningful_1 * self.x[:, 2]**2)

        # --- 4. FAVOR BSPLINES: High Frequency ---
        # y = 10sin(3*pi*x0) + 5x1
        # 3*pi is fast enough that a simple quadratic poly cannot fit it.
        # Requires local basis functions (Splines/Trees).
        elif signal_type == 'high_freq':
            amp_sine = 2.0 * (self.coef_meaningful_1 + self.coef_meaningful_2)
            amp_linear = (self.coef_meaningful_1 + self.coef_meaningful_2)
            return (amp_sine * torch.sin(3.0 * np.pi * self.x[:, 0]) + 
                    amp_linear * self.x[:, 1])
        
        elif signal_type == 'step':
            amp = (self.coef_meaningful_1 + self.coef_meaningful_2)
            return (amp * torch.sign(torch.sin(2.5 * self.x[:, 0])) + 
                    amp * torch.sign(torch.sin(2.5 * self.x[:, 1])) - 
                    amp * torch.sign(torch.sin(2.5 * self.x[:, 2])))
            
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

    def _apply_drift(self, drift_type, drift_magnitude):
        if drift_type == 'meaningful':
            # Concept Drift: The rule P(Y|X) changes
            factor = 1.5 if drift_magnitude == 'strong' else 1.2
            self.coef_meaningful_1 *= factor
            self.coef_meaningful_2 *= factor
            
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