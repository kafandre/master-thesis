import torch
from torch.utils.data import Dataset

# Creating the dataset
class Data(Dataset):
    def __init__(self, data_amount, seed=None):

        if seed is not None:
            torch.manual_seed(seed)
            
        self.data_amount = data_amount
        # Using many more features (20 instead of 5)
        self.x = torch.zeros(data_amount, 20)
        
        # Generate features - fewer samples, more features
        # Core predictive features (similar to original)
        self.x[:, 0] = torch.randn(data_amount) * 1 + -1
        self.x[:, 1] = torch.randn(data_amount) * 3 + 5
        self.x[:, 2] = torch.randn(data_amount) * 0.5 + 0
        
        # Highly noisy features that will cause overfitting
        self.x[:, 3] = torch.randn(data_amount) * 10  # High variance
        self.x[:, 4] = torch.randn(data_amount) * 8 - 4
        
        # Correlated features (variations of the predictive ones)
        self.x[:, 5] = self.x[:, 0] * 0.9 + torch.randn(data_amount) * 0.3  # Correlated with x0
        self.x[:, 6] = self.x[:, 1] * 1.1 - torch.randn(data_amount) * 0.2  # Correlated with x1
        
        # Add some outliers to specific samples
        if data_amount > 10:
            outlier_indices = torch.randint(0, data_amount, (data_amount // 10,))
            self.x[outlier_indices, 7] = torch.randn(len(outlier_indices)) * 20  # Extreme values
        else:
            self.x[:, 7] = torch.randn(data_amount) * 5
            
        # Irrelevant features that will tempt the model to find spurious patterns
        for i in range(8, 20):
            self.x[:, i] = torch.randn(data_amount) * (i % 5 + 0.5)
            
        # Generate target values with complex, sporadic relationships
        # Only a few features actually matter, but with occasional interactions
        self.y = (-5 +
                  torch.mul(self.x[:, 0], 2) +
                  torch.mul(self.x[:, 1], 1) +
                  torch.mul(self.x[:, 2], 3.5) +
                  # Add some interactions that will be hard to generalize
                  torch.mul(self.x[:, 0] * self.x[:, 1], 0.3) +
                  # Add a non-linear transformation that the model might overfit to
                  torch.mul(torch.sin(self.x[:, 3]), 1.5) +
                  # Add small effects from noise features to tempt overfitting
                  torch.mul(self.x[:, 10], 0.1) +
                  torch.mul(self.x[:, 15], 0.2) +
                  # Very high noise level relative to signal
                  torch.randn(data_amount) * 5)
        
        self.len = self.x.shape[0]  # Number of samples
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return self.len