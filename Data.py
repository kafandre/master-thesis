import torch
from torch.utils.data import Dataset

# Creating the dataset
class Data(Dataset):
    def __init__(self, data_amount, seed=None):

        if seed is not None:
            torch.manual_seed(seed)

        self.data_amount = data_amount
        self.x = torch.zeros(data_amount, 5)

        # Generate features
        self.x[:, 0] = torch.randn(data_amount) * 1 + -1   
        self.x[:, 1] = torch.randn(data_amount) * 3 + 5    
        self.x[:, 2] = torch.randn(data_amount) * 0.5 + 0 
        self.x[:, 3] = torch.rand(data_amount) * 11 + 3
        self.x[:, 4] = torch.rand(data_amount) * 3 - 2
          
        # Generate target values
        self.y =  (-5 +
                  torch.mul(self.x[:, 0], 2) + 
                  torch.mul(self.x[:, 1], 1) + 
                  torch.mul(self.x[:, 2], 3.5) + 
                  torch.mul(self.x[:, 3], 0) + 
                  torch.mul(self.x[:, 4], 0) +
                  torch.randn(data_amount) * 2.5)

        self.len = self.x.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len