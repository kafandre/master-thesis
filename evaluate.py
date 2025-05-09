from models.BatchComponentwiseBoostingModel import ComponentwiseBoostingModel
from data.NoisyData import Data
from config import config
from torch.utils.data import random_split
import torch

# Set random seed for reproducibility 200 data points and 423 is overfit
SEED = 8

# Generate synthetic data
dataset = Data(data_amount=1000, seed=SEED)

# Split data into train and test sets
train_size = int(0.0 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

# Load the model from iteration (interesting point in your double descent curve!)
loaded_model_1 = ComponentwiseBoostingModel.load_model("./flooding_model_checkpoints/model_iteration_29.pt")
loaded_model_2 = ComponentwiseBoostingModel.load_model("./flooding_model_checkpoints/model_iteration_1000.pt")

# Evaluate
test_pred = loaded_model_1.predict(X_test)
test_mse = loaded_model_1.get_loss(X_test, y_test)
print(f"Test MSE at iteration 29: {test_mse:.4f}")

test_pred = loaded_model_2.predict(X_test)
test_mse = loaded_model_2.get_loss(X_test, y_test)
print(f"Test MSE at iteration 1000: {test_mse:.4f}")