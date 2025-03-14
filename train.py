import torch
from torch.utils.data import random_split
import numpy as np
from data.NoisyData import Data
from models.BatchComponentwiseBoostingModel import ComponentwiseBoostingModel
import matplotlib.pyplot as plt



# Set random seed for reproducibility   200 data points and 423 is overfit
SEED = 111
torch.manual_seed(SEED)
np.random.seed(SEED)

# Generate synthetic data
dataset = Data(data_amount=200, seed=423)

# Split data into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Convert to tensors for model training
X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

# Training parameters
n_estimators = 1000
learning_rate = 0.1
eval_freq = 1
flood_level = 34
batch_size = 80

# Create and train MSE model
print("Training CWB model with MSE loss...")
mse_model = ComponentwiseBoostingModel(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=SEED,
    loss='mse',
    track_history=True
)

mse_model.fit(
    X=X_train, 
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    batch_size=train_size,    
    eval_freq=eval_freq,
    verbose=True
)

# Create and train Flooding loss model
print("\nTraining CWB model with Flooding loss...")
flooding_model = ComponentwiseBoostingModel(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=SEED,
    loss='flooding',
    track_history=True
)

flooding_model.fit(
    X=X_train, 
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    batch_size=batch_size,
    flood_level=flood_level,
    eval_freq=eval_freq,
    verbose=True
)


# Create evaluation points for plotting
# Since we're evaluating every eval_freq iterations, we need to adjust our x-axis
iterations = list(range(0, n_estimators + 1, eval_freq))
if iterations[0] != 0:
    iterations[0] = 0  # Ensure we include the initial point

# We may need to trim the iterations if early stopping occurred
mse_train_len = len(mse_model.history['train_loss'])
flooding_train_len = len(flooding_model.history['train_loss'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mse_test_len = len(mse_model.history['test_loss'])

# Plot 1: CWB without Flooding
ax1.plot(iterations[:mse_train_len], 
            mse_model.history['train_loss'], 
            label='Train Loss', 
            color='blue')
ax1.plot(iterations[:mse_test_len], 
            mse_model.history['test_loss'], 
            label='Test Loss', 
            color='red')
ax1.set_title('CWB without Flooding')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: CWB with Flooding
flooding_test_len = len(flooding_model.history['test_loss'])

ax2.plot(iterations[:flooding_train_len], 
            flooding_model.history['train_loss'], 
            label='Train Loss', 
            color='blue')
ax2.plot(iterations[:flooding_test_len], 
            flooding_model.history['test_loss'], 
            label='Test Loss', 
            color='red')
ax2.set_title('CWB with Flooding')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('cwb_loss_comparison.png')
plt.show()

# Print final feature importances
print("\nMSE Model Feature Importances:")
for i, importance in enumerate(mse_model.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")

print("\nFlooding Model Feature Importances:")
for i, importance in enumerate(flooding_model.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")

# Print final losses
mse_final_train_loss = mse_model.get_loss(X_train, y_train)
mse_final_test_loss = mse_model.get_loss(X_test, y_test)
flood_final_train_loss = flooding_model.get_loss(X_train, y_train)
flood_final_test_loss = flooding_model.get_loss(X_test, y_test)

print("\nFinal Losses:")
print(f"MSE Model - Train: {mse_final_train_loss:.4f}, Test: {mse_final_test_loss:.4f}")
print(f"Flooding Model - Train: {flood_final_train_loss:.4f}, Test: {flood_final_test_loss:.4f}")

# # Optional: Plot double descent analysis
# print("\nAnalyzing double descent phenomenon...")
# mse_results = mse_model.analyze_double_descent(X_train, y_train, X_test, y_test)
# flooding_results = flooding_model.analyze_double_descent(X_train, y_train, X_test, y_test)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# # MSE Double Descent
# ax1.plot(mse_results['iterations'], mse_results['train_loss'], label='Train Loss', color='blue')
# ax1.plot(mse_results['iterations'], mse_results['test_loss'], label='Test Loss', color='blue', linestyle='--')
# ax1.set_title('MSE Model: Double Descent Analysis')
# ax1.set_xlabel('Number of Iterations')
# ax1.set_ylabel('Loss')
# ax1.set_xscale('log')
# ax1.legend()
# ax1.grid(True)

# # Flooding Double Descent
# ax2.plot(flooding_results['iterations'], flooding_results['train_loss'], label='Train Loss', color='red')
# ax2.plot(flooding_results['iterations'], flooding_results['test_loss'], label='Test Loss', color='red', linestyle='--')
# ax2.set_title('Flooding Model: Double Descent Analysis')
# ax2.set_xlabel('Number of Iterations')
# ax2.set_ylabel('Loss')
# ax2.set_xscale('log')
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
# plt.savefig('cwb_double_descent_analysis.png')
# plt.show()


