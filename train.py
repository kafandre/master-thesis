import torch
from torch.utils.data import random_split
import numpy as np
from data.NoisyData import NoisyData
from models.ComponentwiseBoostingModel import ComponentwiseBoostingModel
from config import config as default_config

def run_experiment(
    seed,
    drift_type=default_config.drift_type,
    drift_magnitude=default_config.drift_magnitude,
    dim_mode=default_config.dim_mode,
    use_momentum=default_config.use_momentum,
    use_top_k=default_config.use_top_k,
    use_flooding=default_config.use_flooding,
    flood_multiplier=default_config.flood_level_sigma_multiplier
):
    # 1. Data Setup
    dataset = NoisyData(
        n_samples=default_config.n_samples,
        dim_mode=dim_mode,
        seed=seed,
        drift_type=drift_type,
        drift_magnitude=drift_magnitude
    )
    
    total_len = len(dataset)
    train_len = int(default_config.train_split * total_len)
    val_len = int(default_config.val_split * total_len)
    test_len = total_len - train_len - val_len
    
    train_data, val_data, test_data = random_split(
        dataset, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    def to_xy(subset):
        return dataset.x[subset.indices], dataset.y[subset.indices]

    X_train, y_train = to_xy(train_data)
    X_val, y_val = to_xy(val_data)
    X_test, y_test = to_xy(test_data)
    
    # 2. Determine Flood Level
    flood_level = dataset.true_noise_var * flood_multiplier
    
    # 3. Model Init
    model = ComponentwiseBoostingModel(
        n_estimators=default_config.n_estimators,
        learning_rate=default_config.learning_rate,
        base_learner=default_config.base_learner,
        poly_degree=default_config.poly_degree,
        loss='flooding' if use_flooding else 'mse',
        flood_level=flood_level,
        use_momentum=use_momentum,
        use_top_k=use_top_k,
        top_k=default_config.top_k,
        momentum_decay=default_config.momentum_decay,
        momentum_strength=default_config.momentum_strength,
        random_state=seed
    )
    
    # 4. Fit
    model.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 5. Evaluate
    # A. Last Epoch
    last_pred = model.predict(X_test, use_best_model=False)
    last_mse = torch.mean((last_pred - y_test)**2).item()
    
    # B. Best Validation Epoch
    best_pred = model.predict(X_test, use_best_model=True)
    best_mse = torch.mean((best_pred - y_test)**2).item()
    
    return {
        'seed': seed,
        'drift': drift_type,
        'dim': dim_mode,
        'method_momentum': use_momentum,
        'method_topk': use_top_k,
        'method_flooding': use_flooding,
        'last_test_mse': last_mse,
        'best_test_mse': best_mse,
        'best_iter': model.best_iteration_,
        'final_train_mse': model.history['train_loss'][-1]
    }

if __name__ == "__main__":
    # Test run
    print(run_experiment(100))