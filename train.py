import torch
from torch.utils.data import random_split
import numpy as np
from data.NoisyData import NoisyData
from models.ComponentwiseBoostingModel import ComponentwiseBoostingModel
from config import config as default_config

def run_experiment(
    seed,
    dim_mode,
    n_samples,
    noise_std,
    base_learner,
    use_momentum,
    use_top_k,
    use_flooding,
    flood_multiplier
):
    # --- 1. Train on CLEAN Data ---
    dataset_clean = NoisyData(
        n_samples=n_samples,
        dim_mode=dim_mode,
        noise_std=noise_std,
        seed=seed,
        drift_type='none'
    )
    
    # Splits
    total_len = len(dataset_clean)
    train_len = int(default_config.train_split * total_len)
    val_len = int(default_config.val_split * total_len)
    test_len = total_len - train_len - val_len
    
    # Deterministic Split
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=g).tolist()
    
    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len+val_len]
    test_idx = indices[train_len+val_len:]
    
    X_train = dataset_clean.x[train_idx]
    y_train = dataset_clean.y[train_idx]
    X_val = dataset_clean.x[val_idx]
    y_val = dataset_clean.y[val_idx]
    X_test_clean = dataset_clean.x[test_idx]
    y_test_clean = dataset_clean.y[test_idx]
    
    # Determine Flood Level
    flood_level = dataset_clean.true_noise_var * flood_multiplier
    
    # Init Model
    model = ComponentwiseBoostingModel(
        n_estimators=default_config.n_estimators,
        learning_rate=default_config.learning_rate,
        base_learner=base_learner,
        poly_degree=default_config.poly_degree,
        tree_max_depth=default_config.tree_depth,
        loss='flooding' if use_flooding else 'mse',
        flood_level=flood_level,
        use_momentum=use_momentum,
        use_top_k=use_top_k,
        top_k=default_config.top_k,
        momentum_decay=default_config.momentum_decay,
        momentum_strength=default_config.momentum_strength,
        random_state=seed
    )
    
    # Fit
    model.fit(X_train, y_train, X_val, y_val)
    
    # --- 2. Evaluate (The 5 Scenarios) ---
    results = {}
    
    # Helper to evaluate
    def get_mse(X, y):
        # Use best model (virtual checkpoint)
        pred = model.predict(X, use_best_model=True)
        return torch.mean((pred - y)**2).item()

    # A. Clean Test
    results['clean'] = get_mse(X_test_clean, y_test_clean)
    
    # B. The 4 Drift Scenarios
    scenarios = [
        ('meaningful', 'weak'),
        ('meaningful', 'strong'),
        ('noise', 'weak'),
        ('noise', 'strong')
    ]
    
    for d_type, d_mag in scenarios:
        # Re-gen universe with drift
        ds_drift = NoisyData(
            n_samples=n_samples,
            dim_mode=dim_mode,
            noise_std=noise_std,
            seed=seed,
            drift_type=d_type,
            drift_magnitude=d_mag
        )
        # Use SAME test indices
        X_test_drift = ds_drift.x[test_idx]
        y_test_drift = ds_drift.y[test_idx]
        
        results[f"{d_type}_{d_mag}"] = get_mse(X_test_drift, y_test_drift)
        
    return {
        'model_obj': model, # Optional: return if you want to save
        'best_iter': model.best_iteration_,
        'scores': results,
        'history': model.history
    }

if __name__ == "__main__":
    # Test run
    print(run_experiment(100, "low", 500, 1.0, "polynomial", False, False, True, 0.5))