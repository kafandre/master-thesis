import torch
from torch.utils.data import random_split
import numpy as np
from data.NoisyData import NoisyData
from data.RealData import RealData
from models.ComponentwiseBoostingModel import ComponentwiseBoostingModel
from config import config as default_config
import matplotlib.pyplot as plt

def run_experiment(
    seed,
    dim_mode,
    n_samples,
    noise_std,
    base_learner,
    use_momentum,
    use_top_k,
    use_flooding,
    flood_multiplier,
    batch_size=None
):
    # --- 1. Train on CLEAN Data ---
    if default_config.DATASET_TYPE == 'synthetic':
        dataset_clean = NoisyData(
            n_samples=n_samples,
            dim_mode=dim_mode,
            noise_std=noise_std,
            seed=seed,
            drift_type='none'
        )
    else:
        # For Real Data
        dataset_clean = RealData(
            dataset_name=default_config.DATASET_NAME, 
            seed=seed
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
    if default_config.flood_level is None:
        flood_level = dataset_clean.true_noise_var * flood_multiplier
    else:
        flood_level = default_config.flood_level
    
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
        batch_size=batch_size,
        random_state=seed
    )
    
    # Fit
    model.fit(
        X_train, y_train,
        X_val, y_val,
        X_test=X_test_clean,
        y_test=y_test_clean
        )

    # --- 2. Evaluate (The 5 Scenarios) ---
    results = {}
    
    # Helper to evaluate
    def get_mse(X, y):
        # Use best model (virtual checkpoint)
        pred = model.predict(X, use_best_model=True)
        return torch.mean((pred - y)**2).item()

    # A. Clean Test
    results['clean'] = get_mse(X_test_clean, y_test_clean)
    
    # B. Drift Scenarios (Synthetic Only)
    if default_config.DATASET_TYPE == 'synthetic':
        scenarios = [
            ('meaningful', 'weak'),
            ('meaningful', 'strong'),
            ('noise', 'weak'),
            ('noise', 'strong')
        ]
        
        for d_type, d_mag in scenarios:
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
        'history': model.history,
        'flood_level': flood_level,
        'batch_size': batch_size,
        'n_samples': n_samples,
        'dim_mode': dim_mode,
        'use_momentum': use_momentum,
        'use_top_k': use_top_k
    }

if __name__ == "__main__":
        
    # Format: (Name, Momentum, Top-K, Batch Size)
    settings_list = [
        {"name": "Top-K Only",          "mom": False, "topk": True,  "batch": None},
        {"name": "Momentum Only",       "mom": True,  "topk": False, "batch": None},
        {"name": "Mini-Batch Only",     "mom": False, "topk": False, "batch": 50},
        {"name": "Combined (All 3)",    "mom": True,  "topk": True,  "batch": 50},
    ]

    for setting in settings_list:
        print(f"\n--- Running Experiment: {setting['name']} ---")
        
        # 1. Run the experiment
        res = run_experiment(
            seed=100, 
            dim_mode="high", 
            n_samples=200, 
            noise_std=5.0, 
            base_learner="polynomial", 
            use_momentum=setting["mom"], 
            use_top_k=setting["topk"], 
            use_flooding=True, 
            flood_multiplier=1, 
            batch_size=setting["batch"]
        )
        
        # 2. Extract Data
        history = res['history']
        flood_level = res['flood_level']
        
        # 3. Plotting
        plt.figure(figsize=(12, 7))
        
        # Plot Losses
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.6, linewidth=1)
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Validation Loss', color='green', alpha=0.8, linewidth=1.5)
            
        if 'test_loss' in history and len(history['test_loss']) > 0:
            plt.plot(history['test_loss'], label='Test Loss (Clean)', color='red', alpha=0.8, linewidth=1.5)
        
        # Plot Flood Level Horizontal Line
        plt.axhline(y=flood_level, color='black', linestyle='--', linewidth=2, label=f'Flood Level ({flood_level:.3f})')
        
        # Styling
        plt.xlabel('Boosting Iterations')
        plt.ylabel('MSE Loss')
        plt.title(f'Loss Curves (n={res["n_samples"]}, Dim_mode: {res["dim_mode"]} Batch size: {res["batch_size"]}, Flood Level: {flood_level:.3f})')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 4. Show Plot
        plt.tight_layout()
        plt.savefig(f'./Plot_n{res["n_samples"]}_dims-{res["dim_mode"]}_batch-{res["batch_size"]}_momentum-{res["use_momentum"]}_top-k-{res["use_top_k"]}_floodlevel-{flood_level:.3f}.png')