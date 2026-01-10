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
        random_state=seed,
        eps_momentum=default_config.eps_momentum,
        eps_linear=default_config.eps_linear
    )
    
    # Fit
    model.fit(
        X_train, y_train,
        X_val, y_val,
        X_test=X_test_clean,
        y_test=y_test_clean
        )

    # --- 2. Evaluate (5 Scenarios) ---
    results = {}
    
    # Helper to evaluate
    def get_mse(X, y):
        # Use best model (virtual checkpoint)
        pred = model.predict(X, use_best_model=True)
        return torch.mean((pred - y)**2).item()

    # record vailadtion loss at best iteration
    results['val_best'] = model.history['val_loss'][model.best_iteration_ - 1]

    # A. Clean Test
    results['clean'] = get_mse(X_test_clean, y_test_clean)
    
    # B. Drift Scenarios (Synthetic Only)
    if default_config.DATASET_TYPE == 'synthetic':
        for d_type, d_mag in default_config.drift_scenarios:
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
    plt.ion()

    # Format: (Name, Momentum, Top-K, Batch Size)
    settings_list = [
        {"name": "Vanilla",             "mom": False, "topk": False, "batch": None},
        {"name": "Top-K Only",          "mom": False, "topk": True,  "batch": None},
        {"name": "Momentum Only",       "mom": True,  "topk": False, "batch": None},
        {"name": "Top-K + Momentum",    "mom": True,  "topk": True, "batch": None},
        {"name": "Mini-Batch Only",     "mom": False, "topk": False, "batch": default_config.demo_batch_size},
        {"name": "Mini-Batch + Top-K",  "mom": False, "topk": True, "batch": default_config.demo_batch_size},
        {"name": "Mini-Batch + Momentum","mom": True, "topk": False, "batch": default_config.demo_batch_size},
        {"name": "Combined (All 3)",    "mom": True,  "topk": True,  "batch": default_config.demo_batch_size},
    ]

    for setting in settings_list:
        print(f"\n--- Running Experiment: {setting['name']} ---")
        
        # 1. Run the experiment
        res = run_experiment(
            seed=default_config.demo_seed, 
            dim_mode=default_config.demo_dim_mode, 
            n_samples=default_config.demo_n_samples, 
            noise_std=default_config.demo_noise_std, 
            base_learner=default_config.demo_base_learner, 
            use_momentum=setting["mom"],
            use_top_k=setting["topk"], 
            use_flooding=default_config.demo_use_flooding,
            flood_multiplier=default_config.demo_flood_multiplier, 
            batch_size=setting["batch"]
        )

        # Evaliation Scores
        scores = res['scores']
        print(f"--- Evaluation Scores (Best Iter: {res['best_iter']}) ---")
        
        # Create a formatted string for both Print and Plot
        score_text = f"Best Iter: {res['best_iter']}\n\n"
        for key, value in scores.items():
            # Format: 'meaningful_weak: 0.1234'
            line = f"{key}: {value:.4f}"
            print(line)
            score_text += line + "\n"
        print("-----------------------------------------------------")

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
        
        # Adding Evaluation Scores to Plot
        # 1. Shrink the plot area slightly to make room on the right
        plt.subplots_adjust(right=0.70) 
        
        # 2. Add the text box (x=1.05 puts it just outside the plot)
        plt.text(
            1.05, 0.5,                  # x, y position (relative to axes)
            score_text,                 # The text string
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # Sanitized name for file system
        safe_setting_name = setting['name'].replace(" ", "_").replace("(", "").replace(")", "")

        # --- OPTIMIZED TITLES & NAMES ---
        
        # 1. Build a suffix based on active hyperparameters
        param_suffix = ""
        if setting['mom']:
            param_suffix += f"_momStr{default_config.momentum_strength}_momDec{default_config.momentum_decay}"
        if setting['topk']:
            param_suffix += f"_topk{default_config.top_k}"
        if setting['batch'] is not None:
            param_suffix += f"_batch{setting['batch']}"
            
        # Sanitized setting name for file system
        safe_setting_name = setting['name'].replace(" ", "_").replace("(", "").replace(")", "")

        # 2. Define Title and Filename based on Dataset Type
        if default_config.DATASET_TYPE == 'real':
            # --- REAL DATA ---
            plot_title = (
                f"{setting['name']} | {default_config.demo_base_learner}\n"
                f"Dataset: {default_config.DATASET_NAME} | "
                f"Flood x{default_config.demo_flood_multiplier} (Lvl: {flood_level:.3f})"
            )
            
            filename = (
                f"Plot_{safe_setting_name}_"
                f"{default_config.demo_base_learner}_"
                f"{default_config.DATASET_NAME}"
                f"{param_suffix}_"
                f"floodMul{default_config.demo_flood_multiplier}.png"
            )
        else:
            # --- SYNTHETIC DATA ---
            plot_title = (
                f"{setting['name']} | {default_config.demo_base_learner}\n"
                f"N={default_config.demo_n_samples} ({default_config.demo_dim_mode}) | "
                f"Noise={default_config.demo_noise_std} | "
                f"Flood x{default_config.demo_flood_multiplier} (Lvl: {flood_level:.3f})"
            )
            
            filename = (
                f"Plot_{safe_setting_name}_"
                f"{default_config.demo_base_learner}_"
                f"n{default_config.demo_n_samples}_{default_config.demo_dim_mode}_"
                f"noise{default_config.demo_noise_std}"
                f"{param_suffix}_"
                f"floodMul{default_config.demo_flood_multiplier}.png"
            )

        # Styling
        plt.xlabel('Boosting Iterations')
        plt.ylabel('MSE Loss')
        plt.title(plot_title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 4. Save and Show
        plt.tight_layout()
        plt.savefig(filename)
        
        plt.draw()
        plt.pause(0.1)

    print("All runs finished. Close plot windows to exit.")
    plt.ioff() # Turn interactive mode off
    plt.show() # Blocking call to keep windows open until you close them