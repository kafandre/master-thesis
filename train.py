import torch
from torch.utils.data import random_split
import numpy as np
from data.SyntheticData import SyntheticData
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
    forced_flood_level=None,
    specific_top_k=None,
    signal_type='linear_interaction',
    feature_dist='normal',
    noise_dist='normal',
    learning_rate=None
):
    # --- 1. Train on CLEAN Data ---
    if default_config.DATASET_TYPE == 'synthetic':
        dataset_clean = SyntheticData(
            n_samples=n_samples,
            dim_mode=dim_mode,
            noise_std=noise_std,
            seed=seed,
            drift_type='none',
            signal_type=signal_type,
            feature_dist=feature_dist,
            noise_dist=noise_dist
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
    
    # --- Determine Flood Level ---
    if forced_flood_level is not None:
        flood_level = forced_flood_level
    elif default_config.flood_level is None:
        flood_level = dataset_clean.true_noise_var * flood_multiplier
    else:
        flood_level = default_config.flood_level
    
    # deteremine top-k
    current_top_k = specific_top_k if specific_top_k is not None else default_config.top_k
    
    # determine LR
    lr = learning_rate if learning_rate is not None else default_config.learning_rate

    # Init Model
    model = ComponentwiseBoostingModel(
        n_estimators=default_config.n_estimators,
        learning_rate=lr,
        base_learner=base_learner,
        poly_degree=default_config.poly_degree,
        tree_max_depth=default_config.tree_depth,
        n_bins=default_config.n_bins,
        loss='flooding' if use_flooding else 'mse',
        flood_level=flood_level,
        use_momentum=use_momentum,
        use_top_k=use_top_k,
        top_k=current_top_k,
        momentum_decay=default_config.momentum_decay,
        momentum_strength=default_config.momentum_strength,
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
    def get_mse(X, y, use_best):
        pred = model.predict(X, use_best_model=use_best)
        return torch.mean((pred - y)**2).item()

    # record validation loss at best iteration
    results['val_best'] = model.history['val_loss'][model.best_iteration_ - 1] if model.best_iteration_ > 0 else model.history['val_loss'][-1]

    # A. Clean Test (Best & Last)
    results['clean_best'] = get_mse(X_test_clean, y_test_clean, use_best=True)
    results['clean_last'] = get_mse(X_test_clean, y_test_clean, use_best=False)
    
    # For backward compatibility with existing code that expects 'clean'
    results['clean'] = results['clean_best']
    
    # B. Drift Scenarios (Synthetic Only)
    if default_config.DATASET_TYPE == 'synthetic':
        for d_type, d_mag in default_config.drift_scenarios:
            ds_drift = SyntheticData(
                n_samples=n_samples,
                dim_mode=dim_mode,
                noise_std=noise_std,
                seed=seed,
                drift_type=d_type,
                drift_magnitude=d_mag,
                signal_type=signal_type,
                feature_dist=feature_dist,
                noise_dist=noise_dist
            )
            X_test_drift = ds_drift.x[test_idx]
            y_test_drift = ds_drift.y[test_idx]
            
            # Record both Best and Last model performance on drift
            results[f"{d_type}_{d_mag}_best"] = get_mse(X_test_drift, y_test_drift, use_best=True)
            results[f"{d_type}_{d_mag}_last"] = get_mse(X_test_drift, y_test_drift, use_best=False)
            
            # For backward compatibility
            results[f"{d_type}_{d_mag}"] = results[f"{d_type}_{d_mag}_best"]
        
    return {
        'model_obj': model, 
        'best_iter': model.best_iteration_,
        'scores': results,
        'history': model.history,
        'flood_level': flood_level,
        'n_samples': n_samples,
        'dim_mode': dim_mode,
        'use_momentum': use_momentum,
        'use_top_k': use_top_k
    }

if __name__ == "__main__":
    plt.ion()
    torch.set_num_threads(1)

    # DEMO
    scenario_name = default_config.demo_scenario
    scen_params = default_config.SCENARIOS[scenario_name]
    
    print(f"Running Demo on Scenario: {scenario_name}")
    print(f"Params: {scen_params}")

    # Format: (Name, Momentum, Top-K)
    settings_list = [
        {"name": "Vanilla",             "mom": False, "topk": False},
        {"name": "Top-K Only",          "mom": False, "topk": True},
        {"name": "Momentum Only",       "mom": True,  "topk": False},
        {"name": "Top-K + Momentum",    "mom": True,  "topk": True},
    ]

    for setting in settings_list:
        print(f"\n--- Running Experiment: {setting['name']} ---")
        
        # 1. Run the experiment
        res = run_experiment(
            seed=default_config.demo_seed, 
            dim_mode=scen_params['dim'], 
            n_samples=scen_params['n_samples'], 
            noise_std=scen_params['noise_std'], 
            base_learner=default_config.demo_base_learner, 
            use_momentum=setting["mom"],
            use_top_k=setting["topk"], 
            use_flooding=default_config.demo_use_flooding,
            flood_multiplier=default_config.demo_flood_multiplier,
            signal_type=scen_params['signal_type'],
            feature_dist=scen_params['feature_dist'],
            noise_dist=scen_params['noise_dist']
        )

        # Evaliation Scores
        scores = res['scores']
        print(f"--- Evaluation Scores (Best Iter: {res['best_iter']}) ---")
        
        score_text = f"Best Iter: {res['best_iter']}\n\n"
        for key, value in scores.items():
            line = f"{key}: {value:.4f}"
            print(line)
            score_text += line + "\n"
        print("-----------------------------------------------------")

        history = res['history']
        flood_level = res['flood_level']
        
        plt.figure(figsize=(12, 7))
        
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.6, linewidth=1)
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Validation Loss', color='green', alpha=0.8, linewidth=1.5)
            
        if 'test_loss' in history and len(history['test_loss']) > 0:
            plt.plot(history['test_loss'], label='Test Loss (Clean)', color='red', alpha=0.8, linewidth=1.5)
        
        plt.axhline(y=flood_level, color='black', linestyle='--', linewidth=2, label=f'Flood Level ({flood_level:.3f})')
        
        plt.subplots_adjust(right=0.70) 
        plt.text(1.05, 0.5, score_text, transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))

        safe_setting_name = setting['name'].replace(" ", "_").replace("(", "").replace(")", "")
        
        plot_title = (
            f"{setting['name']} | {default_config.demo_base_learner} | {scenario_name}\n"
            f"Flood x{default_config.demo_flood_multiplier} (Lvl: {flood_level:.3f})"
        )
        
        filename = f"Plot_Demo_{scenario_name}_{safe_setting_name}.png"

        plt.xlabel('Boosting Iterations')
        plt.ylabel('MSE Loss')
        plt.title(plot_title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(filename)
        
        plt.draw()
        plt.pause(0.1)

    print("All runs finished. Close plot windows to exit.")
    plt.ioff()
    plt.show()