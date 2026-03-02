import torch
from torch.utils.data import random_split
import numpy as np
from data.SyntheticData import SyntheticData
from data.RealData import RealData
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from models.ComponentwiseBoostingModel import ComponentwiseBoostingModel
from config import config as default_config
import matplotlib.pyplot as plt

def run_experiment(
    seed,
    # Scenario Params
    dataset_type,
    dataset_name,
    dim_mode,
    n_samples,
    noise_std,
    signal_type,
    feature_dist,
    rho1, rho2, rho3,
    # Model/Method Params
    competing_learners,
    learning_rate,
    train_split,
    use_momentum,
    use_top_k,
    use_flooding,
    forced_flood_level=None,
    # Hyperparams
    top_k=5,
    momentum_strength=3.0,
    poly_degree=3,
    n_knots=30,
    n_bins=256,
    target_df=1.0,
    method_name="Vanilla"
):
    # Load Data
    if dataset_type == 'synthetic':
        dataset_clean = SyntheticData(
            n_samples=n_samples,
            dim_mode=dim_mode,
            noise_std=noise_std,
            seed=seed,
            drift_type='none',
            signal_type=signal_type,
            feature_dist=feature_dist,
            rho1=rho1, rho2=rho2, rho3=rho3
        )
    else:
        dataset_clean = RealData(
            dataset_name=dataset_name, 
            seed=seed
        )
    
    # Splits
    total_len = len(dataset_clean)
    n_dev = int(train_split * total_len)
    
    # Deterministic pplit using specified seed
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=g).tolist()
    
    dev_idx = indices[:n_dev]
    test_idx = indices[n_dev:]
    
    X_dev = dataset_clean.x[dev_idx]
    y_dev = dataset_clean.y[dev_idx]
    X_test = dataset_clean.x[test_idx]
    y_test = dataset_clean.y[test_idx]
    
    # scale real data using standard scaler
    if dataset_type == 'real':
        scaler = StandardScaler()
        scaler.fit(X_dev.numpy())
        X_dev = torch.tensor(scaler.transform(X_dev.numpy()), dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test.numpy()), dtype=torch.float32)

    # Determine Flood Level based on noise variance if synthetic
    flood_level = 0.0
    if use_flooding:
        if forced_flood_level is not None:
            flood_level = forced_flood_level
        elif dataset_type == 'synthetic':
            flood_level = noise_std ** 2

    # Model Setup
    model_params = dict(
        n_estimators=default_config.n_estimators,
        learning_rate=learning_rate,
        base_learner=competing_learners,
        poly_degree=poly_degree,
        tree_max_depth=1,
        n_bins=n_bins,
        spline_degree=3,
        n_knots=n_knots,
        loss='flooding' if use_flooding else 'mse',
        flood_level=flood_level,
        use_momentum=use_momentum,
        use_top_k=use_top_k,
        top_k=top_k,
        momentum_decay=default_config.momentum_decay,
        momentum_strength=momentum_strength,
        random_state=seed,
        eps_momentum=default_config.eps_momentum,
        eps_linear=default_config.eps_linear,
        target_df=target_df
    )

    # Initialize K-fold CV
    kf = KFold(n_splits=default_config.k_folds, shuffle=True, random_state=seed)
    cv_val_histories = []

    # Train model on each fold and record val loss
    for fold_i, (t_idx, v_idx) in enumerate(kf.split(X_dev)):
        X_f_train, y_f_train = X_dev[t_idx], y_dev[t_idx]
        X_f_val, y_f_val = X_dev[v_idx], y_dev[v_idx]
        
        cv_model = ComponentwiseBoostingModel(**model_params)
        cv_model.fit(X_f_train, y_f_train, X_val=X_f_val, y_val=y_f_val)
        cv_val_histories.append(cv_model.history['val_loss'])

    # Find iteration with min avg val loss
    avg_val_loss = np.mean(np.array(cv_val_histories), axis=0)
    best_iter_cv = np.argmin(avg_val_loss) + 1 
    min_val_loss_cv = avg_val_loss[best_iter_cv - 1]

    # Final fit
    final_model = ComponentwiseBoostingModel(**model_params)
    final_model.fit(
        X_dev, y_dev,
        X_val=None, y_val=None, 
        X_test=X_test, y_test=y_test # Tracking Test Loss here
    )
    final_model.best_iteration_ = best_iter_cv
    final_model.history['val_loss'] = avg_val_loss.tolist()

    # Evaluate on test set
    results = {}
    def get_mse(X, y, use_best):
        pred = final_model.predict(X, use_best_model=use_best)
        return torch.mean((pred - y)**2).item()

    results['val_best'] = min_val_loss_cv
    results['clean_best'] = get_mse(X_test, y_test, use_best=True)
    results['clean'] = results['clean_best']

    # Drift Scenarios
    if dataset_type == 'synthetic':
        drifts = [
            ('meaningful', 'weak'),
            ('meaningful', 'strong'),
            ('noise', 'weak'),
            ('noise', 'strong')
        ]
        
        for d_type, d_mag in drifts:
            ds_drift = SyntheticData(
                n_samples=n_samples,
                dim_mode=dim_mode,
                noise_std=noise_std,
                seed=seed,
                drift_type=d_type,
                drift_magnitude=d_mag,
                signal_type=signal_type,
                feature_dist=feature_dist,
                rho1=rho1, rho2=rho2, rho3=rho3
            )
            # Simulate drift on test set using same indices
            X_test_drift = ds_drift.x[test_idx]
            y_test_drift = ds_drift.y[test_idx]
            
            # Save only the best model performance
            key = f"mse_{d_type}_{d_mag}_best"
            results[key] = get_mse(X_test_drift, y_test_drift, use_best=True)

    return {
        'model_obj': final_model, 
        'best_iter': best_iter_cv,
        'scores': results,
        'history': final_model.history,
        'flood_level': flood_level
    }