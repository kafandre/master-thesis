import pandas as pd
import pickle
import os
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train import run_experiment
from config import config
import sys
import datetime
from joblib import Parallel, delayed 
from filelock import FileLock 

# Setup Directories
RESULTS_DIR = "results_unified"
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "grid_summary.csv")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Custom logger to write output to terminal and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        if "\r" not in message and message.strip():
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.log.write(f"{timestamp}{message}")
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return self.terminal.isatty()

log_file_path = os.path.join(RESULTS_DIR, "grid_log.txt")
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

# Method Configs
method_configs = [
    {"name": "Vanilla",             "mom": False, "topk": False},
    {"name": "TopK",                "mom": False, "topk": True},
    {"name": "Momentum",            "mom": True,  "topk": False},
    {"name": "TopK+Momentum",       "mom": True,  "topk": True},
]

# Helper Functions

def get_filename_base(params, flood_level_val=None):
    name = f"{params['scenario_name']}_{params['method']}_s{params['seed']}"
    if params['use_flooding']:
        lvl = flood_level_val if flood_level_val is not None else 0.0
        name += f"_flood{lvl:.4f}"
    else:
        name += "_clean"
    return name

def save_plot(history, flood_level, params, filename_base):
    plt.figure(figsize=(10, 6))
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss (Final Model)', color='blue', alpha=0.6)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss (CV Avg)', color='green', alpha=0.8)
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss (Hold-out)', color='red', alpha=0.8, linestyle=':')
        
    if params['use_flooding']:
        plt.axhline(y=flood_level, color='black', linestyle='--', label=f'Flood {flood_level:.4f}')

    plt.title(f"{params['scenario_name']} | {params['method']} | Seed {params['seed']}")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    path = os.path.join(PLOTS_DIR, f"Plot_{filename_base}.png")
    plt.savefig(path)
    plt.close()

def save_results_to_csv(row_dict):
    clean_dict = row_dict.copy()
    if 'competing_learners' in clean_dict:
        clean_dict['competing_learners'] = ";".join(clean_dict['competing_learners'])
    
    # Clean up obsolete keys
    keys_to_exclude = ['mse_clean_last']
    for k in keys_to_exclude:
        if k in clean_dict:
            del clean_dict[k]

    df = pd.DataFrame([clean_dict])
    # use filelock to prevent race conditions when writing to summary file
    lock_path = os.path.join(RESULTS_DIR, "grid_summary.csv.lock")
    with FileLock(lock_path):
        hdr = not os.path.exists(SUMMARY_FILE)
        df.to_csv(SUMMARY_FILE, mode='a', header=hdr, index=False)

def run_single_wrapper(params):
    # run single experiment configuration
    torch.set_num_threads(1)
    
    scen_name = params['scenario_name']
    s_params = config.SCENARIOS[scen_name]
    
    # Prepare Arguments
    common_args = {
        'seed': params['seed'],
        'dataset_type': s_params['dataset_type'],
        'dataset_name': s_params['dataset_name'],
        'dim_mode': s_params['dim'],
        'n_samples': s_params['n_samples'],
        'noise_std': s_params['noise_std'],
        'signal_type': s_params['signal_type'],
        'feature_dist': s_params['feature_dist'],
        'rho1': s_params['rho1'],
        'rho2': s_params['rho2'],
        'rho3': s_params['rho3'],
        'competing_learners': s_params['competing_learners'],
        'learning_rate': s_params['learning_rate'],
        'train_split': s_params['train_split'],
        'use_momentum': params['mom'],
        'use_top_k': params['topk'],
        'top_k': s_params['top_k'],
        'momentum_strength': s_params['momentum_strength'],
        'poly_degree': s_params['poly_degree'],
        'n_knots': s_params['n_knots'],
        'n_bins': s_params['n_bins'],
        'target_df': s_params['target_df'],
        'method_name': params['method']
    }

    # Clean Run (no Flooding)
    clean_run_needed = True
    clean_fname = get_filename_base({**params, 'use_flooding': False})
    clean_hist_path = os.path.join(HISTORY_DIR, f"Hist_{clean_fname}.pkl")
    
    clean_history = None
    min_train_loss = None

    # check if clean run already exists
    if os.path.exists(clean_hist_path):
        try:
            with open(clean_hist_path, 'rb') as f:
                clean_history = pickle.load(f)
            min_train_loss = min(clean_history['train_loss'])
            clean_run_needed = False
        except:
            clean_run_needed = True

    # run experiment without flooding
    if clean_run_needed:
        res_clean = run_experiment(
            **common_args,
            use_flooding=False,
            forced_flood_level=None
        )
        clean_history = res_clean['history']
        min_train_loss = min(clean_history['train_loss'])
        
        with open(clean_hist_path, 'wb') as f:
            pickle.dump(clean_history, f)
            
        save_plot(clean_history, 0.0, {**params, 'use_flooding': False}, clean_fname)
        
        row = common_args.copy()
        row.update({
            'scenario': scen_name,
            'method': params['method'],
            'use_flooding': False,
            'flood_level': 0.0,
            'best_iter': res_clean['best_iter'],
            'mse_clean': res_clean['scores']['clean_best'],
            'val_best': res_clean['scores']['val_best']
        })
        
        # Add Drift scores
        for k, v in res_clean['scores'].items():
            if k.startswith("mse_") and k.endswith("_best"):
                row[k] = v
        
        save_results_to_csv(row)

    # Flooding Run
    # calculate flood level
    target_flood_level = 0.0
    if s_params['dataset_type'] == 'synthetic':
        target_flood_level = s_params['noise_std'] ** 2
    else:
        # Empirical flooding level for real data
        val_losses = clean_history['val_loss']
        train_losses = clean_history['train_loss']
        best_val_idx = np.argmin(val_losses)
        min_val_loss = val_losses[best_val_idx]
        
        threshold = min_val_loss * 1.025
        crossing_idx = best_val_idx
        for i in range(len(val_losses)):
            if val_losses[i] < threshold:
                crossing_idx = i
                break
        
        target_idx = min(crossing_idx + 100, len(train_losses) - 1)
        candidate = train_losses[target_idx]
        lower_bound = min_train_loss * 1.05
        target_flood_level = max(candidate, lower_bound)

    # Run experiment with flooding
    flood_fname = get_filename_base({**params, 'use_flooding': True}, target_flood_level)
    flood_hist_path = os.path.join(HISTORY_DIR, f"Hist_{flood_fname}.pkl")
    
    if not os.path.exists(flood_hist_path):
        res_flood = run_experiment(
            **common_args,
            use_flooding=True,
            forced_flood_level=target_flood_level
        )
        
        with open(flood_hist_path, 'wb') as f:
            pickle.dump(res_flood['history'], f)
            
        save_plot(res_flood['history'], target_flood_level, {**params, 'use_flooding': True}, flood_fname)
        
        row = common_args.copy()
        row.update({
            'scenario': scen_name,
            'method': params['method'],
            'use_flooding': True,
            'flood_level': target_flood_level,
            'best_iter': res_flood['best_iter'],
            'mse_clean': res_flood['scores']['clean_best'],
            'val_best': res_flood['scores']['val_best']
        })

        # Add Drift Scores
        for k, v in res_flood['scores'].items():
            if k.startswith("mse_") and k.endswith("_best"):
                row[k] = v
                
        save_results_to_csv(row)

if __name__ == "__main__":
    
    all_jobs = []
    
    print(f"Generating Jobs for {len(config.SCENARIOS)} Scenarios x {len(method_configs)} Methods x {config.n_seeds} Seeds...")
    
    # Generate job configurations for all runs
    for scen_name in config.SCENARIOS.keys():
        for method_conf in method_configs:
            for seed_offset in range(config.n_seeds):
                seed = config.SEED + seed_offset
                
                params = {
                    'scenario_name': scen_name,
                    'method': method_conf['name'],
                    'mom': method_conf['mom'],
                    'topk': method_conf['topk'],
                    'seed': seed
                }
                all_jobs.append(params)
                
    print(f"Total Jobs: {len(all_jobs)}")
    print("Starting execution using n_jobs=-2...")
    
    # Execute jobs parallely
    Parallel(n_jobs=-2, verbose=5, batch_size=1)(
        delayed(run_single_wrapper)(p) for p in all_jobs
    )
    
    print("Grid Search Complete.")