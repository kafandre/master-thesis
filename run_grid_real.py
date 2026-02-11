import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import run_experiment
from config import config
import sys
import datetime
import random 
import torch
from joblib import Parallel, delayed 
from filelock import FileLock 

# --- Setup Directories ---
RESULTS_DIR = "results_real_competing_3"
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "grid_summary_real.csv")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# List of real datasets available in data/RealData.py
REAL_DATASETS = ["bodyfat", "diabetes", "riboflavin"]     # ["diabetes", "bodyfat", "riboflavin"]

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        if "\r" not in message:
            if message.strip(): 
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                self.log.write(f"{timestamp}{message}")
            else:
                self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

log_file_path = os.path.join(RESULTS_DIR, "grid_log_real.txt")
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

# --- Configuration Generation ---
method_configs = [
    {"name": "Vanilla",             "mom": False, "topk": False},
    {"name": "TopK",                "mom": False, "topk": True},
    {"name": "Momentum",            "mom": True,  "topk": False},
    {"name": "TopK+Momentum",       "mom": True,  "topk": True},
]

# --- Helper Functions ---

def get_run_signature(params):
    # Signature adapted for Real Data (uses dataset_name instead of scenario)
    sig = (
        f"{params['dataset_name']}_{params['base_learner']}_"
        f"{params['method']}_s{params['seed']}_flood{params['use_flooding']}"
    )
    return sig.replace(" ", "")

def get_filename_base(params, flood_level_val=None):
    # Filename adapted for Real Data
    name = (
        f"{params['dataset_name']}_{params['base_learner']}_{params['method'].replace(' ', '')}_seed{params['seed']}"
    )
    if params['mom']:
        name += f"_momStr{config.momentum_strength}"
    if params['topk']:
        name += f"_topk{params['top_k_int']}"
    if params['use_flooding'] and flood_level_val is not None:
        name += f"_floodLvl{flood_level_val:.5f}"
    return name

def save_plot(history, flood_level, params, filename_base):
    plt.figure(figsize=(10, 6))
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.6, linewidth=1)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', color='green', alpha=0.8, linewidth=1.5)
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss', color='red', alpha=0.8, linewidth=1.5)
    if params['use_flooding']:
        plt.axhline(y=flood_level, color='black', linestyle='--', label=f'Flood {flood_level:.4f}')

    plt.title(f"{params['dataset_name']} | {params['base_learner']} | {params['method']} | Seed {params['seed']}")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    path = os.path.join(PLOTS_DIR, f"Plot_{filename_base}.png")
    plt.savefig(path)
    plt.close()

def save_results_to_csv(row_dict):
    df = pd.DataFrame([row_dict])
    hdr = not os.path.exists(SUMMARY_FILE)
    df.to_csv(SUMMARY_FILE, mode='a', header=hdr, index=False)

def run_single_wrapper(params):
    # CRITICAL CPU OPTIMIZATION
    torch.set_num_threads(1) 
    
    # --- IMPORTANT: INJECT CONFIG FOR REAL DATA ---
    # Since we are in a worker process, we must set the global config here
    # to ensure RealData loads the correct dataset.
    config.DATASET_TYPE = "real"
    config.DATASET_NAME = params['dataset_name']
    
    lock_path = os.path.join(RESULTS_DIR, "grid_summary_real.csv.lock")
    local_csv_lock = FileLock(lock_path)

    learner_arg = params['base_learner']
    if learner_arg == 'competing':
        learner_arg = config.COMPETING_LEARNERS

    # --- 1. CLEAN RUN SETUP ---
    clean_params = params.copy()
    clean_params['use_flooding'] = False
    
    clean_fname = get_filename_base(clean_params)
    clean_hist_path = os.path.join(HISTORY_DIR, f"Hist_{clean_fname}.pkl")
    
    clean_history = None
    min_train_loss = None
    
    if os.path.exists(clean_hist_path):
        try:
            with open(clean_hist_path, 'rb') as f:
                clean_history = pickle.load(f)
            min_train_loss = min(clean_history['train_loss'])
        except Exception as e:
            print(f"Warning: Corrupt history file {clean_fname}, re-running. Error: {e}")
            clean_history = None 
            
    if clean_history is None:
        try:
            # Note: dim_mode, n_samples, etc. are passed but ignored by RealData class
            res_clean = run_experiment(
                seed=params['seed'],
                dim_mode=0,         # Ignored for Real
                n_samples=0,        # Ignored for Real
                noise_std=0.0,      # Ignored for Real
                base_learner=learner_arg,
                use_momentum=params['mom'],
                use_top_k=params['topk'],
                use_flooding=False,
                flood_multiplier=0.0,
                forced_flood_level=None,
                specific_top_k=params['top_k_int'],
                signal_type="real", # Ignored
                feature_dist="real",# Ignored
                noise_dist="real",  # Ignored
                learning_rate=params['lr']
            )
            clean_history = res_clean['history']
            min_train_loss = min(clean_history['train_loss'])
            
            with open(clean_hist_path, 'wb') as f:
                pickle.dump(clean_history, f)
            save_plot(clean_history, 0.0, clean_params, clean_fname)
            
            row = clean_params.copy()
            row.update({
                'signature': get_run_signature(clean_params),
                'flood_level': 0.0,
                'best_iter': res_clean['best_iter'],
                'mse_clean': res_clean['scores']['clean_best'],
                'mse_clean_last': res_clean['scores']['clean_last'],
                'val_best': res_clean['scores']['val_best'],
            })
            # Add any other scores present (likely just clean/val for real data)
            for k, v in res_clean['scores'].items():
                if k not in ['clean', 'clean_best', 'clean_last', 'val_best']:
                    row[f"mse_{k}"] = v
            
            with local_csv_lock:
                save_results_to_csv(row)
                
        except Exception as e:
            print(f"Error in CLEAN run {clean_fname}: {e}")
            import traceback
            traceback.print_exc()
            return 

    # --- 2. FLOODING RUN SETUP ---
    # Logic: If training loss is very low, apply flooding relative to that min loss
    # For Real Data, we don't have "True Noise Variance", so we rely purely on empirical min train loss.
    if min_train_loss is not None:
        val_losses = clean_history['val_loss']
        train_losses = clean_history['train_loss']
        
        # 1. Find min val loss and its index
        best_val_idx = np.argmin(val_losses)
        min_val_loss = val_losses[best_val_idx]
        
        # 2. Find where val loss first falls below 1.025 * min_val_loss
        threshold = min_val_loss * 1.025
        crossing_idx = best_val_idx  # Default to best index if not found earlier
        
        for i in range(len(val_losses)):
            if val_losses[i] < threshold:
                crossing_idx = i
                break
        
        # 3. Add 100 iterations from that point (clamping to max iterations)
        target_idx = min(crossing_idx + 100, len(train_losses) - 1)
        
        # 4. Use the training loss at that index
        candidate_flood_level = train_losses[target_idx]
        
        # 5. Apply Lower Bound (min_train_loss * 1.1)
        lower_bound = min_train_loss * 1.05
        target_flood_level = max(candidate_flood_level, lower_bound)

        flood_params = params.copy()
        flood_params['use_flooding'] = True
        
        flood_fname = get_filename_base(flood_params, target_flood_level)
        flood_hist_path = os.path.join(HISTORY_DIR, f"Hist_{flood_fname}.pkl")
        
        if not os.path.exists(flood_hist_path):
            try:
                res_flood = run_experiment(
                    seed=params['seed'],
                    dim_mode=0,
                    n_samples=0,
                    noise_std=0.0,
                    base_learner=learner_arg,
                    use_momentum=params['mom'],
                    use_top_k=params['topk'],
                    use_flooding=True,
                    flood_multiplier=0.0, 
                    forced_flood_level=target_flood_level,
                    specific_top_k=params['top_k_int'],
                    signal_type="real",
                    feature_dist="real",
                    noise_dist="real",
                    learning_rate=params['lr']
                )
                
                # Save History
                with open(flood_hist_path, 'wb') as f:
                    pickle.dump(res_flood['history'], f)
                    
                # Save Plot
                save_plot(res_flood['history'], target_flood_level, flood_params, flood_fname)
                
                # Save to CSV (Protected by Lock)
                row = flood_params.copy()
                row.update({
                    'signature': get_run_signature(flood_params),
                    'flood_level': target_flood_level,
                    'best_iter': res_flood['best_iter'],
                    'mse_clean': res_flood['scores']['clean_best'],
                    'mse_clean_last': res_flood['scores']['clean_last'],
                    'val_best': res_flood['scores']['val_best'],
                })
                for k, v in res_flood['scores'].items():
                    if k not in ['clean', 'clean_best', 'clean_last', 'val_best']:
                        row[f"mse_{k}"] = v
                
                with local_csv_lock:
                    save_results_to_csv(row)
                    
            except Exception as e:
                print(f"Error in FLOOD run {flood_fname}: {e}")

# --- Main Grid Setup ---
if __name__ == "__main__":

    total_iterations_est = (
        len(REAL_DATASETS) * len(config.base_learners) * len(method_configs) * config.n_seeds
    )

    print(f"Preparing Parallel Real Data Analysis. Approx Combinations: {total_iterations_est}")
    print(f"Datasets: {REAL_DATASETS}")
    print(f"Resuming is supported: Existing 'Hist_*.pkl' files will be skipped.")

    all_jobs = []
    
    # Iterate over Real Datasets
    for dataset_name in REAL_DATASETS:
        for base_learner in config.base_learners:
            # if base_learner == "bspline" and dataset_name in ["riboflavin", "pcr"]:
            #     continue
            # Use default Learning Rate for Real Data (or define a dictionary if needed)
            current_lr = config.learning_rate
            
            for method_conf in method_configs:
                for seed_offset in range(config.n_seeds):
                    seed = config.SEED + seed_offset
                    
                    actual_k = config.top_k

                    params = {
                        'dataset_name': dataset_name,
                        'base_learner': base_learner,
                        'lr': current_lr,
                        'seed': seed,
                        'method': method_conf['name'],
                        'mom': method_conf['mom'],
                        'topk': method_conf['topk'],
                        'top_k_int': actual_k,
                        'use_flooding': False
                    }
                    all_jobs.append(params)

    print(f"Dispatched {len(all_jobs)} jobs to workers.")
    print("Starting execution using n_jobs=-2 (All CPUs minus 1)...")
    
    Parallel(n_jobs=-2, verbose=10, batch_size=1)(
        delayed(run_single_wrapper)(p) for p in all_jobs
    )

    print("Real Data Analysis Complete.")