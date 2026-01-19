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
from joblib import Parallel, delayed 
from filelock import FileLock 

# --- Setup Directories ---
RESULTS_DIR = "results2"
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "grid_summary.csv")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        # Write to console (terminal) as is
        self.terminal.write(message)
        
        # Write to file with timestamp
        # Filter out carriage returns (\r) to avoid logging progress bar updates
        if "\r" not in message:
            if message.strip(): # If message has content
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                self.log.write(f"{timestamp}{message}")
            else:
                # Keep newlines for formatting
                self.log.write(message)
                
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

# Redirect stdout and stderr to the log file
# Note: In parallel execution, this mostly captures the main process. 
# Worker logs might be interleaved or captured by joblib.
log_file_path = os.path.join(RESULTS_DIR, "grid_log.txt")
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

# --- Configuration Generation ---
method_configs = [
    # 1. Single Mechanics (or None)
    {"name": "Vanilla",             "mom": False, "topk": False},
    {"name": "TopK",                "mom": False, "topk": True},
    {"name": "Momentum",            "mom": True,  "topk": False},
    
    # 2. Double Combinations
    {"name": "TopK+Momentum",       "mom": True,  "topk": True},
]

# --- Helper Functions ---

def get_run_signature(params):
    """Creates a unique string ID for a run to check for duplicates/completion."""
    # Order matters: learner, dim, n, noise, method, seed, flood
    # We do NOT include flood_level value in signature check for the boolean flag,
    # but we DO include 'flooding' status.
    sig = (
        f"{params['base_learner']}_d{params['dim']}_n{params['n_samples']}_"
        f"ns{params['noise_std']}_{params['method']}_"
        f"s{params['seed']}_flood{params['use_flooding']}"
    )
    return sig.replace(" ", "")

def get_filename_base(params, flood_level_val=None):
    """Generates the descriptive filename base (without extension)."""
    # Base: Method_Learner_Size_Dim_Noise_Seed
    name = (
        f"{params['method'].replace(' ', '')}_{params['base_learner']}_"
        f"n{params['n_samples']}_d{params['dim']}_"
        f"noise{params['noise_std']}_seed{params['seed']}"
    )
    
    # Dynamic Hyperparameters
    if params['mom']:
        name += f"_momStr{config.momentum_strength}_momDec{config.momentum_decay}"
    if params['topk']:
        name += f"_topk{params['top_k_int']}"
    
    # Flooding (Only if enabled)
    if params['use_flooding'] and flood_level_val is not None:
        name += f"_floodLvl{flood_level_val:.5f}"
        
    return name

def save_plot(history, flood_level, params, filename_base):
    # Create a new figure for every plot to ensure thread safety
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.6, linewidth=1)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', color='green', alpha=0.8, linewidth=1.5)
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss', color='red', alpha=0.8, linewidth=1.5)
        
    if params['use_flooding']:
        plt.axhline(y=flood_level, color='black', linestyle='--', label=f'Flood {flood_level:.4f}')

    plt.title(f"{params['method']} | Seed {params['seed']} | Flood: {params['use_flooding']}")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    path = os.path.join(PLOTS_DIR, f"Plot_{filename_base}.png")
    plt.savefig(path)
    plt.close()

def save_results_to_csv(row_dict):
    df = pd.DataFrame([row_dict])
    # Append if exists, write header only if new
    hdr = not os.path.exists(SUMMARY_FILE)
    df.to_csv(SUMMARY_FILE, mode='a', header=hdr, index=False)

def run_single_wrapper(params):
    """
    Worker function that handles ONE parameter combination (Clean + Flooding).
    It checks if the run exists (resume logic) and only runs if missing.
    """
    # initialize lock inside the worker
    lock_path = os.path.join(RESULTS_DIR, "grid_summary.csv.lock")
    local_csv_lock = FileLock(lock_path)
    
    # --- 1. CLEAN RUN SETUP ---
    clean_params = params.copy()
    clean_params['use_flooding'] = False
    
    clean_fname = get_filename_base(clean_params)
    clean_hist_path = os.path.join(HISTORY_DIR, f"Hist_{clean_fname}.pkl")
    
    clean_history = None
    min_train_loss = None
    
    # RESUME LOGIC: Check if the pickle file already exists
    # If it exists, we load it instead of re-training (Fast Skip)
    if os.path.exists(clean_hist_path):
        try:
            with open(clean_hist_path, 'rb') as f:
                clean_history = pickle.load(f)
            min_train_loss = min(clean_history['train_loss'])
        except Exception as e:
            # If the file is corrupt, we force a re-run
            print(f"Warning: Corrupt history file {clean_fname}, re-running. Error: {e}")
            clean_history = None 
            
    # EXECUTION: If not found (or corrupt), RUN IT
    if clean_history is None:
        try:
            res_clean = run_experiment(
                seed=params['seed'],
                dim_mode=params['dim'],
                n_samples=params['n_samples'],
                noise_std=params['noise_std'],
                base_learner=params['base_learner'],
                use_momentum=params['mom'],
                use_top_k=params['topk'],
                use_flooding=False,
                flood_multiplier=0.0,
                batch_size=None,
                forced_flood_level=None,
                specific_top_k=params['top_k_int']
            )
            clean_history = res_clean['history']
            min_train_loss = min(clean_history['train_loss'])
            
            # Save History
            with open(clean_hist_path, 'wb') as f:
                pickle.dump(clean_history, f)
                
            # Save Plot
            save_plot(clean_history, 0.0, clean_params, clean_fname)
            
            # Save to CSV (Protected by Lock)
            row = clean_params.copy()
            row.update({
                'signature': get_run_signature(clean_params),
                'flood_level': 0.0,
                'best_iter': res_clean['best_iter'],
                'mse_clean': res_clean['scores']['clean'],
                'val_best': res_clean['scores']['val_best'],
            })
            for k, v in res_clean['scores'].items():
                if k not in ['clean', 'val_best']:
                    row[f"mse_{k}"] = v
            
            with local_csv_lock:
                save_results_to_csv(row)
                
        except Exception as e:
            # Catching errors so one bad run doesn't kill the whole parallel process
            print(f"Error in CLEAN run {clean_fname}: {e}")
            return # Cannot proceed to flooding if clean failed

    # --- 2. FLOODING RUN SETUP ---
    # Only run flooding if method is not "Vanilla" (and if Clean run succeeded)
    if min_train_loss is not None:
        
        # Calculate flood level: train_loss at (best_val_iter + 50)
        best_val_idx = np.argmin(clean_history['val_loss'])
        
        # Add 50 iterations, but clamp to the last iteration if the run wasn't long enough
        target_idx = min(best_val_idx + 50, len(clean_history['train_loss']) - 1)
        
        target_flood_level = clean_history['train_loss'][target_idx]

        # Adjustment for Flat Tails
        # If min_train_loss is within 2% of the target (curve is flat),
        # boost the flood level by 5% to ensure it forces a change in dynamics
        if min_train_loss > 0.98 * target_flood_level:
            target_flood_level = min_train_loss * 1.05
        
        flood_params = params.copy()
        flood_params['use_flooding'] = True
        
        flood_fname = get_filename_base(flood_params, target_flood_level)
        flood_hist_path = os.path.join(HISTORY_DIR, f"Hist_{flood_fname}.pkl")
        
        # RESUME LOGIC: Check if flood pickle exists
        if not os.path.exists(flood_hist_path):
            try:
                res_flood = run_experiment(
                    seed=params['seed'],
                    dim_mode=params['dim'],
                    n_samples=params['n_samples'],
                    noise_std=params['noise_std'],
                    base_learner=params['base_learner'],
                    use_momentum=params['mom'],
                    use_top_k=params['topk'],
                    use_flooding=True,
                    flood_multiplier=0.0, 
                    batch_size=None,
                    forced_flood_level=target_flood_level,
                    specific_top_k=params['top_k_int']
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
                    'mse_clean': res_flood['scores']['clean'],
                    'val_best': res_flood['scores']['val_best'],
                })
                for k, v in res_flood['scores'].items():
                    if k not in ['clean', 'val_best']:
                        row[f"mse_{k}"] = v
                
                with local_csv_lock:
                    save_results_to_csv(row)
                    
            except Exception as e:
                print(f"Error in FLOOD run {flood_fname}: {e}")

# --- Main Grid Setup ---
if __name__ == "__main__":

    total_iterations_est = (
        len(config.base_learners) * len(config.dims) * len(config.sizes) * len(config.noise_levels) * len(method_configs) * config.n_seeds
    )

    print(f"Preparing Parallel Grid Search. Approx Combinations: {total_iterations_est}")
    print(f"Resuming is supported: Existing 'Hist_*.pkl' files will be skipped.")

    # 1. Generate ALL combinations into a list first
    # This replaces the nested loops so we can pass them to the parallel workers
    all_jobs = []
    
    for base_learner in config.base_learners:
        for dim in config.dims:
            for size in config.sizes:
                for noise in config.noise_levels:
                    for method_conf in method_configs:
                        for seed_offset in range(config.n_seeds):
                            seed = config.SEED + seed_offset
                            
                            # --- Dynamic Top-K Logic ---
                            actual_k = 3 if dim == 5 else 5

                            # Pack everything into a dictionary to send to the worker
                            params = {
                                'base_learner': base_learner,
                                'dim': dim,
                                'n_samples': size,
                                'noise_std': noise,
                                'seed': seed,
                                'method': method_conf['name'],
                                'mom': method_conf['mom'],
                                'topk': method_conf['topk'],
                                'top_k_int': actual_k,
                                'use_flooding': False # Start with clean run logic
                            }
                            all_jobs.append(params)

    print(f"Dispatched {len(all_jobs)} jobs to workers.")
    print("Starting execution using n_jobs=-2 (All CPUs minus 1)...")
    
    # Run in Parallel
    # verbose=10 gives nice progress updates in the terminal
    Parallel(n_jobs=-2, verbose=10, batch_size=1)(
        delayed(run_single_wrapper)(p) for p in all_jobs
    )

    print("Grid Search Complete.")