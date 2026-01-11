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

# --- Setup Directories ---
RESULTS_DIR = "results"
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
log_file_path = os.path.join(RESULTS_DIR, "grid_log.txt")
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout

# --- Configuration Generation ---
# 1. Base Methods
base_methods = [
    {"name": "Vanilla",  "mom": False, "topk": False},
    {"name": "Momentum", "mom": True,  "topk": False},
    {"name": "TopK",     "mom": False, "topk": True},
    {"name": "All",      "mom": True,  "topk": True},
]

# 2. Add Batch variants
method_configs = []

for m in base_methods:
    # No Batch
    c = m.copy()
    c['batch'] = None
    method_configs.append(c)
    
    # With Batch
    c_batch = m.copy()
    c_batch['name'] = f"MiniBatch {c_batch['name']}"
    c_batch['batch'] = "half_train"
    method_configs.append(c_batch)

# --- Helper Functions ---

def get_run_signature(params):
    """Creates a unique string ID for a run to check for duplicates/completion."""
    # Order matters: learner, dim, n, noise, method, batch, seed, flood
    # We do NOT include flood_level value in signature check for the boolean flag,
    # but we DO include 'flooding' status.
    sig = (
        f"{params['base_learner']}_d{params['dim']}_n{params['n_samples']}_"
        f"ns{params['noise_std']}_{params['method']}_b{params['batch']}_"
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
    if params['batch'] is not None:
        name += f"_batch{params['batch']}"
    
    # Flooding (Only if enabled)
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

def check_if_done(signature):
    if not os.path.exists(SUMMARY_FILE):
        return False
    # This is a bit slow for massive files, but safe. 
    # For speed, we could load the existing signatures into memory once at startup.
    try:
        df = pd.read_csv(SUMMARY_FILE)
        if 'signature' not in df.columns: return False
        return signature in df['signature'].values
    except:
        return False

# --- Load completed runs for fast checking ---
finished_signatures = set()
if os.path.exists(SUMMARY_FILE):
    try:
        df = pd.read_csv(SUMMARY_FILE)
        if 'signature' in df.columns:
            finished_signatures = set(df['signature'].values)
    except Exception as e:
        print(f"Warning: Could not read existing summary file: {e}")

# --- Main Grid Loop ---

total_iterations = (
    len(config.base_learners) * len(config.dims) * len(config.sizes) * len(config.noise_levels) * len(method_configs) * config.n_seeds
)

print(f"Starting Grid Search. Total Combinations (Clean runs): {total_iterations}")
print(f"Flooding runs will be triggered automatically where applicable.")

pbar = tqdm(total=total_iterations)

for base_learner in config.base_learners:
    for dim in config.dims:
        for size in config.sizes:
            for noise in config.noise_levels:
                for method_conf in method_configs:
                    for seed_offset in range(config.n_seeds):
                        seed = config.SEED + seed_offset
                        
                        # --- Calculate Dynamic Batch Size ---
                        if method_conf['batch'] == "half_train":
                            train_len = int(size * config.train_split) # calculate train set size
                            batch_val = int(train_len / 2)             # Set to half (2 batches)
                        else:
                            batch_val = method_conf['batch']
                            if batch_val is not None:
                                batch_val = int(batch_val)

                        # --- 2. Dynamic Top-K Logic ---
                        # Use 3 for dim=5, otherwise 5
                        actual_k = 3 if dim == 5 else 5

                        # Shared Parameters
                        current_params = {
                            'base_learner': base_learner,
                            'dim': dim,
                            'n_samples': size,
                            'noise_std': noise,
                            'seed': seed,
                            'method': method_conf['name'],
                            'mom': method_conf['mom'],
                            'topk': method_conf['topk'],
                            'batch': batch_val,
                            'top_k_int': actual_k,
                            'use_flooding': False
                        }

                        # --- 1. RUN CLEAN (Flooding=False) ---
                        clean_sig = get_run_signature(current_params)
                        
                        # Placeholders for results to pass to flooding run
                        min_train_loss = None
                        run_clean_performed = False

                        if clean_sig in finished_signatures:
                            # If we skip, we need to retrieve the min_train_loss if we want to run the flooding counterpart
                            # However, for simplicity and safety, if the Clean run is done but Flooding isn't, 
                            # we might need to re-calculate min loss. 
                            # To avoid complexity, we only skip if the Clean run is logged. 
                            # If we need min_train_loss for the next step, we might need to re-run or load history.
                            # Strategy: If Clean is done, try to load its history to find min_train_loss.
                            try:
                                # Construct filename to load history
                                fname_base = get_filename_base(current_params)
                                hist_path = os.path.join(HISTORY_DIR, f"Hist_{fname_base}.pkl")
                                with open(hist_path, 'rb') as f:
                                    h = pickle.load(f)
                                min_train_loss = min(h['train_loss'])
                            except:
                                # If history missing, force re-run
                                pass
                        
                        if min_train_loss is None:
                            try:
                                res_clean = run_experiment(
                                    seed=seed,
                                    dim_mode=dim,
                                    n_samples=size,
                                    noise_std=noise,
                                    base_learner=base_learner,
                                    use_momentum=method_conf['mom'],
                                    use_top_k=method_conf['topk'],
                                    use_flooding=False,
                                    flood_multiplier=0.0, # Irrelevant
                                    batch_size=batch_val,
                                    forced_flood_level=None,
                                    specific_top_k=actual_k
                                )
                                
                                min_train_loss = min(res_clean['history']['train_loss'])
                                run_clean_performed = True
                                
                                # Save Clean Results
                                fname_base = get_filename_base(current_params)
                                
                                # 1. History
                                with open(os.path.join(HISTORY_DIR, f"Hist_{fname_base}.pkl"), 'wb') as f:
                                    pickle.dump(res_clean['history'], f)
                                    
                                # 2. Plot
                                save_plot(res_clean['history'], 0.0, current_params, fname_base)
                                
                                # 3. CSV Summary
                                row = current_params.copy()
                                row.update({
                                    'signature': clean_sig,
                                    'flood_level': 0.0,
                                    'best_iter': res_clean['best_iter'],
                                    'mse_clean': res_clean['scores']['clean'],
                                    'val_best': res_clean['scores']['val_best'],
                                })
                                # Add Drift scores
                                for k, v in res_clean['scores'].items():
                                    if k not in ['clean', 'val_best']:
                                        row[f"mse_{k}"] = v
                                        
                                save_results_to_csv(row)
                                finished_signatures.add(clean_sig)

                            except Exception as e:
                                print(f"\nError in CLEAN run {clean_sig}: {e}")
                                min_train_loss = None # Cannot proceed to flooding

                        
                        # --- 2. RUN FLOODING (Twin Run) ---
                        # Logic: Use min_train_loss * 1.05
                        # Exclusion: Do not run flooding if method is exactly "Vanilla" (No batch)
                        # "MiniBatch Vanilla" IS allowed.
                        
                        should_run_flood = (method_conf['name'] != "Vanilla")
                        
                        if should_run_flood and min_train_loss is not None:
                            
                            # Update params for Flooding
                            flood_params = current_params.copy()
                            flood_params['use_flooding'] = True
                            target_flood_level = min_train_loss * 1.05
                            
                            flood_sig = get_run_signature(flood_params)
                            
                            if flood_sig not in finished_signatures:
                                try:
                                    res_flood = run_experiment(
                                        seed=seed,
                                        dim_mode=dim,
                                        n_samples=size,
                                        noise_std=noise,
                                        base_learner=base_learner,
                                        use_momentum=method_conf['mom'],
                                        use_top_k=method_conf['topk'],
                                        use_flooding=True,
                                        flood_multiplier=0.0, # Ignored due to forced level
                                        batch_size=batch_val,
                                        forced_flood_level=target_flood_level,
                                        specific_top_k=actual_k
                                    )
                                    
                                    # Save Flood Results
                                    fname_base = get_filename_base(flood_params, target_flood_level)
                                    
                                    with open(os.path.join(HISTORY_DIR, f"Hist_{fname_base}.pkl"), 'wb') as f:
                                        pickle.dump(res_flood['history'], f)
                                        
                                    save_plot(res_flood['history'], target_flood_level, flood_params, fname_base)
                                    
                                    row = flood_params.copy()
                                    row.update({
                                        'signature': flood_sig,
                                        'flood_level': target_flood_level,
                                        'best_iter': res_flood['best_iter'],
                                        'mse_clean': res_flood['scores']['clean'],
                                        'val_best': res_flood['scores']['val_best'],
                                    })
                                    for k, v in res_flood['scores'].items():
                                        if k not in ['clean', 'val_best']:
                                            row[f"mse_{k}"] = v
                                            
                                    save_results_to_csv(row)
                                    finished_signatures.add(flood_sig)
                                    
                                except Exception as e:
                                    print(f"\nError in FLOOD run {flood_sig}: {e}")

                        pbar.update(1)

pbar.close()
print("Grid Search Complete.")