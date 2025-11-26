import pandas as pd
import pickle
from tqdm import tqdm
from train import run_experiment
from config import config

# --- Define the Grid ---
method_combinations = [
    # (name, momentum, top_k, flooding)
    ("Vanilla", False, False, False),
    ("Momentum", True, False, False),
    ("TopK", False, True, False),
    ("Flooding", False, False, True),
    ("Flooding+Momentum", True, False, True),
    ("Flooding+TopK", False, True, True),
    ("All", True, True, True),
]

dims = ["low", "high"]
drifts = ["none", "meaningful", "noise"]
flood_multipliers = [0.5, 1.0, 2.0]

# We will separate summary data (CSV) from curve data (Pickle)
summary_table = []
all_curves = {} # Key: specific run ID, Value: history dict

print("Starting Grid Search...")
total_runs = len(method_combinations) * len(dims) * len(drifts) * config.n_seeds
pbar = tqdm(total=total_runs)

run_id = 0

for dim in dims:
    for drift in drifts:
        for method_name, use_mom, use_topk, use_flood in method_combinations:
            
            # If using flooding, test multiple levels; otherwise just run once (level 0)
            current_flood_levels = flood_multipliers if use_flood else [0.0]
            
            for flood_mult in current_flood_levels:
                for seed_offset in range(config.n_seeds):
                    seed = config.SEED + seed_offset
                    run_id += 1
                    
                    try:
                        res = run_experiment(
                            seed=seed,
                            drift_type=drift,
                            drift_magnitude=config.drift_magnitude,
                            dim_mode=dim,
                            use_momentum=use_mom,
                            use_top_k=use_topk,
                            use_flooding=use_flood,
                            flood_multiplier=flood_mult
                        )
                        
                        # 1. Save Summary Data (for CSV Table)
                        summary_row = {
                            'run_id': run_id,
                            'method_name': method_name,
                            'drift': drift,
                            'dim': dim,
                            'seed': seed,
                            'flood_mult': flood_mult,
                            'last_test_mse': res['last_test_mse'],
                            'best_test_mse': res['best_test_mse'],
                            'best_iter': res['best_iter'],
                            'final_train_mse': res['final_train_mse']
                        }
                        summary_table.append(summary_row)
                        
                        # 2. Save Curves (for Plotting later)
                        # We save the whole history dict using a unique key
                        all_curves[run_id] = res['history']
                        
                    except Exception as e:
                        print(f"Error in run {run_id}: {e}")
                        
                    pbar.update(1)

pbar.close()

# --- Save Results ---
# 1. Save Big Table to CSV
df = pd.DataFrame(summary_table)
df.to_csv("grid_search_results.csv", index=False)
print(f"\nSummary table saved to 'grid_search_results.csv'")

# 2. Save Curves to Pickle
with open("grid_search_curves.pkl", "wb") as f:
    pickle.dump(all_curves, f)
print(f"Loss curves saved to 'grid_search_curves.pkl'")

# Quick Sanity Check
print("\nFirst few rows of results:")
print(df.head())