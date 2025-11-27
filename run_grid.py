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

flood_multipliers = [0.5, 1.0, 2.0]

summary_table = []
# all_curves = {} # Uncomment if you want to save curves (warning: large file)

print("Starting Grid Search...")

# Calculate total for progress bar
# Methods * (BaseLearners * Dims * Sizes * Noises * Seeds)
# Note: Flood loop is inside methods, so it's approx 15 method-variants
# 15 * 3 * 2 * 2 * 2 * 10 = 3600
pbar = tqdm(total=3600)

run_id = 0

# --- The Big Loop ---
for base_learner in config.base_learners:
    for dim in config.dims:
        for size in config.sizes:
            for noise in config.noise_levels:
                for method_name, use_mom, use_topk, use_flood in method_combinations:
                    
                    # Flooding Levels Logic
                    current_flood_levels = flood_multipliers if use_flood else [0.0]
                    
                    for flood_mult in current_flood_levels:
                        for seed_offset in range(config.n_seeds):
                            seed = config.SEED + seed_offset
                            run_id += 1
                            
                            try:
                                res = run_experiment(
                                    seed=seed,
                                    dim_mode=dim,
                                    n_samples=size,
                                    noise_std=noise,
                                    base_learner=base_learner,
                                    use_momentum=use_mom,
                                    use_top_k=use_topk,
                                    use_flooding=use_flood,
                                    flood_multiplier=flood_mult
                                )
                                
                                # Unpack Results
                                scores = res['scores']
                                
                                row = {
                                    'run_id': run_id,
                                    'seed': seed,
                                    'base_learner': base_learner,
                                    'dim': dim,
                                    'n_samples': size,
                                    'noise_std': noise,
                                    'method': method_name,
                                    'flood_mult': flood_mult,
                                    'best_iter': res['best_iter'],
                                    # The 5 Key Metrics
                                    'mse_clean': scores['clean'],
                                    'mse_mean_weak': scores['meaningful_weak'],
                                    'mse_mean_strong': scores['meaningful_strong'],
                                    'mse_noise_weak': scores['noise_weak'],
                                    'mse_noise_strong': scores['noise_strong'],
                                }
                                summary_table.append(row)
                                
                                # all_curves[run_id] = res['history']
                                
                            except Exception as e:
                                print(f"Error in run {run_id}: {e}")
                                
                            pbar.update(1)

pbar.close()

# Save
df = pd.DataFrame(summary_table)
df.to_csv("grid_search_results.csv", index=False)
print("Done! Results saved.")