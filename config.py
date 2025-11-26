class config:
    # --- Defaults (can be overridden by run_grid.py) ---
    SEED = 100
    n_seeds = 10  # Number of seeds per configuration

    # Data
    n_samples = 500
    dim_mode = "low"     # 'low' (5), 'high' (50)
    drift_type = "none"  # 'none', 'meaningful', 'noise'
    drift_magnitude = "weak"
    
    # Model
    n_estimators = 1000
    learning_rate = 0.1
    base_learner = "polynomial"
    poly_degree = 2
    
    # Methods (Flags)
    use_momentum = False
    use_top_k = False
    use_flooding = False
    
    # Hyperparameters
    top_k = 5
    momentum_decay = 0.9
    momentum_strength = 1.0
    
    # Flooding: Multiplier of the true noise variance
    # e.g., 0.5 means we flood at 0.5 * sigma^2
    flood_level_sigma_multiplier = 0.0 
    
    # Splits
    train_split = 0.7
    val_split = 0.15
    # test_split = remainder