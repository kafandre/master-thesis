class config:
    SEED = 100
    n_seeds = 10 
    
    # --- Data Levels (2 of each) ---
    dims = ["low", "high"]           # 5 vs 50 features
    sizes = [100, 1000]              # Small vs Large sample size
    noise_levels = [1.0, 3.0]        # Low vs High noise (Sigma)
    
    # --- Base Learners (3 types) ---
    base_learners = ["linear", "polynomial", "tree"]
    
    # Static Hyperparameters
    poly_degree = 2
    tree_depth = 2
    
    # Fixed Method Params
    top_k = 5
    momentum_decay = 0.9
    momentum_strength = 1.0
    
    # Experiment Config
    n_estimators = 1000
    learning_rate = 0.1
    train_split = 0.7
    val_split = 0.15