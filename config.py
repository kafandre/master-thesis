class config:
    SEED = 100
    n_seeds = 10 
    
    # --- Dataset Configuration ---
    # Options: "synthetic" or "real"
    DATASET_TYPE = "real" 
    
    # Options: "diabetes", "bodyfat", "riboflavin", "pcr" (Ignored if type is synthetic)
    DATASET_NAME = "bodyfat"
    
    # --- Data Levels (Synthetic Only) ---
    dims = ["low", "high"]           
    sizes = [100, 1000]              
    noise_levels = [1.0, 3.0]

    # --- Base Learners (3 types) ---
    base_learners = ["linear", "polynomial", "tree"]
    
    # Static Hyperparameters
    poly_degree = 2
    tree_depth = 2
    
    # Fixed Method Params
    top_k = 5
    momentum_decay = 0.9
    momentum_strength = 1.0

    flood_level = 0

    # Batch size for mini-batch processing
    batch_size = None
    
    # Experiment Config
    n_estimators = 1000
    learning_rate = 0.1
    train_split = 0.7
    val_split = 0.15