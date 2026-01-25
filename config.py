class config:
    SEED = 300
    n_seeds = 1
    
    # --- Dataset Configuration ---
    # Options: "synthetic" or "real"
    DATASET_TYPE = "synthetic" 
    
    # Options: "diabetes", "bodyfat", "riboflavin", "pcr" (Ignored if type is synthetic)
    DATASET_NAME = "riboflavin"
    
    # --- Data Levels (Synthetic Only) ---
    dims = [20,100,200]
    sizes = [200,500]              
    noise_levels = [3.0, 10.0]

    # --- Base Learners (4 types) ---
    base_learners = ["linear", "polynomial", "tree", "bspline"]
    
    # Static Hyperparameters
    poly_degree = 2
    tree_depth = 1
    
    # B-Spline Configuration
    spline_degree = 1
    n_knots = 5  # Number of internal knots
    
    # Tree Configuration
    n_bins = 32

    # Fixed Method Params
    top_k = 5
    momentum_decay = 0.9
    momentum_strength = 10.0

    flood_level = None
    
    # Experiment Config
    n_estimators = 1000
    learning_rate = 0.05
    train_split = 0.7
    val_split = 0.15

    # --- Evaluation Configuration ---
    # Scenarios for drift evaluation (Type, Magnitude)
    drift_scenarios = [
        ('meaningful', 'weak'),
        ('meaningful', 'strong'),
        ('noise', 'weak'),
        ('noise', 'strong')
    ]

    # --- Model Stability (Epsilons) ---
    eps_momentum = 1e-6
    eps_linear = 1e-5

    # --- Demo / Single Run Configuration (for train.py __main__) ---
    demo_seed = 100
    demo_dim_mode = 20
    demo_n_samples = 200
    demo_noise_std = 5.0
    demo_base_learner = "linear"
    demo_flood_multiplier = 1.0
    demo_use_flooding = False