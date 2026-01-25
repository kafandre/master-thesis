BASE_LEARNERS_LIST = ["linear", "polynomial", "tree", "bspline"]

class config:
    SEED = 500
    n_seeds = 30
    
    # --- Dataset Configuration ---
    # Options: "synthetic" or "real"
    DATASET_TYPE = "synthetic" 
    
    # Options: "diabetes", "bodyfat", "riboflavin", "pcr" (Ignored if type is synthetic)
    DATASET_NAME = "riboflavin"
    
    # --- Base Learners (4 types) ---
    base_learners = BASE_LEARNERS_LIST

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

    # --- SENSITIVITY ANALYSIS SCENARIOS ---
    # Defines the 7 stress-test environments
    SCENARIOS = {
        "Baseline": {
            "n_samples": 500, "dim": 50, "noise_std": 5.0,
            "signal_type": "linear_interaction", "feature_dist": "normal", "noise_dist": "normal"
        },
        "HighDims": {
            "n_samples": 250, "dim": 250, "noise_std": 5.0,
            "signal_type": "linear_interaction", "feature_dist": "normal", "noise_dist": "normal"
        },
        "Friedman": {
            "n_samples": 500, "dim": 50, "noise_std": 1.0, 
            "signal_type": "friedman", "feature_dist": "normal", "noise_dist": "normal"
        },
        "Step": {
            "n_samples": 500, "dim": 50, "noise_std": 5.0,
            "signal_type": "step", "feature_dist": "normal", "noise_dist": "normal"
        },
        "Multicollinearity": {
            "n_samples": 500, "dim": 50, "noise_std": 5.0,
            "signal_type": "linear_interaction", "feature_dist": "correlated", "noise_dist": "normal"
        },
        "Outlier_Features": {
            "n_samples": 500, "dim": 50, "noise_std": 5.0,
            "signal_type": "linear_interaction", "feature_dist": "student_t", "noise_dist": "normal"
        },
        "Outlier_Target": {
            "n_samples": 500, "dim": 50, "noise_std": 5.0,
            "signal_type": "linear_interaction", "feature_dist": "normal", "noise_dist": "student_t"
        }
    }

    # --- TUNED LEARNING RATES ---
    # Specific learning rates for each (scenario, base_learner) pair.
    # Initialized with default 0.05.
    TUNED_LRS = {
        scenario: {learner: 0.05 for learner in BASE_LEARNERS_LIST}
        for scenario in SCENARIOS
    }

    # TUNED_LRS["Baseline"]["tree"] = 0.1
    # TUNED_LRS["HighDims"]["linear"] = 0.01

    # --- Demo / Single Run Configuration (for train.py __main__) ---
    demo_seed = 100
    demo_scenario = "Baseline" # Used to pick from SCENARIOS in demo
    demo_flood_multiplier = 1.0
    demo_use_flooding = False