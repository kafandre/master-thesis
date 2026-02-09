BASE_LEARNERS_LIST = ["linear", "polynomial", "tree", "bspline"]

class config:
    SEED = 600
    n_seeds = 5
    
    # --- Dataset Configuration ---
    # Options: "synthetic" or "real"
    DATASET_TYPE = "synthetic" 
    
    # Options: "diabetes", "bodyfat", "riboflavin", "pcr" (Ignored if type is synthetic)
    DATASET_NAME = "riboflavin"
    
    # --- Base Learners (4 types) ---
    base_learners = BASE_LEARNERS_LIST

    # Static Hyperparameters
    poly_degree = 3
    tree_depth = 1
    
    # B-Spline Configuration
    spline_degree = 3
    n_knots = 25  # Number of internal knots
    
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
    train_split = 0.4
    val_split = 0.10

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
            "n_samples": 1000, "dim": 100,  "signal_scale": 1.0, "noise_std": 5.0,
            "signal_type": "baseline_composite", "feature_dist": "normal", "noise_dist": "normal"
        },
        "HighDims": {
            "n_samples": 500, "dim": 1000,  "signal_scale": 1.0, "noise_std": 5.0,
            "signal_type": "baseline_composite", "feature_dist": "normal", "noise_dist": "normal"
        },
        "High_Noise": {
            "n_samples": 1000, "dim": 100, "signal_scale": 1.0, "noise_std": 10.0,
            "signal_type": "baseline_composite", "feature_dist": "normal", "noise_dist": "normal"
        },    
        "Multicollinearity": {
            "n_samples": 1000, "dim": 100,  "signal_scale": 1.0, "noise_std": 5.0,
            "signal_type": "baseline_composite", "feature_dist": "correlated", "noise_dist": "normal"
        },
        "Linear": {
            "n_samples": 1000, "dim": 100,  "signal_scale": 1.0, "noise_std": 5.0,
            "signal_type": "simple_additive", "feature_dist": "normal", "noise_dist": "normal"
        },                
        "Smooth": {
            "n_samples": 1000, "dim": 100,  "signal_scale": 1.0, "noise_std": 5.0, 
            "signal_type": "smooth_qubic", "feature_dist": "normal", "noise_dist": "normal"
        },
        "High_Freq": {
            "n_samples": 1000, "dim": 100,  "signal_scale": 1.0, "noise_std": 5.0,
            "signal_type": "high_freq", "feature_dist": "normal", "noise_dist": "normal"
        },
        "Step": {
            "n_samples": 1000, "dim": 100, "signal_scale": 1.0, "noise_std": 2.0,
            "signal_type": "step", "feature_dist": "normal", "noise_dist": "normal"
        }
    }

    # --- TUNED LEARNING RATES ---
    # Specific learning rates for each (scenario, base_learner) pair.
    # Initialized with default 0.05.
    TUNED_LRS = {
        scenario: {learner: 0.05 for learner in BASE_LEARNERS_LIST}
        for scenario in SCENARIOS
    }

    TUNED_LRS["Step"]["tree"] = 0.1
    # TUNED_LRS["Baseline"]["tree"] = 0.1
    # TUNED_LRS["Baseline"]["bspline"] = 0.1
    # TUNED_LRS["HighDims"]["linear"] = 0.1
    # TUNED_LRS["HighDims"]["bspline"] = 0.08
    # TUNED_LRS["HighDims"]["tree"] = 0.1
    # TUNED_LRS["HighDims"]["polynomial"] = 0.1
    