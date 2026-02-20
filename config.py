class config:
    SEED = 1000
    n_seeds = 30
    
    # --- Global Defaults (kept for reference, but manually injected below) ---
    n_estimators = 1000
    k_folds = 5
    
    # Grid Defaults
    top_k = 5
    momentum_decay = 0.9
    momentum_strength = 3.0
    target_df = 1.0
    n_knots = 30
    n_bins = 256
    poly_degree = 5
    
    # Stability
    eps_momentum = 1e-6
    eps_linear = 1e-5

    # Learner Sets
    ALL_LEARNERS = ["linear", "polynomial", "tree", "bspline"]
    LINEAR_SPLINE = ["linear", "bspline"]
    LINEAR = ["linear"]
    POLYNOMIAL = ["polynomial"]
    TREE = ["tree"]
    BSPLINE = ["bspline"]    


    # --- SCENARIOS (Manually Defined with Globals Injected) ---
    SCENARIOS = {
        # --- 1. LINEAR Scenarios (lr=0.05) ---
        "linear_base": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "simple_additive", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": LINEAR,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "linear_highdim": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "simple_additive", "learning_rate": 0.05,
            "n_samples": 500, "dim": 500, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": LINEAR,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "linear_highnoise": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "simple_additive", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 5.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": LINEAR,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "linear_corr": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "simple_additive", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.95, "rho2": 0.8, "rho3": 0.95, "feature_dist": "correlated",
            "competing_learners": LINEAR,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "linear_all": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "simple_additive", "learning_rate": 0.05,
            "n_samples": 500, "dim": 500, "noise_std": 5.0,
            "rho1": 0.75, "rho2": 0.5, "rho3": 0.75, "feature_dist": "correlated",
            "competing_learners": LINEAR,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },

        # --- 2. SMOOTH Scenarios (lr=0.05) ---
        "smooth_base": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "smooth_qubic", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": POLYNOMIAL,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "smooth_highdim": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "smooth_qubic", "learning_rate": 0.05,
            "n_samples": 500, "dim": 500, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": POLYNOMIAL,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "smooth_highnoise": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "smooth_qubic", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 5.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": POLYNOMIAL,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "smooth_corr": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "smooth_qubic", "learning_rate": 0.05,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.95, "rho2": 0.8, "rho3": 0.95, "feature_dist": "correlated",
            "competing_learners": POLYNOMIAL,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "smooth_all": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "smooth_qubic", "learning_rate": 0.05,
            "n_samples": 500, "dim": 500, "noise_std": 5.0,
            "rho1": 0.75, "rho2": 0.5, "rho3": 0.75, "feature_dist": "correlated",
            "competing_learners": POLYNOMIAL,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },

        # --- 3. STEP Scenarios (lr=0.2) ---
        "step_base": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "step", "learning_rate": 0.2,
            "n_samples": 1000, "dim": 200, "noise_std": 1.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": TREE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "step_highdim": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "step", "learning_rate": 0.2,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": TREE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "step_corr": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "step", "learning_rate": 0.2,
            "n_samples": 1000, "dim": 200, "noise_std": 1.0,
            "rho1": 0.95, "rho2": 0.8, "rho3": 0.95, "feature_dist": "correlated",
            "competing_learners": TREE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "step_all": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "step", "learning_rate": 0.2,
            "n_samples": 500, "dim": 500, "noise_std": 2.0,
            "rho1": 0.75, "rho2": 0.5, "rho3": 0.75, "feature_dist": "correlated",
            "competing_learners": TREE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },

        # --- 4. SINE Scenarios (lr=0.025) ---
        "sine_base": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "high_freq", "learning_rate": 0.01,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": BSPLINE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "sine_highdim": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "high_freq", "learning_rate": 0.01,
            "n_samples": 500, "dim": 500, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": BSPLINE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "sine_highnoise": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "high_freq", "learning_rate": 0.01,
            "n_samples": 1000, "dim": 200, "noise_std": 5.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": BSPLINE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "sine_corr": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "high_freq", "learning_rate": 0.01,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.95, "rho2": 0.8, "rho3": 0.95, "feature_dist": "correlated",
            "competing_learners": BSPLINE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "sine_all": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "high_freq", "learning_rate": 0.01,
            "n_samples": 500, "dim": 500, "noise_std": 5.0,
            "rho1": 0.75, "rho2": 0.5, "rho3": 0.75, "feature_dist": "correlated",
            "competing_learners": BSPLINE,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },

        # --- 5. MIXED Scenarios (lr=0.05) ---
        "mixed_base": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "mixed", "learning_rate": 0.025,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": ALL_LEARNERS,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 3.0,
            "n_knots": 15, "n_bins": 256, "poly_degree": 5
        },
        "mixed_highdim": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "mixed", "learning_rate": 0.025,
            "n_samples": 500, "dim": 500, "noise_std": 2.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": ALL_LEARNERS,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 3.0,
            "n_knots": 15, "n_bins": 256, "poly_degree": 5
        },
        "mixed_highnoise": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "mixed", "learning_rate": 0.025,
            "n_samples": 1000, "dim": 200, "noise_std": 5.0,
            "rho1": 0.5, "rho2": 0.25, "rho3": 0.5, "feature_dist": "correlated",
            "competing_learners": ALL_LEARNERS,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 3.0,
            "n_knots": 15, "n_bins": 256, "poly_degree": 5
        },
        "mixed_corr": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "mixed", "learning_rate": 0.025,
            "n_samples": 1000, "dim": 200, "noise_std": 2.0,
            "rho1": 0.95, "rho2": 0.8, "rho3": 0.95, "feature_dist": "correlated",
            "competing_learners": ALL_LEARNERS,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 3.0,
            "n_knots": 15, "n_bins": 256, "poly_degree": 5
        },
        "mixed_all": {
            "dataset_type": "synthetic", "dataset_name": "synthetic",
            "signal_type": "mixed", "learning_rate": 0.025,
            "n_samples": 500, "dim": 500, "noise_std": 5.0,
            "rho1": 0.75, "rho2": 0.5, "rho3": 0.75, "feature_dist": "correlated",
            "competing_learners": ALL_LEARNERS,
            "train_split": 0.5,
            "top_k": 5, "momentum_strength": 3.0, "target_df": 3.0,
            "n_knots": 15, "n_bins": 256, "poly_degree": 5
        },

        # --- 6. REAL Scenarios (lr=0.025) ---
        "real_bodyfat": {
            "dataset_type": "real", "dataset_name": "bodyfat",
            "learning_rate": 0.025,
            "competing_learners": LINEAR_SPLINE,
            "train_split": 0.85,
            "signal_type": "real", "n_samples": 0, "dim": 0, "noise_std": 0.0,
            "rho1": 0, "rho2": 0, "rho3": 0, "feature_dist": "real",
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 30, "n_bins": 256, "poly_degree": 5
        },
        "real_diabetes": {
            "dataset_type": "real", "dataset_name": "diabetes",
            "learning_rate": 0.025,
            "competing_learners": LINEAR_SPLINE,
            "train_split": 0.85,
            "signal_type": "real", "n_samples": 0, "dim": 0, "noise_std": 0.0,
            "rho1": 0, "rho2": 0, "rho3": 0, "feature_dist": "real",
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.5,
            "n_knots": 20, "n_bins": 256, "poly_degree": 5
        },
        "real_riboflavin": {
            "dataset_type": "real", "dataset_name": "riboflavin",
            "learning_rate": 0.025,
            "competing_learners": LINEAR,
            "train_split": 0.85,
            "signal_type": "real", "n_samples": 0, "dim": 0, "noise_std": 0.0,
            "rho1": 0, "rho2": 0, "rho3": 0, "feature_dist": "real",
            "top_k": 5, "momentum_strength": 3.0, "target_df": 1.0,
            "n_knots": 12, "n_bins": 256, "poly_degree": 5
        },
    }