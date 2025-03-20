class config:
    # Data generation parameters
    data_amount = 200       # Number of data points to generate
    SEED = 111              # Global random seed for reproducibility
    data_seed = 423         # Seed specifically for data generation
    train_split = 0.8       # Proportion of data to use for training
    train_size = int(data_amount * train_split)

    # Model parameters
    n_estimators = 1000     # Number of estimators for the CWB model
    learning_rate = 0.1     # Learning rate for the CWB model
    eval_freq = 1           # Evaluation frequency (iterations)
    batch_size = train_size       # Batch size for training
    batch_mode = "all"
    
    # Flooding parameters
    flood_level = 0        # Flooding level for the flooding loss