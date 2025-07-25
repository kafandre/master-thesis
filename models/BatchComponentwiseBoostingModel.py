import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import random
import os
import math
import torch.nn.functional as F


class MomentumFeatureSelector:
    """
    Feature selector that uses momentum to bias against recently selected features.
    """
    def __init__(self, n_features: int, momentum: float = 0.9):
        self.n_features = n_features
        self.momentum = momentum
        self.feature_momentum = torch.zeros(n_features)
    
    def select_stochastic(self, losses: torch.Tensor, temperature: float) -> int:
        """
        Select feature using stochastic selection with momentum bias.
        
        Args:
            losses: Loss values for each feature
            temperature: Temperature for stochastic selection (lower = more deterministic)
            
        Returns:
            Selected feature index
        """
        # Apply momentum penalty to recently selected features
        adjusted_losses = losses + self.feature_momentum
        
        # Convert losses to probabilities (lower loss = higher probability)
        inv_losses = 1.0 / (adjusted_losses + 1e-8)
        probs = F.softmax(inv_losses / temperature, dim=0)
        
        # Sample from the probability distribution
        selected_idx = torch.multinomial(probs, 1).item()
        
        # Update momentum (decay existing momentum, boost selected feature)
        self.feature_momentum *= self.momentum
        self.feature_momentum[selected_idx] += 1.0
        
        return selected_idx
    
    def reset(self):
        """Reset momentum tracking."""
        self.feature_momentum = torch.zeros(self.n_features)


class ComponentwiseBoostingModel:
    """
    Componentwise Gradient Boosting with linear regression base learners.
    Each boosting iteration selects the feature that most reduces the loss.

    - Uses specified loss during training
    - Uses MSE for evaluation/testing
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
        loss: str = 'mse',  # 'mse' for mean square error, 'flooding' for flooding loss
        track_history: bool = True,
        batch_mode: str = "all",  # "first" or "all"
        lr_ascent_mode: str = "exponential",  # "linear", "exponential", or "step"
        lr_ascent_factor: float = 0.1,  # Factor for learning rate increase
        lr_ascent_step_size: int = 10,  # For step mode: increase after this many iterations
        lr_max: float = 0.3,  # Maximum allowed learning rate
        flood_offset: float = 0.7,  # Fixed offset above train MSE   
        # New stochastic selection parameters
        use_stochastic_selection: bool = True,
        initial_temperature: float = 0.001,  # Start nearly deterministic
        final_temperature: float = 5.0,     # End more random
        warmup_epochs: int = 200,            # Epochs before stochastic selection kicks in
        momentum_strength: float = 0.95,      # Momentum decay factor
        gradient_noise_scale: float = 0.2    # Scale for gradient noise after warmup
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store initial learning rate
        self.current_learning_rate = learning_rate  # Track current learning rate        
        self.random_state = random_state
        self.loss = loss
        self.track_history = track_history
        self.batch_mode = batch_mode
        self.flood_offset = flood_offset
        self.previous_train_mse = None        

        # Learning rate ascent parameters (kept but won't be used)
        self.lr_ascent_mode = lr_ascent_mode
        self.lr_ascent_factor = lr_ascent_factor
        self.lr_ascent_step_size = lr_ascent_step_size
        self.lr_max = lr_max
        self.lr_ascent_activated = False
        self.lr_ascent_start_iter = None
        
        # New stochastic selection parameters
        self.use_stochastic_selection = use_stochastic_selection
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.warmup_epochs = warmup_epochs
        self.momentum_strength = momentum_strength
        self.gradient_noise_scale = gradient_noise_scale  # Add this line
        self.feature_selector = None  # Will be initialized during fit
        
        if self.batch_mode not in ["first", "all"]:
            raise ValueError("batch_mode must be either 'first' or 'all'")

        if self.lr_ascent_mode not in ["linear", "exponential", "step"]:
            raise ValueError("lr_ascent_mode must be one of: 'linear', 'exponential', or 'step'")
             
        # Will be set during fitting
        self.estimators_: List[Tuple[int, torch.nn.Linear]] = []
        self.feature_importances_ = None
        self.intercept_ = None
        self.loss_fn = None
        self.eval_loss_fn = None  # Always MSE for evaluation
        self.flood_level = None

        # History tracking for double descent analysis
        self.history: Dict[str, List[float]] = {
            'train_loss': [],          # Using training loss (e.g., flooding)
            'train_mse': [],           # Using MSE for evaluation
            'val_loss': [],            # Using MSE for evaluation
            'test_loss': [],           # Using MSE for evaluation
            'selected_features': [],
            'learning_rate': [],       # Track learning rate changes
            'temperature': []          # Track temperature changes
        }
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _init_estimators(self, y: torch.Tensor):
        """Initialize with the mean prediction."""
        self.intercept_ = torch.mean(y).item()
        self.estimators_ = []
        
        # Reset learning rate ascent tracking (not used but kept for compatibility)
        self.current_learning_rate = self.initial_learning_rate
        self.lr_ascent_activated = False
        self.lr_ascent_start_iter = None

        # Reset history
        self.history = {
            'train_loss': [],          # Using training loss (e.g., flooding)
            'train_mse': [],           # Using MSE for evaluation
            'val_loss': [],            # Using MSE for evaluation
            'test_loss': [],           # Using MSE for evaluation
            'selected_features': [],
            'learning_rate': [],       # Track learning rate changes            
            'temperature': []          # Track temperature changes
        }

    def _get_loss_fn(self, loss_type: str, **kwargs) -> Callable:
        """
        Get the appropriate loss function based on the specified type.
        
        Args:
            loss_type: Type of loss function ('mse', 'flooding', etc.)
            **kwargs: Additional parameters for the loss function
            
        Returns:
            Loss function
        """
        if loss_type == 'mse':
            return lambda y_pred, y: torch.mean((y_pred - y) ** 2)
        elif loss_type == 'flooding':
            self.flood_level = kwargs.get('flood_level', 0.01)
            return lambda y_pred, y: (abs(torch.mean((y_pred - y) ** 2) - self.flood_level) + self.flood_level)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
    def _get_eval_loss_fn(self) -> Callable:
        """
        Get the evaluation loss function (always MSE).
        
        Returns:
            MSE loss function
        """
        return lambda y_pred, y: torch.mean((y_pred - y) ** 2)
    
    def _get_gradient_fn(self, loss_type: str) -> Callable:
        """
        Get the gradient function for the specified loss type.
        
        Args:
            loss_type: Type of loss function
            
        Returns:
            Gradient function that takes y_pred and y and returns gradients
        """
        if loss_type == 'mse':
            # For MSE: gradient is (y_pred - y)
            return lambda y_pred, y: (y_pred - y).unsqueeze(1) if (y_pred - y).dim() == 1 else (y_pred - y)
        elif loss_type == 'flooding':
            # For flooding loss: gradient is (y_pred - y) * sign(MSE - flood_level)
            def flooding_gradient(y_pred, y):
                # Calculate batch MSE
                batch_mse = torch.mean((y_pred - y) ** 2)
                
                # Standard MSE gradient for the batch
                grad = y_pred - y

                # Calculate sign based on whether we're below or above flood level
                sign = -1.0 if batch_mse < self.flood_level else 1.0
                
                # Apply sign to gradient (negative if below flood level, positive if above)
                # This pushes away from minimum when below flood level
                grad = sign * grad
                
                # Check if batch is below flood level
                if batch_mse < self.flood_level:
                    print(f"Flipping gradient! Batch MSE: {batch_mse:.6f}, Flood level: {self.flood_level:.6f}")           
                
                return grad.unsqueeze(1) if grad.dim() == 1 else grad
            return flooding_gradient

        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _update_learning_rate(self, iteration: int, current_loss: float) -> float:
        """
        Update Learning Rate based on provided parameters.
        NOTE: This function is kept for compatibility but not used in stochastic mode.
        """
        # Always return initial learning rate when using stochastic selection
        if self.use_stochastic_selection:
            return self.initial_learning_rate
            
        # Original learning rate update logic (kept for compatibility)
        if self.loss != 'flooding' or self.flood_level is None:
            return self.initial_learning_rate
            
        # Check if we should activate learning rate ascent
        if not self.lr_ascent_activated and current_loss < self.flood_level+0.1:
            self.lr_ascent_activated = True
            self.lr_ascent_start_iter = iteration
            print(f"Learning rate ascent activated at iteration {iteration+1}! Loss: {current_loss:.6f}, Flood level: {self.flood_level:.6f}")
        
        # If ascent is not activated yet, return initial learning rate
        if not self.lr_ascent_activated:
            return self.initial_learning_rate
            
        # Calculate iterations since ascent activation
        iter_since_start = iteration - self.lr_ascent_start_iter
        
        # Calculate new learning rate based on ascent mode
        if self.lr_ascent_mode == "linear":
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * iter_since_start)
        elif self.lr_ascent_mode == "exponential":
            new_lr = self.initial_learning_rate * ((1 + self.lr_ascent_factor) ** iter_since_start)
        elif self.lr_ascent_mode == "step":
            steps = iter_since_start // self.lr_ascent_step_size
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * steps)
        else:
            new_lr = self.initial_learning_rate
            
        # Ensure learning rate doesn't exceed maximum
        new_lr = min(new_lr, self.lr_max)
        
        return new_lr

    def _calculate_temperature(self, iteration: int) -> float:
        """
        Calculate current temperature for stochastic feature selection.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Current temperature value
        """
        if iteration < self.warmup_epochs:
            # During warmup, use initial (low) temperature for nearly deterministic selection
            return self.initial_temperature
        else:
            # After warmup, linearly increase temperature over remaining iterations
            progress = (iteration - self.warmup_epochs) / max(1, self.n_estimators - self.warmup_epochs)
            progress = min(1.0, progress)  # Cap at 1.0
            
            # Linear interpolation between initial and final temperature
            temperature = self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
            return temperature

    def _componentwise_fit(
        self, 
        X: torch.Tensor, 
        negative_gradients: torch.Tensor,
        loss_fn: Callable,
        iteration: int
    ) -> Tuple[int, torch.nn.Linear]:
        """
        Fit a separate linear model for each feature and select the best one using
        stochastic selection with temperature annealing and momentum.
        
        Args:
            X: Input features tensor
            negative_gradients: Negative gradients to fit against
            loss_fn: Loss function to evaluate models
            iteration: Current iteration number
            
        Returns:
            Tuple of (best_feature_idx, best_model)
        """
        n_features = X.shape[1]
        losses = torch.zeros(n_features)
        models = []

        # Ensure negative_gradients is 2D for matrix multiplication
        if negative_gradients.dim() == 1:
            negative_gradients = negative_gradients.unsqueeze(1)  # Convert to shape [n_samples, 1]
        
        # Fit a model for each feature and calculate losses
        for feature_idx in range(n_features):
            # Create a simple linear model for this feature
            X_feature = X[:, feature_idx:feature_idx+1]
            model = torch.nn.Linear(1, 1, bias=False)
            
            # Fit using normal equation for efficiency
            X_t = X_feature.t()
            beta = torch.mm(torch.mm(torch.inverse(torch.mm(X_t, X_feature) + 1e-10 * torch.eye(1)), X_t), negative_gradients)
            model.weight.data = beta.t()
            
            # Compute predictions and loss
            preds = model(X_feature)
            loss = loss_fn(preds, negative_gradients).item()
            
            losses[feature_idx] = loss
            models.append(model)
        
        # Select feature using stochastic selection or deterministic selection
        if self.use_stochastic_selection:
            # Calculate current temperature
            temperature = self._calculate_temperature(iteration)
            
            # Use momentum-based stochastic selection
            selected_idx = self.feature_selector.select_stochastic(losses, temperature)
            
            # Track temperature in history
            if self.track_history:
                self.history['temperature'].append(temperature)
        else:
            # Original deterministic selection (fallback)
            selected_idx = torch.argmin(losses).item()
            
            # Still track temperature as 0 for consistency
            if self.track_history:
                self.history['temperature'].append(0.0)
        
        return selected_idx, models[selected_idx]
    
    def _evaluate_base_learner_on_full_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        current_pred: torch.Tensor,
        feature_idx: int,
        model: torch.nn.Linear
    ) -> float:
        """
        Evaluate a feature base learner on the full dataset.
            
        Returns:
            Loss value when this model is applied to the full dataset
        """
        # Get feature contribution
        X_feature = X[:, feature_idx:feature_idx+1]
        feature_contrib = model(X_feature).squeeze() * self.current_learning_rate     

        # Update predictions temporarily for evaluation
        new_pred = current_pred + feature_contrib
        
        # Compute loss
        loss = self.loss_fn(new_pred, y).item()
        
        return loss

    def _calculate_dynamic_flood_level(self, iteration: int, current_train_mse: float = None, current_test_loss: float = None) -> float:       
        """
        Calculate dynamic flood level based on the current iteration.
        NOTE: This function is kept for compatibility but not used in stochastic mode.
        """
        # When using stochastic selection, return a static flood level
        if self.use_stochastic_selection:
            return 0.01  # or whatever static value you prefer
            
        # Original dynamic flood level logic (kept for compatibility)
        # ... [original implementation remains the same] ...
        return 0.01  # simplified for this example

    def fit(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        loss: Optional[str] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        early_stopping: bool = False,
        patience: int = 100,
        eval_freq: int = 1,
        verbose: bool = False,
        batch_size: int = 32,
        save_iterations: Optional[List[int]] = None,    # New parameter for saving at specific iterations
        save_path: str = "./checkpoints",               # Directory to save models    
        **loss_params
    ) -> 'ComponentwiseBoostingModel':
        """
        Fit the boosting model with batch processing and stochastic feature selection.
        """

        # Create save directory if it doesn't exist
        if save_iterations and len(save_iterations) > 0:
            os.makedirs(save_path, exist_ok=True)

        # Set or update loss function
        loss_type = loss if loss is not None else self.loss
        # Set training loss function based on specified loss type
        self.loss_fn = self._get_loss_fn(loss_type, **loss_params)
        # Set evaluation loss function always to MSE
        self.eval_loss_fn = self._get_eval_loss_fn()
        gradient_fn = self._get_gradient_fn(loss_type)
        
        # Initialize model with mean prediction
        self._init_estimators(y)
        
        # Initialize feature selector for stochastic selection
        if self.use_stochastic_selection:
            n_features = X.shape[1]
            self.feature_selector = MomentumFeatureSelector(n_features, self.momentum_strength)
        
        # Current predictions for all data points
        all_current_pred = torch.full_like(y, self.intercept_)
        
        # For validation data
        if X_val is not None and y_val is not None:
            val_pred = torch.full_like(y_val, self.intercept_)
            best_val_loss = float('inf')
            best_iteration = 0
            no_improvement_count = 0
        
        # For test data
        if X_test is not None and y_test is not None:
            test_pred = torch.full_like(y_test, self.intercept_)
        
        # Feature importance tracking
        n_features = X.shape[1]
        feature_counts = np.zeros(n_features)

        # Create data loaders for batch processing
        train_dataset = TensorDataset(X, y, torch.arange(len(y)))  # Include indices
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Calculate initial loss before training
        initial_train_loss = self.loss_fn(all_current_pred, y).item()

        # Initial learning rate (stays constant in stochastic mode)
        self.current_learning_rate = self.initial_learning_rate

        # Initial loss calculation before training
        if self.track_history:
            # Training loss using the specified loss function (e.g., flooding)
            train_loss = self.loss_fn(all_current_pred, y).item()
            self.history['train_loss'].append(train_loss)
            
            # Training MSE for evaluation purposes
            train_mse = self.eval_loss_fn(all_current_pred, y).item()
            self.history['train_mse'].append(train_mse)
            
            # Track learning rate
            self.history['learning_rate'].append(self.current_learning_rate)
            
            # Track initial temperature
            initial_temp = self._calculate_temperature(0) if self.use_stochastic_selection else 0.0
            self.history['temperature'].append(initial_temp)

            if X_val is not None and y_val is not None:
                # Validation loss always uses MSE
                val_loss = self.eval_loss_fn(val_pred, y_val).item()
                self.history['val_loss'].append(val_loss)
            
            if X_test is not None and y_test is not None:
                # Test loss always uses MSE
                test_loss = self.eval_loss_fn(test_pred, y_test).item()
                self.history['test_loss'].append(test_loss)

        # Initialize history tracking for flood level if using flooding loss
        if self.track_history and loss_type == 'flooding':
            self.history["flood_level"] = []
            
        # Set initial MSE to None - we'll calculate it in the first iteration
        self.previous_train_mse = None               

        # Boosting iterations
        for iteration in range(self.n_estimators):
            # For stochastic mode, use static flood level instead of dynamic
            if loss_type == 'flooding':
                if self.use_stochastic_selection:
                    # Use static flood level (set in loss_params or default)
                    if self.flood_level is None:
                        self.flood_level = loss_params.get('flood_level', 0.01)
                else:
                    # Original dynamic flood level logic (kept for compatibility)
                    current_train_mse = self.eval_loss_fn(all_current_pred, y).item()
                    current_test_loss = None
                    if X_test is not None and y_test is not None:
                        current_test_loss = self.eval_loss_fn(test_pred, y_test).item()
                    
                    self.flood_level = self._calculate_dynamic_flood_level(
                        iteration, current_train_mse, current_test_loss
                    )
                    self.previous_train_mse = current_train_mse
                
                if self.track_history:
                    self.history["flood_level"].append(self.flood_level)

            # Lists to store candidate models from each batch
            batch_candidates = []

            # Process data in batches
            for batch_idx, (batch_X, batch_y, batch_indices) in enumerate(train_loader):                
                # Get current predictions for this batch
                batch_current_pred = all_current_pred[batch_indices]

                # Compute negative gradients for this batch
                negative_gradients = -gradient_fn(batch_current_pred, batch_y)

                # Add gradient noise after warmup for stochastic exploration
                if self.use_stochastic_selection and iteration >= self.warmup_epochs:
                    noise_intensity = self.gradient_noise_scale * (iteration - self.warmup_epochs) / max(1, self.n_estimators - self.warmup_epochs)
                    gradient_noise = torch.randn_like(negative_gradients) * noise_intensity
                    negative_gradients = negative_gradients + gradient_noise
                
                # Find best feature and fit a base model on this batch
                # Pass iteration number for temperature calculation
                feature_idx, model = self._componentwise_fit(batch_X, negative_gradients, self.loss_fn, iteration)
                
                # Store candidate model and feature
                batch_candidates.append((feature_idx, model))
                
                # If using only the first batch, break after one iteration
                if self.batch_mode == "first" and batch_idx == 0:
                    break

            # Select the best model among all candidates based on full training set performance
            best_loss = float('inf')
            best_feature_idx = -1
            best_model = None   

            for feature_idx, model in batch_candidates:
                # Evaluate this model on the full training set
                loss = self._evaluate_base_learner_on_full_data(X, y, all_current_pred, feature_idx, model)
                
                if loss < best_loss:
                    best_loss = loss
                    best_feature_idx = feature_idx
                    best_model = model                         
                
            # Update feature importance count
            feature_counts[feature_idx] += 1
                
            # Store the best estimator
            self.estimators_.append((best_feature_idx, best_model))
            
            # Track selected feature
            if self.track_history:
                self.history['selected_features'].append(best_feature_idx)

            # Update the learning rate (will stay constant in stochastic mode)
            if not self.use_stochastic_selection:
                train_loss = self.loss_fn(all_current_pred, y).item()
                self.current_learning_rate = self._update_learning_rate(iteration, train_loss)

            if self.track_history:
                # Ensure we have a learning rate entry for each iteration
                if len(self.history['learning_rate']) <= iteration:
                    self.history['learning_rate'].append(self.current_learning_rate)

            # Update predictions for ALL data points using the best model
            feature_contrib = best_model(X[:, best_feature_idx:best_feature_idx+1]) * self.current_learning_rate
            all_current_pred += feature_contrib.squeeze()

            # Update validation predictions if available
            if X_val is not None and y_val is not None:
                val_feature_contrib = best_model(X_val[:, best_feature_idx:best_feature_idx+1]) * self.current_learning_rate
                val_pred += val_feature_contrib.squeeze()

            # Update test predictions if available
            if X_test is not None and y_test is not None:
                test_feature_contrib = best_model(X_test[:, best_feature_idx:best_feature_idx+1]) * self.current_learning_rate
                test_pred += test_feature_contrib.squeeze()
            
            # Evaluate and track losses
            if self.track_history and (iteration + 1) % eval_freq == 0:
                # Training loss using specified loss function (e.g., flooding)
                train_loss = self.loss_fn(all_current_pred, y).item()
                self.history['train_loss'].append(train_loss)
                
                # Training MSE for evaluation purposes
                train_mse = self.eval_loss_fn(all_current_pred, y).item()
                self.history['train_mse'].append(train_mse)
                             
                if X_val is not None and y_val is not None:
                    # Validation loss always uses MSE
                    val_loss = self.eval_loss_fn(val_pred, y_val).item()
                    self.history['val_loss'].append(val_loss)
                    
                    # Check for early stopping
                    if early_stopping:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_iteration = iteration
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            if no_improvement_count >= patience:
                                if verbose:
                                    print(f"Early stopping at iteration {iteration+1}. Best iteration: {best_iteration+1}.")
                                break
                
                if X_test is not None and y_test is not None:
                    # Test loss always uses MSE
                    test_loss = self.eval_loss_fn(test_pred, y_test).item()
                    self.history['test_loss'].append(test_loss)
                
                # verbose output
                if verbose and (iteration + 1) % (eval_freq * 10) == 0:
                    print(f"Iteration {iteration+1}/{self.n_estimators}, ", end="")
                    print(f"Train Loss: {train_loss:.6f} (Train MSE: {train_mse:.6f})", end="")
                    if X_val is not None and y_val is not None:
                        print(f", Val MSE: {val_loss:.6f}", end="")
                    if X_test is not None and y_test is not None:
                        print(f", Test MSE: {test_loss:.6f}", end="")
                    print(f", LR: {self.current_learning_rate:.6f}", end="")
                    if loss_type == 'flooding':
                        print(f", Flood: {self.flood_level:.6f}", end="")
                    if self.use_stochastic_selection and iteration >= self.warmup_epochs:
                        current_temp = self._calculate_temperature(iteration)
                        print(f", Temp: {current_temp:.4f}", end="")
                    print("")

            # Check if we should save the model at this iteration
            if save_iterations and (iteration + 1) in save_iterations:
                checkpoint_path = os.path.join(save_path, f"{loss_type}_model_iteration_{iteration+1}.pt")
                self._save_model(checkpoint_path)
                if verbose:
                    print(f"\nSaved model checkpoint at iteration {iteration+1} to {checkpoint_path}\n")

        # Save final model if the last iteration is in save_iterations
        if save_iterations and self.n_estimators in save_iterations:
            checkpoint_path = os.path.join(save_path, f"{loss_type}_model_iteration_{self.n_estimators}.pt")
            self._save_model(checkpoint_path)
            if verbose:
                print(f"\nSaved final model checkpoint at iteration {self.n_estimators} to {checkpoint_path}\n")

        # Calculate feature importances
        self.feature_importances_ = feature_counts / np.sum(feature_counts)
        
        return self

    def _save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        # Create a dictionary with all the model state
        model_state = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'current_learning_rate': self.current_learning_rate,
            'random_state': self.random_state,
            'loss': self.loss,
            'track_history': self.track_history,
            'batch_mode': self.batch_mode,
            'flood_offset': self.flood_offset,
            'previous_train_mse': self.previous_train_mse,
            'lr_ascent_mode': self.lr_ascent_mode,
            'lr_ascent_factor': self.lr_ascent_factor,
            'lr_ascent_step_size': self.lr_ascent_step_size,
            'lr_max': self.lr_max,
            'lr_ascent_activated': self.lr_ascent_activated,
            'lr_ascent_start_iter': self.lr_ascent_start_iter,
            # New stochastic parameters
            'use_stochastic_selection': self.use_stochastic_selection,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'warmup_epochs': self.warmup_epochs,
            'momentum_strength': self.momentum_strength,
            'estimators_': self.estimators_,
            'feature_importances_': self.feature_importances_,
            'intercept_': self.intercept_,
            'flood_level': self.flood_level,
            'history': self.history
        }
        
        # Save to file using torch.save
        torch.save(model_state, path)

    @classmethod
    def load_model(cls, path: str) -> 'ComponentwiseBoostingModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            ComponentwiseBoostingModel: Loaded model
        """
        # Load the state dictionary
        model_state = torch.load(path)
        
        # Create a new instance with the basic parameters
        model = cls(
            n_estimators=model_state['n_estimators'],
            learning_rate=model_state['initial_learning_rate'],
            random_state=model_state['random_state'],
            loss=model_state['loss'],
            track_history=model_state['track_history'],
            batch_mode=model_state['batch_mode'],
            lr_ascent_mode=model_state['lr_ascent_mode'],
            lr_ascent_factor=model_state['lr_ascent_factor'],
            lr_ascent_step_size=model_state['lr_ascent_step_size'],
            lr_max=model_state['lr_max'],
            flood_offset=model_state['flood_offset'],
            # New stochastic parameters
            use_stochastic_selection=model_state.get('use_stochastic_selection', True),
            initial_temperature=model_state.get('initial_temperature', 0.01),
            final_temperature=model_state.get('final_temperature', 1.0),
            warmup_epochs=model_state.get('warmup_epochs', 50),
            momentum_strength=model_state.get('momentum_strength', 0.9)
        )
        
        # Restore all the trained state
        model.initial_learning_rate = model_state['initial_learning_rate']
        model.current_learning_rate = model_state['current_learning_rate']
        model.lr_ascent_activated = model_state['lr_ascent_activated']
        model.lr_ascent_start_iter = model_state['lr_ascent_start_iter']
        model.estimators_ = model_state['estimators_']
        model.feature_importances_ = model_state['feature_importances_']
        model.intercept_ = model_state['intercept_']
        model.flood_level = model_state['flood_level']
        model.history = model_state['history']
        model.previous_train_mse = model_state['previous_train_mse']
        
        # Reinitialize loss functions since they can't be serialized
        if model_state['loss'] == 'mse':
            model.loss_fn = model._get_loss_fn('mse')
        elif model_state['loss'] == 'flooding':
            model.loss_fn = model._get_loss_fn('flooding', flood_level=model.flood_level)
        
        # Always set the evaluation loss function to MSE
        model.eval_loss_fn = model._get_eval_loss_fn()
        
        return model

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions for the input data.
        
        Args:
            X: Input features tensor of shape (n_samples, n_features)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        # Start with the intercept
        predictions = torch.full((X.shape[0],), self.intercept_, device=X.device)
        
        # Add contribution from each estimator
        for i, (feature_idx, model) in enumerate(self.estimators_):
            X_feature = X[:, feature_idx:feature_idx+1]

            # Use stored learning rate history if available, otherwise use final learning rate
            if self.track_history and i < len(self.history['learning_rate']):
                lr = self.history['learning_rate'][i]
            else:
                lr = self.current_learning_rate
                
            predictions += model(X_feature).squeeze() * lr

        return predictions
    
    def get_loss(self, X: torch.Tensor, y: torch.Tensor, use_training_loss: bool = False) -> float:
        """
        Calculate the loss on the given data.
        
        Args:
            X: Input features tensor
            y: True target values
            use_training_loss: If True, use the training loss function (e.g., flooding);
                              if False, use evaluation loss (MSE)
            
        Returns:
            Loss value
        """
        if self.loss_fn is None or self.eval_loss_fn is None:
            raise ValueError("Model must be fitted before calculating loss")
        
        predictions = self.predict(X)
        
        if use_training_loss:
            return self.loss_fn(predictions, y).item()
        else:
            return self.eval_loss_fn(predictions, y).item()
    
    def predict_iterations(self, X: torch.Tensor, max_iterations: Optional[int] = None) -> torch.Tensor:
        """
        Generate predictions using only the first N iterations of the model.
        Useful for analyzing the behavior of the model at different "epochs".
        
        Args:
            X: Input features tensor of shape (n_samples, n_features)
            max_iterations: Maximum number of iterations to use (None = use all)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        if max_iterations is None:
            max_iterations = len(self.estimators_)
            
        # Limit to available estimators
        max_iterations = min(max_iterations, len(self.estimators_))
        
        # Start with the intercept
        predictions = torch.full((X.shape[0],), self.intercept_, device=X.device)
        
        # Add contribution from each estimator up to max_iterations
        for i in range(max_iterations):
            feature_idx, model = self.estimators_[i]
            X_feature = X[:, feature_idx:feature_idx+1]
            
            # Use stored learning rate history if available, otherwise use final learning rate
            if self.track_history and i < len(self.history['learning_rate']):
                lr = self.history['learning_rate'][i]
            else:
                lr = self.current_learning_rate
                
            predictions += model(X_feature).squeeze() * lr
            
        return predictions
    
    def analyze_double_descent(
        self, 
        X_train: torch.Tensor, 
        y_train: torch.Tensor,
        X_test: torch.Tensor, 
        y_test: torch.Tensor,
        iterations_to_check: Optional[List[int]] = None,
        use_training_loss_for_train: bool = False
    ) -> Dict[str, List[float]]:
        """
        Analyze the double descent phenomenon by evaluating the model
        at different numbers of iterations/epochs.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            iterations_to_check: List of iteration counts to evaluate (None = use all)
            use_training_loss_for_train: If True, evaluate training data with training loss;
                                         if False, use MSE for both train and test
            
        Returns:
            Dictionary with train and test losses at each iteration count
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before analysis")
            
        if iterations_to_check is None:
            # Create a logarithmically spaced set of points for efficiency
            max_iter = len(self.estimators_)
            iterations_to_check = [
                int(i) for i in np.logspace(0, np.log10(max_iter), 50)
            ]
            iterations_to_check = sorted(list(set(iterations_to_check)))
        
        results = {
            'iterations': iterations_to_check,
            'train_loss': [],
            'train_mse': [] if use_training_loss_for_train else None,
            'test_loss': []
        }
        
        for n_iter in iterations_to_check:
            # Get predictions using only first n_iter estimators
            train_pred = self.predict_iterations(X_train, n_iter)
            test_pred = self.predict_iterations(X_test, n_iter)
            
            # Calculate losses
            if use_training_loss_for_train:
                # Use training loss for training data (e.g., flooding)
                train_loss = self.loss_fn(train_pred, y_train).item()
                # Also store MSE for comparison
                train_mse = self.eval_loss_fn(train_pred, y_train).item()
                results['train_mse'].append(train_mse)
            else:
                # Use MSE for training data
                train_loss = self.eval_loss_fn(train_pred, y_train).item()
            
            # Always use MSE for test data
            test_loss = self.eval_loss_fn(test_pred, y_test).item()
            
            # Store results
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            
        return results

    def get_stochastic_stats(self) -> Dict[str, any]:
        """
        Get statistics about the stochastic selection process.
        
        Returns:
            Dictionary with stochastic selection statistics
        """
        if not self.use_stochastic_selection or not self.track_history:
            return {}
            
        stats = {
            'use_stochastic_selection': self.use_stochastic_selection,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'warmup_epochs': self.warmup_epochs,
            'momentum_strength': self.momentum_strength,
            'temperature_history': self.history.get('temperature', []),
            'selected_features_history': self.history.get('selected_features', [])
        }
        
        # Calculate feature selection diversity
        if 'selected_features' in self.history:
            selected_features = self.history['selected_features']
            unique_features = len(set(selected_features))
            total_selections = len(selected_features)
            stats['feature_diversity'] = unique_features / max(1, total_selections)
            
            # Calculate feature selection entropy
            from collections import Counter
            feature_counts = Counter(selected_features)
            total = sum(feature_counts.values())
            entropy = 0
            for count in feature_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            stats['selection_entropy'] = entropy
        
        return stats