import torch
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import random
import os
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class ComponentwiseBoostingModel:
    """
    Componentwise Gradient Boosting with various regression base learners.
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
        base_learner: str = "linear",  # "linear", "polynomial", "tree"
        poly_degree: int = 2,  # For polynomial base learner
        tree_max_depth: int = 3,  # For tree base learner
        lr_ascent_mode: str = "step",  # "linear", "exponential", or "step"
        lr_ascent_factor: float = 1.0,  # Factor for learning rate increase
        lr_ascent_step_size: int = 50,  # For step mode: increase after this many iterations
        lr_max: float = 0.5,  # Maximum allowed learning rate
        top_k_selection: int = 10, # Number of top features to randomly select from
        top_k_early: int = 1,
        top_k_late: int = 1
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store initial learning rate
        self.current_learning_rate = learning_rate  # Track current learning rate        
        self.random_state = random_state
        self.loss = loss
        self.track_history = track_history
        self.base_learner = base_learner
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth

        # Learning rate ascent parameters
        self.lr_ascent_mode = lr_ascent_mode
        self.lr_ascent_factor = lr_ascent_factor
        self.lr_ascent_step_size = lr_ascent_step_size
        self.lr_max = lr_max
        self.lr_ascent_activated = False
        self.lr_ascent_start_iter = None 

        # Top-k feature selection parameter
        self.top_k_selection = top_k_selection
        self.top_k_early = top_k_early
        self.top_k_late = top_k_late
        self.stochastic_selection_activated = False

        if self.lr_ascent_mode not in ["linear", "exponential", "step"]:
            raise ValueError("lr_ascent_mode must be one of: 'linear', 'exponential', or 'step'")
            
        if self.base_learner not in ["linear", "polynomial", "tree"]:
            raise ValueError("base_learner must be one of: 'linear', 'polynomial', or 'tree'")
             
        # Will be set during fitting
        self.estimators_: List[Tuple[int, object]] = []
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
            'learning_rate': []        # Track learning rate changes            
        }
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _init_estimators(self, y: torch.Tensor):
        """Initialize with the mean prediction."""
        self.intercept_ = torch.mean(y).item()
        self.estimators_ = []
        
        # Reset learning rate ascent tracking
        self.current_learning_rate = self.initial_learning_rate
        self.lr_ascent_activated = False
        self.lr_ascent_start_iter = None
        self.stochastic_selection_activated = False

        # Reset history
        self.history = {
            'train_loss': [],          # Using training loss (e.g., flooding)
            'train_mse': [],           # Using MSE for evaluation
            'val_loss': [],            # Using MSE for evaluation
            'test_loss': [],           # Using MSE for evaluation
            'selected_features': [],
            'learning_rate': []        # Track learning rate changes            
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
                # Calculate MSE
                mse = torch.mean((y_pred - y) ** 2)
                
                # Standard MSE gradient
                grad = y_pred - y

                # Calculate sign based on whether we're below or above flood level
                sign = -1.0 if mse < self.flood_level else 1.0
                
                # Apply sign to gradient (negative if below flood level, positive if above)
                # This pushes away from minimum when below flood level
                grad = sign * grad
                
                # Check if below flood level
                if mse < self.flood_level:
                    print(f"Flipping gradient! MSE: {mse:.6f}, Flood level: {self.flood_level:.6f}")           
                
                return grad.unsqueeze(1) if grad.dim() == 1 else grad
            return flooding_gradient

        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _update_learning_rate(self, iteration: int, current_loss: float) -> float:
        """
        Update Learning Rate based on provided parameters.
        """
        # If not using flooding loss, don't activate learning rate ascent
        if self.loss != 'flooding' or self.flood_level is None:
            return self.initial_learning_rate
            
        # Check if we should activate learning rate ascent
        if not self.lr_ascent_activated and current_loss < self.flood_level + 0.1:
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
            # Linear increase: lr = initial_lr * (1 + factor * iter_since_start)
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * iter_since_start)
            
        elif self.lr_ascent_mode == "exponential":
            # Exponential increase: lr = initial_lr * (1 + factor) ^ iter_since_start
            new_lr = self.initial_learning_rate * ((1 + self.lr_ascent_factor) ** iter_since_start)
            
        elif self.lr_ascent_mode == "step":
            # Step increase: lr = initial_lr * (1 + factor * floor(iter_since_start / step_size))
            steps = iter_since_start // self.lr_ascent_step_size
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * steps)
            
        else:
            # Default to no change
            new_lr = self.initial_learning_rate
            
        # Ensure learning rate doesn't exceed maximum
        new_lr = min(new_lr, self.lr_max)
        
        return new_lr

    def _create_base_learner(self, feature_idx: int) -> object:
        """
        Create a base learner based on the specified type.
        
        Args:
            feature_idx: Index of the feature (not used for all learner types)
            
        Returns:
            Base learner object
        """
        if self.base_learner == "linear":
            return torch.nn.Linear(1, 1, bias=False)
        elif self.base_learner == "polynomial":
            # Return a polynomial model wrapper
            return PolynomialRegressionWrapper(degree=self.poly_degree)
        elif self.base_learner == "tree":
            return DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown base learner: {self.base_learner}")

    def _fit_base_learner(self, model: object, X_feature: torch.Tensor, y_target: torch.Tensor) -> object:
        """
        Fit a base learner to the data.
        
        Args:
            model: Base learner model
            X_feature: Input features (single feature for linear, multiple for others)
            y_target: Target values
            
        Returns:
            Fitted model
        """
        if self.base_learner == "linear":
            # Original normal equation approach
            X_t = X_feature.t()
            beta = torch.mm(torch.mm(torch.inverse(torch.mm(X_t, X_feature) + 1e-10 * torch.eye(1)), X_t), y_target)
            model.weight.data = beta.t()
            return model
            
        elif self.base_learner in ["polynomial", "tree"]:
            # Convert to numpy for sklearn compatibility
            X_np = X_feature.detach().numpy()
            y_np = y_target.squeeze().detach().numpy()
            
            # Fit the model
            model.fit(X_np, y_np)
            return model
            
        else:
            raise ValueError(f"Unknown base learner: {self.base_learner}")

    def _predict_base_learner(self, model: object, X_feature: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with a base learner.
        
        Args:
            model: Fitted base learner
            X_feature: Input features
            
        Returns:
            Predictions as torch tensor
        """
        if self.base_learner == "linear":
            return model(X_feature)
            
        elif self.base_learner in ["polynomial", "tree"]:
            # Convert to numpy, predict, convert back to torch
            X_np = X_feature.detach().numpy()
            pred_np = model.predict(X_np)
            return torch.tensor(pred_np, dtype=torch.float32).unsqueeze(1)
            
        else:
            raise ValueError(f"Unknown base learner: {self.base_learner}")

    def _componentwise_fit(
        self, 
        X: torch.Tensor, 
        negative_gradients: torch.Tensor,
        iteration: int
    ) -> Tuple[int, object]:
        """
        Fit a separate model for each feature and select the best one.
        
        Args:
            X: Input features tensor
            negative_gradients: Negative gradients to fit against
            iteration: Current iteration number
            
        Returns:
            Tuple of (best_feature_idx, best_model)
        """
        n_features = X.shape[1]
        losses = []
        models = []

        # Ensure negative_gradients is 2D for compatibility
        if negative_gradients.dim() == 1:
            negative_gradients = negative_gradients.unsqueeze(1)

        # Try each feature separately
        for feature_idx in range(n_features):
            # Get feature data
            if self.base_learner == "linear":
                X_feature = X[:, feature_idx:feature_idx+1]
            else:
                # For polynomial and tree learners, we might want to use all features
                # but we'll still iterate through them for consistency
                X_feature = X[:, feature_idx:feature_idx+1]
            
            # Create and fit base learner
            model = self._create_base_learner(feature_idx)
            model = self._fit_base_learner(model, X_feature, negative_gradients)
            
            # Compute predictions and loss
            preds = self._predict_base_learner(model, X_feature)
            loss = torch.mean((preds.squeeze() - negative_gradients.squeeze()) ** 2).item()
            
            losses.append(loss)
            models.append(model)

        # Convert losses to tensor for easier manipulation
        losses_tensor = torch.tensor(losses)
        
        # Select feature based on whether stochastic selection is activated
        if self.stochastic_selection_activated:
            # Random selection from top-k features
            iterations_since_activation = iteration - self.lr_ascent_start_iter
            if iterations_since_activation < 50:  # First 50 iterations after activation
                k = self.top_k_early  # Conservative adaptation
            else:
                k = self.top_k_late  # Aggressive exploration

            top_k_indices = torch.topk(losses_tensor, k, largest=False).indices
            selected_idx = top_k_indices[torch.randint(0, k, (1,))].item()
            print(f"Stochastic top-{k} selection: chose feature {selected_idx} from {top_k_indices.tolist()}")
        else:
            # Deterministic selection (original behavior)
            selected_idx = torch.argmin(losses_tensor).item()
        
        return selected_idx, models[selected_idx]

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
        save_iterations: Optional[List[int]] = None,
        save_path: str = "./checkpoints",
        **loss_params
    ) -> 'ComponentwiseBoostingModel':
        """
        Fit the boosting model using standard componentwise gradient boosting.
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
        
        # Current predictions for all data points
        current_pred = torch.full_like(y, self.intercept_)
        
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

        # Initial learning rate
        self.current_learning_rate = self.initial_learning_rate

        # Initial loss calculation before training
        if self.track_history:
            # Training loss using the specified loss function (e.g., flooding)
            train_loss = self.loss_fn(current_pred, y).item()
            self.history['train_loss'].append(train_loss)
            
            # Training MSE for evaluation purposes
            train_mse = self.eval_loss_fn(current_pred, y).item()
            self.history['train_mse'].append(train_mse)
            
            # Track learning rate
            self.history['learning_rate'].append(self.current_learning_rate)

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

        # Boosting iterations
        for iteration in range(self.n_estimators):
            # Track flood level if using flooding loss
            if loss_type == 'flooding' and self.track_history:
                self.history["flood_level"].append(self.flood_level)

            # Compute negative gradients for the full dataset
            negative_gradients = -gradient_fn(current_pred, y)
            
            # Find best feature and fit a base model
            feature_idx, model = self._componentwise_fit(X, negative_gradients, iteration)
            
            # Update feature importance count
            feature_counts[feature_idx] += 1
                
            # Store the estimator
            self.estimators_.append((feature_idx, model))
            
            # Track selected feature
            if self.track_history:
                self.history['selected_features'].append(feature_idx)

            # Update the learning rate based on current loss
            train_loss = self.loss_fn(current_pred, y).item()
            self.current_learning_rate = self._update_learning_rate(iteration, train_loss)

            # Activate stochastic selection when LR ascent activates
            if (self.loss == 'flooding' and not self.stochastic_selection_activated and 
                train_loss < self.flood_level + 0.1):
                self.stochastic_selection_activated = True
                print(f"Stochastic top-k selection activated at iteration {iteration+1}!")            

            if self.track_history:
                # Ensure we have a learning rate entry for each iteration
                if len(self.history['learning_rate']) <= iteration:
                    self.history['learning_rate'].append(self.current_learning_rate)

            # Update predictions for all data points using the selected model
            if self.base_learner == "linear":
                X_feature = X[:, feature_idx:feature_idx+1]
            else:
                X_feature = X[:, feature_idx:feature_idx+1]
                
            feature_contrib = self._predict_base_learner(model, X_feature).squeeze() * self.current_learning_rate
            current_pred += feature_contrib

            # Update validation predictions if available
            if X_val is not None and y_val is not None:
                if self.base_learner == "linear":
                    val_X_feature = X_val[:, feature_idx:feature_idx+1]
                else:
                    val_X_feature = X_val[:, feature_idx:feature_idx+1]
                    
                val_feature_contrib = self._predict_base_learner(model, val_X_feature).squeeze() * self.current_learning_rate
                val_pred += val_feature_contrib

            # Update test predictions if available
            if X_test is not None and y_test is not None:
                if self.base_learner == "linear":
                    test_X_feature = X_test[:, feature_idx:feature_idx+1]
                else:
                    test_X_feature = X_test[:, feature_idx:feature_idx+1]
                    
                test_feature_contrib = self._predict_base_learner(model, test_X_feature).squeeze() * self.current_learning_rate
                test_pred += test_feature_contrib
            
            # Evaluate and track losses
            if self.track_history and (iteration + 1) % eval_freq == 0:
                # Training loss using specified loss function (e.g., flooding)
                train_loss = self.loss_fn(current_pred, y).item()
                self.history['train_loss'].append(train_loss)
                
                # Training MSE for evaluation purposes
                train_mse = self.eval_loss_fn(current_pred, y).item()
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
                        print(f", Flood: {self.flood_level:.6f}")
                    else:
                        print("")

            # Check if we should save the model at this iteration
            if save_iterations and (iteration + 1) in save_iterations:
                checkpoint_path = os.path.join(save_path, f"{loss_type}_{self.base_learner}_model_iteration_{iteration+1}.pt")
                self._save_model(checkpoint_path)
                if verbose:
                    print(f"\nSaved model checkpoint at iteration {iteration+1} to {checkpoint_path}\n")

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
            'base_learner': self.base_learner,
            'poly_degree': self.poly_degree,
            'tree_max_depth': self.tree_max_depth,
            'lr_ascent_mode': self.lr_ascent_mode,
            'lr_ascent_factor': self.lr_ascent_factor,
            'lr_ascent_step_size': self.lr_ascent_step_size,
            'lr_max': self.lr_max,
            'lr_ascent_activated': self.lr_ascent_activated,
            'lr_ascent_start_iter': self.lr_ascent_start_iter,
            'estimators_': self.estimators_,
            'feature_importances_': self.feature_importances_,
            'intercept_': self.intercept_,
            'flood_level': self.flood_level,
            'history': self.history
        }
        
        # Save to file using torch.save
        torch.save(model_state, path)

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
            if self.base_learner == "linear":
                X_feature = X[:, feature_idx:feature_idx+1]
            else:
                X_feature = X[:, feature_idx:feature_idx+1]

            # Use stored learning rate history if available, otherwise use final learning rate
            if self.track_history and i < len(self.history['learning_rate']):
                lr = self.history['learning_rate'][i]
            else:
                lr = self.current_learning_rate
                
            predictions += self._predict_base_learner(model, X_feature).squeeze() * lr

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


class PolynomialRegressionWrapper:
    """
    Wrapper for polynomial regression to maintain consistent interface.
    """
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.linear_model = LinearRegression()
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit polynomial regression model."""
        X_poly = self.poly_features.fit_transform(X)
        self.linear_model.fit(X_poly, y)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with polynomial regression model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_poly = self.poly_features.transform(X)
        return self.linear_model.predict(X_poly)