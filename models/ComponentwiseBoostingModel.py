import torch
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Union
import random
import os
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix
from scipy.linalg import pinv
import warnings
warnings.filterwarnings('ignore')


class ComponentwiseBoostingModel:
    """
    Componentwise Gradient Boosting with various regression base learners.
    Each boosting iteration selects the best feature-learner combination that most reduces the loss.

    - Uses specified loss during training
    - Uses MSE for evaluation/testing
    - Supports multiple base learner types per feature
    - Implements fair base learner selection with equal degrees of freedom
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
        loss: str = 'mse',  # 'mse' for mean square error, 'flooding' for flooding loss
        track_history: bool = True,
        base_learners: Union[str, List[str]] = ["linear"],  # Can be single string or list
        poly_degree: int = 2,  # For polynomial base learner
        tree_max_depth: int = 3,  # For tree base learner
        spline_knots: int = 5,  # For spline base learner
        degrees_of_freedom: int = 2,  # For fair base learner selection
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
        
        # Handle base learners - convert to list if string
        if isinstance(base_learners, str):
            self.base_learners = [base_learners]
        else:
            self.base_learners = base_learners
            
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth
        self.spline_knots = spline_knots
        self.degrees_of_freedom = degrees_of_freedom

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
            
        valid_learners = ["linear", "polynomial", "tree", "splines"]
        for learner in self.base_learners:
            if learner not in valid_learners:
                raise ValueError(f"base_learner {learner} must be one of: {valid_learners}")
             
        # Will be set during fitting
        self.estimators_: List[Tuple[int, str, object]] = []  # (feature_idx, learner_type, model)
        self.feature_importances_ = None
        self.intercept_ = None
        self.loss_fn = None
        self.eval_loss_fn = None  # Always MSE for evaluation
        self.flood_level = None

        # History tracking for double descent analysis
        self.history: Dict[str, List] = {
            'train_loss': [],          # Using training loss (e.g., flooding)
            'train_mse': [],           # Using MSE for evaluation
            'val_loss': [],            # Using MSE for evaluation
            'test_loss': [],           # Using MSE for evaluation
            'selected_features': [],   # Feature indices
            'selected_learners': [],   # Learner types
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
            'selected_features': [],   # Feature indices
            'selected_learners': [],   # Learner types
            'learning_rate': []        # Track learning rate changes            
        }

    def _get_loss_fn(self, loss_type: str, **kwargs) -> Callable:
        """Get the appropriate loss function based on the specified type."""
        if loss_type == 'mse':
            return lambda y_pred, y: torch.mean((y_pred - y) ** 2)
        elif loss_type == 'flooding':
            self.flood_level = kwargs.get('flood_level', 0.01)
            return lambda y_pred, y: (abs(torch.mean((y_pred - y) ** 2) - self.flood_level) + self.flood_level)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
    def _get_eval_loss_fn(self) -> Callable:
        """Get the evaluation loss function (always MSE)."""
        return lambda y_pred, y: torch.mean((y_pred - y) ** 2)
    
    def _get_gradient_fn(self, loss_type: str) -> Callable:
        """Get the gradient function for the specified loss type."""
        if loss_type == 'mse':
            return lambda y_pred, y: (y_pred - y).unsqueeze(1) if (y_pred - y).dim() == 1 else (y_pred - y)
        elif loss_type == 'flooding':
            def flooding_gradient(y_pred, y):
                mse = torch.mean((y_pred - y) ** 2)
                grad = y_pred - y
                sign = -1.0 if mse < self.flood_level else 1.0
                grad = sign * grad
                
                if mse < self.flood_level:
                    print(f"Flipping gradient! MSE: {mse:.6f}, Flood level: {self.flood_level:.6f}")           
                
                return grad.unsqueeze(1) if grad.dim() == 1 else grad
            return flooding_gradient
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _update_learning_rate(self, iteration: int, current_loss: float) -> float:
        """Update Learning Rate based on provided parameters."""
        if self.loss != 'flooding' or self.flood_level is None:
            return self.initial_learning_rate
            
        if not self.lr_ascent_activated and current_loss < self.flood_level + 0.1:
            self.lr_ascent_activated = True
            self.lr_ascent_start_iter = iteration
            print(f"Learning rate ascent activated at iteration {iteration+1}! Loss: {current_loss:.6f}, Flood level: {self.flood_level:.6f}")
        
        if not self.lr_ascent_activated:
            return self.initial_learning_rate
            
        iter_since_start = iteration - self.lr_ascent_start_iter
        
        if self.lr_ascent_mode == "linear":
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * iter_since_start)
        elif self.lr_ascent_mode == "exponential":
            new_lr = self.initial_learning_rate * ((1 + self.lr_ascent_factor) ** iter_since_start)
        elif self.lr_ascent_mode == "step":
            steps = iter_since_start // self.lr_ascent_step_size
            new_lr = self.initial_learning_rate * (1 + self.lr_ascent_factor * steps)
        else:
            new_lr = self.initial_learning_rate
            
        new_lr = min(new_lr, self.lr_max)
        return new_lr

    def _create_base_learner(self, learner_type: str) -> object:
        """Create a base learner based on the specified type."""
        if learner_type == "linear":
            return LinearBaseLearner(degrees_of_freedom=self.degrees_of_freedom)
        elif learner_type == "polynomial":
            return PolynomialBaseLearner(degree=self.poly_degree, degrees_of_freedom=self.degrees_of_freedom)
        elif learner_type == "tree":
            return TreeBaseLearner(max_depth=self.tree_max_depth, random_state=self.random_state)
        elif learner_type == "splines":
            return SplineBaseLearner(n_knots=self.spline_knots, degrees_of_freedom=self.degrees_of_freedom)
        else:
            raise ValueError(f"Unknown base learner: {learner_type}")

    def _componentwise_fit(
        self, 
        X: torch.Tensor, 
        negative_gradients: torch.Tensor,
        iteration: int
    ) -> Tuple[int, str, object]:
        """
        Fit base learners for each feature-learner combination and select the best one.
        
        Returns:
            Tuple of (best_feature_idx, best_learner_type, best_model)
        """
        n_features = X.shape[1]
        losses = []
        models = []
        feature_learner_pairs = []

        # Ensure negative_gradients is 2D for compatibility
        if negative_gradients.dim() == 1:
            negative_gradients = negative_gradients.unsqueeze(1)

        # Try each feature-learner combination
        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx:feature_idx+1]
            
            for learner_type in self.base_learners:
                # Create and fit base learner
                model = self._create_base_learner(learner_type)
                model = model.fit(X_feature, negative_gradients.squeeze())
                
                # Compute predictions and loss
                preds = model.predict(X_feature)
                if isinstance(preds, np.ndarray):
                    preds = torch.tensor(preds, dtype=torch.float32)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                    
                loss = torch.mean((preds.squeeze() - negative_gradients.squeeze()) ** 2).item()
                
                losses.append(loss)
                models.append(model)
                feature_learner_pairs.append((feature_idx, learner_type))

        # Convert losses to tensor for easier manipulation
        losses_tensor = torch.tensor(losses)
        
        # Select feature-learner combination based on whether stochastic selection is activated
        if self.stochastic_selection_activated:
            # Random selection from top-k combinations
            iterations_since_activation = iteration - self.lr_ascent_start_iter
            if iterations_since_activation < 50:
                k = self.top_k_early
            else:
                k = self.top_k_late

            top_k_indices = torch.topk(losses_tensor, k, largest=False).indices
            selected_idx = top_k_indices[torch.randint(0, k, (1,))].item()
            selected_feature, selected_learner = feature_learner_pairs[selected_idx]
            print(f"Stochastic top-{k} selection: chose feature {selected_feature} with {selected_learner} learner")
        else:
            # Deterministic selection (original behavior)
            selected_idx = torch.argmin(losses_tensor).item()
            selected_feature, selected_learner = feature_learner_pairs[selected_idx]
        
        return selected_feature, selected_learner, models[selected_idx]

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
        """Fit the boosting model using componentwise gradient boosting."""

        # Create save directory if it doesn't exist
        if save_iterations and len(save_iterations) > 0:
            os.makedirs(save_path, exist_ok=True)

        # Set or update loss function
        loss_type = loss if loss is not None else self.loss
        self.loss_fn = self._get_loss_fn(loss_type, **loss_params)
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
        
        # Feature-learner importance tracking
        n_features = X.shape[1]
        n_learners = len(self.base_learners)
        feature_learner_counts = np.zeros((n_features, n_learners))
        learner_names = {i: name for i, name in enumerate(self.base_learners)}

        # Initial learning rate
        self.current_learning_rate = self.initial_learning_rate

        # Initial loss calculation before training
        if self.track_history:
            train_loss = self.loss_fn(current_pred, y).item()
            self.history['train_loss'].append(train_loss)
            
            train_mse = self.eval_loss_fn(current_pred, y).item()
            self.history['train_mse'].append(train_mse)
            
            self.history['learning_rate'].append(self.current_learning_rate)

            if X_val is not None and y_val is not None:
                val_loss = self.eval_loss_fn(val_pred, y_val).item()
                self.history['val_loss'].append(val_loss)
            
            if X_test is not None and y_test is not None:
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
            
            # Find best feature-learner combination and fit a base model
            feature_idx, learner_type, model = self._componentwise_fit(X, negative_gradients, iteration)
            
            # Update feature-learner importance count
            learner_idx = self.base_learners.index(learner_type)
            feature_learner_counts[feature_idx, learner_idx] += 1
                
            # Store the estimator
            self.estimators_.append((feature_idx, learner_type, model))
            
            # Track selected feature and learner
            if self.track_history:
                self.history['selected_features'].append(feature_idx)
                self.history['selected_learners'].append(learner_type)

            # Update the learning rate based on current loss
            train_loss = self.loss_fn(current_pred, y).item()
            self.current_learning_rate = self._update_learning_rate(iteration, train_loss)

            # Activate stochastic selection when LR ascent activates
            if (self.loss == 'flooding' and not self.stochastic_selection_activated and 
                train_loss < self.flood_level + 0.1):
                self.stochastic_selection_activated = True
                print(f"Stochastic top-k selection activated at iteration {iteration+1}!")            

            if self.track_history:
                if len(self.history['learning_rate']) <= iteration:
                    self.history['learning_rate'].append(self.current_learning_rate)

            # Update predictions for all data points using the selected model
            X_feature = X[:, feature_idx:feature_idx+1]
            feature_contrib = model.predict(X_feature)
            if isinstance(feature_contrib, np.ndarray):
                feature_contrib = torch.tensor(feature_contrib, dtype=torch.float32)
            if feature_contrib.dim() == 1:
                feature_contrib = feature_contrib.unsqueeze(1)
            
            current_pred += feature_contrib.squeeze() * self.current_learning_rate

            # Update validation predictions if available
            if X_val is not None and y_val is not None:
                val_X_feature = X_val[:, feature_idx:feature_idx+1]
                val_feature_contrib = model.predict(val_X_feature)
                if isinstance(val_feature_contrib, np.ndarray):
                    val_feature_contrib = torch.tensor(val_feature_contrib, dtype=torch.float32)
                if val_feature_contrib.dim() == 1:
                    val_feature_contrib = val_feature_contrib.unsqueeze(1)
                val_pred += val_feature_contrib.squeeze() * self.current_learning_rate

            # Update test predictions if available
            if X_test is not None and y_test is not None:
                test_X_feature = X_test[:, feature_idx:feature_idx+1]
                test_feature_contrib = model.predict(test_X_feature)
                if isinstance(test_feature_contrib, np.ndarray):
                    test_feature_contrib = torch.tensor(test_feature_contrib, dtype=torch.float32)
                if test_feature_contrib.dim() == 1:
                    test_feature_contrib = test_feature_contrib.unsqueeze(1)
                test_pred += test_feature_contrib.squeeze() * self.current_learning_rate
            
            # Evaluate and track losses
            if self.track_history and (iteration + 1) % eval_freq == 0:
                train_loss = self.loss_fn(current_pred, y).item()
                self.history['train_loss'].append(train_loss)
                
                train_mse = self.eval_loss_fn(current_pred, y).item()
                self.history['train_mse'].append(train_mse)
                             
                if X_val is not None and y_val is not None:
                    val_loss = self.eval_loss_fn(val_pred, y_val).item()
                    self.history['val_loss'].append(val_loss)
                    
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
                    print(f", Selected: F{feature_idx}({learner_type})")

            # Check if we should save the model at this iteration
            if save_iterations and (iteration + 1) in save_iterations:
                checkpoint_path = os.path.join(save_path, f"{loss_type}_model_iteration_{iteration+1}.pt")
                self._save_model(checkpoint_path)
                if verbose:
                    print(f"\nSaved model checkpoint at iteration {iteration+1} to {checkpoint_path}\n")

        # Calculate feature importances (sum across all learner types for each feature)
        feature_totals = np.sum(feature_learner_counts, axis=1)
        self.feature_importances_ = feature_totals / np.sum(feature_totals) if np.sum(feature_totals) > 0 else feature_totals
        
        # Store detailed feature-learner importance
        self.feature_learner_importances_ = feature_learner_counts / np.sum(feature_learner_counts) if np.sum(feature_learner_counts) > 0 else feature_learner_counts
        
        return self

    def _save_model(self, path: str) -> None:
        """Save the model to a file."""
        model_state = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'current_learning_rate': self.current_learning_rate,
            'random_state': self.random_state,
            'loss': self.loss,
            'track_history': self.track_history,
            'base_learners': self.base_learners,
            'poly_degree': self.poly_degree,
            'tree_max_depth': self.tree_max_depth,
            'spline_knots': self.spline_knots,
            'degrees_of_freedom': self.degrees_of_freedom,
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
        torch.save(model_state, path)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate predictions for the input data."""
        predictions = torch.full((X.shape[0],), self.intercept_, device=X.device)
        
        for i, (feature_idx, learner_type, model) in enumerate(self.estimators_):
            X_feature = X[:, feature_idx:feature_idx+1]

            # Use stored learning rate history if available
            if self.track_history and i < len(self.history['learning_rate']):
                lr = self.history['learning_rate'][i]
            else:
                lr = self.current_learning_rate
                
            contrib = model.predict(X_feature)
            if isinstance(contrib, np.ndarray):
                contrib = torch.tensor(contrib, dtype=torch.float32)
            if contrib.dim() == 1:
                contrib = contrib.unsqueeze(1)
                
            predictions += contrib.squeeze() * lr

        return predictions
    
    def get_loss(self, X: torch.Tensor, y: torch.Tensor, use_training_loss: bool = False) -> float:
        """Calculate the loss on the given data."""
        if self.loss_fn is None or self.eval_loss_fn is None:
            raise ValueError("Model must be fitted before calculating loss")
        
        predictions = self.predict(X)
        
        if use_training_loss:
            return self.loss_fn(predictions, y).item()
        else:
            return self.eval_loss_fn(predictions, y).item()


# Base learner classes implementing fair selection through degrees of freedom

class LinearBaseLearner:
    """Linear base learner with regularization for fair selection."""
    
    def __init__(self, degrees_of_freedom: int = 2):
        self.degrees_of_freedom = degrees_of_freedom
        self.coef_ = None
        self.fitted = False
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit linear model with regularization to achieve target degrees of freedom."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.detach().numpy()
        else:
            y_np = y
            
        # Add regularization to achieve target degrees of freedom
        # For linear model: lambda = (n - df) / df where n is number of parameters
        n_params = X_np.shape[1]
        if self.degrees_of_freedom >= n_params:
            lambda_reg = 0.0
        else:
            lambda_reg = (n_params - self.degrees_of_freedom) / self.degrees_of_freedom
        
        # Regularized least squares: (X'X + lambda*I)^-1 X'y
        XtX = X_np.T @ X_np
        if lambda_reg > 0:
            XtX += lambda_reg * np.eye(XtX.shape[0])
        
        try:
            self.coef_ = np.linalg.solve(XtX, X_np.T @ y_np)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            self.coef_ = pinv(XtX) @ (X_np.T @ y_np)
            
        self.fitted = True
        return self
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, torch.Tensor):
            X_np = X.detach().numpy()
        else:
            X_np = X
            
        return X_np @ self.coef_


class PolynomialBaseLearner:
    """Polynomial base learner with regularization for fair selection."""
    
    def __init__(self, degree: int = 2, degrees_of_freedom: int = 2):
        self.degree = degree
        self.degrees_of_freedom = degrees_of_freedom
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.coef_ = None
        self.fitted = False
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit polynomial model with regularization."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.detach().numpy()
        else:
            y_np = y
            
        X_poly = self.poly_features.fit_transform(X_np)
        
        # Add regularization to achieve target degrees of freedom
        n_params = X_poly.shape[1]
        if self.degrees_of_freedom >= n_params:
            lambda_reg = 0.0
        else:
            lambda_reg = (n_params - self.degrees_of_freedom) / self.degrees_of_freedom
        
        XtX = X_poly.T @ X_poly
        if lambda_reg > 0:
            XtX += lambda_reg * np.eye(XtX.shape[0])
        
        try:
            self.coef_ = np.linalg.solve(XtX, X_poly.T @ y_np)
        except np.linalg.LinAlgError:
            self.coef_ = pinv(XtX) @ (X_poly.T @ y_np)
            
        self.fitted = True
        return self
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, torch.Tensor):
            X_np = X.detach().numpy()
        else:
            X_np = X
            
        X_poly = self.poly_features.transform(X_np)
        return X_poly @ self.coef_