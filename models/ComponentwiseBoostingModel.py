import torch
import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ComponentwiseBoostingModel:
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        base_learner: str = "polynomial",
        poly_degree: int = 2,
        tree_max_depth: int = 2,
        loss: str = 'mse', # 'mse' or 'flooding'
        flood_level: float = 0.0,
        use_momentum: bool = False,
        use_top_k: bool = False,
        top_k: int = 5,
        momentum_decay: float = 0.9,
        momentum_strength: float = 1.0,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth
        self.loss = loss
        self.flood_level = flood_level
        
        self.use_momentum = use_momentum
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.momentum_decay = momentum_decay
        self.momentum_strength = momentum_strength
        self.batch_size = batch_size
        
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            
        self.estimators_ = []
        self.intercept_ = 0.0
        self.feature_momentum = {} 
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [], 
            'selected_features': []
        }

    def _get_gradient(self, y_pred, y):
        """Returns gradient of Loss w.r.t prediction (dL/dPred)"""
        grad = (y_pred - y)
        
        if self.loss == 'flooding':
            mse = torch.mean((y_pred - y)**2)
            if mse < self.flood_level:
                # Gradient Ascent (Push away from 0)
                grad *= -1.0 
        
        return grad

    def _select_feature(self, losses_tensor: torch.Tensor) -> int:
        n_features = len(losses_tensor)
        
        # 1. Apply Momentum Adjustment
        if self.use_momentum:
            if len(self.feature_momentum) == 0:
                for i in range(n_features): self.feature_momentum[i] = 0.0
            
            # Update Momentum: Inversely proportional to loss
            for i in range(n_features):
                self.feature_momentum[i] *= self.momentum_decay
                score = 1.0 / (losses_tensor[i].item() + 1e-6)
                self.feature_momentum[i] += self.momentum_strength * score
                
            # Higher momentum -> Lower Adjusted Loss (Better chance to be picked)
            momentum_vec = torch.tensor([self.feature_momentum[i] for i in range(n_features)])
            adjusted_losses = losses_tensor - momentum_vec
        else:
            adjusted_losses = losses_tensor

        # 2. Top-K Filtering
        if self.use_top_k:
            k = min(self.top_k, n_features)
            # Get indices of the k smallest adjusted losses
            top_k_indices = torch.topk(adjusted_losses, k, largest=False).indices
            # Randomly select one from bucket
            selected_idx = top_k_indices[torch.randint(0, k, (1,))].item()
        else:
            # Greedy
            selected_idx = torch.argmin(adjusted_losses).item()
            
        return selected_idx

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        self.intercept_ = torch.mean(y_train).item()
        curr_pred_train = torch.full_like(y_train, self.intercept_)
        
        curr_pred_val = None
        if X_val is not None:
            curr_pred_val = torch.full_like(y_val, self.intercept_)
        
        curr_pred_test = None
        if X_test is not None:
            curr_pred_test = torch.full_like(y_test, self.intercept_)

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        # Best Model Tracking
        best_val_loss = float('inf')
        self.best_iteration_ = 0

        for i in range(self.n_estimators):        
            # Evaluate all features
            for f_idx in range(n_features):
                # Mini-batch sampling
                if self.batch_size is not None and self.batch_size < n_samples:
                    # Randomly sample indices for this iteration
                    batch_idx = np.random.choice(n_samples, self.batch_size, replace=False)   
                    # creating batch views
                    X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    curr_pred_batch = curr_pred_train[batch_idx]

                else:
                    #Fallback to full batch
                    X_batch = X_train
                    y_batch = y_train
                    curr_pred_batch = curr_pred_train

                grad = self._get_gradient(curr_pred_batch, y_batch)
                target = -grad

                feature_losses = []
                feature_models = []

                # Evaluate all features using BATCH data
                for f_idx in range(n_features):
                    x_f = X_batch[:, f_idx:f_idx+1] # Slice batch feature
                    model = self._create_base_learner()
                    model = self._fit_base_learner(model, x_f, target)
                    
                    # Calculate loss against batch gradient
                    pred = self._predict_base_learner(model, x_f).squeeze()
                    loss = torch.mean((pred - target.squeeze())**2)
                    
                    feature_losses.append(loss)
                    feature_models.append(model)

            # Select Best Feature
            best_idx = self._select_feature(torch.tensor(feature_losses))
            best_model = feature_models[best_idx]
            
            # Update State
            self.estimators_.append((best_idx, best_model))
            self.history['selected_features'].append(best_idx)
            
            # Global Update on FULL SET
            # predict on the FULL X_train to keep residuals correct for next iteration
            x_f_train_full = X_train[:, best_idx:best_idx+1]
            update = self._predict_base_learner(best_model, x_f_train_full).squeeze() * self.learning_rate
            curr_pred_train += update
            
            # Val
            if X_val is not None:
                x_f_val = X_val[:, best_idx:best_idx+1]
                update_val = self._predict_base_learner(best_model, x_f_val).squeeze() * self.learning_rate
                curr_pred_val += update_val
                
                val_mse = torch.mean((curr_pred_val - y_val)**2).item()
                self.history['val_loss'].append(val_mse)
                
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    self.best_iteration_ = i + 1
            
            # Test
            if X_test is not None:
                x_f_test = X_test[:, best_idx:best_idx+1]
                update_test = self._predict_base_learner(best_model, x_f_test).squeeze() * self.learning_rate
                curr_pred_test += update_test
                test_mse = torch.mean((curr_pred_test - y_test)**2).item()
                self.history['test_loss'].append(test_mse)

            # Train Loss
            # Log Full Training Loss (to observe Double Descent properly)
            # For history we log MSE to be comparable, even if we optimize with Flooding
            train_mse = torch.mean((curr_pred_train - y_train)**2).item()
            self.history['train_loss'].append(train_mse)

            # Print iteration, train MSE & test MSE
            print(f"Iter {i+1}/{self.n_estimators} | Train MSE: {train_mse:.5f} | Test MSE: {self.history['test_loss'][-1] if len(self.history['test_loss']) > i else 'N/A'}")

    def _create_base_learner(self):
        if self.base_learner == "linear":
            return torch.nn.Linear(1, 1, bias=False)
        elif self.base_learner == "polynomial":
            return PolynomialRegressionWrapper(degree=self.poly_degree)
        elif self.base_learner == "tree":
            return DecisionTreeRegressor(max_depth=self.tree_max_depth)

    def _fit_base_learner(self, model, X, y):
        if self.base_learner == "linear":
            xtx = torch.matmul(X.t(), X)
            xty = torch.matmul(X.t(), y.unsqueeze(1) if y.dim()==1 else y)
            beta = xty / (xtx + 1e-8)
            model.weight.data = beta.t()
            return model
        else:
            X_np = X.detach().numpy()
            y_np = y.detach().numpy()
            model.fit(X_np, y_np)
            return model

    def _predict_base_learner(self, model, X):
        if self.base_learner == "linear":
            return model(X)
        else:
            X_np = X.detach().numpy()
            pred = model.predict(X_np)
            return torch.tensor(pred, dtype=torch.float32).unsqueeze(1)

    def predict(self, X, use_best_model=False):
        pred = torch.full((X.shape[0],), self.intercept_)
        
        limit = self.best_iteration_ if use_best_model and self.best_iteration_ > 0 else len(self.estimators_)
        estimators_to_use = self.estimators_[:limit]
            
        for f_idx, model in estimators_to_use:
            x_f = X[:, f_idx:f_idx+1]
            pred += self._predict_base_learner(model, x_f).squeeze() * self.learning_rate
            
        return pred

class PolynomialRegressionWrapper:
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
    def fit(self, X, y):
        x_poly = self.poly.fit_transform(X)
        self.model.fit(x_poly, y)
    def predict(self, X):
        x_poly = self.poly.transform(X)
        return self.model.predict(x_poly)