import torch
import numpy as np
from typing import Optional, Union, Tuple, List, Dict
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import BSpline

class ComponentwiseBoostingModel:
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        base_learner: str = "polynomial",
        poly_degree: int = 2,
        tree_max_depth: int = 2,
        spline_degree: int = 2,
        n_knots: int = 10,
        loss: str = 'mse', # 'mse' or 'flooding'
        flood_level: float = 0.0,
        use_momentum: bool = False,
        use_top_k: bool = False,
        top_k: int = 5,
        momentum_decay: float = 0.9,
        momentum_strength: float = 1.0,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        eps_momentum: float = 1e-6,
        eps_linear: float = 1e-8
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth
        
        # Store spline params
        self.spline_degree = spline_degree
        self.n_knots = n_knots
        
        self.loss = loss
        self.flood_level = flood_level
        
        self.use_momentum = use_momentum
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.momentum_decay = momentum_decay
        self.momentum_strength = momentum_strength
        self.batch_size = int(batch_size) if batch_size is not None else None
        
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.eps_momentum = eps_momentum
        self.eps_linear = eps_linear   
        
        self.estimators_ = []
        self.intercept_ = 0.0
        self.feature_momentum = {} 
        
        # Store global knots for bsplines: feature_idx -> knots vector
        self.feature_knots_ = {}
        
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
            # vectorized momentum update
            mom_vec = torch.tensor([self.feature_momentum[i] for i in range(n_features)], device=losses_tensor.device)
            mom_vec *= self.momentum_decay
            scores = 1.0 / (losses_tensor + self.eps_momentum)
            mom_vec += self.momentum_strength * scores
            
            # Write back to dictionary
            for i in range(n_features):
                self.feature_momentum[i] = mom_vec[i].item()
                
            adjusted_losses = losses_tensor - mom_vec
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

    # Vectorized Solvers

    def _solve_linear_vectorized(self, X_batch, target):
        """
        Solves y = beta * x for all features simultaneously
        """

        # X_batch: (B, F)
        # target: (B, ) -> (B, 1)
        target = target.unsqueeze(1)
        
        # Numerator: X^T * y -> (F, B) @ (B, 1) -> (F, 1) -> (F,)
        numer = (X_batch * target).sum(dim=0)
        
        # Denom: X^T * X -> Diagonal only since we solve per feature independently
        denom = (X_batch ** 2).sum(dim=0)
        
        beta = numer / (denom + self.eps_linear) # shape (F,)
        
        # Compute MSE for each feature
        # Preds: X * beta -> (B, F) * (F,) no broadcast
        # we need prediction for feature i using beta i
        preds = X_batch * beta.unsqueeze(0) # (B, F)
        
        losses = ((preds - target)**2).mean(dim=0) # (F,)
        
        return beta, losses

    def _solve_poly_vectorized(self, X_batch, target):
        """
        Solves Polynomial Regression for all features simultaneously.
        Matches original logic: Includes Intercept (via LinearRegression default).
        Form: y = b0 + b1*x + b2*x^2 ...
        """
        n_samples, n_features = X_batch.shape
        device = X_batch.device
        
        # 1. Construct Design Matrices for all features
        # We want [1, x, x^2] for every feature.
        # Output shape: (n_features, n_samples, degree+1)
        
        # Powers: 1 to degree
        exponents = torch.arange(1, self.poly_degree + 1, device=device).float()
        
        # X_batch unsqueezed: (N, F, 1)
        X_expanded = X_batch.unsqueeze(-1)
        
        # Feature powers: (N, F, degree)
        # Note: Depending on memory, this can be large.
        poly_features = X_expanded.pow(exponents)
        
        # Add Bias term (column of 1s): (N, F, 1)
        bias = torch.ones(n_samples, n_features, 1, device=device)
        
        # Design Matrix A: (N, F, d+1)
        A = torch.cat([bias, poly_features], dim=2)
        
        # Permute for batch solving: (F, N, d+1)
        A = A.permute(1, 0, 2)
        
        # Target Y: (F, N, 1) (Same target for all features)
        Y = target.view(1, n_samples, 1).expand(n_features, n_samples, 1)
        
        # 2. Solve Normal Equations: (A^T A) beta = A^T Y
        # A_T: (F, d+1, N)
        A_T = A.transpose(1, 2)
        
        # ATA: (F, d+1, d+1)
        ATA = torch.bmm(A_T, A)
        
        # ATY: (F, d+1, 1)
        ATY = torch.bmm(A_T, Y)
        
        # Regularization for stability
        I = torch.eye(self.poly_degree + 1, device=device).unsqueeze(0).expand(n_features, -1, -1)
        ATA_reg = ATA + self.eps_linear * I
        
        # Solve
        # beta shape: (F, d+1, 1)
        beta = torch.linalg.solve(ATA_reg, ATY)
        
        # 3. Compute Losses
        # Preds = A @ beta -> (F, N, d+1) @ (F, d+1, 1) -> (F, N, 1)
        preds = torch.bmm(A, beta).squeeze(-1) # (F, N)
        
        # Target is (N,)
        target_rep = target.unsqueeze(0) # (1, N)
        
        losses = ((preds - target_rep)**2).mean(dim=1) # (F,)
        
        return beta.squeeze(-1), losses

    def _solve_bspline_vectorized(self, X_batch, target):
        """
        Solves B-Spline Regression for all features simultaneously.
        Reuse the design matrix -> Normal Equations logic from Polynomial.
        """
        n_samples, n_features = X_batch.shape
        device = X_batch.device
        
        # Number of basis functions = n_knots + degree + 1 (usually, depending on def)
        
        # loop over features to build the big tensor
        
        X_np = X_batch.detach().cpu().numpy()
        basis_matrices = []
        
        for f_idx in range(n_features):
            knots = self.feature_knots_[f_idx]
            
            # Clip input to knot range to prevent OutOfBounds errors
            x_col = np.clip(X_np[:, f_idx], knots[0], knots[-1])
            
            # Design matrix (N, n_basis)
            # BSpline.design_matrix returns a CSR matrix or dense depending on version/input
            dm = BSpline.design_matrix(x_col, knots, self.spline_degree)
            if not isinstance(dm, np.ndarray):
                dm = dm.toarray()
            basis_matrices.append(dm)
            
        # Stack to (F, N, n_basis)
        A_np = np.stack(basis_matrices, axis=0) 
        A = torch.from_numpy(A_np).float().to(device)
        
        n_basis = A.shape[2]
        
        # Target Y: (F, N, 1)
        Y = target.view(1, n_samples, 1).expand(n_features, n_samples, 1)
        
        # Solve Normal Equations: (A^T A) beta = A^T Y
        A_T = A.transpose(1, 2)
        ATA = torch.bmm(A_T, A)
        ATY = torch.bmm(A_T, Y)
        
        # Regularization
        I = torch.eye(n_basis, device=device).unsqueeze(0).expand(n_features, -1, -1)
        ATA_reg = ATA + self.eps_linear * I
        
        # Solve
        beta = torch.linalg.solve(ATA_reg, ATY)
        
        # Compute Losses
        preds = torch.bmm(A, beta).squeeze(-1) # (F, N)
        target_rep = target.unsqueeze(0)
        losses = ((preds - target_rep)**2).mean(dim=1)
        
        return beta.squeeze(-1), losses

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        # Convert all inputs to float tensors if they aren't already
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, dtype=torch.float32)
        if X_test is not None:
            X_test = torch.as_tensor(X_test, dtype=torch.float32)
            y_test = torch.as_tensor(y_test, dtype=torch.float32)

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
        
        # --- Pre-compute Knots for B-Splines if needed ---
        if self.base_learner == "bspline":
            X_train_np = X_train.detach().cpu().numpy()
            for f_idx in range(n_features):
                # --- Quantile Knots for Stability ---
                f_min = X_train_np[:, f_idx].min()
                f_max = X_train_np[:, f_idx].max()
                
                # Create percentiles (0 to 100)
                # n_knots internal points
                # linspace(0, 100, n_knots + 2) gives [0, ..., 100]
                percentiles = np.linspace(0, 100, self.n_knots + 2)
                knots_all = np.percentile(X_train_np[:, f_idx], percentiles)
                
                # Remove duplicates (if data is very sparse/discrete) to avoid 0-width intervals
                knots_unique = np.unique(knots_all)
                
                # We need at least 2 points to define a range. 
                if len(knots_unique) < 2:
                    internal_knots = np.array([(f_min + f_max)/2])
                else:
                    internal_knots = knots_unique[1:-1]
                
                # Full knot vector for scipy BSpline:
                # k+1 repeats at ends + internal knots
                t = np.concatenate(([f_min]*(self.spline_degree), 
                                    [f_min], 
                                    internal_knots, 
                                    [f_max], 
                                    [f_max]*(self.spline_degree)))
                
                self.feature_knots_[f_idx] = t
        
        # Best Model Tracking
        best_val_loss = float('inf')
        self.best_iteration_ = 0

        for i in range(self.n_estimators):        
            
            # Mini-batch sampling
            if self.batch_size is not None and self.batch_size < n_samples:
                batch_idx = torch.randperm(n_samples)[:self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                curr_pred_batch = curr_pred_train[batch_idx]
            else:
                X_batch = X_train
                y_batch = y_train
                curr_pred_batch = curr_pred_train

            grad = self._get_gradient(curr_pred_batch, y_batch)
            target = -grad

            # --- OPTIMIZED FEATURE SELECTION ---
            
            best_idx = -1
            best_params = None
            best_model_obj = None # Only for trees
            
            if self.base_learner == "linear":
                betas, losses = self._solve_linear_vectorized(X_batch, target)
                best_idx = self._select_feature(losses)
                best_params = betas[best_idx] # Tensor
                
            elif self.base_learner == "polynomial":
                betas, losses = self._solve_poly_vectorized(X_batch, target)
                best_idx = self._select_feature(losses)
                best_params = betas[best_idx] # Tensor (coefficients)
                
            elif self.base_learner == "bspline":
                # Returns betas (F, n_basis) and losses (F,)
                betas, losses = self._solve_bspline_vectorized(X_batch, target)
                best_idx = self._select_feature(losses)
                # Store coeffs AND the knots used for this feature
                best_params = {
                    'coeffs': betas[best_idx], 
                    'knots': self.feature_knots_[best_idx]
                }
                
            elif self.base_learner == "tree":
                # Trees cannot be easily vectorized on CPU without C++, keep loop
                feature_losses = []
                feature_models = []
                
                # Pre-convert to numpy once per batch to avoid overhead in loop?
                # Sklearn needs numpy.
                X_batch_np = X_batch.detach().numpy()
                target_np = target.detach().numpy()
                
                for f_idx in range(n_features):
                    x_f = X_batch_np[:, f_idx:f_idx+1]
                    model = DecisionTreeRegressor(max_depth=self.tree_max_depth)
                    model.fit(x_f, target_np)
                    
                    pred = model.predict(x_f)
                    loss = np.mean((pred - target_np)**2)
                    
                    feature_losses.append(loss)
                    feature_models.append(model)
                
                best_idx = self._select_feature(torch.tensor(feature_losses))
                best_model_obj = feature_models[best_idx]
            
            # --- Store Best Learner ---
            self.estimators_.append({
                'idx': best_idx,
                'learner': self.base_learner,
                'params': best_params,   # Tensor for Lin/Poly, Dict for Bspline
                'model': best_model_obj  # Object for Tree
            })
            self.history['selected_features'].append(best_idx)
            
            # --- Global Update ---
            # We must predict on the FULL sets now
            
            def compute_update(X_data, learner_idx, learner_info):
                x_f = X_data[:, learner_idx:learner_idx+1] # (N, 1)
                
                if self.base_learner == "linear":
                    # y = x * beta
                    return (x_f * learner_info['params']).flatten()
                
                elif self.base_learner == "polynomial":
                    # y = b0 + b1*x + b2*x^2 ...
                    # params is [b0, b1, b2...]
                    params = learner_info['params']
                    N = x_f.shape[0]
                    
                    # Design Matrix: [1, x, x^2...]
                    # Or simple Horner's method / accumulation
                    pred = torch.full((N,), params[0].item(), device=X_data.device)
                    pow_x = x_f.flatten()
                    
                    for p in range(1, len(params)):
                        pred += params[p] * pow_x 
                        pow_x = pow_x * x_f.flatten()
                        
                    return pred
                
                elif self.base_learner == "bspline":
                    # Params is dict {'coeffs': ..., 'knots': ...}
                    coeffs = learner_info['params']['coeffs'].detach().cpu().numpy()
                    knots = learner_info['params']['knots']
                    x_np = x_f.flatten().detach().cpu().numpy()
                    
                    # Clip to knot range to handle unseen data (Val/Test)
                    x_np = np.clip(x_np, knots[0], knots[-1])
                    
                    # Reconstruct B-Spline Design Matrix for this feature
                    dm = BSpline.design_matrix(x_np, knots, self.spline_degree)
                    # Handle sparse/dense
                    if not isinstance(dm, np.ndarray):
                        dm = dm.toarray()
                    
                    # Pred = DM @ coeffs
                    pred_np = dm @ coeffs
                    return torch.from_numpy(pred_np).float().to(X_data.device)
                    
                elif self.base_learner == "tree":
                    # Use Sklearn
                    x_np = x_f.detach().numpy()
                    pred_np = learner_info['model'].predict(x_np)
                    return torch.from_numpy(pred_np).float()

            learner_data = self.estimators_[-1]
            
            # Train Update
            update_train = compute_update(X_train, best_idx, learner_data) * self.learning_rate
            curr_pred_train += update_train
            
            # Val Update
            if X_val is not None:
                update_val = compute_update(X_val, best_idx, learner_data) * self.learning_rate
                curr_pred_val += update_val
                
                val_mse = torch.mean((curr_pred_val - y_val)**2).item()
                self.history['val_loss'].append(val_mse)
                
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    self.best_iteration_ = i + 1
            
            # Test Update
            if X_test is not None:
                update_test = compute_update(X_test, best_idx, learner_data) * self.learning_rate
                curr_pred_test += update_test
                test_mse = torch.mean((curr_pred_test - y_test)**2).item()
                self.history['test_loss'].append(test_mse)

            # Train Loss
            train_mse = torch.mean((curr_pred_train - y_train)**2).item()
            self.history['train_loss'].append(train_mse)

            if (i+1) % 50 == 0:
                print(f"Iter {i+1}/{self.n_estimators} | Train MSE: {train_mse:.5f}")

    def predict(self, X, use_best_model=False):
        X = torch.as_tensor(X, dtype=torch.float32)
        pred = torch.full((X.shape[0],), self.intercept_)
        
        limit = self.best_iteration_ if use_best_model and self.best_iteration_ > 0 else len(self.estimators_)
        estimators_to_use = self.estimators_[:limit]
            
        for est in estimators_to_use:
            f_idx = est['idx']
            x_f = X[:, f_idx:f_idx+1]
            
            update = None
            if est['learner'] == 'linear':
                update = (x_f * est['params']).flatten()
                
            elif est['learner'] == 'polynomial':
                params = est['params']
                N = x_f.shape[0]
                update = torch.full((N,), params[0].item(), device=X.device)
                pow_x = x_f.flatten()
                for p in range(1, len(params)):
                    update += params[p] * pow_x
                    pow_x = pow_x * x_f.flatten()
            
            elif est['learner'] == 'bspline':
                coeffs = est['params']['coeffs'].detach().cpu().numpy()
                knots = est['params']['knots']
                x_np = x_f.flatten().detach().cpu().numpy()
                
                # Clip to knot range
                x_np = np.clip(x_np, knots[0], knots[-1])

                dm = BSpline.design_matrix(x_np, knots, self.spline_degree)
                if not isinstance(dm, np.ndarray):
                    dm = dm.toarray()
                
                pred_np = dm @ coeffs
                update = torch.from_numpy(pred_np).float().to(X.device)

            elif est['learner'] == 'tree':
                x_np = x_f.detach().numpy()
                pred_np = est['model'].predict(x_np)
                update = torch.from_numpy(pred_np).float()
            
            pred += update * self.learning_rate
            
        return pred

    @staticmethod
    def load_model(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_model(self, path):
        import pickle
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)