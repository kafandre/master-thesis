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
        n_bins: int = 32,
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

        # Binning for trees
        self.n_bins = n_bins
        
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

    def _solve_tree_vectorized(self, X_binned, target, bin_edges):
        """
        Solves Decision Stump (Depth=1) for all features simultaneously using
        Histogram-based optimization (Lookup Table).
        
        Args:
            X_binned: (N, F) LongTensor of bin indices (0 to n_bins-1)
            target: (N,) Tensor of gradients/residuals
            bin_edges: (F, n_bins+1) Tensor of bin boundaries
        """
        n_samples, n_features = X_binned.shape
        n_bins = self.n_bins
        device = X_binned.device

        # 1. Efficient Histogram Aggregation (Vectorized)
        # Offsets to shift indices for each feature: [0, n_bins, 2*n_bins, ...]
        offsets = (torch.arange(n_features, device=device) * n_bins).view(1, -1)
        flat_indices = (X_binned + offsets).view(-1)  # (N*F,)
        
        # Expand target for all features
        flat_target = target.view(-1, 1).expand(-1, n_features).reshape(-1)
        
        # Aggregate Sums (G) and Counts (N)
        # G[i] = Sum of targets in absolute bin i
        G_flat = torch.zeros(n_features * n_bins, device=device)
        N_flat = torch.zeros(n_features * n_bins, device=device)
        
        G_flat.index_add_(0, flat_indices, flat_target)
        N_flat.index_add_(0, flat_indices, torch.ones_like(flat_target))
        
        # Reshape back to (F, n_bins)
        G = G_flat.view(n_features, n_bins)
        N = N_flat.view(n_features, n_bins)
        
        # 2. Compute Cumulative Sums (Left vs Right Splits)
        # G_L[k] = Sum of targets for bins 0..k
        G_L = torch.cumsum(G, dim=1)
        N_L = torch.cumsum(N, dim=1)
        
        # Totals for each feature
        G_T = G_L[:, -1:] # (F, 1)
        N_T = N_L[:, -1:] # (F, 1)
        
        # Right splits
        G_R = G_T - G_L
        N_R = N_T - N_L
        
        # 3. Calculate Gain (MSE Reduction Proxy)
        # Gain = G_L^2 / N_L + G_R^2 / N_R
        # Add epsilon to avoid div by zero
        eps = 1e-6
        gain = (G_L**2 / (N_L + eps)) + (G_R**2 / (N_R + eps))
        
        # Mask invalid splits (where Left or Right has 0 samples)
        # We only consider splits at bin boundaries 0 to n_bins-2 (n_bins-1 is the last bucket)
        valid_mask = (N_L > 0) & (N_R > 0)
        # Also, we cannot split after the very last bin (no right child)
        valid_mask[:, -1] = False 
        
        gain[~valid_mask] = -1.0 # Ignore invalid
        
        # 4. Find Best Split per Feature
        # max_gain_per_feat, best_bin_idx = torch.max(gain, dim=1)
        
        # Calculate losses for feature selection logic
        # MSE = sum(y^2) - Gain. Since sum(y^2) is constant, minimizing MSE <=> maximizing Gain
        # We return "negative gain" as loss because the selector picks min(loss)
        neg_gain_per_feat = -torch.max(gain, dim=1).values
        
        return gain, neg_gain_per_feat

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
        
        # --- NEW: Pre-compute Bins for Trees ---
        X_train_binned = None
        self.bin_edges_ = {}
        
        if self.base_learner == "tree":
            # Quantile Binning
            X_train_np = X_train.detach().cpu().numpy()
            X_binned_list = []
            
            # Compute percentiles for bin edges
            percentiles = torch.linspace(0, 1, self.n_bins + 1, device=X_train.device)
            
            # We compute edges for all features using torch.quantile
            # Note: For very large data, do this on CPU or subsample
            self.all_bin_edges = torch.quantile(X_train, percentiles, dim=0).T # (F, n_bins+1)
            
            # Add epsilon to last edge to include max value
            self.all_bin_edges[:, -1] += 1e-4
            
            # Bucketize (Vectorized binning)
            # torch.bucketize only works with 1D boundaries, so we loop or use searchsorted
            # Faster to loop over F for binning step once
            for f_idx in range(n_features):
                # buckets are 0 to n_bins (we clip to n_bins-1)
                edges = self.all_bin_edges[f_idx]
                # Force monotonicity to avoid errors
                edges, _ = torch.sort(edges)
                binned = torch.bucketize(X_train[:, f_idx], edges)
                # Clamp to range [0, n_bins-1]
                binned = torch.clamp(binned - 1, 0, self.n_bins - 1)
                X_binned_list.append(binned)
            
            X_train_binned = torch.stack(X_binned_list, dim=1) # (N, F)

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
                if X_train_binned is not None:
                    X_batch_binned = X_train_binned                

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
                gains, losses = self._solve_tree_vectorized(X_batch_binned, target, self.all_bin_edges)
                
                best_idx = self._select_feature(losses)
                
                # Retrieve best split details for the selected feature
                feat_gains = gains[best_idx]
                best_bin_idx = torch.argmax(feat_gains).item()
                
                # Reconstruct Leaf Values
                # We need to re-calculate means for the chosen split to store them
                # (Or strictly, we could return them from the solver, but re-calc is cheap for 1 feature)
                
                # Get the actual data for this feature to compute exact leaf values (optional)
                # OR use the binned statistics. Let's use binned stats for speed.
                # We need the values S_L, N_L, etc., which we computed inside the solver.
                # To keep code clean, let's just re-compute the leaf means for the WINNER feature only.
                
                f_binned = X_batch_binned[:, best_idx]
                mask_left = f_binned <= best_bin_idx
                
                # Compute leaf values
                val_left = target[mask_left].mean()
                val_right = target[~mask_left].mean()
                
                # The physical threshold is the upper edge of the chosen bin
                threshold = self.all_bin_edges[best_idx, best_bin_idx + 1].item()
                
                best_params = {
                    'threshold': threshold,
                    'left_val': val_left.item(),
                    'right_val': val_right.item()
                }
                best_model_obj = None
            
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
                    params = learner_info['params']
                    # Vectorized conditional
                    # returns left_val where x <= thresh, else right_val
                    thresh = params['threshold']
                    l_val = params['left_val']
                    r_val = params['right_val']
                    
                    pred = torch.where(
                        x_f <= thresh,
                        torch.tensor(l_val, device=x_f.device),
                        torch.tensor(r_val, device=x_f.device)
                        )
                    return pred.flatten()

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
                params = est['params']
                thresh = params['threshold']
                l_val = params['left_val']
                r_val = params['right_val']
                
                update = torch.where(
                    x_f <= thresh,
                    torch.tensor(l_val, device=X.device),
                    torch.tensor(r_val, device=X.device))
                update = update.flatten()
            
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