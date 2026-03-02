import torch
import numpy as np
from typing import Optional, Union, Tuple, List, Dict
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

class ComponentwiseBoostingModel:
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        base_learner: Union[str, List[str]] = "linear",
        poly_degree: int = 2,
        tree_max_depth: int = 1,
        n_bins: int = 256,
        spline_degree: int = 2,
        n_knots: int = 10,
        loss: str = 'mse', # mse or flooding
        flood_level: float = 0.0,
        use_momentum: bool = False,
        use_top_k: bool = False,
        top_k: int = 5,
        momentum_decay: float = 0.9,
        momentum_strength: float = 1.0,
        random_state: Optional[int] = None,
        eps_momentum: float = 1e-6,
        eps_linear: float = 1e-8,
        target_df: float = 1.0 # target degrees of freedom for penalization
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # Modified initialization for competing learners
        if isinstance(base_learner, str):
            self.base_learners = [base_learner]
            self.legacy_mode = True
        elif isinstance(base_learner, list):
            self.base_learners = base_learner
            self.legacy_mode = (len(base_learner) == 1)
        else:
            raise ValueError("base_learner must be a string or a list of strings")
        
        # For legacy compatibility, expose the single learner string if in legacy mode
        self.base_learner = self.base_learners[0] if self.legacy_mode else "competing"
        
        # Setup base learner parameters
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth
        self.n_bins = n_bins
        self.spline_degree = spline_degree
        self.n_knots = n_knots
        
        self.loss = loss
        self.flood_level = flood_level
        
        # Setup method hyperparameters
        self.use_momentum = use_momentum
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.momentum_decay = momentum_decay
        self.momentum_strength = momentum_strength
        
        # Save random state for reproducibility
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.eps_momentum = eps_momentum
        self.eps_linear = eps_linear   
        self.target_df = target_df
        
        self.estimators_ = []
        self.intercept_ = 0.0
        self.feature_momentum = {} 
        self.feature_knots_ = {} # Stores knots for splines
        
        # Cache for competing mode (pre-computed matrices)
        self.competing_assets_ = {} 
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [], 
            'selected_features': [],
            'selected_learners': []
        }

    def _get_gradient(self, y_pred, y):
        # returns gradient of loss with respect to prediction
        grad = (y_pred - y)
        if self.loss == 'flooding':
            # invert gradient if loss falls below flood level to push it back up
            mse = torch.mean((y_pred - y)**2)
            if mse < self.flood_level:
                grad *= -1.0 
        return grad

    def _select_feature(self, losses_tensor: torch.Tensor) -> int:
        # Get total number of features
        n_features = len(losses_tensor)
        
        if self.use_momentum:
            # initialize momentum if empty
            if len(self.feature_momentum) == 0:
                for i in range(n_features): self.feature_momentum[i] = 0.0
            
            worst_loss = torch.max(losses_tensor).detach()
            mom_vec = torch.tensor([self.feature_momentum[i] for i in range(n_features)], device=losses_tensor.device)
            # decay previous momentum
            mom_vec *= self.momentum_decay
            
            # calculate gain relative to worst loss
            gains = worst_loss - losses_tensor
            max_gain = torch.max(gains)
            scores = gains / (max_gain + 1e-8)
            # add normalized score to momentum vector
            mom_vec += scores
            
            # update stored momentum
            for i in range(n_features):
                self.feature_momentum[i] = mom_vec[i].item()
                
            loss_std = torch.std(losses_tensor).detach()
            # compute scale factor for adjustment
            scale_factor = loss_std if loss_std > 1e-9 else 1.0
            adjustment = mom_vec * self.momentum_strength * scale_factor

            # Adjust losses
            adjusted_losses = losses_tensor - adjustment
        else:
            adjusted_losses = losses_tensor

        # Apply top-k logic
        if self.use_top_k:
            k = min(self.top_k, n_features)
            top_k_indices = torch.topk(adjusted_losses, k, largest=False).indices
            # calculate weights and select feature
            weights = torch.arange(k, 0, -1, device=losses_tensor.device, dtype=torch.float32)
            rank_idx = torch.multinomial(weights, 1).item()
            selected_idx = top_k_indices[rank_idx].item()
        else:
            selected_idx = torch.argmin(adjusted_losses).item()
            
        return selected_idx

    # Orthogonal Decomposition & Precomputation
    def _prepare_orthogonal_bases(self, X):
        # Precompute orthogonalized bases and solver matrices for competing learners

        n_samples, n_features = X.shape
        device = X.device
        
        assets = {}

        # Linear basis construction used for projection
        ones = torch.ones(n_samples, 1, device=device)
        X_lin_list = []
        for i in range(n_features):
            X_lin_list.append(torch.cat([ones, X[:, i:i+1]], dim=1))
        X_lin_all = torch.stack(X_lin_list, dim=0) # (F, N, 2)
        
        # Precompute projection matrices
        XTX = torch.bmm(X_lin_all.transpose(1, 2), X_lin_all)
        # Regularize matrix for numerical stability
        XTX += torch.eye(2, device=device).unsqueeze(0) * self.eps_linear
        
        Gamma_proj = torch.linalg.solve(XTX, X_lin_all.transpose(1, 2))

        for learner_type in self.base_learners:
            if learner_type == 'tree' or learner_type == 'linear':
                # no decomp/penalization needed for these
                continue

            B_list = []
            
            if learner_type == 'polynomial':
                # Generate Poly Basis
                exponents = torch.arange(1, self.poly_degree + 1, device=device).float()
                for i in range(n_features):
                    poly_feats = X[:, i:i+1].pow(exponents)
                    bias = torch.ones(n_samples, 1, device=device)
                    B_list.append(torch.cat([bias, poly_feats], dim=1))
                # Ridge Penalty    
                Omega = torch.eye(self.poly_degree + 1, device=device) 

            elif learner_type == 'bspline':
                # Generate B-Spline Basis
                X_np = X.detach().cpu().numpy()
                # calculate fixed target dimensions for basis
                target_K = self.n_knots + self.spline_degree + 1
                # Construct Omega
                dummy_eye = np.eye(target_K)
                D_fixed = np.diff(dummy_eye, n=2, axis=0)
                Omega = torch.from_numpy(D_fixed.T @ D_fixed).float().to(device)

                for i in range(n_features):
                    # Determine knots with quantiles
                    percentiles = np.linspace(0, 100, self.n_knots + 2)
                    knots_all = np.unique(np.percentile(X_np[:, i], percentiles))
                    if len(knots_all) < 2:
                        # Fallback for constant features to prevent singular matrices
                        knots_all = np.array([X_np[:, i].min(), X_np[:, i].max()])
                        
                    # Construct full knot vector
                    f_min, f_max = X_np[:, i].min(), X_np[:, i].max()
                    t = np.concatenate(([f_min]*self.spline_degree, [f_min], knots_all[1:-1], [f_max], [f_max]*self.spline_degree))
                    self.feature_knots_[i] = t
                    
                    # Design Matrix
                    dm_np = BSpline.design_matrix(X_np[:, i], t, self.spline_degree).toarray()
                
                # Paddding logc
                    current_K = dm_np.shape[1]
                    if current_K < target_K:
                        # Pad with zero columns on the right if matrix is smaller than target
                        pad_width = target_K - current_K
                        dm_np = np.pad(dm_np, ((0, 0), (0, pad_width)), mode='constant')
                    elif current_K > target_K:
                        # safety crop if matrix exceeds target dimension
                        dm_np = dm_np[:, :target_K]
                    
                    B_list.append(torch.from_numpy(dm_np).float().to(device))
                
            B_all = torch.stack(B_list, dim=0)
            
            # Orthogonal Transformation
            Gamma = torch.bmm(Gamma_proj, B_all)
            Projected = torch.bmm(X_lin_all, Gamma)
            B_tilde = B_all - Projected
            
            Solver_matrices = []
            
            # Helper to calculate degrees of freedom
            def calc_df(lam, B_mat, Om):
                BtB = B_mat.T @ B_mat
                n_k = BtB.shape[0]
                M = BtB + lam * Om
                try:
                    Inv = torch.linalg.inv(M)
                except:
                    Inv = torch.linalg.inv(M + torch.eye(n_k, device=device)*1e-6)
                S = Inv @ BtB
                return torch.trace(S).item()

            for i in range(n_features):
                b_curr = B_tilde[i] # (N, K)
                
                # Target DF constraint
                max_rank = min(b_curr.shape)

                if torch.all(b_curr.abs() < 1e-9):
                    # zero matrix case if basis is completely flat
                    Solver = torch.zeros(b_curr.shape[1], b_curr.shape[0], device=device)
                else:
                    target = min(self.target_df, max_rank - 0.1)
                    
                    # solve for optimal penalty lambda
                    res = minimize_scalar(
                        lambda l: (calc_df(10**l, b_curr, Omega) - target)**2,
                        bounds=(-5, 5), method='bounded'
                    )
                    best_lam = 10**res.x

                    # compute penalized least squares solver matrix
                    BtB = b_curr.T @ b_curr
                    M_inv = torch.linalg.inv(BtB + best_lam * Omega + torch.eye(BtB.shape[0], device=device)*self.eps_linear)
                    Solver = M_inv @ b_curr.T
                
                Solver_matrices.append(Solver)
                
            # Stack solvers
            Solvers_stacked = torch.stack(Solver_matrices, dim=0)
            
            assets[learner_type] = {
                'B_tilde': B_tilde,
                'Solver': Solvers_stacked,
                'Gamma': Gamma
            }
            
        return assets, X_lin_all

    # Legacy Solvers for single learner mode
    
    def _solve_linear_vectorized(self, X, target):
        # Solve linear equations using vectorized operations
        target = target.unsqueeze(1)
        numer = (X * target).sum(dim=0)
        denom = (X ** 2).sum(dim=0)
        beta = numer / (denom + self.eps_linear)

        # calculate predictions and losses
        preds = X * beta.unsqueeze(0) 
        losses = ((preds - target)**2).mean(dim=0)
        return beta, losses

    def _solve_poly_vectorized(self, X, target):
        n_samples, n_features = X.shape
        device = X.device

        # initialize polynomial feature exponents and expand input dimensions
        exponents = torch.arange(1, self.poly_degree + 1, device=device).float()
        X_expanded = X.unsqueeze(-1)
        poly_features = X_expanded.pow(exponents)
        bias = torch.ones(n_samples, n_features, 1, device=device)

        # construct design matrices with bias term for all features
        A = torch.cat([bias, poly_features], dim=2)
        A = A.permute(1, 0, 2)
        Y = target.view(1, n_samples, 1).expand(n_features, n_samples, 1)
        A_T = A.transpose(1, 2)
        ATA = torch.bmm(A_T, A)
        ATY = torch.bmm(A_T, Y)
        I = torch.eye(self.poly_degree + 1, device=device).unsqueeze(0).expand(n_features, -1, -1)
        ATA_reg = ATA + self.eps_linear * I

        # solve regularized normal equations
        beta = torch.linalg.solve(ATA_reg, ATY)
        preds = torch.bmm(A, beta).squeeze(-1)
        target_rep = target.unsqueeze(0)

        # compute batch predictions and resulting MSE for each feature
        losses = ((preds - target_rep)**2).mean(dim=1)
        return beta.squeeze(-1), losses

    def _solve_tree_vectorized(self, X_binned, target, bin_edges):
        # solves decision stump for all features simultaneously using bins
        n_samples, n_features = X_binned.shape
        n_bins = self.n_bins
        device = X_binned.device

        # Histogram Aggregation (vectorized)
        # create flat indices for fast histogram aggregation
        offsets = (torch.arange(n_features, device=device) * n_bins).view(1, -1)
        flat_indices = (X_binned + offsets).view(-1)  
        flat_target = target.view(-1, 1).expand(-1, n_features).reshape(-1)
        
        G_flat = torch.zeros(n_features * n_bins, device=device)
        N_flat = torch.zeros(n_features * n_bins, device=device)
        
        G_flat.index_add_(0, flat_indices, flat_target)
        N_flat.index_add_(0, flat_indices, torch.ones_like(flat_target))
        
        G = G_flat.view(n_features, n_bins)
        N = N_flat.view(n_features, n_bins)
        
        # compute cumulative sums for left side
        G_L = torch.cumsum(G, dim=1)
        N_L = torch.cumsum(N, dim=1)
        
        G_T = G_L[:, -1:] 
        N_T = N_L[:, -1:]
        # compute right side using totals
        G_R = G_T - G_L
        N_R = N_T - N_L
        
        eps = 1e-6
        # Calculate gain
        gain = (G_L**2 / (N_L + eps)) + (G_R**2 / (N_R + eps))
        
        # mask invalid splits where leaf has zero samples
        valid_mask = (N_L > 0) & (N_R > 0)
        valid_mask[:, -1] = False 
        gain[~valid_mask] = -1.0 
        
        # maximize gain minimizes mse
        max_gain_per_feat = torch.max(gain, dim=1).values
        
        total_ss = torch.sum(target**2)
        
        # Calculate Actual MSE for best split of each feature
        # gives tree loss same scale as Linear/Poly/Spline
        mse_per_feat = (total_ss - max_gain_per_feat) / n_samples
        
        return gain, mse_per_feat
    
    def _solve_bspline_vectorized(self, A, target):
        n_features, n_samples, n_basis = A.shape
        device = A.device

        # reshape target and transpose design matrices for batched operations
        Y = target.view(1, n_samples, 1).expand(n_features, n_samples, 1)
        A_T = A.transpose(1, 2)
        ATA = torch.bmm(A_T, A)
        ATY = torch.bmm(A_T, Y)
        I = torch.eye(n_basis, device=device).unsqueeze(0).expand(n_features, -1, -1)
        ATA_reg = ATA + self.eps_linear * I

        # compute regularized system matrices and solve for spline coefficients
        beta = torch.linalg.solve(ATA_reg, ATY)
        preds = torch.bmm(A, beta).squeeze(-1)
        target_rep = target.unsqueeze(0)

        # calculate preds and losses
        losses = ((preds - target_rep)**2).mean(dim=1)
        return beta.squeeze(-1), losses


    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        # Convert inputs to tensors
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        if X_val is not None:
            # Convert validation inputs
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, dtype=torch.float32)
        if X_test is not None:
            # Convert test inputs
            X_test = torch.as_tensor(X_test, dtype=torch.float32)
            y_test = torch.as_tensor(y_test, dtype=torch.float32)

        # Store intercept as initial prediction
        self.intercept_ = torch.mean(y_train).item()
        curr_pred_train = torch.full_like(y_train, self.intercept_)
        
        curr_pred_val = None
        if X_val is not None: curr_pred_val = torch.full_like(y_val, self.intercept_)
        
        curr_pred_test = None
        if X_test is not None: curr_pred_test = torch.full_like(y_test, self.intercept_)

        n_samples, n_features = X_train.shape

        # Compute B-spline knots if legacy mode is active
        if "bspline" in self.base_learners and self.legacy_mode:
            # Use numpy for quantile calculation
            X_train_np = X_train.detach().cpu().numpy()
            for f_idx in range(n_features):
                if f_idx not in self.feature_knots_: 
                    percentiles = np.linspace(0, 100, self.n_knots + 2)

                    # Determine knots
                    knots_all = np.unique(np.percentile(X_train_np[:, f_idx], percentiles))
                    if len(knots_all) < 2: knots_all = np.array([X_train_np[:, f_idx].min(), X_train_np[:, f_idx].max()])
                    f_min, f_max = X_train_np[:, f_idx].min(), X_train_np[:, f_idx].max()

                    # Construct knot vector
                    t = np.concatenate(([f_min]*self.spline_degree, [f_min], knots_all[1:-1], [f_max], [f_max]*self.spline_degree))
                    self.feature_knots_[f_idx] = t

        # Compute tree bins if needed
        X_train_binned = None
        self.all_bin_edges = None
        if "tree" in self.base_learners:
            X_train_contig = X_train.contiguous()
            percentiles = torch.linspace(0, 1, self.n_bins + 1, device=X_train.device)

            # Calculate bin edges using quantiles
            self.all_bin_edges = torch.quantile(X_train_contig, percentiles, dim=0).T
            self.all_bin_edges[:, -1] += 1e-4
            X_binned_list = []
            for f_idx in range(n_features):
                edges, _ = torch.sort(self.all_bin_edges[f_idx])
                binned = torch.bucketize(X_train_contig[:, f_idx], edges)
                binned = torch.clamp(binned - 1, 0, self.n_bins - 1)
                X_binned_list.append(binned)

            # Map feature values to bins 
            X_train_binned = torch.stack(X_binned_list, dim=1)

        # Competing Mode Pre-computation
        if not self.legacy_mode:
            self.competing_assets_, _ = self._prepare_orthogonal_bases(X_train)
        else:
            # Preparation for legacy B-spline (design matrix)
            if self.base_learner == "bspline":
                X_np = X_train.detach().cpu().numpy()
                basis_matrices = []
                for f_idx in range(n_features):
                    knots = self.feature_knots_[f_idx]
                    x_col = np.clip(X_np[:, f_idx], knots[0], knots[-1])
                    dm = BSpline.design_matrix(x_col, knots, self.spline_degree)
                    # padding logic for legacy
                    target_K = self.n_knots + self.spline_degree + 1
                    dm_arr = dm.toarray()
                    # pad matrix if smaller than target dimension
                    if dm_arr.shape[1] < target_K:
                        pad_w = target_K - dm_arr.shape[1]
                        dm_arr = np.pad(dm_arr, ((0,0), (0, pad_w)), mode='constant')
                    elif dm_arr.shape[1] > target_K:
                        dm_arr = dm_arr[:, :target_K]
                        
                    basis_matrices.append(torch.from_numpy(dm_arr).float().to(X_train.device))
                self.A_bspline_legacy = torch.stack(basis_matrices, dim=0)

        # Boosting loop
        best_val_loss = float('inf')
        self.best_iteration_ = 0

        for i in range(self.n_estimators):        
            grad = self._get_gradient(curr_pred_train, y_train)
            target = -grad

            best_idx = -1
            best_params = None
            best_model_obj = None
            best_learner_type = self.base_learner

            # If using one base learner:
            if self.legacy_mode:
                if self.base_learner == "linear":
                    betas, losses = self._solve_linear_vectorized(X_train, target)
                    best_idx = self._select_feature(losses)
                    best_params = betas[best_idx]
                    
                elif self.base_learner == "polynomial":
                    betas, losses = self._solve_poly_vectorized(X_train, target)
                    best_idx = self._select_feature(losses)
                    best_params = betas[best_idx]
                    
                elif self.base_learner == "bspline":
                    betas, losses = self._solve_bspline_vectorized(self.A_bspline_legacy, target)
                    best_idx = self._select_feature(losses)
                    best_params = {'coeffs': betas[best_idx], 'knots': self.feature_knots_[best_idx]}
                    
                elif self.base_learner == "tree":
                    gains, losses = self._solve_tree_vectorized(X_train_binned, target, self.all_bin_edges)
                    best_idx = self._select_feature(losses)
                    best_bin_idx = torch.argmax(gains[best_idx]).item()
                    
                    # Recompute leaf values using optimal bin index
                    f_binned = X_train_binned[:, best_idx]
                    mask_left = f_binned <= best_bin_idx
                    val_left = target[mask_left].mean()
                    val_right = target[~mask_left].mean()
                    
                    best_params = {
                        'threshold': self.all_bin_edges[best_idx, best_bin_idx + 1].item(),
                        'left_val': val_left.item(),
                        'right_val': val_right.item()
                    }

            # Competing base learner mode
            else:
                competitors_loss = []
                competitors_meta = []
                
                # Evaluate all base learners to find best match
                for l_type in self.base_learners:
                    if l_type == 'tree':
                        # Tree is not pre-computed/vectorized
                        gains, losses = self._solve_tree_vectorized(X_train_binned, target, self.all_bin_edges)
                        competitors_loss.append(losses)
                        competitors_meta.append({'type': 'tree', 'gains': gains})
                        
                    elif l_type == 'linear':
                        betas, losses = self._solve_linear_vectorized(X_train, target)
                        competitors_loss.append(losses)
                        competitors_meta.append({'type': 'linear', 'betas': betas})
                        
                    elif l_type in self.competing_assets_:
                        # Poly and B-spline: orthogonalized and penalized
                        assets = self.competing_assets_[l_type]
                        Solver = assets['Solver']
                        B_tilde = assets['B_tilde']

                        target_exp = target.view(1, n_samples, 1).expand(n_features, n_samples, 1)
                        
                        betas = torch.bmm(Solver, target_exp)
                        preds = torch.bmm(B_tilde, betas).squeeze(-1)
                        
                        # Loss
                        target_rep = target.unsqueeze(0)
                        losses = ((preds - target_rep)**2).mean(dim=1)
                        
                        competitors_loss.append(losses)
                        competitors_meta.append({'type': l_type, 'betas': betas.squeeze(-1)})

                # Global selection
                all_losses = torch.stack(competitors_loss, dim=0)

                min_losses_per_feat, best_learner_indices = torch.min(all_losses, dim=0)
                
                # Select best feature and learner combination
                best_idx = self._select_feature(min_losses_per_feat)
                
                # Retrieve which learner won for this feature
                winner_learner_idx = best_learner_indices[best_idx].item()
                winner_meta = competitors_meta[winner_learner_idx]
                best_learner_type = winner_meta['type']
                
                # Construct params
                if best_learner_type == 'tree':
                    # Reconstruct tree params
                    feat_gains = winner_meta['gains'][best_idx]
                    best_bin_idx = torch.argmax(feat_gains).item()
                    f_binned = X_train_binned[:, best_idx]
                    mask_left = f_binned <= best_bin_idx
                    val_left = target[mask_left].mean().item()
                    val_right = target[~mask_left].mean().item()
                    best_params = {
                        'threshold': self.all_bin_edges[best_idx, best_bin_idx + 1].item(),
                        'left_val': val_left, 'right_val': val_right
                    }
                    
                elif best_learner_type == 'linear':
                    best_params = winner_meta['betas'][best_idx]
                    
                else:
                    # Poly / B-spline
                    beta_orth = winner_meta['betas'][best_idx]
                    
                    Gamma_f = self.competing_assets_[best_learner_type]['Gamma'][best_idx]
                    
                    # Calculate linear adjustment for orthogonalized bases
                    beta_lin_adj = - torch.mv(Gamma_f, beta_orth)
                    
                    best_params = {
                        'beta': beta_orth,
                        'beta_lin': beta_lin_adj
                    }
                    if best_learner_type == 'bspline':
                        best_params['knots'] = self.feature_knots_[best_idx]

            # Store and Update
            self.estimators_.append({
                'idx': best_idx,
                'learner': best_learner_type,
                'params': best_params,
                'model': best_model_obj
            })
            self.history['selected_features'].append(best_idx)
            self.history['selected_learners'].append(best_learner_type)

            # Update Predictions
            def apply_update(X_in, learner_type, f_idx, params):
                x_f = X_in[:, f_idx:f_idx+1]
                N = x_f.shape[0]
                pred = torch.zeros(N, device=X_in.device)

                # Linear learner
                if learner_type == 'linear':
                    pred = (x_f * params).flatten()

                # Decision stump learner
                elif learner_type == 'tree':
                    pred = torch.where(
                        x_f <= params['threshold'],
                        torch.tensor(params['left_val'], device=x_f.device),
                        torch.tensor(params['right_val'], device=x_f.device)
                    ).flatten()

                # Polynomial learner
                elif learner_type == 'polynomial':
                    if isinstance(params, dict) and 'beta_lin' in params:
                        coeffs = params['beta']
                        val_poly = torch.full((N,), coeffs[0].item(), device=X_in.device)
                        pow_x = x_f.flatten()
                        for p in range(1, len(coeffs)):
                            val_poly += coeffs[p] * pow_x
                            pow_x = pow_x * x_f.flatten()
                        
                        lin_coeffs = params['beta_lin']
                        val_lin = lin_coeffs[0] + lin_coeffs[1] * x_f.flatten()
                        
                        pred = val_poly + val_lin
                    else:
                        # Legacy
                        coeffs = params
                        pred = torch.full((N,), coeffs[0].item(), device=X_in.device)
                        pow_x = x_f.flatten()
                        for p in range(1, len(coeffs)):
                            pred += coeffs[p] * pow_x
                            pow_x = pow_x * x_f.flatten()
                            
                elif learner_type == 'bspline':
                    # Similar logic
                    if isinstance(params, dict) and 'beta_lin' in params:
                        coeffs = params['beta'].detach().cpu().numpy()
                        knots = params['knots']
                        lin_coeffs = params['beta_lin']
                        
                        x_np = x_f.flatten().detach().cpu().numpy()
                        x_np = np.clip(x_np, knots[0], knots[-1])
                        dm = BSpline.design_matrix(x_np, knots, self.spline_degree).toarray()

                        # Fix prediction dimension match if dm < coeffs (due to padding)
                        if dm.shape[1] < coeffs.shape[0]:
                            dm = np.pad(dm, ((0,0), (0, coeffs.shape[0] - dm.shape[1])), mode='constant')
                        elif dm.shape[1] > coeffs.shape[0]:
                            dm = dm[:, :coeffs.shape[0]]

                        val_spline = torch.from_numpy(dm @ coeffs).float().to(X_in.device)
                        val_lin = lin_coeffs[0] + lin_coeffs[1] * x_f.flatten()
                        
                        pred = val_spline + val_lin
                    else:
                        # Legacy
                        coeffs = params['coeffs'].detach().cpu().numpy()
                        knots = params['knots']
                        x_np = x_f.flatten().detach().cpu().numpy()
                        x_np = np.clip(x_np, knots[0], knots[-1])
                        dm = BSpline.design_matrix(x_np, knots, self.spline_degree).toarray()
                        if dm.shape[1] < coeffs.shape[0]:
                            dm = np.pad(dm, ((0,0), (0, coeffs.shape[0] - dm.shape[1])), mode='constant')
                        
                        pred = torch.from_numpy(dm @ coeffs).float().to(X_in.device)
                        
                return pred

            learner_data = self.estimators_[-1]

            # compute update
            update_train = apply_update(X_train, best_learner_type, best_idx, best_params) * self.learning_rate
            curr_pred_train += update_train
            
            # update validation predictions
            if X_val is not None:
                curr_pred_val += apply_update(X_val, best_learner_type, best_idx, best_params) * self.learning_rate
                val_mse = torch.mean((curr_pred_val - y_val)**2).item()
                self.history['val_loss'].append(val_mse)
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    self.best_iteration_ = i + 1
            
            # update test predictions
            if X_test is not None:
                curr_pred_test += apply_update(X_test, best_learner_type, best_idx, best_params) * self.learning_rate
                self.history['test_loss'].append(torch.mean((curr_pred_test - y_test)**2).item())

            # record train performance
            train_mse = torch.mean((curr_pred_train - y_train)**2).item()
            self.history['train_loss'].append(train_mse)

            if (i+1) % 50 == 0:
                print(f"Iter {i+1}/{self.n_estimators} | Train MSE: {train_mse:.5f}")

    def predict(self, X, use_best_model=False):
        X = torch.as_tensor(X, dtype=torch.float32)

        # initialize pred with intercept
        pred = torch.full((X.shape[0],), self.intercept_)
        
        limit = self.best_iteration_ if use_best_model and self.best_iteration_ > 0 else len(self.estimators_)

        # determine model depth
        estimators_to_use = self.estimators_[:limit]
        
        # iterate through estimators to accumulate updates from features
        for est in estimators_to_use:
            f_idx = est['idx']
            l_type = est['learner']
            params = est['params']
            x_f = X[:, f_idx:f_idx+1]
            N = x_f.shape[0]
            
            update = None
            
            # update for linear learner
            if l_type == 'linear':
                update = (x_f * params).flatten()

            # update for tree learner
            elif l_type == 'tree':
                update = torch.where(
                    x_f <= params['threshold'],
                    torch.tensor(params['left_val'], device=X.device),
                    torch.tensor(params['right_val'], device=X.device)
                ).flatten()
            
            # update for polynomial learner
            elif l_type == 'polynomial':
                if isinstance(params, dict) and 'beta_lin' in params:
                    coeffs = params['beta']
                    val_poly = torch.full((N,), coeffs[0].item(), device=X.device)
                    pow_x = x_f.flatten()
                    for p in range(1, len(coeffs)):
                        val_poly += coeffs[p] * pow_x
                        pow_x = pow_x * x_f.flatten()
                    lin_coeffs = params['beta_lin']
                    val_lin = lin_coeffs[0] + lin_coeffs[1] * x_f.flatten()
                    update = val_poly + val_lin
                else:
                    coeffs = params
                    update = torch.full((N,), coeffs[0].item(), device=X.device)
                    pow_x = x_f.flatten()
                    for p in range(1, len(coeffs)):
                        update += coeffs[p] * pow_x
                        pow_x = pow_x * x_f.flatten()
            
            # update for B-spline learner
            elif l_type == 'bspline':
                if isinstance(params, dict) and 'beta_lin' in params:
                    coeffs = params['beta'].detach().cpu().numpy()
                    knots = params['knots']
                    lin_coeffs = params['beta_lin']
                    x_np = x_f.flatten().detach().cpu().numpy()
                    x_np = np.clip(x_np, knots[0], knots[-1])
                    dm = BSpline.design_matrix(x_np, knots, self.spline_degree).toarray()
                    
                    # Pad b-spline design matrix if needed
                    if dm.shape[1] < coeffs.shape[0]:
                        dm = np.pad(dm, ((0,0), (0, coeffs.shape[0] - dm.shape[1])), mode='constant')
                    
                    val_spline = torch.from_numpy(dm @ coeffs).float().to(X.device)
                    val_lin = lin_coeffs[0] + lin_coeffs[1] * x_f.flatten()
                    update = val_spline + val_lin
                else:
                    coeffs = params['coeffs'].detach().cpu().numpy()
                    knots = params['knots']
                    x_np = x_f.flatten().detach().cpu().numpy()
                    x_np = np.clip(x_np, knots[0], knots[-1])
                    dm = BSpline.design_matrix(x_np, knots, self.spline_degree).toarray()
                    
                    # Pad b-spline design matrix if needed
                    if dm.shape[1] < coeffs.shape[0]:
                        dm = np.pad(dm, ((0,0), (0, coeffs.shape[0] - dm.shape[1])), mode='constant')

                    update = torch.from_numpy(dm @ coeffs).float().to(X.device)
            
            # Accumulate updates to prediction
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