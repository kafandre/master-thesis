import torch
import numpy as np
from typing import Optional, Callable


class LinearRegressionModel:
    """
    Linear Regression model using PyTorch with support for custom loss functions.
    Uses a single linear layer and SGD optimization.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        loss: str = 'mse',  # 'mse' for mean square error, 'flooding' for flooding loss
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.0
    ):
        self.random_state = random_state
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Will be set during fitting
        self.model = None
        self.loss_fn = None
        self.flood_level = None
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
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
            self.flood_level = kwargs.get('flood_level', 0.02)
            return lambda y_pred, y: (abs(torch.mean((y_pred - y) ** 2) - self.flood_level) + self.flood_level)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
    
    def fit(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        loss: Optional[str] = None,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **loss_params
    ) -> 'LinearRegressionModel':
        """
        Fit the linear regression model.
        
        Args:
            X: Input features tensor of shape (n_samples, n_features)
            y: Target tensor of shape (n_samples,)
            loss: Loss function type (overrides the one specified in __init__)
            epochs: Number of training epochs
            batch_size: Batch size for SGD (None means full batch)
            verbose: Whether to print training progress
            **loss_params: Additional parameters for the loss function
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Create model
        self.model = torch.nn.Linear(n_features, 1)
        
        # Set or update loss function
        loss_type = loss if loss is not None else self.loss
        self.loss_fn = self._get_loss_fn(loss_type, **loss_params)
        
        # Create optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        
        # Create data loader for mini-batch training
        if batch_size is None:
            batch_size = n_samples
        
        # Reshape y to match model output
        y_reshaped = y.view(-1, 1)
        
        # Training loop
        for epoch in range(epochs):
            # Generate random indices for this epoch
            indices = torch.randperm(n_samples)
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                # Get mini-batch indices
                batch_indices = indices[i:i+batch_size]
                
                # Get mini-batch data
                X_batch = X[batch_indices]
                y_batch = y_reshaped[batch_indices]
                
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Compute loss
                loss_value = self.loss_fn(y_pred, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                with torch.no_grad():
                    y_pred = self.model(X)
                    current_loss = self.loss_fn(y_pred, y_reshaped).item()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.6f}")
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions for the input data.
        
        Args:
            X: Input features tensor of shape (n_samples, n_features)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        with torch.no_grad():
            return self.model(X).flatten()
    
    def get_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the current loss on the given data.
        
        Args:
            X: Input features tensor
            y: True target values
            
        Returns:
            Loss value
        """
        if self.model is None or self.loss_fn is None:
            raise ValueError("Model must be fitted before calculating loss")
        
        with torch.no_grad():
            predictions = self.predict(X)
            y_reshaped = y.view(-1, 1) if y.dim() == 1 else y
            return self.loss_fn(predictions.view(-1, 1), y_reshaped).item()