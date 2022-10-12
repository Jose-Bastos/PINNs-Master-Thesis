"""This module implements convergence criteria"""
import torch


class Convergence:
    """Implements convergence criterium. Convergence is when change in patience
    epochs is smaller than delta.
    """

    def __init__(self, patience: int = 200, delta: float = 1e-3) -> None:
        """Implements convergence criterium. Convergence is when change in patience
        epochs is smaller than delta.
        Args:
            patience (int): how often to check for convergence
            delta (float): desired accuracy
        """
        self.patience = patience
        self.delta = delta
        self.start_iteration = None
        self.startMSE_Test = None

    def __call__(self, iteration: int, mse_test: torch.Tensor) -> bool:
        """

        Args:
            epoch (int): Current epoch of the optimization
            l1_norm (torch.Tensor): Value of the L1 norm
        """
        converged = False  # overwrite later

        # Initialize if doesn't exist
        if self.startMSE_Test is None:
            self.startMSE_Test = mse_test
            self.start_iteration = iteration

        # Check if change is smaller than delta and if we've exceeded patience

        elif torch.abs(self.startMSE_Test - mse_test.item()) < self.delta:
            if (iteration - self.start_iteration) >= self.patience:
                converged = True

        # If not, reset and keep going
        else:
            self.startMSE_Test = mse_test
            self.start_iteration = iteration

        return converged
