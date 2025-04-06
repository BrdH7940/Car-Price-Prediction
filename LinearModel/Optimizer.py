import numpy as np
from typing import *

class BaseOptimizer:
    """Base class for optimization algorithms."""

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize the optimizer.

        Parameters:
        -----------
        learning_rate : float, default=0.01
            The step size for parameter updates.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights = None

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_weights(self, n_features: int) -> np.ndarray:
        """
        Initialize weights using Xavier initialization.

        Parameters:
        -----------
        n_features : int
            Number of input features.

        Returns:
        --------
        np.ndarray
            Initialized weights.
        """
        # Xavier initialization
        limit = np.sqrt(6 / (n_features + 1))
        return np.random.uniform(-limit, limit, (n_features,))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to X for the intercept term.

        Parameters:
        -----------
        X : np.ndarray
            Input features.

        Returns:
        --------
        np.ndarray
            X with an added column of ones.
        """
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray,
            l1_penalty: float = 0.0, l2_penalty: float = 0.0) -> np.ndarray:
        """
        Solve for weights using Normal equation.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        l1_penalty : float, default=0.0
            L1 regularization parameter.
        l2_penalty : float, default=0.0
            L2 regularization parameter.

        Returns:
        --------
        np.ndarray
            Optimized weights.
        """
        # Normal equation: w = (X^T X + λI)^(-1) X^T y
        # Note: Normal equation doesn't directly support L1 regularization
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = self._initialize_weights(n_features)

        # For normal equation with L2 regularization
        if l2_penalty > 0:
            identity = np.eye(n_features)
            identity[0, 0] = 0  # Don't regularize the bias term
            XTX = X_with_intercept.T @ X_with_intercept + l2_penalty * identity
        else:
            XTX = X_with_intercept.T @ X_with_intercept

        XTy = X_with_intercept.T @ y

        try:
            # Solve the normal equation
            self.weights = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            self.weights = np.linalg.pinv(XTX) @ XTy

        return self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained weights.

        Parameters:
        -----------
        X : np.ndarray
            Input features.

        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        X_with_intercept = self._add_intercept(X)
        return X_with_intercept @ self.weights


class GradientDescent(BaseOptimizer):
    """Gradient Descent optimizer."""

    def fit(self, X: np.ndarray, y: np.ndarray,
            l1_penalty: float = 0.0, l2_penalty: float = 0.0,
            verbose: bool = False) -> np.ndarray:
        """
        Fit the model using Gradient Descent.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        l1_penalty : float, default=0.0
            L1 regularization parameter.
        l2_penalty : float, default=0.0
            L2 regularization parameter.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        np.ndarray
            Optimized weights.
        """
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = self._initialize_weights(n_features)

        prev_cost = float('inf')

        for iteration in range(self.max_iter):
            # Compute predictions
            y_pred = X_with_intercept @ self.weights

            # Compute gradient of MSE
            gradient = (1 / n_samples) * X_with_intercept.T @ (y_pred - y)

            # Add L2 regularization gradient if specified
            if l2_penalty > 0:
                # Don't regularize intercept
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0
                gradient += l2_penalty * reg_weights

            # Add L1 regularization gradient if specified
            if l1_penalty > 0:
                # Don't regularize intercept
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0
                # L1 gradient is sign of weights
                gradient += l1_penalty * np.sign(reg_weights)

            # Update weights
            self.weights -= self.learning_rate * gradient

            # Compute current cost
            y_pred = X_with_intercept @ self.weights
            mse = np.mean((y_pred - y) ** 2)

            # Add regularization to cost if specified
            if l2_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l2_penalty * np.sum(reg_weights ** 2) / 2

            if l1_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l1_penalty * np.sum(np.abs(reg_weights))

            # Check for convergence
            if abs(prev_cost - mse) < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                break

            prev_cost = mse

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Cost: {mse:.6f}")

        return self.weights


class StochasticGradientDescent(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""

    def fit(self, X: np.ndarray, y: np.ndarray,
            l1_penalty: float = 0.0, l2_penalty: float = 0.0,
            verbose: bool = False) -> np.ndarray:
        """
        Fit the model using Stochastic Gradient Descent.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        l1_penalty : float, default=0.0
            L1 regularization parameter.
        l2_penalty : float, default=0.0
            L2 regularization parameter.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        np.ndarray
            Optimized weights.
        """
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = self._initialize_weights(n_features)

        # For tracking convergence
        prev_cost = float('inf')
        indices = np.arange(n_samples)

        for iteration in range(self.max_iter):
            # Shuffle the data
            np.random.shuffle(indices)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            # Process each sample
            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]
                y_i = y_shuffled[i:i + 1]

                # Compute prediction
                y_pred = x_i @ self.weights

                # Compute gradient
                gradient = x_i.T @ (y_pred - y_i)

                # Add L2 regularization gradient if specified
                if l2_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.weights)
                    reg_weights[0] = 0
                    gradient += l2_penalty * reg_weights

                # Add L1 regularization gradient if specified
                if l1_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.weights)
                    reg_weights[0] = 0
                    # L1 gradient is sign of weights
                    gradient += l1_penalty * np.sign(reg_weights)

                # Update weights
                self.weights -= self.learning_rate * gradient

            # Compute current cost on full dataset
            y_pred = X_with_intercept @ self.weights
            mse = np.mean((y_pred - y) ** 2)

            # Add regularization to cost if specified
            if l2_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l2_penalty * np.sum(reg_weights ** 2) / 2

            if l1_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l1_penalty * np.sum(np.abs(reg_weights))

            # Check for convergence
            if abs(prev_cost - mse) < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                break

            prev_cost = mse

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Cost: {mse:.6f}")

        return self.weights


class MiniBatchGradientDescent(BaseOptimizer):
    """Mini-batch Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 batch_size: int = 32):
        """
        Initialize the optimizer.

        Parameters:
        -----------
        learning_rate : float, default=0.01
            The step size for parameter updates.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        random_state : int, optional
            Random seed for reproducibility.
        batch_size : int, default=32
            Size of mini-batches.
        """
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray,
            l1_penalty: float = 0.0, l2_penalty: float = 0.0,
            verbose: bool = False) -> np.ndarray:
        """
        Fit the model using Mini-batch Gradient Descent.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        l1_penalty : float, default=0.0
            L1 regularization parameter.
        l2_penalty : float, default=0.0
            L2 regularization parameter.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        np.ndarray
            Optimized weights.
        """
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = self._initialize_weights(n_features)

        # For tracking convergence
        prev_cost = float('inf')
        indices = np.arange(n_samples)

        for iteration in range(self.max_iter):
            # Shuffle the data
            np.random.shuffle(indices)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]

            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Compute prediction
                y_pred = X_batch @ self.weights

                # Compute gradient
                gradient = (1 / len(X_batch)) * X_batch.T @ (y_pred - y_batch)

                # Add L2 regularization gradient if specified
                if l2_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.weights)
                    reg_weights[0] = 0
                    gradient += l2_penalty * reg_weights

                # Add L1 regularization gradient if specified
                if l1_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.weights)
                    reg_weights[0] = 0
                    # L1 gradient is sign of weights
                    gradient += l1_penalty * np.sign(reg_weights)

                # Update weights
                self.weights -= self.learning_rate * gradient

            # Compute current cost on full dataset
            y_pred = X_with_intercept @ self.weights
            mse = np.mean((y_pred - y) ** 2)

            # Add regularization to cost if specified
            if l2_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l2_penalty * np.sum(reg_weights ** 2) / 2

            if l1_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l1_penalty * np.sum(np.abs(reg_weights))

            # Check for convergence
            if abs(prev_cost - mse) < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                break

            prev_cost = mse

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Cost: {mse:.6f}")

        return self.weights


class Adam(BaseOptimizer):
    """Adam optimizer."""

    def __init__(self, learning_rate: float = 0.001, max_iter: int = 1000,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the optimizer.

        Parameters:
        -----------
        learning_rate : float, default=0.001
            The step size for parameter updates.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        random_state : int, optional
            Random seed for reproducibility.
        beta1 : float, default=0.9
            Exponential decay rate for the first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for the second moment estimates.
        epsilon : float, default=1e-8
            Small constant for numerical stability.
        """
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: np.ndarray,
            l1_penalty: float = 0.0, l2_penalty: float = 0.0,
            verbose: bool = False) -> np.ndarray:
        """
        Fit the model using Adam optimizer.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        l1_penalty : float, default=0.0
            L1 regularization parameter.
        l2_penalty : float, default=0.0
            L2 regularization parameter.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        np.ndarray
            Optimized weights.
        """
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = self._initialize_weights(n_features)

        # Initialize moment estimates
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)

        # For tracking convergence
        prev_cost = float('inf')

        for iteration in range(1, self.max_iter + 1):
            # Compute prediction
            y_pred = X_with_intercept @ self.weights

            # Compute gradient
            gradient = (1 / n_samples) * X_with_intercept.T @ (y_pred - y)

            # Add L2 regularization gradient if specified
            if l2_penalty > 0:
                # Don't regularize intercept
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0
                gradient += l2_penalty * reg_weights

            # Add L1 regularization gradient if specified
            if l1_penalty > 0:
                # Don't regularize intercept
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0
                # L1 gradient is sign of weights
                gradient += l1_penalty * np.sign(reg_weights)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * gradient
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** iteration)
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2 ** iteration)

            # Update weights
            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Compute current cost
            y_pred = X_with_intercept @ self.weights
            mse = np.mean((y_pred - y) ** 2)

            # Add regularization to cost if specified
            if l2_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l2_penalty * np.sum(reg_weights ** 2) / 2

            if l1_penalty > 0:
                reg_weights = np.copy(self.weights)
                reg_weights[0] = 0  # Don't regularize intercept
                mse += l1_penalty * np.sum(np.abs(reg_weights))

            # Check for convergence
            if abs(prev_cost - mse) < self.tol:
                if verbose:
                    print(f"Converged after {iteration} iterations.")
                break

            prev_cost = mse

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.max_iter}, Cost: {mse:.6f}")

        return self.weights
