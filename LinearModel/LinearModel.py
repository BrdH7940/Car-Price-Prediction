import numpy as np
from typing import Optional, Union, Callable, List, Tuple, Dict
from LinearModel.Optimizer import *



class LinearRegression:
    """Linear Regression model with various optimization methods."""

    def __init__(self, optimizer: str = 'normal', learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-4, random_state: Optional[int] = None,
                 l1_penalty: float = 0.0, l2_penalty: float = 0.0, batch_size: int = 32,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Linear Regression model.

        Parameters:
        -----------
        optimizer : str, default='normal'
            Optimization method. Options: 'normal', 'gd', 'sgd', 'mini_batch_gd', 'adam'.
        learning_rate : float, default=0.01
            The step size for parameter updates.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        random_state : int, optional
            Random seed for reproducibility.
        l1_penalty : float, default=0.0
            L1 regularization parameter (LASSO).
        l2_penalty : float, default=0.0
            L2 regularization parameter (Ridge).
        batch_size : int, default=32
            Size of mini-batches for mini-batch gradient descent.
        beta1 : float, default=0.9
            Exponential decay rate for the first moment estimates in Adam.
        beta2 : float, default=0.999
            Exponential decay rate for the second moment estimates in Adam.
        epsilon : float, default=1e-8
            Small constant for numerical stability in Adam.
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize optimizer
        self._init_optimizer()

    def _init_optimizer(self):
        """Initialize the optimizer based on the specified type."""
        if self.optimizer_name == 'normal':
            self.optimizer = BaseOptimizer(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
        elif self.optimizer_name == 'gd':
            self.optimizer = GradientDescent(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
        elif self.optimizer_name == 'sgd':
            self.optimizer = StochasticGradientDescent(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
        elif self.optimizer_name == 'mini_batch_gd':
            self.optimizer = MiniBatchGradientDescent(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                batch_size=self.batch_size
            )
        elif self.optimizer_name == 'adam':
            self.optimizer = Adam(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LinearRegression':
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        self : LinearRegression
            Fitted model.
        """
        if verbose:
            print(f"Fitting model with {self.optimizer_name} optimizer...")

        if self.optimizer_name in ['gd', 'sgd', 'mini_batch_gd', 'adam']:
            self.optimizer.fit(X, y, self.l1_penalty, self.l2_penalty, verbose)
        else:
            self.optimizer.fit(X, y, self.l1_penalty, self.l2_penalty)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X : np.ndarray
            Input features.

        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        return self.optimizer.predict(X)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                       shuffle: bool = True, verbose: bool = False) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.

        Parameters:
        -----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        n_folds : int, default=5
            Number of folds.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        verbose : bool, default=False
            Whether to print progress.

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary containing lists of evaluation metrics for each fold.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Create folds
        fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
        fold_sizes[:n_samples % n_folds] += 1
        fold_indices = []

        current = 0
        for fold_size in fold_sizes:
            fold_indices.append(indices[current:current + fold_size])
            current += fold_size

        # Initialize metrics
        metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'mae original': []
        }

        # Perform cross-validation
        for i, test_indices in enumerate(fold_indices):
            if verbose:
                print(f"\nFold {i + 1}/{n_folds}")

            # Split data
            train_indices = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Train model
            # Reset optimizer to ensure fresh training
            self._init_optimizer()
            self.fit(X_train, y_train, verbose)

            # Evaluate model
            y_pred = self.predict(X_test)
            fold_metrics = self._evaluate(y_test, y_pred)

            for metric, value in fold_metrics.items():
                metrics[metric].append(value)

            if verbose:
                print(f"Fold {i + 1} Metrics:")
                for metric, value in fold_metrics.items():
                    print(f"  {metric}: {value:.4f}")

        # Calculate average metrics
        avg_metrics = {f"avg_{k}": np.mean(v) for k, v in metrics.items()}
        std_metrics = {f"std_{k}": np.std(v) for k, v in metrics.items()}

        if verbose:
            print("\nCross-validation Results:")
            for metric, value in avg_metrics.items():
                std = std_metrics[f"std_{metric[4:]}"]
                print(f"  {metric}: {value:.4f} (±{std:.4f})")

        # Combine all metrics
        all_metrics = {**metrics, **avg_metrics, **std_metrics}

        return all_metrics

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model using various metrics.

        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.

        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        maee = np.mean(np.abs(np.exp(y_true) - np.exp(y_pred)))

        # R² Score
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_residual / ss_total if ss_total != 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mae original': maee,
            'r2': r2
        }