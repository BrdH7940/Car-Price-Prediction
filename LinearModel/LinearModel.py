import numpy as np
import matplotlib.pyplot as plt
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
            'mae (price)': []
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
            'mae (price)': maee,
            'r2': r2
        }

    def _collect_metrics_during_training(self, X: np.ndarray, y: np.ndarray,
                                         X_val: Optional[np.ndarray] = None,
                                         y_val: Optional[np.ndarray] = None,
                                         verbose: bool = False) -> Dict[str, List[float]]:
        """
        Thu thập các metric trong quá trình huấn luyện mô hình.

        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu huấn luyện.
        y : np.ndarray
            Giá trị mục tiêu huấn luyện.
        X_val : np.ndarray, optional
            Dữ liệu validation.
        y_val : np.ndarray, optional
            Giá trị mục tiêu validation.
        verbose : bool, default=False
            Hiển thị tiến trình hay không.

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary chứa các metric theo từng iteration.
        """
        # Khởi tạo optimizer mới để bắt đầu huấn luyện
        self._init_optimizer()

        X_with_intercept = self.optimizer._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # Khởi tạo biến để theo dõi metrics
        history = {
            'train_mse': [],
            'train_rmse': [],
            'train_mae': [],
            'train_mae (price)': [],
            'train_r2': [],
            'iteration': []
        }

        if X_val is not None and y_val is not None:
            X_val_with_intercept = self.optimizer._add_intercept(X_val)
            history['val_mse'] = []
            history['val_rmse'] = []
            history['val_mae'] = []
            history['val_mae (price)'] = []
            history['val_r2'] = []

        # Khởi tạo weights
        if self.optimizer.weights is None:
            self.optimizer.weights = self.optimizer._initialize_weights(n_features)

        prev_cost = float('inf')

        # Xử lý theo từng loại optimizer
        if self.optimizer_name == 'normal':
            # Normal equation không cần theo dõi quá trình huấn luyện
            self.optimizer.fit(X, y, self.l1_penalty, self.l2_penalty)

            # Thu thập metric sau khi fit hoàn tất
            y_pred_train = X_with_intercept @ self.optimizer.weights
            train_metrics = self._evaluate(y, y_pred_train)

            for key, value in train_metrics.items():
                history[f'train_{key}'] = [value]

            history['iteration'] = [1]

            if X_val is not None and y_val is not None:
                y_pred_val = X_val_with_intercept @ self.optimizer.weights
                val_metrics = self._evaluate(y_val, y_pred_val)

                for key, value in val_metrics.items():
                    history[f'val_{key}'] = [value]

            return history

        elif self.optimizer_name == 'gd':
            # Gradient Descent
            for iteration in range(1, self.optimizer.max_iter + 1):
                # Compute predictions
                y_pred = X_with_intercept @ self.optimizer.weights

                # Compute gradient of MSE
                gradient = (1 / n_samples) * X_with_intercept.T @ (y_pred - y)

                # Add L2 regularization gradient if specified
                if self.l2_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.optimizer.weights)
                    reg_weights[0] = 0
                    gradient += self.l2_penalty * reg_weights

                # Add L1 regularization gradient if specified
                if self.l1_penalty > 0:
                    # Don't regularize intercept
                    reg_weights = np.copy(self.optimizer.weights)
                    reg_weights[0] = 0
                    gradient += self.l1_penalty * np.sign(reg_weights)

                # Update weights
                self.optimizer.weights -= self.optimizer.learning_rate * gradient

                # Compute current cost
                y_pred_train = X_with_intercept @ self.optimizer.weights
                train_metrics = self._evaluate(y, y_pred_train)

                # Collect metrics every 10 iterations or on the first/last iteration
                if iteration % 10 == 0 or iteration == 1 or iteration == self.optimizer.max_iter:
                    history['iteration'].append(iteration)

                    for key, value in train_metrics.items():
                        history[f'train_{key}'].append(value)

                    if X_val is not None and y_val is not None:
                        y_pred_val = X_val_with_intercept @ self.optimizer.weights
                        val_metrics = self._evaluate(y_val, y_pred_val)

                        for key, value in val_metrics.items():
                            history[f'val_{key}'].append(value)

                # Check for convergence
                current_mse = train_metrics['mse']
                if abs(prev_cost - current_mse) < self.optimizer.tol:
                    if verbose:
                        print(f"Converged after {iteration} iterations.")
                    break

                prev_cost = current_mse

                # Print progress if verbose
                if verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration}/{self.optimizer.max_iter}, MSE: {current_mse:.6f}")

        elif self.optimizer_name in ['sgd', 'mini_batch_gd']:
            # Stochastic GD or Mini-batch GD
            indices = np.arange(n_samples)

            for iteration in range(1, self.optimizer.max_iter + 1):
                # Shuffle the data
                np.random.shuffle(indices)
                X_shuffled = X_with_intercept[indices]
                y_shuffled = y[indices]

                if self.optimizer_name == 'sgd':
                    # Process each sample for SGD
                    for i in range(n_samples):
                        x_i = X_shuffled[i:i + 1]
                        y_i = y_shuffled[i:i + 1]

                        # Compute prediction and gradient
                        y_pred = x_i @ self.optimizer.weights
                        gradient = x_i.T @ (y_pred - y_i)

                        # Add regularization
                        if self.l2_penalty > 0:
                            reg_weights = np.copy(self.optimizer.weights)
                            reg_weights[0] = 0  # Don't regularize intercept
                            gradient += self.l2_penalty * reg_weights

                        if self.l1_penalty > 0:
                            reg_weights = np.copy(self.optimizer.weights)
                            reg_weights[0] = 0
                            gradient += self.l1_penalty * np.sign(reg_weights)

                        # Update weights
                        self.optimizer.weights -= self.optimizer.learning_rate * gradient

                else:  # mini_batch_gd
                    # Process mini-batches
                    for i in range(0, n_samples, self.optimizer.batch_size):
                        X_batch = X_shuffled[i:i + self.optimizer.batch_size]
                        y_batch = y_shuffled[i:i + self.optimizer.batch_size]

                        # Compute prediction
                        y_pred = X_batch @ self.optimizer.weights

                        # Compute gradient
                        gradient = (1 / len(X_batch)) * X_batch.T @ (y_pred - y_batch)

                        # Add regularization
                        if self.l2_penalty > 0:
                            reg_weights = np.copy(self.optimizer.weights)
                            reg_weights[0] = 0
                            gradient += self.l2_penalty * reg_weights

                        if self.l1_penalty > 0:
                            reg_weights = np.copy(self.optimizer.weights)
                            reg_weights[0] = 0
                            gradient += self.l1_penalty * np.sign(reg_weights)

                        # Update weights
                        self.optimizer.weights -= self.optimizer.learning_rate * gradient

                # Compute current metrics on full dataset
                y_pred_train = X_with_intercept @ self.optimizer.weights
                train_metrics = self._evaluate(y, y_pred_train)

                # Collect metrics periodically
                if iteration % 10 == 0 or iteration == 1 or iteration == self.optimizer.max_iter:
                    history['iteration'].append(iteration)

                    for key, value in train_metrics.items():
                        history[f'train_{key}'].append(value)

                    if X_val is not None and y_val is not None:
                        y_pred_val = X_val_with_intercept @ self.optimizer.weights
                        val_metrics = self._evaluate(y_val, y_pred_val)

                        for key, value in val_metrics.items():
                            history[f'val_{key}'].append(value)

                # Check for convergence
                current_mse = train_metrics['mse']
                if abs(prev_cost - current_mse) < self.optimizer.tol:
                    if verbose:
                        print(f"Converged after {iteration} iterations.")
                    break

                prev_cost = current_mse

                # Print progress if verbose
                if verbose and iteration % 20 == 0:
                    print(f"Iteration {iteration}/{self.optimizer.max_iter}, MSE: {current_mse:.6f}")

        elif self.optimizer_name == 'adam':
            # Adam optimizer
            m = np.zeros_like(self.optimizer.weights)
            v = np.zeros_like(self.optimizer.weights)

            for iteration in range(1, self.optimizer.max_iter + 1):
                # Compute prediction
                y_pred = X_with_intercept @ self.optimizer.weights

                # Compute gradient
                gradient = (1 / n_samples) * X_with_intercept.T @ (y_pred - y)

                # Add regularization
                if self.l2_penalty > 0:
                    reg_weights = np.copy(self.optimizer.weights)
                    reg_weights[0] = 0
                    gradient += self.l2_penalty * reg_weights

                if self.l1_penalty > 0:
                    reg_weights = np.copy(self.optimizer.weights)
                    reg_weights[0] = 0
                    gradient += self.l1_penalty * np.sign(reg_weights)

                # Update biased first moment estimate
                m = self.optimizer.beta1 * m + (1 - self.optimizer.beta1) * gradient
                # Update biased second raw moment estimate
                v = self.optimizer.beta2 * v + (1 - self.optimizer.beta2) * (gradient ** 2)

                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.optimizer.beta1 ** iteration)
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - self.optimizer.beta2 ** iteration)

                # Update weights
                self.optimizer.weights -= self.optimizer.learning_rate * m_hat / (
                            np.sqrt(v_hat) + self.optimizer.epsilon)

                # Compute current metrics
                y_pred_train = X_with_intercept @ self.optimizer.weights
                train_metrics = self._evaluate(y, y_pred_train)

                # Collect metrics periodically
                if iteration % 10 == 0 or iteration == 1 or iteration == self.optimizer.max_iter:
                    history['iteration'].append(iteration)

                    for key, value in train_metrics.items():
                        history[f'train_{key}'].append(value)

                    if X_val is not None and y_val is not None:
                        y_pred_val = X_val_with_intercept @ self.optimizer.weights
                        val_metrics = self._evaluate(y_val, y_pred_val)

                        for key, value in val_metrics.items():
                            history[f'val_{key}'].append(value)

                # Check for convergence
                current_mse = train_metrics['mse']
                if abs(prev_cost - current_mse) < self.optimizer.tol:
                    if verbose:
                        print(f"Converged after {iteration} iterations.")
                    break

                prev_cost = current_mse

                # Print progress if verbose
                if verbose and iteration % 20 == 0:
                    print(f"Iteration {iteration}/{self.optimizer.max_iter}, MSE: {current_mse:.6f}")

        return history

    def plot_learning_curve(self, X: np.ndarray, y: np.ndarray,
                            val_size: float = 0.2, random_state: Optional[int] = None,
                            metrics: List[str] = ['mse', 'mae'],
                            figsize: Tuple[int, int] = (15, 10),
                            verbose: bool = False) -> Dict[str, List[float]]:
        """
        Vẽ biểu đồ learning curve cho các metric trên tập train và validation.

        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu huấn luyện.
        y : np.ndarray
            Giá trị mục tiêu.
        val_size : float, default=0.2
            Tỷ lệ dữ liệu dành cho validation.
        random_state : int, optional
            Seed cho việc chia dữ liệu.
        metrics : List[str], default=['mse', 'mae']
            Danh sách các metric cần vẽ.
        figsize : Tuple[int, int], default=(15, 10)
            Kích thước của biểu đồ.
        verbose : bool, default=False
            Hiển thị tiến trình hay không.

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary chứa lịch sử các metric.
        """
        # Khởi tạo random_state nếu được cung cấp
        if random_state is not None:
            np.random.seed(random_state)

        # Chia dữ liệu thành tập train và validation
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        val_samples = int(n_samples * val_size)

        X_train = X[indices[val_samples:]]
        y_train = y[indices[val_samples:]]
        X_val = X[indices[:val_samples]]
        y_val = y[indices[:val_samples]]

        if verbose:
            print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")

        # Thu thập metrics trong quá trình huấn luyện
        history = self._collect_metrics_during_training(X_train, y_train, X_val, y_val, verbose)

        # Vẽ biểu đồ learning curve
        if not metrics:
            metrics = ['mse', 'mae']

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        # Xử lý trường hợp chỉ có 1 metric
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            train_metric_key = f'train_{metric}'
            val_metric_key = f'val_{metric}'

            ax.plot(history['iteration'], history[train_metric_key], 'b-', label=f'Training {metric.upper()}')

            if val_metric_key in history:
                ax.plot(history['iteration'], history[val_metric_key], 'r-', label=f'Validation {metric.upper()}')

            ax.set_title(f'Learning Curve - {metric.upper()}')
            ax.set_xlabel('Iterations')
            ax.set_ylabel(metric.upper())
            ax.legend()

            # Set logy scale for better visualization
            if any(x > 0 for x in history[train_metric_key]):
                ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

        return history

    def plot_cross_validation_metrics(self, X: np.ndarray, y: np.ndarray,
                                      n_folds: int = 5, shuffle: bool = True,
                                      metrics: List[str] = ['mse', 'mae', 'r2'],
                                      figsize: Tuple[int, int] = (15, 5),
                                      verbose: bool = False) -> Dict[str, List[float]]:
        """
        Vẽ biểu đồ so sánh các metric trên các fold của cross-validation.

        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu huấn luyện.
        y : np.ndarray
            Giá trị mục tiêu.
        n_folds : int, default=5
            Số lượng fold cho cross-validation.
        shuffle : bool, default=True
            Xáo trộn dữ liệu trước khi chia fold.
        metrics : List[str], default=['mse', 'mae', 'r2']
            Danh sách các metric cần vẽ.
        figsize : Tuple[int, int], default=(15, 5)
            Kích thước của biểu đồ.
        verbose : bool, default=False
            Hiển thị tiến trình hay không.

        Returns:
        --------
        Dict[str, List[float]]
            Kết quả cross-validation.
        """
        # Thực hiện cross-validation
        cv_results = self.cross_validate(X, y, n_folds, shuffle, verbose)

        # Vẽ biểu đồ các metric
        if not metrics:
            metrics = ['mse', 'mae', 'r2']

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        # Xử lý trường hợp chỉ có 1 metric
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            if metric in cv_results:
                # Vẽ biểu đồ boxplot cho metric
                ax.boxplot(cv_results[metric])

                # Thêm các điểm dữ liệu để thấy rõ hơn phân phối
                x = np.random.normal(1, 0.04, size=len(cv_results[metric]))
                ax.plot(x, cv_results[metric], 'r.', alpha=0.5)

                # Thêm giá trị trung bình
                avg_metric = np.mean(cv_results[metric])
                ax.axhline(y=avg_metric, color='b', linestyle='--',
                           label=f'Avg: {avg_metric:.4f}')

                ax.set_title(f'{metric.upper()} across {n_folds} folds')
                ax.set_ylabel(metric.upper())
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f"Metric '{metric}' not available",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

        plt.tight_layout()
        plt.show()

        return cv_results

    def compare_optimizers(self, X: np.ndarray, y: np.ndarray,
                           optimizers: List[str] = ['normal', 'gd', 'sgd', 'mini_batch_gd', 'adam'],
                           metrics: List[str] = ['mse', 'mae', 'r2'],
                           val_size: float = 0.2,
                           random_state: Optional[int] = None,
                           figsize: Tuple[int, int] = (15, 10),
                           verbose: bool = False) -> Dict[str, Dict[str, List[float]]]:
        """
        So sánh hiệu suất của các loại optimizer khác nhau.

        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu huấn luyện.
        y : np.ndarray
            Giá trị mục tiêu.
        optimizers : List[str], default=['normal', 'gd', 'sgd', 'mini_batch_gd', 'adam']
            Danh sách các optimizer cần so sánh.
        metrics : List[str], default=['mse', 'mae', 'r2']
            Danh sách các metric cần vẽ.
        val_size : float, default=0.2
            Tỷ lệ dữ liệu dành cho validation.
        random_state : int, optional
            Seed cho việc chia dữ liệu.
        figsize : Tuple[int, int], default=(15, 10)
            Kích thước của biểu đồ.
        verbose : bool, default=False
            Hiển thị tiến trình hay không.

        Returns:
        --------
        Dict[str, Dict[str, List[float]]]
            Dictionary chứa kết quả của từng optimizer.
        """
        # Lưu lại optimizer hiện tại
        current_optimizer = self.optimizer_name

        # Chia dữ liệu thành tập train và validation
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        val_samples = int(n_samples * val_size)

        X_train = X[indices[val_samples:]]
        y_train = y[indices[val_samples:]]
        X_val = X[indices[:val_samples]]
        y_val = y[indices[:val_samples]]

        if verbose:
            print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")

        # Dictionary để lưu kết quả
        results = {}

        # Đánh giá từng optimizer
        for opt in optimizers:
            if verbose:
                print(f"\nEvaluating {opt} optimizer...")

            # Đặt optimizer hiện tại
            self.optimizer_name = opt
            self._init_optimizer()

            # Huấn luyện mô hình
            self.fit(X_train, y_train, verbose=verbose)

            # Đánh giá trên tập train
            y_pred_train = self.predict(X_train)
            train_metrics = self._evaluate(y_train, y_pred_train)

            # Đánh giá trên tập validation
            y_pred_val = self.predict(X_val)
            val_metrics = self._evaluate(y_val, y_pred_val)

            # Lưu kết quả
            results[opt] = {
                'train': train_metrics,
                'validation': val_metrics
            }

            if verbose:
                print(f"  Train metrics: {train_metrics}")
                print(f"  Validation metrics: {val_metrics}")

        # Khôi phục optimizer ban đầu
        self.optimizer_name = current_optimizer
        self._init_optimizer()

        # Vẽ biểu đồ so sánh
        if not metrics:
            metrics = ['mse', 'mae', 'r2']

        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, n_metrics, figsize=figsize)

        # Tạo danh sách dữ liệu cho mỗi metric
        for i, metric in enumerate(metrics):
            # Dữ liệu cho tập train
            train_values = [results[opt]['train'][metric] for opt in optimizers]
            axes[0, i].bar(optimizers, train_values, color='b', alpha=0.7)
            axes[0, i].set_title(f'Training {metric.upper()}')
            axes[0, i].set_ylabel(metric.upper())
            axes[0, i].set_xticklabels(optimizers, rotation=45)

            # Thêm giá trị lên các cột
            for j, v in enumerate(train_values):
                axes[0, i].text(j, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

            # Dữ liệu cho tập validation
            val_values = [results[opt]['validation'][metric] for opt in optimizers]
            axes[1, i].bar(optimizers, val_values, color='r', alpha=0.7)
            axes[1, i].set_title(f'Validation {metric.upper()}')
            axes[1, i].set_ylabel(metric.upper())
            axes[1, i].set_xticklabels(optimizers, rotation=45)

            # Thêm giá trị lên các cột
            for j, v in enumerate(val_values):
                axes[1, i].text(j, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        return results