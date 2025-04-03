# models/linear_regression.py
import numpy as np
from .optimizer import *
from utils.visualization import plot_learning_curve


class LinearRegression():
    def __init__(self, optimizer='gradient_descent', learning_rate=1e-8,
                 n_iterations=1000, regularization=None, lambda_param=0.1,
                 batch_size=None, random_state=None, tol=1e-6, max_iter=None):
        """
        Khởi tạo mô hình Linear Regression

        Parameters:
        -----------
        optimizer : str hoặc BaseOptimizer
            Thuật toán tối ưu ('gradient_descent', 'sgd', 'mini_batch_gd', 'normal_equation')
            hoặc một đối tượng optimizer tùy chỉnh
        learning_rate : float
            Tốc độ học cho gradient descent
        n_iterations : int
            Số lần lặp tối đa cho gradient descent
        regularization : str, optional ('l1', 'l2', None)
            Phương pháp regularization
        lambda_param : float
            Tham số regularization
        batch_size : int or None
            Kích thước batch cho mini-batch gradient descent
        random_state : int or None
            Seed cho random generator
        tol : float
            Điều kiện dừng khi thay đổi cost function nhỏ hơn tol
        max_iter : int or None
            Giới hạn số lần lặp không cải thiện
        """
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

        # Khởi tạo optimizer
        if isinstance(optimizer, str):
            if optimizer == 'gradient_descent':
                self.optimizer = GradientDescent(learning_rate, random_state)
            elif optimizer == 'sgd':
                self.optimizer = StochasticGradientDescent(
                    learning_rate, random_state)
            elif optimizer == 'mini_batch_gd':
                self.optimizer = MiniBatchGradientDescent(
                    learning_rate, batch_size, random_state)
            elif optimizer == 'normal_equation':
                self.optimizer = NormalEquation(random_state=random_state)
            else:
                raise ValueError(
                    "Unknown optimizer. Use 'gradient_descent', 'sgd', 'mini_batch_gd', or 'normal_equation'.")
        else:
            # Giả sử người dùng truyền vào một đối tượng optimizer tùy chỉnh
            self.optimizer = optimizer

        self.weights = None
        self.bias = None
        self.cost_history = []
        self.iteration_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LinearRegression':
        """
        Train (fit) the Linear Regression model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            Input features (training data), shape (n_samples, n_features).
        y : np.ndarray
            Target values (training labels), shape (n_samples,) or (n_samples, 1).
        verbose : bool, optional (default=False)
            If True, prints progress information during optimization.

        Returns:
        --------
        self : LinearRegression
            The fitted model instance.

        Raises:
        -------
        ValueError:
            If input shapes are incompatible.
        """
        # Ensure data are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Input validation and reshaping
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1) # Ensure y is a column vector

        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError(f"Incompatible shapes: X has {n_samples} samples, "
                            f"but y has {y.shape[0]} samples.")
        if y.shape[1] != 1:
            # Ensure y is a column vector after potential reshaping
            if y.ndim == 2 and y.shape[1] > 1:
                raise ValueError(f"Target y should be a column vector, but got shape {y.shape}")
            # If y was originally 1D, the earlier reshape handled it.
            # If y was somehow passed as (1, n_samples), we might need error handling or transpose.
            # Assuming standard (n_samples,) or (n_samples, 1) input is expected.


        # Initialize parameters using the optimizer's method
        self.weights, self.bias = self.optimizer.initialize(n_features)

        # Perform optimization
        # Type hint for optimizer expected return value (if possible)
        optim_result: tuple[np.ndarray, float, list[float], list[int]] = \
            self.optimizer.optimize(
                X, y, self.weights, self.bias, self.n_iterations,
                self.regularization, self.lambda_param,
                self.tol, self.max_iter, verbose
            )

        self.weights, self.bias, self.cost_history, self.iteration_history = optim_result

        self.is_fitted = True  # Mark model as fitted
        return self

    def predict(self, X):
        """
        Dự đoán giá trị cho dữ liệu mới

        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào

        Returns:
        --------
        y_pred : numpy.ndarray
            Giá trị dự đoán
        """
        # Kiểm tra đã fit hay chưa
        if not self.is_fitted:
            raise Exception("Model must be fitted before predicting")

        # Chuyển đổi dữ liệu thành numpy array nếu cần
        X = np.array(X)

        # Kiểm tra shape
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Tính dự đoán
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        """Trả về parameters của mô hình"""
        return {
            'weights': self.weights,
            'bias': self.bias
        }

    # Các phương thức khác giữ nguyên...

    def plot_learning_curve(self):
        """Vẽ learning curve của mô hình"""
        if not self.is_fitted or len(self.cost_history) <= 1:
            raise Exception(
                "Model must be fitted with iterative method to plot learning curve")

        plot_learning_curve(self.iteration_history, self.cost_history,
                            title=f'Learning Curve - {self.method}')
