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

    def fit(self, X, y, verbose=False):
        """
        Huấn luyện mô hình Linear Regression

        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Giá trị mục tiêu
        verbose : bool
            In thông tin chi tiết trong quá trình huấn luyện

        Returns:
        --------
        self : object
            Returns self.
        """
        # Chuyển đổi dữ liệu thành numpy array nếu cần
        X = np.array(X)
        y = np.array(y)

        # Kiểm tra shape
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Khởi tạo parameters
        n_features = X.shape[1]
        self.weights, self.bias = self.optimizer.initialize(n_features)

        # Huấn luyện với optimizer
        self.weights, self.bias, self.cost_history, self.iteration_history = self.optimizer.optimize(
            X, y, self.weights, self.bias, self.n_iterations,
            self.regularization, self.lambda_param,
            self.tol, self.max_iter, verbose
        )

        self.is_fitted = True
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
