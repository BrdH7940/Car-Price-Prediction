import numpy as np
from LinearModel.Optimizer import *
from Utils.Visualization import plot_learning_curve
from Utils.Metrics import *

class LinearRegression():
    def __init__(self, optimizer='gradient_descent', learning_rate=1e-8,
                 n_iterations=1000, regularization=None, lambda_param=0.01,
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
                self.optimizer = StochasticGradientDescent(learning_rate, random_state)
            elif optimizer == 'mini_batch_gd':
                self.optimizer = MiniBatchGradientDescent(learning_rate, batch_size, random_state)
            elif optimizer == 'normal_equation':
                self.optimizer = NormalEquation(random_state=random_state)
            elif optimizer == 'adam':
                self.optimizer = AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, 
                                               epsilon=1e-8, random_state=random_state)
            else:
                raise ValueError(
                    "Unknown optimizer. Please choose one of: 'gradient_descent', 'sgd', 'mini_batch_gd', or 'normal_equation'.")
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
        print("Initial weights:", self.weights)

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

    def evaluate(self, X, y):
        """Đánh giá mô hình trên tập test"""
        y_pred = self.predict(X)
        
        metrics = {
            'MSE': mean_square_error(y, y_pred),
            'RMSE': root_mean_square_error(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred),
            'R²': r2_score(y, y_pred),
            "MAE exp": np.mean(np.abs(np.exp(y) - np.exp(y_pred)))
        }

        # In ra các metrix
        print("Kết quả đánh giá:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

        return metrics

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
        
    def score(self, X, y):
        """
        Calculate the Coefficient of Determination (R^2) score for the model.

        Parameters:
        -----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Test samples. Features should match the training data.
        y : numpy.ndarray, shape (n_samples,)
            True target values corresponding to X.

        Returns:
        --------
        score : float
            The R^2 score. Ranges from 1 (perfect fit) down to negative values.
            A score of 0 means the model performs no better than predicting the mean.
        """
        # Ensure y is a numpy array
        y = np.array(y)
        if y.ndim != 1:
             y = y.squeeze() # Ensure y is 1D
             if y.ndim != 1:
                 raise ValueError("Target y must be a 1D array or squeezable to 1D.")

        # 1. Get predictions for the input X using the trained model
        y_pred = self.predict(X) # Uses the predict method defined above

        # Ensure y_pred is also 1D for calculations
        y_pred = y_pred.squeeze()

        # Check if shapes match after prediction and squeezing
        if y.shape != y_pred.shape:
             raise ValueError(f"Shape mismatch between true y ({y.shape}) and predicted y ({y_pred.shape}) after prediction.")


        # 2. Calculate the R^2 score
        # Sum of squared residuals (errors)
        ss_res = np.sum((y - y_pred) ** 2)

        # Total sum of squares (variance in y)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # 3. Handle the edge case where ss_tot is zero (y is constant)
        if ss_tot == 0:
            # If ss_res is also ~0, it means y_pred perfectly predicted the constant y
            # Otherwise, R^2 is undefined. Conventionally return 0 or 1.
            # Scikit-learn returns 1.0 if ss_res is also near zero, else 0.0.
            # Let's follow that convention:
            return 1.0 if np.isclose(ss_res, 0) else 0.0

        # 4. Calculate R^2
        r2_score = 1 - (ss_res / ss_tot)

        return r2_score

    def cross_validate(self, X, y, k_folds=5, random_state=None):
        """
        Thực hiện cross-validation để đánh giá mô hình
        
        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Giá trị mục tiêu
        k_folds : int
            Số lượng folds
        random_state : int hoặc None
            Seed cho random generator
        
        Returns:
        --------
        mean_score : float
            Điểm trung bình qua các folds
        scores : list
            Danh sách điểm của từng fold
        """
        # Chuyển đổi dữ liệu thành numpy array nếu cần
        X = np.array(X)
        y = np.array(y)
        
        # Thiết lập random seed nếu cần
        if random_state is not None:
            np.random.seed(random_state)
        
        # Số lượng mẫu
        n_samples = X.shape[0]
        
        # Tạo indices và shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Chia thành k_folds
        fold_size = n_samples // k_folds
        scores = []
        
        for i in range(k_folds):
            # Xác định indices cho tập test
            start = i * fold_size
            end = start + fold_size if i < k_folds - 1 else n_samples
            test_indices = indices[start:end]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Chia dữ liệu
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
            
            # Tạo một bản sao của optimizer hiện tại với cùng tham số
            if isinstance(self.optimizer, GradientDescent):
                optimizer = GradientDescent(
                    learning_rate=self.optimizer.learning_rate,
                    random_state=random_state
                )
            elif isinstance(self.optimizer, StochasticGradientDescent):
                optimizer = StochasticGradientDescent(
                    learning_rate=self.optimizer.learning_rate,
                    random_state=random_state
                )
            elif isinstance(self.optimizer, MiniBatchGradientDescent):
                optimizer = MiniBatchGradientDescent(
                    learning_rate=self.optimizer.learning_rate,
                    batch_size=self.optimizer.batch_size,
                    random_state=random_state
                )
            elif isinstance(self.optimizer, AdamOptimizer):
                optimizer = AdamOptimizer(
                    learning_rate=self.optimizer.learning_rate,
                    beta1=self.optimizer.beta1,
                    beta2=self.optimizer.beta2,
                    epsilon=self.optimizer.epsilon,
                    random_state=random_state
                )
            elif isinstance(self.optimizer, NormalEquation):
                optimizer = NormalEquation(
                    random_state=random_state
                )
            else:
                # Nếu là optimizer không xác định, sử dụng GradientDescent mặc định
                optimizer = GradientDescent(
                    learning_rate=0.01,
                    random_state=random_state
                )
            
            # Huấn luyện mô hình với optimizer mới
            model = LinearRegression(
                optimizer=optimizer,
                n_iterations=self.n_iterations,
                regularization=self.regularization,
                lambda_param=self.lambda_param,
                random_state=random_state,
                tol=self.tol,
                max_iter=self.max_iter
            )
            model.fit(X_train, y_train)
            
            # Tính điểm
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return np.mean(scores), scores