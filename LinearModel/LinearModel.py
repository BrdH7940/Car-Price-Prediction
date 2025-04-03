import numpy as np

import numpy as np


class StandardScaler:
    """
    Chuẩn hóa dữ liệu bằng cách loại bỏ giá trị trung bình và chia cho độ lệch chuẩn.

    Công thức: z = (x - mean) / std

    Parameters
    ----------
    with_mean : bool, default=True
        Nếu True, trừ giá trị trung bình khỏi dữ liệu.

    with_std : bool, default=True
        Nếu True, chia dữ liệu cho độ lệch chuẩn.

    Attributes
    ----------
    mean_ : array, shape (n_features,)
        Giá trị trung bình được tính từ dữ liệu huấn luyện.

    var_ : array, shape (n_features,)
        Phương sai được tính từ dữ liệu huấn luyện.

    scale_ : array, shape (n_features,)
        Độ lệch chuẩn được tính từ dữ liệu huấn luyện.

    n_features_in_ : int
        Số đặc trưng được thấy trong quá trình huấn luyện.

    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Tính toán giá trị trung bình và độ lệch chuẩn.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào mà sẽ được chuẩn hóa.

        y : Ignored
            Không sử dụng, tồn tại để tương thích API.

        Returns
        -------
        self : object
            Returns self.
        """

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Lưu số đặc trưng
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1

        # Tính giá trị trung bình
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_in_)

        # Tính phương sai và độ lệch chuẩn
        if self.with_std:
            self.var_ = np.var(X, axis=0)
            # Xử lý trường hợp phương sai bằng 0
            self.scale_ = np.sqrt(self.var_)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.var_ = np.ones(self.n_features_in_)
            self.scale_ = np.ones(self.n_features_in_)

        return self

    def transform(self, X):
        """
        Thực hiện chuẩn hóa dữ liệu.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào sẽ được chuẩn hóa.

        Returns
        -------
        X_scaled : array, shape (n_samples, n_features)
            Dữ liệu đã được chuẩn hóa.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler chưa được huấn luyện. "
                             "Hãy gọi fit() trước transform().")

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Kiểm tra số đặc trưng
        if X.ndim > 1 and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Số đặc trưng trong dữ liệu ({X.shape[1]}) "
                             f"không trùng khớp với số đặc trưng đã được học ({self.n_features_in_})")

        # Thực hiện chuẩn hóa
        X_scaled = X.copy()

        if self.with_mean:
            X_scaled -= self.mean_

        if self.with_std:
            X_scaled /= self.scale_

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit và transform cùng một lúc.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào mà sẽ được chuẩn hóa.

        y : Ignored
            Không sử dụng, tồn tại để tương thích API.

        Returns
        -------
        X_scaled : array, shape (n_samples, n_features)
            Dữ liệu đã được chuẩn hóa.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Chuyển dữ liệu đã chuẩn hóa về dạng ban đầu.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đã chuẩn hóa.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Dữ liệu sau khi được chuyển về dạng ban đầu.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler chưa được huấn luyện. "
                             "Hãy gọi fit() trước inverse_transform().")

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Kiểm tra số đặc trưng
        if X.ndim > 1 and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Số đặc trưng trong dữ liệu ({X.shape[1]}) "
                             f"không trùng khớp với số đặc trưng đã được học ({self.n_features_in_})")

        # Thực hiện biến đổi ngược
        X_original = X.copy()

        if self.with_std:
            X_original *= self.scale_

        if self.with_mean:
            X_original += self.mean_

        return X_original


# Ví dụ sử dụng
# if __name__ == "__main__":
#     # Tạo dữ liệu ví dụ
#     np.random.seed(42)
#     X = np.random.randn(10, 3) * np.array([10, 5, 2]) + np.array([5, -3, 0])
#
#     # In dữ liệu gốc
#     print("Dữ liệu gốc:")
#     print(X)
#     print("\nMean trước khi chuẩn hóa:", np.mean(X, axis=0))
#     print("Std trước khi chuẩn hóa:", np.std(X, axis=0))
#
#     # Sử dụng StandardScaler
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # In dữ liệu đã chuẩn hóa
#     print("\nDữ liệu đã chuẩn hóa:")
#     print(X_scaled)
#     print("\nMean sau khi chuẩn hóa:", np.mean(X_scaled, axis=0))
#     print("Std sau khi chuẩn hóa:", np.std(X_scaled, axis=0))
#
#     # Thử nghiệm inverse transform
#     X_inverse = scaler.inverse_transform(X_scaled)
#     print("\nDữ liệu sau khi inverse transform:")
#     print(X_inverse)
#
#     # Kiểm tra các thuộc tính của scaler
#     print("\nCác thuộc tính của scaler:")
#     print("mean_:", scaler.mean_)
#     print("var_:", scaler.var_)
#     print("scale_:", scaler.scale_)

class BaseOptimizer:
    """
    Lớp cơ sở cho các thuật toán tối ưu
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def normal_equation(self, X, y):
        """Giải phương trình tuyến tính bằng Normal Equation: w = (X^T X)^(-1) X^T y"""
        A = X.T.dot(X)
        b = X.T.dot(y)
        return np.linalg.solve(A, b)
        # return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def initialize_weights(self, n_features):
        """Khởi tạo trọng số sử dụng phương pháp Xavier"""
        # Xavier initialization: variance ~ 1/n_in
        limit = np.sqrt(6 / (n_features + 1))
        self.weights = np.random.randn(n_features) * limit

    def optimize(self, X, y, verbose=False):
        """Phương thức tối ưu trọng số (sẽ được cài đặt trong các lớp con)"""
        raise NotImplementedError("Phương thức này cần được cài đặt trong lớp con")


class GradientDescent(BaseOptimizer):
    """Thuật toán Gradient Descent"""

    def compute_gradient(self, X, y, y_pred):
        """Tính gradient của hàm mất mát"""
        m = len(y)
        return (1 / m) * X.T.dot(y_pred - y)

    def optimize(self, X, y, verbose=False):
        """Tối ưu bằng phương pháp Gradient Descent"""
        m, n = X.shape
        if self.weights is None:
            self.initialize_weights(n)

        prev_cost = float('inf')

        for i in range(self.max_iter):
            # Dự đoán
            y_pred = X.dot(self.weights)

            # Tính hàm mất mát (MSE)
            # print(y_pred[:10], y[:10])
            cost = np.mean((y_pred - y) ** 2)

            # Kiểm tra điều kiện dừng
            if np.abs(prev_cost - cost) < self.tol:
                if verbose:
                    print(f"Hội tụ tại vòng lặp {i}")
                break

            # Tính gradient và cập nhật trọng số
            gradient = self.compute_gradient(X, y, y_pred)
            self.weights -= self.learning_rate * gradient

            prev_cost = cost

            if verbose and (i % 100 == 0):
                print(f"Vòng lặp {i}, Loss: {cost:.6f}")

        return self.weights


class StochasticGradientDescent(GradientDescent):
    """Thuật toán Stochastic Gradient Descent"""

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, batch_size=1):
        super().__init__(learning_rate, max_iter, tol)
        self.batch_size = batch_size

    def optimize(self, X, y, verbose=False):
        """Tối ưu bằng phương pháp Stochastic Gradient Descent"""
        m, n = X.shape
        if self.weights is None:
            self.initialize_weights(n)

        prev_cost = float('inf')
        indices = np.arange(m)

        for i in range(self.max_iter):
            # Xáo trộn dữ liệu
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batches = [(X_shuffled[j:j + self.batch_size], y_shuffled[j:j + self.batch_size])
                       for j in range(0, m, self.batch_size)]

            epoch_cost = 0

            for batch_X, batch_y in batches:
                # Dự đoán
                y_pred = batch_X.dot(self.weights)

                # Tính loss cho batch
                batch_cost = np.mean((y_pred - batch_y) ** 2)
                epoch_cost += batch_cost * len(batch_y) / m

                # Tính gradient và cập nhật trọng số
                gradient = self.compute_gradient(batch_X, batch_y, y_pred)
                self.weights -= self.learning_rate * gradient

            # Kiểm tra điều kiện dừng
            if np.abs(prev_cost - epoch_cost) < self.tol:
                if verbose:
                    print(f"Hội tụ tại epoch {i}")
                break

            prev_cost = epoch_cost

            if verbose and (i % 10 == 0):
                print(f"Epoch {i}, Loss: {epoch_cost:.6f}")

        return self.weights


class Adam(BaseOptimizer):
    """Thuật toán Adam"""

    def __init__(self, learning_rate=0.001, max_iter=1000, tol=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate, max_iter, tol)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, y, verbose=False):
        """Tối ưu bằng thuật toán Adam"""
        m, n = X.shape
        if self.weights is None:
            self.initialize_weights(n)

        # Khởi tạo moment bậc 1 và bậc 2
        m_t = np.zeros(n)
        v_t = np.zeros(n)

        t = 0
        prev_cost = float('inf')

        for i in range(self.max_iter):
            t += 1

            # Dự đoán
            y_pred = X.dot(self.weights)

            # Tính hàm mất mát
            cost = np.mean((y_pred - y) ** 2)

            # Kiểm tra điều kiện dừng
            if np.abs(prev_cost - cost) < self.tol:
                if verbose:
                    print(f"Hội tụ tại vòng lặp {i}")
                break

            # Tính gradient
            grad = (1 / m) * X.T.dot(y_pred - y)

            # Cập nhật ước lượng moment bậc 1 có độ lệch
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad

            # Cập nhật ước lượng moment bậc 2 có độ lệch
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad ** 2)

            # Tính ước lượng moment bậc 1 được hiệu chỉnh
            m_t_hat = m_t / (1 - self.beta1 ** t)

            # Tính ước lượng moment bậc 2 được hiệu chỉnh
            v_t_hat = v_t / (1 - self.beta2 ** t)

            # Cập nhật trọng số
            self.weights -= self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)

            prev_cost = cost

            if verbose and (i % 100 == 0):
                print(f"Vòng lặp {i}, Loss: {cost:.6f}")

        return self.weights


class LinearRegression:
    """
    Lớp Linear Regression với nhiều phương pháp tối ưu khác nhau
    """

    def __init__(self, optimizer='normal_equation', learning_rate=0.01, max_iter=1000,
                 tol=1e-4, add_intercept=True, **kwargs):
        self.add_intercept = add_intercept
        self.weights = None

        # Tạo optimizer theo yêu cầu
        if optimizer == 'normal_equation':
            self.optimizer = BaseOptimizer(learning_rate, max_iter, tol)
        elif optimizer == 'gradient_descent':
            self.optimizer = GradientDescent(learning_rate, max_iter, tol)
        elif optimizer == 'sgd':
            batch_size = kwargs.get('batch_size', 32)
            self.optimizer = StochasticGradientDescent(learning_rate, max_iter, tol, batch_size)
        elif optimizer == 'adam':
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            epsilon = kwargs.get('epsilon', 1e-8)
            self.optimizer = Adam(learning_rate, max_iter, tol, beta1, beta2, epsilon)
        else:
            raise ValueError(f"Optimizer không hợp lệ: {optimizer}")

    def _add_intercept(self, X):
        """Thêm hệ số chặn vào ma trận X"""
        if self.add_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    def fit(self, X, y, verbose=False):
        """Huấn luyện mô hình trên dữ liệu"""
        X_with_intercept = self._add_intercept(X)

        if verbose:
            print(f"Đang huấn luyện với {type(self.optimizer).__name__}")
            print(f"Kích thước dữ liệu: {X_with_intercept.shape}")

        if isinstance(self.optimizer, BaseOptimizer) and not isinstance(self.optimizer, (
        GradientDescent, StochasticGradientDescent, Adam)):
            # Sử dụng normal equation cho BaseOptimizer
            self.weights = self.optimizer.normal_equation(X_with_intercept, y)
        else:
            # Sử dụng phương pháp tối ưu dựa trên gradient
            self.weights = self.optimizer.optimize(X_with_intercept, y, verbose)

        return self

    def predict(self, X):
        """Dự đoán sử dụng mô hình đã được huấn luyện"""
        if self.weights is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước predict().")

        X_with_intercept = self._add_intercept(X)
        return X_with_intercept.dot(self.weights)

    def evaluate(self, X, y):
        """Đánh giá mô hình trên tập test"""
        y_pred = self.predict(X)

        # Mean Squared Error
        mse = np.mean((y - y_pred) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(y - y_pred))
        maee = np.mean(np.abs(np.exp(y) - np.exp(y_pred)))


        # R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            "MAE exp": maee
        }

        # In ra các metrix
        print("Kết quả đánh giá:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

        return metrics


# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo dữ liệu giả
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_weights = np.array([0.5, -0.3, 0.2, -0.7, 0.1])
    y = X.dot(true_weights) + 0.5 + np.random.randn(100) * 0.1

    # Chia dữ liệu thành tập train và test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Huấn luyện với normal equation
    print("\n=== Huấn luyện bằng Normal Equation ===")
    model1 = LinearRegression(optimizer='normal_equation')
    model1.fit(X_train, y_train, verbose=True)
    model1.evaluate(X_test, y_test)

    # Huấn luyện với gradient descent
    print("\n=== Huấn luyện bằng Gradient Descent ===")
    model2 = LinearRegression(optimizer='gradient_descent', learning_rate=0.1, max_iter=1000)
    model2.fit(X_train, y_train, verbose=True)
    model2.evaluate(X_test, y_test)

    # Huấn luyện với stochastic gradient descent
    print("\n=== Huấn luyện bằng Stochastic Gradient Descent ===")
    model3 = LinearRegression(optimizer='sgd', learning_rate=0.01, max_iter=100, batch_size=16)
    model3.fit(X_train, y_train, verbose=True)
    model3.evaluate(X_test, y_test)

    # Huấn luyện với Adam
    print("\n=== Huấn luyện bằng Adam ===")
    model4 = LinearRegression(optimizer='adam', learning_rate=0.01, max_iter=500)
    model4.fit(X_train, y_train, verbose=True)
    model4.evaluate(X_test, y_test)