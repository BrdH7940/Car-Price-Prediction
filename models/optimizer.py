# utils/optimizers.py
import numpy as np

# Hàm tính cost


def compute_cost(X, y, weights, bias, regularization=None, lambda_param=0):
    m = X.shape[0]
    print(f"X: {X}")
    print(f"W: {weights}")
    y_pred = np.dot(X, weights) + bias
    print(y_pred[0])
    print(y[0])
    cost = (1/(2*m)) * \
        np.sum((y_pred.reshape((-1, 1)) - y.reshape((-1, 1)))**2)
    # Thêm regularization nếu có
    if regularization == 'l2':  # Ridge
        cost += (lambda_param/(2*m)) * np.sum(weights**2)
    elif regularization == 'l1':  # Lasso
        cost += (lambda_param/m) * np.sum(np.abs(weights))

    return cost


class BaseOptimizer:
    """
    Lớp cơ sở cho các thuật toán tối ưu
    """

    def __init__(self, learning_rate=0.01, random_state=None):
        """
        Khởi tạo optimizer

        Parameters:
        -----------
        learning_rate : float
            Tốc độ học cho quá trình tối ưu
        random_state : int hoặc None
            Seed cho random generator
        """
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Thiết lập random seed nếu có
        if random_state is not None:
            np.random.seed(random_state)

    def initialize(self, n_features):
        """
        Khởi tạo tham số

        Parameters:
        -----------
        n_features : int
            Số lượng features

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số khởi tạo
        bias : float
            Bias khởi tạo
        """
        weights = np.random.randn(n_features) * 0.01
        bias = 0
        return weights, bias

    def update(self, weights, bias, gradients):
        """
        Cập nhật tham số theo gradients

        Parameters:
        -----------
        weights : numpy.ndarray
            Trọng số hiện tại
        bias : float
            Bias hiện tại
        gradients : tuple
            (dw, db) - Gradients của weights và bias

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số đã cập nhật
        bias : float
            Bias đã cập nhật
        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_gradients(self, X, y, weights, bias, regularization=None, lambda_param=0):
        """
        Tính gradients cho weights và bias

        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Giá trị mục tiêu
        weights : numpy.ndarray
            Trọng số hiện tại
        bias : float
            Bias hiện tại
        regularization : str, optional ('l1', 'l2', None)
            Phương pháp regularization
        lambda_param : float
            Tham số regularization

        Returns:
        --------
        dw : numpy.ndarray
            Gradients của weights
        db : float
            Gradient của bias
        """
        m = X.shape[0]
        y_pred = np.dot(X, weights) + bias

        dw = (1/m) * (X.T @ (y_pred.reshape((-1, 1)) - y.reshape((-1, 1))))
        db = (1/m) * np.sum(y_pred - y)
        # Thêm đạo hàm của regularization
        if regularization == 'l2':  # Ridge
            dw += (lambda_param/m) * weights
        elif regularization == 'l1':  # Lasso
            dw += (lambda_param/m) * np.sign(weights)
        dw = dw.reshape(-1)
        return dw, db

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=False):
        """
        Thực hiện quá trình tối ưu

        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Giá trị mục tiêu
        weights : numpy.ndarray
            Trọng số ban đầu
        bias : float
            Bias ban đầu
        n_iterations : int
            Số lần lặp tối đa
        regularization : str, optional ('l1', 'l2', None)
            Phương pháp regularization
        lambda_param : float
            Tham số regularization
        tol : float
            Điều kiện dừng khi thay đổi cost function nhỏ hơn tol
        max_iter : int hoặc None
            Giới hạn số lần lặp không cải thiện
        verbose : bool
            In thông tin chi tiết trong quá trình tối ưu

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số tối ưu
        bias : float
            Bias tối ưu
        costs : list
            Cost function qua các lần lặp
        iterations : list
            Số lần lặp tương ứng
        """
        raise NotImplementedError("Subclasses must implement this method")


class GradientDescent(BaseOptimizer):
    """
    Thuật toán Gradient Descent
    """

    def update(self, weights, bias, gradients):
        """
        Cập nhật tham số theo gradients

        Parameters:
        -----------
        weights : numpy.ndarray
            Trọng số hiện tại
        bias : float
            Bias hiện tại
        gradients : tuple
            (dw, db) - Gradients của weights và bias

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số đã cập nhật
        bias : float
            Bias đã cập nhật
        """
        dw, db = gradients
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        # print(dw)
        return weights, bias

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=False):
        """
        Thực hiện quá trình tối ưu với Gradient Descent
        """
        costs = []
        iterations = []
        previous_cost = float('inf')
        no_improvement_count = 0

        for i in range(n_iterations):
            # Tính gradients
            gradients = self.compute_gradients(
                X, y, weights, bias, regularization, lambda_param)

            # Tính cost và lưu vào history
            cost = compute_cost(X, y, weights, bias)
            costs.append(cost)
            iterations.append(i)

            # Cập nhật tham số
            weights, bias = self.update(weights, bias, gradients)

            # In thông tin nếu verbose=True
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")

            # Kiểm tra điều kiện dừng
            if abs(previous_cost - cost) < tol:
                if verbose:
                    print(f"Converged after {i} iterations")
                break

            # Kiểm tra cải thiện
            if cost >= previous_cost:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Dừng nếu không có cải thiện sau một số lần lặp
            if max_iter and no_improvement_count >= max_iter:
                if verbose:
                    print(
                        f"Early stopping after {i} iterations without improvement")
                break

            previous_cost = cost

        return weights, bias, costs, iterations


class StochasticGradientDescent(BaseOptimizer):
    """
    Thuật toán Stochastic Gradient Descent
    """

    def update(self, weights, bias, gradients):
        """Cập nhật tham số với SGD"""
        dw, db = gradients
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=True):
        """
        Thực hiện quá trình tối ưu với Stochastic Gradient Descent
        """
        costs = []
        iterations = []
        previous_cost = float('inf')
        no_improvement_count = 0
        m = X.shape[0]

        for i in range(n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Tiến hành SGD
            for j in range(m):
                # Lấy một mẫu
                X_sample = X_shuffled[j:j+1]
                y_sample = y_shuffled[j:j+1]

                # Tính gradients
                gradients = self.compute_gradients(X_sample, y_sample, weights, bias,
                                                   regularization, lambda_param)

                # Cập nhật tham số
                weights, bias = self.update(weights, bias, gradients)

            # Tính cost và lưu vào history (trên toàn bộ dữ liệu)
            cost = compute_cost(X, y, weights, bias)
            costs.append(cost)
            iterations.append(i)

            # In thông tin nếu verbose=True
            if verbose and i % 10 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

            # Kiểm tra điều kiện dừng
            if abs(previous_cost - cost) < tol:
                if verbose:
                    print(f"Converged after {i} epochs")
                break

            # Kiểm tra cải thiện
            if cost >= previous_cost:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Dừng nếu không có cải thiện sau một số lần lặp
            if max_iter and no_improvement_count >= max_iter:
                if verbose:
                    print(
                        f"Early stopping after {i} epochs without improvement")
                break

            previous_cost = cost

        return weights, bias, costs, iterations


class MiniBatchGradientDescent(BaseOptimizer):
    """
    Thuật toán Mini-batch Gradient Descent
    """

    def __init__(self, learning_rate=0.01, batch_size=32, random_state=None):
        """
        Khởi tạo optimizer

        Parameters:
        -----------
        learning_rate : float
            Tốc độ học cho quá trình tối ưu
        batch_size : int
            Kích thước batch
        random_state : int hoặc None
            Seed cho random generator
        """
        super().__init__(learning_rate, random_state)
        self.batch_size = batch_size

    def update(self, weights, bias, gradients):
        """Cập nhật tham số với Mini-batch GD"""
        dw, db = gradients
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=False):
        """
        Thực hiện quá trình tối ưu với Mini-batch Gradient Descent
        """
        costs = []
        iterations = []
        previous_cost = float('inf')
        no_improvement_count = 0
        m = X.shape[0]
        batch_size = min(self.batch_size, m)

        for i in range(n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Tiến hành mini-batch GD
            for j in range(0, m, batch_size):
                # Lấy một batch
                end = min(j + batch_size, m)
                X_batch = X_shuffled[j:end]
                y_batch = y_shuffled[j:end]

                # Tính gradients
                gradients = self.compute_gradients(X_batch, y_batch, weights, bias,
                                                   regularization, lambda_param)

                # Cập nhật tham số
                weights, bias = self.update(weights, bias, gradients)

            # Tính cost và lưu vào history (trên toàn bộ dữ liệu)
            cost = compute_cost(X, y, weights, bias)
            costs.append(cost)
            iterations.append(i)

            # In thông tin nếu verbose=True
            if verbose and i % 10 == 0:
                print(f"Epoch {i}: Cost = {cost:.6f}")

            # Kiểm tra điều kiện dừng
            if abs(previous_cost - cost) < tol:
                if verbose:
                    print(f"Converged after {i} epochs")
                break

            # Kiểm tra cải thiện
            if cost >= previous_cost:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Dừng nếu không có cải thiện sau một số lần lặp
            if max_iter and no_improvement_count >= max_iter:
                if verbose:
                    print(
                        f"Early stopping after {i} epochs without improvement")
                break

            previous_cost = cost

        return weights, bias, costs, iterations


class NormalEquation(BaseOptimizer):
    """
    Thuật toán Normal Equation
    """

    def optimize(self, X, y, weights, bias, n_iterations=1, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=False):
        """
        Thực hiện tối ưu với Normal Equation (chỉ cần 1 bước)
        """
        m = X.shape[0]
        costs = []
        iterations = [0]

        # Thêm cột bias
        X_b = np.column_stack((np.ones(m), X))

        # Tính weights theo công thức normal equation
        if regularization == 'l2':  # Ridge
            # Formula: θ = (X^T X + λI)^(-1) X^T y
            # Chú ý: không áp dụng regularization cho bias
            reg_matrix = np.eye(X_b.shape[1])
            reg_matrix[0, 0] = 0  # Không regularize bias

            theta = np.linalg.inv(
                X_b.T.dot(X_b) + lambda_param * reg_matrix).dot(X_b.T).dot(y)
        else:
            # Formula: θ = (X^T X)^(-1) X^T y
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        # Tách weights và bias
        bias = theta[0]
        weights = theta[1:]

        cost = compute_cost(X, y, weights, bias)
        costs.append(cost)

        if verbose:
            print(f"Normal Equation - Cost: {cost:.6f}")

        return weights, bias, costs, iterations


class AdamOptimizer(BaseOptimizer):
    """
    Thuật toán Adam Optimizer (Adaptive Moment Estimation)

    Adam kết hợp ưu điểm của Momentum (sử dụng trung bình động của gradient)
    và RMSprop (điều chỉnh learning rate dựa trên trung bình bình phương gradient)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, random_state=None):
        """
        Khởi tạo Adam optimizer

        Parameters:
        -----------
        learning_rate : float
            Tốc độ học
        beta1 : float
            Hệ số suy giảm cho moment bậc nhất (động lượng)
        beta2 : float
            Hệ số suy giảm cho moment bậc hai (RMSprop)
        epsilon : float
            Hằng số nhỏ để tránh chia cho 0
        random_state : int hoặc None
            Seed cho random generator
        """
        super().__init__(learning_rate, random_state)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Khởi tạo moment
        self.m_dw = None  # Momentum cho weights
        self.v_dw = None  # RMSprop cho weights
        self.m_db = None  # Momentum cho bias
        self.v_db = None  # RMSprop cho bias

        # Biến đếm thời gian
        self.t = 0

    def initialize(self, n_features):
        """
        Khởi tạo tham số và các moment

        Parameters:
        -----------
        n_features : int
            Số lượng features

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số khởi tạo
        bias : float
            Bias khởi tạo
        """
        weights, bias = super().initialize(n_features)

        # Khởi tạo moment
        self.m_dw = np.zeros(n_features)
        self.v_dw = np.zeros(n_features)
        self.m_db = 0
        self.v_db = 0
        self.t = 0

        return weights, bias

    def update(self, weights, bias, gradients):
        """
        Cập nhật tham số theo Adam

        Parameters:
        -----------
        weights : numpy.ndarray
            Trọng số hiện tại
        bias : float
            Bias hiện tại
        gradients : tuple
            (dw, db) - Gradients của weights và bias

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số đã cập nhật
        bias : float
            Bias đã cập nhật
        """
        dw, db = gradients
        self.t += 1

        # Cập nhật moment bậc nhất (Momentum)
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # Cập nhật moment bậc hai (RMSprop)
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db**2)

        # Hiệu chỉnh bias (bias correction)
        m_dw_corrected = self.m_dw / (1 - self.beta1**self.t)
        m_db_corrected = self.m_db / (1 - self.beta1**self.t)
        v_dw_corrected = self.v_dw / (1 - self.beta2**self.t)
        v_db_corrected = self.v_db / (1 - self.beta2**self.t)

        # Cập nhật tham số
        weights -= self.learning_rate * m_dw_corrected / \
            (np.sqrt(v_dw_corrected) + self.epsilon)
        bias -= self.learning_rate * m_db_corrected / \
            (np.sqrt(v_db_corrected) + self.epsilon)

        return weights, bias

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                 lambda_param=0, tol=1e-6, max_iter=None, verbose=False):
        """
        Thực hiện quá trình tối ưu với Adam

        Parameters:
        -----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Giá trị mục tiêu
        weights : numpy.ndarray
            Trọng số ban đầu
        bias : float
            Bias ban đầu
        n_iterations : int
            Số lần lặp tối đa
        regularization : str, optional ('l1', 'l2', None)
            Phương pháp regularization
        lambda_param : float
            Tham số regularization
        tol : float
            Điều kiện dừng khi thay đổi cost function nhỏ hơn tol
        max_iter : int hoặc None
            Giới hạn số lần lặp không cải thiện
        verbose : bool
            In thông tin chi tiết trong quá trình tối ưu

        Returns:
        --------
        weights : numpy.ndarray
            Trọng số tối ưu
        bias : float
            Bias tối ưu
        costs : list
            Cost function qua các lần lặp
        iterations : list
            Số lần lặp tương ứng
        """
        costs = []
        iterations = []
        previous_cost = float('inf')
        no_improvement_count = 0

        # Khởi tạo moment
        n_features = weights.shape[0]
        self.m_dw = np.zeros(n_features)
        self.v_dw = np.zeros(n_features)
        self.m_db = 0
        self.v_db = 0
        self.t = 0

        for i in range(n_iterations):
            # Tính gradients
            gradients = self.compute_gradients(
                X, y, weights, bias, regularization, lambda_param)

            # Cập nhật tham số
            weights, bias = self.update(weights, bias, gradients)

            # Tính cost và lưu vào history
            cost = compute_cost(X, y, weights, bias)
            costs.append(cost)
            iterations.append(i)

            # In thông tin nếu verbose=True
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")

            # Kiểm tra điều kiện dừng
            if abs(previous_cost - cost) < tol:
                if verbose:
                    print(f"Converged after {i} iterations")
                break

            # Kiểm tra cải thiện
            if cost >= previous_cost:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Dừng nếu không có cải thiện sau một số lần lặp
            if max_iter and no_improvement_count >= max_iter:
                if verbose:
                    print(
                        f"Early stopping after {i} iterations without improvement")
                break

            previous_cost = cost

        return weights, bias, costs, iterations
