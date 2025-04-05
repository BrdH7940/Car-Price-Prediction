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
    Adam Optimizer (Adaptive Moment Estimation) algorithm.

    Adam combines the advantages of Momentum (using moving average of gradients)
    and RMSprop (adjusting learning rate based on squared gradient averages).
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, random_state=None):
        """
        Initializes the Adam optimizer.

        Parameters:
        -----------
        learning_rate : float
            The learning rate (alpha).
        beta1 : float
            The exponential decay rate for the first moment estimates (momentum).
        beta2 : float
            The exponential decay rate for the second moment estimates (RMSprop).
        epsilon : float
            A small constant to prevent division by zero in the updates.
        random_state : int or None
            Seed for the random number generator (not typically used directly by Adam's core logic,
            but can be useful if initialization involves randomness).
        """
        if not 0.0 <= learning_rate: raise ValueError("Learning rate must be >= 0.")
        if not 0.0 <= beta1 < 1.0: raise ValueError("Beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0: raise ValueError("Beta2 must be in [0, 1).")
        if not 0.0 <= epsilon: raise ValueError("Epsilon must be >= 0.")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.random_state = random_state # Store if needed for initialization or other parts

        # State variables - initialized properly in optimize or when needed
        self.m_dw = None  # First moment vector for weights
        self.v_dw = None  # Second moment vector for weights
        self.m_db = None  # First moment scalar for bias
        self.v_db = None  # Second moment scalar for bias
        self.t = 0        # Timestep counter, starts at 0, increments *before* first use

        # Seed random generator if state is provided
        if random_state is not None:
            np.random.seed(random_state)

    def initialize(self, n_features):
        """
        Initializes parameters (weights, bias) and Adam's moment variables.

        Parameters:
        -----------
        n_features : int
            The number of features in the input data.

        Returns:
        --------
        weights : numpy.ndarray
            Initialized weights array of shape (n_features,).
        bias : float
            Initialized bias scalar.
        """
        # Initialize weights (e.g., small random values or zeros)
        # Using random values helps break symmetry
        # Scale by sqrt(1/n_features) for better initial variance (like Xavier/Glorot for linear)
        scale = np.sqrt(1. / n_features) if n_features > 0 else 1
        weights = np.random.randn(n_features) * scale
        # Initialize bias (often zero)
        bias = 0.0

        # Initialize Adam state variables based on the number of features
        self.m_dw = np.zeros(n_features)
        self.v_dw = np.zeros(n_features)
        self.m_db = 0.0
        self.v_db = 0.0
        self.t = 0 # Reset timestep counter for a fresh optimization run

        return weights, bias

    def update(self, weights, bias, gradients):
        """
        Updates the parameters using one step of the Adam algorithm.

        Parameters:
        -----------
        weights : numpy.ndarray
            Current weights.
        bias : float
            Current bias.
        gradients : tuple
            A tuple containing (dw, db) - gradients of the cost function
            with respect to weights and bias.

        Returns:
        --------
        updated_weights : numpy.ndarray
            Weights after applying the Adam update rule.
        updated_bias : float
            Bias after applying the Adam update rule.
        """
        dw, db = gradients
        self.t += 1 # Increment timestep counter

        # --- Update biased first moment estimate ---
        # m_dw = beta1 * m_dw + (1 - beta1) * dw
        # m_db = beta1 * m_db + (1 - beta1) * db
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # --- Update biased second raw moment estimate ---
        # v_dw = beta2 * v_dw + (1 - beta2) * (dw^2)
        # v_db = beta2 * v_db + (1 - beta2) * (db^2)
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db**2)

        # --- Compute bias-corrected first moment estimate ---
        # m_dw_corrected = m_dw / (1 - beta1^t)
        # m_db_corrected = m_db / (1 - beta1^t)
        m_dw_corr = self.m_dw / (1 - self.beta1**self.t)
        m_db_corr = self.m_db / (1 - self.beta1**self.t)

        # --- Compute bias-corrected second raw moment estimate ---
        # v_dw_corrected = v_dw / (1 - beta2^t)
        # v_db_corrected = v_db / (1 - beta2^t)
        v_dw_corr = self.v_dw / (1 - self.beta2**self.t)
        v_db_corr = self.v_db / (1 - self.beta2**self.t)

        # --- Update parameters ---
        # weights = weights - learning_rate * m_dw_corrected / (sqrt(v_dw_corrected) + epsilon)
        # bias = bias - learning_rate * m_db_corrected / (sqrt(v_db_corrected) + epsilon)
        updated_weights = weights - self.learning_rate * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)
        updated_bias = bias - self.learning_rate * m_db_corr / (np.sqrt(v_db_corr) + self.epsilon)

        return updated_weights, updated_bias

    def optimize(self, X, y, weights, bias, n_iterations, regularization=None,
                lambda_param=0.0, tol=1e-6, max_iter=None, verbose=False):
        """
        Performs the optimization process using Adam.
        ... (rest of the docstring) ...
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            return weights, bias, [], [] # Handle empty input

        # --- Ensure y is a 1D array ---
        y = np.array(y).squeeze() # Convert to numpy array and remove single dimensions
        if y.ndim != 1:
             raise ValueError(f"Target y must be a 1D array, but got shape {y.shape} after squeeze.")
        if y.shape[0] != n_samples:
             raise ValueError(f"Number of samples in X ({n_samples}) does not match y ({y.shape[0]}).")
        # --- End of y shape check ---


        costs = []
        iterations_list = []
        last_cost = float('inf')

        # --- Initialize Adam State Variables ---
        if self.m_dw is None or self.m_dw.shape != weights.shape:
             # Ensure weights is 1D as expected
             if weights.ndim != 1:
                 raise ValueError(f"Initial weights must be 1D, but got shape {weights.shape}")
             self.m_dw = np.zeros_like(weights)
             self.v_dw = np.zeros_like(weights)
             self.m_db = 0.0
             self.v_db = 0.0
             self.t = 0

        for i in range(n_iterations):
            # --- Forward Pass ---
            # Ensure weights are used correctly for dot product
            y_pred = X.dot(weights) + bias # X:(1310, 21), weights:(21,) -> y_pred:(1310,)
            error = y_pred - y # y_pred:(1310,), y:(1310,) -> error:(1310,)

            # --- Cost Calculation (MSE + Regularization) ---
            mse_cost = (1 / (2 * n_samples)) * np.sum(error**2)
            reg_cost = 0
            if regularization == 'l2':
                reg_cost = (lambda_param / (2 * n_samples)) * np.sum(weights**2)
            elif regularization == 'l1':
                reg_cost = (lambda_param / n_samples) * np.sum(np.abs(weights))
            current_cost = mse_cost + reg_cost

            # --- Check Cost List Appending ---
            # Ensure cost is a scalar before appending
            if not isinstance(current_cost, (int, float, np.number)):
                 raise TypeError(f"Calculated cost is not a scalar: {current_cost} (Type: {type(current_cost)})")
            costs.append(current_cost)
            iterations_list.append(i)


            # --- Stopping Criterion (Tolerance) ---
            if i > 0 and abs(last_cost - current_cost) < tol:
                if verbose:
                    print(f"\nConvergence reached at iteration {i}. Change in cost < {tol}")
                break
            last_cost = current_cost

            # --- Gradient Calculation ---
            # X.T:(21, 1310), error:(1310,) -> dw:(21,)
            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error) # Should be scalar

            # Ensure gradients have expected shapes before regularization
            if dw.shape != weights.shape:
                raise ValueError(f"Shape mismatch: dw shape {dw.shape} vs weights shape {weights.shape}")
            if not np.isscalar(db):
                 raise TypeError(f"Bias gradient db is not a scalar: {db}")


            # Add regularization term to gradients (bias is typically not regularized)
            if regularization == 'l2':
                dw += (lambda_param / n_samples) * weights
            elif regularization == 'l1':
                dw += (lambda_param / n_samples) * np.sign(weights)

            # --- Parameter Update ---
            gradients = (dw, db)
            weights, bias = self.update(weights, bias, gradients) # Pass dw:(21,), db:scalar

        if verbose:
            if i < n_iterations -1 :
                pass
            else:
                print(f"\nOptimization finished after {n_iterations} iterations.")
            # Ensure final cost is accessible even if loop breaks early
            final_cost = costs[-1] if costs else float('nan')
            print(f"Final Cost: {final_cost:,.4f}")


        return weights, bias, costs, iterations_list
