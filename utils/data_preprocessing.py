import numpy as np
import pandas as pd

def train_test_split1(X, y, train_size, random_state=None):
    """
        Para:
            X: features 
            y: label
            train_size: size of train test
            random_seed: ...
        Return:
            X_train, y_train, X_test, y_test: splited data with chosen size
    """
    # Initialize random seed
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)
    # Size of data
    n_samples = X.shape[0]

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    # Split data
    n_samples_train = int(n_samples * train_size)
    train_indices = np.array(indices[:n_samples_train])
    test_indices = np.array(indices[n_samples_train:])
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split features (X) and target (y) DataFrames into training and testing sets.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature DataFrame
    y : pandas.DataFrame or pandas.Series
        The target DataFrame or Series
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split (between 0.0 and 1.0)
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split
        Pass an int for reproducible output

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) where all are pandas DataFrames/Series
    """
    # Validate inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError("y must be a pandas DataFrame or Series")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Get the number of rows
    n = len(X)

    # Calculate the number of test samples
    n_test = int(n * test_size)

    # Generate random indices for the test set
    test_indices = np.random.choice(range(n), size=n_test, replace=False)

    # Create boolean mask for test and train sets
    is_test = np.zeros(n, dtype=bool)
    is_test[test_indices] = True

    # Split X - sử dụng iloc để lấy dữ liệu theo vị trí hàng
    # Việc sử dụng .copy() và reset_index(drop=True) không ảnh hưởng đến tên cột
    X_test = X.iloc[is_test].copy().reset_index(drop=True)
    X_train = X.iloc[~is_test].copy().reset_index(drop=True)

    # Split y - tương tự, tên cột được giữ nguyên
    y_test = y.iloc[is_test].copy().reset_index(drop=True)
    y_train = y.iloc[~is_test].copy().reset_index(drop=True)

    return X_train, X_test, y_train, y_test


class StandardScaler:
    """ Normalize data following standard scaling """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """ Calculate mean and std of X """
        self.mean = np.mean(X)
        self.std = np.std(X)

    def transform(self, X):
        """ Normalize """
        return (X - self.mean) / (self.std + 1e-8)  # add epsilon to avoid dividing by zẻo

    def fit_transform(self, X):
        """ Calculate mean, std and normalize """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """ Convert to original """
        return X * self.std + self.mean


class MinMaxScaler:
    """ Normalize data following min-max scaling (0-1)"""

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None
        self.scale = None

    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
        self.scale = (
            self.feature_range[1] - self.feature_range[0]) / (self.max - self.min)

    def transform(self, X):
        return self.feature_range[0] + (X - self.min)*self.scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return (X - self.feature_range[0]) / self.scale + self.min


class PCA:
    """ Reduce dimension of data"""

    def __int__(self):
        self.mean = None
        self.std = None
        self.Z = None
        self.pca_component = None

    def fit(self, X):
        # Standardize data
        self.mean = np.mean(X)
        self.std = np.std(X)
        self.Z = (self.Z - self.mean) / (self.std + 1e-8)

        # Calculate covariance matrix
        covarience = self.Z.cov()

        # Calculate eigenvalue, eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(covarience)

        # Index the eigenvalues in descending order
        index = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        # Calculate explained variance
        explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        # Determine the number of principal components
        n_components = np.argmax(explained_var >= 0.50) + 1

        # PCA component
        self.pca_component = eigenvectors[:, :n_components]

    def transform(self, X):
        Z_pca = self.Z @ self.pca_component
        return Z_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
