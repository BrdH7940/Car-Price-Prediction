import numpy as np
import pandas as pd


def train_test_split(X, y, train_size, random_seed=None):
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
    if random_seed is not None:
        np.random.seed(random_seed)

    # Size of data
    n_samples = X.shape[0]

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split data
    n_samples_train = int(n_samples * train_size)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    X_train = X.iloc[train_indices].to_numpy()
    y_train = y.iloc[train_indices].to_numpy()
    X_test = X.iloc[test_indices].to_numpy()
    y_test = y.iloc[test_indices].to_numpy()

    return X_train, y_train, X_test, y_test


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
