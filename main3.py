import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PreProcessing.FeatureSelector import FeatureSelector


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run solver to fit linear model.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # Normal equations |
        #                  v
        self.theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return np.dot(self.theta, x)


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
clf = GDA()
clf.fit(x_train, y_train)
clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape (m,).
        """

        m, n = x.shape

        phi = np.sum(y) / m
        mu_0 = np.dot(x.T, 1 - y) / np.sum(1 - y)
        mu_1 = np.dot(x.T, y) / np.sum(y)

        # Reshape y to compute pairwise product with mu
        y_reshaped = np.reshape(y, (m, -1))
        print(mu_1.shape)
        print(y_reshaped.shape)
        # Matrix comprises mu_0 and mu_1 based on the value of y. Shape(m, n)
        mu_x = y_reshaped * mu_1.T + (1 - y_reshaped) * mu_0.T

        x_centered = x - mu_x

        sigma = np.dot(x_centered.T, x_centered) / m
        sigma_inv = np.linalg.inv(sigma)

        # Compute theta and theta_0 according to the conclusion from part (c)
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = 1 / 2 * mu_0 @ sigma_inv @ mu_0 - 1 / 2 * \
            mu_1 @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)

        self.theta = np.insert(theta, 0, theta_0)

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return:  Outputs of shape (m,).
        """

        # Add x_0 = 1 convention to make predictions using theta^T x >= 0
        return add_intercept(x) @ self.theta >= 0


def visualize_results(X_test, y_test, y_pred, title="Predictions vs. Actual Values"):
    """Visualize model predictions against actual values"""
    plt.figure(figsize=(10, 6))

    # For 1D feature, plot data points
    if X_test.shape[1] == 1:
        plt.scatter(X_test, y_test, color='blue', label='Actual values')
        plt.scatter(X_test, y_pred, color='red', label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
    else:
        # For multi-dimensional data, just plot predicted vs actual
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        # Add perfect prediction line
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.title(title)
    plt.legend()
    plt.show()


X_train = pd.read_csv("Data/final_X_train.csv")
fs = FeatureSelector(X_train)
_, X_train = fs.get_df(model_id=1, get_Log_Price=False)
y_train = pd.read_csv("Data/final_y_train.csv")
X_test = pd.read_csv("Data/final_X_test.csv")
fs_test = FeatureSelector(X_test)
_, X_test = fs.get_df(model_id=1, get_Log_Price=False)
y_test = pd.read_csv("Data/final_y_test.csv")
X_train, y_train, X_test, y_test = X_train.to_numpy(
), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
clf = GDA()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
visualize_results(X_test, y_test, y_pred,
                  title="Predictions vs. Actual Values")
