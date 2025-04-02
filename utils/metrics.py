import numpy as np


def mean_square_error(y_true, y_pred):
    """ Calculate MSE 

        Para:
            y_true: ground truth
            y_pred: result of model
        Return:
            MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_square_error(y_true, y_pred):
    """ Calculate RMSE"""
    return np.sqrt(mean_square_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """ Calculate MAE """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """ Calculate R2 score """

    # Calculate total sum of square
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    # Calculate residual sum of square
    rss = np.sum((y_pred - y_true) ** 2)

    # Calculat r2 score
    r2 = 1 - (rss / (tss + 1e-8))

    return r2
