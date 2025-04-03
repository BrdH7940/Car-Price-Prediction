import numpy as np
from .data_preprocessing import PCA
import matplotlib.pyplot as plt


def plot_learning_curve(iteration, cost_history, title="Learning Curve"):
    """
    Plot learning curve depend on cost history
    Para:
        iteration: i_th epoch
        cost_history: an array contains all cost in training process
    """
    print(cost_history[0], cost_history[1], cost_history[2])
    plt.figure(figsize=(10, 6))
    plt.scatter(iteration, cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Value of loss function')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_features_w_PCA(X):
    pca = PCA()
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='plasma')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.show()
