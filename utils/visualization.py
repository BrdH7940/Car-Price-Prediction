import numpy as np
from .data_preprocessing import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_results(X_test, y_test, y_pred, title="Predictions vs. Actual Values"):
    """Visualize model predictions against actual values (Corrected)"""
    plt.figure(figsize=(10, 6))

    # Ensure y_test and y_pred are numpy arrays and flattened
    y_test = np.asarray(y_test).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # For 1D feature, plot data points and predictions vs the feature
    if X_test.ndim == 2 and X_test.shape[1] == 1:
        X_test_flat = X_test.flatten() # Flatten X_test for plotting
        plt.scatter(X_test_flat, y_test, color='blue', alpha=0.5, label='Actual values')
        plt.scatter(X_test_flat, y_pred, color='red', alpha=0.5, label='Predictions')
        # Optionally sort for line plot, but scatter is fine
        # sort_idx = np.argsort(X_test_flat)
        # plt.plot(X_test_flat[sort_idx], y_pred[sort_idx], color='red', label='Prediction Line') # If you want a line
        plt.xlabel('Feature X')
        plt.ylabel('Value y')
        plt.legend()
    elif X_test.ndim == 1: # If X_test was already 1D
        plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual values')
        plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predictions')
        plt.xlabel('Feature X')
        plt.ylabel('Value y')
        plt.legend()
    else:
        # For multi-dimensional features, plot predicted vs actual
        print("Plotting Predicted vs Actual (multi-feature input)")
        # Use the flattened y_test and y_pred
        sns.scatterplot(x = y_test, y = y_pred, alpha=0.5, edgecolor=None, hue = y_test)
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        # Add perfect prediction line (y=x)
        # Calculate limits based on the flattened arrays
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], '--r', label='Ideal Fit') # Added label
        # Removed legend() from here, let scatterplot handle it if needed, or add specific labels

    plt.title(title)
    # Add grid for better readability
    plt.grid(True)
    # Show legend only if labels were added
    # Check if any labels were defined before calling legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.show()

def plot_features_w_PCA(X):
    pca = PCA()
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='plasma')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.show()
