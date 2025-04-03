import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import warnings
from utils.data_preprocessing import *
from utils.metrics import *
from utils.visualization import *
from models.model import *
from models.optimizer import *
from PreProcessing.FeatureSelector import FeatureSelector
from PreProcessing.VehicleDataPreprocessor import VehicleDataPreprocessor
warnings.filterwarnings('ignore')

vehicle = VehicleDataPreprocessor()
X_train = pd.read_csv("Data/Final_train.csv")
X_train = vehicle.preprocess(X_train, train=True)
fs = FeatureSelector(X_train)
_, X_train = fs.get_df(model_id=1, get_Log_Price=False)
y_train = pd.read_csv("Data/final_y_train.csv")

X_test = pd.read_csv("Data/final_X_test.csv")
fs_test = FeatureSelector(X_test)
_, X_test = fs_test.get_df(model_id=1, get_Log_Price=False)
y_test = pd.read_csv("Data/final_y_test.csv")

X_train, y_train, X_test, y_test = X_train.to_numpy(
), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


# --- Model Training using Custom LinearRegression ---
print("\nInitializing and fitting custom LinearRegression model...")
# Choose optimizer and hyperparameters
# Options: 'gradient_descent', 'sgd', 'mini_batch_gd', 'normal_equation', 'AdamOptimizer' (if Adam class exists)
# For gradient-based, adjust learning_rate and n_iterations
model = LinearRegression(
    # Try 'normal_equation' if no regularization needed and data isn't too large
    optimizer='normal_equation',
    learning_rate=1e-5,         # May need tuning
    n_iterations=1000,          # May need tuning
    # regularization='l2',        # Optional: 'l1' or 'l2'
    # lambda_param=0.01,          # Regularization strength
    random_state=43,
    tol=1e-5,                   # Convergence tolerance
    max_iter=500                 # Early stopping patience
)

# Fit the model using scaled training data
# Set verbose=True for training progress
model.fit(X_train, y_train, verbose=True)

# --- Prediction using Custom Model ---
print("\nPredicting on the test set...")
y_pred = model.predict(X_test)
# If y was scaled, inverse transform predictions:
# y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
# y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# --- Evaluation using Custom Metrics ---
print("\nEvaluating model performance...")

# Calculate metrics for Log_Price
# Use original y_test and predicted y_pred
mse_log = mean_squared_error(y_test, y_pred)
mae_log = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print weights and intercept from custom model
print("\nFeature weights:")
params = model.get_params()
weights = params.get('weights')  # Use .get for safety
bias = params.get('bias')

# if weights is not None:
#     # Sort features by absolute weight magnitude (optional)
#     # feature_weights = sorted(zip(existing_features, weights), key=lambda item: abs(item[1]), reverse=True)
#     # for feature, weight in feature_weights:
#     #      print(f"{feature}: {weight:.8f}")

#     # Or print in original order
#     for feature, weight in zip(existing_features, weights):
#         print(f"{feature}: {weight:.8f}")

#     print(f"Intercept: {bias:.8f}")
# else:
#     print("Could not retrieve weights or feature length mismatch.")
#     print(
#         f"Number of features: {len(existing_features)}, Number of weights: {len(weights) if weights is not None else 'None'}")


# Print metrics for Log_Price
print("\nMetrics for Log_Price (Test Set):")
print(f"MSE: {mse_log:.4f}")
print(f"MAE: {mae_log:.4f}")
print(f"R^2: {r2:.4f}")

# --- Evaluation on Actual Price ---
print("\nCalculating metrics for Actual Price...")
# Inverse transform y_test and y_pred from log scale to actual price scale
# Use original y_test (which is Log_Price)
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred)

# Calculate metrics for Actual Price using custom functions
mse_actual = mean_squared_error(y_test_exp, y_pred_exp)
mae_actual = mean_absolute_error(y_test_exp, y_pred_exp)
# R^2 score on the exponentiated values might not be directly comparable or as meaningful
# It's often better to evaluate R^2 on the scale the model was trained on (log scale here)
# However, calculating it as requested:
r2_actual = r2_score(y_test_exp, y_pred_exp)

# Print metrics for Actual Price
print("\nMetrics for Actual Price (Test Set):")
print(f"MSE: {mse_actual:.4f}")
print(f"MAE: {mae_actual:.4f}")
print(f"R^2: {r2_actual:.4f}")


# --- Evaluation on Training Set ---
print("\nPredicting and evaluating on the training set...")
y_pred_train = model.predict(X_train)

# Metrics for Log Price (Train Set)
mse_train_log = mean_squared_error(y_train, y_pred_train)
mae_train_log = mean_absolute_error(y_train, y_pred_train)
r2_train_log = r2_score(y_train, y_pred_train)

print("\nMetrics for Log_Price (Train Set):")
print(f"MSE: {mse_train_log:.4f}")
print(f"MAE: {mae_train_log:.4f}")
print(f"R^2: {r2_train_log:.4f}")

# --- Visualization ---
print("\nGenerating plots...")

# Plot Learning Curve (if iterative optimizer was used)
if hasattr(model, 'plot_learning_curve') and model.optimizer.__class__.__name__ != 'NormalEquation':
    try:
        print("Plotting learning curve...")
        model.plot_learning_curve()  # Call the method from the LinearRegression instance
        plt.savefig('custom_learning_curve.png')
        plt.show()
    except Exception as e:
        print(f"Could not plot learning curve: {e}")
else:
    print("Learning curve plot not available for Normal Equation optimizer or model lacks the method.")


# Visualize predictions vs actual (Log Scale - Train)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, alpha=0.3, label='Train Data')
plt.plot([y_train.min(), y_train.max()], [
         y_train.min(), y_train.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted Log Price (Train Set - Custom Model)')
plt.legend()
plt.grid(True)
plt.savefig('custom_prediction_log_train.png')
plt.show()


# Visualize predictions vs actual (Actual Price Scale - Test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_exp, y_pred_exp, alpha=0.5, label='Test Data')
# Plot ideal fit line based on actual price range
min_price = min(y_test_exp.min(), y_pred_exp.min())
max_price = max(y_test_exp.max(), y_pred_exp.max())
plt.plot([min_price, max_price], [min_price,
         max_price], 'r--', label='Ideal Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (Test Set - Custom Model)')
plt.legend()
plt.grid(True)
# Optional: Adjust axis limits if needed, e.g., plt.xlim(0, max_price * 1.1), plt.ylim(0, max_price * 1.1)
plt.savefig('custom_prediction_actual_test.png')
plt.show()

print("\nScript finished.")
