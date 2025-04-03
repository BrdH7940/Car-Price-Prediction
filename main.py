# test_linear_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from models.model import LinearRegression
from utils.visualization import plot_learning_curve


def generate_data(n_samples=100, n_features=3, noise=10.0, random_state=69):
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    y = y.reshape(-1, 1)
    return X, y


def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2
    }


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


def compare_optimizers(X_train, X_test, y_train, y_test):
    """Compare different optimization methods"""
    optimizers = ['gradient_descent', 'sgd',
                  'mini_batch_gd', 'normal_equation']
    results = {}

    for opt in optimizers:
        print(f"\nTesting optimizer: {opt}")

        # Skip normal equation for large datasets to avoid memory issues
        if opt == 'normal_equation' and X_train.shape[0] > 10000:
            print("Skipping normal equation for large dataset")
            continue

        # Set appropriate parameters based on optimizer
        if opt == 'normal_equation':
            model = LinearRegression(optimizer=opt)
        elif opt == 'sgd':
            model = LinearRegression(
                optimizer=opt, learning_rate=0.01, n_iterations=1000)
        elif opt == 'mini_batch_gd':
            model = LinearRegression(
                optimizer=opt, learning_rate=0.01, n_iterations=1000, batch_size=32)
        else:
            model = LinearRegression(
                optimizer=opt, learning_rate=0.01, n_iterations=1000)

        # Train and evaluate
        model.fit(X_train, y_train, verbose=True)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        print(f"Performance metrics for {opt}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Store results
        results[opt] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
        print(model.cost_history[0])
        # Plot learning curve if available
        if opt != 'normal_equation':
            try:
                plot_learning_curve(
                    model.iteration_history, model.cost_history)
            except Exception as e:
                print(f"Could not plot learning curve: {e}")

    return results


def test_regularization(X_train, X_test, y_train, y_test):
    """Test different regularization methods"""
    reg_methods = [None, 'l1', 'l2']
    lambda_values = [0.01, 0.1, 1.0]

    best_mse = float('inf')
    best_config = None

    for reg in reg_methods:
        for lambda_val in lambda_values:
            if reg is None and lambda_val > 0.01:
                continue  # Skip unnecessary combinations

            print(f"\nTesting regularization: {reg}, lambda: {lambda_val}")

            model = LinearRegression(
                optimizer='gradient_descent',
                learning_rate=0.01,
                n_iterations=1000,
                regularization=reg,
                lambda_param=lambda_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            print(f"Performance metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

            if metrics['MSE'] < best_mse:
                best_mse = metrics['MSE']
                best_config = {
                    'regularization': reg,
                    'lambda': lambda_val,
                    'metrics': metrics
                }

    print("\nBest regularization configuration:")
    print(f"  Regularization: {best_config['regularization']}")
    print(f"  Lambda: {best_config['lambda']}")
    print(f"  MSE: {best_config['metrics']['MSE']:.4f}")
    print(f"  R2: {best_config['metrics']['R2']:.4f}")

    return best_config


def main():
    # Test with simple 1D data
    print("=" * 50)
    print("Testing with 1D data")
    print("=" * 50)
    X, y = generate_data(n_samples=100, n_features=30, noise=10.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare optimizers
    results = compare_optimizers(
        X_train_scaled, X_test_scaled, y_train, y_test)

    # Visualize results for gradient descent
    if 'normal_equation' in results:
        visualize_results(
            X_test,
            y_test,
            results['normal_equation']['predictions'],
            "Gradient Descent Predictions (1D data)"
        )

    # Test with multi-dimensional data
    print("\n" + "=" * 50)
    print("Testing with multi-dimensional data")
    print("=" * 50)
    X_multi, y_multi = generate_data(n_samples=500, n_features=5, noise=20.0)
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    # Scale features
    scaler_multi = StandardScaler()
    X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
    X_test_multi_scaled = scaler_multi.transform(X_test_multi)

    # Compare optimizers for multi-dimensional data
    multi_results = compare_optimizers(
        X_train_multi_scaled,
        X_test_multi_scaled,
        y_train_multi,
        y_test_multi
    )

    # Visualize results for normal equation (usually most accurate for this type of problem)
    if 'normal_equation' in multi_results:
        visualize_results(
            X_test_multi,
            y_test_multi,
            multi_results['normal_equation']['predictions'],
            "Normal Equation Predictions (Multi-dimensional data)"
        )

    # Test regularization
    print("\n" + "=" * 50)
    print("Testing regularization methods")
    print("=" * 50)
    # Create data with some collinearity for better regularization testing
    X_reg = np.random.randn(200, 10)
    # Add some collinearity
    X_reg[:, 5] = 0.8 * X_reg[:, 0] + 0.2 * np.random.randn(200)
    X_reg[:, 6] = 0.8 * X_reg[:, 1] + 0.2 * np.random.randn(200)
    # True weights with some zeros to test L1
    true_weights = np.array([1.0, 0.8, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.2, 0.1])
    y_reg = X_reg @ true_weights.reshape(-1, 1) + 0.5 * np.random.randn(200, 1)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Scale features
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Test regularization
    best_reg = test_regularization(
        X_train_reg_scaled,
        X_test_reg_scaled,
        y_train_reg,
        y_test_reg
    )


if __name__ == "__main__":
    main()
