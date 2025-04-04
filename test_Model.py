from PreProcessing.VehicleDataPreprocessor import VehicleDataPreprocessor
from PreProcessing.FeatureSelector import FeatureSelector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearModel.LinearModel import StandardScaler
from models.model import LinearRegression
from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    df = pd.read_csv("Data/train.csv")
    processor = VehicleDataPreprocessor()
    df = processor.preprocess(df, train=True, norm=False)
    # featSelector = FeatureSelector(pre_process_df=df)
    # selected_features, df = featSelector.get_df(model_id=4)
    scaler = StandardScaler()

    y = df["Log_Price"]
    X = df.drop("Log_Price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=43)

    X_train = FeatureSelector._non_linearize_features(X_train)
    X_test = FeatureSelector._non_linearize_features(X_test)

    _, X_train = FeatureSelector.get_df(
        X_train, model_id=1, get_Log_Price=False)
    _, X_test = FeatureSelector.get_df(X_test, model_id=1, get_Log_Price=False)

    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(
    ), y_train.to_numpy().flatten(), y_test.to_numpy().flatten()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print(X_test)
    # print(y_test)

    # Huấn luyện với normal equation
    print("\n=== Huấn luyện bằng Normal Equation ===")
    model1 = LinearRegression(optimizer='normal_equation')
    model1.fit(X_train, y_train, verbose=True)
    model1.evaluate(X_test, y_test)
    mean_score, _ = model1.cross_validate(X_train, y_train)
    print("Mean score of cross validation:", mean_score)

    # # Huấn luyện với gradient descent
    # print("\n=== Huấn luyện bằng Gradient Descent ===")
    # model2 = LinearRegression(optimizer='gradient_descent', learning_rate=0.01, max_iter=1000)
    # model2.fit(X_train, y_train, verbose=True)
    # model2.evaluate(X_test, y_test)

    # # Huấn luyện với stochastic gradient descent
    # print("\n=== Huấn luyện bằng Stochastic Gradient Descent ===")
    # model3 = LinearRegression(optimizer='sgd', learning_rate=0.01, max_iter=10000, batch_size=16)
    # model3.fit(X_train, y_train, verbose=True)
    # model3.evaluate(X_test, y_test)

    # # Huấn luyện với Adam
    # print("\n=== Huấn luyện bằng Adam ===")
    # model4 = LinearRegression(optimizer='adam', learning_rate=0.01, max_iter=500000, tol=1e-34)
    # model4.fit(X_train, y_train, verbose=True)
    # model4.evaluate(X_test, y_test)
    # print(model4.weights)
