from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

from PreProcessing.VehicleDataPreprocessor import VehicleDataPreprocessor
from PreProcessing.FeatureSelector import FeatureSelector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearModel.StandardScaler import StandardScaler
# from models.model import LinearRegression
from LinearModel.LinearModel import LinearRegression
from utils.data_preprocessing import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle


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
    df = pd.read_csv("../Data/train.csv")
    processor = VehicleDataPreprocessor()
    df = processor.preprocess(df, train=True, norm=False)
    # featSelector = FeatureSelector(pre_process_df=df)
    # selected_features, df = featSelector.get_df(model_id=4)
    scaler = StandardScaler()

    y = df["Log_Price"]
    X = df.drop("Log_Price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    X_train_o = FeatureSelector._non_linearize_features(X_train)
    X_test_o = FeatureSelector._non_linearize_features(X_test)

    feats, X_train = FeatureSelector.get_features(X_train)
    _, X_test = FeatureSelector.get_features(X_test)

    for i in range(1,5):
        print(f" === MODEL {i} ===")
        feats, X_train = FeatureSelector.get_df(X_train_o, model_id=i, get_Log_Price=False)
        _, X_test = FeatureSelector.get_df(X_test_o, model_id=i, get_Log_Price=False)

        # print(X_train.shape)

        X_train, X_test= X_train.to_numpy(), X_test.to_numpy()

        # print(X_train.shape)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LinearRegression(
            optimizer='adam',
            learning_rate=0.01,
            max_iter=5000,
            l2_penalty=0.005,
            l1_penalty=0.005,
            random_state=42,
            tol=1e-10,
        )

        # Huấn luyện mô hình
        # model.fit(X_train, y_train, verbose=False)
        # Vẽ learning curve
        history = model.plot_learning_curve(X_train, y_train, val_size=0.1,
                                            figsize=(20,10),
                                            metrics=['mse'], verbose=False)

        # Dự đoán và đánh giá
        # y_pred_train = model.predict(X_train)
        # metrics = model._evaluate(y_train, y_pred_train)
        # print("Train metrics: ", metrics)
        # visualize_results(X_train, y_train, y_pred_train)
        # y_pred = model.predict(X_test)
        # metrics = model._evaluate(y_test, y_pred)
        # print("Test metrics:", metrics)
        #
        # visualize_results(X_test, y_test, y_pred)
        # visualize_results(X_test, np.exp(y_test), np.exp(y_pred))
        #
        # sns.histplot(y_test - y_pred)
        # plt.show()

        # # Cross-validation
        # cv_results = model.cross_validate(X_train, y_train, n_folds=10, verbose=False)
        # y_pred = model.predict(X_test)
        # metrics = model._evaluate(y_test, y_pred)
        # print("Test metrics:", metrics)
        # feats.insert(0, "Intercept")
        # # for col, weight in zip(feats, model.optimizer.weights):
        # #     if np.abs(weight) >= 0.01:
        # #         print(f"'{col}', {weight}")

        # history = model.plot_cross_validation_metrics(X_train, y_train, n_folds=10, figsize=(20,10), metrics=['mse'])
        # print("Cross validation metrics:", history)
        # y_pred = model.predict(X_test)
        # metrics = model._evaluate(y_test, y_pred)
        # print("After Cross-Validate metrics:", metrics)
        #
        # visualize_results(X_test, y_test, y_pred)
        # visualize_results(X_test, np.exp(y_test), np.exp(y_pred))

    # print(X_train[:,1].max())


    # print(X_train.shape)

    # print(X_test.mean())
    # print(y_test)

    # Huấn luyện với normal equation
    # print("\n=== Huấn luyện bằng Normal Equation ===")
    # model1 = LinearRegression(optimizer='normal')
    # # print(X_train.shape)
    # model1.fit(X_train, y_train, verbose=True)
    # y_pred_test = model1.predict(X_test)
    # model1._evaluate(y_test, y_pred_test)
    # mean_score, _ = model1.cross_validate(X_train, y_train)
    # print("Mean score of cross validation:", mean_score)

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
