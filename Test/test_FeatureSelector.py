import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from PreProcessing.FeatureSelector import FeatureSelector
from PreProcessing.VehicleDataPreprocessor import VehicleDataPreprocessor
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv("../Data/train.csv")
    processor = VehicleDataPreprocessor()
    df = processor.preprocess(df, train=True, norm=False)
    # featSelector = FeatureSelector(pre_process_df=df)
    # selected_features, df = featSelector.get_df(model_id=4)

    y = df["Log_Price"]
    X = df.drop("Log_Price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)
    X_train.to_csv("../Data/final_X_train.csv", index=False)
    X_test.to_csv("../Data/final_X_test.csv", index=False)
    y_train.to_csv("../Data/final_y_train.csv", index=False)
    y_test.to_csv("../Data/final_y_test.csv", index=False)

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    #
    # # Dự đoán trên tập test
    # y_pred = model.predict(X_test)
    #
    # # Tính toán các metrics cho Log_Price
    # mse_log = mean_squared_error(y_test, y_pred)
    # mae_log = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    #
    # # In ra weights của mô hình
    # print("Feature weights:")
    # for feature, weight in zip(selected_features, model.coef_):
    #     print(f"{feature}: {weight:.8f}")
    # print(f"Intercept: {model.intercept_:.8f}")
    #
    # # In ra các metrics cho Log_Price
    # print("\nMetrics for Log_Price:")
    # print(f"MSE: {mse_log:.4f}")
    # print(f"MAE: {mae_log:.4f}")
    # print(f"R^2: {r2:.4f}")
    #
    # # Chuyển từ Log_Price sang Price thực tế (exp của Log_Price)
    # y_test_exp = np.exp(y_test)
    # y_pred_exp = np.exp(y_pred)
    #
    # # Tính toán các metrics cho Price thực tế
    # mse_actual = mean_squared_error(y_test_exp, y_pred_exp)
    # mae_actual = mean_absolute_error(y_test_exp, y_pred_exp)
    # r2_actual = r2_score(y_test_exp, y_pred_exp)
    #
    # # In ra các metrics cho Price thực tế
    # print("\nMetrics for Actual Price (exponentiated):")
    # print(f"MSE: {mse_actual:.4f}")
    # print(f"MAE: {mae_actual:.4f}")
    # print(f"R^2: {r2_actual:.4f}")
