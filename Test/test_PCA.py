import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from LinearModel.PCA import PCA
from PreProcessing.FeatureSelector import FeatureSelector
from PreProcessing.VehicleDataPreprocessor import VehicleDataPreprocessor
from LinearModel.StandardScaler import StandardScaler

SAVING_MODEL_PATH = '../ModelSave/linear_regression_model_'
SAVING_SCALER_PATH = '../ModelSave/scaler_'

df = pd.read_csv("../Data/train.csv")
processor = VehicleDataPreprocessor()
df = processor.preprocess(df, train=True, norm=False)

y = df["Log_Price"]
X = df.drop("Log_Price", axis=1)

X = FeatureSelector._non_linearize_features(X)

for i in range(1, 5):
    feats, X_train = FeatureSelector.get_df(X, model_id=i, get_Log_Price=False)

    scaler_path = f"{SAVING_SCALER_PATH}{i}" + '.pkl'
    model_path = SAVING_MODEL_PATH + str(i) + '.pkl'

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_train = scaler.transform(X_train)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_train)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    fig, ax = pca.visualize_3d(X_train, y)
    plt.show()