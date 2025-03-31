import numpy as np
import pandas as pd

def normalization(data):
    #data after preprocessing
    std = data.std()

    return pd.to_numpy((data - data.mean()) / std)

def Gram_Matrix(Span):
     # Span: A Matrix that includes linearly dependent column vectors
    return np.dot(Span.T, Span)

def ProjectionMatrix(Span):
     # Span: A Matrix that includes linearly dependent column vectors
    return np.dot(Span, np.dot(np.linalg.pinv(Gram_Matrix(Span)), Span.T))

def Projection_onto_vector_space(Vector_space, Y):
    return np.dot(ProjectionMatrix(Vector_space), Y)

def Norm(Vector):
    return np.linalg.norm(Vector)

def Vector_correlation(u, v):
    return np.dot(u, v) / (Norm(u) * Norm(v))

def Vector_space_correlation(X, Y):
    X_normalized = normalization(X)
    Y_normalized = normalization(Y)
    projXY = Projection_onto_vector_space(X_normalized, Y_normalized)
    return Vector_correlation(projXY, Y_normalized)
