import numpy as np
from typing import *

class StandardScaler:
    """
    Chuẩn hóa dữ liệu bằng cách loại bỏ giá trị trung bình và chia cho độ lệch chuẩn.

    Công thức: z = (x - mean) / std

    Parameters
    ----------
    with_mean : bool, default=True
        Nếu True, trừ giá trị trung bình khỏi dữ liệu.

    with_std : bool, default=True
        Nếu True, chia dữ liệu cho độ lệch chuẩn.

    Attributes
    ----------
    mean_ : array, shape (n_features,)
        Giá trị trung bình được tính từ dữ liệu huấn luyện.

    var_ : array, shape (n_features,)
        Phương sai được tính từ dữ liệu huấn luyện.

    scale_ : array, shape (n_features,)
        Độ lệch chuẩn được tính từ dữ liệu huấn luyện.

    n_features_in_ : int
        Số đặc trưng được thấy trong quá trình huấn luyện.

    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Tính toán giá trị trung bình và độ lệch chuẩn.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào mà sẽ được chuẩn hóa.

        y : Ignored
            Không sử dụng, tồn tại để tương thích API.

        Returns
        -------
        self : object
            Returns self.
        """

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Lưu số đặc trưng
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1

        # Tính giá trị trung bình
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_in_)

        # Tính phương sai và độ lệch chuẩn
        if self.with_std:
            self.var_ = np.var(X, axis=0)
            # Xử lý trường hợp phương sai bằng 0
            self.scale_ = np.sqrt(self.var_)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.var_ = np.ones(self.n_features_in_)
            self.scale_ = np.ones(self.n_features_in_)

        return self

    def transform(self, X):
        """
        Thực hiện chuẩn hóa dữ liệu.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào sẽ được chuẩn hóa.

        Returns
        -------
        X_scaled : array, shape (n_samples, n_features)
            Dữ liệu đã được chuẩn hóa.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler chưa được huấn luyện. "
                             "Hãy gọi fit() trước transform().")

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Kiểm tra số đặc trưng
        if X.ndim > 1 and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Số đặc trưng trong dữ liệu ({X.shape[1]}) "
                             f"không trùng khớp với số đặc trưng đã được học ({self.n_features_in_})")

        # Thực hiện chuẩn hóa
        X_scaled = X.copy()

        if self.with_mean:
            X_scaled -= self.mean_

        if self.with_std:
            X_scaled /= self.scale_

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit và transform cùng một lúc.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đầu vào mà sẽ được chuẩn hóa.

        y : Ignored
            Không sử dụng, tồn tại để tương thích API.

        Returns
        -------
        X_scaled : array, shape (n_samples, n_features)
            Dữ liệu đã được chuẩn hóa.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Chuyển dữ liệu đã chuẩn hóa về dạng ban đầu.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu đã chuẩn hóa.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Dữ liệu sau khi được chuyển về dạng ban đầu.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler chưa được huấn luyện. "
                             "Hãy gọi fit() trước inverse_transform().")

        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        X = np.array(X, dtype=np.float64)

        # Kiểm tra số đặc trưng
        if X.ndim > 1 and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Số đặc trưng trong dữ liệu ({X.shape[1]}) "
                             f"không trùng khớp với số đặc trưng đã được học ({self.n_features_in_})")

        # Thực hiện biến đổi ngược
        X_original = X.copy()

        if self.with_std:
            X_original *= self.scale_

        if self.with_mean:
            X_original += self.mean_

        return X_original

