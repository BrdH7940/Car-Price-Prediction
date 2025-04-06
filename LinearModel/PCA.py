import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class PCA:
    def __init__(self, n_components=3):
        """
        Khởi tạo PCA class

        Parameters:
        -----------
        n_components : int, default=3
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Parameters:
        -----------
        X : numpy.ndarray hoặc pandas.DataFrame
            Dữ liệu đầu vào có shape (n_samples, n_features)

        Returns:
        --------
        self : object
        """
        # Chuyển đổi DataFrame thành ndarray nếu cần
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples, n_features = X.shape

        # Tính toán giá trị trung bình theo cột
        self.mean = np.mean(X, axis=0)

        # Chuẩn hóa dữ liệu
        X_centered = X - self.mean

        # Tính toán ma trận hiệp phương sai
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Tính toán các giá trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sắp xếp theo thứ tự giảm dần của các giá trị riêng
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Lưu các thành phần chính (principal components)
        self.components = eigenvectors[:, :self.n_components]

        # Lưu phương sai và tỉ lệ phương sai giải thích được
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)

        return self

    def transform(self, X):
        """
        Áp dụng giảm chiều lên dữ liệu X

        Parameters:
        -----------
        X : numpy.ndarray hoặc pandas.DataFrame
            Dữ liệu đầu vào có shape (n_samples, n_features)

        Returns:
        --------
        X_transformed : numpy.ndarray
            Dữ liệu đã được giảm chiều với shape (n_samples, n_components)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Chuẩn hóa dữ liệu
        X_centered = X - self.mean

        # Chiếu dữ liệu lên các thành phần chính
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed

    def fit_transform(self, X):
        """
        Thực hiện fit và transform trên cùng một dữ liệu

        Parameters:
        -----------
        X : numpy.ndarray hoặc pandas.DataFrame
            Dữ liệu đầu vào có shape (n_samples, n_features)

        Returns:
        --------
        X_transformed : numpy.ndarray
            Dữ liệu đã được giảm chiều với shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Chuyển đổi ngược từ dữ liệu đã giảm chiều về dữ liệu ban đầu (xấp xỉ)

        Parameters:
        -----------
        X_transformed : numpy.ndarray
            Dữ liệu đã giảm chiều với shape (n_samples, n_components)

        Returns:
        --------
        X_reconstructed : numpy.ndarray
            Dữ liệu được khôi phục với shape (n_samples, n_features)
        """
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        return X_reconstructed

    def visualize_3d(self, X, y, title='PCA Visualization in 3D'):
        """
        Trực quan hóa dữ liệu 3D (sau khi giảm chiều) với màu sắc dựa trên giá trị y

        Parameters:
        -----------
        X : numpy.ndarray hoặc pandas.DataFrame
            Dữ liệu đầu vào có shape (n_samples, n_features)
        y : numpy.ndarray hoặc pandas.Series
            Nhãn hoặc giá trị mục tiêu cho mỗi mẫu
        title : str, default='PCA Visualization in 3D'
            Tiêu đề của biểu đồ

        Returns:
        --------
        fig, ax : matplotlib Figure and Axes
            Đối tượng Figure và Axes3D của matplotlib để có thể tùy chỉnh thêm
        """
        # Đảm bảo số thành phần là 3 cho trực quan hóa 3D
        if self.n_components < 3:
            raise ValueError("Cần ít nhất 3 thành phần để trực quan hóa 3D")

        # Chuyển đổi DataFrame/Series thành ndarray nếu cần
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel()

        # Áp dụng PCA transformation
        X_pca = self.transform(X)

        # Tạo biểu đồ 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Tạo map màu từ xanh đến đỏ dựa trên giá trị y
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=y, cmap='cool', s=50, alpha=0.8
        )

        # Thêm thanh màu để hiển thị mối quan hệ giữa màu và giá trị y
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Giá trị y')

        # Thêm nhãn và tiêu đề
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%} variance)')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio[2]:.2%} variance)')
        ax.set_title(title)

        # Thêm thông tin về tổng phương sai giải thích được
        total_var = np.sum(self.explained_variance_ratio[:3])
        plt.suptitle(f'Tổng phương sai giải thích được: {total_var:.2%}', y=0.92)

        plt.tight_layout()
        return fig, ax