import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from Preprocessing.VehicleDataPreprocessor import VehicleDataPreprocessor

# Khởi tạo bộ tiền xử lý
preprocessor = VehicleDataPreprocessor()

# Tải dữ liệu của bạn
df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('your_test_data.csv')

# Tiền xử lý cả hai bộ dữ liệu
df_train_processed = preprocessor.preprocess(df_train)
# df_test_processed = preprocessor.preprocess(df_test)

# Bây giờ dữ liệu của bạn đã sẵn sàng cho việc mô hình hóa!
print(f"Kích thước dữ liệu huấn luyện: {df_train_processed.shape}")
print(df_train_processed.to_csv("Final_train.csv", index=False))
# print(f"Kích thước dữ liệu kiểm tra: {df_test_processed.shape}")