# MATHS-FOR-AI

## Thư viện sử dụng:

- numpy, pandas, matplotlib, seaborn, pickle

## Hướng dẫn sử dụng

- Toàn bộ nội dung chạy từ trên xuống được viết trong file "Lab.ipynb"

## Cấu trúc thư mục

```markdown
MATHS-FOR-AI/
├── Data/
│ └── train.csv
├── EDA/
│ ├── DataPreprocessing/
│ │ └── PreprocessingData.ipynb # Giả định là file Jupyter Notebook
│ └── FeatureSelection/
│ ├── FeaturesAnalysis.ipynb
│ └── FeatureSelection.ipynb # Tên file đầy đủ từ hình ảnh
├── LinearModel/
│ ├── **init**.py
│ ├── LinearModel.py
│ ├── Model.py
│ ├── Optimizer.py
│ ├── Scaler.py # Có vẻ là lớp cơ sở hoặc interface
│ └── StandardScaler.py # Triển khai cụ thể
├── models/ # Thư mục lưu trữ khác, có thể là instance model hoặc cấu hình
│ ├── **init**.py
│ └── ModelSave/ # Chứa các model và scaler đã lưu
│ ├── linear*regression_model*... # Nhiều file model .pkl hoặc định dạng khác
│ ├── scaler_1.pkl
│ ├── scaler_2.pkl
│ ├── scaler_3.pkl
│ └── scaler_4.pkl
├── PreProcessing/ # Module tiền xử lý riêng biệt
│ ├── **init**.py
│ ├── FeatureSelector.py
│ ├── GUILDLINE.md # File hướng dẫn
│ ├── README.md # README riêng cho tiền xử lý
│ └── VehicleDataPreprocessor.py # File xử lý dữ liệu xe
├── Test/ # Thư mục chứa các bài test
│ ├── **init**.py
│ ├── test_FeatureSelector.py
│ ├── test_Model.py # Test cho module Model.py
│ └── test_VehicleDataProcessor.py # Test cho VehicleDataPreprocessor.py
├── Utils/ # Thư mục chứa các hàm tiện ích
│ ├── **init**.py
│ ├── data_preprocessing.py # Có thể chứa các hàm tiền xử lý chung
│ ├── Metrics.py # Chứa các hàm tính toán chỉ số đánh giá
│ └── visualization.py # Chứa các hàm vẽ biểu đồ chung
├── Visualization/ # Thư mục chứa các output trực quan hóa
│ ├── analysis/ # Output phân tích
│ └── train/ # Output trong quá trình huấn luyện
├── .gitignore # File cấu hình Git
├── Lab.ipynb # File notebook chính
└── README.md # File README chính của dự án
```
