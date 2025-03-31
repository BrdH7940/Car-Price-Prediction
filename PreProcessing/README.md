## 📋 Tổng Quan Cấu Trúc Class

Class tiền xử lý của chúng ta sẽ trông như thế này:

```python
import pandas as pd
import numpy as np
import pickle

class VehicleDataPreprocessor:
    def __init__(self, make_encoder_path=None):
        # Tải Make encoder nếu được cung cấp
        self.make_encoder = None
        if make_encoder_path:
            try:
                with open(make_encoder_path, 'rb') as f:
                    self.make_encoder = pickle.load(f)
            except Exception as e:
                print(f"Cảnh báo: Không thể tải make encoder: {e}")
                
        # Giá trị toàn cục cho việc điền giá trị còn thiếu (quan trọng cho tập TEST)
        self.global_means = {
            'Max_Power_Value': 117.0,
            'Max_Power_RPM': 4200.0,
            'Max_Torque_Value': 200.0,
            'Max_Torque_RPM': 1900.0,
            'Engine_Value': 1498.0,
            'Width': 1775.0,
            'Kilometer_Log': 10.819798284210286,
            'Length': 4370.0,
            'Height': 1550.0,
            'Fuel Tank Capacity': 50.0,
            'Make_encoded': 13.81  # Giá trị trung bình toàn cục cho mã hóa Make
        }
        
    def preprocess(self, df):
        """Pipeline tiền xử lý chính gọi tất cả các phương thức tiền xử lý riêng lẻ"""
        df = df.copy()  # Không sửa đổi dataframe gốc
        
        # Xử lý từng cột
        df = self._process_price(df)
        df = self._process_engine(df)
        df = self._process_max_power(df)
        df = self._process_max_torque(df)
        df = self._process_width(df)
        df = self._process_kilometer(df)
        df = self._process_make(df)
        df = self._process_fuel_type(df)
        df = self._process_transmission(df)
        df = self._process_color(df)
        df = self._process_owner(df)
        df = self._process_seller_type(df)
        df = self._process_seating_capacity(df)
        df = self._process_remaining_numeric_cols(df)
        
        return df
    
    # Các phương thức xử lý cột riêng lẻ sẽ đặt ở đây...
```

Bây giờ hãy đi sâu vào từng phương thức xử lý cột cụ thể!

## 🧩 Các Phương Thức Xử Lý Cột Riêng Lẻ

### 1️⃣ Cột Price (Giá)

```python
def _process_price(self, df):
    """Xử lý cột Price: chuyển đổi log và loại bỏ cột gốc"""
    if 'Price' in df.columns:
        df['Log_Price'] = np.log(df['Price'])
        df.drop('Price', axis=1, inplace=True)
    return df
```

### 2️⃣ Cột Engine (Động Cơ)
```python
def _process_engine(self, df):
    """Xử lý cột Engine: trích xuất giá trị số, xử lý giá trị thiếu và ngoại lai"""
    if 'Engine' in df.columns:
        # Trích xuất giá trị số từ Engine (loại bỏ 'cc')
        df['Engine_Value'] = pd.to_numeric(df['Engine'].str.replace('cc', ''), errors='coerce')
        
        # CHÚ Ý: Điền giá trị NA cho tập TEST với giá trị 1498.0
        df['Engine_Value'].fillna(self.global_means['Engine_Value'], inplace=True)
        
        # Xử lý ngoại lai bằng chuyển đổi log
        df['Engine_Value_Log'] = np.log(df['Engine_Value'])
        
        # Định nghĩa giới hạn cho xử lý ngoại lai (sử dụng các giá trị cố định)
        # Lưu ý: Sử dụng giá trị từ hướng dẫn gốc để đảm bảo nhất quán train-test
        # Q1 = 7.29  # Sử dụng giá trị cố định từ hướng dẫn
        # Q3 = 7.6   # Sử dụng giá trị cố định từ hướng dẫn
        # IQR = 0.31  # Sử dụng giá trị cố định từ hướng dẫn
        lower_bound = 6.83  # Sử dụng giá trị cố định từ hướng dẫn
        upper_bound = 8.06  # Sử dụng giá trị cố định từ hướng dẫn
        
        # Clip giá trị để xử lý ngoại lai
        df['Engine_Value_Log'] = df['Engine_Value_Log'].clip(lower=lower_bound, upper=upper_bound)
        
        # Chuyển đổi trở lại thang đo ban đầu
        df['Engine_Value'] = np.exp(df['Engine_Value_Log'])
        
        # Loại bỏ cột trung gian và cột gốc
        df.drop(['Engine_Value_Log', 'Engine'], axis=1, inplace=True)
    
    return df
```

### 3️⃣ Cột Max Power (Công Suất Tối Đa)
```python
def _process_max_power(self, df):
    """Xử lý cột Max Power: tách thành giá trị và RPM, xử lý giá trị thiếu và ngoại lai"""
    if 'Max Power' in df.columns:
        # Tách Max Power thành giá trị và RPM
        df[['Max_Power_Value', 'Max_Power_RPM']] = df['Max Power'].str.split('@', expand=True)
        
        # Làm sạch giá trị RPM
        df['Max_Power_RPM'] = df['Max_Power_RPM'].str.strip().str.replace('rpm', '', regex=False)
        
        # Làm sạch giá trị Value (chỉ giữ số và dấu thập phân)
        df['Max_Power_Value'] = df['Max_Power_Value'].str.replace(r'[^\d.]+', '', regex=True)
        
        # Chuyển đổi sang dạng số, xử lý lỗi
        df['Max_Power_RPM'] = pd.to_numeric(df['Max_Power_RPM'], errors='coerce')
        df['Max_Power_Value'] = pd.to_numeric(df['Max_Power_Value'], errors='coerce')
        
        # CHÚ Ý: Điền giá trị NA cho tập TEST
        # Fill Max_Power_Value = 117.0 nếu nan
        # Fill Max_Power_RPM = 4200
        df['Max_Power_Value'].fillna(self.global_means['Max_Power_Value'], inplace=True)
        df['Max_Power_RPM'].fillna(self.global_means['Max_Power_RPM'], inplace=True)
        
        # Xử lý ngoại lai cho Max_Power_Value
        df['Max_Power_Value_Log'] = np.log(df['Max_Power_Value'])
        
        # Sử dụng giá trị cố định từ hướng dẫn gốc để đảm bảo nhất quán train-test
        lower_bound = 3.35
        upper_bound = 6.20
        
        df['Max_Power_Value_Log'] = df['Max_Power_Value_Log'].clip(lower=lower_bound, upper=upper_bound)
        df['Max_Power_Value'] = np.exp(df['Max_Power_Value_Log'])
        
        # Loại bỏ cột trung gian và cột gốc
        df.drop(['Max_Power_Value_Log', 'Max Power'], axis=1, inplace=True)
    
    return df
```

### 4️⃣ Cột Max Torque (Mô-men Xoắn Tối Đa)
```python
def _process_max_torque(self, df):
    """Xử lý cột Max Torque: tách thành giá trị và RPM, xử lý giá trị thiếu và ngoại lai"""
    if 'Max Torque' in df.columns:
        # Tách Max Torque thành giá trị và RPM
        df[['Max_Torque_Value', 'Max_Torque_RPM']] = df['Max Torque'].str.split('@', expand=True)
        
        # Làm sạch giá trị RPM
        df['Max_Torque_RPM'] = df['Max_Torque_RPM'].str.strip().str.replace('rpm', '', regex=False)
        
        # Làm sạch giá trị Value (chỉ giữ số và dấu thập phân)
        df['Max_Torque_Value'] = df['Max_Torque_Value'].str.replace(r'[^\d.]+', '', regex=True)
        
        # Chuyển đổi sang dạng số, xử lý lỗi
        df['Max_Torque_RPM'] = pd.to_numeric(df['Max_Torque_RPM'], errors='coerce')
        df['Max_Torque_Value'] = pd.to_numeric(df['Max_Torque_Value'], errors='coerce')
        
        # CHÚ Ý: Điền giá trị NA cho tập TEST
        # Fill Max_Torque_Value = 200.0
        # Fill Max_Torque_RPM = 1900.0
        df['Max_Torque_Value'].fillna(self.global_means['Max_Torque_Value'], inplace=True)
        df['Max_Torque_RPM'].fillna(self.global_means['Max_Torque_RPM'], inplace=True)
        
        # Xử lý ngoại lai cho Max_Torque_Value
        # Sử dụng giá trị cố định từ hướng dẫn gốc
        lower_bound = -227.00  # Lưu ý: con số này âm trong hướng dẫn gốc nhưng có thể bạn muốn sử dụng 0
        upper_bound = 685.00
        
        df['Max_Torque_Value'] = df['Max_Torque_Value'].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        # Loại bỏ cột gốc
        df.drop('Max Torque', axis=1, inplace=True)
    
    return df
```

### 5️⃣ Cột Width (Chiều Rộng)
```python
def _process_width(self, df):
    """Xử lý cột Width: xử lý giá trị thiếu và ngoại lai"""
    if 'Width' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = 1775.0
        df['Width'].fillna(self.global_means['Width'], inplace=True)
        
        # Xử lý ngoại lai với các giá trị được chỉ định cố định
        lower_bound = 1491.00
        upper_bound = 2035.00
        
        df['Width'] = df['Width'].clip(lower=lower_bound, upper=upper_bound)
    
    return df
```

### 6️⃣ Cột Kilometer (Số Km)
```python
def _process_kilometer(self, df):
    """Xử lý cột Kilometer: chuyển đổi log và xử lý ngoại lai"""
    if 'Kilometer' in df.columns:
        # Chuyển đổi log (log1p = log(1+x))
        df['Kilometer_Log'] = np.log1p(df['Kilometer'])
        
        # CHÚ Ý: Điền giá trị NA cho tập TEST = 10.819798284210286
        df['Kilometer_Log'].fillna(self.global_means['Kilometer_Log'], inplace=True)
        
        # Xử lý ngoại lai với các giá trị đã được chỉ định cố định
        lower_bound = 8.91
        upper_bound = 12.55
        
        df['Kilometer_Log'] = df['Kilometer_Log'].clip(lower=lower_bound, upper=upper_bound)
        
        # Chuyển đổi trở lại thang đo ban đầu
        df['Kilometer'] = np.expm1(df['Kilometer_Log'])
        
        # Loại bỏ cột trung gian
        df.drop('Kilometer_Log', axis=1, inplace=True)
    
    return df
```

### 7️⃣ Các Cột Số Khác
```python
def _process_remaining_numeric_cols(self, df):
    """Xử lý các cột số còn lại: Length, Height, Fuel Tank Capacity"""
    # CHÚ Ý: Điền giá trị NA cho tập TEST với các giá trị cố định
    # Fill 'Length' = 4370.00
    # Fill 'Height' = 1550.00
    # Fill 'Fuel Tank Capacity' = 50.00
    numeric_cols = ['Length', 'Height', 'Fuel Tank Capacity']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(self.global_means[col], inplace=True)
    
    return df
```

### 8️⃣ Cột Make (Hãng Xe)
```python
def _process_make(self, df):
    """Xử lý cột Make: áp dụng mã hóa target"""
    if 'Make' in df.columns:
        # Nếu không có encoder từ file pkl, sử dụng dictionary
        if self.make_encoder is None:
            self.make_encoder = {
                'Maruti Suzuki': 13.0876568212651,
                'Hyundai': 13.332835658211609,
                'Mercedes-Benz': 14.979287245081439,
                'Honda': 13.247622182348156,
                'Toyota': 14.103649799670542,
                'Audi': 14.645617786857711,
                'Mahindra': 13.7548285741869,
                'BMW': 14.970197610360676,
                'Tata': 13.539726337744348,
                'Ford': 13.701192212433263,
                'Renault': 13.085894547754995,
                'Volkswagen': 13.36190214923346,
                'Skoda': 13.940173966008981,
                'Land Rover': 15.278102064185989,
                'Jaguar': 14.766056716400424,
                'Jeep': 14.252383454646322,
                'Volvo': 14.426292544644863,
                'Kia': 14.082939331174831,
                'MG': 14.235879202727373,
                'Nissan': 13.343536423603807,
                'MINI': 14.516270808655126,
                'Porsche': 15.073386981633433,
                'Datsun': 13.076001247571691,
                'Chevrolet': 13.244374287794319,
                'Lexus': 14.414617681160243,
                'Ssangyong': 13.744651156585752,
                'Mitsubishi': 13.977824558629855,
                'Rolls-Royce': 14.920345808680235,
                'Fiat': 13.462345798374823,
                'Lamborghini': 14.337993813142857,
                'Isuzu': 13.905447195264628
            }
            
        # Tạo cột mã hóa
        df['Make_encoded'] = df['Make'].map(self.make_encoder)
        
        # CHÚ Ý: Điền giá trị NA cho tập TEST = 13.81 (Global_mean)
        df['Make_encoded'].fillna(self.global_means['Make_encoded'], inplace=True)
        
        # Loại bỏ cột gốc
        df.drop('Make', axis=1, inplace=True)
    
    return df
```

### 9️⃣ Cột Fuel Type (Loại Nhiên Liệu)
```python
def _process_fuel_type(self, df):
    """Xử lý cột Fuel Type: nhóm các danh mục và mã hóa one-hot"""
    if 'Fuel Type' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = "Diesel"
        df['Fuel Type'].fillna('Diesel', inplace=True)
        
        # Nhóm loại nhiên liệu sử dụng hàm
        def group_fuel_types(fuel_type):
            if fuel_type in ['Diesel']:
                return 'Diesel'
            elif fuel_type in ['Petrol']:
                return 'Petrol'
            elif fuel_type in ['CNG', 'CNG + CNG']:
                return 'CNG'
            else:
                return 'Others'  # Electric, Hybrid, LPG, Petrol + LPG
        
        df['Fuel_Type_Grouped'] = df['Fuel Type'].apply(group_fuel_types)
        
        # Mã hóa one-hot
        fuel_type_dummies = pd.get_dummies(df['Fuel_Type_Grouped'], prefix='Fuel')
        df = pd.concat([df, fuel_type_dummies], axis=1)
        
        # Loại bỏ các cột gốc và nhóm
        if 'Fuel_Diesel' in df.columns:
            df.drop(['Fuel Type', 'Fuel_Type_Grouped', 'Fuel_Diesel'], axis=1, inplace=True)
        else:
            df.drop(['Fuel Type', 'Fuel_Type_Grouped'], axis=1, inplace=True)
    
    return df
```

### 🔟 Cột Transmission (Hộp Số)
```python
def _process_transmission(self, df):
    """Xử lý cột Transmission: chuyển đổi thành tính năng nhị phân"""
    if 'Transmission' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = "Manual"
        df['Transmission'].fillna('Manual', inplace=True)
        
        # Chuyển đổi thành tính năng nhị phân
        df['Transmission_is_Automatic'] = df['Transmission'].apply(lambda x: 1 if x == 'Automatic' else 0)
        
        # Loại bỏ cột gốc
        df.drop('Transmission', axis=1, inplace=True)
    
    return df
```

### 1️⃣1️⃣ Cột Color (Màu Sắc)
```python
def _process_color(self, df):
    """Xử lý cột Color: nhóm màu sắc và mã hóa one-hot"""
    if 'Color' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = "White"
        df['Color'].fillna('White', inplace=True)
        
        # Sử dụng hàm nhóm màu sắc
        def group_colors(color):
            # Premium Colors: Blue, Black (giá cao, tần suất cao)
            if color in ['Blue', 'Black']:
                return 'Premium'
            # White: giữ riêng vì là màu phổ biến nhất
            elif color == 'White':
                return 'White'
            # Standard Colors: Grey, Red (giá trung bình, tần suất cao)
            elif color in ['Grey', 'Red']:
                return 'Standard'
            # Silver: giữ riêng vì có tần suất cao nhưng giá thấp
            elif color == 'Silver':
                return 'Silver'
            # Medium Colors: Brown, Maroon, Bronze, Gold, Green (tần suất trung bình)
            elif color in ['Brown', 'Maroon', 'Bronze', 'Gold', 'Green']:
                return 'Medium'
            # Rare Colors: Beige, Yellow, Orange, Purple, Pink, Others (tần suất thấp)
            else:
                return 'Rare'
                
        # Áp dụng hàm nhóm
        df['Color_Group'] = df['Color'].apply(group_colors)
        
        # Mã hóa one-hot
        color_group_dummies = pd.get_dummies(df['Color_Group'], prefix='Color')
        df = pd.concat([df, color_group_dummies], axis=1)
        
        # Loại bỏ cột gốc, nhóm và một danh mục tham chiếu
        if 'Color_White' in df.columns:
            df.drop(['Color', 'Color_Group', 'Color_White'], axis=1, inplace=True)
        else:
            df.drop(['Color', 'Color_Group'], axis=1, inplace=True)
    
    return df
```

### 1️⃣2️⃣ Cột Owner (Chủ Sở Hữu)
```python
def _process_owner(self, df):
    """Xử lý cột Owner: nhóm chủ sở hữu và mã hóa one-hot"""
    if 'Owner' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = "First"
        df['Owner'].fillna('First', inplace=True)
        
        # Sử dụng hàm phân nhóm chủ sở hữu
        def group_owners(owner):
            if owner == 'UnRegistered Car':
                return 'New'
            elif owner == 'First':
                return 'First_Owner'
            elif owner == 'Second':
                return 'Second_Owner'
            elif owner == 'Third':
                return 'Third_Owner'
            else:  # Fourth, 4 or More
                return 'Fourth_Plus_Owner'
        
        # Áp dụng hàm phân nhóm
        df['Owner_Group'] = df['Owner'].apply(group_owners)
        
        # Mã hóa one-hot
        owner_group_dummies = pd.get_dummies(df['Owner_Group'], prefix='Owner')
        df = pd.concat([df, owner_group_dummies], axis=1)
        
        # Loại bỏ cột gốc và nhóm
        df.drop(['Owner', 'Owner_Group'], axis=1, inplace=True)
    
    return df
```

### 1️⃣3️⃣ Cột Seller Type (Loại Người Bán)
```python
def _process_seller_type(self, df):
    """Xử lý cột Seller Type: mã hóa one-hot"""
    if 'Seller Type' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = "Individual"
        df['Seller Type'].fillna('Individual', inplace=True)
        
        # Mã hóa one-hot
        seller_type_dummies = pd.get_dummies(df['Seller Type'], prefix='Seller')
        df = pd.concat([df, seller_type_dummies], axis=1)
        
        # Loại bỏ cột gốc và danh mục tham chiếu
        if 'Seller_Individual' in df.columns:
            df.drop(['Seller Type', 'Seller_Individual'], axis=1, inplace=True)
        else:
            df.drop(['Seller Type'], axis=1, inplace=True)
    
    return df
```

### 1️⃣4️⃣ Cột Seating Capacity (Số Chỗ Ngồi)
```python
def _process_seating_capacity(self, df):
    """Xử lý cột Seating Capacity: xử lý giá trị thiếu và mã hóa one-hot"""
    if 'Seating Capacity' in df.columns:
        # CHÚ Ý: Điền giá trị NA cho tập TEST = 5
        df['Seating Capacity'].fillna(5, inplace=True)
        
        # Chuyển đổi thành số nguyên
        df['Seating Capacity'] = df['Seating Capacity'].astype(int)
        
        # Mã hóa one-hot
        seating_capacity_dummies = pd.get_dummies(df['Seating Capacity'], prefix='Seating')
        df = pd.concat([df, seating_capacity_dummies], axis=1)
        
        # Loại bỏ cột gốc và danh mục tham chiếu
        if 'Seating_5' in df.columns:
            df.drop(['Seating Capacity', 'Seating_5'], axis=1, inplace=True)
        else:
            df.drop(['Seating Capacity'], axis=1, inplace=True)
    
    return df
```

## 📝 Bảng Tổng Hợp Xử Lý NA cho Tập TEST
Nếu file pkl bị lỗi đây là bảng chứa các thông tin quan trọng

| Cột                     | Giá Trị Điền NA          |
|------------------------|--------------------------|
| Engine_Value            | 1498.0                   |
| Max_Power_Value         | 117.0                    |
| Max_Power_RPM           | 4200.0                   |
| Max_Torque_Value        | 200.0                    |
| Max_Torque_RPM          | 1900.0                   |
| Width                   | 1775.0                   |
| Kilometer_Log           | 10.819798284210286       |
| Length                  | 4370.0                   |
| Height                  | 1550.0                   |
| Fuel Tank Capacity      | 50.0                     |
| Make_encoded            | 13.81                    |
| Fuel Type               | "Diesel"                 |
| Transmission            | "Manual"                 |
| Color                   | "White"                  |
| Owner                   | "First"                  |
| Seller Type             | "Individual"             |
| Seating Capacity        | 5                        |

## 🧪 Usage (Cách sử dụng)
```python
# Khởi tạo bộ tiền xử lý
preprocessor = VehicleDataPreprocessor(make_encoder_path='path/to/make_encoder.pkl')

# Tải dữ liệu của bạn
df_train = pd.read_csv('your_training_data.csv')
df_test = pd.read_csv('your_test_data.csv')

# Tiền xử lý cả hai bộ dữ liệu
df_train_processed = preprocessor.preprocess(df_train)
df_test_processed = preprocessor.preprocess(df_test)

# Bây giờ dữ liệu của bạn đã sẵn sàng cho việc mô hình hóa!
print(f"Kích thước dữ liệu huấn luyện: {df_train_processed.shape}")
print(f"Kích thước dữ liệu kiểm tra: {df_test_processed.shape}")
```
