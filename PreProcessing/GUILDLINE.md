# 🚗 Tiền Xử Lý Dữ Liệu Xe 🚙

## Ý Tưởng

Xây dựng một class `VehicleDataPreprocessor` trong Python có chức năng xử lý từng cột dữ liệu theo các yêu cầu cụ thể. Class này cần đảm bảo việc xử lý nhất quán giữa tập train và test.

## Các Phương Pháp Xử Lý Cột

### 1. Price
- Log cột Price thành cột Log_Price
- Drop cột Price gốc

### 2. Engine
- Xóa bỏ "cc" trong Engine và chuyển thành số: 
  ```
  df['Engine_Value'] = pd.to_numeric(df['Engine'].str.replace('cc', ''), errors='coerce')
  ```
- Fill NaN = 1498.0
- Xử lý outlier qua log transform:
  ```
  df['Engine_Value_Log'] = np.log(df['Engine_Value'])
  ```
- Clip với:
  - Q1 (25%): 7.29
  - Q3 (75%): 7.6
  - IQR: 0.31
  - Lower bound: 6.83
  - Upper bound: 8.06
- Chuyển ngược lại: `np.exp(df['Engine_Value_Log'])`
- Drop cột Engine gốc

### 3. Max Power
- Chia làm 2 cột khác nhau:
  ```
  df[['Max_Power_Value', 'Max_Power_RPM']] = df['Max Power'].str.split('@', expand=True)
  df['Max_Power_RPM'] = df['Max_Power_RPM'].str.strip().str.replace('rpm', '', regex=False)
  df['Max_Power_Value'] = df['Max_Power_Value'].str.replace(r'[^\d.]+', '', regex=True)
  ```
- Chuyển thành dạng số và xử lý NaN:
  - Fill Max_Power_Value = 117.0 nếu nan
  - Fill Max_Power_RPM = 4200.0

#### 3.1. Cột Max_Power_Value
- Log cột này
- Xử lý outliers:
  - Q1 (25%): 4.42
  - Q3 (75%): 5.13
  - IQR: 0.71
  - Lower bound: 3.35
  - Upper bound: 6.20
- Clip lower-upper
- Chuyển về giá trị gốc: `np.exp(df['Max_Power_Value_Log'])`
- Drop cột Max Power gốc

### 4. Max Torque
- Split thành 2 cột:
  ```
  df[['Max_Torque_Value', 'Max_Torque_RPM']] = df['Max Torque'].str.split('@', expand=True)
  df['Max_Torque_RPM'] = df['Max_Torque_RPM'].str.strip().str.replace('rpm', '', regex=False)
  df['Max_Torque_Value'] = df['Max_Torque_Value'].str.replace(r'[^\d.]+', '', regex=True)
  ```
- Chuyển thành dạng số và xử lý NaN:
  - Fill Max_Torque_Value = 200.0
  - Fill Max_Torque_RPM = 1900.0

#### 4.1. Cột Max_Torque_Value
- Xử lý outliers:
  - Q1 (25%): 115.00
  - Q3 (75%): 343.00
  - IQR: 228.00
  - Lower bound: -227.00 (nên dùng 0)
  - Upper bound: 685.00
- Clip giá trị
- Drop cột Max Torque gốc

### 5. Cột Width
- Fill NaN = 1775.0
- Xử lý outliers:
  - Q1 (25%): 1695.00
  - Q3 (75%): 1831.00
  - IQR: 136.00
  - Lower bound: 1491.00
  - Upper bound: 2035.00
- Clip giá trị

### 6. Kilometer
- Log1p transformation: `np.log1p(df['Kilometer'])`
- Xử lý outliers:
  - Q1 (25%): 10.28
  - Q3 (75%): 11.18
  - IQR: 0.91
  - Lower bound: 8.91
  - Upper bound: 12.55
- Fill NaN: 10.819798284210286
- Chuyển lại về cột gốc: `np.expm1(df['Kilometer_Log'])`
- Drop cột Kilometer_Log

### 7. Các cột số khác
- Columns to impute with median:
  - Fill 'Length' = 4370.00
  - Fill 'Height' = 1550.00
  - Fill 'Fuel Tank Capacity' = 50.00

### 8. Make
- Sử dụng smoothed target encoding với k=5
- Global_mean = 13.81 (Nếu test NA thì fill cái này)
- Dictionary chi tiết giá trị encoding:
  ```
  make_encoder = {
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
  ```
- Lưu thành cột Make_encoded và drop cột Make

### 9. Fuel Type
- Nếu test có NA, fill = "Diesel"
- Group theo hàm sau:
  ```python
  def group_fuel_types(fuel_type):
      if fuel_type in ['Diesel']:
          return 'Diesel'
      elif fuel_type in ['Petrol']:
          return 'Petrol'
      elif fuel_type in ['CNG', 'CNG + CNG']:
          return 'CNG'
      else:
          return 'Others'  # Electric, Hybrid, LPG, Petrol + LPG
  ```
- One-hot encoding cho Fuel_Type_Grouped
- Drop cột 'Fuel Type', 'Fuel_Type_Grouped', và 'Fuel_Diesel'

### 10. Transmission
- Nếu test có NA, fill = "Manual"
- Chuyển thành biến nhị phân:
  ```python
  df['Transmission_is_Automatic'] = df['Transmission'].apply(lambda x: 1 if x == 'Automatic' else 0)
  ```
- Drop cột Transmission

### 11. Color
- Nếu test có NA, fill = "White"
- Group theo hàm sau:
  ```python
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
  ```
- One-hot encoding cho Color_Group
- Drop cột 'Color', 'Color_Group', và 'Color_White'

### 12. Owner
- Nếu test có NA, fill = "First"
- Group theo hàm sau:
  ```python
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
  ```
- One-hot encoding cho Owner_Group
- Drop cột 'Owner' và 'Owner_Group'

### 13. Seller Type
- Nếu test có NA, fill = "Individual"
- One-hot encoding cho Seller Type
- Drop cột 'Seller Type' và 'Seller_Individual'

### 14. Seating Capacity
- Fill NA = 5
- Chuyển từ float64 -> int64
- One-hot encoding
- Drop cột 'Seating Capacity' và 'Seating_5'

## 📝 Bảng Tổng Hợp Xử Lý NA cho Tập TEST

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
