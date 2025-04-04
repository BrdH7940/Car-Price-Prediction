import numpy as np
import pandas as pd

class FeatureSelector(object):
    features = {
        0: [
            'Log_Max_Power_Value',
            'Age_Log_Kilometer',
            'Make_encoded',
            'Fuel Tank Capacity',
            'Length_Width_interaction',
            'Max_Power_Value',
            'Age_Volume',
            'Age_Height',
            'Engine_Value',
            'Age_Kilometer',
            'Power_to_CC',
            'Age',
            "Log_Kilometer",
            'Log_Kilometer_Length',
            'Log_Kilometer_Power',

            'Fuel_Others',
            'Fuel_Petrol',

            'Owner_Fourth_Plus_Owner',

            'Seating_2',
            'Seating_4',
        ],
        1: [
            # Numerical
            # # Features cơ bản
            # "Year",
            # 'Sin_Year',
            # 'Year_Inverse',
            # "Age",
            "Log_Age",
            # 'Sin_Age',
            "Age_squared",
            "Age_Log_Kilometer",
            'Age_Length',
            # 'Age_Height',
            # "Kilometer",
            "Log_Kilometer",
            'Log_Kilometer_Length',
            'Log_Kilometer_Power',
            # 'Log_Kilometer_Torque',

            "Make_encoded",
            # 'Drivetrain_encoded',

            # # Features quan trọng về thông số kỹ thuật
            'Log_Max_Power_Value',
            # 'Max_Power_Value',
            # 'Max_Power_RPM',
            'Log_Max_Torque_Value',
            # 'Max_Torque_Value',
            "Fuel Tank Capacity",

            # # Features tính toán
            # 'Efficiency',
            # 'Power_to_CC',

            # # Features kích thước
            "Length",
            'Volume',
            # 'Length_Width_interaction',

            # # Các features kết hợp
            'Age_Volume',
            # 'Age_Kilometer',
            'Kilometer_per_Year',
            # 'Power_to_Torque',
            # 'RPM_Ratio',

            # Categorical
            'Fuel_CNG',
            'Fuel_Others',
            'Fuel_Petrol',

            'Transmission_is_Automatic',

            # 'Owner_First_Owner',
            # 'Owner_Fourth_Plus_Owner',
            # 'Owner_New',
            # 'Owner_Second_Owner',
            # 'Owner_Third_Owner',

            'Seating_2',
            'Seating_4',
            # 'Seating_6',
            # 'Seating_7',
            # 'Seating_8'
        ],
        2: [
            # Numerical
            "Year",
            "Kilometer",
            "Width",
            "Length",
            "Fuel Tank Capacity",
            "Engine_Value",
            'Max_Power_Value',
            "Max_Power_RPM",
            "Max_Torque_Value",
            "Max_Torque_RPM",
            "Make_encoded",
            "Age_Kilometer",
            "Kilometer_per_Year",

            # Categorical
            "Color_Silver",

            'Seating_2',

            "Transmission_is_Automatic",

            "Seller_Commercial Registration",

            'Fuel_CNG',
            'Fuel_Others',
            'Fuel_Petrol',

            'Owner_First_Owner',
            'Owner_Fourth_Plus_Owner',
            'Owner_New',
            'Owner_Second_Owner',
            'Owner_Third_Owner',
        ],
        3: [
            # Numerical
            'Length',
            'Year',
            'Make_encoded',
            'Max_Power_Value',
            'Width',
            'Power_to_CC',
            'Fuel_Petrol',
            'Max_Torque_Value',
            'Power_to_Torque',
            'Kilometer',

            # Categorical
            'Transmission_is_Automatic',

            'Fuel_CNG',
            'Fuel_Others',
            'Fuel_Petrol',

            'Seating_2',
            'Seating_4',
            'Seating_6',
            'Seating_7',
            'Seating_8'
        ],
        4: [
            # Numerical
            # # Features cơ bản
            'Year',
            'Kilometer',
            'Make_encoded',

            # # Features quan trọng về thông số kỹ thuật
            'Max_Power_Value', 'Engine_Value', 'Fuel Tank Capacity',

            # # Features tính toán
            'Efficiency', 'Power_to_CC',

            # Categorical
            'Fuel_CNG',
            'Fuel_Others',
            'Fuel_Petrol',

            'Owner_First_Owner',
            'Owner_Fourth_Plus_Owner',
            'Owner_New',
            'Owner_Second_Owner',
            'Owner_Third_Owner',

            'Seating_2',
            'Seating_4',
            'Seating_6',
            'Seating_7',
            'Seating_8',

            'Transmission_is_Automatic'
        ]
    }
    def __init__(self):
        pass

    @classmethod
    def _non_linearize_features(cls, df):
        df["Volume"] = df["Height"] * df["Width"] * df["Length"]
        df['Power_to_Weight'] = df['Max_Power_Value'] / df['Volume']
        df['Efficiency'] = df['Max_Power_Value'] / df['Fuel Tank Capacity']
        df['Log_Kilometer'] = np.log(df['Kilometer'])
        df['Log_Max_Power_Value'] = np.log(df['Max_Power_Value'])
        df['Log_Max_Torque_Value'] = np.log(df['Max_Torque_Value'])

        # Tương tác RPM
        df['Power_Density'] = df['Max_Power_Value'] / df['Max_Power_RPM']
        df['Torque_Density'] = df['Max_Torque_Value'] / df['Max_Torque_RPM']

        # Tỷ lệ Power/Torque
        df['Power_Torque_Ratio'] = df['Max_Power_Value'] / df['Max_Torque_Value']
        df['Length_Width_interaction'] = df['Length'] * df['Width']

        # Tương tác Year
        df['Sin_Year'] = np.sin(df['Year'])
        df['Year_Inverse'] = 1 / df['Year']
        df['Age'] = 2025 - df['Year']
        df['Age_squared'] = df['Age'] * df['Age']
        df['Log_Age'] = np.log(df['Age'])
        df['Sin_Age'] = np.sin(df['Age'])
        df['Age_Length'] = df['Age'] * df['Length']
        df['Age_Height'] = df['Age'] * df['Height']
        df['Age_Volume'] = df['Age'] * df['Volume']
        df['Age_Kilometer'] = df['Age'] * df['Kilometer']
        df['Age_Log_Kilometer'] = df['Age'] * df['Log_Kilometer']
        df['Kilometer_per_Year'] = df['Kilometer'] / df['Age'].clip(lower=1)
        df['Log_Kilometer_Length'] = df['Log_Kilometer'] * df['Length']
        df['Log_Kilometer_Power'] = df['Log_Kilometer'] * df['Max_Power_Value']
        df['Log_Kilometer_Torque'] = df['Log_Kilometer'] * df['Max_Torque_Value']

        df['Power_to_Weight'] = df['Max_Power_Value'] / df['Volume'].clip(lower=1)
        df['Power_to_CC'] = df['Max_Power_Value'] / df['Engine_Value'].clip(lower=1)
        df['Torque_to_CC'] = df['Max_Torque_Value'] / df['Engine_Value'].clip(lower=1)
        df['Power_to_Torque'] = df['Max_Power_Value'] / df['Max_Torque_Value'].clip(lower=1)
        df['RPM_Ratio'] = df['Max_Power_RPM'] / df['Max_Torque_RPM'].clip(lower=1)

        return df

    @classmethod
    def get_df(cls, df, model_id=1, get_Log_Price=True):
        """Trả về list chứa các tên columns và df"""
        if get_Log_Price: return cls.features[model_id] + ['Log_Price'], df[cls.features[model_id] + ['Log_Price']]
        else: return cls.features[model_id], df[cls.features[model_id]]
