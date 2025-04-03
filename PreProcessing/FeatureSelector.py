import numpy as np
import pandas as pd

class FeatureSelector(object):
    def __init__(self,
                 pre_process_df : pd.DataFrame):
        self.df = pre_process_df
        self._non_linearize_features()

        self.features = {
            1: [
                # Numerical
                # # Features cơ bản
                # "Year",
                "Kilometer",
                "Make_encoded",

                # # Features quan trọng về thông số kỹ thuật
                'Max_Power_Value',
                'Max_Power_RPM',
                'Max_Torque_Value',
                "Fuel Tank Capacity",

                # # Features tính toán
                'Efficiency',
                'Power_to_CC',

                # # Features kích thước
                "Length",
                'Volume',

                # # Các features kết hợp
                'Age_Volume',
                'Age_Kilometer',
                'Kilometer_per_Year',
                'Power_to_Torque',
                'RPM_Ratio',

                # Categorical
                'Fuel_CNG',
                'Fuel_Others',
                'Fuel_Petrol',

                'Transmission_is_Automatic',

                'Owner_First_Owner',
                'Owner_Fourth_Plus_Owner',
                'Owner_New',
                'Owner_Second_Owner',
                'Owner_Third_Owner',

                'Seating_2',
                'Seating_4',
                'Seating_6',
                'Seating_7',
                'Seating_8'
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

    def _non_linearize_features(self):
        self.df["Volume"] = self.df["Height"] * self.df["Width"] * self.df["Length"]
        self.df['Power_to_Weight'] = self.df['Max_Power_Value'] / self.df['Volume']
        self.df['Efficiency'] = self.df['Max_Power_Value'] / self.df['Fuel Tank Capacity']

        # Tương tác RPM
        self.df['Power_Density'] = self.df['Max_Power_Value'] / self.df['Max_Power_RPM']
        self.df['Torque_Density'] = self.df['Max_Torque_Value'] / self.df['Max_Torque_RPM']

        # Tỷ lệ Power/Torque
        self.df['Power_Torque_Ratio'] = self.df['Max_Power_Value'] / self.df['Max_Torque_Value']
        self.df['Length_Width_interaction'] = self.df['Length'] * self.df['Width']

        # Tương tác Year
        self.df['Age'] = 2025 - self.df['Year']
        self.df['Age_Volume'] = self.df['Age'] * self.df['Volume']
        self.df['Age_Kilometer'] = self.df['Age'] * self.df['Kilometer']
        self.df['Kilometer_per_Year'] = self.df['Kilometer'] / self.df['Age'].clip(lower=1)

        self.df['Power_to_Weight'] = self.df['Max_Power_Value'] / self.df['Volume'].clip(lower=1)
        self.df['Power_to_CC'] = self.df['Max_Power_Value'] / self.df['Engine_Value'].clip(lower=1)
        self.df['Torque_to_CC'] = self.df['Max_Torque_Value'] / self.df['Engine_Value'].clip(lower=1)
        self.df['Power_to_Torque'] = self.df['Max_Power_Value'] / self.df['Max_Torque_Value'].clip(lower=1)
        self.df['RPM_Ratio'] = self.df['Max_Power_RPM'] / self.df['Max_Torque_RPM'].clip(lower=1)

    def get_df(self, model_id=1, get_Log_Price=True):
        """Trả về list chứa các tên columns và df"""
        if get_Log_Price: return self.features[model_id] + ['Log_Price'], self.df[self.features[model_id] + ['Log_Price']]
        else: return self.features[model_id], self.df[self.features[model_id]]