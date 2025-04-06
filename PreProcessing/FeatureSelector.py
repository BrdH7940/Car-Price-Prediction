import numpy as np
import pandas as pd

class FeatureSelector(object):
    features = {
        1: [
            'Length',
            'Fuel Tank Capacity',
            'Max_Power_Value',
            'Max_Torque_Value',
            'Make_encoded',
            'Fuel_Petrol',
            'Transmission_is_Automatic',
            'Owner_Third_Owner',
            'Seating_2',
            'Volume',
            'Log_Max_Power_Value',
            'Power_Density',
            'Length_Width_interaction',

            'Age',
            'Age_Length',
            'Age_Volume',
            'Age_Log_Kilometer',
            'Make_encoded_log',
        ],
        2: [
            'Kilometer',

            'Length',
            'Width',
            'Height',

            'Fuel Tank Capacity',
            'Engine_Value',
            'Max_Power_Value',
            'Max_Power_RPM',
            'Max_Torque_Value',
            'Max_Torque_RPM',

            'Make_encoded',

            'Fuel_Diesel',
            'Fuel_Petrol',

            'Transmission_is_Automatic',

            'Owner_Fourth_Plus_Owner',
            'Owner_New',

            'Seating_2',
            'Seating_4',

            'Volume',

            'Power_to_Weight',

            'Log_Kilometer',
            'Log_Max_Power_Value',
            'Log_Max_Torque_Value',

            'Power_Density',
            'Torque_Density',

            'Power_Torque_Ratio',

            'Length_Width_interaction',

            'Year_Inverse',
            'Age',
            'Age_squared',
            'Log_Age',
            'Age_Length',
            'Age_Height',
            'Age_Volume',
            'Age_Log_Kilometer',

            'Kilometer_per_Year',
            'Log_Kilometer_Length',
            'Log_Kilometer_Power',
            'Log_Kilometer_Torque',

            'Make_encoded_sqrt',
            'Make_encoded_log',
            'Make_encoded_squared',

            'Power_to_CC',
            'Torque_to_CC',
            'Power_to_Torque',
            'RPM_Ratio',
        ],
        3: [
            'Age',
            'Log_Max_Power_Value',
            'Age_Log_Kilometer',
            'Make_encoded',
            'Fuel Tank Capacity',
            'Fuel_Petrol',
            'Transmission_is_Automatic',
            'Owner_Fourth_Plus_Owner',
            'Seating_2',
            'Seating_4',
            'Length',
        ],
        4: [
            'Age',
            'Age_Length',
            'Age_Volume',
            'Age_Log_Kilometer',
            'Make_encoded',

            'Fuel Tank Capacity',

            'Length',
            'Width',
            'Length_Width_interaction',

            'Power_Density',

            'Color_Silver',

            'Seating_2',
            'Seating_4',
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

        df['Make_encoded_sqrt'] = np.sqrt(df['Make_encoded'])
        df['Make_encoded_log'] = np.log(df['Make_encoded'])
        df['Make_encoded_squared'] = df['Make_encoded'] ** 2

        df['Power_to_Weight'] = df['Max_Power_Value'] / df['Volume'].clip(lower=1)
        df['Power_to_CC'] = df['Max_Power_Value'] / df['Engine_Value'].clip(lower=1)
        df['Torque_to_CC'] = df['Max_Torque_Value'] / df['Engine_Value'].clip(lower=1)
        df['Power_to_Torque'] = df['Max_Power_Value'] / df['Max_Torque_Value'].clip(lower=1)
        df['RPM_Ratio'] = df['Max_Power_RPM'] / df['Max_Torque_RPM'].clip(lower=1)
        df.drop(['Model', 'Location'], axis=1, inplace=True)

        return df

    @classmethod
    def get_df(cls, df, model_id=1, get_Log_Price=True):
        """Trả về list chứa các tên columns và df"""
        if get_Log_Price: return cls.features[model_id] + ['Log_Price'], df[cls.features[model_id] + ['Log_Price']]
        else: return cls.features[model_id], df[cls.features[model_id]]

    @classmethod
    def get_features(cls, df):
        df = df.select_dtypes(include=[np.number])
        cols = df.columns
        return list(cols), df
