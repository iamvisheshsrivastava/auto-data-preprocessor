from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        numerical_columns = X.select_dtypes(include=["number"]).columns

        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            self.label_encoders[column] = le

        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        return X, y
