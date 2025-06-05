import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    A comprehensive class for preprocessing tabular data for classification tasks.
    
    Features:
    - Automatic detection of categorical and numerical columns
    - Missing value imputation (default strategies)
    - Label encoding for categorical variables
    - Standard scaling for numerical features
    - Target column separation
    """

    def __init__(self,
                 numerical_imputer_strategy="mean",
                 categorical_imputer_strategy="most_frequent"):
        """
        Initializes the preprocessing pipeline.

        Parameters:
        - numerical_imputer_strategy: str, default="mean"
              Strategy for imputing missing values in numerical columns.
        - categorical_imputer_strategy: str, default="most_frequent"
              Strategy for imputing missing values in categorical columns.
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.num_imputer = SimpleImputer(strategy=numerical_imputer_strategy)
        self.cat_imputer = SimpleImputer(strategy=categorical_imputer_strategy)

    def preprocess(self, df: pd.DataFrame, target_column: str):
        """
        Preprocesses the input DataFrame by handling missing values, encoding
        categorical features, and scaling numeric features.

        Parameters:
        - df: pd.DataFrame
              The input DataFrame containing features and target column.
        - target_column: str
              The name of the target column to separate from the feature set.

        Returns:
        - X: pd.DataFrame
              Preprocessed feature set
        - y: pd.Series
              Target labels
        """
        df = df.copy()

        # 1. Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # 2. Identify column types
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

        # 3. Handle missing values
        if numerical_columns:
            X[numerical_columns] = self.num_imputer.fit_transform(X[numerical_columns])
        if categorical_columns:
            X[categorical_columns] = self.cat_imputer.fit_transform(X[categorical_columns])

        # 4. Encode categorical variables
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            self.label_encoders[column] = le

        # 5. Scale numerical features
        if numerical_columns:
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        return X, y

    def transform(self, df: pd.DataFrame):
        """
        Applies the same preprocessing steps on new/unseen data using the 
        fitted transformers from `preprocess()`.

        Parameters:
        - df: pd.DataFrame
              New dataset without the target column.

        Returns:
        - X: pd.DataFrame
              Transformed feature set.
        """
        df = df.copy()

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if numerical_columns:
            df[numerical_columns] = self.num_imputer.transform(df[numerical_columns])

        if categorical_columns:
            df[categorical_columns] = self.cat_imputer.transform(df[categorical_columns])
            for column in categorical_columns:
                if column in self.label_encoders:
                    df[column] = self.label_encoders[column].transform(df[column].astype(str))

        if numerical_columns:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df
