import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    A comprehensive class for preprocessing tabular data for classification tasks.
    
    Features:
    - Automatic detection of categorical and numerical columns
    - Missing value imputation (default strategies)
    - Multiple encoding options for categorical variables (label, onehot, ordinal)
    - Configurable scaling for numerical features (standard or min-max)
    - Target column separation
    """

    def __init__(self,
                 numerical_imputer_strategy="mean",
                 categorical_imputer_strategy="most_frequent",
                 encoding_strategy="label",
                 scaling_strategy="standard"):
        """
        Initializes the preprocessing pipeline.

        Parameters:
        - numerical_imputer_strategy: str, default="mean"
              Strategy for imputing missing values in numerical columns.
        - categorical_imputer_strategy: str, default="most_frequent"
              Strategy for imputing missing values in categorical columns.
        - encoding_strategy: str, default="label"
              Encoding method for categorical variables. Supported values are
              "label", "onehot", and "ordinal".
        - scaling_strategy: str, default="standard"
              Strategy for scaling numerical features. Supported values are
              "standard" and "minmax".
        """
        self.encoding_strategy = encoding_strategy
        self.scaling_strategy = scaling_strategy

        if self.scaling_strategy == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_strategy == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling strategy")
        self.label_encoders = {}
        self.onehot_encoder = None
        self.ordinal_encoder = None
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

        # store columns for later transformations
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        # 3. Handle missing values
        if numerical_columns:
            X[numerical_columns] = self.num_imputer.fit_transform(X[numerical_columns])
        if categorical_columns:
            X[categorical_columns] = self.cat_imputer.fit_transform(X[categorical_columns])

        # 4. Encode categorical variables
        if categorical_columns:
            if self.encoding_strategy == "label":
                for column in categorical_columns:
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column].astype(str))
                    self.label_encoders[column] = le
            elif self.encoding_strategy == "onehot":
                self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                onehot_array = self.onehot_encoder.fit_transform(X[categorical_columns])
                new_cols = self.onehot_encoder.get_feature_names_out(categorical_columns)
                onehot_df = pd.DataFrame(onehot_array, columns=new_cols, index=X.index)
                X = X.drop(columns=categorical_columns)
                X = pd.concat([X, onehot_df], axis=1)
            elif self.encoding_strategy == "ordinal":
                self.ordinal_encoder = OrdinalEncoder()
                X[categorical_columns] = self.ordinal_encoder.fit_transform(X[categorical_columns].astype(str))

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

        categorical_columns = self.categorical_columns
        numerical_columns = self.numerical_columns

        if numerical_columns:
            df[numerical_columns] = self.num_imputer.transform(df[numerical_columns])

        if categorical_columns:
            df[categorical_columns] = self.cat_imputer.transform(df[categorical_columns])
            if self.encoding_strategy == "label":
                for column in categorical_columns:
                    if column in self.label_encoders:
                        df[column] = self.label_encoders[column].transform(df[column].astype(str))
            elif self.encoding_strategy == "onehot" and self.onehot_encoder is not None:
                onehot_array = self.onehot_encoder.transform(df[categorical_columns])
                new_cols = self.onehot_encoder.get_feature_names_out(categorical_columns)
                onehot_df = pd.DataFrame(onehot_array, columns=new_cols, index=df.index)
                df = df.drop(columns=categorical_columns)
                df = pd.concat([df, onehot_df], axis=1)
            elif self.encoding_strategy == "ordinal" and self.ordinal_encoder is not None:
                df[categorical_columns] = self.ordinal_encoder.transform(df[categorical_columns].astype(str))

        if numerical_columns:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df
