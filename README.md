# auto-data-preprocessor

## Overview
`auto-data-preprocessor` is a Python library designed to simplify and automate data preprocessing tasks for classification problems using tabular data. It handles essential steps like data cleaning, missing value imputation, encoding, and feature scaling, making it easier for users to focus on building models and analyzing results. This library is suitable for both beginners and experienced data scientists who want to streamline their data preparation workflows.

---

## Features
- **Data Cleaning:** Automatically identifies and handles missing values and outliers.
- **Data Imputation:** Fills missing values using default strategies (e.g., mean imputation) or user-specified methods.
- **Categorical Encoding:** Encodes categorical variables using Label Encoding or One-Hot Encoding.
- **Feature Scaling:** Scales numerical features using StandardScaler (default) or other scaling options.
- **Target Separation:** Automatically separates the target column for classification tasks.
- **Ease of Use:** Allows users to customize preprocessing steps through simple arguments.
- **Future Enhancements:** Automated EDA, outlier detection, class balancing, and lightweight model evaluation.

---

## Installation

To install the library, use pip:

```bash
pip install auto-data-preprocessor
```

---

## Quick Start Guide

### Example Usage
```python
import pandas as pd
from auto_data_preprocessor import AutoDataPreprocessor

# Sample data
data = {
    "age": [25, 30, None, 35],
    "income": [50000, 60000, 70000, None],
    "gender": ["Male", "Female", "Male", "Female"],
    "purchased": [1, 0, 1, 0],
}

df = pd.DataFrame(data)

# Initialize the preprocessor
preprocessor = AutoDataPreprocessor()

# Preprocess the data
X, y = preprocessor.preprocess(df, target_column="purchased")

# Output results
print("Processed Features:")
print(X)

print("Target Column:")
print(y)
```

#### Output:
```plaintext
Processed Features:
          age    income  gender
0  -1.224745 -1.224745       1
1   0.000000  0.000000       0
2   1.224745  1.224745       1
3   2.449490  2.449490       0

Target Column:
0    1
1    0
2    1
3    0
```

---

## Customization Options (Planned Enhancements)

The following customization options will soon be available:

1. **Missing Value Imputation:**
   - Specify strategies such as `mean`, `median`, `mode`, or custom values.

2. **Categorical Encoding:**
   - Choose between `Label Encoding` and `One-Hot Encoding`.

3. **Scaling Methods:**
   - Use alternative scalers such as `MinMaxScaler` or `RobustScaler`.

4. **Outlier Detection and Handling:**
   - Automated detection using IQR, z-score, or Isolation Forest.

5. **Class Balancing:**
   - Implement resampling techniques like `SMOTE`, `RandomOverSampler`, and `RandomUnderSampler`.

6. **Data Visualization:**
   - Include EDA tools for data distribution insights, histograms, and correlation heatmaps.

7. **Pipeline Export:**
   - Save preprocessing pipelines using `joblib` or `pickle` for reuse.

---

## Directory Structure
```
auto-data-preprocessor/
├── auto_preprocessor/
│   ├── __init__.py           # Makes it a package
│   ├── data_cleaning.py      # Data cleaning utilities
│   ├── data_preprocessor.py  # Core preprocessing logic
│   ├── feature_engineering.py # Feature engineering utilities
├── data/
│   ├── sample_data.csv       # Example dataset
├── examples/
│   ├── usage_example.ipynb   # Jupyter Notebook with example usage
├── tests/
│   ├── test_data_cleaning.py # Unit tests for data cleaning
│   ├── test_data_preprocessor.py # Unit tests for data preprocessing
│   ├── test_feature_engineering.py # Unit tests for feature engineering
├── .gitignore                # Git ignore file
├── LICENSE                   # Project license
├── README.md                 # Documentation
├── requirements.txt          # Dependency list
├── setup.py                  # Package setup file
```

---

## Future Enhancements

Here is a roadmap for planned features:

1. **Automated Exploratory Data Analysis (EDA):**
   - Generate basic insights and visualizations for numerical and categorical variables.

2. **Feature Engineering:**
   - Include transformations like polynomial features, logarithmic scaling, and interaction terms.

3. **Explainability:**
   - Provide insights into preprocessing effects and feature importance.

4. **Interactive Dashboards:**
   - Develop interactive EDA dashboards using `Dash` or `Streamlit`.

5. **Basic Model Evaluation:**
   - Include lightweight model evaluation tools for baseline comparisons.

---

## Contribution Guidelines

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with clear messages.
4. Push your branch and create a pull request.

Please ensure all new features are covered by unit tests and documented.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or suggestions, please reach out:

- **Author:** Vishesh Srivastava
- **Email:** srivastava.vishesh9@gmail.com
- **GitHub:** [iamvisheshsrivastava](https://github.com/iamvisheshsrivastava)
- **Portfolio:** [visheshsrivastava.com](https://visheshsrivastava.com)

