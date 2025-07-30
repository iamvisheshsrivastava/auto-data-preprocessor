[![Project Status: In Progress](https://img.shields.io/badge/Project%20Status-In%20Progress-blue.svg)](#)

# auto-data-preprocessor

**auto-data-preprocessor** is a comprehensive Python library designed to automate the most common data preprocessing tasks for **classification** problems involving tabular data. It frees data scientists and machine learning practitioners from routine data cleaning and preparation chores, so they can focus on building models and interpreting their results.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Why auto-data-preprocessor? (Comparison with Existing Market Solutions)](#why-auto-data-preprocessor)  
4. [Installation](#installation)  
5. [Quick Start Guide](#quick-start-guide)  
6. [Detailed Usage Examples](#detailed-usage-examples)  
7. [Directory Structure](#directory-structure)  
8. [Roadmap and Future Enhancements](#roadmap-and-future-enhancements)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Contact](#contact)  

---

## 1. Overview

Real-world data is often messy. It may include missing values, outliers, mixed data types, and complex relationships. Before training a machine learning model, you need to clean, transform, and standardize your data to ensure robust, reliable results.

**auto-data-preprocessor** aims to simplify and automate these tedious, yet essential, steps for **classification** tasks. It provides an easy-to-use interface to handle:

- Missing value imputation  
- Outlier detection (planned in the next updates)  
- Categorical encoding  
- Numerical feature scaling  
- Basic feature engineering (such as polynomial features or transformations - planned)  

This library is suitable for **both beginners and advanced data scientists** who want to **accelerate** their data preparation workflows.

---

## 2. Key Features

1. **Automatic Missing Value Handling**  
   - Identifies missing values in numerical and categorical columns.  
   - Default imputation strategies (mean for numerical, most frequent for categorical) or **customizable** in upcoming versions.

2. **Categorical Encoding**
   - **Label Encoding** by default.
   - Optional **One-Hot Encoding** via `encoding_strategy="onehot"`.

3. **Feature Scaling**
   - Uses **StandardScaler** by default (mean=0, variance=1).
   - Optionally switch to **MinMaxScaler** via `scaling_strategy="minmax"`.

4. **Target Separation for Classification**
   - Automatically separates the target column from the dataset, simplifying the modeling pipeline.

5. **Column Normalization Utility**
   - Quickly scale numeric columns to the `[0, 1]` range with `normalize_columns()`.
6. **Outlier Detection (Planned)**
   - Will provide automated detection and handling using techniques like **IQR**, **z-score**, or **Isolation Forest**.

7. **Basic Feature Engineering (Planned)**
   - Transformations like logarithmic scaling, polynomial features, and feature interactions.

8. **Customization Options**
   - Upcoming versions will allow specifying advanced imputation methods, encoder types, and scaling techniques.

9. **Ease of Use**
   - Designed to be **beginner-friendly** yet **highly flexible** for experienced users.  
   - Simple function calls for straightforward workflows.

---

## 3. Why auto-data-preprocessor? (Comparison with Existing Market Solutions)

Many libraries in the Python ecosystem address data preprocessing. For example:

- **scikit-learn**: Provides transformers (e.g., `SimpleImputer`, `StandardScaler`, `LabelEncoder`) but requires you to chain multiple objects in a pipeline manually.  
- **Feature-engine**: Offers specialized transformers for encoding, outlier handling, and more, but may require more granular configuration.  
- **Pandas**: Very flexible for manual preprocessing but involves a lot of repetitive coding if you’re always performing the same steps.  

**auto-data-preprocessor** combines several commonly used steps into a **single, unified, high-level interface**, aiming to reduce boilerplate code. While it doesn’t fully replace the customizability of scikit-learn pipelines or advanced libraries for specialized tasks, it provides a **“plug-and-play”** solution for most classification workflows.

**Key Differentiators**:

1. **Simplicity**: Single function (`preprocess`) to handle multiple preprocessing tasks at once.  
2. **Automation**: Detects and handles many data issues automatically.  
3. **Extendibility**: Designed with modularity in mind, so future additions or community contributions can seamlessly extend functionalities.

---

## 4. Installation

The package is hosted on PyPI. You can install it using:

```bash
pip install auto-data-preprocessor
```

Alternatively, clone this repository and install with:

```bash
git clone https://github.com/iamvisheshsrivastava/auto-data-preprocessor.git
cd auto-data-preprocessor
pip install .
```

---

## 5. Quick Start Guide

Here’s a minimal example to show how easy it is to get started:

```python
import pandas as pd
from auto_preprocessor import DataPreprocessor

# Sample dataset
data = {
    "age": [25, 30, None, 35],
    "income": [50000, 60000, 70000, None],
    "gender": ["Male", "Female", "Male", "Female"],
    "purchased": [1, 0, 1, 0],
}

df = pd.DataFrame(data)

# Initialize the preprocessor
preprocessor = DataPreprocessor(scaling_strategy="minmax")

# Preprocess the data
X, y = preprocessor.preprocess(df, target_column="purchased")

print("Processed Features:")
print(X)

print("\nTarget Column:")
print(y)
```

**Output Example**:

```
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

## 6. Detailed Usage Examples

Below are additional examples demonstrating how you might integrate **auto-data-preprocessor** into a typical data science workflow.

### 6.1 Using Custom Target Column

Sometimes your dataset has a different target column name, such as `y` or `Class`. Just specify it:

```python
df = pd.read_csv("path_to_csv.csv")  # your custom dataset

preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(df, target_column="Class")
```

### 6.2 Integrating with scikit-learn Models

Once your features `X` and target `y` are processed, you can directly feed them into any scikit-learn estimator:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Suppose X, y are obtained from the preprocessor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Test Accuracy: {accuracy}")
```

### 6.3 Handling Missing Values (Planned Customization)

The default strategy is **mean imputation for numerical** and **most frequent for categorical** columns. In future releases, you’ll be able to specify:

```python
preprocessor = DataPreprocessor(
    numerical_imputation_strategy='median',
    categorical_imputation_strategy='mode',
    fill_values={'age': 30, 'income': 50000}  # custom fill values
)
```

---

## 7. Directory Structure

```plaintext
auto-data-preprocessor/
├── auto_preprocessor/
│   ├── __init__.py               # Package entry point
│   ├── data_cleaning.py          # Data cleaning utilities
│   ├── data_preprocessor.py      # Core preprocessing logic
│   └── feature_engineering.py    # Feature engineering utilities
├── data/
│   └── sample_data.csv           # Example dataset
├── examples/
│   └── usage_example.ipynb       # Jupyter Notebook with example usage
├── tests/
│   ├── test_data_cleaning.py     # Unit tests for data cleaning
│   ├── test_data_preprocessor.py # Unit tests for data preprocessing
│   └── test_feature_engineering.py # Unit tests for feature engineering
├── .gitignore                    # Files and folders to ignore in Git
├── LICENSE                       # MIT License file
├── README.md                     # Documentation
├── requirements.txt              # Dependency list
└── setup.py                      # Package setup configuration
```

---

## 8. Roadmap and Future Enhancements

Here’s what we’re working on next:

1. **Automated Exploratory Data Analysis (EDA)**  
   - Generate summary statistics, distributions, and correlation heatmaps.

2. **Advanced Outlier Detection**  
   - Implement IQR-based outlier removal, z-score, or advanced methods like Isolation Forest.

3. **Enhanced Categorical Encoding**
   - One-Hot Encoding implemented. Target Encoding and more advanced strategies are planned.

4. **Customizable Scaling**  
   - Options for MinMaxScaler, RobustScaler, and user-defined transformations.

5. **Feature Engineering**  
   - Transformations like polynomial features, logarithmic scaling, interaction terms.

6. **Model Evaluation**  
   - Integrate baseline model evaluations (e.g., a quick random forest or logistic regression) to check data readiness.

7. **Pipeline Export**  
   - Ability to save and load the preprocessing pipeline (joblib/pickle).

8. **Interactive Dashboards**  
   - Use Streamlit or Dash for interactive EDA and data cleaning insights.

---

## 9. Contributing

We **welcome contributions** from the community! Here’s how you can get involved:

1. **Fork the Repository**  
   - Click the “Fork” button in GitHub to create a personal copy of the repo.

2. **Create a Feature Branch**  
   - Use a descriptive name for your branch: `git checkout -b feature/improved-encoding`.

3. **Commit Your Changes**  
   - Write clear and concise commit messages: `git commit -m "Add MinMaxScaler option"`.

4. **Push Your Branch**  
   - `git push origin feature/improved-encoding`.

5. **Create a Pull Request**  
   - Compare your feature branch to the main repository’s `main` branch.  
   - Include details about what you changed and why.

**Guidelines**:

- Ensure that you include or update **unit tests** for any new functionality.  
- Provide relevant **documentation** or **docstrings**.  
- Follow Python’s PEP8 style guidelines for clean and readable code.

---

## 10. License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software in compliance with the license terms. For more details, see the [LICENSE](LICENSE) file.

---

## 11. Contact

For questions, suggestions, or feedback, feel free to reach out:

- **Author**: [Vishesh Srivastava](https://github.com/iamvisheshsrivastava)  
- **Email**: <srivastava.vishesh9@gmail.com>  
- **GitHub**: [iamvisheshsrivastava](https://github.com/iamvisheshsrivastava)  
- **Portfolio**: [visheshsrivastava.com](https://visheshsrivastava.com)

---

*If you find this project helpful, consider giving a star and sharing it with the community.*
