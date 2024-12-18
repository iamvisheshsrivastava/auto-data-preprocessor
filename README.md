# Auto Data Preprocessor

Auto Data Preprocessor is a Python library that automates data preprocessing tasks, including:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features

## Features
- Automatically identifies and preprocesses numerical and categorical features.
- Handles missing values using mean imputation.
- Supports feature scaling with StandardScaler.
- Encodes categorical data using LabelEncoder.

## Installation

Install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

Here is an example of how to use the Auto Data Preprocessor:

### Input Dataset

```python
import pandas as pd
from auto_data_preprocessor import AutoDataPreprocessor

data = {
    "age": [25, 30, None, 35],
    "income": [50000, 60000, 70000, None],
    "gender": ["Male", "Female", "Male", "Female"],
    "purchased": [1, 0, 1, 0],
}

df = pd.DataFrame(data)

preprocessor = AutoDataPreprocessor()
X, y = preprocessor.preprocess(df, target_column="purchased")

print("Processed Features:")
print(X)

print("Target Column:")
print(y)
```

### Output

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

## Requirements
- Python 3.6+
- pandas
- scikit-learn
- numpy

## Project Structure

```
auto-data-preprocessor/
|
├── auto_data_preprocessor.py   # Main script for data preprocessing
├── example_usage.py            # Example showing how to use the preprocessor
├── requirements.txt            # List of dependencies
└── README.md                   # Detailed project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License.
