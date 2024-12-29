from setuptools import setup, find_packages

# Read the README file for a long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto-data-preprocessor",
    version="0.1.0",  # Initial release version
    author="Vishesh Srivastava",
    author_email="srivastava.vishesh9@gmail.com",
    description="A simple library to automate data preprocessing for machine learning pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # README is in Markdown format
    url="https://github.com/iamvisheshsrivastava/auto-data-preprocessor",
    project_urls={
        "Bug Tracker": "https://github.com/iamvisheshsrivastava/auto-data-preprocessor/issues",
        "Source Code": "https://github.com/iamvisheshsrivastava/auto-data-preprocessor",
        "Documentation": "https://visheshsrivastava.com",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "auto_preprocessor"},  # Directory for the package
    packages=find_packages(where="auto_preprocessor"),  # Automatically find packages
    python_requires=">=3.7",  # Specify Python version
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "pyyaml>=5.4.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.5", "pytest-cov>=3.0.0"],
    },
    include_package_data=True,  # Includes data files from MANIFEST.in
    license="MIT",
)
