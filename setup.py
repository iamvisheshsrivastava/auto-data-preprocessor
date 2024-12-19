# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto-data-preprocessor",
    version="0.1.0",
    author="Vishesh Srivastava",
    author_email="srivivastava.vishesh9@gmail.com",
    description="A Python library that automates data preprocessing tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamvisheshsrivastava/auto-data-preprocessor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy"
    ],
)
