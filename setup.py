"""Setup configuration for hyperopter package."""

from setuptools import setup, find_packages

setup(
    name="hyperopter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "loguru",
    ],
    python_requires=">=3.8",
)
