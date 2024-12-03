"""
Data Handler Module

This module is responsible for loading, validating, and preprocessing data from CSV files.
It implements input validation, error handling, and data splitting functionality according
to the specified contract.
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataPreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass

class ValidationRule(Enum):
    """Enumeration of supported validation rules."""
    REQUIRED = "required"
    NUMERIC = "numeric"
    DATE = "date"
    CATEGORICAL = "categorical"
    RANGE = "range"

@dataclass
class DataStatistics:
    """Container for dataset statistics."""
    row_count: int
    column_count: int
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    numeric_stats: Dict[str, Dict[str, float]]

class DataHandler:
    """
    Handles data loading, validation, and preprocessing operations.
    
    Attributes:
        logger: Logger instance for tracking operations
        validation_rules: Dictionary of column-wise validation rules
        preprocessing_specs: Dictionary of preprocessing specifications
    """
    
    def __init__(self, validation_rules: Dict[str, list], preprocessing_specs: Dict[str, Any]):
        """
        Initialize the DataHandler with validation rules and preprocessing specifications.
        
        Args:
            validation_rules: Dictionary mapping column names to their validation rules
            preprocessing_specs: Dictionary containing preprocessing specifications
        """
        self.logger = logging.getLogger(__name__)
        self.validation_rules = validation_rules
        self.preprocessing_specs = preprocessing_specs

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pandas DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            if df.empty:
                raise pd.errors.EmptyDataError("CSV file is empty")
            
            self.logger.info(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data according to specified rules.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Check required columns
            missing_cols = set(self.validation_rules.keys()) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")

            # Apply validation rules
            for column, rules in self.validation_rules.items():
                for rule in rules:
                    if rule == ValidationRule.REQUIRED:
                        if df[column].isnull().any():
                            raise DataValidationError(f"Column {column} contains missing values")
                    elif rule == ValidationRule.NUMERIC:
                        if not pd.api.types.is_numeric_dtype(df[column]):
                            raise DataValidationError(f"Column {column} must be numeric")
                    elif rule == ValidationRule.DATE:
                        try:
                            pd.to_datetime(df[column])
                        except:
                            raise DataValidationError(f"Column {column} contains invalid dates")

            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps according to specifications.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            DataPreprocessingError: If preprocessing fails
        """
        try:
            processed_df = df.copy()
            
            # Apply preprocessing steps
            for step, params in self.preprocessing_specs.items():
                if step == "normalize":
                    for col in params["columns"]:
                        mean = processed_df[col].mean()
                        std = processed_df[col].std()
                        if std != 0:  # Avoid division by zero
                            processed_df[col] = (processed_df[col] - mean) / std
                elif step == "encode_categorical":
                    for col in params["columns"]:
                        processed_df[col] = pd.Categorical(processed_df[col]).codes
                        
            self.logger.info("Data preprocessing completed successfully")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise DataPreprocessingError(str(e))

    def split_data(self, df: pd.DataFrame, 
                  validation_ratio: float = 0.2, 
                  random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            df: Input DataFrame to split
            validation_ratio: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")
            
        np.random.seed(random_state)
        mask = np.random.rand(len(df)) >= validation_ratio
        
        train_data = df[mask]
        val_data = df[~mask]
        
        self.logger.info(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data

    def calculate_statistics(self, df: pd.DataFrame) -> DataStatistics:
        """
        Calculate and return dataset statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataStatistics object containing computed statistics
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_stats = {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            } for col in numeric_columns
        }
        
        stats = DataStatistics(
            row_count=len(df),
            column_count=len(df.columns),
            missing_values=df.isnull().sum().to_dict(),
            data_types=df.dtypes.astype(str).to_dict(),
            numeric_stats=numeric_stats
        )
        
        self.logger.info("Statistics calculated successfully")
        return stats

    def process_dataset(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, DataStatistics]:
        """
        Main method to process the dataset through the complete pipeline.
        
        Args:
            file_path: Path to the input CSV file
            
        Returns:
            Tuple of (training_data, validation_data, statistics)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Validate
        self.validate_data(df)
        
        # Preprocess
        processed_df = self.preprocess_data(df)
        
        # Split data
        train_data, val_data = self.split_data(processed_df)
        
        # Calculate statistics
        statistics = self.calculate_statistics(processed_df)
        
        return train_data, val_data, statistics
