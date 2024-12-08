"""
Data Handler Module

This module is responsible for loading, validating, and preprocessing data from CSV files.
It implements input validation, error handling, and data splitting functionality according
to the specified contract.
"""

from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum
from sklearn.model_selection import train_test_split

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
        required_columns: List of required column names
    """
    
    def __init__(self, validation_rules: Dict[str, list], preprocessing_specs: Dict[str, Any], required_columns: Optional[List[str]] = None):
        """
        Initialize the DataHandler with validation rules and preprocessing specifications.
        
        Args:
            validation_rules: Dictionary mapping column names to their validation rules
            preprocessing_specs: Dictionary containing preprocessing specifications
            required_columns: Optional list of required column names
        """
        self.logger = logging.getLogger(__name__)
        self.validation_rules = validation_rules
        self.preprocessing_specs = preprocessing_specs
        self.required_columns = required_columns or []
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate data from a file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            DataFrame containing validated data
            
        Raises:
            ValueError: If data is invalid
        """
        try:
            data = pd.read_csv(data_path)
            self.validate_data(data)
            
            if data.isnull().any().any():
                self.logger.warning("Found missing values, forward filling...")
                data = data.ffill()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading data: {str(e)}")
    
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
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {missing_cols}")
            
            # Apply validation rules
            for column, rules in self.validation_rules.items():
                if column not in df.columns:
                    continue  # Skip validation for columns not in the data
                
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
        Preprocess the data by normalizing numerical features and encoding categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        for column in processed_df.columns:
            if processed_df[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Normalize numerical features
                std = processed_df[column].std()
                if std != 0:
                    processed_df[column] = (processed_df[column] - processed_df[column].mean()) / std
            else:
                # Encode categorical features
                processed_df[column] = pd.Categorical(processed_df[column]).codes.astype(np.int32)
        
        return processed_df

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
            
        train_data, val_data = train_test_split(
            df, 
            test_size=validation_ratio,
            random_state=random_state
        )
        
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
        
        # Preprocess
        processed_df = self.preprocess_data(df)
        
        # Split data
        train_data, val_data = self.split_data(processed_df)
        
        # Calculate statistics
        statistics = self.calculate_statistics(processed_df)
        
        return train_data, val_data, statistics
