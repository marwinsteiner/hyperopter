"""
Test suite for DataHandler component.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from data_handler import DataHandler, DataValidationError, ValidationRule
import json

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_data.csv"
        
        # Create sample data with required columns
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = {
            'date': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(75, 225, 100),
            'volume': np.random.uniform(1e6, 5e6, 100)
        }
        self.sample_data = pd.DataFrame(data)
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Create sample validation rules and preprocessing specs
        self.validation_rules = {
            'date': [ValidationRule.REQUIRED, ValidationRule.DATE],
            'open': [ValidationRule.REQUIRED, ValidationRule.NUMERIC],
            'high': [ValidationRule.REQUIRED, ValidationRule.NUMERIC],
            'low': [ValidationRule.REQUIRED, ValidationRule.NUMERIC],
            'close': [ValidationRule.REQUIRED, ValidationRule.NUMERIC],
            'volume': [ValidationRule.REQUIRED, ValidationRule.NUMERIC]
        }
        
        self.preprocessing_specs = {
            'normalize': {'columns': ['open', 'high', 'low', 'close', 'volume']},
            'datetime_features': {'columns': ['date']}
        }
        
        # Initialize DataHandler with settings
        self.data_handler = DataHandler(
            validation_rules=self.validation_rules,
            preprocessing_specs=self.preprocessing_specs
        )

    def test_load_data(self):
        """Test data loading functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertEqual(list(df.columns), ['date', 'open', 'high', 'low', 'close', 'volume'])

    def test_validate_data(self):
        """Test data validation functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        
        # Should not raise any exceptions
        self.data_handler.validate_data(df)
        
        # Test with invalid data
        invalid_df = df.copy()
        invalid_df.loc[0, "open"] = None
        with self.assertRaises(DataValidationError):
            self.data_handler.validate_data(invalid_df)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        processed_df = self.data_handler.preprocess_data(df)
        
        # Check normalization for non-zero std columns
        open_std = df["open"].std()
        if open_std != 0:
            self.assertTrue(
                np.allclose(processed_df["open"].mean(), 0, atol=1e-10)
            )
            self.assertTrue(
                np.allclose(processed_df["open"].std(), 1, atol=1e-10)
            )
        
        # Check categorical encoding
        self.assertTrue(
            processed_df["date"].dtype in [np.int32, np.int64]
        )

    def test_split_data(self):
        """Test data splitting functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        train_data, val_data = self.data_handler.split_data(
            df, validation_ratio=0.4, random_state=42
        )
        
        # Check split sizes
        self.assertEqual(len(train_data), 60)
        self.assertEqual(len(val_data), 40)
        
        # Check no data leakage
        train_indices = set(train_data.index)
        val_indices = set(val_data.index)
        self.assertEqual(len(train_indices.intersection(val_indices)), 0)

    def test_calculate_statistics(self):
        """Test statistics calculation functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        stats = self.data_handler.calculate_statistics(df)
        
        self.assertEqual(stats.row_count, 100)
        self.assertEqual(stats.column_count, 6)
        self.assertEqual(stats.missing_values["open"], 0)
        self.assertIn("open", stats.numeric_stats)

    def test_process_dataset(self):
        """Test complete data processing pipeline."""
        train_data, val_data, stats = self.data_handler.process_dataset(
            str(self.csv_path)
        )
        
        # Check outputs
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertEqual(len(train_data) + len(val_data), 100)
        self.assertEqual(stats.row_count, 100)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary files
        if self.csv_path.exists():
            self.csv_path.unlink()
        Path(self.temp_dir).rmdir()

if __name__ == '__main__':
    unittest.main()
