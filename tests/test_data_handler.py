"""
Test suite for DataHandler component.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from data_handler import DataHandler, ValidationRule, DataValidationError

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample validation rules
        self.validation_rules = {
            "feature1": [ValidationRule.REQUIRED, ValidationRule.NUMERIC],
            "feature2": [ValidationRule.REQUIRED, ValidationRule.CATEGORICAL],
            "target": [ValidationRule.REQUIRED, ValidationRule.NUMERIC]
        }
        
        # Create sample preprocessing specifications
        self.preprocessing_specs = {
            "normalize": {"columns": ["feature1"]},
            "encode_categorical": {"columns": ["feature2"]}
        }
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": ["A", "B", "A", "C", "B"],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Create temporary CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_data.csv"
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Initialize DataHandler
        self.data_handler = DataHandler(
            validation_rules=self.validation_rules,
            preprocessing_specs=self.preprocessing_specs
        )

    def test_load_data(self):
        """Test data loading functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ["feature1", "feature2", "target"])

    def test_validate_data(self):
        """Test data validation functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        
        # Should not raise any exceptions
        self.data_handler.validate_data(df)
        
        # Test with invalid data
        invalid_df = df.copy()
        invalid_df.loc[0, "feature1"] = None
        with self.assertRaises(DataValidationError):
            self.data_handler.validate_data(invalid_df)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        processed_df = self.data_handler.preprocess_data(df)
        
        # Check normalization for non-zero std columns
        feature1_std = df["feature1"].std()
        if feature1_std != 0:
            self.assertTrue(
                np.allclose(processed_df["feature1"].mean(), 0, atol=1e-10)
            )
            self.assertTrue(
                np.allclose(processed_df["feature1"].std(), 1, atol=1e-10)
            )
        
        # Check categorical encoding
        self.assertTrue(
            processed_df["feature2"].dtype in [np.int32, np.int64]
        )

    def test_split_data(self):
        """Test data splitting functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        train_data, val_data = self.data_handler.split_data(
            df, validation_ratio=0.4, random_state=42
        )
        
        # Check split sizes
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(val_data), 2)
        
        # Check no data leakage
        train_indices = set(train_data.index)
        val_indices = set(val_data.index)
        self.assertEqual(len(train_indices.intersection(val_indices)), 0)

    def test_calculate_statistics(self):
        """Test statistics calculation functionality."""
        df = self.data_handler.load_data(str(self.csv_path))
        stats = self.data_handler.calculate_statistics(df)
        
        self.assertEqual(stats.row_count, 5)
        self.assertEqual(stats.column_count, 3)
        self.assertEqual(stats.missing_values["feature1"], 0)
        self.assertIn("feature1", stats.numeric_stats)

    def test_process_dataset(self):
        """Test complete data processing pipeline."""
        train_data, val_data, stats = self.data_handler.process_dataset(
            str(self.csv_path)
        )
        
        # Check outputs
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertEqual(len(train_data) + len(val_data), 5)
        self.assertEqual(stats.row_count, 5)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary files
        if self.csv_path.exists():
            self.csv_path.unlink()
        Path(self.temp_dir).rmdir()

if __name__ == '__main__':
    unittest.main()
