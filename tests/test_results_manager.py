"""Tests for the results manager."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
import unittest

import pandas as pd
from loguru import logger

from results_manager import ResultsManager, InvalidResultsError

class TestResultsManager(unittest.TestCase):
    """Test cases for ResultsManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.results_manager = ResultsManager(self.output_dir)
        
        # Sample data
        self.sample_parameters = {
            "learning_rate": 0.01,
            "batch_size": 32
        }
        self.sample_metrics = {
            "accuracy": 0.95,
            "loss": 0.1
        }
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove all loguru handlers
        logger.remove()
        
        import shutil
        import time
        # Give some time for file handles to be released
        time.sleep(0.1)
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass  # Ignore permission errors during cleanup
            
    def test_initialization(self):
        """Test results manager initialization."""
        # Test valid initialization
        rm = ResultsManager(self.output_dir)
        self.assertTrue(rm.output_dir.exists())
        
        # Test invalid directory (use a file path)
        test_file = self.output_dir / "test.txt"
        test_file.touch()
        with self.assertRaises(InvalidResultsError):
            ResultsManager(test_file)
            
    def test_add_optimization_result(self):
        """Test adding optimization results."""
        # Test valid result
        self.results_manager.add_optimization_result(
            1, self.sample_parameters, self.sample_metrics
        )
        self.assertEqual(len(self.results_manager.results_data), 1)
        
        # Test invalid trial ID
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_optimization_result(
                -1, self.sample_parameters, self.sample_metrics
            )
            
        # Test empty parameters
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_optimization_result(
                2, {}, self.sample_metrics
            )
            
        # Test empty metrics
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_optimization_result(
                2, self.sample_parameters, {}
            )
            
    def test_add_config_metadata(self):
        """Test adding configuration metadata."""
        # Test valid metadata
        metadata = {"strategy": "bayesian", "max_trials": 100}
        self.results_manager.add_config_metadata(metadata)
        self.assertEqual(self.results_manager.config_metadata, metadata)
        
        # Test empty metadata
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_config_metadata({})
            
    def test_export_results(self):
        """Test results export functionality."""
        # Add sample result and metadata
        self.results_manager.add_optimization_result(
            1, self.sample_parameters, self.sample_metrics
        )
        self.results_manager.add_config_metadata(
            {"strategy": "bayesian"}
        )
        
        # Test JSON export
        json_path = self.results_manager.export_results("json")
        self.assertTrue(json_path.exists())
        with open(json_path) as f:
            data = json.load(f)
        self.assertIn("metadata", data)
        self.assertIn("results", data)
        
        # Test CSV export
        csv_path = self.results_manager.export_results("csv")
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)
        
        # Test invalid format
        with self.assertRaises(InvalidResultsError):
            self.results_manager.export_results("invalid")
            
    def test_generate_summary(self):
        """Test summary generation."""
        # Test with no results
        with self.assertRaises(InvalidResultsError):
            self.results_manager.generate_summary()
            
        # Add multiple results
        for i in range(3):
            self.results_manager.add_optimization_result(
                i, self.sample_parameters,
                {"accuracy": 0.9 + i*0.01, "loss": 0.1 - i*0.01}
            )
            
        # Test summary generation
        summary = self.results_manager.generate_summary()
        self.assertEqual(summary["total_trials"], 3)
        self.assertIn("accuracy", summary["metrics"])
        self.assertIn("loss", summary["metrics"])
        
    def test_clear_results(self):
        """Test clearing results."""
        # Add result and metadata
        self.results_manager.add_optimization_result(
            1, self.sample_parameters, self.sample_metrics
        )
        self.results_manager.add_config_metadata(
            {"strategy": "bayesian"}
        )
        
        # Clear results
        self.results_manager.clear_results()
        self.assertTrue(self.results_manager.results_data.empty)
        self.assertEqual(self.results_manager.config_metadata, {})

    def test_add_batch_results(self):
        """Test adding batch results."""
        # Prepare batch results
        batch_results = [
            {
                "trial_id": i,
                "parameters": {"learning_rate": 0.01 * i},
                "metrics": {"accuracy": 0.9 + 0.01 * i}
            }
            for i in range(3)
        ]
        
        # Test valid batch
        self.results_manager.add_batch_results(batch_results)
        self.assertEqual(len(self.results_manager.results_data), 3)
        
        # Test empty batch
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_batch_results([])
            
        # Test invalid batch format
        invalid_batch = [{"trial_id": 1}]  # Missing required fields
        with self.assertRaises(InvalidResultsError):
            self.results_manager.add_batch_results(invalid_batch)
            
    def test_export_for_ci(self):
        """Test CI/CD export functionality."""
        # Add sample results
        self.results_manager.add_optimization_result(
            1, {"param": 1}, {"accuracy": 0.95, "result": 0.95}
        )
        self.results_manager.add_config_metadata({"strategy": "random"})
        
        # Test CI export without file
        ci_report = self.results_manager.export_for_ci()
        self.assertEqual(ci_report["status"], "success")
        self.assertEqual(ci_report["metrics"]["total_trials"], 1)
        self.assertIn("metadata", ci_report)
        
        # Test CI export with file
        temp_file = Path(self.temp_dir) / "ci_report.json"
        ci_report = self.results_manager.export_for_ci(temp_file)
        self.assertTrue(temp_file.exists())
        self.assertIn(str(temp_file), ci_report["artifacts"])
        
        # Test export with no results
        self.results_manager.clear_results()
        with self.assertRaises(InvalidResultsError):
            self.results_manager.export_for_ci()

if __name__ == "__main__":
    unittest.main()
