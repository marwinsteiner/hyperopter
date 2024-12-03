"""Tests for the logging system."""

import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd
from loguru import logger

from logging_system import LoggingSystem, InvalidLogPathError

class TestLoggingSystem(unittest.TestCase):
    """Test cases for LoggingSystem class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)
        self.logging_system = LoggingSystem(self.log_dir, "bayesian")

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
        """Test logging system initialization."""
        # Test valid initialization
        ls = LoggingSystem(self.log_dir, "bayesian")
        self.assertEqual(ls.current_strategy, "bayesian")
    
        # Test invalid strategy
        with self.assertRaises(ValueError):
            LoggingSystem(self.log_dir, "invalid_strategy")
    
        # Test invalid directory (use a file path instead of directory)
        test_file = self.log_dir / "test.txt"
        test_file.touch()
        with self.assertRaises(InvalidLogPathError):
            LoggingSystem(test_file, "bayesian")

    def test_log_trial(self):
        """Test logging individual trials."""
        # Test valid trial logging
        trial_params = {"learning_rate": 0.01, "batch_size": 32}
        self.logging_system.log_trial(1, trial_params, 0.5)
        self.assertEqual(len(self.logging_system.trial_results), 1)
        
        # Test invalid trial ID
        with self.assertRaises(ValueError):
            self.logging_system.log_trial(-1, trial_params, 0.5)
            
        # Test empty parameters
        with self.assertRaises(ValueError):
            self.logging_system.log_trial(2, {}, 0.5)

    def test_export_logs(self):
        """Test log export functionality."""
        # Add a trial
        trial_params = {"learning_rate": 0.01, "batch_size": 32}
        self.logging_system.log_trial(1, trial_params, 0.5)
        
        # Test JSON export
        json_path, success = self.logging_system.export_logs("json")
        self.assertTrue(success)
        self.assertTrue(json_path.exists())
        
        # Test CSV export
        csv_path, success = self.logging_system.export_logs("csv")
        self.assertTrue(success)
        self.assertTrue(csv_path.exists())
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.logging_system.export_logs("invalid")

    def test_get_best_trial(self):
        """Test retrieving best trial."""
        # Add multiple trials
        self.logging_system.log_trial(1, {"param": 1}, 0.5)
        self.logging_system.log_trial(2, {"param": 2}, 0.3)
        self.logging_system.log_trial(3, {"param": 3}, 0.7)
        
        # Test maximize
        best_trial = self.logging_system.get_best_trial(maximize=True)
        self.assertEqual(best_trial["result"], 0.7)
        
        # Test minimize
        best_trial = self.logging_system.get_best_trial(maximize=False)
        self.assertEqual(best_trial["result"], 0.3)
        
        # Test empty results
        empty_logger = LoggingSystem(self.log_dir, "bayesian")
        self.assertIsNone(empty_logger.get_best_trial())

    def test_get_trial_history(self):
        """Test retrieving trial history."""
        # Add trials with metrics
        self.logging_system.log_trial(1, {"param": 1}, 0.5, {"loss": 0.1})
        self.logging_system.log_trial(2, {"param": 2}, 0.3, {"loss": 0.2})
        
        # Test basic history
        history = self.logging_system.get_trial_history()
        self.assertEqual(len(history), 2)
        
        # Test history with metric
        history = self.logging_system.get_trial_history("loss")
        self.assertTrue("loss" in history.columns)
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            self.logging_system.get_trial_history("invalid_metric")

    def test_clear_logs(self):
        """Test clearing logs."""
        # Add a trial
        self.logging_system.log_trial(1, {"param": 1}, 0.5)
        self.assertEqual(len(self.logging_system.trial_results), 1)
        
        # Clear logs
        self.logging_system.clear_logs()
        self.assertEqual(len(self.logging_system.trial_results), 0)
