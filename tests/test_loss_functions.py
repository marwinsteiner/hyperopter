"""
Tests for the base loss function class and its core functionality.

This module contains tests for:
1. Base class interface
2. Input validation
3. Error handling
4. Thread safety
5. Performance benchmarks
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any
import threading
import time
import concurrent.futures

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from loss_functions import BaseLossFunction, TimeFrequency


class MockLossFunction(BaseLossFunction):
    """Mock implementation for testing BaseLossFunction."""
    
    def __init__(self, return_value: float = 0.0):
        super().__init__("Mock Loss", "minimize")
        self.return_value = return_value
        
    def calculate_loss(self, trade_data, position_sizes=None, pnl=None, durations=None):
        self.metadata = {'mock_value': self.return_value}
        return self.return_value


def create_sample_data(n_trades: int = 100) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create sample trade data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=n_trades, freq='D')
    trade_data = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.randn(n_trades).cumsum() + 100,
        'volume': np.random.randint(1, 1000, n_trades),
        'capital': np.ones(n_trades) * 10000
    })
    position_sizes = pd.Series(np.random.randint(1, 100, n_trades))
    pnl = pd.Series(np.random.randn(n_trades) * 100)
    durations = pd.Series([timedelta(minutes=np.random.randint(1, 60)) for _ in range(n_trades)])
    return trade_data, position_sizes, pnl, durations


class TestBaseLossFunction(unittest.TestCase):
    """Test cases for BaseLossFunction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = MockLossFunction()
        self.trade_data, self.position_sizes, self.pnl, self.durations = create_sample_data()
        
    def test_interface_compliance(self):
        """Test that the interface matches the contract."""
        # Test required attributes
        self.assertTrue(hasattr(self.loss_fn, 'name'))
        self.assertTrue(hasattr(self.loss_fn, 'direction'))
        self.assertTrue(hasattr(self.loss_fn, 'metadata'))
        
        # Test required methods
        self.assertTrue(callable(self.loss_fn.calculate_loss))
        self.assertTrue(callable(self.loss_fn.get_metadata))
        
        # Test direction validation
        with self.assertRaises(ValueError):
            MockLossFunction().direction = 'invalid'
            
    def test_input_validation(self):
        """Test input validation for trade data and optional inputs."""
        # Test invalid trade_data type
        with self.assertRaises(TypeError):
            self.loss_fn(trade_data=[1, 2, 3])
            
        # Test empty trade_data
        with self.assertRaises(ValueError):
            self.loss_fn(trade_data=pd.DataFrame())
            
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            self.loss_fn(
                trade_data=self.trade_data,
                position_sizes=pd.Series([1, 2])  # Wrong length
            )
            
        # Test invalid optional input types
        with self.assertRaises(TypeError):
            self.loss_fn(
                trade_data=self.trade_data,
                position_sizes=[1, 2, 3]  # Not a Series
            )
            
    def test_error_handling(self):
        """Test error handling and logging."""
        # Test exception propagation
        class ErrorLossFunction(BaseLossFunction):
            def calculate_loss(self, *args, **kwargs):
                raise ValueError("Test error")
                
        loss_fn = ErrorLossFunction("Error Loss", "minimize")
        with self.assertRaises(ValueError):
            loss_fn(self.trade_data)
            
        # Test non-finite value handling
        class InfLossFunction(BaseLossFunction):
            def calculate_loss(self, *args, **kwargs):
                return float('inf')
                
        loss_fn = InfLossFunction("Inf Loss", "minimize")
        result = loss_fn(self.trade_data)
        self.assertTrue(np.isinf(result))
        
    def test_thread_safety(self):
        """Test thread safety of loss function."""
        n_threads = 10
        results = []
        metadata = []
        
        def worker():
            result = self.loss_fn(self.trade_data)
            meta = self.loss_fn.get_metadata()
            results.append(result)
            metadata.append(meta)
            
        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # Check results consistency
        self.assertEqual(len(set(results)), 1)  # All results should be the same
        self.assertEqual(len(metadata), n_threads)  # Each thread should have metadata
        
    def test_performance(self):
        """Test performance benchmarks."""
        # Test large dataset performance
        large_data = create_sample_data(10000)[0]
        start_time = time.time()
        self.loss_fn(large_data)
        duration = time.time() - start_time
        self.assertLess(duration, 1.0)  # Should complete in under 1 second
        
        # Test parallel performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.loss_fn, self.trade_data)
                for _ in range(100)
            ]
            results = [f.result() for f in futures]
        self.assertEqual(len(set(results)), 1)  # All results should be consistent
        
    def test_metadata_safety(self):
        """Test that metadata cannot be modified externally."""
        original_meta = self.loss_fn.get_metadata()
        modified_meta = self.loss_fn.get_metadata()
        modified_meta['new_key'] = 'new_value'
        
        self.assertNotEqual(original_meta, modified_meta)
        self.assertNotIn('new_key', self.loss_fn.get_metadata())
        
    def test_required_columns_validation(self):
        """Test validation of required DataFrame columns."""
        required_cols = ['price', 'volume']
        
        # Test missing columns
        bad_data = self.trade_data.drop(columns=['volume'])
        with self.assertRaises(ValueError):
            self.loss_fn.validate_required_columns(bad_data, required_cols)
            
        # Test with all required columns
        try:
            self.loss_fn.validate_required_columns(self.trade_data, required_cols)
        except ValueError:
            self.fail("validate_required_columns raised ValueError unexpectedly")
            
    def test_numeric_columns_validation(self):
        """Test validation of numeric data types."""
        numeric_cols = ['price', 'volume']
        
        # Test non-numeric data
        bad_data = self.trade_data.copy()
        bad_data['price'] = 'not_numeric'
        with self.assertRaises(ValueError):
            self.loss_fn.validate_numeric_columns(bad_data, numeric_cols)
            
        # Test with numeric data
        try:
            self.loss_fn.validate_numeric_columns(self.trade_data, numeric_cols)
        except ValueError:
            self.fail("validate_numeric_columns raised ValueError unexpectedly")


if __name__ == '__main__':
    unittest.main()
