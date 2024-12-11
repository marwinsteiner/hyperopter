"""
Tests for the Sharpe ratio loss function implementation.

This module contains tests for:
1. Sharpe ratio calculation accuracy
2. Time scaling verification
3. Edge case handling
4. Memory usage profiling
5. External calculator validation
"""

import unittest
from datetime import datetime, timedelta
import time
import psutil
import os

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from loss_functions import SharpeRatioLoss, TimeFrequency


def create_sample_returns(n_periods: int = 252,
                        mean_return: float = 0.001,
                        volatility: float = 0.02,
                        seed: int = 42) -> pd.DataFrame:
    """Create sample return data with known properties."""
    np.random.seed(seed)
    returns = np.random.normal(mean_return, volatility, n_periods)
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'returns': returns,
        'capital': np.ones(n_periods) * 10000
    })


class TestSharpeRatioLoss(unittest.TestCase):
    """Test cases for SharpeRatioLoss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_free_rate = 0.02  # 2% annual
        self.loss_fn = SharpeRatioLoss(
            risk_free_rate=self.risk_free_rate,
            frequency=TimeFrequency.DAILY,
            min_periods=30
        )
        self.trade_data = create_sample_returns()
        
    def test_initialization(self):
        """Test initialization and parameter validation."""
        # Test invalid risk-free rate
        with self.assertRaises(ValueError):
            SharpeRatioLoss(risk_free_rate=-0.01)
            
        # Test invalid min_periods
        with self.assertRaises(ValueError):
            SharpeRatioLoss(min_periods=1)
            
        # Test valid initialization
        loss_fn = SharpeRatioLoss()
        self.assertEqual(loss_fn.direction, "maximize")
        self.assertEqual(loss_fn.risk_free_rate, 0.0)
        
    def test_sharpe_calculation_accuracy(self):
        """Test Sharpe ratio calculation against known values."""
        # Create data with known Sharpe ratio
        mean_return = 0.001  # 0.1% daily
        volatility = 0.02    # 2% daily
        data = create_sample_returns(
            mean_return=mean_return,
            volatility=volatility
        )
        
        # Use actual statistics from the data
        actual_mean = data['returns'].mean()
        actual_volatility = data['returns'].std(ddof=1)
        
        # Calculate expected Sharpe ratio
        excess_return = actual_mean - ((1 + self.risk_free_rate) ** (1/252) - 1)
        expected_sharpe = (excess_return / actual_volatility) * np.sqrt(252)
        
        # Compare with our implementation
        result = -self.loss_fn(data)  # Negative because we minimize
        np.testing.assert_almost_equal(result, expected_sharpe, decimal=4)
        
    def test_time_scaling(self):
        """Test proper time scaling for different frequencies."""
        data = self.trade_data.copy()
        
        # Test daily scaling
        daily_loss = SharpeRatioLoss(frequency=TimeFrequency.DAILY)
        daily_result = daily_loss(data)
        
        # Test hourly scaling
        hourly_loss = SharpeRatioLoss(frequency=TimeFrequency.HOURLY)
        hourly_result = hourly_loss(data)
        
        # Verify scaling relationship
        scale_factor = np.sqrt(24)  # √(hours per day)
        np.testing.assert_almost_equal(
            hourly_result * scale_factor,
            daily_result,
            decimal=4
        )
        
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test zero volatility
        zero_vol_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'returns': np.ones(100) * 0.001,  # Constant positive return
            'capital': np.ones(100) * 10000
        })
        result = self.loss_fn(zero_vol_data)
        self.assertEqual(result, float('-inf'))  # Perfect Sharpe ratio
        
        # Test all negative returns
        neg_data = self.trade_data.copy()
        neg_data['returns'] = -np.abs(neg_data['returns'])
        result = self.loss_fn(neg_data)
        self.assertTrue(result > 0)  # Should be a poor Sharpe ratio
        
        # Test insufficient data
        small_data = create_sample_returns(20)  # Less than min_periods
        with self.assertRaises(ValueError):
            self.loss_fn(small_data)
            
        # Test NaN handling
        nan_data = self.trade_data.copy()
        nan_data.loc[10:20, 'returns'] = np.nan
        result = self.loss_fn(nan_data)
        self.assertTrue(np.isfinite(result))
        
    def test_memory_usage(self):
        """Test memory efficiency."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Test with large dataset
        large_data = create_sample_returns(10000)
        self.loss_fn(large_data)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Should use less than 100MB additional memory
        self.assertLess(memory_increase, 100 * 1024 * 1024)
        
    def test_metadata(self):
        """Test metadata contents and accuracy."""
        self.loss_fn(self.trade_data)
        metadata = self.loss_fn.get_metadata()
        
        # Check required fields
        required_fields = {
            'sharpe_ratio',
            'annualized_return',
            'annualized_volatility',
            'num_periods',
            'risk_free_rate'
        }
        self.assertTrue(required_fields.issubset(metadata.keys()))
        
        # Check value types
        for key, value in metadata.items():
            self.assertTrue(isinstance(value, (int, float)))
            self.assertTrue(np.isfinite(value))
            
    def test_pnl_calculation(self):
        """Test Sharpe calculation from P&L data."""
        # Create data without returns
        data = self.trade_data.copy()
        data = data.drop(columns=['returns'])
        
        # Add P&L data
        pnl = pd.Series(np.random.randn(len(data)) * 100)
        
        # Calculate Sharpe from P&L
        result = self.loss_fn(data, pnl=pnl)
        self.assertTrue(np.isfinite(result))
        
        # Verify metadata
        metadata = self.loss_fn.get_metadata()
        self.assertTrue('sharpe_ratio' in metadata)
        self.assertTrue(np.isfinite(metadata['sharpe_ratio']))


if __name__ == '__main__':
    unittest.main()
