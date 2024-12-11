"""Tests for the integration script."""

import json
import os
import tempfile
from pathlib import Path
import time
import shutil
import pytest
from unittest.mock import patch

import numpy as np
import pandas as pd

from integration import TradingStrategyOptimizer

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "parameter_space": {
            "short_window": {
                "type": "int",
                "range": [5, 50],
                "step": 1
            },
            "long_window": {
                "type": "int",
                "range": [50, 200],
                "step": 1
            }
        },
        "optimization_settings": {
            "max_iterations": 10,
            "convergence_threshold": 0.001,
            "parallel_trials": 2,
            "random_seed": 42
        },
        "strategy": {
            "name": "bayesian",  # Use a valid OptimizationStrategy enum value
            "parameters": {
                "acquisition_function": "expected_improvement",
                "exploration_weight": 0.1
            }
        },
        "data_handler": {
            "validation_rules": {
                "close": ["required", "numeric"],
                "date": ["required", "date"],
                "open": ["numeric"],
                "high": ["numeric"],
                "low": ["numeric"],
                "volume": ["numeric"]
            },
            "preprocessing": {},
            "required_columns": ["date", "open", "high", "low", "close", "volume"]
        }
    }

@pytest.fixture
def sample_data():
    """Create a sample price data DataFrame."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
        "volume": np.random.uniform(1000000, 5000000, 100)
    })
    return data

@pytest.fixture
def strategy_evaluator():
    """Create a mock strategy evaluator function."""
    def evaluator(data, params):
        """Simple moving average crossover strategy evaluator."""
        short_window = params["short_window"]
        long_window = params["long_window"]
        
        # Calculate moving averages
        short_ma = data["close"].rolling(window=short_window).mean()
        long_ma = data["close"].rolling(window=long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        # Calculate returns
        returns = data["close"].pct_change() * signals.shift(1)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        return -sharpe_ratio  # Negative because we want to maximize Sharpe ratio
    
    return evaluator

@pytest.fixture
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up temporary files
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.1)  # Wait briefly before retrying
            else:
                print(f"Warning: Could not delete temp directory {temp_dir}")

@pytest.fixture
def test_paths(test_dir):
    """Create test file paths."""
    return {
        "config": Path(test_dir) / "config.json",
        "data": Path(test_dir) / "data.csv",
        "output": Path(test_dir) / "results"
    }

@pytest.fixture
def setup_files(test_paths):
    """Set up test files with fixture data."""
    def _setup_files(sample_config, sample_data):
        with open(test_paths["config"], "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(test_paths["data"], index=False)
    return _setup_files

class TestTradingStrategyOptimizer:
    """Test suite for TradingStrategyOptimizer."""

    def test_initialization(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test optimizer initialization."""
        setup_files(sample_config, sample_data)
        
        optimizer = TradingStrategyOptimizer(
            config_path=test_paths["config"],
            data_path=test_paths["data"],
            strategy_evaluator=strategy_evaluator,
            output_dir=test_paths["output"]
        )
        
        assert optimizer.config_manager is not None
        assert optimizer.data is not None
        assert optimizer.output_dir == test_paths["output"]
        assert test_paths["output"].exists()
        
    def test_config_validation(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test configuration validation."""
        setup_files(sample_config, sample_data)
        
        # Test valid config
        optimizer = TradingStrategyOptimizer(
            config_path=test_paths["config"],
            data_path=test_paths["data"],
            strategy_evaluator=strategy_evaluator,
            output_dir=test_paths["output"]
        )
        
        # Verify configuration loaded correctly
        assert optimizer.config_manager.config_data is not None
        
        # Test invalid config
        invalid_config = sample_config.copy()
        del invalid_config["strategy"]
        with open(test_paths["config"], "w") as f:
            json.dump(invalid_config, f)
            
        with pytest.raises(Exception):
            TradingStrategyOptimizer(
                config_path=test_paths["config"],
                data_path=test_paths["data"],
                strategy_evaluator=strategy_evaluator,
                output_dir=test_paths["output"]
            )

    def test_data_validation(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test data validation."""
        setup_files(sample_config, sample_data)

        # Test valid data
        optimizer = TradingStrategyOptimizer(
            config_path=test_paths["config"],
            data_path=test_paths["data"],
            strategy_evaluator=strategy_evaluator,
            output_dir=test_paths["output"]
        )
        assert optimizer.data is not None
        assert isinstance(optimizer.data, pd.DataFrame)

        # Test invalid data
        invalid_data = sample_data.copy()
        invalid_data.drop(columns=["close"], inplace=True)
        invalid_data.to_csv(test_paths["data"], index=False)

        # Create a new optimizer instance with invalid data
        with pytest.raises(ValueError, match="Error loading data: Missing required columns: {'close'}"):
            TradingStrategyOptimizer(
                config_path=test_paths["config"],
                data_path=test_paths["data"],
                strategy_evaluator=strategy_evaluator,
                output_dir=test_paths["output"]
            )

    def test_moving_average_signals(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test moving average signal calculation."""
        setup_files(sample_config, sample_data)
        
        optimizer = TradingStrategyOptimizer(
            config_path=test_paths["config"],
            data_path=test_paths["data"],
            strategy_evaluator=strategy_evaluator,
            output_dir=test_paths["output"]
        )

        # Test strategy evaluation
        result = optimizer._evaluate_strategy({
            "short_window": 10,
            "long_window": 50
        })
        assert isinstance(result, float)

    def test_performance_metrics(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test performance metric calculation."""
        setup_files(sample_config, sample_data)
        
        optimizer = TradingStrategyOptimizer(
            config_path=test_paths["config"],
            data_path=test_paths["data"],
            strategy_evaluator=strategy_evaluator,
            output_dir=test_paths["output"]
        )

        # Test strategy evaluation with different parameters
        result1 = optimizer._evaluate_strategy({
            "short_window": 5,
            "long_window": 20
        })
        result2 = optimizer._evaluate_strategy({
            "short_window": 10,
            "long_window": 50
        })
        
        assert isinstance(result1, float)
        assert isinstance(result2, float)
        assert result1 != result2  # Different parameters should give different results

    def test_optimization(self, test_paths, sample_config, sample_data, strategy_evaluator, setup_files):
        """Test full optimization process."""
        setup_files(sample_config, sample_data)

        # Mock optimizer components
        mock_result = {
            "status": "completed",
            "total_trials": 1,
            "completed_trials": 1,
            "best_trial": {
                "parameters": {"short_window": 10, "long_window": 50},
                "result": -1.5  # Negative because we minimize
            },
            "all_results": [{
                "parameters": {"short_window": 10, "long_window": 50},
                "result": -1.5,
                "status": "completed"
            }]
        }

        with patch("integration.ParallelOptimizer") as mock_optimizer:
            mock_instance = mock_optimizer.return_value
            mock_instance.optimize.return_value = mock_result

            optimizer = TradingStrategyOptimizer(
                config_path=test_paths["config"],
                data_path=test_paths["data"],
                strategy_evaluator=strategy_evaluator,
                output_dir=test_paths["output"]
            )

            results = optimizer.optimize()

            assert results["best_trial"] is not None
            assert "parameters" in results["best_trial"]
            assert results["completed_trials"] == 1
            assert results["total_trials"] == 1
