"""Tests for the integration script."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from integration import TradingStrategyOptimizer

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "strategy_name": "moving_average_crossover",
        "parameters": {
            "short_window": {
                "type": "integer",
                "range": [5, 50],
                "step": 1
            },
            "long_window": {
                "type": "integer",
                "range": [50, 200],
                "step": 1
            }
        },
        "optimization": {
            "method": "TPE",
            "trials": 10,
            "timeout": 60
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

class TestTradingStrategyOptimizer:
    """Test suite for TradingStrategyOptimizer."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample files
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.data_path = os.path.join(self.temp_dir, "data.csv")
        self.output_dir = os.path.join(self.temp_dir, "results")
        
    def test_initialization(self, sample_config, sample_data):
        """Test optimizer initialization."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
        # Initialize optimizer
        optimizer = TradingStrategyOptimizer(
            config_path=self.config_path,
            data_path=self.data_path,
            output_dir=self.output_dir
        )
        
        assert optimizer.config_path == Path(self.config_path)
        assert optimizer.data_path == Path(self.data_path)
        assert optimizer.output_dir == Path(self.output_dir)
        assert Path(self.output_dir).exists()
        
    def test_config_validation(self, sample_config, sample_data):
        """Test configuration validation."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
        # Test valid config
        optimizer = TradingStrategyOptimizer(
            config_path=self.config_path,
            data_path=self.data_path,
            output_dir=self.output_dir
        )
        config = optimizer._validate_config()
        assert config == sample_config
        
        # Test invalid config
        invalid_config = sample_config.copy()
        del invalid_config["strategy_name"]
        with open(self.config_path, "w") as f:
            json.dump(invalid_config, f)
            
        with pytest.raises(ValueError):
            optimizer._validate_config()
            
    def test_data_validation(self, sample_config, sample_data):
        """Test data validation."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
        # Test valid data
        optimizer = TradingStrategyOptimizer(
            config_path=self.config_path,
            data_path=self.data_path,
            output_dir=self.output_dir
        )
        data = optimizer._validate_data()
        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ["date", "open", "high", "low", "close", "volume"])
        
        # Test invalid data
        invalid_data = sample_data.copy()
        invalid_data["close"] = "invalid"
        invalid_data.to_csv(self.data_path, index=False)
        
        with pytest.raises(ValueError):
            optimizer._validate_data()
            
    def test_moving_average_signals(self, sample_config, sample_data):
        """Test moving average signal calculation."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
        optimizer = TradingStrategyOptimizer(
            config_path=self.config_path,
            data_path=self.data_path,
            output_dir=self.output_dir
        )
        
        signals_df = optimizer._calculate_moving_average_signals(
            sample_data,
            short_window=10,
            long_window=50
        )
        
        assert "signal" in signals_df.columns
        assert "strategy_returns" in signals_df.columns
        assert all(s in [-1, 0, 1] for s in signals_df["signal"].unique())
        
    def test_performance_metrics(self, sample_config, sample_data):
        """Test performance metric calculation."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
        optimizer = TradingStrategyOptimizer(
            config_path=self.config_path,
            data_path=self.data_path,
            output_dir=self.output_dir
        )
        
        signals_df = optimizer._calculate_moving_average_signals(
            sample_data,
            short_window=10,
            long_window=50
        )
        metrics = optimizer._calculate_performance_metrics(signals_df)
        
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown"], float)
        
    def test_optimization(self, sample_config, sample_data):
        """Test full optimization process."""
        # Write sample files
        with open(self.config_path, "w") as f:
            json.dump(sample_config, f)
        sample_data.to_csv(self.data_path, index=False)
        
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
        
        with patch("parallel_optimizer.ParallelOptimizer") as mock_optimizer:
            mock_optimizer.return_value.optimize.return_value = mock_result
            
            optimizer = TradingStrategyOptimizer(
                config_path=self.config_path,
                data_path=self.data_path,
                output_dir=self.output_dir
            )
            optimizer.optimizer = mock_optimizer.return_value
            
            results = optimizer.optimize()
            
            assert "best_parameters" in results
            assert "performance_metrics" in results
            assert "optimization_history" in results
            assert len(results["optimization_history"]) == 1
            assert results["best_parameters"] == {"short_window": 10, "long_window": 50}
            
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
