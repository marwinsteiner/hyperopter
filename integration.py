"""Integration script for hyperparameter optimization of trading strategies."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from configuration_manager import ConfigurationManager
from data_handler import DataHandler
from parallel_optimizer import ParallelOptimizer
from results_manager import ResultsManager

class TradingStrategyOptimizer:
    """Manager for hyperparameter optimization of trading strategies."""
    
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 output_dir: Optional[str] = None):
        """
        Initialize the trading strategy optimizer.
        
        Args:
            config_path: Path to JSON configuration file
            data_path: Path to CSV data file
            output_dir: Optional directory for output files
        """
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.data_handler = DataHandler(data_path)
        self.optimizer = ParallelOptimizer(
            n_workers=None,  # Use CPU count
            batch_size=10,
            memory_per_worker=0.2,
            log_dir=str(self.output_dir / "logs")
        )
        self.results_manager = ResultsManager(
            output_dir=str(self.output_dir)
        )
        
        # Setup logging
        log_file = self.output_dir / "optimization.log"
        logger.add(log_file, format="{time} - {level} - {message}")
        
    def _validate_config(self) -> Dict[str, Any]:
        """
        Validate the configuration file.
        
        Returns:
            Dictionary containing validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = self.config_manager.load_config()
            
            # Validate required fields
            required_fields = ["strategy_name", "parameters", "optimization"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate parameter ranges
            for param, settings in config["parameters"].items():
                if not all(k in settings for k in ["type", "range"]):
                    raise ValueError(f"Invalid parameter settings for {param}")
                    
            # Validate optimization settings
            opt_settings = config["optimization"]
            if not all(k in opt_settings for k in ["method", "trials", "timeout"]):
                raise ValueError("Invalid optimization settings")
                
            return config
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
            
    def _validate_data(self) -> pd.DataFrame:
        """
        Validate the data file.
        
        Returns:
            DataFrame containing validated data
            
        Raises:
            ValueError: If data is invalid
        """
        try:
            data = self.data_handler.load_data()
            
            # Check required columns
            required_columns = ["date", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check data types
            for col in data.columns:
                if col == "date":
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col])
                else:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(f"Column {col} must be numeric")
                        
            # Handle missing values
            if data.isnull().any().any():
                logger.warning("Found missing values, forward filling...")
                data = data.ffill()
                
            return data
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
            
    def _calculate_moving_average_signals(self,
                                        data: pd.DataFrame,
                                        short_window: int,
                                        long_window: int) -> pd.DataFrame:
        """
        Calculate trading signals based on moving average crossover.
        
        Args:
            data: Price data
            short_window: Short moving average window
            long_window: Long moving average window
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate moving averages
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()
        
        # Generate signals
        df["signal"] = 0
        df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
        df.loc[df["short_ma"] < df["long_ma"], "signal"] = -1
        
        # Calculate returns
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        
        return df
        
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for a trading strategy.
        
        Args:
            df: DataFrame with strategy returns
            
        Returns:
            Dictionary containing performance metrics
        """
        # Calculate Sharpe ratio
        strategy_returns = df["strategy_returns"].dropna()
        sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
        
    def _objective_function(self, **params) -> float:
        """
        Objective function for optimization.
        
        Args:
            **params: Strategy parameters
            
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        try:
            # Calculate signals and returns
            signals_df = self._calculate_moving_average_signals(
                self.data,
                short_window=params["short_window"],
                long_window=params["long_window"]
            )
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(signals_df)
            
            # Return negative Sharpe ratio for minimization
            return -metrics["sharpe_ratio"]
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return float("inf")  # Return worst possible value
            
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary containing optimization results
        """
        try:
            logger.info("Starting optimization process")
            
            # Validate inputs
            config = self._validate_config()
            self.data = self._validate_data()
            
            # Create parameter space
            parameter_space = {}
            for param, settings in config["parameters"].items():
                if settings["type"] == "integer":
                    parameter_space[param] = lambda p=param, s=settings: np.random.randint(
                        s["range"][0],
                        s["range"][1] + 1
                    )
                else:
                    parameter_space[param] = lambda p=param, s=settings: np.random.uniform(
                        s["range"][0],
                        s["range"][1]
                    )
                    
            # Run optimization
            results = self.optimizer.optimize(
                objective_fn=self._objective_function,
                parameter_space=parameter_space,
                n_trials=config["optimization"]["trials"],
                timeout=config["optimization"]["timeout"]
            )
            
            # Process results
            best_trial = results["best_trial"]
            best_metrics = self._calculate_performance_metrics(
                self._calculate_moving_average_signals(
                    self.data,
                    **best_trial["parameters"]
                )
            )
            
            # Format output
            output = {
                "best_parameters": best_trial["parameters"],
                "performance_metrics": best_metrics,
                "optimization_history": [
                    {
                        "trial": i,
                        "parameters": r["parameters"],
                        "sharpe_ratio": -r["result"] if r["status"] == "completed" else None
                    }
                    for i, r in enumerate(results["all_results"])
                ]
            }
            
            # Save results
            self.results_manager.save_results(output)
            
            logger.info("Optimization completed successfully")
            return output
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
