"""Integration script for hyperparameter optimization of trading strategies."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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
                 strategy_evaluator: Callable[[pd.DataFrame, Dict[str, Any]], float],
                 output_dir: Optional[str] = None):
        """
        Initialize the trading strategy optimizer.
        
        Args:
            config_path: Path to JSON configuration file
            data_path: Path to CSV data file
            strategy_evaluator: Function that evaluates strategy performance
            output_dir: Optional directory for output files
        """
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        self.strategy_evaluator = strategy_evaluator
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager = ConfigurationManager()
        self.config_manager.load_configuration(config_path)
        
        # Setup data handler
        data_config = self.config_manager.get_data_handler_config()
        self.data_handler = DataHandler(
            validation_rules=data_config['validation_rules'],
            preprocessing_specs=data_config.get('preprocessing', {}),
            required_columns=data_config.get('required_columns', [])
        )
        
        # Load and validate data
        self.data = self._load_and_validate_data()
        
        # Initialize results manager
        self.results_manager = ResultsManager(output_dir=self.output_dir)
        
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            # Load and validate data
            data = self.data_handler.load_data(str(self.data_path))
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _evaluate_strategy(self, params: Dict[str, Any]) -> float:
        """
        Evaluate strategy with given parameters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Strategy performance metric
        """
        return self.strategy_evaluator(self.data, params)
            
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Get optimization settings
            opt_settings = self.config_manager.get_optimization_settings()
            param_space = self.config_manager.get_parameter_space()
            strategy_config = self.config_manager.get_strategy_config()
            
            # Initialize optimizer
            optimizer = ParallelOptimizer(
                n_workers=min(opt_settings.parallel_trials, 2),  # Limit workers to avoid memory issues
                batch_size=5  # Smaller batch size
            )
            
            # Run optimization
            logger.info("Starting optimization process...")
            results = optimizer.optimize(
                objective_fn=self._evaluate_strategy,
                parameter_space=param_space,
                n_trials=opt_settings.max_iterations,
                timeout=None
            )
            
            # Save results
            self.results_manager.save_results(results)
            logger.info("Optimization completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
            
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters
        """
        return self.results_manager.get_best_trial()

def create_optimizer(
    config_path: str,
    data_path: str,
    strategy_evaluator: Callable[[pd.DataFrame, Dict[str, Any]], float],
    output_dir: Optional[str] = None
) -> TradingStrategyOptimizer:
    """
    Create a TradingStrategyOptimizer instance with the given configuration.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file
        strategy_evaluator: Function to evaluate strategy performance
        output_dir: Optional directory for output files
        
    Returns:
        Configured TradingStrategyOptimizer instance
    """
    return TradingStrategyOptimizer(
        config_path=config_path,
        data_path=data_path,
        strategy_evaluator=strategy_evaluator,
        output_dir=output_dir
    )
