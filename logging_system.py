"""
Hyperopter Logging System

This module provides a comprehensive logging system for the Hyperopter framework,
handling optimization trial results, metrics, and system events.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
from loguru import logger

class LoggingError(Exception):
    """Base exception class for logging-related errors."""
    pass

class InvalidLogPathError(LoggingError):
    """Raised when the log path is invalid or cannot be created."""
    pass

class InvalidLogFormatError(LoggingError):
    """Raised when the log format is invalid or unsupported."""
    pass

class LoggingSystem:
    """
    A comprehensive logging system for the Hyperopter framework.
    
    Handles logging of optimization trials, metrics, and system events.
    Supports multiple output formats and provides query capabilities.
    
    Attributes:
        log_dir: Directory for storing log files
        logger: Python logger instance for system events
        current_strategy: Current optimization strategy being used
        trial_results: DataFrame storing trial results
    """
    
    VALID_STRATEGIES = {'bayesian', 'grid_search', 'random_search', 'evolutionary'}
    VALID_FORMATS = {'json', 'csv'}
    
    def __init__(self, log_dir: Union[str, Path], strategy: str):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory path for storing log files
            strategy: Optimization strategy being used
            
        Raises:
            InvalidLogPathError: If log_dir is invalid or cannot be created
            ValueError: If strategy is not supported
        """
        self.log_dir = Path(log_dir)
        if not self._validate_and_create_log_dir():
            raise InvalidLogPathError(f"Invalid or inaccessible log directory: {log_dir}")
            
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}. Must be one of {self.VALID_STRATEGIES}")
            
        self.current_strategy = strategy
        self.trial_results = pd.DataFrame()
        
        # Setup system logger
        log_file = self.log_dir / f"hyperopter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, format="{time} - {level} - {message}")
        self.log_file = log_file
        
        logger.info(f"Initialized logging system with strategy: {strategy}")

    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        try:
            logger.remove()  # Remove all handlers
        except:
            pass

    def _validate_and_create_log_dir(self) -> bool:
        """
        Validate and create the log directory if it doesn't exist.
        
        Returns:
            bool: True if directory is valid and accessible, False otherwise
        """
        try:
            # Check if path exists and is a file
            if self.log_dir.exists() and self.log_dir.is_file():
                return False
                
            # Try to create directory
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a test file to verify write permissions
            test_file = self.log_dir / '.test_write'
            test_file.touch()
            test_file.unlink()
            
            return True
                   
        except Exception as e:
            return False

    def log_trial(self, trial_id: int, parameters: Dict[str, Any], 
                 result: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log a single optimization trial.
        
        Args:
            trial_id: Unique identifier for the trial
            parameters: Dictionary of hyperparameters used
            result: Primary optimization metric value
            metrics: Optional dictionary of additional metrics
            
        Raises:
            ValueError: If trial_id is negative or parameters is empty
        """
        if trial_id < 0:
            raise ValueError("Trial ID must be non-negative")
            
        if not parameters:
            raise ValueError("Parameters dictionary cannot be empty")
            
        trial_data = {
            "trial_id": trial_id,
            "timestamp": datetime.now(),
            "strategy": self.current_strategy,
            "result": result,
            **parameters
        }
        
        # Add metrics directly to trial data
        if metrics:
            trial_data.update(metrics)
            
        # Convert to DataFrame and concatenate
        new_trial = pd.DataFrame([trial_data])
        self.trial_results = pd.concat([self.trial_results, new_trial], ignore_index=True)
        
        logger.info(f"Logged trial {trial_id} with result {result}")

    def export_logs(self, format: str = "json") -> Tuple[Path, bool]:
        """
        Export logged trials to a file.
        
        Args:
            format: Export format, either 'json' or 'csv'
            
        Returns:
            Tuple of (export file path, success status)
            
        Raises:
            ValueError: If format is not supported
        """
        if format not in ["json", "csv"]:
            raise ValueError("Export format must be either 'json' or 'csv'")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = self.log_dir / f"trials_{timestamp}.{format}"
        
        try:
            if format == "json":
                self.trial_results.to_json(export_file, orient="records", indent=2)
            else:  # csv
                self.trial_results.to_csv(export_file, index=False)
                
            logger.info(f"Exported logs to {export_file}")
            return export_file, True
            
        except Exception as e:
            logger.error(f"Failed to export logs: {str(e)}")
            return export_file, False

    def get_best_trial(self, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the trial with the best result.
        
        Args:
            maximize: If True, return trial with highest result, else lowest
            
        Returns:
            Dictionary containing the best trial data, or None if no trials exist
        """
        if self.trial_results.empty:
            return None
            
        best_idx = self.trial_results["result"].idxmax() if maximize else self.trial_results["result"].idxmin()
        return self.trial_results.iloc[best_idx].to_dict()

    def get_trial_history(self, metric: Optional[str] = None) -> pd.DataFrame:
        """
        Get history of all trials, optionally filtered by metric.
        
        Args:
            metric: Optional metric to include in history
            
        Returns:
            DataFrame containing trial history
            
        Raises:
            ValueError: If specified metric doesn't exist
        """
        if metric and metric not in self.trial_results.columns:
            raise ValueError(f"Metric '{metric}' not found in trial results")
            
        columns = ["trial_id", "timestamp", "strategy", "result"]
        if metric:
            columns.append(metric)
            
        return self.trial_results[columns].copy()

    def clear_logs(self) -> None:
        """Clear all logged trials and reset the logging system."""
        self.trial_results = pd.DataFrame()
        logger.info("Cleared all logged trials")
