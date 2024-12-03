"""Results manager for hyperparameter optimization framework."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
from loguru import logger

class ResultsError(Exception):
    """Base exception class for results-related errors."""
    pass

class InvalidResultsError(ResultsError):
    """Raised when results data is invalid or incomplete."""
    pass

class ResultsManager:
    """Manager for handling optimization results and generating reports."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the results manager.
        
        Args:
            output_dir: Directory for storing results and reports
            
        Raises:
            InvalidResultsError: If output directory is invalid or inaccessible
        """
        self.output_dir = Path(output_dir)
        if not self._validate_and_create_dir():
            raise InvalidResultsError(f"Invalid or inaccessible output directory: {output_dir}")
            
        self.results_data = pd.DataFrame()
        self.config_metadata = {}
        
        # Setup logger
        log_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, format="{time} - {level} - {message}")
        
        logger.info("Initialized results manager")
        
    def _validate_and_create_dir(self) -> bool:
        """
        Validate and create the output directory if it doesn't exist.
        
        Returns:
            bool: True if directory is valid and accessible, False otherwise
        """
        try:
            # Check if path exists and is a file
            if self.output_dir.exists() and self.output_dir.is_file():
                return False
                
            # Try to create directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify write permissions
            test_file = self.output_dir / '.test_write'
            test_file.touch()
            test_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating output directory: {str(e)}")
            return False
            
    def add_optimization_result(self, trial_id: int, parameters: Dict[str, Any],
                              metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Add a single optimization trial result.
        
        Args:
            trial_id: Unique identifier for the trial
            parameters: Dictionary of hyperparameters used
            metrics: Dictionary of performance metrics
            timestamp: Optional timestamp for the result
            
        Raises:
            InvalidResultsError: If result data is invalid
        """
        if trial_id < 0:
            raise InvalidResultsError("Trial ID must be non-negative")
            
        if not parameters:
            raise InvalidResultsError("Parameters dictionary cannot be empty")
            
        if not metrics:
            raise InvalidResultsError("Metrics dictionary cannot be empty")
            
        result_data = {
            "trial_id": trial_id,
            "timestamp": timestamp or datetime.now(),
            **parameters,
            **metrics
        }
        
        # Add to DataFrame
        new_result = pd.DataFrame([result_data])
        self.results_data = pd.concat([self.results_data, new_result], ignore_index=True)
        
        logger.info(f"Added result for trial {trial_id}")
        
    def add_config_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Add configuration metadata.
        
        Args:
            metadata: Dictionary of configuration metadata
            
        Raises:
            InvalidResultsError: If metadata is invalid
        """
        if not metadata:
            raise InvalidResultsError("Metadata dictionary cannot be empty")
            
        self.config_metadata.update(metadata)
        logger.info("Updated configuration metadata")
        
    def export_results(self, format: str = "json") -> Path:
        """
        Export results to a file.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to the exported file
            
        Raises:
            InvalidResultsError: If format is invalid or export fails
        """
        if format not in ["json", "csv"]:
            raise InvalidResultsError("Export format must be either 'json' or 'csv'")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = self.output_dir / f"results_{timestamp}.{format}"
        
        try:
            # Prepare full results dictionary
            full_results = {
                "metadata": self.config_metadata,
                "results": self.results_data.to_dict(orient="records")
            }
            
            # Export based on format
            if format == "json":
                with open(export_file, 'w') as f:
                    json.dump(full_results, f, indent=2, default=str)
            else:  # csv
                self.results_data.to_csv(export_file, index=False)
                
            logger.info(f"Exported results to {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            raise InvalidResultsError(f"Failed to export results: {str(e)}")
            
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of optimization results.
        
        Returns:
            Dictionary containing summary statistics
            
        Raises:
            InvalidResultsError: If no results exist
        """
        if self.results_data.empty:
            raise InvalidResultsError("No results available for summary")
            
        # Calculate summary statistics
        metrics = [col for col in self.results_data.columns 
                  if col not in ["trial_id", "timestamp"]]
                  
        summary = {
            "total_trials": len(self.results_data),
            "timestamp": datetime.now(),
            "metrics": {}
        }
        
        for metric in metrics:
            if pd.api.types.is_numeric_dtype(self.results_data[metric]):
                summary["metrics"][metric] = {
                    "mean": float(self.results_data[metric].mean()),
                    "std": float(self.results_data[metric].std()),
                    "min": float(self.results_data[metric].min()),
                    "max": float(self.results_data[metric].max())
                }
                
        logger.info("Generated results summary")
        return summary
        
    def clear_results(self) -> None:
        """Clear all results data and metadata."""
        self.results_data = pd.DataFrame()
        self.config_metadata = {}
        logger.info("Cleared all results data")
