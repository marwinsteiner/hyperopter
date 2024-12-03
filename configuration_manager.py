"""
Configuration Manager Module

This module handles loading, validating, and managing configuration settings for the optimization system.
It ensures configuration compatibility across all components and validates against defined schemas.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import jsonschema
from dataclasses import dataclass
import logging
from enum import Enum
from data_handler import ValidationRule

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass

class OptimizationStrategy(Enum):
    """Supported optimization strategies."""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"

@dataclass
class StrategyConfig:
    """Container for optimization strategy configuration."""
    name: OptimizationStrategy
    parameters: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class OptimizationSettings:
    """Container for optimization settings."""
    max_iterations: int
    convergence_threshold: float
    timeout_seconds: Optional[int]
    parallel_trials: int
    random_seed: Optional[int]

class ConfigurationManager:
    """
    Manages system configuration including parameter spaces and optimization settings.
    
    Attributes:
        logger: Logger instance for tracking operations
        schema_path: Path to JSON schema file
        config_data: Loaded and validated configuration data
    """

    # Default schema for configuration validation
    DEFAULT_SCHEMA = {
        "type": "object",
        "properties": {
            "parameter_space": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["int", "float", "categorical"]},
                        "range": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            ]
                        },
                        "step": {"type": "number"}
                    },
                    "required": ["type", "range"]
                }
            },
            "optimization_settings": {
                "type": "object",
                "properties": {
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "convergence_threshold": {"type": "number", "minimum": 0},
                    "parallel_trials": {"type": "integer", "minimum": 1},
                    "random_seed": {"type": "integer"}
                },
                "required": ["max_iterations", "convergence_threshold", "parallel_trials"]
            },
            "strategy": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parameters": {"type": "object"}
                },
                "required": ["name", "parameters"]
            }
        },
        "required": ["parameter_space", "optimization_settings", "strategy"]
    }

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the ConfigurationManager.
        
        Args:
            schema_path: Optional path to custom JSON schema file
        """
        self.logger = logging.getLogger(__name__)
        self.schema_path = schema_path
        self.config_data: Dict[str, Any] = {}
        
        # Load custom schema if provided, otherwise use default
        if schema_path:
            try:
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading schema: {str(e)}")
                self.schema = self.DEFAULT_SCHEMA
        else:
            self.schema = self.DEFAULT_SCHEMA

    def load_configuration(self, config_path: str) -> None:
        """
        Load and validate configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Raises:
            ConfigurationError: If configuration file cannot be loaded or is invalid
        """
        try:
            if not Path(config_path).exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                self.config_data = json.load(f)
                
            self._validate_schema()
            self._validate_parameter_ranges()
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def _validate_schema(self) -> None:
        """
        Validate configuration against JSON schema.
        
        Raises:
            SchemaValidationError: If configuration doesn't match schema
        """
        try:
            jsonschema.validate(instance=self.config_data, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            raise SchemaValidationError(f"Schema validation failed: {str(e)}")

    def _validate_parameter_ranges(self) -> None:
        """
        Validate parameter ranges and types.
        
        Raises:
            ConfigurationError: If parameter ranges are invalid
        """
        for param, config in self.config_data["parameter_space"].items():
            range_values = config["range"]
            
            if config["type"] in ["int", "float"]:
                if range_values[0] >= range_values[1]:
                    raise ConfigurationError(
                        f"Invalid range for parameter {param}: min value must be less than max value"
                    )
                if "step" in config and config["step"] <= 0:
                    raise ConfigurationError(
                        f"Invalid step size for parameter {param}: must be positive"
                    )
            elif config["type"] == "categorical":
                if not isinstance(range_values, list):
                    raise ConfigurationError(
                        f"Invalid categorical values for parameter {param}: must be a list"
                    )

    def get_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get validated parameter space configuration.
        
        Returns:
            Dictionary containing parameter space configuration
        """
        return self.config_data["parameter_space"]

    def get_optimization_settings(self) -> OptimizationSettings:
        """
        Get optimization settings as a structured object.
        
        Returns:
            OptimizationSettings object containing validated settings
        """
        settings = self.config_data["optimization_settings"]
        return OptimizationSettings(
            max_iterations=settings["max_iterations"],
            convergence_threshold=settings["convergence_threshold"],
            timeout_seconds=settings.get("timeout_seconds"),
            parallel_trials=settings["parallel_trials"],
            random_seed=settings.get("random_seed")
        )

    def get_strategy_config(self) -> StrategyConfig:
        """
        Get strategy configuration as a structured object.
        
        Returns:
            StrategyConfig object containing strategy settings
        """
        strategy = self.config_data["strategy"]
        return StrategyConfig(
            name=OptimizationStrategy(strategy["name"]),
            parameters=strategy["parameters"],
            constraints=strategy.get("constraints")
        )

    def get_data_handler_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to DataHandler component.
        
        Returns:
            Dictionary containing DataHandler configuration including validation rules
            and preprocessing specifications
        """
        # Extract and transform relevant configuration for DataHandler
        validation_rules = {}
        preprocessing_specs = {}
        
        # Map parameter types to validation rules
        for param, config in self.config_data["parameter_space"].items():
            rules = [ValidationRule.REQUIRED]  # All parameters are required
            
            if config["type"] in ["int", "float"]:
                rules.append(ValidationRule.NUMERIC)
            elif config["type"] == "categorical":
                rules.append(ValidationRule.CATEGORICAL)
                
            validation_rules[param] = rules
            
            # Add preprocessing specifications based on parameter type
            if config["type"] in ["int", "float"]:
                if "normalize" not in preprocessing_specs:
                    preprocessing_specs["normalize"] = {"columns": []}
                preprocessing_specs["normalize"]["columns"].append(param)
            elif config["type"] == "categorical":
                if "encode_categorical" not in preprocessing_specs:
                    preprocessing_specs["encode_categorical"] = {"columns": []}
                preprocessing_specs["encode_categorical"]["columns"].append(param)
        
        return {
            "validation_rules": validation_rules,
            "preprocessing_specs": preprocessing_specs
        }

    def validate_compatibility(self, component_name: str) -> bool:
        """
        Validate configuration compatibility with specific component.
        
        Args:
            component_name: Name of the component to validate against
            
        Returns:
            True if configuration is compatible, False otherwise
        """
        try:
            if component_name == "data_handler":
                # Verify that all parameters have valid validation rules
                config = self.get_data_handler_config()
                return all(
                    isinstance(rules, list) and all(isinstance(r, ValidationRule) for r in rules)
                    for rules in config["validation_rules"].values()
                )
            # Add other component compatibility checks as needed
            return True
        except Exception as e:
            self.logger.error(f"Compatibility validation failed: {str(e)}")
            return False
