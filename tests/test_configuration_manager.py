"""
Test suite for ConfigurationManager component.
"""

import unittest
import tempfile
import json
from pathlib import Path
from configuration_manager import (
    ConfigurationManager,
    ConfigurationError,
    SchemaValidationError,
    OptimizationStrategy,
    OptimizationSettings,
    StrategyConfig
)

class TestConfigurationManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample configuration with simpler categorical values
        self.sample_config = {
            "parameter_space": {
                "learning_rate": {
                    "type": "float",
                    "range": [0.0001, 0.1],
                    "step": 0.0001
                },
                "num_layers": {
                    "type": "int",
                    "range": [1, 5],
                    "step": 1
                },
                "activation": {
                    "type": "categorical",
                    "range": ["relu", "tanh"]  # Reduced list
                }
            },
            "optimization_settings": {
                "max_iterations": 100,
                "convergence_threshold": 0.001,
                "timeout_seconds": 3600,
                "parallel_trials": 4,
                "random_seed": 42
            },
            "strategy": {
                "name": "bayesian",
                "parameters": {
                    "n_startup_trials": 10
                }
            }
        }
        
        # Save sample configuration
        self.config_path = Path(self.temp_dir) / "test_config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.sample_config, f)
            
        # Initialize ConfigurationManager
        self.config_manager = ConfigurationManager()

    def test_load_configuration(self):
        """Test configuration loading functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        self.assertIsNotNone(self.config_manager.config_data)
        self.assertEqual(
            len(self.config_manager.config_data["parameter_space"]), 
            3
        )

    def test_validate_schema(self):
        """Test schema validation functionality."""
        # Should not raise any exceptions
        self.config_manager.load_configuration(str(self.config_path))
        
        # Test with invalid configuration
        invalid_config = self.sample_config.copy()
        del invalid_config["parameter_space"]
        invalid_path = Path(self.temp_dir) / "invalid_config.json"
        with open(invalid_path, 'w') as f:
            json.dump(invalid_config, f)
            
        with self.assertRaises(SchemaValidationError):
            self.config_manager.load_configuration(str(invalid_path))

    def test_validate_parameter_ranges(self):
        """Test parameter range validation functionality."""
        # Test with invalid ranges
        invalid_config = self.sample_config.copy()
        invalid_config["parameter_space"]["learning_rate"]["range"] = [0.1, 0.0001]
        invalid_path = Path(self.temp_dir) / "invalid_ranges.json"
        with open(invalid_path, 'w') as f:
            json.dump(invalid_config, f)
            
        with self.assertRaises(ConfigurationError):
            self.config_manager.load_configuration(str(invalid_path))

    def test_get_parameter_space(self):
        """Test parameter space retrieval functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        param_space = self.config_manager.get_parameter_space()
        
        self.assertEqual(len(param_space), 3)
        self.assertIn("learning_rate", param_space)
        self.assertIn("num_layers", param_space)
        self.assertIn("activation", param_space)

    def test_get_optimization_settings(self):
        """Test optimization settings retrieval functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        settings = self.config_manager.get_optimization_settings()
        
        self.assertIsInstance(settings, OptimizationSettings)
        self.assertEqual(settings.max_iterations, 100)
        self.assertEqual(settings.convergence_threshold, 0.001)
        self.assertEqual(settings.parallel_trials, 4)

    def test_get_strategy_config(self):
        """Test strategy configuration retrieval functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        strategy = self.config_manager.get_strategy_config()
        
        self.assertIsInstance(strategy, StrategyConfig)
        self.assertEqual(strategy.name, OptimizationStrategy.BAYESIAN)
        self.assertIn("n_startup_trials", strategy.parameters)

    def test_get_data_handler_config(self):
        """Test data handler configuration retrieval functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        config = self.config_manager.get_data_handler_config()
        
        self.assertIn("validation_rules", config)
        self.assertIn("preprocessing_specs", config)
        self.assertIn("learning_rate", config["validation_rules"])

    def test_validate_compatibility(self):
        """Test component compatibility validation functionality."""
        self.config_manager.load_configuration(str(self.config_path))
        
        # Test data handler compatibility
        self.assertTrue(
            self.config_manager.validate_compatibility("data_handler")
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        try:
            # Remove all files in the temporary directory
            if Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {str(e)}")

if __name__ == '__main__':
    unittest.main()
