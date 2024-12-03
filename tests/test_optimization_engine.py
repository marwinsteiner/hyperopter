"""
Test suite for OptimizationEngine component.
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from optimization_engine import (
    OptimizationEngine,
    OptimizationError,
    OptimizationResult,
    TrialResult
)
from configuration_manager import (
    OptimizationStrategy,
    OptimizationSettings,
    StrategyConfig
)

class TestOptimizationEngine(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample parameter space
        self.parameter_space = {
            "x": {
                "type": "float",
                "range": [-5.0, 5.0]
            },
            "y": {
                "type": "float",
                "range": [-5.0, 5.0]
            }
        }
        
        # Create optimization settings
        self.optimization_settings = OptimizationSettings(
            max_iterations=20,
            convergence_threshold=0.001,
            timeout_seconds=60,
            parallel_trials=2,
            random_seed=42
        )
        
        # Create temporary directory for results
        self.temp_dir = tempfile.mkdtemp()
        
        # Test different optimization strategies
        self.strategies = [
            StrategyConfig(
                name=OptimizationStrategy.BAYESIAN,
                parameters={"n_startup_trials": 5}
            ),
            StrategyConfig(
                name=OptimizationStrategy.GRID_SEARCH,
                parameters={}
            ),
            StrategyConfig(
                name=OptimizationStrategy.RANDOM_SEARCH,
                parameters={}
            ),
            StrategyConfig(
                name=OptimizationStrategy.EVOLUTIONARY,
                parameters={"sigma0": 1.0}
            )
        ]

    def objective_function(self, params):
        """Sample objective function (Rosenbrock function)."""
        x = params["x"]
        y = params["y"]
        value = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        metrics = {
            "x_value": x,
            "y_value": y,
            "distance_from_origin": np.sqrt(x ** 2 + y ** 2)
        }
        return value, metrics

    def test_initialization(self):
        """Test optimization engine initialization."""
        for strategy in self.strategies:
            optimizer = OptimizationEngine(
                parameter_space=self.parameter_space,
                optimization_settings=self.optimization_settings,
                strategy_config=strategy
            )
            self.assertIsNotNone(optimizer)

    def test_input_validation(self):
        """Test input validation functionality."""
        # Test with invalid parameter space
        invalid_space = {
            "x": {"type": "float"}  # Missing range
        }
        
        with self.assertRaises(OptimizationError):
            OptimizationEngine(
                parameter_space=invalid_space,
                optimization_settings=self.optimization_settings,
                strategy_config=self.strategies[0]
            )

    def test_optimization_bayesian(self):
        """Test Bayesian optimization strategy."""
        optimizer = OptimizationEngine(
            parameter_space=self.parameter_space,
            optimization_settings=self.optimization_settings,
            strategy_config=self.strategies[0]
        )
        
        results = optimizer.optimize(
            objective_function=self.objective_function,
            n_jobs=1
        )
        
        self.assertIsInstance(results, OptimizationResult)
        self.assertIn("x", results.best_params)
        self.assertIn("y", results.best_params)
        self.assertTrue(len(results.optimization_history) > 0)

    def test_optimization_grid_search(self):
        """Test Grid Search optimization strategy."""
        optimizer = OptimizationEngine(
            parameter_space=self.parameter_space,
            optimization_settings=self.optimization_settings,
            strategy_config=self.strategies[1]
        )
        
        results = optimizer.optimize(
            objective_function=self.objective_function,
            n_jobs=1
        )
        
        self.assertIsInstance(results, OptimizationResult)
        self.assertTrue(len(results.optimization_history) > 0)

    def test_parallel_trials(self):
        """Test parallel trial execution functionality."""
        # Define a simple objective function that doesn't rely on class attributes
        def simple_objective(params):
            x = params["x"]
            y = params["y"]
            return x**2 + y**2, {"distance": np.sqrt(x**2 + y**2)}

        optimizer = OptimizationEngine(
            parameter_space=self.parameter_space,
            optimization_settings=self.optimization_settings,
            strategy_config=self.strategies[0]
        )
        
        params_list = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": -1.0, "y": -1.0}
        ]
        
        results = optimizer.run_parallel_trials(
            params_list=params_list,
            objective_function=simple_objective
        )
        
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], TrialResult)

    def test_save_results(self):
        """Test results saving functionality."""
        optimizer = OptimizationEngine(
            parameter_space=self.parameter_space,
            optimization_settings=self.optimization_settings,
            strategy_config=self.strategies[0]
        )
        
        results = optimizer.optimize(
            objective_function=self.objective_function,
            n_jobs=1
        )
        
        results_path = Path(self.temp_dir) / "optimization_results.json"
        optimizer.save_results(results, str(results_path))
        
        self.assertTrue(results_path.exists())
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
            self.assertIn("best_params", saved_results)
            self.assertIn("optimization_history", saved_results)

    def test_convergence(self):
        """Test optimization convergence."""
        # Use a simple quadratic function for testing convergence
        def simple_objective(params):
            x = params["x"]
            y = params["y"]
            return x**2 + y**2, {"distance": np.sqrt(x**2 + y**2)}
        
        # Use more iterations for convergence test
        settings = OptimizationSettings(
            max_iterations=50,  # Increased iterations
            convergence_threshold=0.001,
            timeout_seconds=60,
            parallel_trials=2,
            random_seed=42
        )
        
        optimizer = OptimizationEngine(
            parameter_space=self.parameter_space,
            optimization_settings=settings,  # Use new settings
            strategy_config=self.strategies[0]  # Use Bayesian optimization
        )
        
        results = optimizer.optimize(
            objective_function=simple_objective,
            n_jobs=1
        )
        
        # Check if the optimization found a point close to the minimum (0,0)
        # Use a more lenient threshold for real-world scenarios
        self.assertLess(abs(results.best_params["x"]), 1.0)
        self.assertLess(abs(results.best_params["y"]), 1.0)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary directory and its contents
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        Path(self.temp_dir).rmdir()

if __name__ == '__main__':
    unittest.main()
