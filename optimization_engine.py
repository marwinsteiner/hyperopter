"""
Optimization Engine Module

This module implements various optimization strategies including Bayesian, Grid Search,
Random Search, and Evolutionary optimization methods. It integrates with the configuration
manager and data handler to perform hyperparameter optimization.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional, Union
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import concurrent.futures
from configuration_manager import OptimizationStrategy, OptimizationSettings, StrategyConfig

class OptimizationError(Exception):
    """Custom exception for optimization-related errors."""
    pass

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_value: float
    optimization_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    execution_time: float

@dataclass
class TrialResult:
    """Container for individual trial results."""
    params: Dict[str, Any]
    value: float
    metrics: Dict[str, float]

class OptimizationEngine:
    """
    Manages hyperparameter optimization using various strategies.
    
    Attributes:
        logger: Logger instance for tracking operations
        parameter_space: Dictionary defining parameter search space
        optimization_settings: Settings for optimization process
        strategy_config: Configuration for optimization strategy
    """
    
    def __init__(
        self,
        parameter_space: Dict[str, Dict[str, Any]],
        optimization_settings: OptimizationSettings,
        strategy_config: StrategyConfig
    ):
        """
        Initialize the OptimizationEngine.
        
        Args:
            parameter_space: Dictionary defining parameter search space
            optimization_settings: Settings for optimization process
            strategy_config: Configuration for optimization strategy
        """
        self.logger = logging.getLogger(__name__)
        self.parameter_space = parameter_space
        self.optimization_settings = optimization_settings
        self.strategy_config = strategy_config
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validate input parameters and configuration.
        
        Raises:
            OptimizationError: If inputs are invalid
        """
        try:
            # Validate parameter space
            if not self.parameter_space:
                raise OptimizationError("Parameter space cannot be empty")
                
            for param, config in self.parameter_space.items():
                if not isinstance(config, dict):
                    raise OptimizationError(f"Invalid configuration for parameter: {param}")
                if "type" not in config or "range" not in config:
                    raise OptimizationError(f"Missing required fields for parameter: {param}")
                    
            # Validate optimization settings
            if self.optimization_settings.max_iterations < 1:
                raise OptimizationError("max_iterations must be positive")
            if self.optimization_settings.convergence_threshold <= 0:
                raise OptimizationError("convergence_threshold must be positive")
            if self.optimization_settings.parallel_trials < 1:
                raise OptimizationError("parallel_trials must be positive")
                
            # Validate strategy configuration
            if not isinstance(self.strategy_config.name, OptimizationStrategy):
                raise OptimizationError("Invalid optimization strategy")
                
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise

    def _create_optuna_study(self) -> optuna.Study:
        """
        Create an Optuna study with appropriate sampler based on strategy.
        
        Returns:
            Configured Optuna study object
        """
        if self.strategy_config.name == OptimizationStrategy.BAYESIAN:
            sampler = TPESampler(
                seed=self.optimization_settings.random_seed
            )
        elif self.strategy_config.name == OptimizationStrategy.RANDOM_SEARCH:
            sampler = RandomSampler(
                seed=self.optimization_settings.random_seed
            )
        elif self.strategy_config.name == OptimizationStrategy.GRID_SEARCH:
            search_space = self._create_grid_search_space()
            sampler = GridSampler(search_space)
        else:  # Evolutionary
            sampler = optuna.samplers.CmaEsSampler(
                seed=self.optimization_settings.random_seed,
                **self.strategy_config.parameters
            )
            
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
        )
        
        return optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="minimize"
        )

    def _create_grid_search_space(self) -> Dict[str, List[Any]]:
        """
        Create grid search space from parameter configurations.
        
        Returns:
            Dictionary mapping parameter names to their possible values
        """
        search_space = {}
        for param, config in self.parameter_space.items():
            if config["type"] in ["int", "float"]:
                start, end = config["range"]
                step = config.get("step", (end - start) / 10)
                values = np.arange(start, end + step, step)
                if config["type"] == "int":
                    values = values.astype(int)
                search_space[param] = values.tolist()
            else:  # categorical
                search_space[param] = config["range"]
        return search_space

    def _create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create parameter suggestions for an Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        for param, config in self.parameter_space.items():
            if config["type"] == "int":
                params[param] = trial.suggest_int(
                    param,
                    int(config["range"][0]),
                    int(config["range"][1]),
                    step=config.get("step", 1)
                )
            elif config["type"] == "float":
                params[param] = trial.suggest_float(
                    param,
                    float(config["range"][0]),
                    float(config["range"][1]),
                    step=config.get("step")
                )
            else:  # categorical
                params[param] = trial.suggest_categorical(
                    param,
                    config["range"]
                )
        return params

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], Union[float, Tuple[float, Dict[str, float]]]],
        n_jobs: int = -1
    ) -> OptimizationResult:
        """
        Run optimization process using configured strategy.
        
        Args:
            objective_function: Function to minimize, should return either a float value
                              or a tuple of (float value, metrics dict)
            n_jobs: Number of parallel jobs, -1 for all available cores
            
        Returns:
            OptimizationResult containing best parameters and optimization history
        """
        start_time = datetime.now()
        
        try:
            study = self._create_optuna_study()
            
            def objective(trial: optuna.Trial) -> float:
                params = self._create_trial_params(trial)
                result = objective_function(params)
                
                if isinstance(result, tuple):
                    value, metrics = result
                    # Store metrics in trial user attributes
                    for metric_name, metric_value in metrics.items():
                        trial.set_user_attr(f"metric_{metric_name}", metric_value)
                else:
                    value = result
                    
                return value

            # Run optimization
            study.optimize(
                objective,
                n_trials=self.optimization_settings.max_iterations,
                timeout=self.optimization_settings.timeout_seconds,
                n_jobs=n_jobs,
                gc_after_trial=True
            )
            
            # Collect optimization history
            history = []
            for trial in study.trials:
                trial_metrics = {
                    k.replace("metric_", ""): v
                    for k, v in trial.user_attrs.items()
                    if k.startswith("metric_")
                }
                history.append({
                    "params": trial.params,
                    "value": trial.value,
                    "metrics": trial_metrics
                })
            
            # Calculate performance metrics
            performance_metrics = {
                "mean_value": np.mean([t.value for t in study.trials]),
                "std_value": np.std([t.value for t in study.trials]),
                "convergence_rate": len(study.trials) / self.optimization_settings.max_iterations
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                best_params=study.best_params,
                best_value=study.best_value,
                optimization_history=history,
                performance_metrics=performance_metrics,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            raise OptimizationError(f"Optimization failed: {str(e)}")

    def run_parallel_trials(
        self,
        params_list: List[Dict[str, Any]],
        objective_function: Callable[[Dict[str, Any]], Union[float, Tuple[float, Dict[str, float]]]]
    ) -> List[TrialResult]:
        """
        Run multiple trials in parallel for parameter combinations.
        
        Args:
            params_list: List of parameter dictionaries to evaluate
            objective_function: Function to minimize
            
        Returns:
            List of TrialResult objects
        """
        def _run_single_trial(params):
            try:
                result = objective_function(params)
                if isinstance(result, tuple):
                    value, metrics = result
                else:
                    value = result
                    metrics = {}
                return TrialResult(params=params, value=value, metrics=metrics)
            except Exception as e:
                self.logger.error(f"Trial error with params {params}: {str(e)}")
                return None

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.optimization_settings.parallel_trials, len(params_list))
        ) as executor:
            future_results = list(executor.map(_run_single_trial, params_list))
            results = [r for r in future_results if r is not None]
                    
        return results

    def save_results(self, results: OptimizationResult, filepath: str) -> None:
        """
        Save optimization results to JSON file.
        
        Args:
            results: OptimizationResult object to save
            filepath: Path to save results to
        """
        try:
            output = {
                "best_params": results.best_params,
                "best_value": results.best_value,
                "optimization_history": results.optimization_history,
                "performance_metrics": results.performance_metrics,
                "execution_time": results.execution_time,
                "optimization_settings": {
                    "max_iterations": self.optimization_settings.max_iterations,
                    "convergence_threshold": self.optimization_settings.convergence_threshold,
                    "timeout_seconds": self.optimization_settings.timeout_seconds,
                    "parallel_trials": self.optimization_settings.parallel_trials,
                    "random_seed": self.optimization_settings.random_seed
                },
                "strategy": {
                    "name": self.strategy_config.name.value,
                    "parameters": self.strategy_config.parameters
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2)
                
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
