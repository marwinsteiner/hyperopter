"""Parallel optimizer for distributed hyperparameter optimization."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import os
import time
import numpy as np

from loguru import logger

from results_manager import ResultsManager

@dataclass
class WorkerConfig:
    """Configuration for a worker process."""
    worker_id: int
    batch_size: int
    memory_limit: float
    log_dir: Path

class ParallelError(Exception):
    """Base exception class for parallel optimization errors."""
    pass

class WorkerInitError(ParallelError):
    """Raised when worker initialization fails."""
    pass

class TaskDistributionError(ParallelError):
    """Raised when task distribution fails."""
    pass

class ResultCollectionError(ParallelError):
    """Raised when result collection fails."""
    pass

class ParallelOptimizer:
    """Manager for parallel hyperparameter optimization."""
    
    def __init__(self,
                 n_workers: int = None,
                 batch_size: int = 10,
                 log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the parallel optimizer.
        
        Args:
            n_workers: Number of worker processes (default: CPU count)
            batch_size: Number of trials per batch
            log_dir: Directory for worker logs
            
        Raises:
            ParallelError: If initialization fails
        """
        # Validate worker count
        if n_workers is not None and n_workers < 1:
            raise ParallelError("Number of workers must be positive")
            
        # Validate batch size
        if batch_size < 1:
            raise ParallelError("Batch size must be positive")
            
        self.n_workers = n_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "worker_logs"
        
        # Create components
        try:
            self.results_manager = ResultsManager(
                output_dir=self.log_dir
            )
            
            # Create log directory
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize worker pool
            self.worker_configs = [
                WorkerConfig(
                    worker_id=i,
                    batch_size=self.batch_size,
                    memory_limit=0.0,
                    log_dir=self.log_dir
                )
                for i in range(self.n_workers)
            ]
            
            logger.info(f"Initialized parallel optimizer with {self.n_workers} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize parallel optimizer: {str(e)}")
            raise ParallelError(f"Initialization failed: {str(e)}")
            
    def _worker_setup(self, config: WorkerConfig) -> None:
        """
        Setup function for worker processes.
        
        Args:
            config: Worker configuration
            
        Raises:
            WorkerInitError: If worker setup fails
        """
        try:
            # Configure worker-specific logger
            log_file = config.log_dir / f"worker_{config.worker_id}.log"
            logger.add(log_file, format="{time} - {level} - Worker {extra[worker_id]} - {message}")
            logger.configure(extra={"worker_id": config.worker_id})
            
            # Set process name for monitoring
            mp.current_process().name = f"worker_{config.worker_id}"
            
            logger.info("Worker setup complete")
            
        except Exception as e:
            logger.error(f"Worker setup failed: {str(e)}")
            raise WorkerInitError(f"Worker {config.worker_id} setup failed: {str(e)}")
            
    def _worker_cleanup(self, config: WorkerConfig) -> None:
        """
        Cleanup function for worker processes.
        
        Args:
            config: Worker configuration
        """
        try:
            logger.info("Worker cleanup started")
            # Remove worker-specific handlers
            logger.remove()
            
        except Exception as e:
            logger.error(f"Worker cleanup failed: {str(e)}")
            
    def _execute_batch(self,
                      config: WorkerConfig,
                      objective_fn: Callable,
                      trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a batch of trials on a worker.
        
        Args:
            config: Worker configuration
            objective_fn: Objective function to optimize
            trials: List of trial configurations
            
        Returns:
            List of trial results
            
        Raises:
            TaskDistributionError: If batch execution fails
        """
        try:
            self._worker_setup(config)
            results = []
            
            for trial in trials:
                start_time = time.time()
                
                try:
                    # Call the objective function with parameters
                    result = objective_fn(trial["parameters"])
                    
                    # Record successful result
                    results.append({
                        "trial_id": trial["trial_id"],
                        "parameters": trial["parameters"],
                        "result": float(result),  # Ensure result is numeric
                        "status": "completed",
                        "duration": time.time() - start_time,
                        "worker_id": config.worker_id,
                        "error": None
                    })
                    
                except Exception as e:
                    logger.error(f"Trial {trial['trial_id']} failed: {str(e)}")
                    results.append({
                        "trial_id": trial["trial_id"],
                        "parameters": trial["parameters"],
                        "result": None,
                        "status": "failed",
                        "duration": time.time() - start_time,
                        "worker_id": config.worker_id,
                        "error": str(e)
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            raise TaskDistributionError(f"Batch execution failed: {str(e)}")
            
        finally:
            self._worker_cleanup(config)
            
    def optimize(self,
                objective_fn: Callable,
                parameter_space: Dict[str, Any],
                n_trials: int,
                timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Run parallel optimization.
        
        Args:
            objective_fn: Function to optimize
            parameter_space: Dictionary defining parameter search space
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing optimization results
            
        Raises:
            ParallelError: If optimization fails
        """
        start_time = time.time()
        active_workers = []
        
        try:
            # Generate trial configurations
            trials = []
            for i in range(n_trials):
                params = {}
                for k, v in parameter_space.items():
                    if isinstance(v, dict):
                        if v["type"] == "int":
                            start, end = v["range"]
                            step = v.get("step", 1)
                            value = np.random.randint(start, end + 1)
                            params[k] = value
                        else:
                            params[k] = v["range"][0]  # Default to first value
                    else:
                        params[k] = v
                        
                trials.append({
                    "trial_id": i,
                    "parameters": params
                })
                
            # Split trials into batches
            batches = [
                trials[i:i + self.batch_size]
                for i in range(0, len(trials), self.batch_size)
            ]
            
            # Execute batches in parallel
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_batch = {
                    executor.submit(
                        self._execute_batch,
                        config,
                        objective_fn,
                        batch
                    ): batch
                    for config, batch in zip(self.worker_configs, batches)
                }
                
                # Collect results
                all_results = []
                for future in as_completed(future_to_batch, timeout=timeout):
                    batch = future_to_batch[future]
                    try:
                        results = future.result(timeout=timeout)
                        all_results.extend(results)
                        
                        # Update results manager
                        self.results_manager.add_batch_results([
                            {
                                "trial_id": r["trial_id"],
                                "parameters": r["parameters"],
                                "metrics": {"result": r["result"]} if r["status"] == "completed"
                                else {"error": r["error"]}
                            }
                            for r in results
                        ])
                        
                    except TimeoutError as e:
                        msg = "Trial execution timeout"
                        logger.error(msg)
                        raise ParallelError(msg) from e
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        # Mark all trials in batch as failed
                        failed_results = [
                            {
                                "trial_id": t["trial_id"],
                                "parameters": t["parameters"],
                                "error": str(e),
                                "status": "failed",
                                "duration": time.time() - start_time
                            }
                            for t in batch
                        ]
                        all_results.extend(failed_results)
                        
            # Generate optimization summary
            completed_trials = [r for r in all_results if r["status"] == "completed"]
            best_trial = max(completed_trials, key=lambda x: x["result"]) if completed_trials else None
            
            summary = {
                "status": "completed",
                "total_trials": n_trials,
                "completed_trials": len(completed_trials),
                "failed_trials": len(all_results) - len(completed_trials),
                "duration": time.time() - start_time,
                "best_trial": best_trial,
                "trials": all_results
            }
            
            logger.info(f"Optimization completed: {len(completed_trials)}/{n_trials} trials successful")
            
            return summary
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise ParallelError(f"Optimization failed: {str(e)}")
            
    def get_worker_status(self) -> Dict[str, Any]:
        """
        Get status of worker processes.
        
        Returns:
            Dictionary containing worker status information
        """
        try:
            return {
                "error": None,
                "timestamp": datetime.now(),
                "worker_metrics": {},
                "total_workers": self.n_workers
            }
        except Exception as e:
            logger.error(f"Failed to get worker status: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now(),
                "worker_metrics": {},
                "total_workers": self.n_workers
            }
