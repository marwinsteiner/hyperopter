"""Tests for the parallel optimizer module."""

import unittest
from pathlib import Path
import tempfile
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import logging

from parallel_optimizer import (
    ParallelOptimizer,
    ParallelError,
    WorkerInitError,
    TaskDistributionError,
    ResultCollectionError
)

class TestParallelOptimizer(unittest.TestCase):
    """Test cases for ParallelOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Disable loguru logging during tests
        logging.getLogger("loguru").setLevel(logging.WARNING)
        
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "worker_logs"
        
        # Use lower memory limits for testing
        self.optimizer = ParallelOptimizer(
            n_workers=2,
            batch_size=5,
            memory_per_worker=0.1,  # 10% per worker
            log_dir=self.log_dir
        )
        self.optimizer.memory_manager.memory_limit = 0.5  # 50% total limit for testing
        
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            # Stop any active monitoring
            if hasattr(self.optimizer, 'memory_manager'):
                self.optimizer.memory_manager.stop_monitoring()
                
            # Close any open file handles
            import gc
            gc.collect()
            time.sleep(0.1)  # Give OS time to release handles
            
            # Remove temp directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Cleanup error (non-fatal): {str(e)}")
        
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_workers, 2)
        self.assertEqual(self.optimizer.batch_size, 5)
        self.assertEqual(self.optimizer.memory_per_worker, 0.1)
        self.assertTrue(self.log_dir.exists())
        
        # Test invalid worker count
        with self.assertRaises(ParallelError):
            ParallelOptimizer(n_workers=0)
            
    def test_worker_setup(self):
        """Test worker process setup."""
        config = self.optimizer.worker_configs[0]
        
        # Test successful setup
        try:
            self.optimizer._worker_setup(config)
        except WorkerInitError:
            self.fail("Worker setup raised WorkerInitError unexpectedly")
            
        # Test worker log file creation
        log_file = self.log_dir / f"worker_{config.worker_id}.log"
        self.assertTrue(log_file.exists())
        
    def test_worker_cleanup(self):
        """Test worker process cleanup."""
        config = self.optimizer.worker_configs[0]
        
        # Setup and cleanup should not raise errors
        try:
            self.optimizer._worker_setup(config)
            self.optimizer._worker_cleanup(config)
        except Exception as e:
            self.fail(f"Worker cleanup raised an exception: {str(e)}")
            
    def test_batch_execution(self):
        """Test batch trial execution."""
        def objective_fn(x, y):
            return x**2 + y**2
            
        trials = [
            {
                "trial_id": i,
                "parameters": {"x": i * 0.1, "y": i * 0.2}
            }
            for i in range(5)
        ]
        
        # Test successful batch execution
        results = self.optimizer._execute_batch(
            self.optimizer.worker_configs[0],
            objective_fn,
            trials
        )
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("trial_id", result)
            self.assertIn("parameters", result)
            self.assertIn("result", result)
            self.assertEqual(result["status"], "completed")
            
        # Test batch execution with failing trials
        def failing_objective(x, y):
            if x > 0.2:
                raise ValueError("x too large")
            return x**2 + y**2
            
        results = self.optimizer._execute_batch(
            self.optimizer.worker_configs[0],
            failing_objective,
            trials
        )
        
        failed_trials = [r for r in results if r["status"] == "failed"]
        self.assertTrue(len(failed_trials) > 0)
        
    def test_optimization(self):
        """Test full optimization process."""
        def objective_fn(x, y):
            return x**2 + y**2
            
        parameter_space = {
            "x": lambda: np.random.uniform(-1, 1),
            "y": lambda: np.random.uniform(-1, 1)
        }
        
        # Create a proper Future mock
        class MockFuture:
            def __init__(self, result_value, raise_timeout=False):
                self._result = result_value
                self._condition = Mock()
                self._condition.__enter__ = Mock(return_value=None)
                self._condition.__exit__ = Mock(return_value=None)
                self._state = "FINISHED"
                self._waiters = []
                self._raise_timeout = raise_timeout
                
            def result(self, timeout=None):
                if self._raise_timeout:
                    raise TimeoutError("Timeout")
                return self._result
                
            def add_done_callback(self, fn):
                pass
                
            def done(self):
                return True
                
            def set_running_or_notify_cancel(self):
                return True
                
            def set_result(self, result):
                self._result = result
        
        def mock_as_completed(futures, timeout=None):
            for future in futures:
                yield future
        
        # Mock memory monitoring and process pool
        with patch('memory_manager.MemoryManager.start_continuous_monitoring') as mock_monitor, \
             patch('parallel_optimizer.ProcessPoolExecutor') as mock_executor, \
             patch('concurrent.futures.as_completed', side_effect=mock_as_completed), \
             patch('concurrent.futures._base.FINISHED', "FINISHED"), \
             patch('concurrent.futures._base.CANCELLED_AND_NOTIFIED', "CANCELLED_AND_NOTIFIED"):
             
            mock_monitor.return_value = None
            
            # Test successful optimization
            mock_result = [{
                "trial_id": 0,
                "parameters": {"x": 0.1, "y": 0.2},
                "result": 0.05,
                "status": "completed",
                "duration": 0.1,
                "worker_id": 1,
                "error": None
            }]
            
            # Mock executor to return results directly
            mock_future = MockFuture(mock_result)
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            results = self.optimizer.optimize(
                objective_fn,
                parameter_space,
                n_trials=1,  # Reduce trials for simpler test
                timeout=30
            )
            
            self.assertEqual(results["status"], "completed")
            self.assertEqual(results["total_trials"], 1)
            self.assertEqual(results["completed_trials"], 1)
            self.assertIsNotNone(results["best_trial"])
            self.assertEqual(results["best_trial"]["result"], 0.05)
            
            # Test optimization with timeout
            mock_future_timeout = MockFuture(None, raise_timeout=True)
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future_timeout
            
            with self.assertRaises(ParallelError):
                self.optimizer.optimize(
                    lambda x, y: time.sleep(1) or (x**2 + y**2),
                    parameter_space,
                    n_trials=1,
                    timeout=1
                )
            
    def test_worker_status(self):
        """Test worker status monitoring."""
        status = self.optimizer.get_worker_status()
        
        self.assertIn("worker_metrics", status)
        self.assertIn("total_workers", status)
        self.assertEqual(status["total_workers"], 2)
        
    @patch('parallel_optimizer.ProcessPoolExecutor')
    def test_optimization_error_handling(self, mock_executor):
        """Test error handling during optimization."""
        # Mock executor to simulate failures
        mock_executor.return_value.__enter__.return_value.submit.side_effect = \
            TaskDistributionError("Task distribution failed")
            
        with self.assertRaises(ParallelError):
            self.optimizer.optimize(
                lambda x: x**2,
                {"x": lambda: np.random.uniform(-1, 1)},
                n_trials=10
            )
            
if __name__ == '__main__':
    unittest.main()
