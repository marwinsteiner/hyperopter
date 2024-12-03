"""Tests for the memory manager."""

import unittest
from datetime import datetime
import gc
import os
import time
import threading

import psutil
from loguru import logger

from memory_manager import (
    MemoryManager,
    MemoryError,
    MemoryLimitError,
    ResourceCleanupError
)

class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.memory_manager = MemoryManager(
            memory_limit=0.9,
            cleanup_threshold=0.8,
            monitoring_interval=30
        )
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.memory_manager.reset_metrics()
        gc.collect()
        
    def test_initialization(self):
        """Test memory manager initialization."""
        # Test valid initialization
        mm = MemoryManager()
        self.assertEqual(mm.memory_limit, 0.8)
        self.assertEqual(mm.cleanup_threshold, 0.7)
        
        # Test invalid memory limit
        with self.assertRaises(MemoryError):
            MemoryManager(memory_limit=1.5)
            
        # Test invalid cleanup threshold
        with self.assertRaises(MemoryError):
            MemoryManager(cleanup_threshold=0.9, memory_limit=0.8)
            
        # Test invalid monitoring interval
        with self.assertRaises(MemoryError):
            MemoryManager(monitoring_interval=0)
            
    def test_get_memory_usage(self):
        """Test getting memory usage metrics."""
        metrics = self.memory_manager.get_memory_usage()
        
        # Check required metrics are present
        self.assertIn("system_percent", metrics)
        self.assertIn("process_rss", metrics)
        self.assertIn("process_vms", metrics)
        self.assertIn("available_percent", metrics)
        
        # Check metric values are valid
        for value in metrics.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
            
    def test_check_memory_status(self):
        """Test memory status checking."""
        cleanup_needed, metrics = self.memory_manager.check_memory_status()
        
        # Check return types
        self.assertIsInstance(cleanup_needed, bool)
        self.assertIsInstance(metrics, dict)
        
        # Check history is updated
        self.assertEqual(len(self.memory_manager.usage_history), 1)
        self.assertIn("timestamp", self.memory_manager.usage_history[0])
        
    def test_cleanup_resources(self):
        """Test resource cleanup."""
        # Basic cleanup test
        result = self.memory_manager.cleanup_resources()
        self.assertIsInstance(result, bool)
        
        # Test cleanup with memory allocation
        large_list = [i for i in range(1000000)]
        del large_list
        result = self.memory_manager.cleanup_resources()
        
        # Check if cleanup was attempted (don't assert success as it depends on system state)
        self.assertIsInstance(result, bool)
        self.assertGreater(len(self.memory_manager.usage_history), 0)
        
    def test_monitor_memory(self):
        """Test memory monitoring."""
        # Create a test instance with higher limits for system testing
        test_mm = MemoryManager(memory_limit=0.95, cleanup_threshold=0.85)
        
        # Test normal monitoring
        try:
            result = test_mm.monitor_memory()
            self.assertIn("timestamp", result)
            self.assertIn("metrics", result)
            self.assertIn("cleanup_needed", result)
            self.assertIn("actions_taken", result)
        except MemoryLimitError:
            # Skip test if system memory is too high
            self.skipTest("System memory usage too high for test")
        
        # Test with simulated high memory usage
        original_get_memory_usage = test_mm.get_memory_usage
        def mock_high_memory():
            return {
                "system_percent": 0.99,  # Very high memory usage
                "process_rss": 0.85,
                "process_vms": 0.8,
                "available_percent": 0.1
            }
        test_mm.get_memory_usage = mock_high_memory
        
        with self.assertRaises(MemoryLimitError):
            test_mm.monitor_memory()
            
        # Restore original method
        test_mm.get_memory_usage = original_get_memory_usage
        
    def test_get_usage_report(self):
        """Test usage report generation."""
        # Test with no history
        report = self.memory_manager.get_usage_report()
        self.assertIn("error", report)
        
        # Add some history and test again
        self.memory_manager.check_memory_status()
        report = self.memory_manager.get_usage_report()
        
        self.assertIn("timestamp", report)
        self.assertIn("total_measurements", report)
        self.assertIn("system_memory", report)
        self.assertIn("process_memory", report)
        
        # Check memory stats
        for memory_type in ["system_memory", "process_memory"]:
            stats = report[memory_type]
            self.assertIn("current", stats)
            self.assertIn("average", stats)
            self.assertIn("peak", stats)
            
    def test_reset_metrics(self):
        """Test metrics reset."""
        # Add some history
        self.memory_manager.check_memory_status()
        self.assertGreater(len(self.memory_manager.usage_history), 0)
        
        # Reset and verify
        self.memory_manager.reset_metrics()
        self.assertEqual(len(self.memory_manager.usage_history), 0)

    def test_monitor_optimization_engine(self):
        """Test optimization engine monitoring."""
        # Test monitoring current process
        metrics = self.memory_manager.monitor_optimization_engine()
        self.assertIn("rss", metrics)
        self.assertIn("vms", metrics)
        self.assertIn("percent", metrics)
        self.assertIn("num_threads", metrics)
        self.assertIn("cpu_percent", metrics)
        
        # Test monitoring invalid PID
        with self.assertRaises(MemoryError):
            self.memory_manager.monitor_optimization_engine(999999)
            
    def test_coordinate_parallel_workers(self):
        """Test parallel worker coordination."""
        # Test with no workers
        with self.assertRaises(MemoryError):
            self.memory_manager.coordinate_parallel_workers([])
            
        # Test with current process as worker
        current_pid = os.getpid()
        result = self.memory_manager.coordinate_parallel_workers([current_pid])
        
        self.assertIn("worker_metrics", result)
        self.assertIn("actions_needed", result)
        self.assertEqual(result["total_workers"], 1)
        self.assertIn(current_pid, result["worker_metrics"])
        
        # Test with invalid PID
        result = self.memory_manager.coordinate_parallel_workers([999999])
        self.assertEqual(result["worker_metrics"][999999]["status"], "not_found")
        
    def test_continuous_monitoring(self):
        """Test continuous monitoring functionality."""
        # Start monitoring in a separate thread
        import threading
        stop_event = threading.Event()
        
        def monitor_thread():
            try:
                self.memory_manager.start_continuous_monitoring()
            except Exception:
                pass
            finally:
                stop_event.set()
                
        thread = threading.Thread(target=monitor_thread)
        thread.daemon = True
        thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop monitoring
        self.memory_manager.stop_monitoring()
        stop_event.wait(timeout=1.0)
        
        # Verify monitoring was active
        self.assertGreater(len(self.memory_manager.usage_history), 0)
