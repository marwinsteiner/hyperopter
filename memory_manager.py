"""Memory manager for monitoring and optimizing resource usage."""

import gc
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import time

from loguru import logger

class MemoryError(Exception):
    """Base exception class for memory-related errors."""
    pass

class MemoryLimitError(MemoryError):
    """Raised when memory usage exceeds defined limits."""
    pass

class ResourceCleanupError(MemoryError):
    """Raised when resource cleanup fails."""
    pass

class MemoryManager:
    """Manager for monitoring and optimizing memory usage."""
    
    def __init__(self, 
                 memory_limit: float = 0.95,  # 95% of system memory
                 cleanup_threshold: float = 0.9,  # 90% of memory limit
                 monitoring_interval: int = 60,  # 60 seconds
                 cleanup_interval: float = 300.0):  # Cleanup every 5 minutes
        """
        Initialize the memory manager.
        
        Args:
            memory_limit: Maximum memory usage as fraction of total system memory
            cleanup_threshold: Memory threshold to trigger cleanup as fraction
            monitoring_interval: Interval between memory checks in seconds
            cleanup_interval: Minimum interval between cleanups in seconds
            
        Raises:
            MemoryError: If invalid parameters are provided
        """
        if not 0 < memory_limit <= 1:
            raise MemoryError("Memory limit must be between 0 and 1")
        if not 0 < cleanup_threshold < memory_limit:
            raise MemoryError("Cleanup threshold must be between 0 and memory limit")
        if monitoring_interval < 1:
            raise MemoryError("Monitoring interval must be positive")
        if cleanup_interval <= 0.0:
            raise MemoryError("Cleanup interval must be positive")
            
        self.memory_limit = memory_limit
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        self.cleanup_interval = cleanup_interval
        
        self.process = psutil.Process()
        self.total_memory = psutil.virtual_memory().total
        self.last_check = datetime.now()
        self.is_monitoring = False
        
        # Initialize metrics storage
        self.usage_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized memory manager")
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage metrics.
        
        Returns:
            Dictionary containing memory usage metrics
        """
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            "system_percent": memory.percent / 100,  # Convert to fraction
            "process_rss": process_memory.rss / self.total_memory,  # Resident Set Size
            "process_vms": process_memory.vms / self.total_memory,  # Virtual Memory Size
            "available_percent": memory.available / self.total_memory
        }
        
    def check_memory_status(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check current memory status and determine if cleanup is needed.
        
        Returns:
            Tuple of (cleanup_needed, memory_metrics)
        """
        metrics = self.get_memory_usage()
        
        # Store metrics in history
        self.usage_history.append({
            "timestamp": datetime.now(),
            **metrics
        })
        
        # Keep only last 1000 measurements
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
            
        cleanup_needed = (
            metrics["system_percent"] >= self.cleanup_threshold or
            metrics["process_rss"] >= self.cleanup_threshold
        )
        
        return cleanup_needed, metrics
        
    def cleanup_resources(self) -> bool:
        """
        Perform memory cleanup operations.
        
        Returns:
            bool: True if cleanup was successful
            
        Raises:
            ResourceCleanupError: If cleanup fails
        """
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear process working set
            if hasattr(psutil, "Process"):
                try:
                    self.process.memory_info()
                except Exception as e:
                    logger.warning(f"Failed to get process memory info: {str(e)}")
                    
            # Check if cleanup was effective
            _, metrics = self.check_memory_status()
            cleanup_successful = (
                metrics["system_percent"] < self.cleanup_threshold and
                metrics["process_rss"] < self.cleanup_threshold
            )
            
            if cleanup_successful:
                logger.info("Memory cleanup successful")
            else:
                logger.warning("Memory cleanup completed but usage still high")
                
            return cleanup_successful
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {str(e)}")
            raise ResourceCleanupError(f"Failed to cleanup resources: {str(e)}")
            
    def monitor_memory(self) -> Dict[str, Any]:
        """
        Check memory usage and perform cleanup if needed.
        
        Returns:
            Dictionary containing monitoring results and actions taken
            
        Raises:
            MemoryLimitError: If memory usage exceeds limit
        """
        cleanup_needed, metrics = self.check_memory_status()
        actions_taken = []
        
        # Check if memory limit is exceeded
        if metrics["system_percent"] >= self.memory_limit:
            msg = f"System memory usage ({metrics['system_percent']:.1%}) exceeds limit ({self.memory_limit:.1%})"
            logger.error(msg)
            raise MemoryLimitError(msg)
            
        # Perform cleanup if needed
        if cleanup_needed:
            logger.warning("Memory usage high, initiating cleanup")
            cleanup_success = self.cleanup_resources()
            actions_taken.append("cleanup_performed")
            if cleanup_success:
                actions_taken.append("cleanup_successful")
            else:
                actions_taken.append("cleanup_insufficient")
                
        return {
            "timestamp": datetime.now(),
            "metrics": metrics,
            "cleanup_needed": cleanup_needed,
            "actions_taken": actions_taken
        }
        
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Generate a report of memory usage statistics.
        
        Returns:
            Dictionary containing usage statistics and trends
        """
        if not self.usage_history:
            return {"error": "No usage history available"}
            
        # Calculate statistics
        system_usage = [entry["system_percent"] for entry in self.usage_history]
        process_rss = [entry["process_rss"] for entry in self.usage_history]
        
        return {
            "timestamp": datetime.now(),
            "total_measurements": len(self.usage_history),
            "system_memory": {
                "current": system_usage[-1],
                "average": sum(system_usage) / len(system_usage),
                "peak": max(system_usage)
            },
            "process_memory": {
                "current": process_rss[-1],
                "average": sum(process_rss) / len(process_rss),
                "peak": max(process_rss)
            },
            "total_memory_bytes": self.total_memory,
            "memory_limit": self.memory_limit,
            "cleanup_threshold": self.cleanup_threshold
        }
        
    def reset_metrics(self) -> None:
        """Reset all stored metrics and history."""
        self.usage_history.clear()
        gc.collect()
        logger.info("Reset memory metrics and history")

    def monitor_optimization_engine(self, engine_pid: Optional[int] = None) -> Dict[str, Any]:
        """
        Monitor memory usage of the optimization engine process.
        
        Args:
            engine_pid: Process ID of the optimization engine (if different from current)
            
        Returns:
            Dictionary containing engine-specific memory metrics
            
        Raises:
            MemoryError: If engine process cannot be monitored
        """
        try:
            # Get process to monitor
            process = psutil.Process(engine_pid) if engine_pid else self.process
            
            # Get memory info
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get additional process metrics
            metrics = {
                "rss": memory_info.rss / self.total_memory,
                "vms": memory_info.vms / self.total_memory,
                "percent": memory_percent / 100,  # Convert to fraction
                "num_threads": process.num_threads(),
                "cpu_percent": process.cpu_percent()
            }
            
            logger.info(f"Optimization engine memory usage: {metrics['percent']:.1%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to monitor optimization engine: {str(e)}")
            raise MemoryError(f"Failed to monitor optimization engine: {str(e)}")
            
    def coordinate_parallel_workers(self, 
                                  worker_pids: List[int],
                                  memory_per_worker: float = 0.2) -> Dict[str, Any]:
        """
        Monitor and coordinate memory usage across parallel workers.
        
        Args:
            worker_pids: List of worker process IDs
            memory_per_worker: Maximum memory fraction per worker
            
        Returns:
            Dictionary containing worker memory status and actions
            
        Raises:
            MemoryError: If worker coordination fails
        """
        if not worker_pids:
            raise MemoryError("No worker PIDs provided")
            
        try:
            worker_metrics = {}
            actions_needed = []
            
            # Monitor each worker
            for pid in worker_pids:
                try:
                    process = psutil.Process(pid)
                    memory_percent = process.memory_percent() / 100  # Convert to fraction
                    
                    worker_metrics[pid] = {
                        "memory_percent": memory_percent,
                        "status": "ok" if memory_percent < memory_per_worker else "high"
                    }
                    
                    # Check if worker needs cleanup
                    if memory_percent >= memory_per_worker:
                        actions_needed.append({
                            "pid": pid,
                            "action": "cleanup",
                            "current_usage": memory_percent
                        })
                        
                except psutil.NoSuchProcess:
                    worker_metrics[pid] = {"status": "not_found"}
                    
            # Log status
            active_workers = sum(1 for m in worker_metrics.values() if m.get("status") == "ok")
            logger.info(f"Monitoring {len(worker_pids)} workers, {active_workers} within limits")
            
            return {
                "timestamp": datetime.now(),
                "worker_metrics": worker_metrics,
                "actions_needed": actions_needed,
                "total_workers": len(worker_pids),
                "active_workers": active_workers
            }
            
        except Exception as e:
            logger.error(f"Failed to coordinate parallel workers: {str(e)}")
            raise MemoryError(f"Failed to coordinate parallel workers: {str(e)}")
            
    def start_continuous_monitoring(self, 
                                  engine_pid: Optional[int] = None,
                                  worker_pids: Optional[List[int]] = None) -> None:
        """
        Start continuous memory monitoring for optimization processes.
        
        Args:
            engine_pid: Optional optimization engine process ID
            worker_pids: Optional list of worker process IDs
            
        Raises:
            MemoryError: If monitoring cannot be started
        """
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        logger.info("Starting continuous memory monitoring")
        
        try:
            while self.is_monitoring:
                # Monitor main process
                self.monitor_memory()
                
                # Monitor optimization engine if specified
                if engine_pid:
                    self.monitor_optimization_engine(engine_pid)
                    
                # Monitor workers if specified
                if worker_pids:
                    self.coordinate_parallel_workers(worker_pids)
                    
                # Wait for next interval
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            self.is_monitoring = False
            logger.error(f"Continuous monitoring failed: {str(e)}")
            raise MemoryError(f"Continuous monitoring failed: {str(e)}")
            
    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self.is_monitoring = False
        logger.info("Stopped memory monitoring")
