"""Memory manager for monitoring and optimizing resource usage."""

import gc
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

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
                 memory_limit: float = 0.8,  # 80% of system memory
                 cleanup_threshold: float = 0.7,  # 70% of system memory
                 monitoring_interval: int = 60):  # 60 seconds
        """
        Initialize the memory manager.
        
        Args:
            memory_limit: Maximum memory usage as fraction of total system memory
            cleanup_threshold: Memory threshold to trigger cleanup as fraction
            monitoring_interval: Interval between memory checks in seconds
            
        Raises:
            MemoryError: If invalid parameters are provided
        """
        if not 0 < memory_limit <= 1:
            raise MemoryError("Memory limit must be between 0 and 1")
        if not 0 < cleanup_threshold < memory_limit:
            raise MemoryError("Cleanup threshold must be between 0 and memory limit")
        if monitoring_interval < 1:
            raise MemoryError("Monitoring interval must be positive")
            
        self.memory_limit = memory_limit
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
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
