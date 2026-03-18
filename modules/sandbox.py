import threading
import multiprocessing
import time
import sys
import os
import traceback
import signal
import resource
import psutil
import logging
from queue import Queue, Empty
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import gc
import tracemalloc
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SandboxStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"

@dataclass
class SandboxConfig:
    max_memory_mb: int = 512  # Maximum memory in MB
    timeout_seconds: int = 30  # Maximum execution time in seconds
    max_cpu_percent: int = 80  # Maximum CPU usage percentage
    enable_memory_tracking: bool = True
    enable_timeout: bool = True
    cleanup_on_exit: bool = True
    max_retries: int = 1

@dataclass
class SandboxResult:
    status: SandboxStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    traceback: Optional[str] = None

class ResourceMonitor:
    """Monitors resource usage of a process/thread"""
    
    def __init__(self, pid: Optional[int] = None):
        self.pid = pid or os.getpid()
        self.process = psutil.Process(self.pid)
        self.start_memory = 0
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start resource monitoring"""
        try:
            self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
            self.peak_memory = self.start_memory
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.start_memory = 0
            
    def update_peak_memory(self):
        """Update peak memory usage"""
        try:
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, current_memory)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
            
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.process.cpu_percent(interval=0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
            
    def get_memory_delta(self) -> float:
        """Get memory increase since monitoring started"""
        return self.get_memory_usage() - self.start_memory

class WatchdogTimer:
    """Watchdog timer for enforcing timeouts"""
    
    def __init__(self, timeout_seconds: int, callback: Callable):
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self.timer = None
        self.timed_out = False
        
    def start(self):
        """Start the watchdog timer"""
        if self.timeout_seconds > 0:
            self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
            self.timer.daemon = True
            self.timer.start()
            
    def _timeout_handler(self):
        """Handle timeout event"""
        self.timed_out = True
        if self.callback:
            self.callback()
            
    def cancel(self):
        """Cancel the watchdog timer"""
        if self.timer:
            self.timer.cancel()
            self.timer = None

class MemoryTracker:
    """Tracks memory allocations for sandboxed code"""
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.tracking_enabled = False
        
    def start_tracking(self):
        """Start memory tracking"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        self.tracking_enabled = True
        
    def stop_tracking(self):
        """Stop memory tracking"""
        if self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            
    def check_memory_limit(self) -> bool:
        """Check if memory limit is exceeded"""
        if not self.tracking_enabled:
            return False
            
        current, peak = tracemalloc.get_traced_memory()
        return peak > self.max_memory_bytes
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        if not self.tracking_enabled:
            return {"current": 0, "peak": 0}
            
        current, peak = tracemalloc.get_traced_memory()
        return {
            "current": current / (1024 * 1024),  # MB
            "peak": peak / (1024 * 1024)  # MB
        }

class SandboxWorker:
    """Worker that executes code in a sandboxed environment"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.memory_tracker = MemoryTracker(config.max_memory_mb)
        self.watchdog = None
        self.result_queue = Queue()
        self.interrupt_event = threading.Event()
        self.execution_thread = None
        
    def _interrupt_handler(self):
        """Handle interruption (timeout/memory limit)"""
        self.interrupt_event.set()
        logger.warning("Sandbox execution interrupted")
        
    def execute(self, func: Callable, *args, **kwargs) -> SandboxResult:
        """Execute a function in the sandbox"""
        start_time = time.time()
        result = SandboxResult(status=SandboxStatus.IDLE)
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        if self.config.enable_memory_tracking:
            self.memory_tracker.start_tracking()
            
        # Start watchdog timer
        if self.config.enable_timeout:
            self.watchdog = WatchdogTimer(
                self.config.timeout_seconds,
                self._interrupt_handler
            )
            self.watchdog.start()
            
        # Execute in thread
        self.execution_thread = threading.Thread(
            target=self._execute_wrapper,
            args=(func, args, kwargs)
        )
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        # Monitor execution
        try:
            while self.execution_thread.is_alive():
                # Check for interruption
                if self.interrupt_event.is_set():
                    result.status = SandboxStatus.TIMEOUT
                    result.error = "Execution timeout"
                    break
                    
                # Check memory limit
                if self.config.enable_memory_tracking and self.memory_tracker.check_memory_limit():
                    result.status = SandboxStatus.MEMORY_LIMIT
                    result.error = f"Memory limit exceeded: {self.config.max_memory_mb}MB"
                    self.interrupt_event.set()
                    break
                    
                # Check CPU usage
                cpu_usage = self.resource_monitor.get_cpu_usage()
                if cpu_usage > self.config.max_cpu_percent:
                    logger.warning(f"High CPU usage detected: {cpu_usage}%")
                    
                time.sleep(0.1)  # Poll interval
                
            # Wait for thread to finish (with timeout)
            self.execution_thread.join(timeout=1.0)
            
        except Exception as e:
            result.status = SandboxStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()
            
        # Collect results
        if result.status == SandboxStatus.IDLE:
            try:
                execution_result = self.result_queue.get_nowait()
                result.status = execution_result.get("status", SandboxStatus.COMPLETED)
                result.result = execution_result.get("result")
                result.error = execution_result.get("error")
                result.traceback = execution_result.get("traceback")
            except Empty:
                result.status = SandboxStatus.FAILED
                result.error = "No result returned from execution"
                
        # Cleanup
        if self.watchdog:
            self.watchdog.cancel()
            
        if self.config.enable_memory_tracking:
            memory_stats = self.memory_tracker.get_memory_stats()
            result.memory_used_mb = memory_stats["peak"]
            self.memory_tracker.stop_tracking()
            
        result.execution_time = time.time() - start_time
        
        # Force garbage collection
        if self.config.cleanup_on_exit:
            gc.collect()
            
        return result
        
    def _execute_wrapper(self, func: Callable, args: tuple, kwargs: dict):
        """Wrapper for executing the function with error handling"""
        try:
            # Set resource limits (Unix-like systems only)
            if hasattr(resource, 'setrlimit'):
                try:
                    # Set memory limit
                    memory_bytes = self.config.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                except (ValueError, resource.error):
                    pass
                    
            # Execute the function
            result = func(*args, **kwargs)
            
            self.result_queue.put({
                "status": SandboxStatus.COMPLETED,
                "result": result
            })
            
        except MemoryError:
            self.result_queue.put({
                "status": SandboxStatus.MEMORY_LIMIT,
                "error": "Memory limit exceeded"
            })
            
        except Exception as e:
            self.result_queue.put({
                "status": SandboxStatus.FAILED,
                "error": str(e),
                "traceback": traceback.format_exc()
            })

class ExtensionSandbox:
    """Main sandbox manager for extensions"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExtensionSandbox, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            self.default_config = SandboxConfig()
            self.extension_configs: Dict[str, SandboxConfig] = {}
            self.active_sandboxes: Dict[str, SandboxWorker] = {}
            self.execution_history: List[Dict] = []
            self._initialized = True
            
    def configure_extension(self, extension_name: str, config: SandboxConfig):
        """Configure sandbox settings for a specific extension"""
        self.extension_configs[extension_name] = config
        logger.info(f"Configured sandbox for extension: {extension_name}")
        
    def get_config(self, extension_name: str) -> SandboxConfig:
        """Get sandbox configuration for an extension"""
        return self.extension_configs.get(extension_name, self.default_config)
        
    def execute_sandboxed(self, extension_name: str, func: Callable, *args, **kwargs) -> SandboxResult:
        """Execute a function in a sandboxed environment for an extension"""
        config = self.get_config(extension_name)
        worker = SandboxWorker(config)
        
        # Store active sandbox
        self.active_sandboxes[extension_name] = worker
        
        try:
            result = worker.execute(func, *args, **kwargs)
            
            # Record execution history
            self.execution_history.append({
                "extension": extension_name,
                "timestamp": time.time(),
                "status": result.status,
                "execution_time": result.execution_time,
                "memory_used_mb": result.memory_used_mb,
                "error": result.error
            })
            
            # Keep history limited
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
                
            return result
            
        finally:
            # Cleanup
            if extension_name in self.active_sandboxes:
                del self.active_sandboxes[extension_name]
                
    def sandbox_extension(self, extension_name: str):
        """Decorator to sandbox an entire extension module"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_sandboxed(extension_name, func, *args, **kwargs)
            return wrapper
        return decorator
        
    def get_extension_stats(self, extension_name: str) -> Dict:
        """Get execution statistics for an extension"""
        extension_history = [
            h for h in self.execution_history 
            if h["extension"] == extension_name
        ]
        
        if not extension_history:
            return {}
            
        total_executions = len(extension_history)
        successful = sum(1 for h in extension_history if h["status"] == SandboxStatus.COMPLETED)
        failed = sum(1 for h in extension_history if h["status"] == SandboxStatus.FAILED)
        timeouts = sum(1 for h in extension_history if h["status"] == SandboxStatus.TIMEOUT)
        memory_limits = sum(1 for h in extension_history if h["status"] == SandboxStatus.MEMORY_LIMIT)
        
        avg_execution_time = sum(h["execution_time"] for h in extension_history) / total_executions
        avg_memory_used = sum(h["memory_used_mb"] for h in extension_history) / total_executions
        
        return {
            "extension_name": extension_name,
            "total_executions": total_executions,
            "successful": successful,
            "failed": failed,
            "timeouts": timeouts,
            "memory_limits": memory_limits,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "average_memory_used_mb": avg_memory_used
        }
        
    def cleanup_all(self):
        """Cleanup all active sandboxes"""
        for extension_name in list(self.active_sandboxes.keys()):
            try:
                worker = self.active_sandboxes[extension_name]
                worker.interrupt_event.set()
                if worker.execution_thread and worker.execution_thread.is_alive():
                    worker.execution_thread.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error cleaning up sandbox for {extension_name}: {e}")
                
        self.active_sandboxes.clear()
        gc.collect()
        
    def emergency_stop(self):
        """Emergency stop all sandboxes"""
        logger.warning("Emergency stop initiated for all sandboxes")
        self.cleanup_all()
        
        # Force kill any remaining threads (use with caution)
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.daemon:
                try:
                    # This is a last resort - may cause instability
                    pass
                except:
                    pass

# Integration with existing extension system
class SandboxedExtension:
    """Wrapper for extensions to run in sandbox"""
    
    def __init__(self, extension_module, extension_name: str, config: Optional[SandboxConfig] = None):
        self.extension_module = extension_module
        self.extension_name = extension_name
        self.sandbox = ExtensionSandbox()
        
        if config:
            self.sandbox.configure_extension(extension_name, config)
            
    def __getattr__(self, name):
        """Proxy attribute access to sandboxed execution"""
        attr = getattr(self.extension_module, name, None)
        
        if attr is None:
            raise AttributeError(f"Extension {self.extension_name} has no attribute {name}")
            
        if callable(attr):
            def sandboxed_call(*args, **kwargs):
                result = self.sandbox.execute_sandboxed(
                    self.extension_name,
                    attr,
                    *args,
                    **kwargs
                )
                
                if result.status == SandboxStatus.COMPLETED:
                    return result.result
                elif result.status == SandboxStatus.TIMEOUT:
                    raise TimeoutError(f"Extension {self.extension_name}.{name} timed out")
                elif result.status == SandboxStatus.MEMORY_LIMIT:
                    raise MemoryError(f"Extension {self.extension_name}.{name} exceeded memory limit")
                else:
                    raise RuntimeError(f"Extension {self.extension_name}.{name} failed: {result.error}")
                    
            return sandboxed_call
        else:
            return attr

# Global sandbox instance
sandbox_manager = ExtensionSandbox()

# Decorator for easy sandboxing
def sandboxed(extension_name: str, config: Optional[SandboxConfig] = None):
    """Decorator to sandbox a function"""
    def decorator(func):
        if config:
            sandbox_manager.configure_extension(extension_name, config)
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            return sandbox_manager.execute_sandboxed(extension_name, func, *args, **kwargs)
        return wrapper
    return decorator

# Integration hooks for existing extension loading system
def patch_extension_loader():
    """Patch the extension loader to use sandboxing"""
    try:
        # This would integrate with the actual extension loading system
        # For now, we provide the interface
        logger.info("Extension sandboxing system initialized")
        
        # Register cleanup on exit
        import atexit
        atexit.register(sandbox_manager.cleanup_all)
        
        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, cleaning up sandboxes")
            sandbox_manager.cleanup_all()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    except Exception as e:
        logger.error(f"Failed to patch extension loader: {e}")

# Auto-initialize when module is imported
patch_extension_loader()

# Export public API
__all__ = [
    'SandboxConfig',
    'SandboxResult',
    'SandboxStatus',
    'ExtensionSandbox',
    'SandboxedExtension',
    'sandbox_manager',
    'sandboxed',
    'patch_extension_loader'
]