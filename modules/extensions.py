from __future__ import annotations

import configparser
import dataclasses
import os
import threading
import re
import multiprocessing
import time
import signal
import resource
import traceback
import queue
import psutil
import importlib
import sys
import hashlib
import json
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Set, Tuple
from enum import Enum
from collections import defaultdict

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions: list[Extension] = []
extension_paths: dict[str, Extension] = {}
loaded_extensions: dict[str, Exception] = {}


os.makedirs(extensions_dir, exist_ok=True)


class SandboxStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    CRASHED = "crashed"
    DISABLED = "disabled"


@dataclasses.dataclass
class SandboxConfig:
    memory_limit_mb: int = 512  # Default 512MB per extension
    timeout_seconds: int = 30   # Default 30 second timeout
    enable_sandbox: bool = True
    max_restarts: int = 2       # Auto-restart extension on crash up to N times
    watchdog_interval: float = 1.0  # Check interval in seconds


@dataclasses.dataclass
class SandboxResult:
    status: SandboxStatus
    result: Any = None
    exception: Optional[Exception] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    restart_count: int = 0


class ExtensionSandbox:
    """Manages sandboxed execution of extension code with resource limits."""
    
    def __init__(self, extension_name: str, config: Optional[SandboxConfig] = None):
        self.extension_name = extension_name
        self.config = config or SandboxConfig()
        self.process: Optional[multiprocessing.Process] = None
        self.result_queue = multiprocessing.Queue()
        self.exception_queue = multiprocessing.Queue()
        self.start_time = 0.0
        self.watchdog_thread: Optional[threading.Thread] = None
        self.stop_watchdog = threading.Event()
        self.restart_count = 0
        self._last_memory_check = 0.0
        self._peak_memory = 0.0
        
    def execute_sandboxed(self, func: Callable, *args, **kwargs) -> SandboxResult:
        """Execute a function in a sandboxed process with resource limits."""
        if not self.config.enable_sandbox:
            # Bypass sandbox if disabled
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                return SandboxResult(
                    status=SandboxStatus.DISABLED,
                    result=result,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return SandboxResult(
                    status=SandboxStatus.CRASHED,
                    exception=e,
                    execution_time=time.time() - start_time
                )
        
        self.stop_watchdog.clear()
        self.start_time = time.time()
        
        # Create and start sandboxed process
        self.process = multiprocessing.Process(
            target=self._run_in_sandbox,
            args=(func, args, kwargs, self.result_queue, self.exception_queue)
        )
        
        self.process.start()
        
        # Start watchdog thread
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_monitor,
            daemon=True
        )
        self.watchdog_thread.start()
        
        # Wait for process completion
        try:
            self.process.join(timeout=self.config.timeout_seconds)
            
            if self.process.is_alive():
                # Timeout occurred
                self._terminate_process()
                return SandboxResult(
                    status=SandboxStatus.TIMEOUT,
                    execution_time=time.time() - self.start_time,
                    restart_count=self.restart_count
                )
            
            # Check for exceptions
            if not self.exception_queue.empty():
                exception = self.exception_queue.get_nowait()
                return SandboxResult(
                    status=SandboxStatus.CRASHED,
                    exception=exception,
                    execution_time=time.time() - self.start_time,
                    memory_used_mb=self._peak_memory,
                    restart_count=self.restart_count
                )
            
            # Get result
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                return SandboxResult(
                    status=SandboxStatus.COMPLETED,
                    result=result,
                    execution_time=time.time() - self.start_time,
                    memory_used_mb=self._peak_memory,
                    restart_count=self.restart_count
                )
            
            # Process ended without result or exception
            return SandboxResult(
                status=SandboxStatus.CRASHED,
                exception=RuntimeError("Extension process ended unexpectedly"),
                execution_time=time.time() - self.start_time,
                restart_count=self.restart_count
            )
            
        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.CRASHED,
                exception=e,
                execution_time=time.time() - self.start_time,
                restart_count=self.restart_count
            )
        finally:
            self.stop_watchdog.set()
            if self.watchdog_thread and self.watchdog_thread.is_alive():
                self.watchdog_thread.join(timeout=1.0)
            self._cleanup_resources()
    
    def _run_in_sandbox(self, func: Callable, args: tuple, kwargs: dict, 
                       result_queue: multiprocessing.Queue, 
                       exception_queue: multiprocessing.Queue):
        """Target function that runs in the sandboxed process."""
        try:
            # Set resource limits for this process
            self._set_resource_limits()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Send result back
            result_queue.put(result)
            
        except Exception as e:
            # Capture and send exception
            exception_queue.put(e)
            traceback.print_exc()
    
    def _set_resource_limits(self):
        """Set resource limits for the sandboxed process."""
        try:
            # Set memory limit (virtual memory)
            memory_bytes = self.config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # Set CPU time limit (soft and hard)
            resource.setrlimit(resource.RLIMIT_CPU, 
                             (self.config.timeout_seconds, self.config.timeout_seconds))
            
            # Limit number of child processes (prevent fork bombs)
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
            
        except (ValueError, resource.error) as e:
            # Resource limits might not be available on all systems
            print(f"Warning: Could not set resource limits: {e}")
    
    def _watchdog_monitor(self):
        """Monitor the sandboxed process for resource violations."""
        while not self.stop_watchdog.is_set():
            try:
                if self.process and self.process.is_alive():
                    # Check memory usage
                    try:
                        process = psutil.Process(self.process.pid)
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                        
                        self._peak_memory = max(self._peak_memory, memory_mb)
                        
                        if memory_mb > self.config.memory_limit_mb:
                            self._terminate_process()
                            return
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    # Check CPU usage
                    try:
                        cpu_percent = process.cpu_percent(interval=0.1)
                        # If CPU usage is consistently high, might indicate infinite loop
                        # This is a simple heuristic
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                time.sleep(self.config.watchdog_interval)
            except Exception as e:
                print(f"Watchdog error: {e}")
                break
    
    def _terminate_process(self):
        """Terminate the sandboxed process and all its children."""
        if self.process and self.process.is_alive():
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                
                # Terminate children first
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                # Terminate parent
                self.process.terminate()
                
                # Wait a bit for graceful termination
                self.process.join(timeout=2.0)
                
                # Force kill if still alive
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1.0)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process already terminated
                pass
    
    def _cleanup_resources(self):
        """Clean up resources used by the sandbox."""
        try:
            # Clear queues
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.exception_queue.empty():
                try:
                    self.exception_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass


@dataclasses.dataclass
class ExtensionVersion:
    """Represents an extension version with semantic versioning support."""
    major: int = 0
    minor: int = 0
    patch: int = 0
    build: str = ""
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ExtensionVersion':
        """Parse version string like '1.2.3' or '1.2.3-beta'."""
        if not version_str:
            return cls()
        
        # Remove build metadata if present
        version_str = version_str.split('+')[0]
        
        # Handle pre-release versions
        if '-' in version_str:
            version_str, build = version_str.split('-', 1)
        else:
            build = ""
        
        parts = version_str.split('.')
        try:
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return cls(major, minor, patch, build)
        except (ValueError, IndexError):
            return cls()
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"-{self.build}"
        return version
    
    def is_compatible_with(self, other: 'ExtensionVersion') -> bool:
        """Check if this version is backward compatible with another version."""
        # Same major version means backward compatible
        return self.major == other.major
    
    def __eq__(self, other):
        if not isinstance(other, ExtensionVersion):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch and 
                self.build == other.build)
    
    def __lt__(self, other):
        if not isinstance(other, ExtensionVersion):
            return NotImplemented
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        return self.build < other.build


class ExtensionProxy:
    """Proxy object for extension interfaces that can be swapped at runtime."""
    
    def __init__(self, target=None):
        self._target = target
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def set_target(self, target):
        """Set new target for the proxy."""
        with self._lock:
            old_target = self._target
            self._target = target
            # Notify callbacks about target change
            for callback in self._callbacks.get('target_changed', []):
                try:
                    callback(old_target, target)
                except Exception as e:
                    print(f"Error in proxy callback: {e}")
    
    def get_target(self):
        """Get current target."""
        with self._lock:
            return self._target
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for proxy events."""
        with self._lock:
            self._callbacks[event].append(callback)
    
    def __getattr__(self, name):
        """Delegate attribute access to target."""
        with self._lock:
            if self._target is None:
                raise AttributeError(f"Extension proxy has no target set")
            return getattr(self._target, name)
    
    def __call__(self, *args, **kwargs):
        """Delegate calls to target."""
        with self._lock:
            if self._target is None:
                raise RuntimeError("Extension proxy has no target set")
            if callable(self._target):
                return self._target(*args, **kwargs)
            raise TypeError(f"Extension target {self._target} is not callable")
    
    def __repr__(self):
        with self._lock:
            return f"<ExtensionProxy target={self._target}>"


class DependencyResolver:
    """Resolves dependencies between extensions."""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, extension: str, depends_on: str):
        """Add a dependency relationship."""
        self.dependency_graph[extension].add(depends_on)
        self.reverse_graph[depends_on].add(extension)
    
    def get_dependencies(self, extension: str, recursive: bool = True) -> Set[str]:
        """Get all dependencies for an extension."""
        if not recursive:
            return self.dependency_graph.get(extension, set()).copy()
        
        visited = set()
        stack = [extension]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for dep in self.dependency_graph.get(current, set()):
                if dep not in visited:
                    stack.append(dep)
        
        visited.remove(extension)  # Remove the extension itself
        return visited
    
    def get_dependents(self, extension: str, recursive: bool = True) -> Set[str]:
        """Get all extensions that depend on this extension."""
        if not recursive:
            return self.reverse_graph.get(extension, set()).copy()
        
        visited = set()
        stack = [extension]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for dep in self.reverse_graph.get(current, set()):
                if dep not in visited:
                    stack.append(dep)
        
        visited.remove(extension)  # Remove the extension itself
        return visited
    
    def has_circular_dependency(self) -> bool:
        """Check if there are circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def visit(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if node not in visited:
                if visit(node):
                    return True
        return False
    
    def topological_sort(self) -> List[str]:
        """Get extensions in dependency order (dependencies first)."""
        in_degree = defaultdict(int)
        for ext in self.dependency_graph:
            in_degree[ext] = in_degree.get(ext, 0)
            for dep in self.dependency_graph[ext]:
                in_degree[dep] = in_degree.get(dep, 0) + 1
        
        queue = [ext for ext in in_degree if in_degree[ext] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for dependent in self.reverse_graph.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(in_degree):
            raise ValueError("Circular dependency detected")
        
        return result


class ExtensionManager:
    """Manages hot-reloadable extensions with versioning and dependency resolution."""
    
    def __init__(self):
        self.extensions: Dict[str, 'Extension'] = {}
        self.enabled_extensions: Set[str] = set()
        self.disabled_extensions: Set[str] = set()
        self.dependency_resolver = DependencyResolver()
        self.proxies: Dict[str, ExtensionProxy] = {}
        self.update_lock = threading.RLock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog = threading.Event()
        self._update_queue: queue.Queue = queue.Queue()
        self._extension_states: Dict[str, Dict] = {}  # For storing state during reload
    
    def start_watchdog(self):
        """Start the watchdog thread for monitoring extensions."""
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._stop_watchdog.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="ExtensionWatchdog"
            )
            self._watchdog_thread.start()
    
    def stop_watchdog(self):
        """Stop the watchdog thread."""
        self._stop_watchdog.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5.0)
    
    def _watchdog_loop(self):
        """Main watchdog loop for monitoring and auto-recovery."""
        while not self._stop_watchdog.is_set():
            try:
                # Process update queue
                self._process_update_queue()
                
                # Check for crashed extensions
                self._check_extension_health()
                
                time.sleep(1.0)  # Check every second
            except Exception as e:
                print(f"Watchdog error: {e}")
                time.sleep(5.0)
    
    def _process_update_queue(self):
        """Process pending extension updates."""
        try:
            while True:
                try:
                    update_task = self._update_queue.get_nowait()
                    self._perform_extension_update(update_task)
                    self._update_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error processing update queue: {e}")
    
    def _check_extension_health(self):
        """Check health of all extensions and restart if needed."""
        for ext_name, ext in list(self.extensions.items()):
            if ext_name in self.enabled_extensions:
                try:
                    # Check if extension is responsive
                    if hasattr(ext, 'health_check'):
                        if not ext.health_check():
                            print(f"Extension {ext_name} failed health check, attempting restart")
                            self.reload_extension(ext_name)
                except Exception as e:
                    print(f"Health check error for {ext_name}: {e}")
    
    def register_extension(self, extension: 'Extension'):
        """Register an extension with the manager."""
        with self.update_lock:
            self.extensions[extension.name] = extension
            
            # Create proxy if not exists
            if extension.name not in self.proxies:
                self.proxies[extension.name] = ExtensionProxy()
            
            # Update dependency graph
            for dep in extension.dependencies:
                self.dependency_resolver.add_dependency(extension.name, dep)
    
    def unregister_extension(self, extension_name: str):
        """Unregister an extension from the manager."""
        with self.update_lock:
            if extension_name in self.extensions:
                # Disable first
                self.disable_extension(extension_name)
                
                # Remove from collections
                del self.extensions[extension_name]
                if extension_name in self.proxies:
                    del self.proxies[extension_name]
                self.enabled_extensions.discard(extension_name)
                self.disabled_extensions.discard(extension_name)
    
    def enable_extension(self, extension_name: str):
        """Enable an extension."""
        with self.update_lock:
            if extension_name not in self.extensions:
                raise ValueError(f"Extension {extension_name} not found")
            
            # Check dependencies
            dependencies = self.dependency_resolver.get_dependencies(extension_name)
            for dep in dependencies:
                if dep not in self.enabled_extensions:
                    raise RuntimeError(f"Dependency {dep} not enabled for {extension_name}")
            
            # Enable the extension
            self.enabled_extensions.add(extension_name)
            self.disabled_extensions.discard(extension_name)
            
            # Load the extension
            self._load_extension(extension_name)
    
    def disable_extension(self, extension_name: str):
        """Disable an extension."""
        with self.update_lock:
            if extension_name not in self.extensions:
                return
            
            # Check if other enabled extensions depend on this
            dependents = self.dependency_resolver.get_dependents(extension_name)
            enabled_dependents = dependents.intersection(self.enabled_extensions)
            if enabled_dependents:
                raise RuntimeError(
                    f"Cannot disable {extension_name}: extensions {enabled_dependents} depend on it"
                )
            
            # Unload the extension
            self._unload_extension(extension_name)
            
            # Disable
            self.enabled_extensions.discard(extension_name)
            self.disabled_extensions.add(extension_name)
    
    def reload_extension(self, extension_name: str):
        """Reload an extension without restarting the UI."""
        with self.update_lock:
            if extension_name not in self.extensions:
                raise ValueError(f"Extension {extension_name} not found")
            
            if extension_name not in self.enabled_extensions:
                return  # Nothing to reload
            
            print(f"Reloading extension: {extension_name}")
            
            # Store current state
            self._store_extension_state(extension_name)
            
            try:
                # Unload
                self._unload_extension(extension_name)
                
                # Reload module
                self._reload_extension_module(extension_name)
                
                # Load again
                self._load_extension(extension_name)
                
                # Restore state
                self._restore_extension_state(extension_name)
                
                print(f"Successfully reloaded extension: {extension_name}")
                
            except Exception as e:
                print(f"Failed to reload extension {extension_name}: {e}")
                # Try to restore previous state
                try:
                    self._load_extension(extension_name)
                    self._restore_extension_state(extension_name)
                except Exception as restore_error:
                    print(f"Failed to restore extension {extension_name}: {restore_error}")
                    # Mark as disabled
                    self.enabled_extensions.discard(extension_name)
                    self.disabled_extensions.add(extension_name)
                raise
    
    def update_extension(self, extension_name: str, version: Optional[str] = None):
        """Queue an extension update."""
        with self.update_lock:
            if extension_name not in self.extensions:
                raise ValueError(f"Extension {extension_name} not found")
            
            self._update_queue.put({
                'extension_name': extension_name,
                'target_version': version,
                'timestamp': time.time()
            })
    
    def _perform_extension_update(self, update_task: Dict):
        """Perform the actual extension update."""
        ext_name = update_task['extension_name']
        target_version = update_task.get('target_version')
        
        try:
            extension = self.extensions[ext_name]
            
            # Check if update is needed
            current_version = extension.version
            if target_version and current_version == ExtensionVersion.from_string(target_version):
                print(f"Extension {ext_name} already at version {target_version}")
                return
            
            # Perform update (e.g., git pull)
            if hasattr(extension, 'update'):
                print(f"Updating extension {ext_name}...")
                extension.update(target_version)
            
            # Reload the extension
            if ext_name in self.enabled_extensions:
                self.reload_extension(ext_name)
            
            print(f"Successfully updated extension {ext_name}")
            
        except Exception as e:
            print(f"Failed to update extension {ext_name}: {e}")
            traceback.print_exc()
    
    def _load_extension(self, extension_name: str):
        """Load an extension module."""
        extension = self.extensions[extension_name]
        
        try:
            # Import or reload the module
            module_name = f"extensions.{extension_name}"
            
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    os.path.join(extension.path, "__init__.py")
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            
            # Update proxy
            if extension_name in self.proxies:
                self.proxies[extension_name].set_target(module)
            
            # Call extension's load callback if exists
            if hasattr(module, 'on_load'):
                module.on_load()
            
            print(f"Loaded extension: {extension_name}")
            
        except Exception as e:
            print(f"Failed to load extension {extension_name}: {e}")
            raise
    
    def _unload_extension(self, extension_name: str):
        """Unload an extension module."""
        try:
            module_name = f"extensions.{extension_name}"
            
            if module_name in sys.modules:
                module = sys.modules[module_name]
                
                # Call extension's unload callback if exists
                if hasattr(module, 'on_unload'):
                    try:
                        module.on_unload()
                    except Exception as e:
                        print(f"Error in on_unload for {extension_name}: {e}")
                
                # Clear proxy
                if extension_name in self.proxies:
                    self.proxies[extension_name].set_target(None)
                
                # Remove from sys.modules
                del sys.modules[module_name]
                
                print(f"Unloaded extension: {extension_name}")
                
        except Exception as e:
            print(f"Error unloading extension {extension_name}: {e}")
    
    def _reload_extension_module(self, extension_name: str):
        """Reload the extension module."""
        module_name = f"extensions.{extension_name}"
        
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception as e:
                print(f"Error reloading module for {extension_name}: {e}")
                raise
    
    def _store_extension_state(self, extension_name: str):
        """Store extension state before reload."""
        module_name = f"extensions.{extension_name}"
        
        if module_name in sys.modules:
            module = sys.modules[module_name]
            
            # Store state if extension has a state dict
            if hasattr(module, '__extension_state__'):
                self._extension_states[extension_name] = module.__extension_state__.copy()
            else:
                # Try to store common state attributes
                state = {}
                for attr in dir(module):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(module, attr)
                            # Only store simple types
                            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                                state[attr] = value
                        except:
                            pass
                self._extension_states[extension_name] = state
    
    def _restore_extension_state(self, extension_name: str):
        """Restore extension state after reload."""
        if extension_name in self._extension_states:
            module_name = f"extensions.{extension_name}"
            
            if module_name in sys.modules:
                module = sys.modules[module_name]
                state = self._extension_states[extension_name]
                
                # Restore state if extension has a state dict
                if hasattr(module, '__extension_state__'):
                    module.__extension_state__.update(state)
                else:
                    # Try to restore attributes
                    for attr, value in state.items():
                        try:
                            if hasattr(module, attr):
                                setattr(module, attr, value)
                        except:
                            pass
            
            # Clean up stored state
            del self._extension_states[extension_name]
    
    def get_extension_info(self, extension_name: str) -> Dict:
        """Get information about an extension."""
        if extension_name not in self.extensions:
            return {}
        
        extension = self.extensions[extension_name]
        dependencies = self.dependency_resolver.get_dependencies(extension_name)
        dependents = self.dependency_resolver.get_dependents(extension_name)
        
        return {
            'name': extension.name,
            'version': str(extension.version),
            'enabled': extension_name in self.enabled_extensions,
            'path': extension.path,
            'dependencies': list(dependencies),
            'dependents': list(dependents),
            'has_proxy': extension_name in self.proxies,
            'proxy_target': self.proxies[extension_name].get_target() if extension_name in self.proxies else None
        }
    
    def get_all_extensions_info(self) -> List[Dict]:
        """Get information about all extensions."""
        return [self.get_extension_info(name) for name in self.extensions.keys()]
    
    def validate_dependencies(self) -> List[str]:
        """Validate all extension dependencies and return any issues."""
        issues = []
        
        for ext_name, ext in self.extensions.items():
            for dep in ext.dependencies:
                if dep not in self.extensions:
                    issues.append(f"Extension {ext_name} depends on missing extension {dep}")
                elif not ext.version.is_compatible_with(self.extensions[dep].version):
                    issues.append(
                        f"Extension {ext_name} version {ext.version} may not be compatible with "
                        f"{dep} version {self.extensions[dep].version}"
                    )
        
        if self.dependency_resolver.has_circular_dependency():
            issues.append("Circular dependency detected in extensions")
        
        return issues


class Extension:
    """Represents an extension with hot-reload capabilities."""
    
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.version = self._detect_version()
        self.dependencies = self._detect_dependencies()
        self.enabled = True
        self.sandbox_config = SandboxConfig()
        self.sandbox: Optional[ExtensionSandbox] = None
        self._module_hash: Optional[str] = None
        self._last_modified: float = 0.0
    
    def _detect_version(self) -> ExtensionVersion:
        """Detect extension version from various sources."""
        # Try to read from metadata file
        metadata_files = [
            os.path.join(self.path, 'metadata.json'),
            os.path.join(self.path, 'version.txt'),
            os.path.join(self.path, 'package.json')
        ]
        
        for metadata_file in metadata_files:
            if os.path.exists(metadata_file):
                try:
                    if metadata_file.endswith('.json'):
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            version_str = data.get('version', '')
                    else:
                        with open(metadata_file, 'r') as f:
                            version_str = f.read().strip()
                    
                    if version_str:
                        return ExtensionVersion.from_string(version_str)
                except Exception as e:
                    print(f"Error reading version from {metadata_file}: {e}")
        
        # Try to get from git tags
        if os.path.exists(os.path.join(self.path, '.git')):
            try:
                repo = Repo(self.path)
                tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
                if tags:
                    return ExtensionVersion.from_string(str(tags[-1]))
            except Exception:
                pass
        
        # Default version
        return ExtensionVersion()
    
    def _detect_dependencies(self) -> List[str]:
        """Detect extension dependencies from various sources."""
        dependencies = []
        
        # Try to read from metadata file
        metadata_files = [
            os.path.join(self.path, 'metadata.json'),
            os.path.join(self.path, 'requirements.txt'),
            os.path.join(self.path, 'dependencies.txt')
        ]
        
        for metadata_file in metadata_files:
            if os.path.exists(metadata_file):
                try:
                    if metadata_file.endswith('.json'):
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            deps = data.get('dependencies', [])
                            if isinstance(deps, list):
                                dependencies.extend(deps)
                            elif isinstance(deps, dict):
                                dependencies.extend(deps.keys())
                    else:
                        with open(metadata_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    # Extract extension name from dependency spec
                                    # Simple implementation - assumes extension name is first part
                                    dep_name = re.split(r'[>=<]', line)[0].strip()
                                    if dep_name:
                                        dependencies.append(dep_name)
                except Exception as e:
                    print(f"Error reading dependencies from {metadata_file}: {e}")
        
        # Remove duplicates and self-dependencies
        dependencies = list(set(dependencies))
        if self.name in dependencies:
            dependencies.remove(self.name)
        
        return dependencies
    
    def update(self, target_version: Optional[str] = None):
        """Update the extension to a specific version."""
        if not os.path.exists(os.path.join(self.path, '.git')):
            raise RuntimeError(f"Extension {self.name} is not a git repository")
        
        try:
            repo = Repo(self.path)
            
            # Fetch latest changes
            origin = repo.remotes.origin
            origin.fetch()
            
            if target_version:
                # Checkout specific version
                repo.git.checkout(target_version)
            else:
                # Pull latest changes
                origin.pull()
            
            # Update version
            self.version = self._detect_version()
            self.dependencies = self._detect_dependencies()
            
            print(f"Updated extension {self.name} to version {self.version}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to update extension {self.name}: {e}")
    
    def get_file_hash(self) -> str:
        """Get hash of extension files for change detection."""
        hasher = hashlib.md5()
        
        for root, dirs, files in os.walk(self.path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in sorted(files):
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            hasher.update(f.read())
                    except Exception:
                        pass
        
        return hasher.hexdigest()
    
    def has_changed(self) -> bool:
        """Check if extension files have changed since last check."""
        current_hash = self.get_file_hash()
        current_modified = os.path.getmtime(self.path)
        
        if (self._module_hash != current_hash or 
            self._last_modified != current_modified):
            self._module_hash = current_hash
            self._last_modified = current_modified
            return True
        
        return False
    
    def health_check(self) -> bool:
        """Perform health check on the extension."""
        try:
            # Check if main files exist
            init_file = os.path.join(self.path, '__init__.py')
            if not os.path.exists(init_file):
                return False
            
            # Check if extension is importable
            module_name = f"extensions.{self.name}"
            if module_name in sys.modules:
                module = sys.modules[module_name]
                # Check if module has required attributes
                if hasattr(module, 'health_check'):
                    return module.health_check()
            
            return True
        except Exception:
            return False


# Global extension manager instance
extension_manager = ExtensionManager()


def initialize():
    """Initialize the extension system."""
    # Start the watchdog
    extension_manager.start_watchdog()
    
    # Load built-in extensions
    load_builtin_extensions()
    
    # Load user extensions
    load_user_extensions()


def load_builtin_extensions():
    """Load built-in extensions."""
    if not os.path.exists(extensions_builtin_dir):
        return
    
    for item in os.listdir(extensions_builtin_dir):
        item_path = os.path.join(extensions_builtin_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            try:
                extension = Extension(item, item_path)
                extension_manager.register_extension(extension)
                extension_manager.enable_extension(item)
                print(f"Loaded built-in extension: {item}")
            except Exception as e:
                print(f"Failed to load built-in extension {item}: {e}")


def load_user_extensions():
    """Load user extensions from extensions directory."""
    if not os.path.exists(extensions_dir):
        return
    
    for item in os.listdir(extensions_dir):
        item_path = os.path.join(extensions_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            try:
                extension = Extension(item, item_path)
                extension_manager.register_extension(extension)
                
                # Check if extension should be enabled by default
                # Could be controlled by a config file
                enabled_by_default = True
                config_file = os.path.join(item_path, 'config.json')
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            enabled_by_default = config.get('enabled', True)
                    except Exception:
                        pass
                
                if enabled_by_default:
                    extension_manager.enable_extension(item)
                    print(f"Loaded user extension: {item}")
                else:
                    extension_manager.disable_extension(item)
                    print(f"Registered disabled extension: {item}")
                    
            except Exception as e:
                print(f"Failed to load user extension {item}: {e}")


def reload_extension(extension_name: str):
    """Reload a specific extension."""
    extension_manager.reload_extension(extension_name)


def update_extension(extension_name: str, version: Optional[str] = None):
    """Update a specific extension."""
    extension_manager.update_extension(extension_name, version)


def enable_extension(extension_name: str):
    """Enable a specific extension."""
    extension_manager.enable_extension(extension_name)


def disable_extension(extension_name: str):
    """Disable a specific extension."""
    extension_manager.disable_extension(extension_name)


def get_extension_proxy(extension_name: str) -> Optional[ExtensionProxy]:
    """Get proxy for a specific extension."""
    return extension_manager.proxies.get(extension_name)


def get_extension_info(extension_name: str) -> Dict:
    """Get information about a specific extension."""
    return extension_manager.get_extension_info(extension_name)


def get_all_extensions_info() -> List[Dict]:
    """Get information about all extensions."""
    return extension_manager.get_all_extensions_info()


def validate_dependencies() -> List[str]:
    """Validate all extension dependencies."""
    return extension_manager.validate_dependencies()


def cleanup():
    """Clean up the extension system."""
    extension_manager.stop_watchdog()
    
    # Unload all extensions
    for ext_name in list(extension_manager.extensions.keys()):
        try:
            extension_manager.disable_extension(ext_name)
        except Exception as e:
            print(f"Error disabling extension {ext_name}: {e}")
    
    print("Extension system cleaned up")


# Register cleanup handler
import atexit
atexit.register(cleanup)