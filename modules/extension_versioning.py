import os
import sys
import json
import hashlib
import importlib
import importlib.util
import threading
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from packaging import version as packaging_version
import inspect
import weakref

# Import existing modules for integration
try:
    from modules import extensions, script_callbacks, shared
    from modules.paths import extensions_dir, script_path
except ImportError:
    # Fallback for standalone testing
    extensions_dir = os.path.join(os.path.dirname(__file__), '..', 'extensions')
    script_path = os.path.join(os.path.dirname(__file__), '..')


class ExtensionState(Enum):
    """Extension lifecycle states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class ExtensionVersion:
    """Semantic version with compatibility tracking"""
    major: int = 0
    minor: int = 0
    patch: int = 0
    build: str = ""
    api_version: str = "1.0.0"
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ExtensionVersion':
        """Parse version string like '1.2.3-beta'"""
        if not version_str:
            return cls()
        
        parts = version_str.split('-')
        version_parts = parts[0].split('.')
        
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        build = parts[1] if len(parts) > 1 else ""
        
        return cls(major=major, minor=minor, patch=patch, build=build)
    
    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            base += f"-{self.build}"
        return base
    
    def __eq__(self, other):
        if not isinstance(other, ExtensionVersion):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch)
    
    def __lt__(self, other):
        if not isinstance(other, ExtensionVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def is_compatible_with(self, other: 'ExtensionVersion') -> bool:
        """Check backward compatibility (same major version)"""
        return self.major == other.major


@dataclass
class ExtensionDependency:
    """Extension dependency specification"""
    name: str
    version_constraint: str = "*"  # e.g., ">=1.0.0", "~1.2.0", "^2.0.0"
    optional: bool = False
    
    def satisfies(self, ext_version: ExtensionVersion) -> bool:
        """Check if version satisfies constraint"""
        if self.version_constraint == "*":
            return True
        
        try:
            # Parse constraint
            if self.version_constraint.startswith(">="):
                required = ExtensionVersion.from_string(self.version_constraint[2:])
                return ext_version >= required
            elif self.version_constraint.startswith(">"):
                required = ExtensionVersion.from_string(self.version_constraint[1:])
                return ext_version > required
            elif self.version_constraint.startswith("<="):
                required = ExtensionVersion.from_string(self.version_constraint[2:])
                return ext_version <= required
            elif self.version_constraint.startswith("<"):
                required = ExtensionVersion.from_string(self.version_constraint[1:])
                return ext_version < required
            elif self.version_constraint.startswith("~"):
                # Tilde: same major.minor, patch can be higher
                required = ExtensionVersion.from_string(self.version_constraint[1:])
                return (ext_version.major == required.major and 
                        ext_version.minor == required.minor and
                        ext_version >= required)
            elif self.version_constraint.startswith("^"):
                # Caret: same major, minor/patch can be higher
                required = ExtensionVersion.from_string(self.version_constraint[1:])
                return ext_version.major == required.major and ext_version >= required
            else:
                # Exact version
                required = ExtensionVersion.from_string(self.version_constraint)
                return ext_version == required
        except (ValueError, IndexError):
            return False


class ExtensionProxy:
    """Proxy object for hot-swappable extension interfaces"""
    
    def __init__(self, extension_name: str):
        self._extension_name = extension_name
        self._implementation = None
        self._interface_version = "1.0.0"
        self._compatibility_layer = {}
        self._lock = threading.RLock()
        self._callbacks = weakref.WeakKeyDictionary()
        
    def swap_implementation(self, new_impl: Any, preserve_state: bool = True) -> bool:
        """Swap the underlying implementation at runtime"""
        with self._lock:
            old_impl = self._implementation
            
            # Migrate state if requested and possible
            if preserve_state and old_impl and hasattr(old_impl, '__getstate__'):
                try:
                    state = old_impl.__getstate__()
                    if hasattr(new_impl, '__setstate__'):
                        new_impl.__setstate__(state)
                except Exception as e:
                    print(f"Warning: Failed to migrate state for {self._extension_name}: {e}")
            
            # Install compatibility shims if needed
            self._install_compatibility_shims(new_impl)
            
            self._implementation = new_impl
            
            # Notify observers
            for callback in list(self._callbacks.values()):
                try:
                    callback(old_impl, new_impl)
                except Exception:
                    pass
            
            return True
    
    def _install_compatibility_shims(self, impl: Any):
        """Install backward compatibility layers"""
        if not hasattr(impl, '_api_version'):
            impl._api_version = self._interface_version
        
        # Add missing methods with default implementations
        for method_name, default_impl in self._compatibility_layer.items():
            if not hasattr(impl, method_name):
                setattr(impl, method_name, default_impl.__get__(impl, type(impl)))
    
    def register_compatibility_layer(self, method_name: str, default_impl: Callable):
        """Register a backward compatibility method"""
        self._compatibility_layer[method_name] = default_impl
    
    def register_change_callback(self, obj: Any, callback: Callable):
        """Register callback for implementation changes"""
        self._callbacks[id(obj)] = callback
    
    def unregister_change_callback(self, obj: Any):
        """Unregister callback"""
        self._callbacks.pop(id(obj), None)
    
    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if self._implementation is None:
            raise RuntimeError(f"Extension '{self._extension_name}' is not loaded")
        
        with self._lock:
            attr = getattr(self._implementation, name)
            
            # Wrap methods to handle runtime swapping
            if callable(attr):
                def wrapped(*args, **kwargs):
                    # Re-resolve method in case implementation changed
                    current_impl = self._implementation
                    if current_impl is None:
                        raise RuntimeError(f"Extension '{self._extension_name}' was unloaded")
                    
                    current_attr = getattr(current_impl, name)
                    return current_attr(*args, **kwargs)
                
                return wrapped
            return attr
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if self._implementation is None:
                raise RuntimeError(f"Extension '{self._extension_name}' is not loaded")
            setattr(self._implementation, name, value)
    
    def __call__(self, *args, **kwargs):
        if self._implementation is None:
            raise RuntimeError(f"Extension '{self._extension_name}' is not loaded")
        
        if callable(self._implementation):
            return self._implementation(*args, **kwargs)
        raise TypeError(f"Extension '{self._extension_name}' is not callable")
    
    def __repr__(self):
        return f"<ExtensionProxy '{self._extension_name}' impl={type(self._implementation).__name__ if self._implementation else 'None'}>"


class DependencyResolver:
    """Resolves extension dependencies and conflicts"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}
    
    def add_dependency(self, extension: str, depends_on: str):
        """Add a dependency relationship"""
        if extension not in self.dependency_graph:
            self.dependency_graph[extension] = set()
        self.dependency_graph[extension].add(depends_on)
        
        if depends_on not in self.reverse_graph:
            self.reverse_graph[depends_on] = set()
        self.reverse_graph[depends_on].add(extension)
    
    def get_dependents(self, extension: str) -> Set[str]:
        """Get all extensions that depend on this one"""
        visited = set()
        stack = [extension]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for dependent in self.reverse_graph.get(current, set()):
                if dependent not in visited:
                    stack.append(dependent)
        
        visited.discard(extension)
        return visited
    
    def can_disable(self, extension: str, active_extensions: Set[str]) -> Tuple[bool, List[str]]:
        """Check if extension can be disabled without breaking others"""
        dependents = self.get_dependents(extension)
        active_dependents = dependents.intersection(active_extensions)
        
        if active_dependents:
            return False, list(active_dependents)
        return True, []
    
    def resolve_update_order(self, extensions_to_update: Set[str], 
                           all_extensions: Dict[str, 'ExtensionInfo']) -> List[str]:
        """Determine safe order to update extensions"""
        # Topological sort considering dependencies
        visited = set()
        temp_mark = set()
        order = []
        
        def visit(ext_name: str):
            if ext_name in temp_mark:
                raise ValueError(f"Circular dependency detected involving {ext_name}")
            if ext_name in visited:
                return
            
            temp_mark.add(ext_name)
            
            # Visit dependencies first
            for dep_name in self.dependency_graph.get(ext_name, set()):
                if dep_name in all_extensions:
                    visit(dep_name)
            
            temp_mark.discard(ext_name)
            visited.add(ext_name)
            order.append(ext_name)
        
        for ext_name in extensions_to_update:
            if ext_name not in visited:
                visit(ext_name)
        
        return order


@dataclass
class ExtensionInfo:
    """Complete extension metadata and state"""
    name: str
    path: Path
    version: ExtensionVersion = field(default_factory=ExtensionVersion)
    state: ExtensionState = ExtensionState.UNLOADED
    dependencies: List[ExtensionDependency] = field(default_factory=list)
    api_version: str = "1.0.0"
    enabled: bool = True
    last_modified: float = 0.0
    checksum: str = ""
    module: Any = None
    proxy: Optional[ExtensionProxy] = None
    error: Optional[str] = None
    load_time: float = 0.0
    
    def calculate_checksum(self) -> str:
        """Calculate checksum of extension files"""
        hasher = hashlib.sha256()
        
        for root, dirs, files in os.walk(self.path):
            for file in sorted(files):
                if file.endswith('.py') or file.endswith('.json'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            hasher.update(f.read())
                    except Exception:
                        continue
        
        return hasher.hexdigest()
    
    def update_last_modified(self):
        """Update last modified timestamp"""
        max_mtime = 0
        for root, dirs, files in os.walk(self.path):
            for file in files:
                file_path = Path(root) / file
                try:
                    mtime = file_path.stat().st_mtime
                    max_mtime = max(max_mtime, mtime)
                except Exception:
                    continue
        self.last_modified = max_mtime


class ExtensionVersionManager:
    """Manages extension versioning and hot-reloading"""
    
    def __init__(self, extensions_dir: Optional[str] = None):
        self.extensions_dir = Path(extensions_dir or globals().get('extensions_dir', 'extensions'))
        self.extensions: Dict[str, ExtensionInfo] = {}
        self.proxies: Dict[str, ExtensionProxy] = {}
        self.dependency_resolver = DependencyResolver()
        self.watcher_thread: Optional[threading.Thread] = None
        self.watcher_running = False
        self.lock = threading.RLock()
        self.update_callbacks: List[Callable] = []
        
        # Cache for loaded modules
        self._module_cache: Dict[str, Any] = {}
        
        # Initialize
        self._scan_extensions()
    
    def _scan_extensions(self):
        """Scan extensions directory for available extensions"""
        if not self.extensions_dir.exists():
            return
        
        for item in self.extensions_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                self._register_extension(item)
    
    def _register_extension(self, path: Path):
        """Register an extension directory"""
        try:
            # Load extension metadata
            metadata_file = path / 'extension.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            ext_name = metadata.get('name', path.name)
            ext_version = ExtensionVersion.from_string(metadata.get('version', '0.0.0'))
            
            # Parse dependencies
            dependencies = []
            for dep in metadata.get('dependencies', []):
                if isinstance(dep, str):
                    dependencies.append(ExtensionDependency(name=dep))
                elif isinstance(dep, dict):
                    dependencies.append(ExtensionDependency(
                        name=dep['name'],
                        version_constraint=dep.get('version', '*'),
                        optional=dep.get('optional', False)
                    ))
            
            # Create extension info
            ext_info = ExtensionInfo(
                name=ext_name,
                path=path,
                version=ext_version,
                dependencies=dependencies,
                api_version=metadata.get('api_version', '1.0.0'),
                enabled=metadata.get('enabled', True)
            )
            
            ext_info.update_last_modified()
            ext_info.checksum = ext_info.calculate_checksum()
            
            # Create proxy
            proxy = ExtensionProxy(ext_name)
            ext_info.proxy = proxy
            self.proxies[ext_name] = proxy
            
            # Register in dependency graph
            for dep in dependencies:
                self.dependency_resolver.add_dependency(ext_name, dep.name)
            
            with self.lock:
                self.extensions[ext_name] = ext_info
            
            print(f"Registered extension: {ext_name} v{ext_version}")
            
        except Exception as e:
            print(f"Failed to register extension at {path}: {e}")
    
    def load_extension(self, ext_name: str, force_reload: bool = False) -> bool:
        """Load or reload an extension"""
        with self.lock:
            ext_info = self.extensions.get(ext_name)
            if not ext_info:
                print(f"Extension not found: {ext_name}")
                return False
            
            if ext_info.state == ExtensionState.LOADING:
                print(f"Extension {ext_name} is already loading")
                return False
            
            # Check dependencies
            if not self._check_dependencies(ext_info):
                ext_info.state = ExtensionState.ERROR
                ext_info.error = "Unmet dependencies"
                return False
            
            ext_info.state = ExtensionState.LOADING
            
            try:
                start_time = time.time()
                
                # Check if already loaded and needs reload
                if ext_info.module and not force_reload:
                    # Check if files changed
                    current_checksum = ext_info.calculate_checksum()
                    if current_checksum == ext_info.checksum:
                        ext_info.state = ExtensionState.LOADED
                        return True
                
                # Unload previous version if exists
                if ext_info.module:
                    self._unload_extension_module(ext_info)
                
                # Load the extension module
                module = self._load_extension_module(ext_info)
                
                if module:
                    ext_info.module = module
                    ext_info.state = ExtensionState.LOADED
                    ext_info.load_time = time.time() - start_time
                    ext_info.error = None
                    
                    # Update proxy
                    if ext_info.proxy:
                        ext_info.proxy.swap_implementation(module)
                    
                    # Update checksum
                    ext_info.checksum = ext_info.calculate_checksum()
                    
                    # Notify callbacks
                    self._notify_update(ext_name, "loaded")
                    
                    print(f"Loaded extension: {ext_name} in {ext_info.load_time:.2f}s")
                    return True
                else:
                    ext_info.state = ExtensionState.ERROR
                    ext_info.error = "Failed to load module"
                    return False
                    
            except Exception as e:
                ext_info.state = ExtensionState.ERROR
                ext_info.error = str(e)
                print(f"Failed to load extension {ext_name}: {e}")
                traceback.print_exc()
                return False
    
    def _load_extension_module(self, ext_info: ExtensionInfo) -> Any:
        """Load extension module with hot-reload support"""
        try:
            # Find main script
            main_script = ext_info.path / 'scripts' / 'main.py'
            if not main_script.exists():
                main_script = ext_info.path / 'main.py'
            
            if not main_script.exists():
                # Try to find any Python file
                py_files = list(ext_info.path.glob('*.py'))
                if py_files:
                    main_script = py_files[0]
                else:
                    return None
            
            # Generate unique module name
            module_name = f"extension_{ext_info.name}_{int(time.time())}"
            
            # Load module spec
            spec = importlib.util.spec_from_file_location(
                module_name,
                main_script,
                submodule_search_locations=[str(ext_info.path)]
            )
            
            if not spec or not spec.loader:
                return None
            
            # Create module
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules before execution (for imports within extension)
            sys.modules[module_name] = module
            
            # Set extension metadata
            module.__extension_name__ = ext_info.name
            module.__extension_version__ = str(ext_info.version)
            module.__extension_path__ = str(ext_info.path)
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Cache module
            self._module_cache[ext_info.name] = module
            
            return module
            
        except Exception as e:
            print(f"Error loading extension module {ext_info.name}: {e}")
            traceback.print_exc()
            return None
    
    def _unload_extension_module(self, ext_info: ExtensionInfo):
        """Unload extension module and clean up"""
        if ext_info.module:
            # Call cleanup if available
            if hasattr(ext_info.module, 'unload'):
                try:
                    ext_info.module.unload()
                except Exception as e:
                    print(f"Error during extension unload {ext_info.name}: {e}")
            
            # Remove from sys.modules
            module_name = ext_info.module.__name__
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Clear from cache
            self._module_cache.pop(ext_info.name, None)
            
            ext_info.module = None
    
    def _check_dependencies(self, ext_info: ExtensionInfo) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in ext_info.dependencies:
            dep_ext = self.extensions.get(dep.name)
            
            if not dep_ext:
                if dep.optional:
                    continue
                print(f"Missing dependency: {dep.name} for {ext_info.name}")
                return False
            
            if not dep.satisfies(dep_ext.version):
                if dep.optional:
                    continue
                print(f"Version mismatch: {dep.name} {dep_ext.version} does not satisfy {dep.version_constraint}")
                return False
            
            if dep_ext.state not in [ExtensionState.LOADED, ExtensionState.ACTIVE]:
                if dep.optional:
                    continue
                print(f"Dependency not loaded: {dep.name}")
                return False
        
        return True
    
    def enable_extension(self, ext_name: str) -> bool:
        """Enable an extension"""
        with self.lock:
            ext_info = self.extensions.get(ext_name)
            if not ext_info:
                return False
            
            if ext_info.state == ExtensionState.DISABLED:
                ext_info.enabled = True
                if self.load_extension(ext_name):
                    ext_info.state = ExtensionState.ACTIVE
                    self._notify_update(ext_name, "enabled")
                    return True
            
            return False
    
    def disable_extension(self, ext_name: str, force: bool = False) -> bool:
        """Disable an extension"""
        with self.lock:
            ext_info = self.extensions.get(ext_name)
            if not ext_info:
                return False
            
            # Check if other extensions depend on this one
            can_disable, dependents = self.dependency_resolver.can_disable(
                ext_name, 
                {name for name, info in self.extensions.items() 
                 if info.state in [ExtensionState.ACTIVE, ExtensionState.LOADED]}
            )
            
            if not can_disable and not force:
                print(f"Cannot disable {ext_name}: required by {', '.join(dependents)}")
                return False
            
            # Unload if loaded
            if ext_info.module:
                self._unload_extension_module(ext_info)
            
            ext_info.enabled = False
            ext_info.state = ExtensionState.DISABLED
            
            # Update proxy to return None/defaults
            if ext_info.proxy:
                ext_info.proxy.swap_implementation(None)
            
            self._notify_update(ext_name, "disabled")
            return True
    
    def update_extension(self, ext_name: str, new_path: Optional[Path] = None) -> bool:
        """Update an extension to a new version"""
        with self.lock:
            ext_info = self.extensions.get(ext_name)
            if not ext_info:
                return False
            
            old_state = ext_info.state
            ext_info.state = ExtensionState.UPDATING
            
            try:
                # If new path provided, update the path
                if new_path and new_path.exists():
                    ext_info.path = new_path
                
                # Re-scan metadata
                metadata_file = ext_info.path / 'extension.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    new_version = ExtensionVersion.from_string(metadata.get('version', '0.0.0'))
                    
                    # Check backward compatibility
                    if not ext_info.version.is_compatible_with(new_version):
                        print(f"Warning: Major version change for {ext_name}: {ext_info.version} -> {new_version}")
                    
                    ext_info.version = new_version
                
                # Reload if was loaded
                if old_state in [ExtensionState.LOADED, ExtensionState.ACTIVE]:
                    if self.load_extension(ext_name, force_reload=True):
                        if ext_info.enabled:
                            ext_info.state = ExtensionState.ACTIVE
                        else:
                            ext_info.state = ExtensionState.LOADED
                        self._notify_update(ext_name, "updated")
                        return True
                    else:
                        ext_info.state = ExtensionState.ERROR
                        return False
                else:
                    # Just update metadata
                    ext_info.update_last_modified()
                    ext_info.checksum = ext_info.calculate_checksum()
                    self._notify_update(ext_name, "updated")
                    return True
                    
            except Exception as e:
                ext_info.state = ExtensionState.ERROR
                ext_info.error = str(e)
                print(f"Failed to update extension {ext_name}: {e}")
                return False
    
    def hot_reload_extension(self, ext_name: str) -> bool:
        """Hot-reload a single extension (for development)"""
        return self.load_extension(ext_name, force_reload=True)
    
    def hot_reload_all(self):
        """Hot-reload all modified extensions"""
        for ext_name, ext_info in self.extensions.items():
            if ext_info.enabled and ext_info.state in [ExtensionState.LOADED, ExtensionState.ACTIVE]:
                # Check if files changed
                current_checksum = ext_info.calculate_checksum()
                if current_checksum != ext_info.checksum:
                    print(f"Hot-reloading modified extension: {ext_name}")
                    self.hot_reload_extension(ext_name)
    
    def start_file_watcher(self, interval: float = 2.0):
        """Start background thread to watch for file changes"""
        if self.watcher_running:
            return
        
        self.watcher_running = True
        
        def watcher_loop():
            while self.watcher_running:
                try:
                    self.hot_reload_all()
                except Exception as e:
                    print(f"Error in file watcher: {e}")
                time.sleep(interval)
        
        self.watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
        self.watcher_thread.start()
        print(f"Started extension file watcher (interval: {interval}s)")
    
    def stop_file_watcher(self):
        """Stop the file watcher thread"""
        self.watcher_running = False
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5.0)
            self.watcher_thread = None
    
    def register_update_callback(self, callback: Callable):
        """Register callback for extension updates"""
        self.update_callbacks.append(callback)
    
    def _notify_update(self, ext_name: str, action: str):
        """Notify all registered callbacks of an extension update"""
        for callback in self.update_callbacks:
            try:
                callback(ext_name, action)
            except Exception as e:
                print(f"Error in update callback: {e}")
    
    def get_extension_info(self, ext_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an extension"""
        ext_info = self.extensions.get(ext_name)
        if not ext_info:
            return None
        
        return {
            'name': ext_info.name,
            'version': str(ext_info.version),
            'state': ext_info.state.value,
            'enabled': ext_info.enabled,
            'path': str(ext_info.path),
            'dependencies': [
                {'name': dep.name, 'version': dep.version_constraint, 'optional': dep.optional}
                for dep in ext_info.dependencies
            ],
            'api_version': ext_info.api_version,
            'load_time': ext_info.load_time,
            'error': ext_info.error,
            'last_modified': ext_info.last_modified,
            'checksum': ext_info.checksum[:16] + '...' if ext_info.checksum else ''
        }
    
    def list_extensions(self) -> List[Dict[str, Any]]:
        """List all registered extensions"""
        return [
            self.get_extension_info(name)
            for name in sorted(self.extensions.keys())
        ]
    
    def resolve_dependencies(self, ext_name: str) -> List[str]:
        """Get ordered list of dependencies for an extension"""
        ext_info = self.extensions.get(ext_name)
        if not ext_info:
            return []
        
        resolved = []
        visited = set()
        
        def resolve(name: str):
            if name in visited:
                return
            visited.add(name)
            
            info = self.extensions.get(name)
            if not info:
                return
            
            for dep in info.dependencies:
                resolve(dep.name)
            
            resolved.append(name)
        
        resolve(ext_name)
        return resolved[:-1]  # Exclude the extension itself


# Global instance for integration with existing codebase
extension_manager = ExtensionVersionManager()

# Integration with existing modules.extensions system
def integrate_with_existing_system():
    """Integrate with the existing extension system in forge"""
    try:
        from modules import extensions as existing_extensions
        
        # Override the extension loading mechanism
        original_load_extensions = getattr(existing_extensions, 'load_extensions', None)
        
        def enhanced_load_extensions():
            """Enhanced extension loading with versioning support"""
            # Load built-in extensions first
            if original_load_extensions:
                original_load_extensions()
            
            # Load versioned extensions
            for ext_name, ext_info in extension_manager.extensions.items():
                if ext_info.enabled and ext_info.state == ExtensionState.UNLOADED:
                    extension_manager.load_extension(ext_name)
                    if ext_info.state == ExtensionState.LOADED:
                        ext_info.state = ExtensionState.ACTIVE
        
        # Replace if exists
        if original_load_extensions:
            existing_extensions.load_extensions = enhanced_load_extensions
        
        # Add hot-reload capability
        existing_extensions.hot_reload = extension_manager.hot_reload_all
        existing_extensions.hot_reload_extension = extension_manager.hot_reload_extension
        existing_extensions.get_extension_info = extension_manager.get_extension_info
        existing_extensions.list_extensions = extension_manager.list_extensions
        
        # Start file watcher in development mode
        if os.getenv('WEBUI_RELOAD_EXTENSIONS', 'false').lower() == 'true':
            extension_manager.start_file_watcher()
        
        print("Extension versioning system integrated successfully")
        
    except ImportError:
        print("Warning: Could not integrate with existing extensions module")
    except Exception as e:
        print(f"Error during integration: {e}")
        traceback.print_exc()


# Script callback integration
def on_script_loaded(callback):
    """Register callback for when scripts are loaded"""
    extension_manager.register_update_callback(
        lambda name, action: callback(name) if action == 'loaded' else None
    )


# Auto-integrate when module is imported
try:
    integrate_with_existing_system()
except Exception as e:
    print(f"Auto-integration failed: {e}")


# Example extension interface for compatibility
class ExtensionInterface:
    """Base class for extension interfaces with versioning support"""
    
    def __init__(self):
        self._api_version = "1.0.0"
    
    def get_api_version(self) -> str:
        return self._api_version
    
    def is_compatible(self, required_version: str) -> bool:
        """Check if this extension is compatible with required API version"""
        try:
            current = packaging_version.parse(self._api_version)
            required = packaging_version.parse(required_version)
            return current >= required
        except Exception:
            return False
    
    def __getstate__(self):
        """Get state for hot-reload migration"""
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Restore state after hot-reload"""
        self.__dict__.update(state)


# Utility functions for extension developers
def get_extension_proxy(ext_name: str) -> Optional[ExtensionProxy]:
    """Get proxy object for an extension"""
    return extension_manager.proxies.get(ext_name)


def require_extension(ext_name: str, min_version: Optional[str] = None):
    """Decorator to require another extension"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            ext_info = extension_manager.extensions.get(ext_name)
            if not ext_info:
                raise ImportError(f"Required extension not found: {ext_name}")
            
            if ext_info.state not in [ExtensionState.LOADED, ExtensionState.ACTIVE]:
                raise ImportError(f"Required extension not loaded: {ext_name}")
            
            if min_version:
                required = ExtensionVersion.from_string(min_version)
                if ext_info.version < required:
                    raise ImportError(
                        f"Extension {ext_name} version {ext_info.version} < required {min_version}"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Export public API
__all__ = [
    'ExtensionVersionManager',
    'ExtensionProxy',
    'ExtensionInfo',
    'ExtensionVersion',
    'ExtensionDependency',
    'ExtensionState',
    'extension_manager',
    'get_extension_proxy',
    'require_extension',
    'on_script_loaded',
    'ExtensionInterface',
]