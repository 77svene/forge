"""Plugin registry and discovery system for forge.

This module provides a formal plugin ecosystem that allows community contributions
of new models, training strategies, and evaluation metrics without modifying core code.
Includes automatic plugin discovery, version compatibility checking, and CLI management.
"""

import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from packaging import version
import pkg_resources

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class PluginType(Enum):
    """Types of plugins supported by forge."""
    MODEL = "model"
    TRAINER = "trainer"
    METRIC = "metric"
    DATA_PROCESSOR = "data_processor"
    CALLBACK = "callback"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    compatible_versions: List[str] = field(default_factory=list)  # SemVer ranges
    tags: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    installed: bool = False
    enabled: bool = True
    
    def is_compatible(self, factory_version: str) -> bool:
        """Check if plugin is compatible with forge version."""
        if not self.compatible_versions:
            return True
        
        try:
            factory_ver = version.parse(factory_version)
            for compat_range in self.compatible_versions:
                # Support simple range like ">=1.0.0,<2.0.0"
                if "," in compat_range:
                    min_ver, max_ver = compat_range.split(",")
                    min_ver = min_ver.strip().lstrip(">=")
                    max_ver = max_ver.strip().lstrip("<")
                    if version.parse(min_ver) <= factory_ver < version.parse(max_ver):
                        return True
                else:
                    # Single version constraint
                    if compat_range.startswith(">="):
                        if factory_ver >= version.parse(compat_range[2:]):
                            return True
                    elif compat_range.startswith(">"):
                        if factory_ver > version.parse(compat_range[1:]):
                            return True
                    elif compat_range.startswith("<="):
                        if factory_ver <= version.parse(compat_range[2:]):
                            return True
                    elif compat_range.startswith("<"):
                        if factory_ver < version.parse(compat_range[1:]):
                            return True
                    elif compat_range.startswith("=="):
                        if factory_ver == version.parse(compat_range[2:]):
                            return True
                    else:
                        # Exact version match
                        if factory_ver == version.parse(compat_range):
                            return True
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {e}")
            return False
        
        return False


class PluginInterface(ABC):
    """Base interface for all forge plugins."""
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @classmethod
    def validate_dependencies(cls) -> bool:
        """Validate that all plugin dependencies are installed."""
        metadata = cls.get_metadata()
        for dep in metadata.dependencies:
            try:
                pkg_resources.require(dep)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
                logger.error(f"Dependency not satisfied for plugin {metadata.name}: {e}")
                return False
        return True
    
    @classmethod
    def get_compatibility_info(cls) -> Dict[str, Any]:
        """Get compatibility information for the plugin."""
        metadata = cls.get_metadata()
        return {
            "name": metadata.name,
            "version": metadata.version,
            "compatible_versions": metadata.compatible_versions,
            "dependencies": metadata.dependencies
        }


class ModelPlugin(PluginInterface):
    """Base class for model plugins."""
    
    @classmethod
    @abstractmethod
    def get_model_class(cls) -> Type:
        """Return the model class."""
        pass
    
    @classmethod
    @abstractmethod
    def get_tokenizer_class(cls) -> Type:
        """Return the tokenizer class."""
        pass
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Return default model configuration."""
        return {}


class TrainerPlugin(PluginInterface):
    """Base class for trainer plugins."""
    
    @classmethod
    @abstractmethod
    def get_trainer_class(cls) -> Type:
        """Return the trainer class."""
        pass
    
    @classmethod
    def get_training_arguments(cls) -> Dict[str, Any]:
        """Return default training arguments."""
        return {}


class MetricPlugin(PluginInterface):
    """Base class for metric plugins."""
    
    @classmethod
    @abstractmethod
    def compute(cls, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        """Compute the metric."""
        pass
    
    @classmethod
    def get_metric_name(cls) -> str:
        """Return the metric name."""
        metadata = cls.get_metadata()
        return metadata.name


class PluginRegistry:
    """Central registry for managing forge plugins."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[PluginType, Dict[str, Type[PluginInterface]]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
        self._plugin_paths: List[Path] = []
        self._factory_version = self._get_factory_version()
        self._initialized = True
        
        # Default plugin directories
        self._default_plugin_dirs = [
            Path.home() / ".forge" / "plugins",
            Path("/usr/local/share/forge/plugins"),
            Path("/etc/forge/plugins")
        ]
        
        # Add current directory plugins
        self._default_plugin_dirs.append(Path.cwd() / "plugins")
    
    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """Get singleton instance of PluginRegistry."""
        if cls._instance is None:
            cls()
        return cls._instance
    
    def _get_factory_version(self) -> str:
        """Get current forge version."""
        try:
            return importlib.metadata.version("forge")
        except importlib.metadata.PackageNotFoundError:
            # Fallback for development
            try:
                from forge import __version__
                return __version__
            except ImportError:
                return "0.0.0"
    
    def register_plugin(self, plugin_class: Type[PluginInterface], force: bool = False) -> bool:
        """Register a plugin class.
        
        Args:
            plugin_class: The plugin class to register
            force: If True, overwrite existing plugin with same name
            
        Returns:
            bool: True if registration successful
        """
        try:
            metadata = plugin_class.get_metadata()
        except Exception as e:
            logger.error(f"Failed to get metadata for plugin class {plugin_class}: {e}")
            return False
        
        # Validate dependencies
        if not plugin_class.validate_dependencies():
            logger.error(f"Dependencies not satisfied for plugin {metadata.name}")
            return False
        
        # Check compatibility
        if not metadata.is_compatible(self._factory_version):
            logger.warning(
                f"Plugin {metadata.name} v{metadata.version} may not be compatible "
                f"with forge v{self._factory_version}"
            )
        
        plugin_type = metadata.plugin_type
        plugin_name = metadata.name
        
        if plugin_name in self._plugins[plugin_type] and not force:
            logger.warning(f"Plugin {plugin_name} already registered for type {plugin_type.value}")
            return False
        
        self._plugins[plugin_type][plugin_name] = plugin_class
        self._plugin_metadata[plugin_name] = metadata
        metadata.installed = True
        
        logger.info(f"Registered plugin: {plugin_name} ({plugin_type.value})")
        return True
    
    def unregister_plugin(self, plugin_name: str, plugin_type: Optional[PluginType] = None) -> bool:
        """Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            plugin_type: Optional plugin type. If None, unregister from all types.
            
        Returns:
            bool: True if unregistration successful
        """
        found = False
        
        if plugin_type:
            if plugin_name in self._plugins[plugin_type]:
                del self._plugins[plugin_type][plugin_name]
                found = True
        else:
            for ptype in PluginType:
                if plugin_name in self._plugins[ptype]:
                    del self._plugins[ptype][plugin_name]
                    found = True
        
        if found and plugin_name in self._plugin_metadata:
            del self._plugin_metadata[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
        
        return found
    
    def get_plugin(self, plugin_name: str, plugin_type: PluginType) -> Optional[Type[PluginInterface]]:
        """Get a plugin class by name and type."""
        return self._plugins[plugin_type].get(plugin_name)
    
    def get_model_plugin(self, model_name: str) -> Optional[Type[ModelPlugin]]:
        """Get a model plugin by name."""
        return self.get_plugin(model_name, PluginType.MODEL)
    
    def get_trainer_plugin(self, trainer_name: str) -> Optional[Type[TrainerPlugin]]:
        """Get a trainer plugin by name."""
        return self.get_plugin(trainer_name, PluginType.TRAINER)
    
    def get_metric_plugin(self, metric_name: str) -> Optional[Type[MetricPlugin]]:
        """Get a metric plugin by name."""
        return self.get_plugin(metric_name, PluginType.METRIC)
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None, 
                     enabled_only: bool = True) -> Dict[str, List[PluginMetadata]]:
        """List all registered plugins.
        
        Args:
            plugin_type: Optional filter by plugin type
            enabled_only: If True, only list enabled plugins
            
        Returns:
            Dict mapping plugin types to lists of plugin metadata
        """
        result = {}
        
        types_to_check = [plugin_type] if plugin_type else list(PluginType)
        
        for ptype in types_to_check:
            plugins = []
            for name, plugin_class in self._plugins[ptype].items():
                metadata = self._plugin_metadata.get(name)
                if metadata and (not enabled_only or metadata.enabled):
                    plugins.append(metadata)
            
            if plugins:
                result[ptype.value] = plugins
        
        return result
    
    def discover_plugins(self, paths: Optional[List[Union[str, Path]]] = None,
                         entry_points: bool = True) -> int:
        """Discover and register plugins from various sources.
        
        Args:
            paths: Additional paths to search for plugins
            entry_points: If True, discover plugins via entry points
            
        Returns:
            int: Number of plugins discovered and registered
        """
        discovered_count = 0
        
        # Discover via entry points
        if entry_points:
            discovered_count += self._discover_entry_points()
        
        # Discover from filesystem paths
        search_paths = list(self._default_plugin_dirs)
        if paths:
            search_paths.extend([Path(p) for p in paths])
        
        for path in search_paths:
            if path.exists() and path.is_dir():
                discovered_count += self._discover_from_path(path)
        
        logger.info(f"Discovered {discovered_count} plugins")
        return discovered_count
    
    def _discover_entry_points(self) -> int:
        """Discover plugins via setuptools entry points."""
        count = 0
        
        # Define entry point groups for each plugin type
        entry_point_groups = {
            PluginType.MODEL: "forge.models",
            PluginType.TRAINER: "forge.trainers",
            PluginType.METRIC: "forge.metrics",
            PluginType.DATA_PROCESSOR: "forge.data_processors",
            PluginType.CALLBACK: "forge.callbacks"
        }
        
        for plugin_type, group in entry_point_groups.items():
            try:
                for entry_point in importlib.metadata.entry_points().get(group, []):
                    try:
                        plugin_class = entry_point.load()
                        if self._is_valid_plugin_class(plugin_class, plugin_type):
                            if self.register_plugin(plugin_class):
                                count += 1
                                logger.debug(f"Discovered plugin via entry point: {entry_point.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load entry point {entry_point.name}: {e}")
            except Exception as e:
                logger.debug(f"No entry points found for group {group}: {e}")
        
        return count
    
    def _discover_from_path(self, path: Path) -> int:
        """Discover plugins from a filesystem path."""
        count = 0
        
        # Add path to Python path if not already there
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        
        # Look for Python files and packages
        for item in path.iterdir():
            if item.is_file() and item.suffix == ".py":
                count += self._discover_from_file(item)
            elif item.is_dir() and (item / "__init__.py").exists():
                count += self._discover_from_package(item)
        
        return count
    
    def _discover_from_file(self, file_path: Path) -> int:
        """Discover plugins from a single Python file."""
        count = 0
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"forge_plugin_{file_path.stem}",
                file_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_plugin_class(obj):
                        if self.register_plugin(obj):
                            count += 1
        except Exception as e:
            logger.warning(f"Failed to load plugin from {file_path}: {e}")
        
        return count
    
    def _discover_from_package(self, package_path: Path) -> int:
        """Discover plugins from a Python package."""
        count = 0
        
        try:
            # Import the package
            package_name = package_path.name
            spec = importlib.util.spec_from_file_location(
                package_name,
                package_path / "__init__.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = module
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_plugin_class(obj):
                        if self.register_plugin(obj):
                            count += 1
        except Exception as e:
            logger.warning(f"Failed to load plugin package from {package_path}: {e}")
        
        return count
    
    def _is_plugin_class(self, obj: Any) -> bool:
        """Check if an object is a valid plugin class."""
        return (
            inspect.isclass(obj) and
            issubclass(obj, PluginInterface) and
            obj is not PluginInterface and
            not inspect.isabstract(obj)
        )
    
    def _is_valid_plugin_class(self, obj: Any, expected_type: PluginType) -> bool:
        """Check if an object is a valid plugin class of expected type."""
        if not self._is_plugin_class(obj):
            return False
        
        try:
            metadata = obj.get_metadata()
            return metadata.plugin_type == expected_type
        except:
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self._plugin_metadata:
            self._plugin_metadata[plugin_name].enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self._plugin_metadata:
            self._plugin_metadata[plugin_name].enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
        return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        if plugin_name not in self._plugin_metadata:
            return None
        
        metadata = self._plugin_metadata[plugin_name]
        info = asdict(metadata)
        
        # Find which types this plugin is registered for
        registered_types = []
        for plugin_type, plugins in self._plugins.items():
            if plugin_name in plugins:
                registered_types.append(plugin_type.value)
        
        info["registered_types"] = registered_types
        return info
    
    def validate_all_plugins(self) -> Dict[str, List[str]]:
        """Validate all registered plugins.
        
        Returns:
            Dict with validation results for each plugin
        """
        results = {}
        
        for plugin_name, metadata in self._plugin_metadata.items():
            issues = []
            
            # Check compatibility
            if not metadata.is_compatible(self._factory_version):
                issues.append(f"Incompatible with forge v{self._factory_version}")
            
            # Check dependencies
            plugin_class = None
            for ptype in PluginType:
                if plugin_name in self._plugins[ptype]:
                    plugin_class = self._plugins[ptype][plugin_name]
                    break
            
            if plugin_class and not plugin_class.validate_dependencies():
                issues.append("Missing dependencies")
            
            if issues:
                results[plugin_name] = issues
        
        return results
    
    def export_registry(self, output_path: Union[str, Path]) -> None:
        """Export registry to a JSON file."""
        export_data = {
            "factory_version": self._factory_version,
            "plugins": {}
        }
        
        for plugin_name, metadata in self._plugin_metadata.items():
            export_data["plugins"][plugin_name] = asdict(metadata)
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported registry to {output_path}")
    
    def import_registry(self, input_path: Union[str, Path]) -> int:
        """Import plugins from a registry JSON file.
        
        Returns:
            int: Number of plugins imported
        """
        with open(input_path, "r") as f:
            import_data = json.load(f)
        
        imported_count = 0
        for plugin_name, plugin_data in import_data.get("plugins", {}).items():
            # Note: This only imports metadata, not the actual plugin code
            # The actual plugin code must be installed separately
            metadata = PluginMetadata(**plugin_data)
            self._plugin_metadata[plugin_name] = metadata
            imported_count += 1
        
        logger.info(f"Imported {imported_count} plugin metadata entries")
        return imported_count


# Global registry instance
registry = PluginRegistry.get_instance()


def get_model_registry() -> Dict[str, Type[ModelPlugin]]:
    """Get all registered model plugins."""
    return registry._plugins[PluginType.MODEL].copy()


def get_trainer_registry() -> Dict[str, Type[TrainerPlugin]]:
    """Get all registered trainer plugins."""
    return registry._plugins[PluginType.TRAINER].copy()


def get_metric_registry() -> Dict[str, Type[MetricPlugin]]:
    """Get all registered metric plugins."""
    return registry._plugins[PluginType.METRIC].copy()


def load_plugin(plugin_name: str, plugin_type: PluginType) -> Optional[Type[PluginInterface]]:
    """Load a plugin by name and type."""
    return registry.get_plugin(plugin_name, plugin_type)


def discover_plugins(paths: Optional[List[Union[str, Path]]] = None,
                     entry_points: bool = True) -> int:
    """Discover and register plugins."""
    return registry.discover_plugins(paths, entry_points)


# Decorator for easy plugin registration
def register_plugin(plugin_class: Type[PluginInterface]) -> Type[PluginInterface]:
    """Decorator to register a plugin class."""
    registry.register_plugin(plugin_class)
    return plugin_class


# Example plugin implementations (for testing/documentation)
@register_plugin
class ExampleModelPlugin(ModelPlugin):
    """Example model plugin for demonstration."""
    
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="example_model",
            version="1.0.0",
            plugin_type=PluginType.MODEL,
            description="Example model plugin for testing",
            author="forge Team",
            homepage="https://github.com/forge/forge",
            repository="https://github.com/forge/forge",
            license="Apache-2.0",
            compatible_versions=[">=1.0.0"],
            tags=["example", "test"]
        )
    
    @classmethod
    def get_model_class(cls) -> Type:
        from transformers import AutoModel
        return AutoModel
    
    @classmethod
    def get_tokenizer_class(cls) -> Type:
        from transformers import AutoTokenizer
        return AutoTokenizer


# Initialize plugin discovery on module import
def _initialize_plugins():
    """Initialize plugin discovery when module is imported."""
    try:
        discover_plugins()
    except Exception as e:
        logger.warning(f"Plugin discovery failed during initialization: {e}")


# Auto-discover plugins when module is imported
_initialize_plugins()