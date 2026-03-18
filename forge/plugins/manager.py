"""
forge Plugin System - Manager Module
===========================================

Provides plugin discovery, loading, version management, and marketplace integration.
"""

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import hashlib
import tempfile
import shutil
import subprocess
from urllib.parse import urlparse

import requests
from packaging import version

from ..utils import logging

logger = logging.get_logger(__name__)


class PluginType(Enum):
    """Types of plugins supported by forge."""
    MODEL = "model"
    TRAINER = "trainer"
    METRIC = "metric"
    DATASET = "dataset"
    STRATEGY = "strategy"
    EVALUATOR = "evaluator"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    plugin_type: PluginType
    author: str = ""
    description: str = ""
    homepage: str = ""
    license: str = ""
    dependencies: List[str] = field(default_factory=list)
    supported_versions: List[str] = field(default_factory=list)  # forge versions
    entry_point: str = ""
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.plugin_type.value,
            "author": self.author,
            "description": self.description,
            "homepage": self.homepage,
            "license": self.license,
            "dependencies": self.dependencies,
            "supported_versions": self.supported_versions,
            "entry_point": self.entry_point,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            plugin_type=PluginType(data["type"]),
            author=data.get("author", ""),
            description=data.get("description", ""),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
            dependencies=data.get("dependencies", []),
            supported_versions=data.get("supported_versions", []),
            entry_point=data.get("entry_point", ""),
            checksum=data.get("checksum", ""),
        )


class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def check_compatibility(self, forge_version: str) -> bool:
        """Check if plugin is compatible with given forge version."""
        metadata = self.get_metadata()
        if not metadata.supported_versions:
            return True
        
        for supported in metadata.supported_versions:
            if version.parse(forge_version) >= version.parse(supported):
                return True
        return False
    
    def validate(self) -> bool:
        """Validate plugin configuration and dependencies."""
        return True


class ModelPlugin(PluginInterface):
    """Base class for model plugins."""
    
    @abstractmethod
    def get_model_class(self) -> Type:
        """Return the model class provided by this plugin."""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        pass
    
    def get_tokenizer_class(self) -> Optional[Type]:
        """Return tokenizer class if provided."""
        return None


class TrainerPlugin(PluginInterface):
    """Base class for trainer plugins."""
    
    @abstractmethod
    def get_trainer_class(self) -> Type:
        """Return the trainer class provided by this plugin."""
        pass
    
    @abstractmethod
    def get_training_arguments(self) -> Dict[str, Any]:
        """Return default training arguments."""
        pass


class MetricPlugin(PluginInterface):
    """Base class for metric plugins."""
    
    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        """Compute metric values."""
        pass
    
    def get_metric_name(self) -> str:
        """Return metric name."""
        metadata = self.get_metadata()
        return metadata.name


class DatasetPlugin(PluginInterface):
    """Base class for dataset plugins."""
    
    @abstractmethod
    def load_dataset(self, **kwargs) -> Any:
        """Load and return dataset."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        pass


class StrategyPlugin(PluginInterface):
    """Base class for training strategy plugins."""
    
    @abstractmethod
    def apply_strategy(self, model: Any, **kwargs) -> Any:
        """Apply training strategy to model."""
        pass


class EvaluatorPlugin(PluginInterface):
    """Base class for evaluator plugins."""
    
    @abstractmethod
    def evaluate(self, model: Any, dataset: Any, **kwargs) -> Dict[str, Any]:
        """Run evaluation and return results."""
        pass


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    metadata: PluginMetadata
    plugin_class: Type[PluginInterface]
    module_path: str
    instance: Optional[PluginInterface] = None
    is_active: bool = True
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def plugin_type(self) -> PluginType:
        return self.metadata.plugin_type


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    _instance = None
    _forge_version = "0.1.0"  # Should be imported from main package
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_dirs: List[Path] = []
        self._marketplace_url = "https://forge.io/plugins/marketplace.json"
        self._cache_dir = Path.home() / ".cache" / "forge" / "plugins"
        self._config_file = Path.home() / ".forge" / "plugins.json"
        
        # Initialize plugin directories
        self._setup_plugin_dirs()
        self._load_config()
        
        # Try to get forge version
        try:
            from .. import __version__
            self._forge_version = __version__
        except ImportError:
            pass
        
        self._initialized = True
    
    def _setup_plugin_dirs(self):
        """Setup plugin directories."""
        # User plugins directory
        user_plugin_dir = Path.home() / ".forge" / "plugins"
        user_plugin_dir.mkdir(parents=True, exist_ok=True)
        self._plugin_dirs.append(user_plugin_dir)
        
        # System plugins directory (if exists)
        system_plugin_dir = Path("/usr/local/share/forge/plugins")
        if system_plugin_dir.exists():
            self._plugin_dirs.append(system_plugin_dir)
        
        # Current directory plugins
        cwd_plugin_dir = Path.cwd() / "plugins"
        if cwd_plugin_dir.exists():
            self._plugin_dirs.append(cwd_plugin_dir)
    
    def _load_config(self):
        """Load plugin configuration."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load plugin config: {e}")
                self._config = {}
        else:
            self._config = {"active_plugins": [], "installed_plugins": []}
    
    def _save_config(self):
        """Save plugin configuration."""
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w") as f:
            json.dump(self._config, f, indent=2)
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins from all sources."""
        discovered = []
        
        # Discover from plugin directories
        for plugin_dir in self._plugin_dirs:
            discovered.extend(self._discover_from_directory(plugin_dir))
        
        # Discover from entry points
        discovered.extend(self._discover_from_entry_points())
        
        return discovered
    
    def _discover_from_directory(self, directory: Path) -> List[PluginMetadata]:
        """Discover plugins from a directory."""
        plugins = []
        
        if not directory.exists():
            return plugins
        
        for item in directory.iterdir():
            if item.is_file() and item.suffix == ".py":
                # Single file plugin
                try:
                    metadata = self._load_plugin_metadata_from_file(item)
                    if metadata:
                        plugins.append(metadata)
                except Exception as e:
                    logger.debug(f"Failed to load plugin from {item}: {e}")
            
            elif item.is_dir() and (item / "__init__.py").exists():
                # Package plugin
                try:
                    metadata = self._load_plugin_metadata_from_package(item)
                    if metadata:
                        plugins.append(metadata)
                except Exception as e:
                    logger.debug(f"Failed to load plugin from {item}: {e}")
        
        return plugins
    
    def _discover_from_entry_points(self) -> List[PluginMetadata]:
        """Discover plugins from Python entry points."""
        plugins = []
        
        try:
            # Python 3.10+ has entry_points with group parameter
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group="forge.plugins")
            else:
                eps = importlib.metadata.entry_points().get("forge.plugins", [])
            
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if hasattr(plugin_class, "get_metadata"):
                        instance = plugin_class()
                        metadata = instance.get_metadata()
                        plugins.append(metadata)
                except Exception as e:
                    logger.debug(f"Failed to load entry point {ep.name}: {e}")
        
        except Exception as e:
            logger.debug(f"Failed to discover entry points: {e}")
        
        return plugins
    
    def _load_plugin_metadata_from_file(self, file_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a Python file."""
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            return None
        
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
        
        # Look for plugin class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, PluginInterface) and 
                obj is not PluginInterface and
                not inspect.isabstract(obj)):
                try:
                    instance = obj()
                    return instance.get_metadata()
                except Exception:
                    continue
        
        return None
    
    def _load_plugin_metadata_from_package(self, package_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a package directory."""
        package_name = package_path.name
        
        # Add package to path temporarily
        sys.path.insert(0, str(package_path.parent))
        try:
            module = importlib.import_module(package_name)
            
            # Look for plugin class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj is not PluginInterface and
                    not inspect.isabstract(obj)):
                    try:
                        instance = obj()
                        return instance.get_metadata()
                    except Exception:
                        continue
        finally:
            sys.path.pop(0)
        
        return None
    
    def load_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Load a specific plugin by name."""
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]
        
        # Search in discovered plugins
        discovered = self.discover_plugins()
        for metadata in discovered:
            if metadata.name == plugin_name:
                plugin_info = self._load_plugin_from_metadata(metadata)
                if plugin_info:
                    self._plugins[plugin_name] = plugin_info
                    return plugin_info
        
        logger.warning(f"Plugin '{plugin_name}' not found")
        return None
    
    def _load_plugin_from_metadata(self, metadata: PluginMetadata) -> Optional[PluginInfo]:
        """Load plugin from metadata."""
        try:
            # Check compatibility
            if not self._check_version_compatibility(metadata):
                logger.warning(
                    f"Plugin '{metadata.name}' version {metadata.version} "
                    f"is not compatible with forge {self._forge_version}"
                )
                return None
            
            # Load the plugin class
            if metadata.entry_point:
                # Load from entry point
                plugin_class = self._load_from_entry_point(metadata.entry_point)
            else:
                # Load from discovery
                plugin_class = self._load_plugin_class(metadata)
            
            if not plugin_class:
                return None
            
            # Create instance
            instance = plugin_class()
            
            # Validate plugin
            if not instance.validate():
                logger.warning(f"Plugin '{metadata.name}' validation failed")
                return None
            
            return PluginInfo(
                metadata=metadata,
                plugin_class=plugin_class,
                module_path=metadata.entry_point or "",
                instance=instance,
                is_active=True
            )
        
        except Exception as e:
            logger.error(f"Failed to load plugin '{metadata.name}': {e}")
            return None
    
    def _check_version_compatibility(self, metadata: PluginMetadata) -> bool:
        """Check if plugin is compatible with current forge version."""
        if not metadata.supported_versions:
            return True
        
        for supported_version in metadata.supported_versions:
            try:
                if version.parse(self._forge_version) >= version.parse(supported_version):
                    return True
            except Exception:
                continue
        
        return False
    
    def _load_from_entry_point(self, entry_point_str: str) -> Optional[Type[PluginInterface]]:
        """Load plugin class from entry point string."""
        try:
            # Parse entry point: "module:attribute"
            module_path, attr_name = entry_point_str.split(":", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except Exception as e:
            logger.error(f"Failed to load entry point '{entry_point_str}': {e}")
            return None
    
    def _load_plugin_class(self, metadata: PluginMetadata) -> Optional[Type[PluginInterface]]:
        """Load plugin class by searching in plugin directories."""
        for plugin_dir in self._plugin_dirs:
            # Try as single file
            plugin_file = plugin_dir / f"{metadata.name}.py"
            if plugin_file.exists():
                spec = importlib.util.spec_from_file_location(metadata.name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, PluginInterface) and 
                            obj is not PluginInterface and
                            not inspect.isabstract(obj)):
                            return obj
            
            # Try as package
            plugin_package = plugin_dir / metadata.name
            if plugin_package.is_dir() and (plugin_package / "__init__.py").exists():
                sys.path.insert(0, str(plugin_dir))
                try:
                    module = importlib.import_module(metadata.name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, PluginInterface) and 
                            obj is not PluginInterface and
                            not inspect.isabstract(obj)):
                            return obj
                finally:
                    sys.path.pop(0)
        
        return None
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a plugin."""
        if plugin_name not in self._plugins:
            if not self.load_plugin(plugin_name):
                return False
        
        plugin_info = self._plugins[plugin_name]
        plugin_info.is_active = True
        
        # Update config
        if plugin_name not in self._config["active_plugins"]:
            self._config["active_plugins"].append(plugin_name)
            self._save_config()
        
        return True
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].is_active = False
        
        # Update config
        if plugin_name in self._config["active_plugins"]:
            self._config["active_plugins"].remove(plugin_name)
            self._save_config()
        
        return True
    
    def get_active_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginInfo]:
        """Get all active plugins, optionally filtered by type."""
        active = []
        for plugin_info in self._plugins.values():
            if plugin_info.is_active:
                if plugin_type is None or plugin_info.plugin_type == plugin_type:
                    active.append(plugin_info)
        return active
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return list(self._plugins.values())
    
    def install_plugin(self, source: str, plugin_name: Optional[str] = None) -> bool:
        """Install a plugin from various sources."""
        try:
            if source.startswith(("http://", "https://")):
                # URL source
                return self._install_from_url(source, plugin_name)
            elif source.endswith(".whl") or source.endswith(".tar.gz"):
                # Local file
                return self._install_from_file(source, plugin_name)
            elif "/" in source and not source.startswith(("/", ".")):
                # GitHub repository (owner/repo)
                return self._install_from_github(source, plugin_name)
            else:
                # Marketplace name
                return self._install_from_marketplace(source, plugin_name)
        except Exception as e:
            logger.error(f"Failed to install plugin from {source}: {e}")
            return False
    
    def _install_from_url(self, url: str, plugin_name: Optional[str] = None) -> bool:
        """Install plugin from URL."""
        try:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Determine filename
            if plugin_name:
                filename = f"{plugin_name}.py"
            else:
                filename = url.split("/")[-1]
                if not filename.endswith(".py"):
                    filename += ".py"
            
            # Save to plugin directory
            plugin_dir = self._plugin_dirs[0]  # User plugin directory
            plugin_path = plugin_dir / filename
            
            with open(plugin_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Calculate checksum
            checksum = self._calculate_checksum(plugin_path)
            
            # Try to load and validate
            metadata = self._load_plugin_metadata_from_file(plugin_path)
            if metadata:
                # Update config
                self._config["installed_plugins"].append({
                    "name": metadata.name,
                    "version": metadata.version,
                    "source": url,
                    "checksum": checksum,
                    "installed_at": str(Path(plugin_path).stat().st_mtime)
                })
                self._save_config()
                
                logger.info(f"Successfully installed plugin '{metadata.name}' from {url}")
                return True
            else:
                # Remove invalid plugin
                plugin_path.unlink()
                logger.error(f"Invalid plugin file from {url}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to install from URL {url}: {e}")
            return False
    
    def _install_from_file(self, file_path: str, plugin_name: Optional[str] = None) -> bool:
        """Install plugin from local file."""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Determine destination
            plugin_dir = self._plugin_dirs[0]  # User plugin directory
            if plugin_name:
                dest_path = plugin_dir / f"{plugin_name}.py"
            else:
                dest_path = plugin_dir / source_path.name
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(dest_path)
            
            # Try to load and validate
            metadata = self._load_plugin_metadata_from_file(dest_path)
            if metadata:
                # Update config
                self._config["installed_plugins"].append({
                    "name": metadata.name,
                    "version": metadata.version,
                    "source": str(source_path.absolute()),
                    "checksum": checksum,
                    "installed_at": str(dest_path.stat().st_mtime)
                })
                self._save_config()
                
                logger.info(f"Successfully installed plugin '{metadata.name}' from {file_path}")
                return True
            else:
                # Remove invalid plugin
                dest_path.unlink()
                logger.error(f"Invalid plugin file: {file_path}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to install from file {file_path}: {e}")
            return False
    
    def _install_from_github(self, repo: str, plugin_name: Optional[str] = None) -> bool:
        """Install plugin from GitHub repository."""
        try:
            # Parse repository
            if repo.count("/") == 1:
                owner, repo_name = repo.split("/")
                branch = "main"
            elif repo.count("/") == 2:
                owner, repo_name, branch = repo.split("/")
            else:
                logger.error(f"Invalid GitHub repository format: {repo}")
                return False
            
            # Construct raw URL
            url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{branch}/plugin.py"
            
            # Download and install
            return self._install_from_url(url, plugin_name or repo_name)
        
        except Exception as e:
            logger.error(f"Failed to install from GitHub {repo}: {e}")
            return False
    
    def _install_from_marketplace(self, plugin_name: str, target_name: Optional[str] = None) -> bool:
        """Install plugin from marketplace."""
        try:
            # Fetch marketplace data
            marketplace_data = self._fetch_marketplace_data()
            if not marketplace_data:
                logger.error("Failed to fetch marketplace data")
                return False
            
            # Find plugin
            plugin_data = None
            for plugin in marketplace_data.get("plugins", []):
                if plugin["name"] == plugin_name:
                    plugin_data = plugin
                    break
            
            if not plugin_data:
                logger.error(f"Plugin '{plugin_name}' not found in marketplace")
                return False
            
            # Install from source
            return self._install_from_url(plugin_data["download_url"], target_name)
        
        except Exception as e:
            logger.error(f"Failed to install from marketplace {plugin_name}: {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        try:
            # Find plugin file
            plugin_file = None
            for plugin_dir in self._plugin_dirs:
                potential_file = plugin_dir / f"{plugin_name}.py"
                if potential_file.exists():
                    plugin_file = potential_file
                    break
            
            if not plugin_file:
                logger.error(f"Plugin file for '{plugin_name}' not found")
                return False
            
            # Remove file
            plugin_file.unlink()
            
            # Remove from config
            self._config["installed_plugins"] = [
                p for p in self._config["installed_plugins"]
                if p["name"] != plugin_name
            ]
            self._config["active_plugins"] = [
                p for p in self._config["active_plugins"]
                if p != plugin_name
            ]
            self._save_config()
            
            # Remove from loaded plugins
            if plugin_name in self._plugins:
                del self._plugins[plugin_name]
            
            logger.info(f"Successfully uninstalled plugin '{plugin_name}'")
            return True
        
        except Exception as e:
            logger.error(f"Failed to uninstall plugin '{plugin_name}': {e}")
            return False
    
    def update_plugin(self, plugin_name: str) -> bool:
        """Update a plugin to latest version."""
        try:
            # Find plugin in installed plugins
            plugin_data = None
            for p in self._config["installed_plugins"]:
                if p["name"] == plugin_name:
                    plugin_data = p
                    break
            
            if not plugin_data:
                logger.error(f"Plugin '{plugin_name}' not installed")
                return False
            
            # Reinstall from source
            source = plugin_data["source"]
            if source.startswith(("http://", "https://")):
                return self._install_from_url(source, plugin_name)
            else:
                logger.error(f"Cannot update plugin from source: {source}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to update plugin '{plugin_name}': {e}")
            return False
    
    def _fetch_marketplace_data(self) -> Optional[Dict[str, Any]]:
        """Fetch marketplace data from remote."""
        try:
            cache_file = self._cache_dir / "marketplace.json"
            
            # Check cache (valid for 1 hour)
            if cache_file.exists():
                cache_age = Path(cache_file).stat().st_mtime
                if (Path(cache_file).stat().st_mtime - cache_age) < 3600:
                    with open(cache_file, "r") as f:
                        return json.load(f)
            
            # Fetch fresh data
            response = requests.get(self._marketplace_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the data
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
            
            return data
        
        except Exception as e:
            logger.warning(f"Failed to fetch marketplace data: {e}")
            return None
    
    def search_marketplace(self, query: str = "", plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """Search plugins in marketplace."""
        marketplace_data = self._fetch_marketplace_data()
        if not marketplace_data:
            return []
        
        results = []
        for plugin in marketplace_data.get("plugins", []):
            # Filter by type
            if plugin_type and plugin.get("type") != plugin_type.value:
                continue
            
            # Filter by query
            if query:
                query_lower = query.lower()
                searchable = f"{plugin['name']} {plugin.get('description', '')} {plugin.get('author', '')}".lower()
                if query_lower not in searchable:
                    continue
            
            results.append(plugin)
        
        return results
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        # Check local plugins
        plugin_info = self.get_plugin(plugin_name)
        if plugin_info:
            return plugin_info.metadata.to_dict()
        
        # Check marketplace
        marketplace_data = self._fetch_marketplace_data()
        if marketplace_data:
            for plugin in marketplace_data.get("plugins", []):
                if plugin["name"] == plugin_name:
                    return plugin
        
        return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def verify_plugin(self, plugin_name: str) -> bool:
        """Verify plugin integrity."""
        try:
            # Find plugin in installed plugins
            plugin_data = None
            for p in self._config["installed_plugins"]:
                if p["name"] == plugin_name:
                    plugin_data = p
                    break
            
            if not plugin_data:
                logger.error(f"Plugin '{plugin_name}' not installed")
                return False
            
            # Find plugin file
            plugin_file = None
            for plugin_dir in self._plugin_dirs:
                potential_file = plugin_dir / f"{plugin_name}.py"
                if potential_file.exists():
                    plugin_file = potential_file
                    break
            
            if not plugin_file:
                logger.error(f"Plugin file for '{plugin_name}' not found")
                return False
            
            # Verify checksum
            current_checksum = self._calculate_checksum(plugin_file)
            if current_checksum != plugin_data.get("checksum"):
                logger.error(f"Checksum mismatch for plugin '{plugin_name}'")
                return False
            
            # Try to load and validate
            metadata = self._load_plugin_metadata_from_file(plugin_file)
            if not metadata:
                logger.error(f"Failed to load plugin '{plugin_name}'")
                return False
            
            logger.info(f"Plugin '{plugin_name}' verification successful")
            return True
        
        except Exception as e:
            logger.error(f"Failed to verify plugin '{plugin_name}': {e}")
            return False
    
    def export_plugin_list(self, output_file: str) -> bool:
        """Export list of installed plugins to JSON file."""
        try:
            data = {
                "forge_version": self._forge_version,
                "plugins": self._config["installed_plugins"],
                "active_plugins": self._config["active_plugins"]
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Plugin list exported to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export plugin list: {e}")
            return False
    
    def import_plugin_list(self, input_file: str) -> bool:
        """Import and install plugins from JSON file."""
        try:
            with open(input_file, "r") as f:
                data = json.load(f)
            
            success_count = 0
            for plugin_data in data.get("plugins", []):
                plugin_name = plugin_data["name"]
                source = plugin_data["source"]
                
                if self.install_plugin(source, plugin_name):
                    success_count += 1
            
            logger.info(f"Successfully imported {success_count} plugins from {input_file}")
            return success_count > 0
        
        except Exception as e:
            logger.error(f"Failed to import plugin list: {e}")
            return False


# Global plugin manager instance
plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return plugin_manager


def register_plugin(plugin_class: Type[PluginInterface]) -> Type[PluginInterface]:
    """Decorator to register a plugin class."""
    def wrapper(cls):
        # Validate that it's a proper plugin class
        if not issubclass(cls, PluginInterface):
            raise TypeError(f"Class {cls.__name__} must inherit from PluginInterface")
        
        # Add to entry points (for development)
        if not hasattr(sys.modules[__name__], "_registered_plugins"):
            sys.modules[__name__]._registered_plugins = []
        sys.modules[__name__]._registered_plugins.append(cls)
        
        return cls
    return wrapper(plugin_class)


# CLI integration functions
def cli_install_plugin(source: str, name: Optional[str] = None) -> int:
    """CLI function to install a plugin."""
    if plugin_manager.install_plugin(source, name):
        return 0
    return 1


def cli_uninstall_plugin(name: str) -> int:
    """CLI function to uninstall a plugin."""
    if plugin_manager.uninstall_plugin(name):
        return 0
    return 1


def cli_list_plugins(plugin_type: Optional[str] = None) -> int:
    """CLI function to list plugins."""
    type_filter = PluginType(plugin_type) if plugin_type else None
    plugins = plugin_manager.get_active_plugins(type_filter)
    
    if not plugins:
        print("No active plugins found.")
        return 0
    
    print(f"Active Plugins ({len(plugins)}):")
    print("-" * 60)
    for plugin in plugins:
        print(f"  {plugin.name} ({plugin.metadata.version})")
        print(f"    Type: {plugin.plugin_type.value}")
        print(f"    Author: {plugin.metadata.author}")
        print(f"    Description: {plugin.metadata.description[:50]}...")
        print()
    
    return 0


def cli_search_marketplace(query: str = "", plugin_type: Optional[str] = None) -> int:
    """CLI function to search marketplace."""
    type_filter = PluginType(plugin_type) if plugin_type else None
    results = plugin_manager.search_marketplace(query, type_filter)
    
    if not results:
        print("No plugins found in marketplace.")
        return 0
    
    print(f"Marketplace Results ({len(results)}):")
    print("-" * 60)
    for plugin in results:
        print(f"  {plugin['name']} ({plugin.get('version', 'N/A')})")
        print(f"    Type: {plugin.get('type', 'N/A')}")
        print(f"    Author: {plugin.get('author', 'N/A')}")
        print(f"    Downloads: {plugin.get('downloads', 0)}")
        print(f"    Description: {plugin.get('description', '')[:50]}...")
        print()
    
    return 0


def cli_update_plugin(name: str) -> int:
    """CLI function to update a plugin."""
    if plugin_manager.update_plugin(name):
        return 0
    return 1


def cli_verify_plugin(name: str) -> int:
    """CLI function to verify a plugin."""
    if plugin_manager.verify_plugin(name):
        return 0
    return 1


def cli_export_plugins(output_file: str) -> int:
    """CLI function to export plugin list."""
    if plugin_manager.export_plugin_list(output_file):
        return 0
    return 1


def cli_import_plugins(input_file: str) -> int:
    """CLI function to import plugin list."""
    if plugin_manager.import_plugin_list(input_file):
        return 0
    return 1


# Integration with existing forge components
def get_model_plugins() -> List[PluginInfo]:
    """Get all active model plugins."""
    return plugin_manager.get_active_plugins(PluginType.MODEL)


def get_trainer_plugins() -> List[PluginInfo]:
    """Get all active trainer plugins."""
    return plugin_manager.get_active_plugins(PluginType.TRAINER)


def get_metric_plugins() -> List[PluginInfo]:
    """Get all active metric plugins."""
    return plugin_manager.get_active_plugins(PluginType.METRIC)


def get_dataset_plugins() -> List[PluginInfo]:
    """Get all active dataset plugins."""
    return plugin_manager.get_active_plugins(PluginType.DATASET)


def get_strategy_plugins() -> List[PluginInfo]:
    """Get all active strategy plugins."""
    return plugin_manager.get_active_plugins(PluginType.STRATEGY)


def get_evaluator_plugins() -> List[PluginInfo]:
    """Get all active evaluator plugins."""
    return plugin_manager.get_active_plugins(PluginType.EVALUATOR)


# Example plugin implementation for testing
class ExampleModelPlugin(ModelPlugin):
    """Example model plugin for testing."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_model",
            version="1.0.0",
            plugin_type=PluginType.MODEL,
            author="forge Team",
            description="Example model plugin for testing",
            supported_versions=["0.1.0"],
        )
    
    def get_model_class(self) -> Type:
        # Return a placeholder class
        class ExampleModel:
            pass
        return ExampleModel
    
    def get_model_config(self) -> Dict[str, Any]:
        return {"model_type": "example", "hidden_size": 768}