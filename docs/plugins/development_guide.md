# Plugin Development Guide for forge

## Table of Contents
1. [Introduction](#introduction)
2. [Plugin Architecture](#plugin-architecture)
3. [Creating Your First Plugin](#creating-your-first-plugin)
4. [Model Plugins](#model-plugins)
5. [Trainer Plugins](#trainer-plugins)
6. [Metric Plugins](#metric-plugins)
7. [Plugin Discovery & Loading](#plugin-discovery--loading)
8. [Version Compatibility](#version-compatibility)
9. [CLI Plugin Manager](#cli-plugin-manager)
10. [Plugin Marketplace](#plugin-marketplace)
11. [Testing & Quality Assurance](#testing--quality-assurance)
12. [Best Practices](#best-practices)
13. [Examples](#examples)

## Introduction

forge's plugin system enables community-driven extensibility without modifying core code. Plugins can introduce new models, training strategies, evaluation metrics, and utilities. This guide covers the complete development lifecycle from creation to distribution.

## Plugin Architecture

### Core Components
```
forge/
├── plugins/
│   ├── __init__.py
│   ├── base.py           # Base plugin interfaces
│   ├── registry.py       # Plugin registry
│   ├── loader.py         # Dynamic loader
│   ├── compatibility.py  # Version checking
│   └── marketplace.py    # Marketplace client
├── cli/
│   └── plugin.py         # CLI plugin commands
└── plugins/              # Plugin installation directory
```

### Plugin Types
- **Model Plugins**: New model architectures and configurations
- **Trainer Plugins**: Custom training loops and strategies
- **Metric Plugins**: Evaluation metrics and benchmarks
- **Utility Plugins**: Data processors, callbacks, and tools

## Creating Your First Plugin

### Basic Plugin Structure
```python
# my_plugin/__init__.py
from forge.plugins.base import PluginBase

class MyPlugin(PluginBase):
    """Example plugin demonstrating basic structure."""
    
    PLUGIN_NAME = "my_awesome_plugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_AUTHOR = "Your Name"
    PLUGIN_DESCRIPTION = "An awesome plugin for forge"
    PLUGIN_URL = "https://github.com/username/my-plugin"
    COMPATIBLE_VERSIONS = ["0.2.0", "0.3.0"]
    
    def __init__(self):
        super().__init__()
        self.registered = False
        
    def register(self, registry):
        """Register plugin components with the registry."""
        from .models import MyCustomModel
        from .trainers import MyCustomTrainer
        from .metrics import MyCustomMetric
        
        registry.register_model(MyCustomModel)
        registry.register_trainer(MyCustomTrainer)
        registry.register_metric(MyCustomMetric)
        
        self.registered = True
        return True
        
    def get_metadata(self):
        """Return plugin metadata."""
        return {
            "name": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "author": self.PLUGIN_AUTHOR,
            "description": self.PLUGIN_DESCRIPTION,
            "url": self.PLUGIN_URL,
            "compatible_versions": self.COMPATIBLE_VERSIONS,
            "type": "full",  # model, trainer, metric, or full
            "entry_points": {
                "models": ["MyCustomModel"],
                "trainers": ["MyCustomTrainer"],
                "metrics": ["MyCustomMetric"]
            }
        }
```

### Plugin Entry Point (setup.py)
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="forge-plugin-myplugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "forge>=0.2.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "forge.plugins": [
            "my_plugin = my_plugin:MyPlugin",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="An awesome plugin for forge",
    url="https://github.com/username/my-plugin",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

## Model Plugins

### Creating a Custom Model Plugin
```python
# my_plugin/models/custom_model.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from forge.models.base import BaseModel
from forge.plugins.base import ModelPlugin

class MyCustomModelPlugin(ModelPlugin):
    """Plugin for custom model architecture."""
    
    PLUGIN_NAME = "custom_model"
    PLUGIN_VERSION = "1.0.0"
    
    @staticmethod
    def get_model_class():
        """Return the model class."""
        return MyCustomModel
    
    @staticmethod
    def get_model_config(config_name: str = "default"):
        """Return model configuration."""
        configs = {
            "default": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "vocab_size": 32000,
                "max_seq_len": 2048,
            },
            "large": {
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "vocab_size": 32000,
                "max_seq_len": 4096,
            }
        }
        return configs.get(config_name, configs["default"])
    
    @staticmethod
    def supports_architecture(arch_name: str) -> bool:
        """Check if this plugin supports the given architecture."""
        return arch_name in ["custom", "my_custom", "my-model"]
    
    @staticmethod
    def convert_from_hf(hf_model, config: Dict[str, Any]):
        """Convert HuggingFace model to custom format."""
        # Implement conversion logic
        pass

class MyCustomModel(BaseModel):
    """Custom model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        # Build model layers
        self.embedding = nn.Embedding(
            config["vocab_size"],
            config["hidden_size"]
        )
        
        self.layers = nn.ModuleList([
            MyCustomLayer(config) for _ in range(config["num_layers"])
        ])
        
        self.output = nn.Linear(
            config["hidden_size"],
            config["vocab_size"],
            bias=False
        )
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass."""
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, **kwargs)
            
        return self.output(x)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pretrained model."""
        # Implement loading logic
        pass
    
    def save_pretrained(self, save_path: str):
        """Save model to disk."""
        # Implement saving logic
        pass
```

## Trainer Plugins

### Custom Training Strategy Plugin
```python
# my_plugin/trainers/custom_trainer.py
import torch
from typing import Dict, Any, Optional, List
from forge.trainers.base import BaseTrainer
from forge.plugins.base import TrainerPlugin

class CustomTrainerPlugin(TrainerPlugin):
    """Plugin for custom training strategy."""
    
    PLUGIN_NAME = "custom_trainer"
    PLUGIN_VERSION = "1.0.0"
    
    @staticmethod
    def get_trainer_class():
        """Return the trainer class."""
        return CustomTrainer
    
    @staticmethod
    def get_trainer_config(config_name: str = "default"):
        """Return trainer configuration."""
        return {
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "custom_param": 42,  # Custom parameter
        }
    
    @staticmethod
    def supports_strategy(strategy_name: str) -> bool:
        """Check if this trainer supports the given strategy."""
        return strategy_name in ["custom", "my_strategy"]

class CustomTrainer(BaseTrainer):
    """Custom trainer with advanced features."""
    
    def __init__(self, model, config: Dict[str, Any], **kwargs):
        super().__init__(model, config, **kwargs)
        self.custom_param = config.get("custom_param", 42)
        
    def create_optimizer(self):
        """Create custom optimizer."""
        # Custom optimizer implementation
        return super().create_optimizer()
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Custom training step."""
        # Implement custom training logic
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Custom loss modification
        if self.custom_param > 0:
            loss = loss * self.custom_param
            
        return loss
    
    def compute_loss(self, outputs, labels):
        """Custom loss computation."""
        # Implement custom loss function
        return super().compute_loss(outputs, labels)
    
    def get_train_dataloader(self):
        """Custom data loading."""
        # Implement custom data loading logic
        return super().get_train_dataloader()
```

## Metric Plugins

### Custom Evaluation Metric Plugin
```python
# my_plugin/metrics/custom_metric.py
import numpy as np
from typing import List, Dict, Any, Optional
from forge.metrics.base import BaseMetric
from forge.plugins.base import MetricPlugin

class CustomMetricPlugin(MetricPlugin):
    """Plugin for custom evaluation metric."""
    
    PLUGIN_NAME = "custom_metric"
    PLUGIN_VERSION = "1.0.0"
    
    @staticmethod
    def get_metric_class():
        """Return the metric class."""
        return CustomMetric
    
    @staticmethod
    def get_metric_config(config_name: str = "default"):
        """Return metric configuration."""
        return {
            "threshold": 0.5,
            "normalize": True,
            "custom_weight": 1.0,
        }
    
    @staticmethod
    def supports_metric(metric_name: str) -> bool:
        """Check if this plugin supports the given metric."""
        return metric_name in ["custom", "my_metric", "custom_score"]

class CustomMetric(BaseMetric):
    """Custom evaluation metric."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.5)
        self.normalize = config.get("normalize", True)
        self.custom_weight = config.get("custom_weight", 1.0)
        
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute the metric score."""
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Custom scoring logic
            score = self._compute_single(pred, ref)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        if self.normalize:
            avg_score = self._normalize_score(avg_score)
            
        return {
            "custom_score": avg_score * self.custom_weight,
            "details": {
                "individual_scores": scores,
                "threshold_used": self.threshold,
            }
        }
    
    def _compute_single(self, prediction: str, reference: str) -> float:
        """Compute score for a single prediction-reference pair."""
        # Implement custom scoring algorithm
        # Example: similarity-based scoring
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if not pred_words or not ref_words:
            return 0.0
            
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range."""
        return min(max(score, 0.0), 1.0)
```

## Plugin Discovery & Loading

### Automatic Discovery Mechanism
```python
# forge/plugins/loader.py
import importlib
import pkg_resources
from typing import Dict, List, Type, Optional
from pathlib import Path
from .base import PluginBase
from .registry import PluginRegistry

class PluginLoader:
    """Dynamic plugin loader with discovery."""
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.loaded_plugins = {}
        self.plugin_dirs = [
            Path.home() / ".forge" / "plugins",
            Path("/usr/local/share/forge/plugins"),
            Path(__file__).parent.parent / "plugins",
        ]
        
    def discover_plugins(self) -> Dict[str, Type[PluginBase]]:
        """Discover all available plugins."""
        plugins = {}
        
        # Discover from entry points
        entry_point_plugins = self._discover_entry_points()
        plugins.update(entry_point_plugins)
        
        # Discover from plugin directories
        dir_plugins = self._discover_from_directories()
        plugins.update(dir_plugins)
        
        return plugins
    
    def _discover_entry_points(self) -> Dict[str, Type[PluginBase]]:
        """Discover plugins via setuptools entry points."""
        plugins = {}
        
        for entry_point in pkg_resources.iter_entry_points("forge.plugins"):
            try:
                plugin_class = entry_point.load()
                if self._validate_plugin(plugin_class):
                    plugins[entry_point.name] = plugin_class
                    print(f"Discovered plugin: {entry_point.name} ({entry_point})")
            except Exception as e:
                print(f"Failed to load plugin {entry_point.name}: {e}")
                
        return plugins
    
    def _discover_from_directories(self) -> Dict[str, Type[PluginBase]]:
        """Discover plugins from filesystem directories."""
        plugins = {}
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
                
            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    plugin_name = item.name
                    try:
                        # Add to Python path if not already
                        if str(item.parent) not in sys.path:
                            sys.path.insert(0, str(item.parent))
                            
                        module = importlib.import_module(plugin_name)
                        plugin_class = getattr(module, "Plugin", None)
                        
                        if plugin_class and self._validate_plugin(plugin_class):
                            plugins[plugin_name] = plugin_class
                            print(f"Discovered plugin from directory: {plugin_name}")
                    except Exception as e:
                        print(f"Failed to load plugin {plugin_name} from {item}: {e}")
                        
        return plugins
    
    def _validate_plugin(self, plugin_class: Type[PluginBase]) -> bool:
        """Validate plugin class."""
        if not issubclass(plugin_class, PluginBase):
            return False
            
        required_attrs = [
            "PLUGIN_NAME", "PLUGIN_VERSION", "PLUGIN_AUTHOR",
            "PLUGIN_DESCRIPTION", "register"
        ]
        
        for attr in required_attrs:
            if not hasattr(plugin_class, attr):
                print(f"Plugin missing required attribute: {attr}")
                return False
                
        return True
    
    def load_plugin(self, plugin_name: str, plugin_class: Type[PluginBase]) -> bool:
        """Load and register a plugin."""
        try:
            plugin_instance = plugin_class()
            
            # Check version compatibility
            if not self._check_compatibility(plugin_instance):
                print(f"Plugin {plugin_name} is not compatible with current forge version")
                return False
                
            # Register plugin
            success = plugin_instance.register(self.registry)
            
            if success:
                self.loaded_plugins[plugin_name] = plugin_instance
                print(f"Successfully loaded plugin: {plugin_name} v{plugin_instance.PLUGIN_VERSION}")
                return True
            else:
                print(f"Failed to register plugin: {plugin_name}")
                return False
                
        except Exception as e:
            print(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def _check_compatibility(self, plugin: PluginBase) -> bool:
        """Check if plugin is compatible with current version."""
        from .compatibility import VersionChecker
        
        checker = VersionChecker()
        return checker.is_compatible(
            plugin.PLUGIN_VERSION,
            plugin.COMPATIBLE_VERSIONS
        )
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Discover and load all available plugins."""
        discovered = self.discover_plugins()
        results = {}
        
        for name, plugin_class in discovered.items():
            results[name] = self.load_plugin(name, plugin_class)
            
        return results
```

## Version Compatibility

### Compatibility Checking System
```python
# forge/plugins/compatibility.py
import re
from typing import List, Tuple, Optional
from packaging import version

class VersionChecker:
    """Check version compatibility between plugins and core."""
    
    CORE_VERSION = "0.3.0"  # Current forge version
    
    def __init__(self, core_version: str = None):
        self.core_version = core_version or self.CORE_VERSION
        
    def is_compatible(self, plugin_version: str, 
                     compatible_versions: List[str]) -> bool:
        """Check if plugin version is compatible."""
        if not compatible_versions:
            return True
            
        plugin_ver = version.parse(plugin_version)
        
        for compat_range in compatible_versions:
            if self._check_single_compatibility(plugin_ver, compat_range):
                return True
                
        return False
    
    def _check_single_compatibility(self, plugin_ver, compat_range: str) -> bool:
        """Check compatibility with a single version range."""
        # Handle different range formats
        if "," in compat_range:
            # Range like ">=0.2.0,<0.4.0"
            parts = compat_range.split(",")
            return all(self._check_single_compatibility(plugin_ver, part) 
                      for part in parts)
        
        # Handle operators
        if compat_range.startswith(">="):
            min_ver = version.parse(compat_range[2:].strip())
            return plugin_ver >= min_ver
        elif compat_range.startswith("<="):
            max_ver = version.parse(compat_range[2:].strip())
            return plugin_ver <= max_ver
        elif compat_range.startswith(">"):
            min_ver = version.parse(compat_range[1:].strip())
            return plugin_ver > min_ver
        elif compat_range.startswith("<"):
            max_ver = version.parse(compat_range[1:].strip())
            return plugin_ver < max_ver
        elif compat_range.startswith("=="):
            exact_ver = version.parse(compat_range[2:].strip())
            return plugin_ver == exact_ver
        else:
            # Exact version match
            exact_ver = version.parse(compat_range.strip())
            return plugin_ver == exact_ver
    
    def get_compatibility_report(self, plugin_info: dict) -> dict:
        """Generate detailed compatibility report."""
        report = {
            "compatible": False,
            "plugin_version": plugin_info.get("version"),
            "core_version": self.core_version,
            "compatible_versions": plugin_info.get("compatible_versions", []),
            "issues": [],
            "warnings": []
        }
        
        # Check basic compatibility
        if self.is_compatible(
            plugin_info["version"],
            plugin_info.get("compatible_versions", [])
        ):
            report["compatible"] = True
        else:
            report["issues"].append(
                f"Plugin version {plugin_info['version']} is not compatible "
                f"with core version {self.core_version}"
            )
        
        # Check for deprecated features
        if self._has_deprecated_features(plugin_info):
            report["warnings"].append(
                "Plugin uses deprecated APIs. Consider updating."
            )
        
        return report
    
    def _has_deprecated_features(self, plugin_info: dict) -> bool:
        """Check if plugin uses deprecated features."""
        # This would check against a list of deprecated APIs
        deprecated_apis = [
            "old_model_interface",
            "legacy_trainer_method",
            "deprecated_metric_api"
        ]
        
        # Check plugin's entry points for deprecated APIs
        entry_points = plugin_info.get("entry_points", {})
        for category, items in entry_points.items():
            for item in items:
                if any(dep in item.lower() for dep in deprecated_apis):
                    return True
                    
        return False
```

## CLI Plugin Manager

### Command Line Interface
```python
# forge/cli/plugin.py
import click
import json
import sys
from pathlib import Path
from typing import Optional
from ..plugins.loader import PluginLoader
from ..plugins.registry import PluginRegistry
from ..plugins.marketplace import PluginMarketplace

@click.group()
def plugin():
    """Manage forge plugins."""
    pass

@plugin.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list(verbose):
    """List all available plugins."""
    registry = PluginRegistry()
    loader = PluginLoader(registry)
    
    plugins = loader.discover_plugins()
    
    if not plugins:
        click.echo("No plugins found.")
        return
    
    click.echo(f"Found {len(plugins)} plugins:\n")
    
    for name, plugin_class in plugins.items():
        try:
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            
            click.echo(f"📦 {name}")
            click.echo(f"   Version: {metadata['version']}")
            click.echo(f"   Author: {metadata['author']}")
            click.echo(f"   Description: {metadata['description']}")
            
            if verbose:
                click.echo(f"   URL: {metadata.get('url', 'N/A')}")
                click.echo(f"   Compatible with: {', '.join(metadata.get('compatible_versions', ['all']))}")
                click.echo(f"   Type: {metadata.get('type', 'unknown')}")
                
            click.echo("")
            
        except Exception as e:
            click.echo(f"❌ {name}: Error loading - {e}\n")

@plugin.command()
@click.argument('plugin_name')
@click.option('--version', '-v', help='Specific version to install')
@click.option('--source', '-s', default='marketplace', 
              type=click.Choice(['marketplace', 'local', 'git']),
              help='Installation source')
def install(plugin_name, version, source):
    """Install a plugin."""
    marketplace = PluginMarketplace()
    
    click.echo(f"Installing plugin: {plugin_name}")
    
    if source == 'marketplace':
        success = marketplace.install_plugin(plugin_name, version)
    elif source == 'local':
        success = _install_local(plugin_name)
    elif source == 'git':
        success = _install_from_git(plugin_name)
    
    if success:
        click.echo(f"✅ Successfully installed {plugin_name}")
    else:
        click.echo(f"❌ Failed to install {plugin_name}")
        sys.exit(1)

@plugin.command()
@click.argument('plugin_name')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation')
def uninstall(plugin_name, confirm):
    """Uninstall a plugin."""
    if not confirm:
        click.confirm(f'Are you sure you want to uninstall {plugin_name}?', abort=True)
    
    marketplace = PluginMarketplace()
    
    if marketplace.uninstall_plugin(plugin_name):
        click.echo(f"✅ Successfully uninstalled {plugin_name}")
    else:
        click.echo(f"❌ Failed to uninstall {plugin_name}")
        sys.exit(1)

@plugin.command()
def update():
    """Update all installed plugins."""
    marketplace = PluginMarketplace()
    
    click.echo("Checking for plugin updates...")
    updates = marketplace.check_updates()
    
    if not updates:
        click.echo("All plugins are up to date.")
        return
    
    for plugin_name, update_info in updates.items():
        click.echo(f"📦 {plugin_name}: {update_info['current']} -> {update_info['latest']}")
        
        if click.confirm(f"Update {plugin_name}?"):
            if marketplace.update_plugin(plugin_name):
                click.echo(f"✅ Updated {plugin_name}")
            else:
                click.echo(f"❌ Failed to update {plugin_name}")

@plugin.command()
@click.argument('plugin_name')
def info(plugin_name):
    """Show detailed information about a plugin."""
    registry = PluginRegistry()
    loader = PluginLoader(registry)
    
    plugins = loader.discover_plugins()
    
    if plugin_name not in plugins:
        click.echo(f"Plugin '{plugin_name}' not found.")
        sys.exit(1)
    
    plugin_class = plugins[plugin_name]
    plugin_instance = plugin_class()
    metadata = plugin_instance.get_metadata()
    
    click.echo(f"Plugin: {metadata['name']}")
    click.echo(f"Version: {metadata['version']}")
    click.echo(f"Author: {metadata['author']}")
    click.echo(f"Description: {metadata['description']}")
    click.echo(f"URL: {metadata.get('url', 'N/A')}")
    click.echo(f"Type: {metadata.get('type', 'unknown')}")
    click.echo(f"Compatible versions: {', '.join(metadata.get('compatible_versions', ['all']))}")
    
    # Show entry points
    click.echo("\nEntry Points:")
    for category, items in metadata.get('entry_points', {}).items():
        click.echo(f"  {category}:")
        for item in items:
            click.echo(f"    - {item}")

@plugin.command()
def search():
    """Search the plugin marketplace."""
    marketplace = PluginMarketplace()
    
    click.echo("Searching plugin marketplace...")
    plugins = marketplace.search_plugins()
    
    if not plugins:
        click.echo("No plugins found in marketplace.")
        return
    
    click.echo(f"Found {len(plugins)} plugins:\n")
    
    for plugin in plugins:
        click.echo(f"📦 {plugin['name']} (v{plugin['version']})")
        click.echo(f"   {plugin['description']}")
        click.echo(f"   Downloads: {plugin.get('downloads', 0)}")
        click.echo(f"   Rating: {plugin.get('rating', 'N/A')}")
        click.echo("")

def _install_local(plugin_path: str) -> bool:
    """Install plugin from local directory."""
    path = Path(plugin_path)
    
    if not path.exists():
        click.echo(f"Path does not exist: {plugin_path}")
        return False
    
    # Implementation for local installation
    # This would copy the plugin to the plugins directory
    # and handle dependencies
    return True

def _install_from_git(git_url: str) -> bool:
    """Install plugin from Git repository."""
    # Implementation for Git installation
    # This would clone the repository and install the plugin
    return True
```

## Plugin Marketplace

### Marketplace Client
```python
# forge/plugins/marketplace.py
import requests
import json
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import zipfile
import shutil

class PluginMarketplace:
    """Client for interacting with the plugin marketplace."""
    
    MARKETPLACE_URL = "https://plugins.forge.ai/api/v1"
    CACHE_DIR = Path.home() / ".forge" / "marketplace_cache"
    
    def __init__(self, marketplace_url: str = None):
        self.marketplace_url = marketplace_url or self.MARKETPLACE_URL
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def search_plugins(self, query: str = None, 
                      category: str = None) -> List[Dict]:
        """Search for plugins in the marketplace."""
        params = {}
        if query:
            params['q'] = query
        if category:
            params['category'] = category
            
        try:
            response = requests.get(
                f"{self.marketplace_url}/plugins",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json().get('plugins', [])
        except requests.RequestException as e:
            print(f"Error searching marketplace: {e}")
            return []
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Get detailed information about a plugin."""
        cache_file = self.cache_dir / f"{plugin_name}.json"
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                # Cache for 1 hour
                if time.time() - cached.get('timestamp', 0) < 3600:
                    return cached.get('data')
        
        try:
            response = requests.get(
                f"{self.marketplace_url}/plugins/{plugin_name}",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
                
            return data
        except requests.RequestException as e:
            print(f"Error getting plugin info: {e}")
            return None
    
    def install_plugin(self, plugin_name: str, version: str = None) -> bool:
        """Install a plugin from the marketplace."""
        plugin_info = self.get_plugin_info(plugin_name)
        
        if not plugin_info:
            print(f"Plugin '{plugin_name}' not found in marketplace.")
            return False
        
        # Get download URL
        download_url = plugin_info.get('download_url')
        if not download_url:
            print(f"No download URL for plugin '{plugin_name}'.")
            return False
        
        # Download plugin
        try:
            print(f"Downloading {plugin_name}...")
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Verify checksum
            if not self._verify_checksum(tmp_path, plugin_info.get('checksum')):
                print("Checksum verification failed!")
                Path(tmp_path).unlink()
                return False
            
            # Extract plugin
            install_dir = Path.home() / ".forge" / "plugins" / plugin_name
            install_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(install_dir)
            
            # Install dependencies
            self._install_dependencies(install_dir)
            
            # Cleanup
            Path(tmp_path).unlink()
            
            print(f"Successfully installed {plugin_name} to {install_dir}")
            return True
            
        except Exception as e:
            print(f"Error installing plugin: {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        install_dir = Path.home() / ".forge" / "plugins" / plugin_name
        
        if not install_dir.exists():
            print(f"Plugin '{plugin_name}' is not installed.")
            return False
        
        try:
            shutil.rmtree(install_dir)
            print(f"Successfully uninstalled {plugin_name}")
            return True
        except Exception as e:
            print(f"Error uninstalling plugin: {e}")
            return False
    
    def check_updates(self) -> Dict[str, Dict]:
        """Check for updates to installed plugins."""
        updates = {}
        plugins_dir = Path.home() / ".forge" / "plugins"
        
        if not plugins_dir.exists():
            return updates
        
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir():
                plugin_name = plugin_dir.name
                local_version = self._get_local_version(plugin_dir)
                
                if local_version:
                    marketplace_info = self.get_plugin_info(plugin_name)
                    
                    if marketplace_info:
                        marketplace_version = marketplace_info.get('version')
                        
                        if marketplace_version and marketplace_version != local_version:
                            updates[plugin_name] = {
                                'current': local_version,
                                'latest': marketplace_version,
                                'changelog': marketplace_info.get('changelog', '')
                            }
        
        return updates
    
    def update_plugin(self, plugin_name: str) -> bool:
        """Update a plugin to the latest version."""
        # First uninstall old version
        if not self.uninstall_plugin(plugin_name):
            return False
        
        # Install new version
        return self.install_plugin(plugin_name)
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Verify file checksum."""
        if not expected_checksum:
            return True
            
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        return file_hash == expected_checksum
    
    def _install_dependencies(self, plugin_dir: Path) -> bool:
        """Install plugin dependencies."""
        requirements_file = plugin_dir / "requirements.txt"
        
        if requirements_file.exists():
            import subprocess
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "-r", str(requirements_file)
                ])
                return True
            except subprocess.CalledProcessError:
                return False
        
        return True
    
    def _get_local_version(self, plugin_dir: Path) -> Optional[str]:
        """Get locally installed plugin version."""
        init_file = plugin_dir / "__init__.py"
        
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
                # Look for version in plugin class
                import re
                match = re.search(r'PLUGIN_VERSION\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        
        return None
```

## Testing & Quality Assurance

### Plugin Testing Framework
```python
# tests/test_plugin_system.py
import pytest
import tempfile
from pathlib import Path
from forge.plugins.loader import PluginLoader
from forge.plugins.registry import PluginRegistry
from forge.plugins.compatibility import VersionChecker

class TestPluginSystem:
    """Test suite for plugin system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self.checker = VersionChecker()
        
    def test_plugin_discovery(self):
        """Test plugin discovery mechanism."""
        plugins = self.loader.discover_plugins()
        assert isinstance(plugins, dict)
        
    def test_plugin_loading(self):
        """Test plugin loading and registration."""
        # Create a test plugin
        test_plugin = self._create_test_plugin()
        
        # Load plugin
        success = self.loader.load_plugin("test_plugin", test_plugin)
        assert success
        
        # Check registration
        assert "test_plugin" in self.loader.loaded_plugins
        
    def test_version_compatibility(self):
        """Test version compatibility checking."""
        # Test compatible versions
        assert self.checker.is_compatible("1.0.0", [">=0.2.0,<2.0.0"])
        assert not self.checker.is_compatible("3.0.0", [">=0.2.0,<2.0.0"])
        
        # Test exact version
        assert self.checker.is_compatible("1.0.0", ["1.0.0"])
        assert not self.checker.is_compatible("1.0.1", ["1.0.0"])
        
    def test_model_plugin(self):
        """Test model plugin functionality."""
        from my_plugin.models.custom_model import MyCustomModelPlugin
        
        plugin = MyCustomModelPlugin()
        model_class = plugin.get_model_class()
        assert model_class is not None
        
        config = plugin.get_model_config("default")
        assert "hidden_size" in config
        
    def test_trainer_plugin(self):
        """Test trainer plugin functionality."""
        from my_plugin.trainers.custom_trainer import CustomTrainerPlugin
        
        plugin = CustomTrainerPlugin()
        trainer_class = plugin.get_trainer_class()
        assert trainer_class is not None
        
    def test_metric_plugin(self):
        """Test metric plugin functionality."""
        from my_plugin.metrics.custom_metric import CustomMetricPlugin
        
        plugin = CustomMetricPlugin()
        metric_class = plugin.get_metric_class()
        assert metric_class is not None
        
        metric = metric_class({"threshold": 0.5})
        result = metric.compute(["hello world"], ["hello there"])
        assert "custom_score" in result
        
    def _create_test_plugin(self):
        """Create a test plugin class."""
        from forge.plugins.base import PluginBase
        
        class TestPlugin(PluginBase):
            PLUGIN_NAME = "test_plugin"
            PLUGIN_VERSION = "1.0.0"
            PLUGIN_AUTHOR = "Test"
            PLUGIN_DESCRIPTION = "Test plugin"
            COMPATIBLE_VERSIONS = ["0.2.0", "0.3.0"]
            
            def register(self, registry):
                return True
                
            def get_metadata(self):
                return {
                    "name": self.PLUGIN_NAME,
                    "version": self.PLUGIN_VERSION,
                    "author": self.PLUGIN_AUTHOR,
                    "description": self.PLUGIN_DESCRIPTION
                }
        
        return TestPlugin

# Plugin validation script
def validate_plugin(plugin_path: Path) -> Dict[str, Any]:
    """Validate a plugin before submission to marketplace."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": []
    }
    
    # Check directory structure
    required_files = ["__init__.py", "setup.py", "README.md"]
    for file in required_files:
        if not (plugin_path / file).exists():
            results["errors"].append(f"Missing required file: {file}")
            results["valid"] = False
    
    # Check plugin class
    try:
        sys.path.insert(0, str(plugin_path.parent))
        module = importlib.import_module(plugin_path.name)
        plugin_class = getattr(module, "Plugin", None)
        
        if not plugin_class:
            results["errors"].append("No Plugin class found in __init__.py")
            results["valid"] = False
        else:
            # Check required attributes
            required_attrs = ["PLUGIN_NAME", "PLUGIN_VERSION", "register"]
            for attr in required_attrs:
                if not hasattr(plugin_class, attr):
                    results["errors"].append(f"Missing required attribute: {attr}")
                    results["valid"] = False
            
            # Check version format
            version = getattr(plugin_class, "PLUGIN_VERSION", "")
            if not re.match(r'^\d+\.\d+\.\d+$', version):
                results["warnings"].append("Version should follow semantic versioning")
            
    except Exception as e:
        results["errors"].append(f"Error loading plugin: {e}")
        results["valid"] = False
    
    # Check dependencies
    requirements_file = plugin_path / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            deps = f.read().splitlines()
            for dep in deps:
                if "==" not in dep and ">=" not in dep:
                    results["warnings"].append(f"Dependency {dep} should specify version")
    
    # Run tests if available
    test_dir = plugin_path / "tests"
    if test_dir.exists():
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                results["errors"].append(f"Tests failed: {result.stderr}")
                results["valid"] = False
            else:
                results["checks"].append("All tests passed")
        except Exception as e:
            results["warnings"].append(f"Could not run tests: {e}")
    
    return results
```

## Best Practices

### Plugin Development Guidelines

1. **Naming Conventions**
   - Plugin names: lowercase with underscores
   - Class names: PascalCase
   - Configuration keys: snake_case

2. **Documentation**
   - Include comprehensive README.md
   - Document all public APIs
   - Provide usage examples
   - Include changelog

3. **Dependencies**
   - Specify exact versions in requirements.txt
   - Minimize external dependencies
   - Handle optional dependencies gracefully

4. **Error Handling**
   - Provide clear error messages
   - Validate inputs
   - Handle missing dependencies
   - Log appropriately

5. **Testing**
   - Include unit tests
   - Test with different configurations
   - Test error conditions
   - Test compatibility

6. **Performance**
   - Optimize critical paths
   - Use lazy loading where appropriate
   - Cache expensive computations
   - Profile memory usage

7. **Security**
   - Validate all inputs
   - Sanitize file paths
   - Avoid code injection
   - Handle credentials securely

### Plugin Submission Checklist

- [ ] Plugin follows naming conventions
- [ ] All required files present
- [ ] Version follows semantic versioning
- [ ] Compatible with specified forge versions
- [ ] Tests pass
- [ ] Documentation complete
- [ ] No security vulnerabilities
- [ ] Performance acceptable
- [ ] License specified
- [ ] README includes usage examples

## Examples

### Complete Example: Custom LoRA Implementation

```python
# plugins/custom_lora/__init__.py
from forge.plugins.base import PluginBase

class CustomLoRAPlugin(PluginBase):
    """Plugin for custom LoRA implementation with advanced features."""
    
    PLUGIN_NAME = "custom_lora"
    PLUGIN_VERSION = "1.2.0"
    PLUGIN_AUTHOR = "AI Research Team"
    PLUGIN_DESCRIPTION = "Advanced LoRA implementation with adaptive rank and merging"
    PLUGIN_URL = "https://github.com/forge/custom-lora"
    COMPATIBLE_VERSIONS = [">=0.2.0,<0.5.0"]
    
    def register(self, registry):
        from .models import CustomLoRAModel
        from .trainers import CustomLoRATrainer
        from .callbacks import LoRACallback
        
        registry.register_model(CustomLoRAModel)
        registry.register_trainer(CustomLoRATrainer)
        registry.register_callback(LoRACallback)
        
        return True
    
    def get_metadata(self):
        return {
            "name": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "author": self.PLUGIN_AUTHOR,
            "description": self.PLUGIN_DESCRIPTION,
            "url": self.PLUGIN_URL,
            "compatible_versions": self.COMPATIBLE_VERSIONS,
            "type": "full",
            "tags": ["lora", "peft", "efficient-training"],
            "entry_points": {
                "models": ["CustomLoRAModel"],
                "trainers": ["CustomLoRATrainer"],
                "callbacks": ["LoRACallback"]
            },
            "requirements": [
                "torch>=2.0.0",
                "peft>=0.5.0",
                "transformers>=4.30.0"
            ]
        }

# Plugin entry point
Plugin = CustomLoRAPlugin
```

This comprehensive plugin development guide provides everything needed to create, test, and distribute plugins for forge. The system is designed to be extensible, maintainable, and community-friendly, enabling rapid innovation while maintaining stability and compatibility.