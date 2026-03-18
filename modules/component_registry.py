"""
Component Registry for Stable Diffusion WebUI
Replaces monkey-patching with proper dependency injection for model components.
"""

import logging
import importlib
from typing import Dict, Any, Optional, Type, Callable, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of components that can be registered"""
    VQ_MODEL = "vq_model"
    DDPM = "ddpm"
    LATENT_DIFFUSION = "latent_diffusion"
    AUTOENCODER = "autoencoder"
    SAMPLER = "sampler"
    SCHEDULER = "scheduler"
    UPSCALER = "upscaler"
    TEXT_ENCODER = "text_encoder"
    UNET = "unet"
    CUSTOM = "custom"

@dataclass
class ComponentRegistration:
    """Registration information for a component"""
    name: str
    component_type: ComponentType
    factory: Union[Type, Callable]
    version: str = "1.0.0"
    priority: int = 0  # Higher priority overrides lower
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ComponentRegistry:
    """
    Central registry for dependency injection of model components.
    Allows extensions to register alternative implementations without monkey-patching.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._components: Dict[str, Dict[str, ComponentRegistration]] = {}
        self._active_components: Dict[str, ComponentRegistration] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialization_hooks: list = []
        self._initialized = True
        
        # Initialize with default components from original modules
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default implementations from original modules"""
        try:
            # Default VQModel from original autoencoder
            from ldm.models.autoencoder import VQModel
            self.register_component(
                name="default_vq_model",
                component_type=ComponentType.VQ_MODEL,
                factory=VQModel,
                version="1.0.0",
                priority=0
            )
            
            # Default DDPM from original diffusion module
            from ldm.models.diffusion.ddpm import DDPM
            self.register_component(
                name="default_ddpm",
                component_type=ComponentType.DDPM,
                factory=DDPM,
                version="1.0.0",
                priority=0
            )
            
            # Default LatentDiffusion
            from ldm.models.diffusion.ddpm import LatentDiffusion
            self.register_component(
                name="default_latent_diffusion",
                component_type=ComponentType.LATENT_DIFFUSION,
                factory=LatentDiffusion,
                version="1.0.0",
                priority=0
            )
            
            logger.info("Registered default components from original modules")
            
        except ImportError as e:
            logger.warning(f"Could not import original modules for defaults: {e}")
    
    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        factory: Union[Type, Callable],
        version: str = "1.0.0",
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a component implementation.
        
        Args:
            name: Unique name for the component
            component_type: Type of component (VQ_MODEL, DDPM, etc.)
            factory: Class or factory function to create the component
            version: Version string
            priority: Priority level (higher overrides lower)
            metadata: Additional metadata
            
        Returns:
            bool: True if registration was successful
        """
        if metadata is None:
            metadata = {}
            
        registration = ComponentRegistration(
            name=name,
            component_type=component_type,
            factory=factory,
            version=version,
            priority=priority,
            metadata=metadata
        )
        
        # Initialize component type dict if needed
        if component_type.value not in self._components:
            self._components[component_type.value] = {}
        
        # Check for existing registration with same name
        if name in self._components[component_type.value]:
            existing = self._components[component_type.value][name]
            if existing.priority >= priority:
                logger.warning(
                    f"Component {name} already registered with equal/higher priority. "
                    f"Existing: {existing.priority}, New: {priority}"
                )
                return False
        
        # Store registration
        self._components[component_type.value][name] = registration
        
        # Update active component if this has higher priority
        self._update_active_component(component_type, registration)
        
        logger.info(
            f"Registered component: {name} ({component_type.value}) "
            f"v{version} with priority {priority}"
        )
        
        # Run initialization hooks
        for hook in self._initialization_hooks:
            try:
                hook(registration)
            except Exception as e:
                logger.error(f"Error in initialization hook: {e}")
        
        return True
    
    def _update_active_component(
        self, 
        component_type: ComponentType, 
        registration: ComponentRegistration
    ):
        """Update the active component for a given type"""
        current_active = self._active_components.get(component_type.value)
        
        if current_active is None or registration.priority > current_active.priority:
            self._active_components[component_type.value] = registration
            logger.debug(
                f"Set active {component_type.value} to {registration.name} "
                f"(priority: {registration.priority})"
            )
    
    def get_component(
        self, 
        component_type: ComponentType,
        name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a component instance.
        
        Args:
            component_type: Type of component to retrieve
            name: Specific component name (optional, uses active if not provided)
            **kwargs: Arguments to pass to the factory
            
        Returns:
            Component instance
        """
        if name:
            # Get specific component by name
            components = self._components.get(component_type.value, {})
            if name not in components:
                raise KeyError(
                    f"Component '{name}' not found for type {component_type.value}. "
                    f"Available: {list(components.keys())}"
                )
            registration = components[name]
        else:
            # Get active component for type
            registration = self._active_components.get(component_type.value)
            if registration is None:
                raise ValueError(
                    f"No active component registered for type {component_type.value}. "
                    f"Available types: {list(self._active_components.keys())}"
                )
        
        # Create instance using factory
        try:
            if callable(registration.factory):
                return registration.factory(**kwargs)
            else:
                # Assume it's a class
                return registration.factory(**kwargs)
        except Exception as e:
            logger.error(
                f"Failed to instantiate component {registration.name}: {e}"
            )
            raise
    
    def get_factory(
        self, 
        component_type: ComponentType,
        name: Optional[str] = None
    ) -> Callable:
        """
        Get a factory function for a component.
        
        Args:
            component_type: Type of component
            name: Specific component name (optional)
            
        Returns:
            Factory function or class
        """
        if name:
            components = self._components.get(component_type.value, {})
            if name not in components:
                raise KeyError(f"Component '{name}' not found")
            return components[name].factory
        else:
            registration = self._active_components.get(component_type.value)
            if registration is None:
                raise ValueError(f"No active component for type {component_type.value}")
            return registration.factory
    
    def unregister_component(
        self, 
        component_type: ComponentType, 
        name: str
    ) -> bool:
        """
        Unregister a component.
        
        Args:
            component_type: Type of component
            name: Component name to unregister
            
        Returns:
            bool: True if unregistered successfully
        """
        components = self._components.get(component_type.value, {})
        if name not in components:
            return False
        
        del components[name]
        
        # Update active component if we removed the active one
        active = self._active_components.get(component_type.value)
        if active and active.name == name:
            # Find next highest priority
            if components:
                new_active = max(components.values(), key=lambda x: x.priority)
                self._active_components[component_type.value] = new_active
            else:
                del self._active_components[component_type.value]
        
        logger.info(f"Unregistered component: {name} ({component_type.value})")
        return True
    
    def list_components(
        self, 
        component_type: Optional[ComponentType] = None
    ) -> Dict[str, Any]:
        """
        List registered components.
        
        Args:
            component_type: Filter by component type (optional)
            
        Returns:
            Dict of component info
        """
        if component_type:
            components = self._components.get(component_type.value, {})
            return {
                name: {
                    "type": reg.component_type.value,
                    "version": reg.version,
                    "priority": reg.priority,
                    "metadata": reg.metadata,
                    "active": self._active_components.get(component_type.value) == reg
                }
                for name, reg in components.items()
            }
        else:
            result = {}
            for comp_type, components in self._components.items():
                for name, reg in components.items():
                    key = f"{comp_type}:{name}"
                    result[key] = {
                        "type": reg.component_type.value,
                        "version": reg.version,
                        "priority": reg.priority,
                        "metadata": reg.metadata,
                        "active": self._active_components.get(comp_type) == reg
                    }
            return result
    
    def add_initialization_hook(self, hook: Callable):
        """Add a hook to run when components are registered"""
        self._initialization_hooks.append(hook)
    
    def clear(self):
        """Clear all registrations (useful for testing)"""
        self._components.clear()
        self._active_components.clear()
        self._register_default_components()
        logger.info("Component registry cleared and defaults re-registered")
    
    def validate_dependencies(self) -> Dict[str, list]:
        """
        Validate that all required components are available.
        
        Returns:
            Dict with validation results
        """
        required_types = [
            ComponentType.VQ_MODEL,
            ComponentType.DDPM,
            ComponentType.LATENT_DIFFUSION
        ]
        
        missing = []
        for comp_type in required_types:
            if comp_type.value not in self._active_components:
                missing.append(comp_type.value)
        
        return {
            "valid": len(missing) == 0,
            "missing_components": missing,
            "available_types": list(self._active_components.keys())
        }

# Global registry instance
_registry = ComponentRegistry()

# Public API functions
def register_component(
    name: str,
    component_type: ComponentType,
    factory: Union[Type, Callable],
    version: str = "1.0.0",
    priority: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Register a component implementation"""
    return _registry.register_component(
        name, component_type, factory, version, priority, metadata
    )

def get_component(
    component_type: ComponentType,
    name: Optional[str] = None,
    **kwargs
) -> Any:
    """Get a component instance"""
    return _registry.get_component(component_type, name, **kwargs)

def get_factory(
    component_type: ComponentType,
    name: Optional[str] = None
) -> Callable:
    """Get a component factory"""
    return _registry.get_factory(component_type, name)

def unregister_component(component_type: ComponentType, name: str) -> bool:
    """Unregister a component"""
    return _registry.unregister_component(component_type, name)

def list_components(component_type: Optional[ComponentType] = None) -> Dict[str, Any]:
    """List registered components"""
    return _registry.list_components(component_type)

def add_initialization_hook(hook: Callable):
    """Add initialization hook"""
    _registry.add_initialization_hook(hook)

def clear_registry():
    """Clear registry (for testing)"""
    _registry.clear()

def validate_dependencies() -> Dict[str, list]:
    """Validate component dependencies"""
    return _registry.validate_dependencies()

# Context manager for temporary component overrides
class ComponentOverride:
    """Context manager for temporarily overriding a component"""
    
    def __init__(
        self,
        component_type: ComponentType,
        factory: Union[Type, Callable],
        name: str = "temporary_override",
        priority: int = 1000
    ):
        self.component_type = component_type
        self.factory = factory
        self.name = name
        self.priority = priority
        self.previous_active = None
    
    def __enter__(self):
        # Store current active component
        self.previous_active = _registry._active_components.get(
            self.component_type.value
        )
        
        # Register temporary override
        register_component(
            name=self.name,
            component_type=self.component_type,
            factory=self.factory,
            priority=self.priority
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove temporary override
        unregister_component(self.component_type, self.name)
        
        # Restore previous active if it existed
        if self.previous_active:
            _registry._active_components[self.component_type.value] = self.previous_active

# Decorator for registering components
def register(
    component_type: ComponentType,
    name: Optional[str] = None,
    version: str = "1.0.0",
    priority: int = 0,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator to register a class or function as a component.
    
    Usage:
        @register(ComponentType.VQ_MODEL, name="my_vq_model")
        class MyVQModel(VQModel):
            ...
    """
    def decorator(cls_or_func):
        component_name = name or cls_or_func.__name__
        register_component(
            name=component_name,
            component_type=component_type,
            factory=cls_or_func,
            version=version,
            priority=priority,
            metadata=metadata
        )
        return cls_or_func
    
    return decorator

# Integration helpers for existing hijack system
def migrate_from_hijack(hijack_module_name: str) -> Dict[str, Any]:
    """
    Helper to migrate from existing hijack modules to component registry.
    
    Args:
        hijack_module_name: Name of the hijack module (e.g., 'sd_hijack_autoencoder')
        
    Returns:
        Dict with migration results
    """
    try:
        module = importlib.import_module(hijack_module_name)
        
        migrated = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Look for classes that might be hijack replacements
            if isinstance(attr, type) and hasattr(attr, '__module__'):
                if 'VQModel' in attr_name or 'DDPM' in attr_name or 'LatentDiffusion' in attr_name:
                    # Determine component type
                    if 'VQModel' in attr_name:
                        comp_type = ComponentType.VQ_MODEL
                    elif 'DDPM' in attr_name:
                        comp_type = ComponentType.DDPM
                    elif 'LatentDiffusion' in attr_name:
                        comp_type = ComponentType.LATENT_DIFFUSION
                    else:
                        comp_type = ComponentType.CUSTOM
                    
                    # Register as high-priority component
                    register_component(
                        name=f"migrated_{attr_name.lower()}",
                        component_type=comp_type,
                        factory=attr,
                        version="1.0.0",
                        priority=100,  # Higher than default
                        metadata={"migrated_from": hijack_module_name}
                    )
                    migrated.append(attr_name)
        
        return {
            "success": True,
            "module": hijack_module_name,
            "migrated_components": migrated,
            "count": len(migrated)
        }
        
    except ImportError as e:
        return {
            "success": False,
            "module": hijack_module_name,
            "error": str(e)
        }

# Example usage for extensions
"""
# In an extension's __init__.py or setup.py:

from modules.component_registry import register, ComponentType

@register(ComponentType.VQ_MODEL, name="my_optimized_vq", priority=100)
class OptimizedVQModel:
    def __init__(self, *args, **kwargs):
        # Custom implementation
        pass

# Or manually:
from modules.component_registry import register_component

register_component(
    name="my_ddpm_variant",
    component_type=ComponentType.DDPM,
    factory=MyDDPMClass,
    version="2.0.0",
    priority=50,
    metadata={"optimizations": ["xformers", "flash_attention"]}
)
"""

# Initialize logging
logger.info("Component registry initialized")
logger.debug(f"Default components validated: {validate_dependencies()}")