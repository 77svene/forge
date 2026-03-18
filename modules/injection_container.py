"""
Dependency Injection Container for Stable Diffusion WebUI
Replaces monkey-patching with clean component registry system.
"""

import importlib
import logging
import threading
from typing import Dict, Any, Callable, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components that can be injected."""
    AUTOENCODER = "autoencoder"
    DDPM = "ddpm"
    VQ_MODEL = "vq_model"
    LDM = "ldm"
    SAMPLER = "sampler"
    SCHEDULER = "scheduler"
    TEXT_ENCODER = "text_encoder"
    UNET = "unet"
    VAE = "vae"
    CUSTOM = "custom"


@dataclass
class ComponentRegistration:
    """Registration information for a component."""
    factory: Callable[..., Any]
    priority: int = 0
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider: str = "core"


class InjectionContainer:
    """
    Central dependency injection container for model components.
    
    Allows extensions to register alternative implementations without
    monkey-patching or global state pollution.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the container."""
        self._registries: Dict[ComponentType, Dict[str, ComponentRegistration]] = {
            component_type: {} for component_type in ComponentType
        }
        self._custom_registries: Dict[str, Dict[str, ComponentRegistration]] = {}
        self._active_components: Dict[ComponentType, str] = {}
        self._component_cache: Dict[Tuple[ComponentType, str], Any] = {}
        self._extension_components: Dict[str, Set[str]] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        
        # Register default implementations
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default component implementations."""
        # These will be populated as modules are loaded
        pass
    
    def register_component(
        self,
        component_type: ComponentType,
        name: str,
        factory: Callable[..., Any],
        priority: int = 0,
        version: str = "1.0.0",
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provider: str = "core",
        extension_name: Optional[str] = None
    ) -> None:
        """
        Register a component implementation.
        
        Args:
            component_type: Type of component (AUTOENCODER, DDPM, etc.)
            name: Unique name for this implementation
            factory: Factory function that creates the component
            priority: Higher priority implementations are preferred
            version: Semantic version of the implementation
            dependencies: List of component names this depends on
            metadata: Additional metadata about the component
            provider: Who provides this implementation
            extension_name: Name of extension providing this component
        """
        with self._lock:
            if dependencies is None:
                dependencies = []
            if metadata is None:
                metadata = {}
            
            registration = ComponentRegistration(
                factory=factory,
                priority=priority,
                version=version,
                dependencies=dependencies,
                metadata=metadata,
                provider=provider
            )
            
            if component_type == ComponentType.CUSTOM:
                # Custom components use their own namespace
                if name not in self._custom_registries:
                    self._custom_registries[name] = {}
                self._custom_registries[name][name] = registration
            else:
                self._registries[component_type][name] = registration
            
            # Track extension components
            if extension_name:
                if extension_name not in self._extension_components:
                    self._extension_components[extension_name] = set()
                self._extension_components[extension_name].add(
                    f"{component_type.value}:{name}"
                )
            
            # Update dependency graph
            for dep in dependencies:
                if dep not in self._dependency_graph:
                    self._dependency_graph[dep] = set()
                self._dependency_graph[dep].add(name)
            
            logger.info(
                f"Registered component: {component_type.value}/{name} "
                f"(priority: {priority}, provider: {provider})"
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
            name: Specific implementation name, or None for best available
            **kwargs: Arguments to pass to the factory
            
        Returns:
            Component instance
            
        Raises:
            ComponentNotFoundError: If no suitable component is found
            CircularDependencyError: If circular dependencies are detected
        """
        with self._lock:
            cache_key = (component_type, name)
            
            # Check cache first
            if cache_key in self._component_cache:
                return self._component_cache[cache_key]
            
            # Get registration
            registration = self._get_registration(component_type, name)
            if registration is None:
                raise ComponentNotFoundError(
                    f"No component found for {component_type.value}"
                    + (f"/{name}" if name else "")
                )
            
            # Check dependencies
            self._resolve_dependencies(registration.dependencies, **kwargs)
            
            # Create instance
            try:
                instance = registration.factory(**kwargs)
                self._component_cache[cache_key] = instance
                return instance
            except Exception as e:
                logger.error(
                    f"Failed to create component {component_type.value}/{name}: {e}"
                )
                raise
    
    def _get_registration(
        self,
        component_type: ComponentType,
        name: Optional[str] = None
    ) -> Optional[ComponentRegistration]:
        """Get the best matching registration."""
        if component_type == ComponentType.CUSTOM:
            # Custom components require a name
            if name is None:
                return None
            return self._custom_registries.get(name, {}).get(name)
        
        registry = self._registries.get(component_type, {})
        
        if name:
            # Specific name requested
            return registry.get(name)
        
        if not registry:
            return None
        
        # Return highest priority implementation
        return max(
            registry.values(),
            key=lambda reg: (reg.priority, reg.version),
            default=None
        )
    
    def _resolve_dependencies(
        self,
        dependencies: List[str],
        **kwargs
    ) -> None:
        """Resolve and validate dependencies."""
        visited = set()
        visiting = set()
        
        def visit(dep_name: str):
            if dep_name in visiting:
                raise CircularDependencyError(
                    f"Circular dependency detected: {dep_name}"
                )
            if dep_name in visited:
                return
            
            visiting.add(dep_name)
            
            # Find the component type for this dependency
            for comp_type, registry in self._registries.items():
                if dep_name in registry:
                    # Recursively resolve its dependencies
                    self._resolve_dependencies(
                        registry[dep_name].dependencies,
                        **kwargs
                    )
                    break
            
            visiting.remove(dep_name)
            visited.add(dep_name)
        
        for dep in dependencies:
            visit(dep)
    
    def list_components(
        self,
        component_type: Optional[ComponentType] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all registered components.
        
        Args:
            component_type: Optional filter by component type
            
        Returns:
            Dictionary mapping component types to lists of component info
        """
        with self._lock:
            result = {}
            
            types_to_check = (
                [component_type] if component_type
                else list(ComponentType)
            )
            
            for comp_type in types_to_check:
                if comp_type == ComponentType.CUSTOM:
                    components = []
                    for name, registry in self._custom_registries.items():
                        for reg_name, reg in registry.items():
                            components.append({
                                "name": reg_name,
                                "priority": reg.priority,
                                "version": reg.version,
                                "provider": reg.provider,
                                "metadata": reg.metadata
                            })
                else:
                    components = []
                    registry = self._registries.get(comp_type, {})
                    for name, reg in registry.items():
                        components.append({
                            "name": name,
                            "priority": reg.priority,
                            "version": reg.version,
                            "provider": reg.provider,
                            "metadata": reg.metadata
                        })
                
                if components:
                    result[comp_type.value] = components
            
            return result
    
    def unregister_component(
        self,
        component_type: ComponentType,
        name: str,
        extension_name: Optional[str] = None
    ) -> bool:
        """
        Unregister a component.
        
        Args:
            component_type: Type of component
            name: Name of implementation
            extension_name: If provided, only unregister if from this extension
            
        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if component_type == ComponentType.CUSTOM:
                if name in self._custom_registries:
                    if extension_name:
                        # Check if this extension owns it
                        reg = self._custom_registries[name].get(name)
                        if reg and reg.provider != extension_name:
                            return False
                    del self._custom_registries[name]
                    return True
                return False
            
            registry = self._registries.get(component_type, {})
            if name in registry:
                if extension_name:
                    reg = registry[name]
                    if reg.provider != extension_name:
                        return False
                del registry[name]
                
                # Clear cache
                cache_key = (component_type, name)
                self._component_cache.pop(cache_key, None)
                
                # Update extension tracking
                if extension_name and extension_name in self._extension_components:
                    self._extension_components[extension_name].discard(
                        f"{component_type.value}:{name}"
                    )
                
                return True
            return False
    
    def clear_cache(
        self,
        component_type: Optional[ComponentType] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Clear the component cache.
        
        Args:
            component_type: Optional specific type to clear
            name: Optional specific name to clear
        """
        with self._lock:
            if component_type and name:
                cache_key = (component_type, name)
                self._component_cache.pop(cache_key, None)
            elif component_type:
                keys_to_remove = [
                    k for k in self._component_cache.keys()
                    if k[0] == component_type
                ]
                for key in keys_to_remove:
                    del self._component_cache[key]
            else:
                self._component_cache.clear()
    
    def get_extension_components(
        self,
        extension_name: str
    ) -> List[str]:
        """
        Get all components provided by an extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            List of component identifiers
        """
        with self._lock:
            return list(self._extension_components.get(extension_name, set()))
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate all component dependencies.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        with self._lock:
            # Check for missing dependencies
            for comp_type, registry in self._registries.items():
                for name, reg in registry.items():
                    for dep in reg.dependencies:
                        found = False
                        for check_type, check_registry in self._registries.items():
                            if dep in check_registry:
                                found = True
                                break
                        if not found:
                            errors.append(
                                f"{comp_type.value}/{name} depends on "
                                f"missing component: {dep}"
                            )
            
            # Check for circular dependencies
            try:
                for comp_type, registry in self._registries.items():
                    for name in registry:
                        self._resolve_dependencies(registry[name].dependencies)
            except CircularDependencyError as e:
                errors.append(str(e))
        
        return errors
    
    def reset(self) -> None:
        """Reset the container to initial state (for testing)."""
        with self._lock:
            self._registries = {
                component_type: {} for component_type in ComponentType
            }
            self._custom_registries.clear()
            self._active_components.clear()
            self._component_cache.clear()
            self._extension_components.clear()
            self._dependency_graph.clear()
            self._register_defaults()


# Custom Exceptions
class ComponentNotFoundError(Exception):
    """Raised when a requested component is not found."""
    pass


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class RegistrationError(Exception):
    """Raised when component registration fails."""
    pass


# Global container instance
_container = None


def get_container() -> InjectionContainer:
    """Get the global injection container instance."""
    global _container
    if _container is None:
        _container = InjectionContainer()
    return _container


def register_autoencoder(
    name: str,
    factory: Callable[..., Any],
    priority: int = 0,
    **kwargs
) -> None:
    """Convenience function to register an autoencoder."""
    get_container().register_component(
        ComponentType.AUTOENCODER, name, factory, priority, **kwargs
    )


def register_ddpm(
    name: str,
    factory: Callable[..., Any],
    priority: int = 0,
    **kwargs
) -> None:
    """Convenience function to register a DDPM model."""
    get_container().register_component(
        ComponentType.DDPM, name, factory, priority, **kwargs
    )


def register_vq_model(
    name: str,
    factory: Callable[..., Any],
    priority: int = 0,
    **kwargs
) -> None:
    """Convenience function to register a VQ model."""
    get_container().register_component(
        ComponentType.VQ_MODEL, name, factory, priority, **kwargs
    )


def get_autoencoder(name: Optional[str] = None, **kwargs) -> Any:
    """Convenience function to get an autoencoder instance."""
    return get_container().get_component(
        ComponentType.AUTOENCODER, name, **kwargs
    )


def get_ddpm(name: Optional[str] = None, **kwargs) -> Any:
    """Convenience function to get a DDPM instance."""
    return get_container().get_component(ComponentType.DDPM, name, **kwargs)


def get_vq_model(name: Optional[str] = None, **kwargs) -> Any:
    """Convenience function to get a VQ model instance."""
    return get_container().get_component(ComponentType.VQ_MODEL, name, **kwargs)


# Migration helpers for existing hijack code
def migrate_hijack_to_injection():
    """
    Helper function to migrate existing hijack code to use injection.
    This should be called during the transition period.
    """
    container = get_container()
    
    # Import existing hijack modules
    try:
        from modules import sd_hijack_autoencoder
        from modules import sd_hijack_ddpm_v1
        
        # Register their implementations
        if hasattr(sd_hijack_autoencoder, 'AutoencoderKL'):
            register_autoencoder(
                "original_hijack",
                lambda: sd_hijack_autoencoder.AutoencoderKL,
                priority=10,
                provider="legacy_hijack",
                metadata={"source": "sd_hijack_autoencoder.py"}
            )
        
        if hasattr(sd_hijack_ddpm_v1, 'DDPM'):
            register_ddpm(
                "original_hijack",
                lambda: sd_hijack_ddpm_v1.DDPM,
                priority=10,
                provider="legacy_hijack",
                metadata={"source": "sd_hijack_ddpm_v1.py"}
            )
        
        logger.info("Migrated existing hijack code to injection container")
        
    except ImportError as e:
        logger.warning(f"Could not migrate hijack code: {e}")


# Extension integration API
class ExtensionRegistrar:
    """Helper class for extensions to register components."""
    
    def __init__(self, extension_name: str):
        self.extension_name = extension_name
        self.container = get_container()
    
    def register_autoencoder(
        self,
        name: str,
        factory: Callable[..., Any],
        priority: int = 0,
        **kwargs
    ) -> None:
        """Register an autoencoder for this extension."""
        self.container.register_component(
            ComponentType.AUTOENCODER,
            name,
            factory,
            priority,
            provider=self.extension_name,
            extension_name=self.extension_name,
            **kwargs
        )
    
    def register_ddpm(
        self,
        name: str,
        factory: Callable[..., Any],
        priority: int = 0,
        **kwargs
    ) -> None:
        """Register a DDPM model for this extension."""
        self.container.register_component(
            ComponentType.DDPM,
            name,
            factory,
            priority,
            provider=self.extension_name,
            extension_name=self.extension_name,
            **kwargs
        )
    
    def register_custom(
        self,
        name: str,
        factory: Callable[..., Any],
        priority: int = 0,
        **kwargs
    ) -> None:
        """Register a custom component for this extension."""
        self.container.register_component(
            ComponentType.CUSTOM,
            name,
            factory,
            priority,
            provider=self.extension_name,
            extension_name=self.extension_name,
            **kwargs
        )
    
    def unregister_all(self) -> None:
        """Unregister all components from this extension."""
        components = self.container.get_extension_components(self.extension_name)
        for component_id in components:
            comp_type_str, comp_name = component_id.split(":", 1)
            comp_type = ComponentType(comp_type_str)
            self.container.unregister_component(
                comp_type, comp_name, self.extension_name
            )


# Example usage for extensions
"""
# In an extension's __init__.py:

from modules.injection_container import ExtensionRegistrar

registrar = ExtensionRegistrar("my_extension")

# Register alternative autoencoder
def create_my_autoencoder(**kwargs):
    from my_extension.autoencoder import MyAutoencoderKL
    return MyAutoencoderKL(**kwargs)

registrar.register_autoencoder(
    "my_autoencoder",
    create_my_autoencoder,
    priority=20,  # Higher than default
    version="1.0.0",
    metadata={"description": "Custom autoencoder with enhanced features"}
)

# Register alternative DDPM
def create_my_ddpm(**kwargs):
    from my_extension.ddpm import MyDDPM
    return MyDDPM(**kwargs)

registrar.register_ddpm(
    "my_ddpm",
    create_my_ddpm,
    priority=20,
    version="1.0.0"
)
"""

# Auto-migrate on import
migrate_hijack_to_injection()