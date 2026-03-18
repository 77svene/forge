"""
Unified Model Registry & Dynamic Adapter System for forge.

This module replaces static model conversion scripts with a dynamic model registry
that supports instant model addition via configuration files and adapter plugins.
"""

import importlib
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a registered model architecture."""
    name: str
    model_type: str
    config_class: Type[PretrainedConfig]
    model_class: Type[PreTrainedModel]
    adapter_class: Optional[Type["BaseAdapter"]] = None
    default_checkpoint: Optional[str] = None
    supported_methods: List[str] = field(default_factory=lambda: ["full", "lora", "qlora"])
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    All model-specific logic should be implemented in subclasses of this adapter.
    This includes weight loading, LoRA application, and special token handling.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        **kwargs
    ) -> PreTrainedModel:
        """
        Load a model from checkpoint or HuggingFace Hub.
        
        Args:
            model_name_or_path: Path to model checkpoint or HuggingFace model ID
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        pass
    
    @abstractmethod
    def load_weights(self, model: PreTrainedModel, checkpoint_path: str, **kwargs) -> PreTrainedModel:
        """
        Load weights from a checkpoint into an existing model.
        
        Args:
            model: Target model to load weights into
            checkpoint_path: Path to checkpoint file or directory
            **kwargs: Additional arguments for weight loading
            
        Returns:
            Model with loaded weights
        """
        pass
    
    @abstractmethod
    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: Dict[str, Any],
        **kwargs
    ) -> PreTrainedModel:
        """
        Apply LoRA adapters to the model.
        
        Args:
            model: Base model to apply LoRA to
            lora_config: LoRA configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Model with LoRA adapters applied
        """
        pass
    
    @abstractmethod
    def handle_special_tokens(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        special_tokens: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Handle special tokens for the model and tokenizer.
        
        Args:
            model: Model to update
            tokenizer: Tokenizer to update
            special_tokens: Dictionary of special tokens
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (updated_model, updated_tokenizer)
        """
        pass
    
    @abstractmethod
    def convert_to_hf(self, model: PreTrainedModel, **kwargs) -> PreTrainedModel:
        """
        Convert model to HuggingFace format if needed.
        
        Args:
            model: Model to convert
            **kwargs: Additional arguments
            
        Returns:
            Converted model
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        return True


class ModelRegistry:
    """
    Central registry for model architectures and their adapters.
    
    This is a singleton class that maintains a mapping of model names to their
    configurations and adapter classes. New models can be registered at runtime.
    """
    
    _instance = None
    _registry: Dict[str, ModelConfig] = {}
    _adapters: Dict[str, BaseAdapter] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize registry with built-in models."""
        self._load_builtin_adapters()
        self._load_config_adapters()
    
    def _load_builtin_adapters(self):
        """Load built-in model adapters."""
        builtin_adapters = {
            "llama": "forge.model.adapters.llama.LlamaAdapter",
            "qwen": "forge.model.adapters.qwen.QwenAdapter",
            "baichuan": "forge.model.adapters.baichuan.BaichuanAdapter",
            "mistral": "forge.model.adapters.mistral.MistralAdapter",
            "chatglm": "forge.model.adapters.chatglm.ChatGLMAdapter",
            "bloom": "forge.model.adapters.bloom.BloomAdapter",
            "gpt2": "forge.model.adapters.gpt2.GPT2Adapter",
        }
        
        for model_type, adapter_path in builtin_adapters.items():
            try:
                self._load_adapter_from_path(model_type, adapter_path)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not load built-in adapter for {model_type}: {e}")
    
    def _load_config_adapters(self):
        """Load adapters from configuration files."""
        config_dir = Path(__file__).parent / "adapters" / "configs"
        if not config_dir.exists():
            return
        
        for config_file in config_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                
                if "model_type" in config_data and "adapter_path" in config_data:
                    self._load_adapter_from_path(
                        config_data["model_type"],
                        config_data["adapter_path"],
                        config_data
                    )
            except Exception as e:
                logger.warning(f"Failed to load adapter config from {config_file}: {e}")
    
    def _load_adapter_from_path(
        self,
        model_type: str,
        adapter_path: str,
        config_data: Optional[Dict] = None
    ):
        """Dynamically load an adapter from a module path."""
        try:
            module_path, class_name = adapter_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            
            if not issubclass(adapter_class, BaseAdapter):
                raise TypeError(f"{adapter_path} is not a subclass of BaseAdapter")
            
            # Get model configuration
            if config_data and "config_class" in config_data:
                config_module, config_class_name = config_data["config_class"].rsplit(".", 1)
                config_class = getattr(importlib.import_module(config_module), config_class_name)
            else:
                # Try to infer from transformers
                config_class = AutoConfig.for_model(model_type)
            
            # Get model class
            if config_data and "model_class" in config_data:
                model_module, model_class_name = config_data["model_class"].rsplit(".", 1)
                model_class = getattr(importlib.import_module(model_module), model_class_name)
            else:
                model_class = AutoModelForCausalLM
            
            self.register(
                name=model_type,
                model_type=model_type,
                config_class=config_class,
                model_class=model_class,
                adapter_class=adapter_class,
                metadata=config_data.get("metadata", {}) if config_data else {}
            )
            
            logger.info(f"Registered adapter for {model_type} from {adapter_path}")
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_path}: {e}")
            raise
    
    def register(
        self,
        name: str,
        model_type: str,
        config_class: Type[PretrainedConfig],
        model_class: Type[PreTrainedModel],
        adapter_class: Optional[Type[BaseAdapter]] = None,
        default_checkpoint: Optional[str] = None,
        supported_methods: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new model architecture.
        
        Args:
            name: Unique name for the model
            model_type: Type identifier for the model
            config_class: HuggingFace config class for the model
            model_class: HuggingFace model class
            adapter_class: Adapter class for model-specific operations
            default_checkpoint: Default checkpoint path or HuggingFace ID
            supported_methods: List of supported training methods
            metadata: Additional metadata
        """
        if name in self._registry:
            logger.warning(f"Overwriting existing registration for {name}")
        
        if adapter_class is None:
            adapter_class = DefaultAdapter
        
        self._registry[name] = ModelConfig(
            name=name,
            model_type=model_type,
            config_class=config_class,
            model_class=model_class,
            adapter_class=adapter_class,
            default_checkpoint=default_checkpoint,
            supported_methods=supported_methods or ["full", "lora", "qlora"],
            metadata=metadata or {}
        )
        
        logger.debug(f"Registered model: {name} (type: {model_type})")
    
    def get(self, name: str) -> ModelConfig:
        """Get configuration for a registered model."""
        if name not in self._registry:
            # Try to find by model_type
            for config in self._registry.values():
                if config.model_type == name:
                    return config
            
            available = list(self._registry.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        return self._registry[name]
    
    def get_adapter(self, name: str) -> BaseAdapter:
        """Get or create an adapter instance for a model."""
        config = self.get(name)
        
        if name not in self._adapters:
            self._adapters[name] = config.adapter_class(config)
        
        return self._adapters[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._registry.keys())
    
    def list_by_type(self, model_type: str) -> List[str]:
        """List all models of a specific type."""
        return [
            name for name, config in self._registry.items()
            if config.model_type == model_type
        ]
    
    def unregister(self, name: str):
        """Remove a model from the registry."""
        if name in self._registry:
            del self._registry[name]
        if name in self._adapters:
            del self._adapters[name]
    
    def clear(self):
        """Clear all registrations (mainly for testing)."""
        self._registry.clear()
        self._adapters.clear()


class DefaultAdapter(BaseAdapter):
    """Default adapter implementation using HuggingFace Auto classes."""
    
    def load_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        **kwargs
    ) -> PreTrainedModel:
        """Load model using AutoModelForCausalLM."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
            self.model = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_name_or_path}: {e}")
            raise
    
    def load_weights(self, model: PreTrainedModel, checkpoint_path: str, **kwargs) -> PreTrainedModel:
        """Load weights using standard HuggingFace loading."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Handle potential key mismatches
            model_dict = model.state_dict()
            for key in state_dict:
                if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                    model_dict[key] = state_dict[key]
            
            model.load_state_dict(model_dict, strict=False)
            return model
        except Exception as e:
            logger.error(f"Failed to load weights from {checkpoint_path}: {e}")
            raise
    
    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: Dict[str, Any],
        **kwargs
    ) -> PreTrainedModel:
        """Apply LoRA using PEFT library."""
        try:
            from peft import LoraConfig, get_peft_model
            
            config = LoraConfig(**lora_config)
            model = get_peft_model(model, config)
            return model
        except ImportError:
            logger.error("PEFT library not installed. Please install with: pip install peft")
            raise
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise
    
    def handle_special_tokens(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        special_tokens: Dict[str, str],
        **kwargs
    ) -> tuple:
        """Handle special tokens by resizing embeddings."""
        try:
            # Add special tokens to tokenizer
            special_tokens_dict = {"additional_special_tokens": list(special_tokens.values())}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            
            if num_added > 0:
                # Resize model embeddings
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Added {num_added} special tokens and resized embeddings")
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to handle special tokens: {e}")
            raise
    
    def convert_to_hf(self, model: PreTrainedModel, **kwargs) -> PreTrainedModel:
        """No conversion needed for HuggingFace models."""
        return model
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Basic config validation."""
        required_fields = ["model_type", "hidden_size", "num_attention_heads"]
        return all(field in config for field in required_fields)


def register_model(
    name: str,
    model_type: str,
    config_class: Type[PretrainedConfig],
    model_class: Type[PreTrainedModel],
    adapter_class: Optional[Type[BaseAdapter]] = None,
    **kwargs
):
    """
    Decorator to register a model adapter.
    
    Example usage:
        @register_model("my_llama", "llama", LlamaConfig, LlamaForCausalLM)
        class MyLlamaAdapter(BaseAdapter):
            # Implement adapter methods
            pass
    """
    def decorator(adapter_cls: Type[BaseAdapter]):
        registry = ModelRegistry()
        registry.register(
            name=name,
            model_type=model_type,
            config_class=config_class,
            model_class=model_class,
            adapter_class=adapter_cls,
            **kwargs
        )
        return adapter_cls
    return decorator


def load_model(
    model_name_or_path: str,
    model_type: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    **kwargs
) -> tuple:
    """
    Unified model loading function using the registry.
    
    Args:
        model_name_or_path: Model name, path, or HuggingFace ID
        model_type: Optional model type override
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, tokenizer, adapter)
    """
    from transformers import AutoTokenizer
    
    registry = ModelRegistry()
    
    # Determine model type
    if model_type is None:
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            model_type = config.model_type
        except Exception:
            # Try to infer from name
            for name in registry.list_models():
                if name in model_name_or_path.lower():
                    model_type = name
                    break
    
    if model_type is None:
        raise ValueError(
            f"Could not determine model type for {model_name_or_path}. "
            "Please specify model_type explicitly."
        )
    
    # Get adapter
    adapter = registry.get_adapter(model_type)
    
    # Load model
    model = adapter.load_model(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, adapter


def convert_checkpoint(
    source_path: str,
    target_path: str,
    source_type: str,
    target_type: str = "llama",
    **kwargs
):
    """
    Convert checkpoint between different model formats.
    
    Args:
        source_path: Path to source checkpoint
        target_path: Path to save converted checkpoint
        source_type: Source model type
        target_type: Target model type (default: llama)
        **kwargs: Additional conversion arguments
    """
    registry = ModelRegistry()
    
    # Get source adapter
    try:
        source_adapter = registry.get_adapter(source_type)
    except KeyError:
        raise ValueError(f"Unsupported source model type: {source_type}")
    
    # Load source model
    logger.info(f"Loading source model from {source_path}")
    model = source_adapter.load_model(source_path, device_map="cpu")
    
    # Convert to target format
    if target_type != source_type:
        logger.info(f"Converting from {source_type} to {target_type}")
        model = source_adapter.convert_to_hf(model, **kwargs)
    
    # Save converted model
    logger.info(f"Saving converted model to {target_path}")
    model.save_pretrained(target_path)
    
    # Also save tokenizer if available
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
        tokenizer.save_pretrained(target_path)
    except Exception as e:
        logger.warning(f"Could not save tokenizer: {e}")
    
    logger.info("Conversion completed successfully")


# Auto-discover and register adapters from the adapters directory
def _auto_discover_adapters():
    """Automatically discover and register adapters from the adapters directory."""
    adapters_dir = Path(__file__).parent / "adapters"
    if not adapters_dir.exists():
        return
    
    for adapter_file in adapters_dir.glob("*_adapter.py"):
        try:
            module_name = f"forge.model.adapters.{adapter_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, adapter_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for adapter classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAdapter) and 
                    obj != BaseAdapter):
                    # Try to register if not already registered
                    model_type = adapter_file.stem.replace("_adapter", "")
                    if model_type not in ModelRegistry()._registry:
                        ModelRegistry().register(
                            name=model_type,
                            model_type=model_type,
                            config_class=AutoConfig.for_model(model_type),
                            model_class=AutoModelForCausalLM,
                            adapter_class=obj
                        )
        except Exception as e:
            logger.debug(f"Could not auto-discover adapter from {adapter_file}: {e}")


# Initialize the registry and auto-discover adapters
registry = ModelRegistry()
_auto_discover_adapters()

# Export public API
__all__ = [
    "ModelRegistry",
    "BaseAdapter",
    "ModelConfig",
    "DefaultAdapter",
    "register_model",
    "load_model",
    "convert_checkpoint",
    "registry",
]