"""
forge/adapters/base.py
Unified Model Registry & Dynamic Adapter System
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Central model registry
MODEL_REGISTRY: Dict[str, Type["BaseAdapter"]] = {}


@dataclass
class AdapterConfig:
    """Configuration for model adapters."""
    model_name_or_path: str
    adapter_name: str = "default"
    adapter_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: List[str] = field(default_factory=list)
    special_tokens: Dict[str, str] = field(default_factory=dict)
    torch_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    padding_side: str = "right"
    model_max_length: int = 2048
    config_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Base adapter interface for unified model handling."""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self._special_tokens_map: Dict[str, str] = {}
        
    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """Load the base model with architecture-specific handling."""
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer with special token handling."""
        pass
    
    @abstractmethod
    def get_target_modules(self) -> List[str]:
        """Get LoRA target modules for this architecture."""
        pass
    
    def setup_special_tokens(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """Configure special tokens for the model."""
        special_tokens = self.config.special_tokens.copy()
        
        # Add architecture-specific special tokens
        arch_tokens = self._get_architecture_special_tokens()
        special_tokens.update(arch_tokens)
        
        # Update tokenizer with special tokens
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            
        # Set padding side
        tokenizer.padding_side = self.config.padding_side
        
        # Set model max length
        if hasattr(tokenizer, 'model_max_length'):
            tokenizer.model_max_length = self.config.model_max_length
            
        return tokenizer
    
    def _get_architecture_special_tokens(self) -> Dict[str, str]:
        """Get architecture-specific special tokens."""
        return {}
    
    def apply_lora(self, model: PreTrainedModel) -> PeftModel:
        """Apply LoRA adapters to the model."""
        target_modules = self.get_target_modules()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            modules_to_save=self.config.modules_to_save,
        )
        
        logger.info(f"Applying LoRA with rank {self.config.lora_rank} to modules: {target_modules}")
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def load_adapter(self, adapter_path: str) -> PeftModel:
        """Load a pre-trained adapter."""
        if self.peft_model is None:
            raise ValueError("Base model must be loaded before loading adapter")
            
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=self.config.torch_dtype,
        )
        return self.peft_model
    
    def merge_adapter(self) -> PreTrainedModel:
        """Merge adapter weights into base model."""
        if self.peft_model is None:
            raise ValueError("No adapter to merge")
            
        merged_model = self.peft_model.merge_and_unload()
        self.model = merged_model
        self.peft_model = None
        return merged_model
    
    def save_adapter(self, save_path: str):
        """Save adapter weights."""
        if self.peft_model is None:
            raise ValueError("No adapter to save")
            
        self.peft_model.save_pretrained(save_path)
        logger.info(f"Adapter saved to {save_path}")
    
    def prepare_for_training(self):
        """Prepare model for training."""
        if self.model is None:
            self.load_model()
        if self.tokenizer is None:
            self.load_tokenizer()
            
        # Resize embeddings if tokenizer was updated
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        # Apply LoRA if specified
        if self.config.adapter_type == "lora" and self.peft_model is None:
            self.peft_model = self.apply_lora(self.model)
    
    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Get information about trainable parameters."""
        if self.peft_model is not None:
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.peft_model.parameters())
        elif self.model is not None:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
        else:
            return {}
            
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": 100 * trainable_params / all_params if all_params > 0 else 0
        }


def register_adapter(name: str):
    """Decorator to register adapter classes in the global registry."""
    def decorator(adapter_class: Type[BaseAdapter]):
        if name in MODEL_REGISTRY:
            logger.warning(f"Adapter {name} already registered, overwriting")
        MODEL_REGISTRY[name] = adapter_class
        return adapter_class
    return decorator


def get_adapter_class(model_name_or_path: str) -> Type[BaseAdapter]:
    """Get adapter class for a model, falling back to auto-detection."""
    # First check explicit registry
    if model_name_or_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_path]
    
    # Try to detect from model config
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        model_type = getattr(config, "model_type", "").lower()
        
        if model_type in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_type]
        
        # Try partial matching
        for registered_name, adapter_class in MODEL_REGISTRY.items():
            if registered_name.lower() in model_type or model_type in registered_name.lower():
                logger.info(f"Using adapter {registered_name} for model type {model_type}")
                return adapter_class
                
    except Exception as e:
        logger.warning(f"Could not detect model type from config: {e}")
    
    # Default to generic adapter
    if "generic" in MODEL_REGISTRY:
        return MODEL_REGISTRY["generic"]
    
    raise ValueError(
        f"No adapter found for {model_name_or_path}. "
        f"Available adapters: {list(MODEL_REGISTRY.keys())}"
    )


def load_adapter_from_config(config: AdapterConfig) -> BaseAdapter:
    """Load an adapter instance from configuration."""
    adapter_class = get_adapter_class(config.model_name_or_path)
    return adapter_class(config)


def discover_adapters(adapters_dir: Optional[Union[str, Path]] = None):
    """Discover and load adapter plugins from directory."""
    if adapters_dir is None:
        # Default to adapters directory relative to this file
        adapters_dir = Path(__file__).parent
    
    adapters_dir = Path(adapters_dir)
    
    if not adapters_dir.exists():
        logger.warning(f"Adapters directory {adapters_dir} does not exist")
        return
    
    for adapter_file in adapters_dir.glob("*.py"):
        if adapter_file.name.startswith("_"):
            continue
            
        module_name = f"forge.adapters.{adapter_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, adapter_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all adapter classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAdapter) and 
                    obj != BaseAdapter):
                    # Auto-register if not already registered
                    adapter_name = getattr(obj, "ADAPTER_NAME", adapter_file.stem)
                    if adapter_name not in MODEL_REGISTRY:
                        MODEL_REGISTRY[adapter_name] = obj
                        logger.info(f"Discovered adapter: {adapter_name}")
                        
        except Exception as e:
            logger.error(f"Failed to load adapter from {adapter_file}: {e}")


# Auto-discover adapters on module import
discover_adapters()


class GenericAdapter(BaseAdapter):
    """Generic adapter for models without specific implementation."""
    
    ADAPTER_NAME = "generic"
    
    def load_model(self) -> PreTrainedModel:
        logger.info(f"Loading model {self.config.model_name_or_path}")
        
        model_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            **self.config.model_kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        
        return self.model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        logger.info(f"Loading tokenizer for {self.config.model_name_or_path}")
        
        tokenizer_kwargs = {
            "use_fast": self.config.use_fast_tokenizer,
            "trust_remote_code": self.config.trust_remote_code,
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            **tokenizer_kwargs
        )
        
        self.tokenizer = self.setup_special_tokens(self.tokenizer)
        
        return self.tokenizer
    
    def get_target_modules(self) -> List[str]:
        """Try to detect target modules from model architecture."""
        if self.model is None:
            return self.config.target_modules
            
        # Common module patterns for different architectures
        module_patterns = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "qwen": ["c_attn", "c_proj"],
            "baichuan": ["W_pack", "o_proj"],
            "chatglm": ["query_key_value", "dense"],
            "bloom": ["query_key_value", "dense"],
            "gpt_neox": ["query_key_value", "dense"],
            "falcon": ["query_key_value", "dense"],
        }
        
        model_type = getattr(self.model.config, "model_type", "").lower()
        
        for arch, modules in module_patterns.items():
            if arch in model_type:
                # Verify modules exist in model
                available_modules = []
                for name, module in self.model.named_modules():
                    for target in modules:
                        if target in name:
                            available_modules.append(target)
                            break
                if available_modules:
                    return list(set(available_modules))
        
        # Fallback to config
        return self.config.target_modules
    
    def _get_architecture_special_tokens(self) -> Dict[str, str]:
        """Get special tokens based on model architecture."""
        if self.model is None:
            return {}
            
        special_tokens = {}
        model_type = getattr(self.model.config, "model_type", "").lower()
        
        # Architecture-specific special tokens
        if "llama" in model_type:
            special_tokens.update({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            })
        elif "qwen" in model_type:
            special_tokens.update({
                "bos_token": "",
                "eos_token": "",
                "pad_token": "",
            })
        elif "baichuan" in model_type:
            special_tokens.update({
                "bos_token": "<s>",
                "eos_token": "</s>",
            })
            
        return special_tokens


# Register generic adapter
MODEL_REGISTRY["generic"] = GenericAdapter


# Convenience functions for backward compatibility
def get_model_and_tokenizer(
    model_name_or_path: str,
    adapter_config: Optional[AdapterConfig] = None,
    **kwargs
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Get model and tokenizer using the adapter system."""
    if adapter_config is None:
        adapter_config = AdapterConfig(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    
    adapter = load_adapter_from_config(adapter_config)
    adapter.load_model()
    adapter.load_tokenizer()
    
    return adapter.model, adapter.tokenizer


def apply_lora_to_model(
    model: PreTrainedModel,
    adapter_config: AdapterConfig,
) -> PeftModel:
    """Apply LoRA to an existing model."""
    adapter = GenericAdapter(adapter_config)
    adapter.model = model
    return adapter.apply_lora(model)


# Example adapter registration for specific models
@register_adapter("llama")
class LlamaAdapter(BaseAdapter):
    """Adapter for LLaMA models."""
    
    ADAPTER_NAME = "llama"
    
    def load_model(self) -> PreTrainedModel:
        from transformers import LlamaForCausalLM
        
        model_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": self.config.device_map,
            **self.config.model_kwargs
        }
        
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        return self.model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        from transformers import LlamaTokenizer
        
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
        )
        self.tokenizer = self.setup_special_tokens(self.tokenizer)
        return self.tokenizer
    
    def get_target_modules(self) -> List[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _get_architecture_special_tokens(self) -> Dict[str, str]:
        return {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }


@register_adapter("mistral")
class MistralAdapter(LlamaAdapter):
    """Adapter for Mistral models (inherits from LLaMA)."""
    ADAPTER_NAME = "mistral"


@register_adapter("qwen")
class QwenAdapter(BaseAdapter):
    """Adapter for Qwen models."""
    
    ADAPTER_NAME = "qwen"
    
    def load_model(self) -> PreTrainedModel:
        from transformers import AutoModelForCausalLM
        
        model_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            **self.config.model_kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        return self.model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=True,
        )
        self.tokenizer = self.setup_special_tokens(self.tokenizer)
        return self.tokenizer
    
    def get_target_modules(self) -> List[str]:
        return ["c_attn", "c_proj"]
    
    def _get_architecture_special_tokens(self) -> Dict[str, str]:
        return {
            "bos_token": "",
            "eos_token": "",
            "pad_token": "",
        }


@register_adapter("baichuan")
class BaichuanAdapter(BaseAdapter):
    """Adapter for Baichuan models."""
    
    ADAPTER_NAME = "baichuan"
    
    def load_model(self) -> PreTrainedModel:
        from transformers import AutoModelForCausalLM
        
        model_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            **self.config.model_kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        return self.model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=True,
        )
        self.tokenizer = self.setup_special_tokens(self.tokenizer)
        return self.tokenizer
    
    def get_target_modules(self) -> List[str]:
        return ["W_pack", "o_proj"]
    
    def _get_architecture_special_tokens(self) -> Dict[str, str]:
        return {
            "bos_token": "<s>",
            "eos_token": "</s>",
        }


# Integration with existing conversion scripts
def migrate_conversion_script(script_path: str) -> Type[BaseAdapter]:
    """Helper to migrate existing conversion scripts to adapters."""
    # This would parse existing conversion scripts and create adapter classes
    # For now, we'll just log a warning
    logger.warning(
        f"Conversion script {script_path} should be migrated to an adapter. "
        "See documentation for details."
    )
    return GenericAdapter


if __name__ == "__main__":
    # Example usage
    config = AdapterConfig(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        lora_rank=16,
        lora_alpha=32,
    )
    
    adapter = load_adapter_from_config(config)
    adapter.prepare_for_training()
    
    print(f"Loaded model: {type(adapter.model).__name__}")
    print(f"Trainable parameters: {adapter.get_trainable_parameters()}")