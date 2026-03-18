#!/usr/bin/env python3
"""
Architecture Registry for forge Model Conversion
Unified converter with auto-detection and transformation registry
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from tqdm import tqdm
import importlib.util

logger = logging.getLogger(__name__)

class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"

@dataclass
class ConversionConfig:
    """Configuration for model conversion"""
    source_path: str
    target_path: str
    target_format: ModelFormat = ModelFormat.SAFETENSORS
    target_architecture: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    validate: bool = True
    progress: bool = True

@dataclass
class ArchitectureInfo:
    """Information about a model architecture"""
    name: str
    config_keys: List[str] = field(default_factory=list)
    conversion_func: Optional[Callable] = None
    validator: Optional[Callable] = None
    description: str = ""

class ArchitectureRegistry:
    """Registry for model architectures and their conversion functions"""
    
    _instance = None
    _registry: Dict[str, ArchitectureInfo] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the registry with built-in architectures"""
        self._register_builtin_architectures()
    
    def _register_builtin_architectures(self):
        """Register built-in architecture converters"""
        # Llama family
        self.register(
            name="llama",
            config_keys=["architectures", "hidden_size", "intermediate_size"],
            conversion_func=self._convert_to_llama,
            validator=self._validate_llama,
            description="Llama/Llama2 architecture"
        )
        
        # Baichuan2
        self.register(
            name="baichuan2",
            config_keys=["architectures", "hidden_size", "num_attention_heads"],
            conversion_func=self._convert_baichuan2,
            validator=self._validate_baichuan2,
            description="Baichuan2 architecture"
        )
        
        # Qwen
        self.register(
            name="qwen",
            config_keys=["architectures", "hidden_size", "num_attention_heads", "kv_channels"],
            conversion_func=self._convert_qwen,
            validator=self._validate_qwen,
            description="Qwen architecture"
        )
        
        # Mistral
        self.register(
            name="mistral",
            config_keys=["architectures", "hidden_size", "num_attention_heads", "sliding_window"],
            conversion_func=self._convert_mistral,
            validator=self._validate_mistral,
            description="Mistral architecture"
        )
        
        # Yi
        self.register(
            name="yi",
            config_keys=["architectures", "hidden_size", "num_attention_heads", "multi_query_group_num"],
            conversion_func=self._convert_yi,
            validator=self._validate_yi,
            description="Yi architecture"
        )
        
        # Phi
        self.register(
            name="phi",
            config_keys=["architectures", "hidden_size", "num_attention_heads", "partial_rotary_factor"],
            conversion_func=self._convert_phi,
            validator=self._validate_phi,
            description="Phi architecture"
        )
    
    def register(self, 
                 name: str,
                 config_keys: List[str],
                 conversion_func: Callable,
                 validator: Optional[Callable] = None,
                 description: str = ""):
        """Register a new architecture"""
        if name in self._registry:
            logger.warning(f"Architecture {name} already registered, overwriting")
        
        self._registry[name] = ArchitectureInfo(
            name=name,
            config_keys=config_keys,
            conversion_func=conversion_func,
            validator=validator,
            description=description
        )
        logger.info(f"Registered architecture: {name}")
    
    def detect_architecture(self, config: Dict[str, Any]) -> Optional[str]:
        """Auto-detect model architecture from config"""
        if not config:
            return None
        
        # Check for explicit architecture in config
        if "architectures" in config:
            architectures = config["architectures"]
            if isinstance(architectures, list) and architectures:
                arch_name = architectures[0].lower()
                
                # Direct mapping
                arch_map = {
                    "llamaforcausallm": "llama",
                    "baichuan2forcausallm": "baichuan2",
                    "qwenlmheadmodel": "qwen",
                    "mistralforcausallm": "mistral",
                    "yiforcausallm": "yi",
                    "phiforcausallm": "phi",
                }
                
                for key, value in arch_map.items():
                    if key in arch_name:
                        return value
        
        # Pattern matching based on config keys
        for arch_name, arch_info in self._registry.items():
            if self._matches_config_pattern(config, arch_info.config_keys):
                return arch_name
        
        return None
    
    def _matches_config_pattern(self, config: Dict[str, Any], pattern_keys: List[str]) -> bool:
        """Check if config matches pattern keys"""
        matches = 0
        for key in pattern_keys:
            if key in config:
                matches += 1
        return matches >= len(pattern_keys) * 0.7  # 70% match threshold
    
    def get_converter(self, architecture: str) -> Optional[Callable]:
        """Get conversion function for architecture"""
        arch_info = self._registry.get(architecture)
        if arch_info:
            return arch_info.conversion_func
        return None
    
    def get_validator(self, architecture: str) -> Optional[Callable]:
        """Get validation function for architecture"""
        arch_info = self._registry.get(architecture)
        if arch_info:
            return arch_info.validator
        return None
    
    def list_architectures(self) -> List[str]:
        """List all registered architectures"""
        return list(self._registry.keys())
    
    # Built-in conversion functions
    def _convert_to_llama(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert to Llama architecture"""
        # This is a simplified version - actual implementation would be more complex
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Example transformation patterns
            if "attention" in key:
                key = key.replace("attention", "self_attn")
            if "mlp" in key:
                key = key.replace("mlp", "feed_forward")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _convert_baichuan2(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert Baichuan2 to Llama format"""
        # Import existing conversion logic
        try:
            spec = importlib.util.spec_from_file_location(
                "llamafy_baichuan2",
                Path(__file__).parent / "llamafy_baichuan2.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Use existing conversion function
            return module.convert_state_dict(state_dict, config)
        except Exception as e:
            logger.error(f"Failed to import Baichuan2 converter: {e}")
            raise
    
    def _convert_qwen(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert Qwen to Llama format"""
        try:
            spec = importlib.util.spec_from_file_location(
                "llamafy_qwen",
                Path(__file__).parent / "llamafy_qwen.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module.convert_state_dict(state_dict, config)
        except Exception as e:
            logger.error(f"Failed to import Qwen converter: {e}")
            raise
    
    def _convert_mistral(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert Mistral to Llama format"""
        # Mistral is already very similar to Llama, minimal conversion needed
        return state_dict
    
    def _convert_yi(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert Yi to Llama format"""
        # Yi conversion logic
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Yi-specific transformations
            if "query_key_value" in key:
                # Split QKV into separate components
                pass
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _convert_phi(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert Phi to Llama format"""
        # Phi conversion logic
        return state_dict
    
    # Validation functions
    def _validate_llama(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Llama conversion"""
        required_keys = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]
        return all(key in state_dict for key in required_keys)
    
    def _validate_baichuan2(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Baichuan2 conversion"""
        return self._validate_llama(state_dict, config)
    
    def _validate_qwen(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Qwen conversion"""
        return self._validate_llama(state_dict, config)
    
    def _validate_mistral(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Mistral conversion"""
        return self._validate_llama(state_dict, config)
    
    def _validate_yi(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Yi conversion"""
        return self._validate_llama(state_dict, config)
    
    def _validate_phi(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> bool:
        """Validate Phi conversion"""
        return self._validate_llama(state_dict, config)

class ModelConverter:
    """Generic model converter with auto-detection"""
    
    def __init__(self, registry: Optional[ArchitectureRegistry] = None):
        self.registry = registry or ArchitectureRegistry()
        self._safetensors_available = self._check_safetensors()
    
    def _check_safetensors(self) -> bool:
        """Check if safetensors is available"""
        try:
            import safetensors
            return True
        except ImportError:
            logger.warning("safetensors not available, falling back to PyTorch format")
            return False
    
    def load_model(self, model_path: str, device: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load model from path"""
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load model weights
        state_dict = {}
        
        # Try safetensors first
        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists() and self._safetensors_available:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path), device=device)
        else:
            # Fall back to PyTorch bin files
            bin_files = list(model_path.glob("*.bin"))
            if not bin_files:
                raise FileNotFoundError(f"No model files found in {model_path}")
            
            for bin_file in tqdm(bin_files, desc="Loading PyTorch files", disable=not self.progress):
                loaded = torch.load(bin_file, map_location=device)
                state_dict.update(loaded)
        
        return state_dict, config
    
    def save_model(self, 
                   state_dict: Dict[str, torch.Tensor], 
                   save_path: str, 
                   format: ModelFormat = ModelFormat.SAFETENSORS,
                   config: Optional[Dict[str, Any]] = None,
                   progress: bool = True):
        """Save model to path"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config if provided
        if config:
            config_path = save_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        
        # Save model weights
        if format == ModelFormat.SAFETENSORS and self._safetensors_available:
            from safetensors.torch import save_file
            safetensors_path = save_path / "model.safetensors"
            
            # Convert tensors to float16 for efficiency if dtype not specified
            save_dict = {}
            for key, tensor in tqdm(state_dict.items(), desc="Preparing tensors", disable=not progress):
                if tensor.dtype == torch.float32:
                    save_dict[key] = tensor.half()
                else:
                    save_dict[key] = tensor
            
            save_file(save_dict, str(safetensors_path))
            logger.info(f"Saved model in safetensors format to {safetensors_path}")
        else:
            # Save as PyTorch bin files (sharded if large)
            total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
            max_shard_size = 2 * 1024 * 1024 * 1024  # 2GB per shard
            
            if total_size > max_shard_size:
                self._save_sharded(state_dict, save_path, max_shard_size, progress)
            else:
                torch.save(state_dict, save_path / "pytorch_model.bin")
                logger.info(f"Saved model in PyTorch format to {save_path / 'pytorch_model.bin'}")
    
    def _save_sharded(self, 
                      state_dict: Dict[str, torch.Tensor], 
                      save_path: Path, 
                      max_shard_size: int,
                      progress: bool):
        """Save model in sharded format"""
        current_shard = {}
        current_size = 0
        shard_idx = 0
        
        for key, tensor in tqdm(state_dict.items(), desc="Saving shards", disable=not progress):
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_shard_size and current_shard:
                # Save current shard
                shard_path = save_path / f"pytorch_model-{shard_idx:05d}.bin"
                torch.save(current_shard, shard_path)
                logger.info(f"Saved shard {shard_idx} to {shard_path}")
                
                # Start new shard
                current_shard = {}
                current_size = 0
                shard_idx += 1
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # Save final shard
        if current_shard:
            shard_path = save_path / f"pytorch_model-{shard_idx:05d}.bin"
            torch.save(current_shard, shard_path)
            logger.info(f"Saved final shard {shard_idx} to {shard_path}")
        
        # Save shard index
        shard_index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
            "weight_map": {key: f"pytorch_model-{i:05d}.bin" 
                          for i, key in enumerate(state_dict.keys())}
        }
        with open(save_path / "pytorch_model.bin.index.json", "w") as f:
            json.dump(shard_index, f, indent=2)
    
    def convert(self, config: ConversionConfig) -> bool:
        """Convert model according to configuration"""
        try:
            logger.info(f"Loading model from {config.source_path}")
            state_dict, model_config = self.load_model(config.source_path, config.device)
            
            # Detect architecture
            if config.target_architecture:
                architecture = config.target_architecture
                logger.info(f"Using specified architecture: {architecture}")
            else:
                architecture = self.registry.detect_architecture(model_config)
                if not architecture:
                    logger.error("Could not detect model architecture")
                    return False
                logger.info(f"Detected architecture: {architecture}")
            
            # Get converter
            converter = self.registry.get_converter(architecture)
            if not converter:
                logger.error(f"No converter found for architecture: {architecture}")
                return False
            
            # Apply conversion
            logger.info("Converting model...")
            converted_state_dict = converter(state_dict, model_config)
            
            # Apply dtype conversion if specified
            if config.dtype:
                converted_state_dict = {
                    k: v.to(config.dtype) if v.dtype.is_floating_point else v
                    for k, v in converted_state_dict.items()
                }
            
            # Validate conversion
            if config.validate:
                validator = self.registry.get_validator(architecture)
                if validator:
                    if not validator(converted_state_dict, model_config):
                        logger.warning("Model validation failed")
                    else:
                        logger.info("Model validation passed")
            
            # Save converted model
            logger.info(f"Saving converted model to {config.target_path}")
            self.save_model(
                converted_state_dict,
                config.target_path,
                config.target_format,
                model_config,
                config.progress
            )
            
            logger.info("Conversion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Command-line interface for the converter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Model Converter")
    parser.add_argument("source", help="Path to source model directory")
    parser.add_argument("target", help="Path to save converted model")
    parser.add_argument("--format", choices=["safetensors", "pytorch"], default="safetensors",
                       help="Output format (default: safetensors)")
    parser.add_argument("--architecture", help="Target architecture (auto-detect if not specified)")
    parser.add_argument("--device", default="cpu", help="Device to use for conversion")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], 
                       help="Target dtype for conversion")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars")
    parser.add_argument("--list-architectures", action="store_true", 
                       help="List available architectures and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create registry and converter
    registry = ArchitectureRegistry()
    converter = ModelConverter(registry)
    
    if args.list_architectures:
        print("Available architectures:")
        for arch in registry.list_architectures():
            arch_info = registry._registry[arch]
            print(f"  - {arch}: {arch_info.description}")
        return
    
    # Map dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    # Create conversion config
    config = ConversionConfig(
        source_path=args.source,
        target_path=args.target,
        target_format=ModelFormat(args.format),
        target_architecture=args.architecture,
        device=args.device,
        dtype=dtype_map.get(args.dtype),
        validate=not args.no_validate,
        progress=not args.no_progress
    )
    
    # Perform conversion
    success = converter.convert(config)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()