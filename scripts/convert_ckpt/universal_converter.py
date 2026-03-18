#!/usr/bin/env python3
"""
Universal Model Converter for forge
Auto-detects model architecture and applies appropriate transformations
Replaces model-specific conversion scripts with unified converter
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
from tqdm import tqdm
import gc

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import is_safetensors_available
    if is_safetensors_available():
        from safetensors.torch import load_file, save_file
    SAFETENSORS_AVAILABLE = is_safetensors_available()
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: transformers or safetensors not installed. Some features may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Supported model architectures"""
    LLAMA = "llama"
    BAICHUAN = "baichuan"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    CHATGLM = "chatglm"
    INTERNLM = "internlm"
    YI = "yi"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    PHI = "phi"
    GEMMA = "gemma"
    UNKNOWN = "unknown"


@dataclass
class ConversionConfig:
    """Configuration for model conversion"""
    input_dir: str
    output_dir: str
    output_format: str = "safetensors"  # safetensors or pytorch
    shard_size: Optional[int] = None  # In bytes, None for no sharding
    safe_serialization: bool = True
    max_shard_size: str = "2GB"
    device: str = "cpu"
    dtype: Optional[str] = None  # float16, bfloat16, float32
    trust_remote_code: bool = False
    verbose: bool = False


class TransformationRegistry:
    """Registry for model-specific transformations"""
    
    def __init__(self):
        self._transformations: Dict[ModelArchitecture, Dict[str, Callable]] = {}
        self._register_default_transformations()
    
    def _register_default_transformations(self):
        """Register default transformation functions"""
        # Llama architecture transformations
        self.register(
            ModelArchitecture.LLAMA,
            "weight_mapping",
            self._llama_weight_mapping
        )
        
        # Baichuan architecture transformations
        self.register(
            ModelArchitecture.BAICHUAN,
            "weight_mapping",
            self._baichuan_weight_mapping
        )
        self.register(
            ModelArchitecture.BAICHUAN,
            "config_mapping",
            self._baichuan_config_mapping
        )
        
        # Qwen architecture transformations
        self.register(
            ModelArchitecture.QWEN,
            "weight_mapping",
            self._qwen_weight_mapping
        )
        self.register(
            ModelArchitecture.QWEN,
            "config_mapping",
            self._qwen_config_mapping
        )
        
        # Qwen2 architecture transformations
        self.register(
            ModelArchitecture.QWEN2,
            "weight_mapping",
            self._qwen2_weight_mapping
        )
        
        # ChatGLM architecture transformations
        self.register(
            ModelArchitecture.CHATGLM,
            "weight_mapping",
            self._chatglm_weight_mapping
        )
        
        # InternLM architecture transformations
        self.register(
            ModelArchitecture.INTERNLM,
            "weight_mapping",
            self._internlm_weight_mapping
        )
        
        # Yi architecture transformations
        self.register(
            ModelArchitecture.YI,
            "weight_mapping",
            self._yi_weight_mapping
        )
        
        # Mistral/Mixtral architecture transformations
        self.register(
            ModelArchitecture.MISTRAL,
            "weight_mapping",
            self._mistral_weight_mapping
        )
        self.register(
            ModelArchitecture.MIXTRAL,
            "weight_mapping",
            self._mixtral_weight_mapping
        )
        
        # Phi architecture transformations
        self.register(
            ModelArchitecture.PHI,
            "weight_mapping",
            self._phi_weight_mapping
        )
        
        # Gemma architecture transformations
        self.register(
            ModelArchitecture.GEMMA,
            "weight_mapping",
            self._gemma_weight_mapping
        )
    
    def register(self, architecture: ModelArchitecture, transformation_type: str, func: Callable):
        """Register a transformation function for an architecture"""
        if architecture not in self._transformations:
            self._transformations[architecture] = {}
        self._transformations[architecture][transformation_type] = func
    
    def get_transformation(self, architecture: ModelArchitecture, transformation_type: str) -> Optional[Callable]:
        """Get transformation function for architecture"""
        return self._transformations.get(architecture, {}).get(transformation_type)
    
    def has_architecture(self, architecture: ModelArchitecture) -> bool:
        """Check if architecture is supported"""
        return architecture in self._transformations
    
    # Default transformation implementations
    def _llama_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Llama weight mapping (identity transformation)"""
        return state_dict
    
    def _baichuan_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Baichuan weights to Llama format"""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Transform attention layer names
            if "self_attn" in key:
                key = key.replace("self_attn.W_pack", "self_attn.q_proj")
                key = key.replace("self_attn.o_proj", "self_attn.o_proj")
            
            # Transform MLP layer names
            elif "mlp" in key:
                key = key.replace("mlp.gate_up_proj", "mlp.gate_proj")
                key = key.replace("mlp.down_proj", "mlp.down_proj")
            
            # Transform embedding layer names
            elif "embed_tokens" in key:
                key = key.replace("embed_tokens", "model.embed_tokens")
            
            # Transform norm layer names
            elif "input_layernorm" in key:
                key = key.replace("input_layernorm", "input_layernorm")
            elif "post_attention_layernorm" in key:
                key = key.replace("post_attention_layernorm", "post_attention_layernorm")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _baichuan_config_mapping(self, config: Dict) -> Dict:
        """Transform Baichuan config to Llama format"""
        new_config = config.copy()
        new_config["architectures"] = ["LlamaForCausalLM"]
        new_config["model_type"] = "llama"
        
        # Map Baichuan-specific parameters
        if "hidden_size" in config:
            new_config["hidden_size"] = config["hidden_size"]
        if "num_attention_heads" in config:
            new_config["num_attention_heads"] = config["num_attention_heads"]
        if "num_hidden_layers" in config:
            new_config["num_hidden_layers"] = config["num_hidden_layers"]
        if "intermediate_size" in config:
            new_config["intermediate_size"] = config["intermediate_size"]
        
        return new_config
    
    def _qwen_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Qwen weights to Llama format"""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Qwen uses different naming conventions
            if "transformer.h" in key:
                key = key.replace("transformer.h", "model.layers")
            elif "transformer.wte" in key:
                key = key.replace("transformer.wte", "model.embed_tokens")
            elif "transformer.ln_f" in key:
                key = key.replace("transformer.ln_f", "model.norm")
            
            # Attention transformations
            if "attn" in key:
                key = key.replace("attn.c_attn", "self_attn.qkv_proj")
                key = key.replace("attn.c_proj", "self_attn.o_proj")
            
            # MLP transformations
            if "mlp" in key:
                key = key.replace("mlp.w1", "mlp.gate_proj")
                key = key.replace("mlp.w2", "mlp.down_proj")
                key = key.replace("mlp.c_proj", "mlp.up_proj")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _qwen_config_mapping(self, config: Dict) -> Dict:
        """Transform Qwen config to Llama format"""
        new_config = config.copy()
        new_config["architectures"] = ["LlamaForCausalLM"]
        new_config["model_type"] = "llama"
        
        # Map Qwen-specific parameters
        if "hidden_size" in config:
            new_config["hidden_size"] = config["hidden_size"]
        if "num_attention_heads" in config:
            new_config["num_attention_heads"] = config["num_attention_heads"]
        if "num_hidden_layers" in config:
            new_config["num_hidden_layers"] = config["num_hidden_layers"]
        
        return new_config
    
    def _qwen2_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Qwen2 weights to Llama format"""
        # Qwen2 is already similar to Llama, minimal transformation needed
        return state_dict
    
    def _chatglm_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform ChatGLM weights to Llama format"""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # ChatGLM has different layer structure
            if "transformer.encoder" in key:
                key = key.replace("transformer.encoder", "model")
            if "embedding" in key:
                key = key.replace("embedding.word_embeddings", "embed_tokens")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _internlm_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform InternLM weights to Llama format"""
        # InternLM is already similar to Llama
        return state_dict
    
    def _yi_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Yi weights to Llama format"""
        # Yi is already similar to Llama
        return state_dict
    
    def _mistral_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Mistral weights to Llama format"""
        # Mistral is already similar to Llama
        return state_dict
    
    def _mixtral_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Mixtral weights to Llama format"""
        # Mixtral has MoE layers, need special handling
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Handle MoE layers
            if "block_sparse_moe" in key:
                # Convert Mixtral MoE to Llama format (simplified)
                key = key.replace("block_sparse_moe", "mlp")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _phi_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Phi weights to Llama format"""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Phi has different naming
            if "transformer" in key:
                key = key.replace("transformer", "model")
            if "layers" in key:
                key = key.replace("layers", "layers")
            
            new_state_dict[key] = value
        
        return new_state_dict
    
    def _gemma_weight_mapping(self, state_dict: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
        """Transform Gemma weights to Llama format"""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Gemma has different naming conventions
            if "model.layers" in key:
                # Already in correct format
                pass
            elif "embed_tokens" in key:
                key = key.replace("embed_tokens", "model.embed_tokens")
            
            new_state_dict[key] = value
        
        return new_state_dict


class ArchitectureDetector:
    """Detects model architecture from config"""
    
    # Architecture detection patterns
    ARCHITECTURE_PATTERNS = {
        ModelArchitecture.LLAMA: ["llama", "LlamaForCausalLM"],
        ModelArchitecture.BAICHUAN: ["baichuan", "BaichuanForCausalLM", "Baichuan2ForCausalLM"],
        ModelArchitecture.QWEN: ["qwen", "QWenLMHeadModel", "QwenForCausalLM"],
        ModelArchitecture.QWEN2: ["qwen2", "Qwen2ForCausalLM"],
        ModelArchitecture.CHATGLM: ["chatglm", "ChatGLMForCausalLM", "ChatGLMModel"],
        ModelArchitecture.INTERNLM: ["internlm", "InternLMForCausalLM"],
        ModelArchitecture.YI: ["yi", "YiForCausalLM"],
        ModelArchitecture.MISTRAL: ["mistral", "MistralForCausalLM"],
        ModelArchitecture.MIXTRAL: ["mixtral", "MixtralForCausalLM"],
        ModelArchitecture.PHI: ["phi", "PhiForCausalLM"],
        ModelArchitecture.GEMMA: ["gemma", "GemmaForCausalLM"],
    }
    
    @classmethod
    def detect(cls, config: Dict) -> ModelArchitecture:
        """Detect model architecture from config"""
        # Check model_type field
        model_type = config.get("model_type", "").lower()
        for arch, patterns in cls.ARCHITECTURE_PATTERNS.items():
            if any(pattern.lower() in model_type for pattern in patterns):
                return arch
        
        # Check architectures field
        architectures = config.get("architectures", [])
        if architectures:
            arch_name = architectures[0].lower()
            for arch, patterns in cls.ARCHITECTURE_PATTERNS.items():
                if any(pattern.lower() in arch_name for pattern in patterns):
                    return arch
        
        # Check for specific config keys
        if "num_key_value_heads" in config:
            return ModelArchitecture.LLAMA
        if "multi_query_attention" in config:
            return ModelArchitecture.QWEN
        
        return ModelArchitecture.UNKNOWN
    
    @classmethod
    def is_supported(cls, architecture: ModelArchitecture) -> bool:
        """Check if architecture is supported for conversion"""
        return architecture != ModelArchitecture.UNKNOWN


class UniversalConverter:
    """Universal model converter with auto-detection"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.registry = TransformationRegistry()
        self.detector = ArchitectureDetector()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def convert(self) -> bool:
        """Main conversion method"""
        try:
            logger.info(f"Starting conversion from {self.config.input_dir} to {self.config.output_dir}")
            
            # Step 1: Load and analyze config
            input_config = self._load_config()
            if not input_config:
                logger.error("Failed to load input config")
                return False
            
            # Step 2: Detect architecture
            architecture = self.detector.detect(input_config)
            logger.info(f"Detected architecture: {architecture.value}")
            
            if not self.detector.is_supported(architecture):
                logger.error(f"Unsupported architecture: {architecture.value}")
                return False
            
            if not self.registry.has_architecture(architecture):
                logger.error(f"No transformations registered for architecture: {architecture.value}")
                return False
            
            # Step 3: Convert config
            output_config = self._convert_config(input_config, architecture)
            self._save_config(output_config)
            
            # Step 4: Convert model weights
            success = self._convert_weights(architecture, input_config)
            if not success:
                logger.error("Failed to convert model weights")
                return False
            
            # Step 5: Copy tokenizer and other files
            self._copy_additional_files()
            
            logger.info("Conversion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _load_config(self) -> Optional[Dict]:
        """Load model config from input directory"""
        config_path = os.path.join(self.config.input_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return None
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return None
    
    def _convert_config(self, input_config: Dict, architecture: ModelArchitecture) -> Dict:
        """Convert config to target format"""
        # Get config transformation
        config_transform = self.registry.get_transformation(architecture, "config_mapping")
        if config_transform:
            output_config = config_transform(input_config)
        else:
            # Default: keep config as is but update model_type
            output_config = input_config.copy()
            output_config["model_type"] = "llama"
            output_config["architectures"] = ["LlamaForCausalLM"]
        
        # Ensure required fields
        if "torch_dtype" in output_config and self.config.dtype:
            dtype_map = {
                "float16": "float16",
                "bfloat16": "bfloat16",
                "float32": "float32"
            }
            if self.config.dtype in dtype_map:
                output_config["torch_dtype"] = dtype_map[self.config.dtype]
        
        return output_config
    
    def _save_config(self, config: Dict):
        """Save converted config"""
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved config to {config_path}")
    
    def _convert_weights(self, architecture: ModelArchitecture, config: Dict) -> bool:
        """Convert model weights"""
        # Find model weight files
        weight_files = self._find_weight_files()
        if not weight_files:
            logger.error("No model weight files found")
            return False
        
        logger.info(f"Found {len(weight_files)} weight file(s)")
        
        # Process each weight file
        all_state_dicts = []
        for weight_file in tqdm(weight_files, desc="Loading weight files"):
            state_dict = self._load_weight_file(weight_file)
            if state_dict is None:
                return False
            
            # Apply transformation
            weight_transform = self.registry.get_transformation(architecture, "weight_mapping")
            if weight_transform:
                state_dict = weight_transform(state_dict, config)
            
            all_state_dicts.append(state_dict)
        
        # Merge state dicts if multiple files
        if len(all_state_dicts) > 1:
            merged_state_dict = {}
            for state_dict in all_state_dicts:
                merged_state_dict.update(state_dict)
            all_state_dicts = [merged_state_dict]
        
        # Convert dtype if specified
        if self.config.dtype:
            all_state_dicts = [self._convert_dtype(sd) for sd in all_state_dicts]
        
        # Save converted weights
        return self._save_weights(all_state_dicts[0])
    
    def _find_weight_files(self) -> List[str]:
        """Find model weight files in input directory"""
        weight_files = []
        
        # Check for safetensors files
        for file in os.listdir(self.config.input_dir):
            if file.endswith(".safetensors"):
                weight_files.append(os.path.join(self.config.input_dir, file))
        
        # Check for pytorch bin files
        if not weight_files:
            for file in os.listdir(self.config.input_dir):
                if file.endswith(".bin") and "pytorch_model" in file:
                    weight_files.append(os.path.join(self.config.input_dir, file))
        
        # Sort files for consistent ordering
        weight_files.sort()
        return weight_files
    
    def _load_weight_file(self, file_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load weights from file"""
        try:
            if file_path.endswith(".safetensors"):
                if not SAFETENSORS_AVAILABLE:
                    logger.error("safetensors library not available")
                    return None
                return load_file(file_path, device=self.config.device)
            else:
                return torch.load(file_path, map_location=self.config.device)
        except Exception as e:
            logger.error(f"Failed to load weight file {file_path}: {str(e)}")
            return None
    
    def _convert_dtype(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert tensor dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        
        if self.config.dtype not in dtype_map:
            return state_dict
        
        target_dtype = dtype_map[self.config.dtype]
        converted_dict = {}
        
        for key, tensor in state_dict.items():
            if tensor.dtype != target_dtype:
                converted_dict[key] = tensor.to(target_dtype)
            else:
                converted_dict[key] = tensor
        
        return converted_dict
    
    def _save_weights(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Save converted weights"""
        try:
            if self.config.output_format == "safetensors":
                if not SAFETENSORS_AVAILABLE:
                    logger.error("safetensors library not available")
                    return False
                
                output_path = os.path.join(self.config.output_dir, "model.safetensors")
                save_file(state_dict, output_path)
                logger.info(f"Saved weights to {output_path} (safetensors format)")
            else:
                output_path = os.path.join(self.config.output_dir, "pytorch_model.bin")
                torch.save(state_dict, output_path)
                logger.info(f"Saved weights to {output_path} (pytorch format)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save weights: {str(e)}")
            return False
    
    def _copy_additional_files(self):
        """Copy tokenizer and other files"""
        files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json"
        ]
        
        for file_name in files_to_copy:
            src_path = os.path.join(self.config.input_dir, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(self.config.output_dir, file_name)
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to copy {file_name}: {str(e)}")
        
        # Copy any other JSON files that might be important
        for file in os.listdir(self.config.input_dir):
            if file.endswith(".json") and file not in ["config.json"] + files_to_copy:
                src_path = os.path.join(self.config.input_dir, file)
                dst_path = os.path.join(self.config.output_dir, file)
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {file}")
                except Exception as e:
                    logger.warning(f"Failed to copy {file}: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Universal Model Converter for forge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Baichuan2 model to Llama format
  python universal_converter.py --input_dir /path/to/baichuan2 --output_dir /path/to/output
  
  # Convert with specific output format
  python universal_converter.py --input_dir /path/to/model --output_dir /path/to/output --output_format pytorch
  
  # Convert with dtype conversion
  python universal_converter.py --input_dir /path/to/model --output_dir /path/to/output --dtype float16
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing model files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Output format for model weights (default: safetensors)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Convert model to specified dtype"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for conversion (default: cpu)"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Check for safetensors availability
    if args.output_format == "safetensors" and not SAFETENSORS_AVAILABLE:
        logger.error("safetensors library not available. Please install with: pip install safetensors")
        sys.exit(1)
    
    # Create conversion config
    config = ConversionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose
    )
    
    # Perform conversion
    converter = UniversalConverter(config)
    success = converter.convert()
    
    if success:
        logger.info("Conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()