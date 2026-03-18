"""
forge Model Conversion & Merging Toolkit
Unified converter with plugin architecture and Gradio web interface.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import importlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
import gradio as gr
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('converter.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    HUGGINGFACE = "huggingface"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    BAICHUAN = "baichuan"
    CHATGLM = "chatglm"
    FALCON = "falcon"
    MPT = "mpt"
    GPTJ = "gptj"
    GPT2 = "gpt2"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"


class MergeMethod(Enum):
    """Supported model merging methods."""
    LINEAR = "linear"
    DARE = "dare"
    TIES = "ties"
    TASK_ARITHMETIC = "task_arithmetic"
    SLERP = "slerp"
    PASSTHROUGH = "passthrough"


class QuantizationMethod(Enum):
    """Supported quantization methods."""
    BITSANDBYTES_4BIT = "bitsandbytes_4bit"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    SQUEEZELLM = "squeezellm"
    GGUF = "gguf"


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    input_path: str
    output_path: str
    input_format: Optional[ModelFormat] = None
    output_format: Optional[ModelFormat] = None
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    max_shard_size: str = "10GB"
    safe_serialization: bool = True
    progress_callback: Optional[Any] = None


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    model_paths: List[str]
    output_path: str
    merge_method: MergeMethod = MergeMethod.LINEAR
    weights: Optional[List[float]] = None
    normalize: bool = True
    density: float = 0.5
    epsilon: float = 1e-6
    device: str = "auto"
    dtype: str = "float16"
    tokenizer_path: Optional[str] = None
    progress_callback: Optional[Any] = None


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    input_path: str
    output_path: str
    method: QuantizationMethod = QuantizationMethod.BITSANDBYTES_4BIT
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = False
    sym: bool = True
    device: str = "auto"
    tokenizer_path: Optional[str] = None
    progress_callback: Optional[Any] = None


class BaseConverter(ABC):
    """Base class for all format converters."""
    
    @abstractmethod
    def detect_format(self, path: str) -> bool:
        """Detect if the given path matches this converter's format."""
        pass
    
    @abstractmethod
    def convert(self, config: ConversionConfig) -> None:
        """Convert model from this format to HuggingFace format."""
        pass
    
    @abstractmethod
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Extract metadata from the model."""
        pass


class HuggingFaceConverter(BaseConverter):
    """Converter for HuggingFace models."""
    
    def detect_format(self, path: str) -> bool:
        """Check for HuggingFace model files."""
        path = Path(path)
        if not path.exists():
            return False
        
        # Check for config.json and model files
        has_config = (path / "config.json").exists()
        has_model = any([
            (path / "pytorch_model.bin").exists(),
            (path / "model.safetensors").exists(),
            (path / "pytorch_model.bin.index.json").exists(),
            (path / "model.safetensors.index.json").exists()
        ])
        
        return has_config and has_model
    
    def convert(self, config: ConversionConfig) -> None:
        """HuggingFace to HuggingFace conversion (essentially a copy/resave)."""
        logger.info(f"Converting HuggingFace model from {config.input_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            config.input_path,
            torch_dtype=getattr(torch, config.dtype),
            device_map=config.device,
            trust_remote_code=config.trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        tokenizer = None
        if config.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path,
                trust_remote_code=config.trust_remote_code,
                use_fast=config.use_fast_tokenizer
            )
        elif (Path(config.input_path) / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code,
                use_fast=config.use_fast_tokenizer
            )
        
        # Save model
        model.save_pretrained(
            config.output_path,
            max_shard_size=config.max_shard_size,
            safe_serialization=config.safe_serialization
        )
        
        if tokenizer:
            tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"Model saved to {config.output_path}")
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Extract HuggingFace model metadata."""
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        return {
            "model_type": config.model_type,
            "architectures": getattr(config, "architectures", []),
            "vocab_size": getattr(config, "vocab_size", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
            "num_heads": getattr(config, "num_attention_heads", None),
            "torch_dtype": getattr(config, "torch_dtype", None)
        }


class LlamaConverter(BaseConverter):
    """Converter for LLaMA format models."""
    
    def detect_format(self, path: str) -> bool:
        """Check for LLaMA model files."""
        path = Path(path)
        if not path.exists():
            return False
        
        # LLaMA models typically have consolidated.*.pth files
        llama_files = list(path.glob("consolidated.*.pth"))
        has_params = (path / "params.json").exists()
        
        return len(llama_files) > 0 or has_params
    
    def convert(self, config: ConversionConfig) -> None:
        """Convert LLaMA format to HuggingFace format."""
        logger.info(f"Converting LLaMA model from {config.input_path}")
        
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
        
        input_path = Path(config.input_path)
        
        # Load params
        params_path = input_path / "params.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)
        else:
            params = {}
        
        # Create config
        hf_config = LlamaConfig(
            vocab_size=params.get("vocab_size", 32000),
            hidden_size=params.get("dim", 4096),
            intermediate_size=params.get("hidden_dim", 11008),
            num_hidden_layers=params.get("n_layers", 32),
            num_attention_heads=params.get("n_heads", 32),
            num_key_value_heads=params.get("n_kv_heads", None),
            hidden_act="silu",
            max_position_embeddings=params.get("max_seq_len", 2048),
            initializer_range=0.02,
            rms_norm_eps=params.get("norm_eps", 1e-6),
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=params.get("rope_theta", 10000.0),
            rope_scaling=None,
            attention_bias=False,
        )
        
        # Initialize model
        model = LlamaForCausalLM(hf_config)
        
        # Load weights from consolidated files
        self._load_llama_weights(model, input_path, config)
        
        # Save in HuggingFace format
        model.save_pretrained(
            config.output_path,
            max_shard_size=config.max_shard_size,
            safe_serialization=config.safe_serialization
        )
        
        # Handle tokenizer
        tokenizer_path = input_path / "tokenizer.model"
        if tokenizer_path.exists():
            tokenizer = LlamaTokenizer(vocab_file=str(tokenizer_path))
            tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"LLaMA model converted and saved to {config.output_path}")
    
    def _load_llama_weights(self, model, input_path: Path, config: ConversionConfig):
        """Load weights from LLaMA consolidated files."""
        # This is a simplified implementation
        # Real implementation would handle sharded checkpoints
        consolidated_files = sorted(input_path.glob("consolidated.*.pth"))
        
        if not consolidated_files:
            raise FileNotFoundError(f"No consolidated files found in {input_path}")
        
        # Load first shard as example
        state_dict = torch.load(consolidated_files[0], map_location="cpu")
        
        # Map LLaMA keys to HuggingFace keys
        # This is a simplified mapping - real implementation needs complete mapping
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("tok_embeddings."):
                new_key = key.replace("tok_embeddings.", "model.embed_tokens.")
            elif key.startswith("layers."):
                new_key = key.replace("layers.", "model.layers.")
                new_key = new_key.replace("attention.wq.", "self_attn.q_proj.")
                new_key = new_key.replace("attention.wk.", "self_attn.k_proj.")
                new_key = new_key.replace("attention.wv.", "self_attn.v_proj.")
                new_key = new_key.replace("attention.wo.", "self_attn.o_proj.")
                new_key = new_key.replace("feed_forward.w1.", "mlp.gate_proj.")
                new_key = new_key.replace("feed_forward.w2.", "mlp.down_proj.")
                new_key = new_key.replace("feed_forward.w3.", "mlp.up_proj.")
                new_key = new_key.replace("attention_norm.", "input_layernorm.")
                new_key = new_key.replace("ffn_norm.", "post_attention_layernorm.")
            elif key.startswith("norm."):
                new_key = key.replace("norm.", "model.norm.")
            elif key.startswith("output."):
                new_key = key.replace("output.", "lm_head.")
            else:
                new_key = key
            
            new_state_dict[new_key] = value
        
        # Load weights
        model.load_state_dict(new_state_dict, strict=False)
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Extract LLaMA model metadata."""
        params_path = Path(path) / "params.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                return json.load(f)
        return {}


class QwenConverter(BaseConverter):
    """Converter for Qwen format models."""
    
    def detect_format(self, path: str) -> bool:
        """Check for Qwen model files."""
        path = Path(path)
        if not path.exists():
            return False
        
        # Qwen models have specific naming patterns
        qwen_files = list(path.glob("qwen*.bin")) + list(path.glob("model*.safetensors"))
        has_config = (path / "config.json").exists()
        
        return len(qwen_files) > 0 and has_config
    
    def convert(self, config: ConversionConfig) -> None:
        """Convert Qwen format to HuggingFace format."""
        logger.info(f"Converting Qwen model from {config.input_path}")
        
        # Load original config
        with open(Path(config.input_path) / "config.json", "r") as f:
            original_config = json.load(f)
        
        # Create HuggingFace config
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Load model with trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            config.input_path,
            torch_dtype=getattr(torch, config.dtype),
            device_map=config.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Save in HuggingFace format
        model.save_pretrained(
            config.output_path,
            max_shard_size=config.max_shard_size,
            safe_serialization=config.safe_serialization
        )
        
        # Handle tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.input_path,
            trust_remote_code=True,
            use_fast=config.use_fast_tokenizer
        )
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"Qwen model converted and saved to {config.output_path}")
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Extract Qwen model metadata."""
        config_path = Path(path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}


class BaichuanConverter(BaseConverter):
    """Converter for Baichuan format models."""
    
    def detect_format(self, path: str) -> bool:
        """Check for Baichuan model files."""
        path = Path(path)
        if not path.exists():
            return False
        
        # Baichuan models have specific naming
        baichuan_files = list(path.glob("baichuan*.bin")) + list(path.glob("pytorch_model*.bin"))
        has_config = (path / "config.json").exists()
        
        return len(baichuan_files) > 0 and has_config
    
    def convert(self, config: ConversionConfig) -> None:
        """Convert Baichuan format to HuggingFace format."""
        logger.info(f"Converting Baichuan model from {config.input_path}")
        
        # Use existing conversion script logic
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            config.input_path,
            torch_dtype=getattr(torch, config.dtype),
            device_map=config.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        model.save_pretrained(
            config.output_path,
            max_shard_size=config.max_shard_size,
            safe_serialization=config.safe_serialization
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.input_path,
            trust_remote_code=True,
            use_fast=config.use_fast_tokenizer
        )
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"Baichuan model converted and saved to {config.output_path}")
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Extract Baichuan model metadata."""
        config_path = Path(path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}


class ModelMerger:
    """Handles model merging operations."""
    
    def __init__(self):
        self.merge_methods = {
            MergeMethod.LINEAR: self._linear_merge,
            MergeMethod.DARE: self._dare_merge,
            MergeMethod.TIES: self._ties_merge,
            MergeMethod.TASK_ARITHMETIC: self._task_arithmetic_merge,
            MergeMethod.SLERP: self._slerp_merge,
            MergeMethod.PASSTHROUGH: self._passthrough_merge
        }
    
    def merge(self, config: MergeConfig) -> None:
        """Merge multiple models using specified method."""
        logger.info(f"Merging {len(config.model_paths)} models using {config.merge_method.value}")
        
        if len(config.model_paths) < 2:
            raise ValueError("At least 2 models are required for merging")
        
        # Load models
        models = []
        for i, path in enumerate(config.model_paths):
            if config.progress_callback:
                config.progress_callback(i / len(config.model_paths), f"Loading model {i+1}/{len(config.model_paths)}")
            
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=getattr(torch, config.dtype),
                device_map=config.device,
                low_cpu_mem_usage=True
            )
            models.append(model)
        
        # Perform merge
        if config.progress_callback:
            config.progress_callback(0.5, "Merging models...")
        
        merged_model = self.merge_methods[config.merge_method](models, config)
        
        # Save merged model
        if config.progress_callback:
            config.progress_callback(0.9, "Saving merged model...")
        
        merged_model.save_pretrained(
            config.output_path,
            max_shard_size="10GB",
            safe_serialization=True
        )
        
        # Handle tokenizer
        if config.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_paths[0])
        
        tokenizer.save_pretrained(config.output_path)
        
        if config.progress_callback:
            config.progress_callback(1.0, "Merge complete!")
        
        logger.info(f"Merged model saved to {config.output_path}")
    
    def _linear_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """Linear interpolation merge."""
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        if config.normalize:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Get state dicts
        state_dicts = [model.state_dict() for model in models]
        
        # Merge weights
        merged_state_dict = {}
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            merged = sum(w * t for w, t in zip(weights, tensors))
            merged_state_dict[key] = merged
        
        # Create new model with merged weights
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _dare_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """DARE (Drop And REscale) merge."""
        # Simplified DARE implementation
        weights = config.weights or [1.0 / len(models)] * len(models)
        density = config.density
        
        state_dicts = [model.state_dict() for model in models]
        merged_state_dict = {}
        
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            
            # Drop and rescale
            merged = torch.zeros_like(tensors[0])
            for i, (tensor, weight) in enumerate(zip(tensors, weights)):
                # Create mask
                mask = torch.rand_like(tensor) < density
                # Apply mask and rescale
                masked_tensor = tensor * mask / density
                merged += weight * masked_tensor
            
            merged_state_dict[key] = merged
        
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _ties_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """TIES (Task-Informed Expert Stitching) merge."""
        # Simplified TIES implementation
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        state_dicts = [model.state_dict() for model in models]
        merged_state_dict = {}
        
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            
            # Elect sign based on magnitude
            signs = [torch.sign(t) for t in tensors]
            elected_sign = torch.sign(sum(signs))
            
            # Disjoint merge
            merged = torch.zeros_like(tensors[0])
            for tensor, weight in zip(tensors, weights):
                # Only keep parameters that agree with elected sign
                mask = torch.sign(tensor) == elected_sign
                merged += weight * tensor * mask
            
            merged_state_dict[key] = merged
        
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _task_arithmetic_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """Task arithmetic merge."""
        # Base model is first, task vectors are the rest
        base_model = models[0]
        task_models = models[1:]
        
        base_state = base_model.state_dict()
        task_vectors = []
        
        for model in task_models:
            task_state = model.state_dict()
            vector = {k: task_state[k] - base_state[k] for k in base_state.keys()}
            task_vectors.append(vector)
        
        # Combine task vectors
        weights = config.weights or [1.0] * len(task_vectors)
        combined_vector = {}
        
        for key in base_state.keys():
            combined = sum(w * tv[key] for w, tv in zip(weights, task_vectors))
            combined_vector[key] = combined
        
        # Apply to base
        merged_state = {k: base_state[k] + combined_vector[k] for k in base_state.keys()}
        
        merged_model = base_model.__class__(base_model.config)
        merged_model.load_state_dict(merged_state)
        
        return merged_model
    
    def _slerp_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """Spherical linear interpolation merge."""
        if len(models) != 2:
            raise ValueError("SLERP merge requires exactly 2 models")
        
        t = config.weights[0] if config.weights else 0.5
        
        state_dict1 = models[0].state_dict()
        state_dict2 = models[1].state_dict()
        
        merged_state_dict = {}
        for key in state_dict1.keys():
            v1 = state_dict1[key].flatten().double()
            v2 = state_dict2[key].flatten().double()
            
            # Normalize
            v1_norm = v1 / torch.norm(v1)
            v2_norm = v2 / torch.norm(v2)
            
            # Compute angle
            dot = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
            omega = torch.acos(dot)
            
            # SLERP
            if omega.abs() < config.epsilon:
                # Vectors are nearly parallel, use linear interpolation
                merged = (1 - t) * v1 + t * v2
            else:
                sin_omega = torch.sin(omega)
                merged = (torch.sin((1 - t) * omega) / sin_omega) * v1 + (torch.sin(t * omega) / sin_omega) * v2
            
            merged_state_dict[key] = merged.reshape(state_dict1[key].shape).to(state_dict1[key].dtype)
        
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _passthrough_merge(self, models: List[Any], config: MergeConfig) -> Any:
        """Passthrough merge (concatenate layers)."""
        # This is a simplified version that would need more complex logic
        # for actual layer concatenation
        logger.warning("Passthrough merge is not fully implemented")
        return models[0]


class Quantizer:
    """Handles model quantization."""
    
    def __init__(self):
        self.quantization_methods = {
            QuantizationMethod.BITSANDBYTES_4BIT: self._quantize_bnb_4bit,
            QuantizationMethod.BITSANDBYTES_8BIT: self._quantize_bnb_8bit,
            QuantizationMethod.GPTQ: self._quantize_gptq,
            QuantizationMethod.AWQ: self._quantize_awq,
            QuantizationMethod.SQUEEZELLM: self._quantize_squeezellm,
            QuantizationMethod.GGUF: self._quantize_gguf
        }
    
    def quantize(self, config: QuantizationConfig) -> None:
        """Quantize model using specified method."""
        logger.info(f"Quantizing model using {config.method.value}")
        
        if config.method not in self.quantization_methods:
            raise ValueError(f"Unsupported quantization method: {config.method}")
        
        self.quantization_methods[config.method](config)
    
    def _quantize_bnb_4bit(self, config: QuantizationConfig) -> None:
        """4-bit quantization using bitsandbytes."""
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.dtype),
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.input_path,
            quantization_config=quantization_config,
            device_map=config.device,
            trust_remote_code=True
        )
        
        model.save_pretrained(config.output_path, safe_serialization=True)
        
        if config.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.input_path)
        
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"4-bit quantized model saved to {config.output_path}")
    
    def _quantize_bnb_8bit(self, config: QuantizationConfig) -> None:
        """8-bit quantization using bitsandbytes."""
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.input_path,
            quantization_config=quantization_config,
            device_map=config.device,
            trust_remote_code=True
        )
        
        model.save_pretrained(config.output_path, safe_serialization=True)
        
        if config.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.input_path)
        
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"8-bit quantized model saved to {config.output_path}")
    
    def _quantize_gptq(self, config: QuantizationConfig) -> None:
        """GPTQ quantization."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError("auto-gptq is required for GPTQ quantization. Install with: pip install auto-gptq")
        
        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            damp_percent=config.damp_percent,
            desc_act=config.desc_act,
            sym=config.sym
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(
            config.input_path,
            quantize_config=quantize_config,
            trust_remote_code=True
        )
        
        # Quantize
        model.quantize([])
        
        model.save_quantized(config.output_path)
        
        if config.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.input_path)
        
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"GPTQ quantized model saved to {config.output_path}")
    
    def _quantize_awq(self, config: QuantizationConfig) -> None:
        """AWQ quantization."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("autoawq is required for AWQ quantization. Install with: pip install autoawq")
        
        model = AutoAWQForCausalLM.from_pretrained(
            config.input_path,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.input_path,
            trust_remote_code=True
        )
        
        quant_config = {
            "zero_point": True,
            "q_group_size": config.group_size,
            "w_bit": config.bits,
            "version": "GEMM"
        }
        
        model.quantize(tokenizer, quant_config=quant_config)
        
        model.save_quantized(config.output_path)
        tokenizer.save_pretrained(config.output_path)
        
        logger.info(f"AWQ quantized model saved to {config.output_path}")
    
    def _quantize_squeezellm(self, config: QuantizationConfig) -> None:
        """SqueezeLLM quantization."""
        logger.warning("SqueezeLLM quantization not yet implemented")
        # Implementation would go here
    
    def _quantize_gguf(self, config: QuantizationConfig) -> None:
        """GGUF quantization."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for GGUF quantization")
        
        logger.warning("GGUF quantization requires llama.cpp conversion tools")
        # Implementation would convert to GGUF format


class ConverterRegistry:
    """Registry for all converters."""
    
    def __init__(self):
        self.converters: Dict[ModelFormat, BaseConverter] = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Register built-in converters."""
        self.register(ModelFormat.HUGGINGFACE, HuggingFaceConverter())
        self.register(ModelFormat.LLAMA, LlamaConverter())
        self.register(ModelFormat.QWEN, QwenConverter())
        self.register(ModelFormat.BAICHUAN, BaichuanConverter())
    
    def register(self, format: ModelFormat, converter: BaseConverter):
        """Register a converter for a specific format."""
        self.converters[format] = converter
        logger.info(f"Registered converter for {format.value}")
    
    def detect_format(self, path: str) -> Optional[ModelFormat]:
        """Auto-detect model format."""
        for format, converter in self.converters.items():
            try:
                if converter.detect_format(path):
                    logger.info(f"Detected format: {format.value} for path: {path}")
                    return format
            except Exception as e:
                logger.debug(f"Format detection failed for {format.value}: {e}")
        
        logger.warning(f"Could not detect format for path: {path}")
        return None
    
    def get_converter(self, format: ModelFormat) -> Optional[BaseConverter]:
        """Get converter for specific format."""
        return self.converters.get(format)
    
    def list_supported_formats(self) -> List[str]:
        """List all supported formats."""
        return [fmt.value for fmt in self.converters.keys()]


class UnifiedConverter:
    """Main converter class that orchestrates all operations."""
    
    def __init__(self):
        self.registry = ConverterRegistry()
        self.merger = ModelMerger()
        self.quantizer = Quantizer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def convert_model(self, config: ConversionConfig) -> None:
        """Convert model between formats."""
        # Auto-detect input format if not specified
        if config.input_format is None:
            config.input_format = self.registry.detect_format(config.input_path)
            if config.input_format is None:
                raise ValueError(f"Could not detect format for {config.input_path}")
        
        # Get converter
        converter = self.registry.get_converter(config.input_format)
        if converter is None:
            raise ValueError(f"No converter available for format: {config.input_format}")
        
        # Perform conversion
        converter.convert(config)
    
    def merge_models(self, config: MergeConfig) -> None:
        """Merge multiple models."""
        self.merger.merge(config)
    
    def quantize_model(self, config: QuantizationConfig) -> None:
        """Quantize model."""
        self.quantizer.quantize(config)
    
    def batch_convert(self, configs: List[ConversionConfig]) -> List[Tuple[bool, str]]:
        """Batch convert multiple models."""
        results = []
        
        futures = {
            self.executor.submit(self.convert_model, config): i 
            for i, config in enumerate(configs)
        }
        
        for future in tqdm(as_completed(futures), total=len(configs), desc="Batch converting"):
            idx = futures[future]
            try:
                future.result()
                results.append((True, f"Successfully converted model {idx}"))
            except Exception as e:
                results.append((False, f"Failed to convert model {idx}: {str(e)}"))
        
        return results
    
    def get_model_info(self, path: str) -> Dict[str, Any]:
        """Get information about a model."""
        format = self.registry.detect_format(path)
        if format is None:
            return {"error": "Could not detect model format"}
        
        converter = self.registry.get_converter(format)
        if converter is None:
            return {"error": f"No converter for format: {format}"}
        
        metadata = converter.get_metadata(path)
        metadata["detected_format"] = format.value
        metadata["path"] = path
        
        return metadata


class GradioInterface:
    """Gradio web interface for the converter toolkit."""
    
    def __init__(self, converter: UnifiedConverter):
        self.converter = converter
        self.current_operation = None
        self.progress = 0
        self.status = "Ready"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title="forge Model Converter", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🦙 forge Model Conversion & Merging Toolkit")
            gr.Markdown("Unified toolkit for model conversion, merging, and quantization")
            
            with gr.Tabs():
                # Conversion Tab
                with gr.TabItem("🔄 Model Conversion"):
                    with gr.Row():
                        with gr.Column():
                            input_path = gr.Textbox(
                                label="Input Model Path",
                                placeholder="Path to model or HuggingFace repo ID",
                                info="Local path or HuggingFace model ID"
                            )
                            output_path = gr.Textbox(
                                label="Output Path",
                                placeholder="Path to save converted model"
                            )
                            
                            with gr.Row():
                                input_format = gr.Dropdown(
                                    choices=["Auto-detect"] + self.converter.registry.list_supported_formats(),
                                    label="Input Format",
                                    value="Auto-detect"
                                )
                                output_format = gr.Dropdown(
                                    choices=self.converter.registry.list_supported_formats(),
                                    label="Output Format",
                                    value="huggingface"
                                )
                            
                            with gr.Accordion("Advanced Options", open=False):
                                dtype = gr.Dropdown(
                                    choices=["float16", "bfloat16", "float32"],
                                    label="Data Type",
                                    value="float16"
                                )
                                device = gr.Dropdown(
                                    choices=["auto", "cpu", "cuda", "mps"],
                                    label="Device",
                                    value="auto"
                                )
                                trust_remote_code = gr.Checkbox(
                                    label="Trust Remote Code",
                                    value=False
                                )
                                max_shard_size = gr.Textbox(
                                    label="Max Shard Size",
                                    value="10GB"
                                )
                            
                            convert_btn = gr.Button("Convert Model", variant="primary")
                        
                        with gr.Column():
                            conversion_output = gr.Textbox(
                                label="Conversion Output",
                                interactive=False,
                                lines=10
                            )
                            conversion_progress = gr.Slider(
                                label="Progress",
                                minimum=0,
                                maximum=100,
                                value=0,
                                interactive=False
                            )
                
                # Merging Tab
                with gr.TabItem("🔀 Model Merging"):
                    with gr.Row():
                        with gr.Column():
                            model_paths = gr.Textbox(
                                label="Model Paths (one per line)",
                                placeholder="Enter model paths, one per line",
                                lines=5
                            )
                            merge_method = gr.Dropdown(
                                choices=[m.value for m in MergeMethod],
                                label="Merge Method",
                                value="linear"
                            )
                            
                            with gr.Row():
                                weights = gr.Textbox(
                                    label="Weights (comma-separated)",
                                    placeholder="e.g., 0.7, 0.3",
                                    info="Optional: specify weights for each model"
                                )
                                density = gr.Slider(
                                    label="Density (for DARE/TIES)",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1
                                )
                            
                            merge_output_path = gr.Textbox(
                                label="Output Path",
                                placeholder="Path to save merged model"
                            )
                            
                            with gr.Accordion("Advanced Options", open=False):
                                merge_dtype = gr.Dropdown(
                                    choices=["float16", "bfloat16", "float32"],
                                    label="Data Type",
                                    value="float16"
                                )
                                merge_device = gr.Dropdown(
                                    choices=["auto", "cpu", "cuda", "mps"],
                                    label="Device",
                                    value="auto"
                                )
                                normalize = gr.Checkbox(
                                    label="Normalize Weights",
                                    value=True
                                )
                            
                            merge_btn = gr.Button("Merge Models", variant="primary")
                        
                        with gr.Column():
                            merge_output = gr.Textbox(
                                label="Merge Output",
                                interactive=False,
                                lines=10
                            )
                            merge_progress = gr.Slider(
                                label="Progress",
                                minimum=0,
                                maximum=100,
                                value=0,
                                interactive=False
                            )
                
                # Quantization Tab
                with gr.TabItem("⚡ Model Quantization"):
                    with gr.Row():
                        with gr.Column():
                            quant_input_path = gr.Textbox(
                                label="Input Model Path",
                                placeholder="Path to model"
                            )
                            quant_output_path = gr.Textbox(
                                label="Output Path",
                                placeholder="Path to save quantized model"
                            )
                            quant_method = gr.Dropdown(
                                choices=[m.value for m in QuantizationMethod],
                                label="Quantization Method",
                                value="bitsandbytes_4bit"
                            )
                            
                            with gr.Row():
                                quant_bits = gr.Slider(
                                    label="Bits",
                                    minimum=2,
                                    maximum=8,
                                    value=4,
                                    step=2
                                )
                                group_size = gr.Slider(
                                    label="Group Size",
                                    minimum=32,
                                    maximum=256,
                                    value=128,
                                    step=32
                                )
                            
                            with gr.Accordion("Advanced Options", open=False):
                                quant_dtype = gr.Dropdown(
                                    choices=["float16", "bfloat16", "float32"],
                                    label="Compute Data Type",
                                    value="float16"
                                )
                                quant_device = gr.Dropdown(
                                    choices=["auto", "cpu", "cuda", "mps"],
                                    label="Device",
                                    value="auto"
                                )
                                desc_act = gr.Checkbox(
                                    label="Desc Act (for GPTQ)",
                                    value=False
                                )
                                sym = gr.Checkbox(
                                    label="Symmetric Quantization",
                                    value=True
                                )
                            
                            quantize_btn = gr.Button("Quantize Model", variant="primary")
                        
                        with gr.Column():
                            quant_output = gr.Textbox(
                                label="Quantization Output",
                                interactive=False,
                                lines=10
                            )
                            quant_progress = gr.Slider(
                                label="Progress",
                                minimum=0,
                                maximum=100,
                                value=0,
                                interactive=False
                            )
                
                # Batch Processing Tab
                with gr.TabItem("📦 Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_config = gr.Textbox(
                                label="Batch Configuration (JSON)",
                                placeholder='[{"input_path": "...", "output_path": "..."}]',
                                lines=10
                            )
                            batch_btn = gr.Button("Run Batch Processing", variant="primary")
                        
                        with gr.Column():
                            batch_output = gr.Textbox(
                                label="Batch Output",
                                interactive=False,
                                lines=15
                            )
                            batch_progress = gr.Slider(
                                label="Progress",
                                minimum=0,
                                maximum=100,
                                value=0,
                                interactive=False
                            )
                
                # Model Info Tab
                with gr.TabItem("ℹ️ Model Info"):
                    with gr.Row():
                        info_path = gr.Textbox(
                            label="Model Path",
                            placeholder="Enter model path to inspect"
                        )
                        info_btn = gr.Button("Get Model Info", variant="primary")
                    
                    info_output = gr.JSON(
                        label="Model Information"
                    )
            
            # Event handlers
            convert_btn.click(
                fn=self._handle_conversion,
                inputs=[
                    input_path, output_path, input_format, output_format,
                    dtype, device, trust_remote_code, max_shard_size
                ],
                outputs=[conversion_output, conversion_progress]
            )
            
            merge_btn.click(
                fn=self._handle_merge,
                inputs=[
                    model_paths, merge_method, weights, density,
                    merge_output_path, merge_dtype, merge_device, normalize
                ],
                outputs=[merge_output, merge_progress]
            )
            
            quantize_btn.click(
                fn=self._handle_quantization,
                inputs=[
                    quant_input_path, quant_output_path, quant_method,
                    quant_bits, group_size, quant_dtype, quant_device,
                    desc_act, sym
                ],
                outputs=[quant_output, quant_progress]
            )
            
            batch_btn.click(
                fn=self._handle_batch,
                inputs=[batch_config],
                outputs=[batch_output, batch_progress]
            )
            
            info_btn.click(
                fn=self._handle_info,
                inputs=[info_path],
                outputs=[info_output]
            )
        
        return interface
    
    def _update_progress(self, progress: float, status: str):
        """Update progress callback."""
        self.progress = progress * 100
        self.status = status
    
    def _handle_conversion(self, input_path, output_path, input_format, output_format,
                          dtype, device, trust_remote_code, max_shard_size):
        """Handle conversion request."""
        try:
            if not input_path or not output_path:
                return "Please provide both input and output paths", 0
            
            # Parse format
            in_fmt = None if input_format == "Auto-detect" else ModelFormat(input_format)
            out_fmt = ModelFormat(output_format)
            
            config = ConversionConfig(
                input_path=input_path,
                output_path=output_path,
                input_format=in_fmt,
                output_format=out_fmt,
                dtype=dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                max_shard_size=max_shard_size,
                progress_callback=self._update_progress
            )
            
            self.converter.convert_model(config)
            return f"✅ Conversion complete!\nModel saved to: {output_path}", 100
        
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return f"❌ Conversion failed: {str(e)}", 0
    
    def _handle_merge(self, model_paths, merge_method, weights, density,
                     output_path, dtype, device, normalize):
        """Handle merge request."""
        try:
            if not model_paths or not output_path:
                return "Please provide model paths and output path", 0
            
            # Parse paths
            paths = [p.strip() for p in model_paths.split('\n') if p.strip()]
            if len(paths) < 2:
                return "At least 2 models are required for merging", 0
            
            # Parse weights
            weight_list = None
            if weights:
                try:
                    weight_list = [float(w.strip()) for w in weights.split(',')]
                    if len(weight_list) != len(paths):
                        return f"Number of weights ({len(weight_list)}) must match number of models ({len(paths)})", 0
                except ValueError:
                    return "Invalid weights format. Use comma-separated numbers.", 0
            
            config = MergeConfig(
                model_paths=paths,
                output_path=output_path,
                merge_method=MergeMethod(merge_method),
                weights=weight_list,
                density=density,
                dtype=dtype,
                device=device,
                normalize=normalize,
                progress_callback=self._update_progress
            )
            
            self.converter.merge_models(config)
            return f"✅ Merge complete!\nMerged model saved to: {output_path}", 100
        
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return f"❌ Merge failed: {str(e)}", 0
    
    def _handle_quantization(self, input_path, output_path, method, bits,
                           group_size, dtype, device, desc_act, sym):
        """Handle quantization request."""
        try:
            if not input_path or not output_path:
                return "Please provide both input and output paths", 0
            
            config = QuantizationConfig(
                input_path=input_path,
                output_path=output_path,
                method=QuantizationMethod(method),
                bits=int(bits),
                group_size=int(group_size),
                dtype=dtype,
                device=device,
                desc_act=desc_act,
                sym=sym,
                progress_callback=self._update_progress
            )
            
            self.converter.quantize_model(config)
            return f"✅ Quantization complete!\nQuantized model saved to: {output_path}", 100
        
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return f"❌ Quantization failed: {str(e)}", 0
    
    def _handle_batch(self, config_json):
        """Handle batch processing request."""
        try:
            if not config_json:
                return "Please provide batch configuration", 0
            
            # Parse JSON config
            configs_data = json.loads(config_json)
            if not isinstance(configs_data, list):
                return "Configuration must be a JSON array", 0
            
            # Create conversion configs
            configs = []
            for i, cfg in enumerate(configs_data):
                config = ConversionConfig(
                    input_path=cfg["input_path"],
                    output_path=cfg["output_path"],
                    input_format=ModelFormat(cfg.get("input_format")) if cfg.get("input_format") else None,
                    output_format=ModelFormat(cfg.get("output_format", "huggingface")),
                    dtype=cfg.get("dtype", "float16"),
                    device=cfg.get("device", "auto"),
                    trust_remote_code=cfg.get("trust_remote_code", False),
                    progress_callback=lambda p, s: self._update_progress(
                        (i + p) / len(configs_data), s
                    )
                )
                configs.append(config)
            
            # Run batch conversion
            results = self.converter.batch_convert(configs)
            
            # Format output
            output_lines = []
            success_count = 0
            for success, message in results:
                if success:
                    success_count += 1
                    output_lines.append(f"✅ {message}")
                else:
                    output_lines.append(f"❌ {message}")
            
            summary = f"\n📊 Batch complete: {success_count}/{len(configs)} successful"
            return '\n'.join(output_lines) + summary, 100
        
        except json.JSONDecodeError:
            return "Invalid JSON configuration", 0
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return f"❌ Batch processing failed: {str(e)}", 0
    
    def _handle_info(self, path):
        """Handle model info request."""
        try:
            if not path:
                return {"error": "Please provide a model path"}
            
            info = self.converter.get_model_info(path)
            return info
        
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="forge Model Conversion & Merging Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert model
  python -m forge.tools.converter convert --input ./llama-model --output ./hf-model --format llama
  
  # Merge models
  python -m forge.tools.converter merge --models model1 model2 --output merged --method linear
  
  # Launch web interface
  python -m forge.tools.converter gui --port 7860
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert model format')
    convert_parser.add_argument('--input', required=True, help='Input model path')
    convert_parser.add_argument('--output', required=True, help='Output model path')
    convert_parser.add_argument('--format', choices=[f.value for f in ModelFormat],
                               help='Input format (auto-detect if not specified)')
    convert_parser.add_argument('--output-format', default='huggingface',
                               choices=[f.value for f in ModelFormat],
                               help='Output format')
    convert_parser.add_argument('--dtype', default='float16',
                               choices=['float16', 'bfloat16', 'float32'],
                               help='Data type')
    convert_parser.add_argument('--device', default='auto',
                               choices=['auto', 'cpu', 'cuda', 'mps'],
                               help='Device')
    convert_parser.add_argument('--trust-remote-code', action='store_true',
                               help='Trust remote code')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge models')
    merge_parser.add_argument('--models', nargs='+', required=True,
                             help='Model paths to merge')
    merge_parser.add_argument('--output', required=True, help='Output path')
    merge_parser.add_argument('--method', default='linear',
                             choices=[m.value for m in MergeMethod],
                             help='Merge method')
    merge_parser.add_argument('--weights', nargs='+', type=float,
                             help='Weights for each model')
    merge_parser.add_argument('--density', type=float, default=0.5,
                             help='Density for DARE/TIES')
    
    # Quantize command
    quant_parser = subparsers.add_parser('quantize', help='Quantize model')
    quant_parser.add_argument('--input', required=True, help='Input model path')
    quant_parser.add_argument('--output', required=True, help='Output model path')
    quant_parser.add_argument('--method', default='bitsandbytes_4bit',
                             choices=[m.value for m in QuantizationMethod],
                             help='Quantization method')
    quant_parser.add_argument('--bits', type=int, default=4,
                             help='Quantization bits')
    quant_parser.add_argument('--group-size', type=int, default=128,
                             help='Group size')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch web interface')
    gui_parser.add_argument('--port', type=int, default=7860,
                           help='Port to run on')
    gui_parser.add_argument('--share', action='store_true',
                           help='Create public link')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get model information')
    info_parser.add_argument('--path', required=True, help='Model path')
    
    args = parser.parse_args()
    
    converter = UnifiedConverter()
    
    if args.command == 'convert':
        config = ConversionConfig(
            input_path=args.input,
            output_path=args.output,
            input_format=ModelFormat(args.format) if args.format else None,
            output_format=ModelFormat(args.output_format),
            dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code
        )
        converter.convert_model(config)
        print(f"✅ Model converted successfully to {args.output}")
    
    elif args.command == 'merge':
        config = MergeConfig(
            model_paths=args.models,
            output_path=args.output,
            merge_method=MergeMethod(args.method),
            weights=args.weights,
            density=args.density
        )
        converter.merge_models(config)
        print(f"✅ Models merged successfully to {args.output}")
    
    elif args.command == 'quantize':
        config = QuantizationConfig(
            input_path=args.input,
            output_path=args.output,
            method=QuantizationMethod(args.method),
            bits=args.bits,
            group_size=args.group_size
        )
        converter.quantize_model(config)
        print(f"✅ Model quantized successfully to {args.output}")
    
    elif args.command == 'gui':
        interface = GradioInterface(converter)
        app = interface.create_interface()
        print(f"🚀 Starting web interface on port {args.port}")
        app.launch(server_port=args.port, share=args.share)
    
    elif args.command == 'info':
        info = converter.get_model_info(args.path)
        print(json.dumps(info, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()