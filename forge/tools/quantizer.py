import os
import sys
import json
import yaml
import torch
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

import gradio as gr
import numpy as np
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from forge import logger
    from forge.extras.misc import get_device_count
except ImportError:
    # Fallback logging if forge not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class ConversionFormat(Enum):
    """Supported model conversion formats."""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"
    EXL2 = "exl2"
    MEGATRON = "megatron"
    LLAMA_CPP = "llama_cpp"
    ONNX = "onnx"

class QuantizationMethod(Enum):
    """Supported quantization methods."""
    GPTQ = "gptq"
    AWQ = "awq"
    EXL2 = "exl2"
    BITSANDBYTES = "bitsandbytes"
    QUANTO = "quanto"
    AQLM = "aqlm"
    HQQ = "hqq"

class MergeMethod(Enum):
    """Supported model merging methods."""
    LINEAR = "linear"
    DARE = "dare"
    TIES = "ties"
    TASK_ARITHMETIC = "task_arithmetic"
    SLERP = "slerp"
    PASSTHROUGH = "passthrough"

@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    input_path: str
    output_path: str
    input_format: ConversionFormat = ConversionFormat.HUGGINGFACE
    output_format: ConversionFormat = ConversionFormat.HUGGINGFACE
    tokenizer_path: Optional[str] = None
    dtype: str = "float16"
    device: str = "auto"
    max_shard_size: str = "10GB"
    safe_serialization: bool = True
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    config_overrides: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    input_path: str
    output_path: str
    method: QuantizationMethod = QuantizationMethod.GPTQ
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    damp_percent: float = 0.01
    dataset: Optional[str] = None
    nsamples: int = 128
    seed: int = 42
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False

@dataclass
class MergeConfig:
    """Configuration for model merging."""
    model_paths: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    output_path: str = ""
    method: MergeMethod = MergeMethod.LINEAR
    normalize: bool = True
    density: float = 0.5
    epsilon: float = 1e-6
    device: str = "auto"
    dtype: str = "float16"
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False

class BaseModelConverter(ABC):
    """Base class for model converters."""
    
    @abstractmethod
    def convert(self, config: ConversionConfig) -> bool:
        """Convert model from one format to another."""
        pass
    
    @abstractmethod
    def validate_input(self, config: ConversionConfig) -> Tuple[bool, str]:
        """Validate input configuration."""
        pass

class HuggingFaceConverter(BaseModelConverter):
    """Converter for HuggingFace models."""
    
    def convert(self, config: ConversionConfig) -> bool:
        """Convert model to HuggingFace format."""
        try:
            logger.info(f"Loading model from {config.input_path}")
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                config.input_path,
                torch_dtype=getattr(torch, config.dtype),
                device_map=config.device,
                trust_remote_code=config.trust_remote_code,
                **config.config_overrides
            )
            
            tokenizer_path = config.tokenizer_path or config.input_path
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=config.trust_remote_code,
                use_fast=config.use_fast_tokenizer
            )
            
            # Save model
            logger.info(f"Saving model to {config.output_path}")
            model.save_pretrained(
                config.output_path,
                max_shard_size=config.max_shard_size,
                safe_serialization=config.safe_serialization
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(config.output_path)
            
            logger.info("Conversion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False
    
    def validate_input(self, config: ConversionConfig) -> Tuple[bool, str]:
        """Validate HuggingFace conversion config."""
        if not os.path.exists(config.input_path):
            return False, f"Input path does not exist: {config.input_path}"
        
        if not os.path.isdir(config.input_path):
            return False, f"Input path is not a directory: {config.input_path}"
        
        # Check for required files
        required_files = ["config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(config.input_path, file)):
                return False, f"Missing required file: {file}"
        
        return True, "Validation passed"

class GGUFConverter(BaseModelConverter):
    """Converter for GGUF format (requires llama.cpp)."""
    
    def convert(self, config: ConversionConfig) -> bool:
        """Convert model to GGUF format."""
        try:
            # This would require llama.cpp conversion scripts
            # For now, we'll provide a placeholder implementation
            logger.warning("GGUF conversion requires llama.cpp tools")
            logger.info("Please use convert.py from llama.cpp repository")
            
            # Create output directory
            os.makedirs(config.output_path, exist_ok=True)
            
            # Write conversion instructions
            instructions = f"""
            To convert to GGUF format:
            1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp
            2. Run: python convert.py {config.input_path} --outtype {config.dtype} --outfile {config.output_path}/model.gguf
            
            Alternatively, use quantize tool for quantized GGUF:
            ./quantize {config.input_path}/model.gguf {config.output_path}/model-q4_0.gguf q4_0
            """
            
            with open(os.path.join(config.output_path, "CONVERSION_INSTRUCTIONS.md"), "w") as f:
                f.write(instructions)
            
            return True
            
        except Exception as e:
            logger.error(f"GGUF conversion failed: {str(e)}")
            return False
    
    def validate_input(self, config: ConversionConfig) -> Tuple[bool, str]:
        """Validate GGUF conversion config."""
        if not os.path.exists(config.input_path):
            return False, f"Input path does not exist: {config.input_path}"
        return True, "Validation passed"

class GPTQConverter(BaseModelConverter):
    """Converter for GPTQ quantized models."""
    
    def convert(self, config: ConversionConfig) -> bool:
        """Convert model to GPTQ format."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            logger.info(f"Loading model from {config.input_path}")
            
            # Create quantization config
            quantize_config = BaseQuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False
            )
            
            # Load model
            model = AutoGPTQForCausalLM.from_pretrained(
                config.input_path,
                quantize_config=quantize_config,
                trust_remote_code=config.trust_remote_code
            )
            
            # Save model
            logger.info(f"Saving GPTQ model to {config.output_path}")
            model.save_quantized(config.output_path)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path or config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            tokenizer.save_pretrained(config.output_path)
            
            return True
            
        except ImportError:
            logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
            return False
        except Exception as e:
            logger.error(f"GPTQ conversion failed: {str(e)}")
            return False
    
    def validate_input(self, config: ConversionConfig) -> Tuple[bool, str]:
        """Validate GPTQ conversion config."""
        if not os.path.exists(config.input_path):
            return False, f"Input path does not exist: {config.input_path}"
        return True, "Validation passed"

class ModelQuantizer:
    """Unified model quantization class."""
    
    def __init__(self):
        self.method_handlers = {
            QuantizationMethod.GPTQ: self._quantize_gptq,
            QuantizationMethod.AWQ: self._quantize_awq,
            QuantizationMethod.BITSANDBYTES: self._quantize_bitsandbytes,
            QuantizationMethod.QUANTO: self._quantize_quanto,
        }
    
    def quantize(self, config: QuantizationConfig) -> bool:
        """Quantize model using specified method."""
        handler = self.method_handlers.get(config.method)
        if not handler:
            logger.error(f"Unsupported quantization method: {config.method}")
            return False
        
        try:
            return handler(config)
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            return False
    
    def _quantize_gptq(self, config: QuantizationConfig) -> bool:
        """Quantize using GPTQ method."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from datasets import load_dataset
            
            logger.info(f"Loading model from {config.input_path}")
            
            # Load dataset for calibration
            if config.dataset:
                dataset = load_dataset(config.dataset, split=f"train[:{config.nsamples}]")
                dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512))
            else:
                # Use default dataset
                dataset = ["auto-gptq is an easy-to-use model quantization library"]
            
            # Create quantization config
            quantize_config = BaseQuantizeConfig(
                bits=config.bits,
                group_size=config.group_size,
                desc_act=config.desc_act,
                sym=config.sym,
                damp_percent=config.damp_percent
            )
            
            # Load and quantize model
            model = AutoGPTQForCausalLM.from_pretrained(
                config.input_path,
                quantize_config=quantize_config,
                trust_remote_code=config.trust_remote_code
            )
            
            # Quantize
            logger.info("Quantizing model...")
            model.quantize(dataset)
            
            # Save
            logger.info(f"Saving quantized model to {config.output_path}")
            model.save_quantized(config.output_path)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            tokenizer.save_pretrained(config.output_path)
            
            return True
            
        except ImportError:
            logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
            return False
    
    def _quantize_awq(self, config: QuantizationConfig) -> bool:
        """Quantize using AWQ method."""
        try:
            from awq import AutoAWQForCausalLM
            
            logger.info(f"Loading model from {config.input_path}")
            
            # Load model
            model = AutoAWQForCausalLM.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            
            # Quantize
            logger.info("Quantizing model with AWQ...")
            model.quantize(
                tokenizer,
                quant_config={
                    "zero_point": True,
                    "q_group_size": config.group_size,
                    "w_bit": config.bits,
                    "version": "GEMM"
                }
            )
            
            # Save
            logger.info(f"Saving AWQ model to {config.output_path}")
            model.save_quantized(config.output_path)
            tokenizer.save_pretrained(config.output_path)
            
            return True
            
        except ImportError:
            logger.error("autoawq not installed. Install with: pip install autoawq")
            return False
    
    def _quantize_bitsandbytes(self, config: QuantizationConfig) -> bool:
        """Quantize using bitsandbytes."""
        try:
            from transformers import BitsAndBytesConfig
            
            logger.info(f"Loading model from {config.input_path}")
            
            # Create quantization config
            if config.bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, config.dtype),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif config.bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                raise ValueError(f"Bitsandbytes only supports 4-bit or 8-bit, got {config.bits}")
            
            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                config.input_path,
                quantization_config=quantization_config,
                device_map=config.device,
                trust_remote_code=config.trust_remote_code
            )
            
            # Save
            logger.info(f"Saving quantized model to {config.output_path}")
            model.save_pretrained(config.output_path)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            tokenizer.save_pretrained(config.output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Bitsandbytes quantization failed: {str(e)}")
            return False
    
    def _quantize_quanto(self, config: QuantizationConfig) -> bool:
        """Quantize using quanto library."""
        try:
            from quanto import quantize, freeze, qint8, qint4
            from quanto import QLinear
            
            logger.info(f"Loading model from {config.input_path}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.input_path,
                torch_dtype=getattr(torch, config.dtype),
                trust_remote_code=config.trust_remote_code
            )
            
            # Select quantization type
            if config.bits == 4:
                quant_type = qint4
            elif config.bits == 8:
                quant_type = qint8
            else:
                raise ValueError(f"Quanto only supports 4-bit or 8-bit, got {config.bits}")
            
            # Quantize
            logger.info("Quantizing model with quanto...")
            quantize(model, weights=quant_type)
            freeze(model)
            
            # Save
            logger.info(f"Saving quantized model to {config.output_path}")
            model.save_pretrained(config.output_path)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.input_path,
                trust_remote_code=config.trust_remote_code
            )
            tokenizer.save_pretrained(config.output_path)
            
            return True
            
        except ImportError:
            logger.error("quanto not installed. Install with: pip install quanto")
            return False

class ModelMerger:
    """Unified model merging class."""
    
    def merge(self, config: MergeConfig) -> bool:
        """Merge multiple models using specified method."""
        try:
            if len(config.model_paths) < 2:
                logger.error("At least 2 models required for merging")
                return False
            
            if len(config.weights) != len(config.model_paths):
                # Normalize weights if not provided
                config.weights = [1.0 / len(config.model_paths)] * len(config.model_paths)
            
            # Load models
            models = []
            for path in config.model_paths:
                logger.info(f"Loading model from {path}")
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=getattr(torch, config.dtype),
                    device_map=config.device,
                    trust_remote_code=config.trust_remote_code
                )
                models.append(model)
            
            # Merge based on method
            if config.method == MergeMethod.LINEAR:
                merged_model = self._merge_linear(models, config.weights, config.normalize)
            elif config.method == MergeMethod.DARE:
                merged_model = self._merge_dare(models, config.weights, config.density)
            elif config.method == MergeMethod.TIES:
                merged_model = self._merge_ties(models, config.weights, config.density)
            elif config.method == MergeMethod.TASK_ARITHMETIC:
                merged_model = self._merge_task_arithmetic(models, config.weights)
            elif config.method == MergeMethod.SLERP:
                merged_model = self._merge_slerp(models, config.weights)
            else:
                merged_model = self._merge_passthrough(models, config.weights)
            
            # Save merged model
            logger.info(f"Saving merged model to {config.output_path}")
            merged_model.save_pretrained(config.output_path)
            
            # Save tokenizer (use first model's tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path or config.model_paths[0],
                trust_remote_code=config.trust_remote_code
            )
            tokenizer.save_pretrained(config.output_path)
            
            logger.info("Merging completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Merging failed: {str(e)}")
            return False
    
    def _merge_linear(self, models: List, weights: List[float], normalize: bool) -> Any:
        """Linear merging of models."""
        logger.info("Performing linear merge")
        
        if normalize:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Get state dicts
        state_dicts = [model.state_dict() for model in models]
        
        # Initialize merged state dict
        merged_state_dict = {}
        
        # Get all parameter names
        param_names = set()
        for sd in state_dicts:
            param_names.update(sd.keys())
        
        # Merge parameters
        for name in param_names:
            if not all(name in sd for sd in state_dicts):
                logger.warning(f"Parameter {name} not found in all models, skipping")
                continue
            
            # Get parameters from all models
            params = [sd[name] for sd in state_dicts]
            
            # Check shapes match
            shapes = [p.shape for p in params]
            if len(set(shapes)) > 1:
                logger.warning(f"Parameter {name} has mismatched shapes: {shapes}, skipping")
                continue
            
            # Weighted sum
            merged_param = torch.zeros_like(params[0])
            for param, weight in zip(params, weights):
                merged_param += weight * param
            
            merged_state_dict[name] = merged_param
        
        # Create new model with merged weights
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _merge_dare(self, models: List, weights: List[float], density: float) -> Any:
        """DARE (Drop And REscale) merging."""
        logger.info(f"Performing DARE merge with density={density}")
        
        # Implementation of DARE merging
        # Based on: https://arxiv.org/abs/2311.03099
        
        state_dicts = [model.state_dict() for model in models]
        merged_state_dict = {}
        
        param_names = set()
        for sd in state_dicts:
            param_names.update(sd.keys())
        
        for name in param_names:
            if not all(name in sd for sd in state_dicts):
                continue
            
            params = [sd[name] for sd in state_dicts]
            shapes = [p.shape for p in params]
            
            if len(set(shapes)) > 1:
                continue
            
            # Apply DARE merging
            merged_param = torch.zeros_like(params[0])
            
            for param, weight in zip(params, weights):
                # Create mask for dropping parameters
                mask = torch.bernoulli(torch.full_like(param, density))
                
                # Rescale
                scaled_param = param * mask / density
                
                # Weighted sum
                merged_param += weight * scaled_param
            
            merged_state_dict[name] = merged_param
        
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _merge_ties(self, models: List, weights: List[float], density: float) -> Any:
        """TIES (Task-Informed Ensembling) merging."""
        logger.info(f"Performing TIES merge with density={density}")
        
        # Implementation of TIES merging
        # Based on: https://arxiv.org/abs/2306.01708
        
        state_dicts = [model.state_dict() for model in models]
        merged_state_dict = {}
        
        param_names = set()
        for sd in state_dicts:
            param_names.update(sd.keys())
        
        for name in param_names:
            if not all(name in sd for sd in state_dicts):
                continue
            
            params = [sd[name] for sd in state_dicts]
            shapes = [p.shape for p in params]
            
            if len(set(shapes)) > 1:
                continue
            
            # Reselect: Keep only top-k% parameters by magnitude
            all_params = torch.stack(params)
            magnitudes = torch.abs(all_params)
            threshold = torch.quantile(magnitudes, 1 - density, dim=0)
            mask = magnitudes >= threshold
            
            # Elect: Resolve sign conflicts
            signs = torch.sign(all_params)
            majority_sign = torch.sign(torch.sum(signs * mask.float(), dim=0))
            
            # Merge: Weighted average of selected parameters
            merged_param = torch.zeros_like(params[0])
            for param, weight, m in zip(params, weights, mask):
                # Only include parameters that agree with majority sign
                sign_agreement = (torch.sign(param) == majority_sign) | (param == 0)
                final_mask = m & sign_agreement
                merged_param += weight * param * final_mask
            
            merged_state_dict[name] = merged_param
        
        merged_model = models[0].__class__(models[0].config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _merge_task_arithmetic(self, models: List, weights: List[float]) -> Any:
        """Task arithmetic merging."""
        logger.info("Performing task arithmetic merge")
        
        # Task arithmetic: weighted sum of task vectors
        base_model = models[0]
        base_state_dict = base_model.state_dict()
        
        merged_state_dict = base_state_dict.copy()
        
        for model, weight in zip(models[1:], weights[1:]):
            task_vector = {}
            for name in base_state_dict:
                if name in model.state_dict():
                    task_vector[name] = model.state_dict()[name] - base_state_dict[name]
            
            # Add weighted task vector
            for name in merged_state_dict:
                if name in task_vector:
                    merged_state_dict[name] += weight * task_vector[name]
        
        merged_model = base_model.__class__(base_model.config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _merge_slerp(self, models: List, weights: List[float]) -> Any:
        """Spherical Linear Interpolation (SLERP) merging."""
        logger.info("Performing SLERP merge")
        
        # SLERP between two models
        if len(models) != 2:
            logger.warning("SLERP only supports 2 models, falling back to linear merge")
            return self._merge_linear(models, weights, normalize=True)
        
        model1, model2 = models
        weight1, weight2 = weights
        
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        merged_state_dict = {}
        
        for name in state_dict1:
            if name not in state_dict2:
                continue
            
            param1 = state_dict1[name].float()
            param2 = state_dict2[name].float()
            
            # Flatten for SLERP
            flat1 = param1.view(-1)
            flat2 = param2.view(-1)
            
            # Compute dot product
            dot = torch.dot(flat1, flat2) / (torch.norm(flat1) * torch.norm(flat2))
            dot = torch.clamp(dot, -1.0, 1.0)
            
            # Compute angle
            theta = torch.acos(dot)
            
            # SLERP formula
            if theta.abs() < 1e-6:
                # Linear interpolation for small angles
                merged_flat = weight1 * flat1 + weight2 * flat2
            else:
                # SLERP
                sin_theta = torch.sin(theta)
                merged_flat = (torch.sin((1 - weight1) * theta) / sin_theta * flat1 +
                              torch.sin(weight1 * theta) / sin_theta * flat2)
            
            merged_state_dict[name] = merged_flat.view(param1.shape).to(param1.dtype)
        
        merged_model = model1.__class__(model1.config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _merge_passthrough(self, models: List, weights: List[float]) -> Any:
        """Passthrough merging (no merging, just return first model)."""
        logger.warning("Passthrough merge selected, returning first model")
        return models[0]

class ModelToolkit:
    """Unified model conversion, merging, and quantization toolkit."""
    
    def __init__(self):
        self.converters = {
            ConversionFormat.HUGGINGFACE: HuggingFaceConverter(),
            ConversionFormat.GGUF: GGUFConverter(),
            ConversionFormat.GPTQ: GPTQConverter(),
        }
        self.quantizer = ModelQuantizer()
        self.merger = ModelMerger()
        
        # Auto-detect format from files
        self.format_detectors = {
            "config.json": ConversionFormat.HUGGINGFACE,
            "model.safetensors": ConversionFormat.HUGGINGFACE,
            "pytorch_model.bin": ConversionFormat.HUGGINGFACE,
            "gguf": ConversionFormat.GGUF,
            "gptq": ConversionFormat.GPTQ,
        }
    
    def detect_format(self, path: str) -> ConversionFormat:
        """Auto-detect model format from files."""
        path = Path(path)
        
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in [".gguf", ".ggml"]:
                return ConversionFormat.GGUF
            elif suffix in [".safetensors", ".bin"]:
                return ConversionFormat.HUGGINGFACE
        elif path.is_dir():
            files = [f.name for f in path.iterdir()]
            
            # Check for GGUF files
            for f in files:
                if f.endswith(".gguf"):
                    return ConversionFormat.GGUF
            
            # Check for GPTQ files
            if "quantize_config.json" in files:
                return ConversionFormat.GPTQ
            
            # Check for HuggingFace files
            if "config.json" in files:
                return ConversionFormat.HUGGINGFACE
        
        # Default to HuggingFace
        return ConversionFormat.HUGGINGFACE
    
    def convert_model(self, config: ConversionConfig) -> bool:
        """Convert model format."""
        converter = self.converters.get(config.output_format)
        if not converter:
            logger.error(f"No converter available for format: {config.output_format}")
            return False
        
        # Validate
        valid, message = converter.validate_input(config)
        if not valid:
            logger.error(f"Validation failed: {message}")
            return False
        
        # Convert
        return converter.convert(config)
    
    def quantize_model(self, config: QuantizationConfig) -> bool:
        """Quantize model."""
        return self.quantizer.quantize(config)
    
    def merge_models(self, config: MergeConfig) -> bool:
        """Merge multiple models."""
        return self.merger.merge(config)
    
    def batch_process(self, configs: List[Dict], operation: str) -> Dict[str, bool]:
        """Batch process multiple operations."""
        results = {}
        
        for i, config_dict in enumerate(configs):
            logger.info(f"Processing batch item {i+1}/{len(configs)}")
            
            try:
                if operation == "convert":
                    config = ConversionConfig(**config_dict)
                    results[f"item_{i}"] = self.convert_model(config)
                elif operation == "quantize":
                    config = QuantizationConfig(**config_dict)
                    results[f"item_{i}"] = self.quantize_model(config)
                elif operation == "merge":
                    config = MergeConfig(**config_dict)
                    results[f"item_{i}"] = self.merge_models(config)
                else:
                    logger.error(f"Unknown operation: {operation}")
                    results[f"item_{i}"] = False
            except Exception as e:
                logger.error(f"Batch item {i} failed: {str(e)}")
                results[f"item_{i}"] = False
        
        return results

class GradioInterface:
    """Gradio web interface for the model toolkit."""
    
    def __init__(self, toolkit: ModelToolkit):
        self.toolkit = toolkit
        self.css = """
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .section { margin: 20px 0; padding: 20px; border-radius: 10px; background: #f8f9fa; }
        .progress { height: 20px; }
        """
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(css=self.css, title="LLaMA Factory Model Toolkit") as demo:
            gr.Markdown("# 🦙 LLaMA Factory Model Toolkit")
            gr.Markdown("Unified toolkit for model conversion, quantization, and merging")
            
            with gr.Tabs():
                # Conversion Tab
                with gr.Tab("🔄 Model Conversion"):
                    self._create_conversion_tab()
                
                # Quantization Tab
                with gr.Tab("⚡ Model Quantization"):
                    self._create_quantization_tab()
                
                # Merging Tab
                with gr.Tab("🔀 Model Merging"):
                    self._create_merging_tab()
                
                # Batch Processing Tab
                with gr.Tab("📦 Batch Processing"):
                    self._create_batch_tab()
                
                # Model Info Tab
                with gr.Tab("ℹ️ Model Info"):
                    self._create_info_tab()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("Built with ❤️ by LLaMA Factory | [GitHub](https://github.com/hiyouga/LLaMA-Factory)")
        
        return demo
    
    def _create_conversion_tab(self):
        """Create conversion interface."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Model")
                input_path = gr.Textbox(
                    label="Input Path/URL",
                    placeholder="Path to model or HuggingFace Hub ID",
                    info="Can be local path or HuggingFace model ID"
                )
                input_format = gr.Dropdown(
                    choices=[f.value for f in ConversionFormat],
                    label="Input Format",
                    value="huggingface",
                    info="Auto-detected if not specified"
                )
                auto_detect = gr.Checkbox(
                    label="Auto-detect format",
                    value=True,
                    info="Automatically detect model format from files"
                )
                
                gr.Markdown("### Output Settings")
                output_path = gr.Textbox(
                    label="Output Path",
                    placeholder="Path to save converted model"
                )
                output_format = gr.Dropdown(
                    choices=[f.value for f in ConversionFormat],
                    label="Output Format",
                    value="huggingface"
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    dtype = gr.Dropdown(
                        choices=["float16", "bfloat16", "float32"],
                        label="Data Type",
                        value="float16"
                    )
                    device = gr.Textbox(
                        label="Device",
                        value="auto",
                        info="auto, cpu, cuda, cuda:0, etc."
                    )
                    max_shard_size = gr.Textbox(
                        label="Max Shard Size",
                        value="10GB"
                    )
                    safe_serialization = gr.Checkbox(
                        label="Safe Serialization",
                        value=True,
                        info="Use safetensors format"
                    )
                    trust_remote_code = gr.Checkbox(
                        label="Trust Remote Code",
                        value=False,
                        info="Allow custom model code"
                    )
            
            with gr.Column():
                gr.Markdown("### Conversion Log")
                convert_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    interactive=False
                )
                
                convert_btn = gr.Button("🚀 Convert Model", variant="primary")
                convert_progress = gr.Progress()
        
        # Event handlers
        convert_btn.click(
            fn=self._run_conversion,
            inputs=[
                input_path, input_format, auto_detect,
                output_path, output_format,
                dtype, device, max_shard_size,
                safe_serialization, trust_remote_code
            ],
            outputs=[convert_output]
        )
    
    def _create_quantization_tab(self):
        """Create quantization interface."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model to Quantize")
                quant_input_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to model or HuggingFace Hub ID"
                )
                
                gr.Markdown("### Quantization Settings")
                quant_method = gr.Dropdown(
                    choices=[m.value for m in QuantizationMethod],
                    label="Quantization Method",
                    value="gptq"
                )
                quant_bits = gr.Slider(
                    minimum=2,
                    maximum=8,
                    step=1,
                    value=4,
                    label="Bits",
                    info="Number of bits for quantization"
                )
                quant_group_size = gr.Slider(
                    minimum=32,
                    maximum=256,
                    step=32,
                    value=128,
                    label="Group Size"
                )
                
                with gr.Accordion("Method-specific Options", open=False):
                    # GPTQ options
                    with gr.Group(visible=True) as gptq_options:
                        desc_act = gr.Checkbox(label="Desc Act", value=False)
                        sym = gr.Checkbox(label="Symmetric", value=True)
                        damp_percent = gr.Slider(
                            minimum=0.001,
                            maximum=0.1,
                            step=0.001,
                            value=0.01,
                            label="Damp Percent"
                        )
                    
                    # Dataset options
                    dataset = gr.Textbox(
                        label="Calibration Dataset",
                        placeholder="Dataset for calibration (e.g., wikitext)",
                        info="Leave empty for default"
                    )
                    nsamples = gr.Slider(
                        minimum=16,
                        maximum=512,
                        step=16,
                        value=128,
                        label="Number of Samples"
                    )
                
                quant_output_path = gr.Textbox(
                    label="Output Path",
                    placeholder="Path to save quantized model"
                )
            
            with gr.Column():
                gr.Markdown("### Quantization Log")
                quant_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    interactive=False
                )
                
                quant_btn = gr.Button("⚡ Quantize Model", variant="primary")
                quant_progress = gr.Progress()
        
        # Event handlers
        quant_btn.click(
            fn=self._run_quantization,
            inputs=[
                quant_input_path, quant_method,
                quant_bits, quant_group_size,
                desc_act, sym, damp_percent,
                dataset, nsamples,
                quant_output_path
            ],
            outputs=[quant_output]
        )
    
    def _create_merging_tab(self):
        """Create merging interface."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Models to Merge")
                
                # Dynamic model inputs
                model_inputs = []
                weight_inputs = []
                
                for i in range(3):
                    with gr.Row():
                        model_path = gr.Textbox(
                            label=f"Model {i+1}",
                            placeholder=f"Path to model {i+1}",
                            scale=3
                        )
                        weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                            label=f"Weight {i+1}",
                            scale=1
                        )
                    model_inputs.append(model_path)
                    weight_inputs.append(weight)
                
                add_model_btn = gr.Button("➕ Add Another Model")
                
                gr.Markdown("### Merge Settings")
                merge_method = gr.Dropdown(
                    choices=[m.value for m in MergeMethod],
                    label="Merge Method",
                    value="linear"
                )
                
                with gr.Accordion("Method Options", open=False):
                    normalize = gr.Checkbox(
                        label="Normalize Weights",
                        value=True,
                        info="Ensure weights sum to 1"
                    )
                    density = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        label="Density",
                        info="For DARE/TIES methods"
                    )
                
                merge_output_path = gr.Textbox(
                    label="Output Path",
                    placeholder="Path to save merged model"
                )
            
            with gr.Column():
                gr.Markdown("### Merging Log")
                merge_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    interactive=False
                )
                
                merge_btn = gr.Button("🔀 Merge Models", variant="primary")
                merge_progress = gr.Progress()
        
        # Event handlers
        def add_model_row():
            # This would add more model input rows dynamically
            pass
        
        add_model_btn.click(
            fn=add_model_row,
            outputs=[]
        )
        
        merge_btn.click(
            fn=self._run_merging,
            inputs=[
                *model_inputs, *weight_inputs,
                merge_method, normalize, density,
                merge_output_path
            ],
            outputs=[merge_output]
        )
    
    def _create_batch_tab(self):
        """Create batch processing interface."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Batch Configuration")
                batch_config = gr.Textbox(
                    label="Batch Config (JSON)",
                    placeholder='[{"input_path": "model1", "output_path": "output1", ...}]',
                    lines=10,
                    info="JSON array of conversion/quantization/merge configs"
                )
                batch_operation = gr.Dropdown(
                    choices=["convert", "quantize", "merge"],
                    label="Operation",
                    value="convert"
                )
                batch_btn = gr.Button("📦 Run Batch", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Batch Results")
                batch_output = gr.JSON(
                    label="Results",
                    show_label=True
                )
                batch_progress = gr.Progress()
        
        batch_btn.click(
            fn=self._run_batch,
            inputs=[batch_config, batch_operation],
            outputs=[batch_output]
        )
    
    def _create_info_tab(self):
        """Create model info interface."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Information")
                info_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to model or HuggingFace Hub ID"
                )
                info_btn = gr.Button("ℹ️ Get Info", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Model Details")
                info_output = gr.JSON(
                    label="Model Info",
                    show_label=True
                )
        
        info_btn.click(
            fn=self._get_model_info,
            inputs=[info_path],
            outputs=[info_output]
        )
    
    def _run_conversion(self, *args):
        """Run model conversion."""
        # Convert inputs to config
        config = ConversionConfig(
            input_path=args[0],
            input_format=ConversionFormat(args[1]) if not args[2] else self.toolkit.detect_format(args[0]),
            output_path=args[3],
            output_format=ConversionFormat(args[4]),
            dtype=args[5],
            device=args[6],
            max_shard_size=args[7],
            safe_serialization=args[8],
            trust_remote_code=args[9]
        )
        
        # Run conversion
        success = self.toolkit.convert_model(config)
        
        if success:
            return "✅ Conversion completed successfully!"
        else:
            return "❌ Conversion failed. Check logs for details."
    
    def _run_quantization(self, *args):
        """Run model quantization."""
        config = QuantizationConfig(
            input_path=args[0],
            method=QuantizationMethod(args[1]),
            bits=int(args[2]),
            group_size=int(args[3]),
            desc_act=args[4],
            sym=args[5],
            damp_percent=args[6],
            dataset=args[7] if args[7] else None,
            nsamples=int(args[8]),
            output_path=args[9]
        )
        
        success = self.toolkit.quantize_model(config)
        
        if success:
            return "✅ Quantization completed successfully!"
        else:
            return "❌ Quantization failed. Check logs for details."
    
    def _run_merging(self, *args):
        """Run model merging."""
        # Extract models and weights
        models = [args[i] for i in range(3) if args[i]]
        weights = [args[i+3] for i in range(3) if args[i]]
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        
        config = MergeConfig(
            model_paths=models,
            weights=weights,
            method=MergeMethod(args[6]),
            normalize=args[7],
            density=args[8],
            output_path=args[9]
        )
        
        success = self.toolkit.merge_models(config)
        
        if success:
            return "✅ Merging completed successfully!"
        else:
            return "❌ Merging failed. Check logs for details."
    
    def _run_batch(self, config_json, operation):
        """Run batch processing."""
        try:
            configs = json.loads(config_json)
            results = self.toolkit.batch_process(configs, operation)
            return results
        except json.JSONDecodeError:
            return {"error": "Invalid JSON configuration"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_model_info(self, model_path):
        """Get model information."""
        try:
            # Try to load config
            if os.path.exists(model_path):
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                info = {
                    "model_type": config.model_type,
                    "architectures": config.architectures,
                    "vocab_size": config.vocab_size,
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                    "torch_dtype": str(config.torch_dtype),
                    "tokenizer_vocab_size": len(tokenizer),
                    "model_path": model_path,
                }
                
                # Try to get model size
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    param_count = sum(p.numel() for p in model.parameters())
                    info["parameters"] = f"{param_count:,} ({param_count/1e9:.2f}B)"
                    del model
                except:
                    info["parameters"] = "Unknown"
                
                return info
            else:
                # Try HuggingFace Hub
                api = HfApi()
                model_info = api.model_info(model_path)
                return {
                    "model_id": model_info.modelId,
                    "author": model_info.author,
                    "tags": model_info.tags,
                    "downloads": model_info.downloads,
                    "likes": model_info.likes,
                    "last_modified": str(model_info.lastModified),
                }
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main entry point for CLI and web interface."""
    parser = argparse.ArgumentParser(description="LLaMA Factory Model Toolkit")
    parser.add_argument("--mode", choices=["cli", "web"], default="web",
                       help="Run mode: cli or web interface")
    parser.add_argument("--operation", choices=["convert", "quantize", "merge"],
                       help="Operation to perform (CLI mode)")
    parser.add_argument("--config", type=str, help="JSON config file (CLI mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Initialize toolkit
    toolkit = ModelToolkit()
    
    if args.mode == "cli":
        # CLI mode
        if not args.config:
            logger.error("Config file required for CLI mode")
            return
        
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        
        if args.operation == "convert":
            config = ConversionConfig(**config_dict)
            success = toolkit.convert_model(config)
        elif args.operation == "quantize":
            config = QuantizationConfig(**config_dict)
            success = toolkit.quantize_model(config)
        elif args.operation == "merge":
            config = MergeConfig(**config_dict)
            success = toolkit.merge_models(config)
        else:
            logger.error(f"Unknown operation: {args.operation}")
            return
        
        if success:
            print("✅ Operation completed successfully!")
        else:
            print("❌ Operation failed!")
            sys.exit(1)
    
    else:
        # Web interface mode
        interface = GradioInterface(toolkit)
        demo = interface.create_interface()
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )

if __name__ == "__main__":
    main()