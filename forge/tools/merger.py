#!/usr/bin/env python3
"""
forge Model Conversion & Merging Toolkit
Unified toolkit with GUI for model merging, conversion, and quantization.
Supports automatic format detection and batch processing.
"""

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
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import gradio as gr
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
import numpy as np
from tqdm import tqdm
import threading
import queue
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FORMATS = {
    'hf': 'HuggingFace',
    'llama': 'LLaMA',
    'qwen': 'Qwen',
    'baichuan': 'Baichuan',
    'bloom': 'BLOOM',
    'mpt': 'MPT',
    'falcon': 'Falcon',
    'mistral': 'Mistral',
    'gpt2': 'GPT-2',
    'opt': 'OPT',
    'phi': 'Phi',
    'gemma': 'Gemma',
}

MERGE_METHODS = {
    'linear': 'Linear Interpolation',
    'dare': 'DARE (Drop And REscale)',
    'ties': 'TIES (TRIM, ELECT SIGN & MERGE)',
    'task_arithmetic': 'Task Arithmetic',
    'slerp': 'SLERP (Spherical Linear Interpolation)',
    'model_stock': 'Model Stock',
}

QUANTIZATION_METHODS = {
    'gptq': 'GPTQ',
    'awq': 'AWQ',
    'bnb': 'BitsAndBytes',
    'gguf': 'GGUF',
    'exl2': 'ExLlamaV2',
}


class FormatDetector:
    """Automatic model format detection."""
    
    @staticmethod
    def detect_format(model_path: str) -> Tuple[str, Dict]:
        """Detect model format from path or HuggingFace Hub."""
        try:
            if os.path.exists(model_path):
                # Local path detection
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    arch = config.get("architectures", [""])[0].lower()
                    
                    if "llama" in arch:
                        return "llama", config
                    elif "qwen" in arch:
                        return "qwen", config
                    elif "baichuan" in arch:
                        return "baichuan", config
                    elif "bloom" in arch:
                        return "bloom", config
                    elif "mpt" in arch:
                        return "mpt", config
                    elif "falcon" in arch:
                        return "falcon", config
                    elif "mistral" in arch:
                        return "mistral", config
                    elif "gpt2" in arch:
                        return "gpt2", config
                    elif "opt" in arch:
                        return "opt", config
                    elif "phi" in arch:
                        return "phi", config
                    elif "gemma" in arch:
                        return "gemma", config
                    else:
                        # Try to infer from model_type
                        model_type = config.get("model_type", "").lower()
                        if model_type in SUPPORTED_FORMATS:
                            return model_type, config
                        else:
                            return "hf", config
                else:
                    # Check for other format indicators
                    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                        return "hf", {}
                    elif os.path.exists(os.path.join(model_path, "model.safetensors")):
                        return "hf", {}
                    else:
                        raise ValueError(f"Could not detect format for {model_path}")
            else:
                # HuggingFace Hub detection
                api = HfApi()
                model_info = api.model_info(model_path)
                config = model_info.config or {}
                arch = config.get("architectures", [""])[0].lower()
                
                for fmt in SUPPORTED_FORMATS:
                    if fmt in arch or fmt in model_path.lower():
                        return fmt, config
                
                return "hf", config
                
        except Exception as e:
            logger.warning(f"Format detection failed: {e}. Defaulting to HF format.")
            return "hf", {}


class BaseConverter:
    """Base class for model converters."""
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
    
    def convert(self, 
                input_path: str, 
                output_path: str, 
                config: Dict = None,
                **kwargs) -> bool:
        """Convert model from source to target format."""
        raise NotImplementedError
    
    def validate(self, model_path: str) -> bool:
        """Validate if conversion is possible."""
        return True


class HFConverter(BaseConverter):
    """HuggingFace format converter."""
    
    def convert(self, input_path: str, output_path: str, config: Dict = None, **kwargs):
        """Convert to/from HuggingFace format."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                input_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                input_path,
                trust_remote_code=True
            )
            
            # Save in target format
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Successfully converted {input_path} to HuggingFace format at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False


class LlamaConverter(BaseConverter):
    """LLaMA format converter."""
    
    def convert(self, input_path: str, output_path: str, config: Dict = None, **kwargs):
        """Convert to LLaMA format."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if self.source_format == "hf":
                # Convert from HuggingFace to LLaMA format
                from scripts.convert_ckpt.llamafy_qwen import convert_qwen_to_llama
                convert_qwen_to_llama(input_path, output_path)
            else:
                raise NotImplementedError(f"Conversion from {self.source_format} to LLaMA not implemented")
            
            logger.info(f"Successfully converted to LLaMA format at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"LLaMA conversion failed: {e}")
            return False


class QwenConverter(BaseConverter):
    """Qwen format converter."""
    
    def convert(self, input_path: str, output_path: str, config: Dict = None, **kwargs):
        """Convert to Qwen format."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if self.source_format == "hf":
                # Convert from HuggingFace to Qwen format
                from scripts.convert_ckpt.llamafy_qwen import convert_llama_to_qwen
                convert_llama_to_qwen(input_path, output_path)
            else:
                raise NotImplementedError(f"Conversion from {self.source_format} to Qwen not implemented")
            
            logger.info(f"Successfully converted to Qwen format at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Qwen conversion failed: {e}")
            return False


class BaichuanConverter(BaseConverter):
    """Baichuan format converter."""
    
    def convert(self, input_path: str, output_path: str, config: Dict = None, **kwargs):
        """Convert to Baichuan format."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if self.source_format == "hf":
                # Convert from HuggingFace to Baichuan format
                from scripts.convert_ckpt.llamafy_baichuan2 import convert_baichuan2_to_llama
                convert_baichuan2_to_llama(input_path, output_path)
            else:
                raise NotImplementedError(f"Conversion from {self.source_format} to Baichuan not implemented")
            
            logger.info(f"Successfully converted to Baichuan format at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Baichuan conversion failed: {e}")
            return False


class BaseMerger:
    """Base class for model merging methods."""
    
    def __init__(self, method: str):
        self.method = method
    
    def merge(self, 
              models: List[str], 
              output_path: str,
              weights: List[float] = None,
              **kwargs) -> bool:
        """Merge multiple models."""
        raise NotImplementedError


class LinearMerger(BaseMerger):
    """Linear interpolation merger."""
    
    def merge(self, models: List[str], output_path: str, weights: List[float] = None, **kwargs):
        """Merge models using linear interpolation."""
        try:
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Load first model to get structure
            logger.info(f"Loading base model: {models[0]}")
            base_model = AutoModelForCausalLM.from_pretrained(
                models[0],
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Initialize merged state dict
            merged_state_dict = {}
            for key in base_model.state_dict().keys():
                merged_state_dict[key] = torch.zeros_like(base_model.state_dict()[key])
            
            # Merge models
            for i, (model_path, weight) in enumerate(zip(models, weights)):
                logger.info(f"Merging model {i+1}/{len(models)}: {model_path} with weight {weight}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                for key in merged_state_dict.keys():
                    if key in model.state_dict():
                        merged_state_dict[key] += weight * model.state_dict()[key].to(merged_state_dict[key].dtype)
                
                del model
                torch.cuda.empty_cache()
            
            # Save merged model
            os.makedirs(output_path, exist_ok=True)
            
            # Load tokenizer from first model
            tokenizer = AutoTokenizer.from_pretrained(models[0], trust_remote_code=True)
            
            # Save merged model
            base_model.load_state_dict(merged_state_dict)
            base_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Successfully merged models to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Linear merging failed: {e}")
            return False


class DAREMerger(BaseMerger):
    """DARE (Drop And REscale) merger."""
    
    def merge(self, models: List[str], output_path: str, weights: List[float] = None, 
              drop_rate: float = 0.5, rescale: bool = True, **kwargs):
        """Merge models using DARE method."""
        try:
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            
            # Load models
            loaded_models = []
            for model_path in tqdm(models, desc="Loading models"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                loaded_models.append(model)
            
            # Get base model structure
            base_model = loaded_models[0]
            merged_state_dict = {}
            
            # Apply DARE merging
            for key in tqdm(base_model.state_dict().keys(), desc="Merging parameters"):
                params = []
                for i, model in enumerate(loaded_models):
                    if key in model.state_dict():
                        param = model.state_dict()[key].clone()
                        
                        # Apply dropout mask
                        if drop_rate > 0:
                            mask = torch.rand_like(param) > drop_rate
                            param = param * mask.float()
                        
                        # Rescale
                        if rescale and drop_rate > 0:
                            param = param / (1 - drop_rate)
                        
                        params.append(param * weights[i])
                
                # Sum parameters
                if params:
                    merged_param = sum(params)
                    merged_state_dict[key] = merged_param
            
            # Save merged model
            os.makedirs(output_path, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(models[0], trust_remote_code=True)
            
            base_model.load_state_dict(merged_state_dict)
            base_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Cleanup
            for model in loaded_models:
                del model
            torch.cuda.empty_cache()
            
            logger.info(f"Successfully merged models using DARE to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"DARE merging failed: {e}")
            return False


class TIESMerger(BaseMerger):
    """TIES (TRIM, ELECT SIGN & MERGE) merger."""
    
    def merge(self, models: List[str], output_path: str, weights: List[float] = None,
              top_k: float = 0.2, **kwargs):
        """Merge models using TIES method."""
        try:
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            
            # Load models
            loaded_models = []
            for model_path in tqdm(models, desc="Loading models"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                loaded_models.append(model)
            
            # Get base model structure
            base_model = loaded_models[0]
            merged_state_dict = {}
            
            # Apply TIES merging
            for key in tqdm(base_model.state_dict().keys(), desc="Merging parameters"):
                params = []
                for i, model in enumerate(loaded_models):
                    if key in model.state_dict():
                        param = model.state_dict()[key].clone()
                        
                        # Trim: Keep only top-k% of parameters by magnitude
                        if top_k < 1.0:
                            flat_param = param.flatten()
                            k = int(len(flat_param) * top_k)
                            if k > 0:
                                threshold = torch.topk(torch.abs(flat_param), k).values[-1]
                                mask = torch.abs(param) >= threshold
                                param = param * mask.float()
                        
                        params.append(param * weights[i])
                
                # Elect sign: Majority voting on parameter signs
                if len(params) > 1:
                    signs = [torch.sign(p) for p in params]
                    majority_sign = torch.sign(sum(signs))
                    
                    # Keep only parameters that agree with majority sign
                    trimmed_params = []
                    for p in params:
                        mask = torch.sign(p) == majority_sign
                        trimmed_params.append(p * mask.float())
                    
                    merged_param = sum(trimmed_params)
                else:
                    merged_param = sum(params)
                
                merged_state_dict[key] = merged_param
            
            # Save merged model
            os.makedirs(output_path, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(models[0], trust_remote_code=True)
            
            base_model.load_state_dict(merged_state_dict)
            base_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Cleanup
            for model in loaded_models:
                del model
            torch.cuda.empty_cache()
            
            logger.info(f"Successfully merged models using TIES to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"TIES merging failed: {e}")
            return False


class Quantizer:
    """Model quantization handler."""
    
    def __init__(self, method: str = 'gptq'):
        self.method = method
    
    def quantize(self, 
                 model_path: str, 
                 output_path: str,
                 bits: int = 4,
                 group_size: int = 128,
                 **kwargs) -> bool:
        """Quantize model using specified method."""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if self.method == 'gptq':
                return self._quantize_gptq(model_path, output_path, bits, group_size, **kwargs)
            elif self.method == 'awq':
                return self._quantize_awq(model_path, output_path, bits, group_size, **kwargs)
            elif self.method == 'bnb':
                return self._quantize_bnb(model_path, output_path, bits, **kwargs)
            elif self.method == 'gguf':
                return self._quantize_gguf(model_path, output_path, bits, **kwargs)
            else:
                raise ValueError(f"Unsupported quantization method: {self.method}")
                
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False
    
    def _quantize_gptq(self, model_path: str, output_path: str, bits: int, group_size: int, **kwargs):
        """Quantize using GPTQ."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=kwargs.get('desc_act', False),
            )
            
            model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                quantize_config=quantize_config,
                trust_remote_code=True
            )
            
            # Quantize
            model.quantize(
                examples=kwargs.get('examples', None),
                batch_size=kwargs.get('batch_size', 1),
            )
            
            # Save
            model.save_quantized(output_path)
            
            logger.info(f"Successfully quantized model using GPTQ to {output_path}")
            return True
            
        except ImportError:
            logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
            return False
    
    def _quantize_awq(self, model_path: str, output_path: str, bits: int, group_size: int, **kwargs):
        """Quantize using AWQ."""
        try:
            from awq import AutoAWQForCausalLM
            
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Quantize
            model.quantize(
                tokenizer=kwargs.get('tokenizer', None),
                quant_config={
                    "zero_point": True,
                    "q_group_size": group_size,
                    "w_bit": bits,
                    "version": "GEMM"
                }
            )
            
            # Save
            model.save_quantized(output_path)
            
            logger.info(f"Successfully quantized model using AWQ to {output_path}")
            return True
            
        except ImportError:
            logger.error("autoawq not installed. Install with: pip install autoawq")
            return False
    
    def _quantize_bnb(self, model_path: str, output_path: str, bits: int, **kwargs):
        """Quantize using BitsAndBytes."""
        try:
            from transformers import BitsAndBytesConfig
            
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"BitsAndBytes only supports 4 or 8 bits, got {bits}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Save
            model.save_pretrained(output_path)
            
            logger.info(f"Successfully quantized model using BitsAndBytes to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            return False
    
    def _quantize_gguf(self, model_path: str, output_path: str, bits: int, **kwargs):
        """Quantize to GGUF format."""
        try:
            # This would require llama.cpp conversion
            logger.warning("GGUF quantization requires llama.cpp. Use convert.py from llama.cpp")
            return False
        except Exception as e:
            logger.error(f"GGUF quantization failed: {e}")
            return False


class ConverterRegistry:
    """Registry for model converters."""
    
    _converters = {}
    
    @classmethod
    def register(cls, source_format: str, target_format: str):
        """Register a converter."""
        def decorator(converter_class):
            key = f"{source_format}->{target_format}"
            cls._converters[key] = converter_class
            return converter_class
        return decorator
    
    @classmethod
    def get_converter(cls, source_format: str, target_format: str) -> Optional[BaseConverter]:
        """Get converter for specified formats."""
        key = f"{source_format}->{target_format}"
        converter_class = cls._converters.get(key)
        if converter_class:
            return converter_class(source_format, target_format)
        return None


class MergerRegistry:
    """Registry for model mergers."""
    
    _mergers = {}
    
    @classmethod
    def register(cls, method: str):
        """Register a merger."""
        def decorator(merger_class):
            cls._mergers[method] = merger_class
            return merger_class
        return decorator
    
    @classmethod
    def get_merger(cls, method: str) -> Optional[BaseMerger]:
        """Get merger for specified method."""
        merger_class = cls._mergers.get(method)
        if merger_class:
            return merger_class(method)
        return None


# Register converters
ConverterRegistry.register('hf', 'llama')(LlamaConverter)
ConverterRegistry.register('hf', 'qwen')(QwenConverter)
ConverterRegistry.register('hf', 'baichuan')(BaichuanConverter)
ConverterRegistry.register('llama', 'hf')(HFConverter)
ConverterRegistry.register('qwen', 'hf')(HFConverter)
ConverterRegistry.register('baichuan', 'hf')(HFConverter)

# Register mergers
MergerRegistry.register('linear')(LinearMerger)
MergerRegistry.register('dare')(DAREMerger)
MergerRegistry.register('ties')(TIESMerger)


class ModelToolkit:
    """Unified model conversion, merging, and quantization toolkit."""
    
    def __init__(self):
        self.format_detector = FormatDetector()
        self.converter_registry = ConverterRegistry
        self.merger_registry = MergerRegistry
        self.quantizer = None
        self.progress_queue = queue.Queue()
        self.current_operation = None
    
    def detect_format(self, model_path: str) -> Tuple[str, Dict]:
        """Detect model format."""
        return self.format_detector.detect_format(model_path)
    
    def convert_model(self,
                     input_path: str,
                     output_path: str,
                     target_format: str = 'hf',
                     source_format: str = None,
                     **kwargs) -> bool:
        """Convert model to target format."""
        try:
            # Auto-detect source format if not provided
            if source_format is None:
                source_format, config = self.detect_format(input_path)
                logger.info(f"Detected source format: {source_format}")
            
            # Get converter
            converter = self.converter_registry.get_converter(source_format, target_format)
            if converter is None:
                # Try direct HF conversion
                converter = HFConverter(source_format, target_format)
            
            # Perform conversion
            self.current_operation = "conversion"
            result = converter.convert(input_path, output_path, **kwargs)
            self.current_operation = None
            
            return result
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            self.current_operation = None
            return False
    
    def merge_models(self,
                    models: List[str],
                    output_path: str,
                    method: str = 'linear',
                    weights: List[float] = None,
                    **kwargs) -> bool:
        """Merge multiple models."""
        try:
            # Get merger
            merger = self.merger_registry.get_merger(method)
            if merger is None:
                raise ValueError(f"Unsupported merge method: {method}")
            
            # Perform merging
            self.current_operation = "merging"
            result = merger.merge(models, output_path, weights, **kwargs)
            self.current_operation = None
            
            return result
            
        except Exception as e:
            logger.error(f"Model merging failed: {e}")
            self.current_operation = None
            return False
    
    def quantize_model(self,
                      model_path: str,
                      output_path: str,
                      method: str = 'gptq',
                      bits: int = 4,
                      **kwargs) -> bool:
        """Quantize model."""
        try:
            # Create quantizer
            self.quantizer = Quantizer(method)
            
            # Perform quantization
            self.current_operation = "quantization"
            result = self.quantizer.quantize(model_path, output_path, bits, **kwargs)
            self.current_operation = None
            
            return result
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            self.current_operation = None
            return False
    
    def batch_process(self,
                     operations: List[Dict],
                     output_dir: str,
                     **kwargs) -> Dict[str, bool]:
        """Process multiple operations in batch."""
        results = {}
        
        for i, op in enumerate(operations):
            op_type = op.get('type')
            op_id = op.get('id', f"op_{i}")
            
            logger.info(f"Processing operation {i+1}/{len(operations)}: {op_type}")
            
            try:
                if op_type == 'convert':
                    result = self.convert_model(
                        input_path=op['input_path'],
                        output_path=os.path.join(output_dir, op.get('output_name', f"converted_{i}")),
                        target_format=op.get('target_format', 'hf'),
                        source_format=op.get('source_format'),
                        **kwargs
                    )
                elif op_type == 'merge':
                    result = self.merge_models(
                        models=op['models'],
                        output_path=os.path.join(output_dir, op.get('output_name', f"merged_{i}")),
                        method=op.get('method', 'linear'),
                        weights=op.get('weights'),
                        **kwargs
                    )
                elif op_type == 'quantize':
                    result = self.quantize_model(
                        model_path=op['model_path'],
                        output_path=os.path.join(output_dir, op.get('output_name', f"quantized_{i}")),
                        method=op.get('method', 'gptq'),
                        bits=op.get('bits', 4),
                        **kwargs
                    )
                else:
                    logger.warning(f"Unknown operation type: {op_type}")
                    result = False
                
                results[op_id] = result
                
            except Exception as e:
                logger.error(f"Batch operation {op_id} failed: {e}")
                results[op_id] = False
        
        return results


class ToolkitGUI:
    """Gradio web interface for the Model Toolkit."""
    
    def __init__(self, toolkit: ModelToolkit):
        self.toolkit = toolkit
        self.demo = None
        self.progress = 0
        self.status = "Ready"
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="forge Model Toolkit", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🦙 forge Model Conversion & Merging Toolkit")
            gr.Markdown("Unified toolkit for model conversion, merging, and quantization")
            
            with gr.Tabs():
                # Tab 1: Model Conversion
                with gr.TabItem("🔄 Model Conversion"):
                    with gr.Row():
                        with gr.Column():
                            convert_input = gr.Textbox(
                                label="Input Model Path or HuggingFace ID",
                                placeholder="e.g., meta-llama/Llama-2-7b-hf or /path/to/model"
                            )
                            convert_source_format = gr.Dropdown(
                                choices=["Auto-detect"] + list(SUPPORTED_FORMATS.keys()),
                                value="Auto-detect",
                                label="Source Format"
                            )
                            convert_target_format = gr.Dropdown(
                                choices=list(SUPPORTED_FORMATS.keys()),
                                value="hf",
                                label="Target Format"
                            )
                            convert_output = gr.Textbox(
                                label="Output Path",
                                placeholder="e.g., ./converted_model"
                            )
                            convert_btn = gr.Button("Convert Model", variant="primary")
                        
                        with gr.Column():
                            convert_output_log = gr.Textbox(
                                label="Conversion Log",
                                lines=10,
                                interactive=False
                            )
                    
                    convert_btn.click(
                        fn=self.run_conversion,
                        inputs=[convert_input, convert_source_format, convert_target_format, convert_output],
                        outputs=[convert_output_log]
                    )
                
                # Tab 2: Model Merging
                with gr.TabItem("🔀 Model Merging"):
                    with gr.Row():
                        with gr.Column():
                            merge_models = gr.Textbox(
                                label="Model Paths (one per line)",
                                placeholder="Enter model paths, one per line",
                                lines=5
                            )
                            merge_method = gr.Dropdown(
                                choices=list(MERGE_METHODS.keys()),
                                value="linear",
                                label="Merge Method"
                            )
                            merge_weights = gr.Textbox(
                                label="Weights (comma-separated, optional)",
                                placeholder="e.g., 0.5,0.3,0.2"
                            )
                            merge_output = gr.Textbox(
                                label="Output Path",
                                placeholder="e.g., ./merged_model"
                            )
                            
                            # Method-specific parameters
                            with gr.Accordion("Method Parameters", open=False):
                                merge_drop_rate = gr.Slider(
                                    minimum=0, maximum=1, value=0.5, step=0.05,
                                    label="DARE: Drop Rate",
                                    visible=False
                                )
                                merge_top_k = gr.Slider(
                                    minimum=0, maximum=1, value=0.2, step=0.05,
                                    label="TIES: Top-k Percentage",
                                    visible=False
                                )
                            
                            merge_btn = gr.Button("Merge Models", variant="primary")
                        
                        with gr.Column():
                            merge_output_log = gr.Textbox(
                                label="Merging Log",
                                lines=10,
                                interactive=False
                            )
                    
                    # Update visibility based on method
                    merge_method.change(
                        fn=self.update_merge_params,
                        inputs=[merge_method],
                        outputs=[merge_drop_rate, merge_top_k]
                    )
                    
                    merge_btn.click(
                        fn=self.run_merging,
                        inputs=[merge_models, merge_method, merge_weights, merge_output,
                               merge_drop_rate, merge_top_k],
                        outputs=[merge_output_log]
                    )
                
                # Tab 3: Model Quantization
                with gr.TabItem("📉 Model Quantization"):
                    with gr.Row():
                        with gr.Column():
                            quantize_input = gr.Textbox(
                                label="Input Model Path",
                                placeholder="e.g., ./merged_model or meta-llama/Llama-2-7b-hf"
                            )
                            quantize_method = gr.Dropdown(
                                choices=list(QUANTIZATION_METHODS.keys()),
                                value="gptq",
                                label="Quantization Method"
                            )
                            quantize_bits = gr.Dropdown(
                                choices=[2, 3, 4, 8, 16],
                                value=4,
                                label="Bit Width"
                            )
                            quantize_output = gr.Textbox(
                                label="Output Path",
                                placeholder="e.g., ./quantized_model"
                            )
                            
                            # Method-specific parameters
                            with gr.Accordion("Advanced Parameters", open=False):
                                quantize_group_size = gr.Slider(
                                    minimum=32, maximum=256, value=128, step=32,
                                    label="Group Size (for GPTQ/AWQ)"
                                )
                            
                            quantize_btn = gr.Button("Quantize Model", variant="primary")
                        
                        with gr.Column():
                            quantize_output_log = gr.Textbox(
                                label="Quantization Log",
                                lines=10,
                                interactive=False
                            )
                    
                    quantize_btn.click(
                        fn=self.run_quantization,
                        inputs=[quantize_input, quantize_method, quantize_bits,
                               quantize_output, quantize_group_size],
                        outputs=[quantize_output_log]
                    )
                
                # Tab 4: Batch Processing
                with gr.TabItem("📦 Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_config = gr.File(
                                label="Batch Configuration (JSON/YAML)",
                                file_types=[".json", ".yaml", ".yml"]
                            )
                            batch_output_dir = gr.Textbox(
                                label="Output Directory",
                                placeholder="e.g., ./batch_output"
                            )
                            batch_btn = gr.Button("Run Batch Processing", variant="primary")
                        
                        with gr.Column():
                            batch_output_log = gr.Textbox(
                                label="Batch Processing Log",
                                lines=15,
                                interactive=False
                            )
                    
                    batch_btn.click(
                        fn=self.run_batch_processing,
                        inputs=[batch_config, batch_output_dir],
                        outputs=[batch_output_log]
                    )
                
                # Tab 5: Model Info
                with gr.TabItem("ℹ️ Model Info"):
                    with gr.Row():
                        with gr.Column():
                            info_input = gr.Textbox(
                                label="Model Path or HuggingFace ID",
                                placeholder="Enter model path to analyze"
                            )
                            info_btn = gr.Button("Analyze Model", variant="primary")
                        
                        with gr.Column():
                            info_output = gr.JSON(
                                label="Model Information"
                            )
                    
                    info_btn.click(
                        fn=self.get_model_info,
                        inputs=[info_input],
                        outputs=[info_output]
                    )
            
            # Progress bar
            progress_bar = gr.Slider(
                minimum=0, maximum=100, value=0, 
                label="Progress", interactive=False
            )
            status_text = gr.Textbox(
                label="Status", value="Ready", interactive=False
            )
            
            # Footer
            gr.Markdown("""
            ---
            ### Supported Formats
            - **Source/Target**: HuggingFace, LLaMA, Qwen, Baichuan, BLOOM, MPT, Falcon, Mistral, GPT-2, OPT, Phi, Gemma
            
            ### Merge Methods
            - **Linear**: Simple weighted average
            - **DARE**: Drop And REscale (random dropout with rescaling)
            - **TIES**: TRIM, ELECT SIGN & MERGE (sign-based merging)
            - **Task Arithmetic**: Task vector arithmetic
            - **SLERP**: Spherical Linear Interpolation
            - **Model Stock**: Model averaging
            
            ### Quantization Methods
            - **GPTQ**: Post-training quantization
            - **AWQ**: Activation-aware weight quantization
            - **BitsAndBytes**: 4/8-bit quantization
            - **GGUF**: GGML Universal Format
            - **ExLlamaV2**: ExLlamaV2 format
            """)
        
        self.demo = demo
        return demo
    
    def update_merge_params(self, method):
        """Update merge parameter visibility based on selected method."""
        return (
            gr.update(visible=(method == 'dare')),
            gr.update(visible=(method == 'ties'))
        )
    
    def run_conversion(self, input_path, source_format, target_format, output_path):
        """Run model conversion."""
        if not input_path or not output_path:
            return "Error: Please provide input and output paths"
        
        if source_format == "Auto-detect":
            source_format = None
        
        try:
            self.status = "Converting model..."
            result = self.toolkit.convert_model(
                input_path=input_path,
                output_path=output_path,
                target_format=target_format,
                source_format=source_format
            )
            
            if result:
                return f"✅ Conversion completed successfully!\nOutput saved to: {output_path}"
            else:
                return "❌ Conversion failed. Check logs for details."
                
        except Exception as e:
            return f"❌ Error during conversion: {str(e)}"
    
    def run_merging(self, models_text, method, weights_text, output_path, drop_rate, top_k):
        """Run model merging."""
        if not models_text or not output_path:
            return "Error: Please provide model paths and output path"
        
        # Parse models
        models = [m.strip() for m in models_text.split('\n') if m.strip()]
        if len(models) < 2:
            return "Error: At least 2 models are required for merging"
        
        # Parse weights
        weights = None
        if weights_text:
            try:
                weights = [float(w.strip()) for w in weights_text.split(',')]
                if len(weights) != len(models):
                    return f"Error: Number of weights ({len(weights)}) doesn't match number of models ({len(models)})"
            except ValueError:
                return "Error: Invalid weights format. Use comma-separated numbers."
        
        # Prepare kwargs
        kwargs = {}
        if method == 'dare':
            kwargs['drop_rate'] = drop_rate
        elif method == 'ties':
            kwargs['top_k'] = top_k
        
        try:
            self.status = "Merging models..."
            result = self.toolkit.merge_models(
                models=models,
                output_path=output_path,
                method=method,
                weights=weights,
                **kwargs
            )
            
            if result:
                return f"✅ Merging completed successfully!\nOutput saved to: {output_path}"
            else:
                return "❌ Merging failed. Check logs for details."
                
        except Exception as e:
            return f"❌ Error during merging: {str(e)}"
    
    def run_quantization(self, model_path, method, bits, output_path, group_size):
        """Run model quantization."""
        if not model_path or not output_path:
            return "Error: Please provide input and output paths"
        
        try:
            self.status = "Quantizing model..."
            result = self.toolkit.quantize_model(
                model_path=model_path,
                output_path=output_path,
                method=method,
                bits=int(bits),
                group_size=int(group_size)
            )
            
            if result:
                return f"✅ Quantization completed successfully!\nOutput saved to: {output_path}"
            else:
                return "❌ Quantization failed. Check logs for details."
                
        except Exception as e:
            return f"❌ Error during quantization: {str(e)}"
    
    def run_batch_processing(self, config_file, output_dir):
        """Run batch processing from configuration file."""
        if not config_file or not output_dir:
            return "Error: Please provide configuration file and output directory"
        
        try:
            # Load configuration
            if config_file.name.endswith('.json'):
                with open(config_file.name, 'r') as f:
                    config = json.load(f)
            elif config_file.name.endswith(('.yaml', '.yml')):
                with open(config_file.name, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                return "Error: Unsupported file format. Use JSON or YAML."
            
            operations = config.get('operations', [])
            if not operations:
                return "Error: No operations found in configuration"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            self.status = "Running batch processing..."
            results = self.toolkit.batch_process(operations, output_dir)
            
            # Format results
            output_lines = ["Batch Processing Results:"]
            output_lines.append("=" * 50)
            
            success_count = sum(1 for r in results.values() if r)
            total_count = len(results)
            
            for op_id, result in results.items():
                status = "✅ Success" if result else "❌ Failed"
                output_lines.append(f"{op_id}: {status}")
            
            output_lines.append("=" * 50)
            output_lines.append(f"Summary: {success_count}/{total_count} operations successful")
            
            return '\n'.join(output_lines)
            
        except Exception as e:
            return f"❌ Error during batch processing: {str(e)}"
    
    def get_model_info(self, model_path):
        """Get model information."""
        if not model_path:
            return {"error": "Please provide a model path"}
        
        try:
            # Detect format
            fmt, config = self.toolkit.detect_format(model_path)
            
            # Get additional info
            info = {
                "format": fmt,
                "format_name": SUPPORTED_FORMATS.get(fmt, "Unknown"),
                "config": config,
            }
            
            # Try to get model size
            if os.path.exists(model_path):
                total_size = 0
                file_count = 0
                
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                
                info["file_count"] = file_count
                info["total_size_gb"] = total_size / (1024**3)
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if self.demo is None:
            self.create_interface()
        
        self.demo.launch(**kwargs)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="forge Model Conversion & Merging Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert model to HuggingFace format
  python merger.py convert --input meta-llama/Llama-2-7b-hf --output ./llama2-7b-hf --target hf
  
  # Merge multiple models
  python merger.py merge --models model1 model2 model3 --output ./merged --method linear --weights 0.5 0.3 0.2
  
  # Quantize model
  python merger.py quantize --input ./merged --output ./quantized --method gptq --bits 4
  
  # Launch GUI
  python merger.py gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert model format')
    convert_parser.add_argument('--input', required=True, help='Input model path or HuggingFace ID')
    convert_parser.add_argument('--output', required=True, help='Output path')
    convert_parser.add_argument('--target', default='hf', choices=SUPPORTED_FORMATS.keys(),
                               help='Target format')
    convert_parser.add_argument('--source', default=None, help='Source format (auto-detect if not provided)')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple models')
    merge_parser.add_argument('--models', nargs='+', required=True, help='Model paths to merge')
    merge_parser.add_argument('--output', required=True, help='Output path')
    merge_parser.add_argument('--method', default='linear', choices=MERGE_METHODS.keys(),
                             help='Merge method')
    merge_parser.add_argument('--weights', nargs='+', type=float, help='Merge weights')
    
    # Quantize command
    quantize_parser = subparsers.add_parser('quantize', help='Quantize model')
    quantize_parser.add_argument('--input', required=True, help='Input model path')
    quantize_parser.add_argument('--output', required=True, help='Output path')
    quantize_parser.add_argument('--method', default='gptq', choices=QUANTIZATION_METHODS.keys(),
                                help='Quantization method')
    quantize_parser.add_argument('--bits', type=int, default=4, help='Bit width')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch web interface')
    gui_parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    gui_parser.add_argument('--share', action='store_true', help='Create public link')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch processing')
    batch_parser.add_argument('--config', required=True, help='Configuration file (JSON/YAML)')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create toolkit
    toolkit = ModelToolkit()
    
    if args.command == 'convert':
        result = toolkit.convert_model(
            input_path=args.input,
            output_path=args.output,
            target_format=args.target,
            source_format=args.source
        )
        sys.exit(0 if result else 1)
    
    elif args.command == 'merge':
        result = toolkit.merge_models(
            models=args.models,
            output_path=args.output,
            method=args.method,
            weights=args.weights
        )
        sys.exit(0 if result else 1)
    
    elif args.command == 'quantize':
        result = toolkit.quantize_model(
            model_path=args.input,
            output_path=args.output,
            method=args.method,
            bits=args.bits
        )
        sys.exit(0 if result else 1)
    
    elif args.command == 'gui':
        gui = ToolkitGUI(toolkit)
        gui.launch(server_port=args.port, share=args.share)
    
    elif args.command == 'batch':
        # Load configuration
        if args.config.endswith('.json'):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        operations = config.get('operations', [])
        results = toolkit.batch_process(operations, args.output_dir)
        
        # Print results
        print("Batch Processing Results:")
        print("=" * 50)
        for op_id, result in results.items():
            status = "Success" if result else "Failed"
            print(f"{op_id}: {status}")
        
        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)
        print(f"\nSummary: {success_count}/{total_count} operations successful")
        
        sys.exit(0 if success_count == total_count else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()