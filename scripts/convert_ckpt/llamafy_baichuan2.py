# Copyright 2025 the forge team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

import fire
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CONFIG_NAME = "config.json"


# Transformation registry for different model architectures
TRANSFORMATION_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_transformation(model_type: str, patterns: Dict[str, str], config_transform: Optional[Callable] = None):
    """Register a transformation pattern for a specific model architecture."""
    def decorator(func: Callable) -> Callable:
        TRANSFORMATION_REGISTRY[model_type] = {
            "transform": func,
            "patterns": patterns,
            "config_transform": config_transform
        }
        return func
    return decorator


def detect_model_architecture(input_dir: str) -> Tuple[str, Dict[str, Any]]:
    """Auto-detect model architecture from config.json."""
    config_path = os.path.join(input_dir, CONFIG_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    
    # Try to detect from architectures field first
    architectures = config.get("architectures", [])
    if architectures:
        arch_name = architectures[0].lower()
        if "baichuan" in arch_name:
            return "baichuan", config
        elif "qwen" in arch_name:
            return "qwen", config
        elif "llama" in arch_name:
            return "llama", config
    
    # Fallback to model_type field
    model_type = config.get("model_type", "").lower()
    if "baichuan" in model_type:
        return "baichuan", config
    elif "qwen" in model_type:
        return "qwen", config
    elif "llama" in model_type:
        return "llama", config
    
    # Try pattern matching on weight keys
    weight_files = [f for f in os.listdir(input_dir) if f.endswith((".bin", ".safetensors"))]
    if weight_files:
        # Load first weight file to inspect keys
        first_file = os.path.join(input_dir, weight_files[0])
        if first_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sample_weights = load_file(first_file, device="cpu")
        else:
            sample_weights = torch.load(first_file, map_location="cpu", weights_only=True)
        
        keys = list(sample_weights.keys())
        if any("W_pack" in key for key in keys):
            return "baichuan", config
        elif any("c_attn" in key for key in keys):
            return "qwen", config
    
    raise ValueError(f"Could not detect model architecture from config: {config.get('model_type', 'unknown')}")


@register_transformation(
    model_type="baichuan",
    patterns={
        "W_pack": r"W_pack",
        "lm_head": r"lm_head"
    }
)
def transform_baichuan(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Transform Baichuan2 weights to LLaMA format."""
    llama_state_dict = OrderedDict()
    
    for key, value in tqdm(state_dict.items(), desc="Converting Baichuan2 weights"):
        if "W_pack" in key:
            proj_size = value.size(0) // 3
            llama_state_dict[key.replace("W_pack", "q_proj")] = value[:proj_size, :]
            llama_state_dict[key.replace("W_pack", "k_proj")] = value[proj_size : 2 * proj_size, :]
            llama_state_dict[key.replace("W_pack", "v_proj")] = value[2 * proj_size :, :]
        elif "lm_head" in key:
            llama_state_dict[key] = torch.nn.functional.normalize(value)
        else:
            llama_state_dict[key] = value
    
    return llama_state_dict


@register_transformation(
    model_type="qwen",
    patterns={
        "c_attn": r"c_attn",
        "c_proj": r"c_proj",
        "wte": r"wte",
        "lm_head": r"lm_head"
    }
)
def transform_qwen(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Transform Qwen weights to LLaMA format."""
    llama_state_dict = OrderedDict()
    
    for key, value in tqdm(state_dict.items(), desc="Converting Qwen weights"):
        if "c_attn" in key:
            # Qwen uses combined attention projection
            proj_size = value.size(0) // 3
            llama_state_dict[key.replace("c_attn", "q_proj")] = value[:proj_size, :]
            llama_state_dict[key.replace("c_attn", "k_proj")] = value[proj_size : 2 * proj_size, :]
            llama_state_dict[key.replace("c_attn", "v_proj")] = value[2 * proj_size :, :]
        elif "c_proj" in key:
            llama_state_dict[key.replace("c_proj", "o_proj")] = value
        elif "wte" in key:
            llama_state_dict[key.replace("wte", "embed_tokens")] = value
        elif "lm_head" in key:
            llama_state_dict[key] = value
        else:
            # Copy other weights as-is
            llama_state_dict[key] = value
    
    return llama_state_dict


@register_transformation(
    model_type="llama",
    patterns={},
    config_transform=lambda config: config  # No transformation needed for LLaMA
)
def transform_identity(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Identity transformation for models already in LLaMA format."""
    return state_dict


def transform_config_baichuan(config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Baichuan2 config to LLaMA format."""
    llama_config = config.copy()
    llama_config["architectures"] = ["LlamaForCausalLM"]
    llama_config.pop("auto_map", None)
    llama_config.pop("tokenizer_class", None)
    llama_config["model_type"] = "llama"
    return llama_config


def transform_config_qwen(config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Qwen config to LLaMA format."""
    llama_config = config.copy()
    llama_config["architectures"] = ["LlamaForCausalLM"]
    llama_config.pop("auto_map", None)
    llama_config.pop("tokenizer_class", None)
    llama_config["model_type"] = "llama"
    
    # Qwen-specific config adjustments
    if "kv_channels" in llama_config:
        llama_config["num_key_value_heads"] = llama_config.get("num_attention_heads", 32) // llama_config["kv_channels"]
        llama_config.pop("kv_channels", None)
    
    if "seq_length" in llama_config:
        llama_config["max_position_embeddings"] = llama_config.pop("seq_length")
    
    return llama_config


def load_state_dict(input_dir: str) -> Dict[str, torch.Tensor]:
    """Load model weights from input directory."""
    state_dict = OrderedDict()
    weight_files = []
    
    # Collect all weight files
    for filename in tqdm(os.listdir(input_dir), desc="Scanning weight files"):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath) and (filename.endswith(".bin") or filename.endswith(".safetensors")):
            weight_files.append(filepath)
    
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {input_dir}")
    
    # Load weights with progress tracking
    for filepath in tqdm(weight_files, desc="Loading weights"):
        if filepath.endswith(".safetensors"):
            from safetensors.torch import load_file
            shard_weight = load_file(filepath, device="cpu")
        else:
            shard_weight = torch.load(filepath, map_location="cpu", weights_only=True)
        state_dict.update(shard_weight)
    
    return state_dict


def save_weight(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    shard_size: str,
    save_safetensors: bool,
    model_type: str
):
    """Save transformed weights with sharding and validation."""
    # Validate transformation
    if model_type != "llama":
        # Check if transformation was applied correctly
        sample_keys = list(state_dict.keys())[:10]
        if model_type == "baichuan" and any("W_pack" in key for key in sample_keys):
            print("Warning: Baichuan transformation may not have been applied correctly")
        elif model_type == "qwen" and any("c_attn" in key for key in sample_keys):
            print("Warning: Qwen transformation may not have been applied correctly")
    
    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
    
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=shard_size
    )
    
    for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="Saving weights"):
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))
    
    if not state_dict_split.is_sharded:
        print(f"Model weights saved in {os.path.join(output_dir, weights_name)}.")
    else:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print(f"Model weights saved in {output_dir}.")


def save_config(config: Dict[str, Any], output_dir: str):
    """Save transformed config to output directory."""
    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved in {os.path.join(output_dir, CONFIG_NAME)}")


def convert_model(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = True,
    target_architecture: str = "llama"
):
    r"""Convert various model architectures to LLaMA format with auto-detection.
    
    Usage: python convert_model.py --input_dir input --output_dir output
    Supports: Baichuan2, Qwen, and other architectures with registered transformations
    """
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise RuntimeError(f"Output directory already exists or cannot be created: {e}")
    
    # Auto-detect model architecture
    print("Detecting model architecture...")
    model_type, config = detect_model_architecture(input_dir)
    print(f"Detected model type: {model_type}")
    
    # Get transformation functions from registry
    if model_type not in TRANSFORMATION_REGISTRY:
        raise ValueError(f"No transformation registered for model type: {model_type}")
    
    transformation_info = TRANSFORMATION_REGISTRY[model_type]
    weight_transform = transformation_info["transform"]
    config_transform = transformation_info.get("config_transform")
    
    # Load and transform weights
    print("Loading weights...")
    state_dict = load_state_dict(input_dir)
    
    print(f"Applying {model_type} transformation...")
    transformed_state_dict = weight_transform(state_dict)
    
    # Transform config
    if config_transform:
        transformed_config = config_transform(config)
    else:
        # Default config transformation to LLaMA
        transformed_config = config.copy()
        transformed_config["architectures"] = ["LlamaForCausalLM"]
        transformed_config.pop("auto_map", None)
        transformed_config.pop("tokenizer_class", None)
        transformed_config["model_type"] = target_architecture
    
    # Save transformed weights and config
    print("Saving transformed model...")
    save_weight(transformed_state_dict, output_dir, shard_size, save_safetensors, model_type)
    save_config(transformed_config, output_dir)
    
    print(f"Conversion complete! Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(convert_model)