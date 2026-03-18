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
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import gradio as gr
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils import check_min_version


try:
    check_min_version("4.34.0")
except Exception:
    raise ValueError("Please upgrade `transformers` to 4.34.0")


CONFIG_NAME = "config.json"


class ModelConverter(ABC):
    """Base class for model converters with plugin architecture."""
    
    @abstractmethod
    def detect_format(self, input_dir: str) -> bool:
        """Detect if input directory contains model in supported format."""
        pass
    
    @abstractmethod
    def convert(self, input_dir: str, output_dir: str, shard_size: str = "2GB", 
                save_safetensors: bool = False, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Convert model and return (torch_dtype, config_dict)."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported format names."""
        pass


class QwenToLlamaConverter(ModelConverter):
    """Converter for Qwen models to LLaMA format."""
    
    def detect_format(self, input_dir: str) -> bool:
        """Check if directory contains Qwen model."""
        config_path = os.path.join(input_dir, CONFIG_NAME)
        if not os.path.exists(config_path):
            return False
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        return config.get("model_type") == "qwen"
    
    def get_supported_formats(self) -> List[str]:
        return ["qwen_to_llama"]
    
    def convert(self, input_dir: str, output_dir: str, shard_size: str = "2GB", 
                save_safetensors: bool = False, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Convert Qwen model to LLaMA format."""
        os.makedirs(output_dir, exist_ok=True)
        
        torch_dtype = self._save_weight(input_dir, output_dir, shard_size, save_safetensors)
        config_dict = self._save_config(input_dir, output_dir, torch_dtype)
        
        return torch_dtype, config_dict
    
    def _save_weight(self, input_dir: str, output_dir: str, shard_size: str, 
                    save_safetensors: bool) -> str:
        """Save converted weights."""
        qwen_state_dict: Dict[str, torch.Tensor] = OrderedDict()
        for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
            if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".safetensors"):
                with safe_open(os.path.join(input_dir, filepath), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        qwen_state_dict[key] = f.get_tensor(key)
        
        llama_state_dict: Dict[str, torch.Tensor] = OrderedDict()
        torch_dtype = None
        for key, value in tqdm(qwen_state_dict.items(), desc="Convert format"):
            if torch_dtype is None:
                torch_dtype = value.dtype
            if "wte" in key:
                llama_state_dict["model.embed_tokens.weight"] = value
            elif "ln_f" in key:
                llama_state_dict["model.norm.weight"] = value
            else:
                key = key.replace("transformer.h", "model.layers")
                if "attn.c_attn" in key:
                    proj_size = value.size(0) // 3
                    llama_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
                    llama_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value[
                        proj_size : 2 * proj_size, ...
                    ]
                    llama_state_dict[key.replace("attn.c_attn", "self_attn.v_proj")] = value[2 * proj_size :, ...]
                elif "attn.c_proj" in key:
                    llama_state_dict[key.replace("attn.c_proj", "self_attn.o_proj")] = value
                    llama_state_dict[key.replace("attn.c_proj.weight", "self_attn.o_proj.bias")] = torch.zeros_like(
                        value[:, 0]
                    ).squeeze()
                elif "ln_1" in key:
                    llama_state_dict[key.replace("ln_1", "input_layernorm")] = value
                elif "ln_2" in key:
                    llama_state_dict[key.replace("ln_2", "post_attention_layernorm")] = value
                elif "mlp.w1" in key:
                    llama_state_dict[key.replace("mlp.w1", "mlp.up_proj")] = value
                elif "mlp.w2" in key:
                    llama_state_dict[key.replace("mlp.w2", "mlp.gate_proj")] = value
                elif "mlp.c_proj" in key:
                    llama_state_dict[key.replace("mlp.c_proj", "mlp.down_proj")] = value
                elif "lm_head" in key:
                    llama_state_dict[key] = value
                else:
                    raise KeyError(f"Unable to process key {key}")
        
        weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            llama_state_dict, filename_pattern=filename_pattern, max_shard_size=shard_size
        )
        for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="Save weights"):
            shard = {tensor: llama_state_dict[tensor].contiguous() for tensor in tensors}
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
        
        return str(torch_dtype).replace("torch.", "")
    
    def _save_config(self, input_dir: str, output_dir: str, torch_dtype: str) -> Dict[str, Any]:
        """Save converted config."""
        with open(os.path.join(input_dir, CONFIG_NAME), encoding="utf-8") as f:
            qwen_config_dict: Dict[str, Any] = json.load(f)
        
        llama2_config_dict: Dict[str, Any] = OrderedDict()
        llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
        llama2_config_dict["hidden_act"] = "silu"
        llama2_config_dict["hidden_size"] = qwen_config_dict["hidden_size"]
        llama2_config_dict["initializer_range"] = qwen_config_dict["initializer_range"]
        llama2_config_dict["intermediate_size"] = qwen_config_dict["intermediate_size"] // 2
        llama2_config_dict["max_position_embeddings"] = qwen_config_dict["max_position_embeddings"]
        llama2_config_dict["model_type"] = "llama"
        llama2_config_dict["num_attention_heads"] = qwen_config_dict["num_attention_heads"]
        llama2_config_dict["num_hidden_layers"] = qwen_config_dict["num_hidden_layers"]
        llama2_config_dict["num_key_value_heads"] = qwen_config_dict["hidden_size"] // qwen_config_dict["kv_channels"]
        llama2_config_dict["pretraining_tp"] = 1
        llama2_config_dict["rms_norm_eps"] = qwen_config_dict["layer_norm_epsilon"]
        llama2_config_dict["rope_scaling"] = None
        llama2_config_dict["tie_word_embeddings"] = qwen_config_dict["tie_word_embeddings"]
        llama2_config_dict["torch_dtype"] = torch_dtype
        llama2_config_dict["transformers_version"] = "4.34.0"
        llama2_config_dict["use_cache"] = True
        llama2_config_dict["vocab_size"] = qwen_config_dict["vocab_size"]
        llama2_config_dict["attention_bias"] = True
        
        with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(llama2_config_dict, f, indent=2)
        
        print(f"Model config saved in {os.path.join(output_dir, CONFIG_NAME)}")
        return llama2_config_dict


class ModelMerger:
    """Handles model merging operations."""
    
    @staticmethod
    def merge_dare(models: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        """Merge models using DARE (Drop And REscale) method."""
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("Weights must be between 0 and 1")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        merged = {}
        for key in models[0].keys():
            tensors = [model[key] for model in models]
            # DARE: randomly drop weights and rescale
            merged_tensor = torch.zeros_like(tensors[0])
            for tensor, weight in zip(tensors, weights):
                mask = torch.bernoulli(torch.full_like(tensor, weight))
                merged_tensor += tensor * mask
            merged[key] = merged_tensor
        
        return merged
    
    @staticmethod
    def merge_ties(models: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        """Merge models using TIES (Task Arithmetic) method."""
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        merged = {}
        for key in models[0].keys():
            tensors = [model[key] for model in models]
            # TIES: weighted sum with sign consensus
            merged_tensor = torch.zeros_like(tensors[0])
            signs = torch.stack([torch.sign(t) for t in tensors])
            consensus = torch.mode(signs, dim=0).values
            
            for tensor, weight in zip(tensors, weights):
                # Only keep weights that agree with consensus
                mask = (torch.sign(tensor) == consensus).float()
                merged_tensor += tensor * weight * mask
            
            merged[key] = merged_tensor
        
        return merged
    
    @staticmethod
    def merge_linear(models: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        """Merge models using linear interpolation."""
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        merged = {}
        for key in models[0].keys():
            tensors = [model[key] for model in models]
            merged_tensor = torch.zeros_like(tensors[0])
            for tensor, weight in zip(tensors, weights):
                merged_tensor += tensor * weight
            merged[key] = merged_tensor
        
        return merged


class ModelQuantizer:
    """Handles model quantization operations."""
    
    @staticmethod
    def quantize_gguf(model_path: str, output_path: str, quantization: str = "Q4_K_M"):
        """Quantize model to GGUF format (requires llama.cpp)."""
        try:
            from llama_cpp import llama_cpp as llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for GGUF quantization")
        
        # This is a placeholder - actual implementation would use llama.cpp
        print(f"Quantizing {model_path} to {output_path} with {quantization}")
        # Implementation would go here
    
    @staticmethod
    def quantize_awq(model_path: str, output_path: str, w_bit: int = 4, group_size: int = 128):
        """Quantize model using AWQ."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("autoawq is required for AWQ quantization")
        
        print(f"Quantizing {model_path} to AWQ format")
        # Implementation would go here
    
    @staticmethod
    def quantize_gptq(model_path: str, output_path: str, bits: int = 4, group_size: int = 128):
        """Quantize model using GPTQ."""
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError:
            raise ImportError("auto-gptq is required for GPTQ quantization")
        
        print(f"Quantizing {model_path} to GPTQ format")
        # Implementation would go here


class ConversionToolkit:
    """Unified toolkit for model conversion, merging, and quantization."""
    
    def __init__(self):
        self.converters = {
            "qwen_to_llama": QwenToLlamaConverter(),
        }
        self.merger = ModelMerger()
        self.quantizer = ModelQuantizer()
    
    def auto_detect_format(self, input_dir: str) -> Optional[str]:
        """Automatically detect model format."""
        for format_name, converter in self.converters.items():
            if converter.detect_format(input_dir):
                return format_name
        return None
    
    def convert_model(self, input_dir: str, output_dir: str, format_name: Optional[str] = None,
                     shard_size: str = "2GB", save_safetensors: bool = False, **kwargs) -> Dict[str, Any]:
        """Convert model with automatic format detection."""
        if format_name is None:
            format_name = self.auto_detect_format(input_dir)
            if format_name is None:
                raise ValueError(f"Could not detect model format in {input_dir}")
        
        if format_name not in self.converters:
            raise ValueError(f"Unsupported format: {format_name}")
        
        converter = self.converters[format_name]
        torch_dtype, config_dict = converter.convert(
            input_dir, output_dir, shard_size, save_safetensors, **kwargs
        )
        
        return {
            "format": format_name,
            "torch_dtype": torch_dtype,
            "config": config_dict,
            "output_dir": output_dir
        }
    
    def batch_convert(self, input_dirs: List[str], output_base_dir: str, 
                     format_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Batch convert multiple models."""
        results = []
        for i, input_dir in enumerate(input_dirs):
            output_dir = os.path.join(output_base_dir, f"model_{i}")
            try:
                result = self.convert_model(input_dir, output_dir, format_name, **kwargs)
                results.append({"status": "success", **result})
            except Exception as e:
                results.append({
                    "status": "error",
                    "input_dir": input_dir,
                    "error": str(e)
                })
        return results
    
    def merge_models(self, model_paths: List[str], output_dir: str, 
                    method: str = "linear", weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Merge multiple models."""
        if weights is None:
            weights = [1.0] * len(model_paths)
        
        if len(weights) != len(model_paths):
            raise ValueError("Number of weights must match number of models")
        
        # Load all models
        models = []
        for path in model_paths:
            model_dict = self._load_model_weights(path)
            models.append(model_dict)
        
        # Merge based on method
        if method == "dare":
            merged = self.merger.merge_dare(models, weights)
        elif method == "ties":
            merged = self.merger.merge_ties(models, weights)
        elif method == "linear":
            merged = self.merger.merge_linear(models, weights)
        else:
            raise ValueError(f"Unsupported merge method: {method}")
        
        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        self._save_model_weights(merged, output_dir)
        
        return {
            "method": method,
            "weights": weights,
            "output_dir": output_dir,
            "num_models": len(model_paths)
        }
    
    def _load_model_weights(self, model_path: str) -> Dict[str, torch.Tensor]:
        """Load model weights from directory."""
        weights = {}
        
        # Try safetensors first
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        if safetensor_files:
            for file in safetensor_files:
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
        else:
            # Try pytorch bin files
            bin_files = list(Path(model_path).glob("*.bin"))
            for file in bin_files:
                data = torch.load(file, map_location="cpu")
                weights.update(data)
        
        return weights
    
    def _save_model_weights(self, weights: Dict[str, torch.Tensor], output_dir: str):
        """Save model weights to directory."""
        # Save as single safetensors file for simplicity
        save_file(weights, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})
        print(f"Model saved to {output_dir}")


def create_gradio_interface():
    """Create Gradio web interface for the toolkit."""
    toolkit = ConversionToolkit()
    
    with gr.Blocks(title="forge Model Toolkit", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🦙 forge Model Conversion & Merging Toolkit")
        gr.Markdown("Unified toolkit for model conversion, merging, and quantization")
        
        with gr.Tabs():
            with gr.TabItem("🔄 Model Conversion"):
                with gr.Row():
                    with gr.Column():
                        convert_input_dir = gr.Textbox(label="Input Directory", 
                                                      placeholder="Path to model directory")
                        convert_output_dir = gr.Textbox(label="Output Directory",
                                                       placeholder="Path for converted model")
                        convert_format = gr.Dropdown(
                            choices=["auto"] + list(toolkit.converters.keys()),
                            value="auto",
                            label="Model Format"
                        )
                        convert_shard_size = gr.Textbox(value="2GB", label="Shard Size")
                        convert_safetensors = gr.Checkbox(value=True, label="Save as Safetensors")
                        convert_btn = gr.Button("Convert Model", variant="primary")
                    
                    with gr.Column():
                        convert_output = gr.JSON(label="Conversion Results")
                
                def run_conversion(input_dir, output_dir, format_name, shard_size, save_safetensors):
                    try:
                        if format_name == "auto":
                            format_name = None
                        result = toolkit.convert_model(
                            input_dir, output_dir, format_name, shard_size, save_safetensors
                        )
                        return {"status": "success", **result}
                    except Exception as e:
                        return {"status": "error", "error": str(e)}
                
                convert_btn.click(
                    run_conversion,
                    inputs=[convert_input_dir, convert_output_dir, convert_format, 
                           convert_shard_size, convert_safetensors],
                    outputs=convert_output
                )
            
            with gr.TabItem("🔀 Model Merging"):
                with gr.Row():
                    with gr.Column():
                        merge_model_paths = gr.Textbox(
                            label="Model Paths (comma-separated)",
                            placeholder="path1,path2,path3"
                        )
                        merge_output_dir = gr.Textbox(label="Output Directory")
                        merge_method = gr.Dropdown(
                            choices=["linear", "dare", "ties"],
                            value="linear",
                            label="Merge Method"
                        )
                        merge_weights = gr.Textbox(
                            label="Weights (comma-separated, optional)",
                            placeholder="1.0,1.0,1.0"
                        )
                        merge_btn = gr.Button("Merge Models", variant="primary")
                    
                    with gr.Column():
                        merge_output = gr.JSON(label="Merge Results")
                
                def run_merging(model_paths_str, output_dir, method, weights_str):
                    try:
                        model_paths = [p.strip() for p in model_paths_str.split(",")]
                        weights = None
                        if weights_str:
                            weights = [float(w.strip()) for w in weights_str.split(",")]
                        
                        result = toolkit.merge_models(model_paths, output_dir, method, weights)
                        return {"status": "success", **result}
                    except Exception as e:
                        return {"status": "error", "error": str(e)}
                
                merge_btn.click(
                    run_merging,
                    inputs=[merge_model_paths, merge_output_dir, merge_method, merge_weights],
                    outputs=merge_output
                )
            
            with gr.TabItem("⚡ Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        batch_input_dir = gr.Textbox(label="Input Base Directory",
                                                    placeholder="Directory containing multiple models")
                        batch_output_dir = gr.Textbox(label="Output Base Directory")
                        batch_format = gr.Dropdown(
                            choices=["auto"] + list(toolkit.converters.keys()),
                            value="auto",
                            label="Model Format"
                        )
                        batch_btn = gr.Button("Batch Convert", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.JSON(label="Batch Results")
                
                def run_batch(input_base_dir, output_base_dir, format_name):
                    try:
                        # Find all model directories
                        model_dirs = []
                        for item in Path(input_base_dir).iterdir():
                            if item.is_dir() and (item / CONFIG_NAME).exists():
                                model_dirs.append(str(item))
                        
                        if not model_dirs:
                            return {"status": "error", "error": "No model directories found"}
                        
                        if format_name == "auto":
                            format_name = None
                        
                        results = toolkit.batch_convert(model_dirs, output_base_dir, format_name)
                        return {
                            "status": "success",
                            "total_models": len(model_dirs),
                            "results": results
                        }
                    except Exception as e:
                        return {"status": "error", "error": str(e)}
                
                batch_btn.click(
                    run_batch,
                    inputs=[batch_input_dir, batch_output_dir, batch_format],
                    outputs=batch_output
                )
            
            with gr.TabItem("📊 Format Detection"):
                with gr.Row():
                    with gr.Column():
                        detect_input_dir = gr.Textbox(label="Input Directory")
                        detect_btn = gr.Button("Detect Format", variant="primary")
                    
                    with gr.Column():
                        detect_output = gr.JSON(label="Detection Results")
                
                def run_detection(input_dir):
                    try:
                        format_name = toolkit.auto_detect_format(input_dir)
                        if format_name:
                            return {
                                "status": "success",
                                "detected_format": format_name,
                                "input_dir": input_dir
                            }
                        else:
                            return {
                                "status": "not_detected",
                                "input_dir": input_dir,
                                "message": "No supported format detected"
                            }
                    except Exception as e:
                        return {"status": "error", "error": str(e)}
                
                detect_btn.click(
                    run_detection,
                    inputs=[detect_input_dir],
                    outputs=detect_output
                )
        
        gr.Markdown("### Supported Formats")
        gr.Markdown("- Qwen to LLaMA conversion")
        gr.Markdown("- Model merging (Linear, DARE, TIES)")
        gr.Markdown("- Batch processing")
        gr.Markdown("- Automatic format detection")
    
    return demo


def llamafy_qwen(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = False,
):
    r"""Convert the Qwen models in the same format as LLaMA2.

    Usage: python llamafy_qwen.py --input_dir input --output_dir output
    Converted model: https://huggingface.co/hiyouga/Qwen-14B-Chat-LLaMAfied
    """
    toolkit = ConversionToolkit()
    result = toolkit.convert_model(input_dir, output_dir, "qwen_to_llama", shard_size, save_safetensors)
    print(f"Conversion completed: {result}")


def launch_gradio(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch Gradio web interface."""
    demo = create_gradio_interface()
    demo.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        # Launch Gradio interface
        import argparse
        parser = argparse.ArgumentParser(description="Launch Gradio interface")
        parser.add_argument("--share", action="store_true", help="Create a public link")
        parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
        parser.add_argument("--server_port", type=int, default=7860, help="Server port")
        args = parser.parse_args(sys.argv[2:])
        launch_gradio(args.share, args.server_name, args.server_port)
    else:
        # Use Fire for CLI
        fire.Fire({
            "llamafy_qwen": llamafy_qwen,
            "convert": ConversionToolkit().convert_model,
            "batch_convert": ConversionToolkit().batch_convert,
            "merge": ConversionToolkit().merge_models,
            "detect": ConversionToolkit().auto_detect_format,
            "gradio": launch_gradio
        })