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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CONFIG_NAME = "config.json"


class BaseConverter:
    """Base class for model converters with plugin architecture."""
    
    def __init__(self, input_dir: str, output_dir: str, shard_size: str = "2GB", save_safetensors: bool = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.save_safetensors = save_safetensors
        
    def detect_format(self) -> str:
        """Detect model format automatically."""
        raise NotImplementedError
        
    def convert(self) -> None:
        """Main conversion method."""
        raise NotImplementedError
        
    def save_weight(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Save weights with sharding support."""
        weights_name = SAFE_WEIGHTS_NAME if self.save_safetensors else WEIGHTS_NAME
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=self.shard_size
        )
        for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="Save weights"):
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            if self.save_safetensors:
                save_file(shard, os.path.join(self.output_dir, shard_file), metadata={"format": "pt"})
            else:
                torch.save(shard, os.path.join(self.output_dir, shard_file))

        if not state_dict_split.is_sharded:
            print(f"Model weights saved in {os.path.join(self.output_dir, weights_name)}.")
        else:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            index_name = SAFE_WEIGHTS_INDEX_NAME if self.save_safetensors else WEIGHTS_INDEX_NAME
            with open(os.path.join(self.output_dir, index_name), "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, sort_keys=True)

            print(f"Model weights saved in {self.output_dir}.")


class Baichuan2Converter(BaseConverter):
    """Converter for Baichuan2 models to LLaMA format."""
    
    def detect_format(self) -> str:
        """Detect if input is Baichuan2 format."""
        config_path = os.path.join(self.input_dir, CONFIG_NAME)
        if not os.path.exists(config_path):
            return "unknown"
            
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            
        # Check for Baichuan2 specific patterns
        if "W_pack" in str(config.get("auto_map", {})):
            return "baichuan2"
        if "BaichuanForCausalLM" in config.get("architectures", []):
            return "baichuan2"
            
        return "unknown"
    
    def convert(self) -> None:
        """Convert Baichuan2 model to LLaMA format."""
        try:
            os.makedirs(self.output_dir, exist_ok=False)
        except Exception as e:
            raise ValueError("Output dir already exists") from e
            
        state_dict = self._load_and_convert_weights()
        self.save_weight(state_dict)
        self._save_config()
        
    def _load_and_convert_weights(self) -> Dict[str, torch.Tensor]:
        """Load and convert Baichuan2 weights to LLaMA format."""
        baichuan2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
        for filepath in tqdm(os.listdir(self.input_dir), desc="Load weights"):
            if os.path.isfile(os.path.join(self.input_dir, filepath)) and filepath.endswith(".bin"):
                shard_weight = torch.load(os.path.join(self.input_dir, filepath), map_location="cpu", weights_only=True)
                baichuan2_state_dict.update(shard_weight)

        llama_state_dict: Dict[str, torch.Tensor] = OrderedDict()
        for key, value in tqdm(baichuan2_state_dict.items(), desc="Convert format"):
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
        
    def _save_config(self) -> None:
        """Save LLaMA-compatible config."""
        with open(os.path.join(self.input_dir, CONFIG_NAME), encoding="utf-8") as f:
            config_dict: Dict[str, Any] = json.load(f)

        config_dict["architectures"] = ["LlamaForCausalLM"]
        config_dict.pop("auto_map", None)
        config_dict.pop("tokenizer_class", None)
        config_dict["model_type"] = "llama"

        with open(os.path.join(self.output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Model config saved in {os.path.join(self.output_dir, CONFIG_NAME)}")


class ModelConversionToolkit:
    """Unified toolkit for model conversion, merging, and quantization."""
    
    CONVERTERS = {
        "baichuan2": Baichuan2Converter,
    }
    
    @classmethod
    def get_converter(cls, model_format: str, **kwargs) -> BaseConverter:
        """Get converter instance for specified format."""
        if model_format not in cls.CONVERTERS:
            raise ValueError(f"Unsupported format: {model_format}. Supported: {list(cls.CONVERTERS.keys())}")
        return cls.CONVERTERS[model_format](**kwargs)
    
    @classmethod
    def auto_convert(cls, input_dir: str, output_dir: str, **kwargs) -> None:
        """Automatically detect format and convert."""
        # Try each converter to detect format
        for format_name, converter_class in cls.CONVERTERS.items():
            converter = converter_class(input_dir, output_dir, **kwargs)
            detected_format = converter.detect_format()
            if detected_format != "unknown":
                print(f"Detected format: {detected_format}")
                converter.convert()
                return
                
        raise ValueError("Could not detect model format. Please specify format explicitly.")
    
    @classmethod
    def batch_convert(cls, conversions: List[Dict[str, Any]]) -> None:
        """Batch process multiple conversions."""
        for i, conversion in enumerate(conversions):
            print(f"\nProcessing conversion {i+1}/{len(conversions)}")
            try:
                if "format" in conversion:
                    converter = cls.get_converter(conversion["format"], **conversion)
                    converter.convert()
                else:
                    cls.auto_convert(**conversion)
            except Exception as e:
                print(f"Error in conversion {i+1}: {e}")


def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool):
    """Legacy function for backward compatibility."""
    converter = Baichuan2Converter(input_dir, output_dir, shard_size, save_safetensors)
    state_dict = converter._load_and_convert_weights()
    converter.save_weight(state_dict)


def save_config(input_dir: str, output_dir: str):
    """Legacy function for backward compatibility."""
    converter = Baichuan2Converter(input_dir, output_dir)
    converter._save_config()


def llamafy_baichuan2(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = True,
):
    r"""Convert the Baichuan2-7B model in the same format as LLaMA2-7B.

    Usage: python llamafy_baichuan2.py --input_dir input --output_dir output
    Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied
    """
    converter = Baichuan2Converter(input_dir, output_dir, shard_size, save_safetensors)
    converter.convert()


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)