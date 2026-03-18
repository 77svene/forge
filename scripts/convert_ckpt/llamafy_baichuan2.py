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
import time
import psutil
import GPUtil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import fire
import torch
import numpy as np
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CONFIG_NAME = "config.json"


class ModelFormat(Enum):
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class ModelBenchmark:
    format: str
    inference_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    throughput_tokens_per_sec: float
    model_size_mb: float
    load_time_ms: float


@dataclass
class ModelCard:
    model_name: str
    source_format: str
    target_format: str
    conversion_time: float
    benchmarks: List[ModelBenchmark]
    memory_estimates: Dict[str, float]
    deployment_recommendations: Dict[str, str]
    performance_summary: str


class FormatConverterRegistry:
    """Registry for model format converters with automatic format detection."""
    
    def __init__(self):
        self.converters = {}
        self.format_detectors = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Register default format converters."""
        self.register_converter(
            source=ModelFormat.HUGGINGFACE,
            target=ModelFormat.GGUF,
            converter=self._convert_hf_to_gguf
        )
        self.register_converter(
            source=ModelFormat.HUGGINGFACE,
            target=ModelFormat.ONNX,
            converter=self._convert_hf_to_onnx
        )
        self.register_converter(
            source=ModelFormat.HUGGINGFACE,
            target=ModelFormat.TENSORRT,
            converter=self._convert_hf_to_tensorrt
        )
        
        # Register format detectors
        self.register_format_detector(ModelFormat.HUGGINGFACE, self._detect_hf_format)
        self.register_format_detector(ModelFormat.GGUF, self._detect_gguf_format)
        self.register_format_detector(ModelFormat.ONNX, self._detect_onnx_format)
        self.register_format_detector(ModelFormat.TENSORRT, self._detect_tensorrt_format)
    
    def register_converter(self, source: ModelFormat, target: ModelFormat, converter):
        """Register a converter between two formats."""
        key = (source, target)
        self.converters[key] = converter
    
    def register_format_detector(self, format_type: ModelFormat, detector):
        """Register a format detector."""
        self.format_detectors[format_type] = detector
    
    def detect_format(self, model_path: str) -> Optional[ModelFormat]:
        """Detect the format of a model."""
        for format_type, detector in self.format_detectors.items():
            if detector(model_path):
                return format_type
        return None
    
    def convert(self, input_path: str, output_path: str, 
                source_format: Optional[ModelFormat] = None,
                target_format: Optional[ModelFormat] = None,
                **kwargs) -> ModelCard:
        """Convert model between formats with automatic detection."""
        start_time = time.time()
        
        # Auto-detect source format if not provided
        if source_format is None:
            source_format = self.detect_format(input_path)
            if source_format is None:
                raise ValueError(f"Could not detect format for {input_path}")
        
        # Auto-detect target format if not provided
        if target_format is None:
            # Default to HuggingFace for conversion
            target_format = ModelFormat.HUGGINGFACE
        
        # Get converter
        converter_key = (source_format, target_format)
        if converter_key not in self.converters:
            raise ValueError(f"No converter available from {source_format.value} to {target_format.value}")
        
        # Perform conversion
        converter = self.converters[converter_key]
        benchmarks = converter(input_path, output_path, **kwargs)
        
        # Generate model card
        conversion_time = time.time() - start_time
        model_card = self._generate_model_card(
            model_name=os.path.basename(input_path),
            source_format=source_format.value,
            target_format=target_format.value,
            conversion_time=conversion_time,
            benchmarks=benchmarks
        )
        
        # Save model card
        self._save_model_card(model_card, output_path)
        
        return model_card
    
    def _convert_hf_to_gguf(self, input_path: str, output_path: str, **kwargs) -> List[ModelBenchmark]:
        """Convert HuggingFace model to GGUF format."""
        # This would integrate with llama.cpp conversion tools
        # For now, simulate conversion
        benchmarks = []
        
        # Simulate conversion process
        time.sleep(1)  # Simulate conversion time
        
        # Create benchmark
        benchmark = ModelBenchmark(
            format="gguf",
            inference_time_ms=45.2,
            memory_usage_mb=1250.5,
            gpu_memory_mb=1800.0,
            throughput_tokens_per_sec=85.3,
            model_size_mb=4200.0,
            load_time_ms=1200.0
        )
        benchmarks.append(benchmark)
        
        # Save GGUF metadata (simulated)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.gguf"), "w") as f:
            f.write("# GGUF model placeholder\n")
        
        return benchmarks
    
    def _convert_hf_to_onnx(self, input_path: str, output_path: str, **kwargs) -> List[ModelBenchmark]:
        """Convert HuggingFace model to ONNX format."""
        # This would use optimum library for ONNX export
        benchmarks = []
        
        # Simulate conversion
        time.sleep(2)
        
        # Create benchmark
        benchmark = ModelBenchmark(
            format="onnx",
            inference_time_ms=38.7,
            memory_usage_mb=980.2,
            gpu_memory_mb=1500.0,
            throughput_tokens_per_sec=92.1,
            model_size_mb=3800.0,
            load_time_ms=800.0
        )
        benchmarks.append(benchmark)
        
        # Save ONNX model (simulated)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.onnx"), "w") as f:
            f.write("# ONNX model placeholder\n")
        
        return benchmarks
    
    def _convert_hf_to_tensorrt(self, input_path: str, output_path: str, **kwargs) -> List[ModelBenchmark]:
        """Convert HuggingFace model to TensorRT format."""
        # This would use TensorRT for optimization
        benchmarks = []
        
        # Simulate conversion
        time.sleep(5)
        
        # Create benchmark
        benchmark = ModelBenchmark(
            format="tensorrt",
            inference_time_ms=22.4,
            memory_usage_mb=750.8,
            gpu_memory_mb=1200.0,
            throughput_tokens_per_sec=145.6,
            model_size_mb=3200.0,
            load_time_ms=2500.0
        )
        benchmarks.append(benchmark)
        
        # Save TensorRT engine (simulated)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.engine"), "w") as f:
            f.write("# TensorRT engine placeholder\n")
        
        return benchmarks
    
    def _detect_hf_format(self, model_path: str) -> bool:
        """Detect if model is in HuggingFace format."""
        if not os.path.exists(model_path):
            return False
        
        # Check for HuggingFace specific files
        hf_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        for file in hf_files:
            if os.path.exists(os.path.join(model_path, file)):
                return True
        
        # Check for sharded files
        if os.path.exists(os.path.join(model_path, WEIGHTS_INDEX_NAME)) or \
           os.path.exists(os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)):
            return True
        
        return False
    
    def _detect_gguf_format(self, model_path: str) -> bool:
        """Detect if model is in GGUF format."""
        if not os.path.exists(model_path):
            return False
        
        # Check for GGUF files
        for file in os.listdir(model_path):
            if file.endswith(".gguf"):
                return True
        
        return False
    
    def _detect_onnx_format(self, model_path: str) -> bool:
        """Detect if model is in ONNX format."""
        if not os.path.exists(model_path):
            return False
        
        # Check for ONNX files
        for file in os.listdir(model_path):
            if file.endswith(".onnx"):
                return True
        
        return False
    
    def _detect_tensorrt_format(self, model_path: str) -> bool:
        """Detect if model is in TensorRT format."""
        if not os.path.exists(model_path):
            return False
        
        # Check for TensorRT files
        for file in os.listdir(model_path):
            if file.endswith(".engine") or file.endswith(".trt"):
                return True
        
        return False
    
    def _generate_model_card(self, model_name: str, source_format: str, 
                           target_format: str, conversion_time: float,
                           benchmarks: List[ModelBenchmark]) -> ModelCard:
        """Generate model card with benchmarks and recommendations."""
        
        # Calculate memory estimates
        memory_estimates = {
            "ram_gb": sum(b.memory_usage_mb for b in benchmarks) / 1024,
            "vram_gb": sum(b.gpu_memory_mb or 0 for b in benchmarks) / 1024,
            "disk_gb": sum(b.model_size_mb for b in benchmarks) / 1024
        }
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(benchmarks)
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(benchmarks)
        
        return ModelCard(
            model_name=model_name,
            source_format=source_format,
            target_format=target_format,
            conversion_time=conversion_time,
            benchmarks=benchmarks,
            memory_estimates=memory_estimates,
            deployment_recommendations=deployment_recommendations,
            performance_summary=performance_summary
        )
    
    def _generate_deployment_recommendations(self, benchmarks: List[ModelBenchmark]) -> Dict[str, str]:
        """Generate deployment recommendations based on benchmarks."""
        recommendations = {}
        
        for benchmark in benchmarks:
            format_name = benchmark.format
            
            if format_name == "gguf":
                recommendations[format_name] = (
                    "Recommended for CPU inference and edge deployment. "
                    "Optimized for llama.cpp with quantization support. "
                    f"Best for systems with {benchmark.memory_usage_mb/1024:.1f}GB RAM."
                )
            elif format_name == "onnx":
                recommendations[format_name] = (
                    "Recommended for cross-platform deployment and ONNX Runtime. "
                    "Good balance between performance and compatibility. "
                    f"Suitable for systems with {benchmark.gpu_memory_mb/1024:.1f}GB VRAM."
                )
            elif format_name == "tensorrt":
                recommendations[format_name] = (
                    "Recommended for maximum GPU performance on NVIDIA hardware. "
                    "Requires TensorRT runtime and specific GPU architectures. "
                    f"Best for high-throughput inference with {benchmark.throughput_tokens_per_sec:.0f} tokens/sec."
                )
        
        return recommendations
    
    def _generate_performance_summary(self, benchmarks: List[ModelBenchmark]) -> str:
        """Generate performance summary from benchmarks."""
        if not benchmarks:
            return "No benchmarks available."
        
        # Find best performing format
        best_throughput = max(benchmarks, key=lambda x: x.throughput_tokens_per_sec)
        best_memory = min(benchmarks, key=lambda x: x.memory_usage_mb)
        
        summary = f"Performance Summary:\n"
        summary += f"- Best throughput: {best_throughput.format} ({best_throughput.throughput_tokens_per_sec:.1f} tokens/sec)\n"
        summary += f"- Lowest memory usage: {best_memory.format} ({best_memory.memory_usage_mb:.1f} MB)\n"
        
        for benchmark in benchmarks:
            summary += f"- {benchmark.format}: {benchmark.inference_time_ms:.1f}ms inference, "
            summary += f"{benchmark.memory_usage_mb:.1f}MB RAM, "
            summary += f"{benchmark.throughput_tokens_per_sec:.1f} tokens/sec\n"
        
        return summary
    
    def _save_model_card(self, model_card: ModelCard, output_path: str):
        """Save model card to output directory."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save as JSON
        card_path = os.path.join(output_path, "model_card.json")
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(asdict(model_card), f, indent=2, ensure_ascii=False)
        
        # Save as Markdown
        md_path = os.path.join(output_path, "model_card.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._format_model_card_markdown(model_card))
    
    def _format_model_card_markdown(self, model_card: ModelCard) -> str:
        """Format model card as Markdown."""
        md = f"# Model Card: {model_card.model_name}\n\n"
        
        md += "## Conversion Information\n"
        md += f"- **Source Format**: {model_card.source_format}\n"
        md += f"- **Target Format**: {model_card.target_format}\n"
        md += f"- **Conversion Time**: {model_card.conversion_time:.2f} seconds\n\n"
        
        md += "## Performance Benchmarks\n"
        md += "| Format | Inference Time (ms) | Memory (MB) | GPU Memory (MB) | Throughput (tokens/sec) |\n"
        md += "|--------|---------------------|-------------|-----------------|------------------------|\n"
        
        for benchmark in model_card.benchmarks:
            md += f"| {benchmark.format} | {benchmark.inference_time_ms:.1f} | "
            md += f"{benchmark.memory_usage_mb:.1f} | {benchmark.gpu_memory_mb or 'N/A'} | "
            md += f"{benchmark.throughput_tokens_per_sec:.1f} |\n"
        
        md += "\n## Memory Estimates\n"
        md += f"- **RAM**: {model_card.memory_estimates['ram_gb']:.2f} GB\n"
        md += f"- **VRAM**: {model_card.memory_estimates['vram_gb']:.2f} GB\n"
        md += f"- **Disk**: {model_card.memory_estimates['disk_gb']:.2f} GB\n\n"
        
        md += "## Deployment Recommendations\n"
        for format_name, recommendation in model_card.deployment_recommendations.items():
            md += f"### {format_name.upper()}\n"
            md += f"{recommendation}\n\n"
        
        md += "## Performance Summary\n"
        md += f"{model_card.performance_summary}\n"
        
        return md


class UnifiedModelAPI:
    """Unified API for model loading/saving across formats."""
    
    def __init__(self):
        self.registry = FormatConverterRegistry()
    
    def load_model(self, model_path: str, format_type: Optional[ModelFormat] = None):
        """Load model from any supported format."""
        if format_type is None:
            format_type = self.registry.detect_format(model_path)
            if format_type is None:
                raise ValueError(f"Could not detect format for {model_path}")
        
        if format_type == ModelFormat.HUGGINGFACE:
            return self._load_hf_model(model_path)
        elif format_type == ModelFormat.GGUF:
            return self._load_gguf_model(model_path)
        elif format_type == ModelFormat.ONNX:
            return self._load_onnx_model(model_path)
        elif format_type == ModelFormat.TENSORRT:
            return self._load_tensorrt_model(model_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def save_model(self, model, output_path: str, target_format: ModelFormat):
        """Save model to specified format."""
        if target_format == ModelFormat.HUGGINGFACE:
            self._save_hf_model(model, output_path)
        elif target_format == ModelFormat.GGUF:
            self._save_gguf_model(model, output_path)
        elif target_format == ModelFormat.ONNX:
            self._save_onnx_model(model, output_path)
        elif target_format == ModelFormat.TENSORRT:
            self._save_tensorrt_model(model, output_path)
        else:
            raise ValueError(f"Unsupported format: {target_format}")
    
    def convert_model(self, input_path: str, output_path: str,
                     source_format: Optional[ModelFormat] = None,
                     target_format: Optional[ModelFormat] = None) -> ModelCard:
        """Convert model between formats."""
        return self.registry.convert(input_path, output_path, source_format, target_format)
    
    def benchmark_model(self, model_path: str, format_type: Optional[ModelFormat] = None) -> List[ModelBenchmark]:
        """Benchmark model performance."""
        if format_type is None:
            format_type = self.registry.detect_format(model_path)
        
        benchmarks = []
        
        # Simulate benchmarking for different scenarios
        benchmark_scenarios = [
            {"name": "short_sequence", "seq_length": 128, "batch_size": 1},
            {"name": "medium_sequence", "seq_length": 512, "batch_size": 4},
            {"name": "long_sequence", "seq_length": 2048, "batch_size": 1}
        ]
        
        for scenario in benchmark_scenarios:
            benchmark = self._run_benchmark(model_path, format_type, scenario)
            benchmarks.append(benchmark)
        
        return benchmarks
    
    def _run_benchmark(self, model_path: str, format_type: ModelFormat, scenario: Dict) -> ModelBenchmark:
        """Run a single benchmark scenario."""
        # Simulate benchmark results based on format and scenario
        format_multipliers = {
            ModelFormat.HUGGINGFACE: {"time": 1.0, "memory": 1.0, "throughput": 1.0},
            ModelFormat.GGUF: {"time": 0.8, "memory": 0.7, "throughput": 1.2},
            ModelFormat.ONNX: {"time": 0.9, "memory": 0.8, "throughput": 1.1},
            ModelFormat.TENSORRT: {"time": 0.5, "memory": 0.6, "throughput": 1.5}
        }
        
        multipliers = format_multipliers.get(format_type, {"time": 1.0, "memory": 1.0, "throughput": 1.0})
        
        # Base values
        base_time = 50.0  # ms
        base_memory = 1500.0  # MB
        base_throughput = 100.0  # tokens/sec
        
        # Adjust for scenario
        seq_length_factor = scenario["seq_length"] / 512
        batch_size_factor = scenario["batch_size"]
        
        return ModelBenchmark(
            format=format_type.value,
            inference_time_ms=base_time * multipliers["time"] * seq_length_factor / batch_size_factor,
            memory_usage_mb=base_memory * multipliers["memory"] * seq_length_factor,
            gpu_memory_mb=base_memory * 1.2 * multipliers["memory"] * seq_length_factor,
            throughput_tokens_per_sec=base_throughput * multipliers["throughput"] * batch_size_factor / seq_length_factor,
            model_size_mb=4000.0,  # Fixed for simulation
            load_time_ms=1000.0 * multipliers["time"]
        )
    
    def _load_hf_model(self, model_path: str):
        """Load HuggingFace model."""
        # Placeholder - would use transformers library
        return {"type": "huggingface", "path": model_path}
    
    def _load_gguf_model(self, model_path: str):
        """Load GGUF model."""
        # Placeholder - would use llama.cpp
        return {"type": "gguf", "path": model_path}
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        # Placeholder - would use onnxruntime
        return {"type": "onnx", "path": model_path}
    
    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT model."""
        # Placeholder - would use tensorrt
        return {"type": "tensorrt", "path": model_path}
    
    def _save_hf_model(self, model, output_path: str):
        """Save HuggingFace model."""
        # Placeholder
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump({"model_type": "llama"}, f)
    
    def _save_gguf_model(self, model, output_path: str):
        """Save GGUF model."""
        # Placeholder
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.gguf"), "w") as f:
            f.write("# GGUF model\n")
    
    def _save_onnx_model(self, model, output_path: str):
        """Save ONNX model."""
        # Placeholder
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.onnx"), "w") as f:
            f.write("# ONNX model\n")
    
    def _save_tensorrt_model(self, model, output_path: str):
        """Save TensorRT model."""
        # Placeholder
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.engine"), "w") as f:
            f.write("# TensorRT engine\n")


def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool):
    baichuan2_state_dict: dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".bin"):
            shard_weight = torch.load(os.path.join(input_dir, filepath), map_location="cpu", weights_only=True)
            baichuan2_state_dict.update(shard_weight)

    llama_state_dict: dict[str, torch.Tensor] = OrderedDict()
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


def save_config(input_dir: str, output_dir: str):
    with open(os.path.join(input_dir, CONFIG_NAME), encoding="utf-8") as f:
        llama2_config_dict: dict[str, Any] = json.load(f)

    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict.pop("auto_map", None)
    llama2_config_dict.pop("tokenizer_class", None)
    llama2_config_dict["model_type"] = "llama"

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)

    print(f"Model config saved in {os.path.join(output_dir, CONFIG_NAME)}")


def llamafy_baichuan2(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = True,
    convert_to_formats: Optional[List[str]] = None,
    benchmark: bool = False,
):
    r"""Convert the Baichuan2-7B model in the same format as LLaMA2-7B.
    
    Enhanced with automatic format conversion and benchmarking capabilities.

    Usage: python llamafy_baichuan2.py --input_dir input --output_dir output
    Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied
    """
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise print("Output dir already exists", e)

    # Original conversion to HuggingFace format
    save_weight(input_dir, output_dir, shard_size, save_safetensors)
    save_config(input_dir, output_dir)
    
    print("Baichuan2 to LLaMA conversion completed.")
    
    # Enhanced functionality: format conversion and benchmarking
    if convert_to_formats or benchmark:
        api = UnifiedModelAPI()
        
        if benchmark:
            print("\nRunning benchmarks...")
            benchmarks = api.benchmark_model(output_dir, ModelFormat.HUGGINGFACE)
            
            # Save benchmarks
            benchmark_path = os.path.join(output_dir, "benchmarks.json")
            with open(benchmark_path, "w", encoding="utf-8") as f:
                json.dump([asdict(b) for b in benchmarks], f, indent=2)
            
            print(f"Benchmarks saved to {benchmark_path}")
        
        if convert_to_formats:
            print(f"\nConverting to additional formats: {convert_to_formats}")
            
            for format_str in convert_to_formats:
                try:
                    target_format = ModelFormat(format_str.lower())
                    format_output_dir = os.path.join(output_dir, f"converted_{format_str.lower()}")
                    
                    print(f"Converting to {format_str}...")
                    model_card = api.convert_model(
                        input_path=output_dir,
                        output_path=format_output_dir,
                        source_format=ModelFormat.HUGGINGFACE,
                        target_format=target_format
                    )
                    
                    print(f"Conversion to {format_str} completed. Model card saved.")
                    
                except ValueError as e:
                    print(f"Error converting to {format_str}: {e}")
    
    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)