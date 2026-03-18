"""
Model Registry with Automatic Format Conversion
Centralized registry for model management, conversion, and benchmarking across formats.
"""

import os
import sys
import json
import time
import logging
import hashlib
import tempfile
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from contextlib import contextmanager

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel

# Setup logging
logger = logging.getLogger(__name__)

# Constants
REGISTRY_DIR = os.getenv("LLAMAFACTORY_REGISTRY_DIR", os.path.expanduser("~/.forge/registry"))
MODEL_CARD_FILE = "model_card.json"
BENCHMARK_FILE = "benchmarks.json"


class ModelFormat(Enum):
    """Supported model formats"""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    SAFETENSORS = "safetensors"


@dataclass
class BenchmarkResult:
    """Benchmark results for a specific format"""
    format: str
    inference_time_ms: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    accuracy_score: Optional[float] = None
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentRecommendation:
    """Deployment recommendations for a model format"""
    format: str
    min_gpu_memory_gb: float
    recommended_batch_size: int
    supported_hardware: List[str]
    optimization_tips: List[str]
    quantization_options: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCard:
    """Model card with metadata and performance information"""
    model_id: str
    model_name: str
    description: str
    base_format: ModelFormat
    available_formats: List[ModelFormat]
    parameters_billions: float
    architecture: str
    license: str
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    deployment_recommendations: Dict[str, DeploymentRecommendation] = field(default_factory=dict)
    conversion_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["base_format"] = self.base_format.value
        data["available_formats"] = [f.value for f in self.available_formats]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        data["base_format"] = ModelFormat(data["base_format"])
        data["available_formats"] = [ModelFormat(f) for f in data["available_formats"]]
        return cls(**data)


class FormatConverter:
    """Base class for format converters"""
    
    @staticmethod
    def detect_format(model_path: Union[str, Path]) -> ModelFormat:
        """Detect model format from path"""
        model_path = Path(model_path)
        
        if model_path.is_dir():
            # Check for HuggingFace format
            if (model_path / "config.json").exists() and (model_path / "pytorch_model.bin").exists():
                return ModelFormat.HUGGINGFACE
            elif (model_path / "config.json").exists() and (model_path / "model.safetensors").exists():
                return ModelFormat.SAFETENSORS
            elif any(f.suffix == ".onnx" for f in model_path.iterdir()):
                return ModelFormat.ONNX
            elif any(f.suffix == ".engine" for f in model_path.iterdir()):
                return ModelFormat.TENSORRT
        else:
            if model_path.suffix == ".gguf":
                return ModelFormat.GGUF
            elif model_path.suffix == ".onnx":
                return ModelFormat.ONNX
        
        raise ValueError(f"Could not detect format for {model_path}")
    
    @staticmethod
    def convert_to_gguf(model_path: Union[str, Path], output_path: Union[str, Path], 
                       quantization: str = "q4_0") -> Path:
        """Convert HuggingFace model to GGUF format"""
        try:
            from llama_cpp import convert_hf_to_gguf
        except ImportError:
            raise ImportError("llama-cpp-python required for GGUF conversion. Install with: pip install llama-cpp-python")
        
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use existing conversion script if available
        convert_script = Path(__file__).parent.parent.parent / "scripts" / "convert_ckpt"
        if convert_script.exists():
            # Try to find appropriate conversion script
            for script in convert_script.glob("*.py"):
                if "gguf" in script.name.lower() or "convert" in script.name.lower():
                    cmd = [sys.executable, str(script), 
                          "--model_path", str(model_path),
                          "--output_path", str(output_path),
                          "--quantization", quantization]
                    subprocess.run(cmd, check=True)
                    return output_path / "model.gguf"
        
        # Fallback to llama_cpp conversion
        convert_hf_to_gguf(model_path, output_path, quantization=quantization)
        return output_path / "model.gguf"
    
    @staticmethod
    def convert_to_onnx(model_path: Union[str, Path], output_path: Union[str, Path],
                       opset_version: int = 14) -> Path:
        """Convert model to ONNX format"""
        try:
            from optimum.exporters.onnx import main_export
        except ImportError:
            raise ImportError("optimum required for ONNX conversion. Install with: pip install optimum[onnxruntime]")
        
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        main_export(
            model_name_or_path=str(model_path),
            output=str(output_path),
            opset=opset_version,
            for_ort=False
        )
        
        return output_path / "model.onnx"
    
    @staticmethod
    def convert_to_tensorrt(onnx_path: Union[str, Path], output_path: Union[str, Path],
                          precision: str = "fp16", max_batch_size: int = 32) -> Path:
        """Convert ONNX model to TensorRT format"""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT required for TensorRT conversion")
        
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
        
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        
        profile = builder.create_optimization_profile()
        # Set dynamic shapes (example for text models)
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 1), (max_batch_size // 2, 128), (max_batch_size, 256))
        config.add_optimization_profile(profile)
        
        engine = builder.build_serialized_network(network, config)
        
        engine_path = output_path / "model.engine"
        with open(engine_path, "wb") as f:
            f.write(engine)
        
        return engine_path


class ModelBenchmark:
    """Benchmarking utilities for different model formats"""
    
    @staticmethod
    def benchmark_huggingface(model_path: Union[str, Path], 
                            device: str = "cuda" if torch.cuda.is_available() else "cpu",
                            batch_size: int = 1,
                            sequence_length: int = 128,
                            num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark HuggingFace model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Warmup
        dummy_input = tokenizer("Hello, world!", return_tensors="pt").to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(**dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        total_tokens = 0
        for i in range(num_iterations):
            # Generate random input
            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length)).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            total_tokens += batch_size * sequence_length
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000 / num_iterations
        throughput = total_tokens / (end_time - start_time)
        
        # Memory usage
        if device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            format=ModelFormat.HUGGINGFACE.value,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_mb,
            throughput_tokens_per_sec=throughput,
            hardware_info={"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
        )
    
    @staticmethod
    def benchmark_gguf(model_path: Union[str, Path], 
                      num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark GGUF model using llama.cpp"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required for GGUF benchmarking")
        
        model = Llama(model_path=str(model_path), n_ctx=512, n_gpu_layers=-1)
        
        # Warmup
        for _ in range(10):
            _ = model.create_completion("Hello, world!", max_tokens=1)
        
        # Benchmark
        start_time = time.time()
        total_tokens = 0
        
        for i in range(num_iterations):
            output = model.create_completion(
                "The quick brown fox jumps over the lazy dog. " * 10,
                max_tokens=50,
                temperature=0.0
            )
            total_tokens += len(output["choices"][0]["text"].split())
        
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000 / num_iterations
        throughput = total_tokens / (end_time - start_time)
        
        # Memory usage (approximate)
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            format=ModelFormat.GGUF.value,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_mb,
            throughput_tokens_per_sec=throughput,
            hardware_info={"backend": "llama.cpp"}
        )


class ModelRegistry:
    """Centralized model registry with format conversion and benchmarking"""
    
    def __init__(self, registry_dir: Optional[Union[str, Path]] = None):
        self.registry_dir = Path(registry_dir or REGISTRY_DIR)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelCard] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load existing model cards from registry"""
        for model_dir in self.registry_dir.iterdir():
            if model_dir.is_dir():
                card_path = model_dir / MODEL_CARD_FILE
                if card_path.exists():
                    try:
                        with open(card_path, "r") as f:
                            data = json.load(f)
                            model_card = ModelCard.from_dict(data)
                            self.models[model_card.model_id] = model_card
                    except Exception as e:
                        logger.warning(f"Failed to load model card from {card_path}: {e}")
    
    def _save_model_card(self, model_card: ModelCard):
        """Save model card to registry"""
        model_dir = self.registry_dir / model_card.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        card_path = model_dir / MODEL_CARD_FILE
        with open(card_path, "w") as f:
            json.dump(model_card.to_dict(), f, indent=2)
    
    def register_model(self, 
                      model_id: str,
                      model_path: Union[str, Path],
                      model_name: Optional[str] = None,
                      description: str = "",
                      license: str = "apache-2.0") -> ModelCard:
        """Register a new model in the registry"""
        model_path = Path(model_path)
        
        # Detect format
        base_format = FormatConverter.detect_format(model_path)
        
        # Get model info
        try:
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            parameters = sum(p.numel() for p in model.parameters()) / 1e9
            architecture = config.architectures[0] if hasattr(config, "architectures") else "Unknown"
        except Exception as e:
            logger.warning(f"Could not load model info: {e}")
            parameters = 0.0
            architecture = "Unknown"
        
        model_card = ModelCard(
            model_id=model_id,
            model_name=model_name or model_path.name,
            description=description,
            base_format=base_format,
            available_formats=[base_format],
            parameters_billions=parameters,
            architecture=architecture,
            license=license
        )
        
        self.models[model_id] = model_card
        self._save_model_card(model_card)
        
        # Copy or link model to registry
        dest_dir = self.registry_dir / model_id / base_format.value
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path.is_dir():
            # Symlink for directories to save space
            if not dest_dir.exists():
                os.symlink(model_path.absolute(), dest_dir)
        else:
            # Copy files
            import shutil
            shutil.copy2(model_path, dest_dir / model_path.name)
        
        logger.info(f"Registered model {model_id} with format {base_format.value}")
        return model_card
    
    def convert_model(self,
                     model_id: str,
                     target_format: ModelFormat,
                     quantization: Optional[str] = None,
                     precision: str = "fp16",
                     **kwargs) -> Path:
        """Convert model to target format"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        source_path = self.registry_dir / model_id / model_card.base_format.value
        output_path = self.registry_dir / model_id / target_format.value
        
        # Check if already converted
        if output_path.exists() and any(output_path.iterdir()):
            logger.info(f"Model already converted to {target_format.value}")
            return output_path
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert based on target format
        if target_format == ModelFormat.GGUF:
            result_path = FormatConverter.convert_to_gguf(
                source_path, output_path, 
                quantization=quantization or "q4_0"
            )
        elif target_format == ModelFormat.ONNX:
            result_path = FormatConverter.convert_to_onnx(
                source_path, output_path,
                **kwargs
            )
        elif target_format == ModelFormat.TENSORRT:
            # First convert to ONNX if needed
            if model_card.base_format != ModelFormat.ONNX:
                onnx_path = self.convert_model(model_id, ModelFormat.ONNX)
            else:
                onnx_path = source_path
            
            result_path = FormatConverter.convert_to_tensorrt(
                onnx_path, output_path,
                precision=precision,
                **kwargs
            )
        else:
            raise ValueError(f"Conversion to {target_format.value} not supported")
        
        # Update model card
        if target_format not in model_card.available_formats:
            model_card.available_formats.append(target_format)
        
        model_card.conversion_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_format": model_card.base_format.value,
            "to_format": target_format.value,
            "quantization": quantization,
            "precision": precision
        })
        
        model_card.updated_at = datetime.now().isoformat()
        self._save_model_card(model_card)
        
        logger.info(f"Converted {model_id} to {target_format.value}")
        return result_path
    
    def benchmark_model(self,
                       model_id: str,
                       format: Optional[ModelFormat] = None,
                       **kwargs) -> BenchmarkResult:
        """Benchmark model in specified format"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        target_format = format or model_card.base_format
        
        if target_format not in model_card.available_formats:
            raise ValueError(f"Model not available in {target_format.value}")
        
        model_path = self.registry_dir / model_id / target_format.value
        
        # Run benchmark based on format
        if target_format == ModelFormat.HUGGINGFACE:
            result = ModelBenchmark.benchmark_huggingface(model_path, **kwargs)
        elif target_format == ModelFormat.GGUF:
            result = ModelBenchmark.benchmark_gguf(model_path, **kwargs)
        else:
            raise ValueError(f"Benchmarking for {target_format.value} not implemented")
        
        # Update model card
        model_card.benchmarks[target_format.value] = result
        model_card.updated_at = datetime.now().isoformat()
        self._save_model_card(model_card)
        
        return result
    
    def get_deployment_recommendation(self,
                                    model_id: str,
                                    format: Optional[ModelFormat] = None) -> DeploymentRecommendation:
        """Get deployment recommendations for model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        target_format = format or model_card.base_format
        
        # Generate recommendations based on model size and format
        if target_format == ModelFormat.HUGGINGFACE:
            min_gpu = max(4, model_card.parameters_billions * 2)  # Rough estimate
            batch_size = max(1, int(8 / model_card.parameters_billions))
            hardware = ["NVIDIA GPU", "AMD GPU", "CPU"]
            tips = [
                "Use mixed precision (fp16) for better performance",
                "Consider gradient checkpointing for large models",
                "Use model parallelism for multi-GPU setups"
            ]
        elif target_format == ModelFormat.GGUF:
            min_gpu = max(2, model_card.parameters_billions * 1.5)
            batch_size = 1  # GGUF typically used for single inference
            hardware = ["CPU", "NVIDIA GPU (with CUDA)", "Apple Silicon"]
            tips = [
                "Use quantized versions (q4_0, q5_0) for reduced memory",
                "Adjust n_gpu_layers based on available VRAM",
                "Use mlock to keep model in RAM"
            ]
            quantization_options = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
        elif target_format == ModelFormat.ONNX:
            min_gpu = max(2, model_card.parameters_billions * 1.2)
            batch_size = max(1, int(16 / model_card.parameters_billions))
            hardware = ["NVIDIA GPU", "AMD GPU", "Intel CPU", "CPU"]
            tips = [
                "Use ONNX Runtime for optimized inference",
                "Consider quantization with Intel Neural Compressor",
                "Use dynamic axes for variable input sizes"
            ]
        else:
            min_gpu = model_card.parameters_billions * 2
            batch_size = 1
            hardware = ["NVIDIA GPU"]
            tips = ["Refer to format-specific documentation"]
        
        recommendation = DeploymentRecommendation(
            format=target_format.value,
            min_gpu_memory_gb=min_gpu,
            recommended_batch_size=batch_size,
            supported_hardware=hardware,
            optimization_tips=tips,
            quantization_options=quantization_options if target_format == ModelFormat.GGUF else []
        )
        
        # Update model card
        model_card.deployment_recommendations[target_format.value] = recommendation
        model_card.updated_at = datetime.now().isoformat()
        self._save_model_card(model_card)
        
        return recommendation
    
    def load_model(self,
                  model_id: str,
                  format: Optional[ModelFormat] = None,
                  device: str = "auto",
                  **kwargs) -> Tuple[Any, Any]:
        """Load model and tokenizer from registry"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        target_format = format or model_card.base_format
        
        if target_format not in model_card.available_formats:
            # Try to convert first
            self.convert_model(model_id, target_format)
        
        model_path = self.registry_dir / model_id / target_format.value
        
        if target_format == ModelFormat.HUGGINGFACE:
            model = AutoModel.from_pretrained(model_path, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif target_format == ModelFormat.GGUF:
            from llama_cpp import Llama
            model = Llama(model_path=str(model_path / "model.gguf"), **kwargs)
            tokenizer = None  # GGUF models handle tokenization internally
        else:
            raise ValueError(f"Loading for {target_format.value} not implemented")
        
        return model, tokenizer
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in registry"""
        return [
            {
                "model_id": card.model_id,
                "model_name": card.model_name,
                "parameters_billions": card.parameters_billions,
                "base_format": card.base_format.value,
                "available_formats": [f.value for f in card.available_formats],
                "description": card.description
            }
            for card in self.models.values()
        ]
    
    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        """Get model card by ID"""
        return self.models.get(model_id)
    
    def delete_model(self, model_id: str, format: Optional[ModelFormat] = None):
        """Delete model from registry"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if format:
            # Delete specific format
            format_dir = self.registry_dir / model_id / format.value
            if format_dir.exists():
                import shutil
                shutil.rmtree(format_dir)
                
                # Update model card
                model_card = self.models[model_id]
                if format in model_card.available_formats:
                    model_card.available_formats.remove(format)
                    self._save_model_card(model_card)
        else:
            # Delete entire model
            model_dir = self.registry_dir / model_id
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            
            del self.models[model_id]
        
        logger.info(f"Deleted {model_id}" + (f" format {format.value}" if format else ""))


# Global registry instance
_registry_instance = None


def get_registry(registry_dir: Optional[Union[str, Path]] = None) -> ModelRegistry:
    """Get or create global registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(registry_dir)
    return _registry_instance


def register_model(*args, **kwargs) -> ModelCard:
    """Convenience function to register model"""
    return get_registry().register_model(*args, **kwargs)


def convert_model(*args, **kwargs) -> Path:
    """Convenience function to convert model"""
    return get_registry().convert_model(*args, **kwargs)


def benchmark_model(*args, **kwargs) -> BenchmarkResult:
    """Convenience function to benchmark model"""
    return get_registry().benchmark_model(*args, **kwargs)


def load_model(*args, **kwargs) -> Tuple[Any, Any]:
    """Convenience function to load model"""
    return get_registry().load_model(*args, **kwargs)


# Integration with existing forge scripts
def integrate_with_existing_scripts():
    """Integrate registry with existing conversion scripts"""
    import importlib.util
    
    scripts_dir = Path(__file__).parent.parent.parent / "scripts" / "convert_ckpt"
    if not scripts_dir.exists():
        return
    
    for script_path in scripts_dir.glob("*.py"):
        try:
            # Dynamically import conversion functions
            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for conversion functions
            for name in dir(module):
                if name.startswith("convert") or name.startswith("llamafy"):
                    func = getattr(module, name)
                    if callable(func):
                        # Register as converter if it has the right signature
                        import inspect
                        sig = inspect.signature(func)
                        if "model_path" in sig.parameters or "input_path" in sig.parameters:
                            logger.info(f"Found conversion function {name} in {script_path.name}")
        except Exception as e:
            logger.debug(f"Could not load {script_path}: {e}")


# Initialize on import
integrate_with_existing_scripts()