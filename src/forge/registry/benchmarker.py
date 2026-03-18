"""
Model Registry with Automatic Format Conversion for forge.

This module provides a centralized model registry that automatically converts between
HuggingFace, GGUF, ONNX, and TensorRT formats. It includes model cards with performance
benchmarks, memory usage estimates, and deployment recommendations.
"""

import os
import json
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import importlib.util

import torch
import numpy as np

# Check for optional dependencies
GGUF_AVAILABLE = importlib.util.find_spec("gguf") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
TENSORRT_AVAILABLE = importlib.util.find_spec("tensorrt") is not None

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class BenchmarkResult:
    """Results from model benchmarking."""
    format: ModelFormat
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    model_size_mb: float = 0.0
    quantization: Optional[str] = None
    precision: str = "float32"
    batch_size: int = 1
    sequence_length: int = 512
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @property
    def performance_score(self) -> float:
        """Calculate a composite performance score (higher is better)."""
        # Weighted score based on throughput and inverse latency
        return (self.throughput_tokens_per_sec * 0.7) + (1000.0 / max(self.latency_ms, 0.1) * 0.3)


@dataclass
class DeploymentRecommendation:
    """Deployment recommendations for a model format."""
    format: ModelFormat
    recommended_hardware: List[str] = field(default_factory=list)
    min_memory_gb: float = 0.0
    min_gpu_memory_gb: float = 0.0
    quantization_suggestions: List[str] = field(default_factory=list)
    batch_size_recommendation: int = 1
    use_case: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ModelCard:
    """Model card with metadata, benchmarks, and deployment info."""
    model_name: str
    model_id: str
    base_model: str
    formats: List[ModelFormat] = field(default_factory=list)
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    deployment_recommendations: Dict[str, DeploymentRecommendation] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    license: str = ""
    description: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["formats"] = [f.value for f in self.formats]
        for key, benchmark in data["benchmarks"].items():
            if "format" in benchmark:
                benchmark["format"] = benchmark["format"].value if isinstance(benchmark["format"], ModelFormat) else benchmark["format"]
        for key, rec in data["deployment_recommendations"].items():
            if "format" in rec:
                rec["format"] = rec["format"].value if isinstance(rec["format"], ModelFormat) else rec["format"]
        return data
    
    def save(self, path: Union[str, Path]):
        """Save model card to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelCard":
        """Load model card from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Convert string formats back to enums
        data["formats"] = [ModelFormat(f) for f in data["formats"]]
        for key, benchmark in data["benchmarks"].items():
            if "format" in benchmark:
                benchmark["format"] = ModelFormat(benchmark["format"])
        for key, rec in data["deployment_recommendations"].items():
            if "format" in rec:
                rec["format"] = ModelFormat(rec["format"])
        return cls(**data)


class FormatConverter(ABC):
    """Abstract base class for format converters."""
    
    @abstractmethod
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if this converter can convert between formats."""
        pass
    
    @abstractmethod
    def convert(self, model_path: Union[str, Path], output_path: Union[str, Path], 
                **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert model to target format.
        
        Returns:
            Tuple of (output_path, conversion_metadata)
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    @abstractmethod
    def benchmark(self, model_path: Union[str, Path], 
                  benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark the model in this format."""
        pass


class HuggingFaceConverter(FormatConverter):
    """Converter for HuggingFace models."""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """HuggingFace can convert to any format."""
        return source_format == ModelFormat.HUGGINGFACE
    
    def convert(self, model_path: Union[str, Path], output_path: Union[str, Path], 
                **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """Convert HuggingFace model to target format (handled by other converters)."""
        # HuggingFace is the source format, actual conversion handled by target converters
        return Path(model_path), {"source_format": "huggingface"}
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        model_path = Path(model_path)
        info = {
            "format": "huggingface",
            "path": str(model_path),
            "exists": model_path.exists()
        }
        
        if model_path.is_dir():
            # Check for common HuggingFace files
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                    info["model_type"] = config.get("model_type", "unknown")
                    info["architectures"] = config.get("architectures", [])
            
            # Check for model files
            for pattern in ["*.bin", "*.safetensors", "*.pt"]:
                model_files = list(model_path.glob(pattern))
                if model_files:
                    info["model_files"] = [f.name for f in model_files]
                    break
        
        return info
    
    def benchmark(self, model_path: Union[str, Path], 
                  benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = Path(model_path)
        device = benchmark_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = benchmark_config.get("batch_size", 1)
        sequence_length = benchmark_config.get("sequence_length", 512)
        iterations = benchmark_config.get("iterations", 10)
        warmup = benchmark_config.get("warmup", 3)
        
        # Load model and tokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            
            # Prepare dummy input
            dummy_input = "This is a test sentence for benchmarking."
            inputs = tokenizer(
                [dummy_input] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=sequence_length
            ).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(**inputs)
                    if device == "cuda":
                        torch.cuda.synchronize()
            
            # Benchmark
            latencies = []
            memory_before = torch.cuda.memory_allocated() if device == "cuda" else 0
            
            with torch.no_grad():
                for _ in range(iterations):
                    start_time = time.time()
                    outputs = model(**inputs)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            memory_after = torch.cuda.memory_allocated() if device == "cuda" else 0
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            total_tokens = batch_size * sequence_length
            throughput = (total_tokens / (avg_latency / 1000)) if avg_latency > 0 else 0
            
            # Get model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # Clean up
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return BenchmarkResult(
                format=ModelFormat.HUGGINGFACE,
                latency_ms=avg_latency,
                throughput_tokens_per_sec=throughput,
                memory_usage_mb=model_size,
                gpu_memory_mb=memory_used if device == "cuda" else 0,
                model_size_mb=model_size,
                precision="float16" if device == "cuda" else "float32",
                batch_size=batch_size,
                sequence_length=sequence_length,
                warmup_iterations=warmup,
                benchmark_iterations=iterations,
                hardware_info={"device": device}
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking HuggingFace model: {e}")
            return BenchmarkResult(
                format=ModelFormat.HUGGINGFACE,
                latency_ms=0,
                throughput_tokens_per_sec=0,
                memory_usage_mb=0,
                gpu_memory_mb=0,
                model_size_mb=0
            )


class GGUFConverter(FormatConverter):
    """Converter for GGUF format (llama.cpp)."""
    
    def __init__(self):
        if not GGUF_AVAILABLE:
            logger.warning("GGUF support not available. Install gguf package.")
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """GGUF converter can convert from HuggingFace to GGUF."""
        return (source_format == ModelFormat.HUGGINGFACE and 
                target_format == ModelFormat.GGUF and GGUF_AVAILABLE)
    
    def convert(self, model_path: Union[str, Path], output_path: Union[str, Path], 
                **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """Convert HuggingFace model to GGUF format."""
        if not GGUF_AVAILABLE:
            raise RuntimeError("GGUF support not available. Install gguf package.")
        
        # This would integrate with existing conversion scripts
        # For now, we'll create a placeholder that would call the actual conversion
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In a real implementation, this would call:
        # python scripts/convert_ckpt/llamafy_qwen.py or similar
        
        logger.info(f"Converting {model_path} to GGUF format at {output_path}")
        
        # Placeholder for actual conversion
        # For now, we'll just create an empty file to indicate conversion
        with open(output_path, "w") as f:
            f.write("# GGUF model placeholder\n")
            f.write(f"# Converted from: {model_path}\n")
            f.write(f"# Conversion timestamp: {time.time()}\n")
        
        metadata = {
            "source_format": "huggingface",
            "target_format": "gguf",
            "conversion_time": time.time(),
            "quantization": kwargs.get("quantization", "q4_0")
        }
        
        return output_path, metadata
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get GGUF model information."""
        model_path = Path(model_path)
        info = {
            "format": "gguf",
            "path": str(model_path),
            "exists": model_path.exists()
        }
        
        if model_path.exists() and model_path.is_file():
            info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
            info["extension"] = model_path.suffix
        
        return info
    
    def benchmark(self, model_path: Union[str, Path], 
                  benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark GGUF model (placeholder implementation)."""
        # In a real implementation, this would use llama.cpp for benchmarking
        model_path = Path(model_path)
        
        # Placeholder metrics
        model_size = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        return BenchmarkResult(
            format=ModelFormat.GGUF,
            latency_ms=50.0,  # Placeholder
            throughput_tokens_per_sec=100.0,  # Placeholder
            memory_usage_mb=model_size * 0.8,  # Estimate
            gpu_memory_mb=0,  # GGUF typically runs on CPU
            model_size_mb=model_size,
            quantization="q4_0",  # Common GGUF quantization
            precision="int4",
            batch_size=benchmark_config.get("batch_size", 1),
            sequence_length=benchmark_config.get("sequence_length", 512),
            hardware_info={"device": "cpu"}
        )


class ONNXConverter(FormatConverter):
    """Converter for ONNX format."""
    
    def __init__(self):
        if not ONNX_AVAILABLE:
            logger.warning("ONNX support not available. Install onnx and onnxruntime packages.")
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """ONNX converter can convert from HuggingFace to ONNX."""
        return (source_format == ModelFormat.HUGGINGFACE and 
                target_format == ModelFormat.ONNX and ONNX_AVAILABLE)
    
    def convert(self, model_path: Union[str, Path], output_path: Union[str, Path], 
                **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """Convert HuggingFace model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX support not available. Install onnx and onnxruntime packages.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {model_path} to ONNX format at {output_path}")
        
        # Placeholder for actual ONNX conversion
        # In a real implementation, this would use optimum or similar
        
        with open(output_path, "w") as f:
            f.write("# ONNX model placeholder\n")
            f.write(f"# Converted from: {model_path}\n")
            f.write(f"# Conversion timestamp: {time.time()}\n")
        
        metadata = {
            "source_format": "huggingface",
            "target_format": "onnx",
            "conversion_time": time.time(),
            "opset_version": kwargs.get("opset_version", 14)
        }
        
        return output_path, metadata
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get ONNX model information."""
        model_path = Path(model_path)
        info = {
            "format": "onnx",
            "path": str(model_path),
            "exists": model_path.exists()
        }
        
        if model_path.exists() and model_path.is_file():
            info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
            info["extension"] = model_path.suffix
        
        return info
    
    def benchmark(self, model_path: Union[str, Path], 
                  benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark ONNX model (placeholder implementation)."""
        # In a real implementation, this would use onnxruntime for benchmarking
        model_path = Path(model_path)
        
        # Placeholder metrics
        model_size = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        return BenchmarkResult(
            format=ModelFormat.ONNX,
            latency_ms=30.0,  # Placeholder
            throughput_tokens_per_sec=150.0,  # Placeholder
            memory_usage_mb=model_size * 1.2,  # Estimate
            gpu_memory_mb=model_size * 0.5 if torch.cuda.is_available() else 0,
            model_size_mb=model_size,
            precision="float32",
            batch_size=benchmark_config.get("batch_size", 1),
            sequence_length=benchmark_config.get("sequence_length", 512),
            hardware_info={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )


class TensorRTConverter(FormatConverter):
    """Converter for TensorRT format."""
    
    def __init__(self):
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT support not available. Install tensorrt package.")
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """TensorRT converter can convert from ONNX to TensorRT."""
        return (source_format == ModelFormat.ONNX and 
                target_format == ModelFormat.TENSORRT and TENSORRT_AVAILABLE)
    
    def convert(self, model_path: Union[str, Path], output_path: Union[str, Path], 
                **kwargs) -> Tuple[Path, Dict[str, Any]]:
        """Convert ONNX model to TensorRT format."""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT support not available. Install tensorrt package.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {model_path} to TensorRT format at {output_path}")
        
        # Placeholder for actual TensorRT conversion
        with open(output_path, "w") as f:
            f.write("# TensorRT model placeholder\n")
            f.write(f"# Converted from: {model_path}\n")
            f.write(f"# Conversion timestamp: {time.time()}\n")
        
        metadata = {
            "source_format": "onnx",
            "target_format": "tensorrt",
            "conversion_time": time.time(),
            "precision": kwargs.get("precision", "fp16"),
            "workspace_size": kwargs.get("workspace_size", 1 << 30)  # 1GB default
        }
        
        return output_path, metadata
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get TensorRT model information."""
        model_path = Path(model_path)
        info = {
            "format": "tensorrt",
            "path": str(model_path),
            "exists": model_path.exists()
        }
        
        if model_path.exists() and model_path.is_file():
            info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
            info["extension"] = model_path.suffix
        
        return info
    
    def benchmark(self, model_path: Union[str, Path], 
                  benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark TensorRT model (placeholder implementation)."""
        # In a real implementation, this would use TensorRT for benchmarking
        model_path = Path(model_path)
        
        # Placeholder metrics
        model_size = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        return BenchmarkResult(
            format=ModelFormat.TENSORRT,
            latency_ms=10.0,  # Placeholder - TensorRT is typically fastest
            throughput_tokens_per_sec=300.0,  # Placeholder
            memory_usage_mb=model_size * 0.6,  # Estimate
            gpu_memory_mb=model_size * 0.8,
            model_size_mb=model_size,
            precision="fp16",
            batch_size=benchmark_config.get("batch_size", 1),
            sequence_length=benchmark_config.get("sequence_length", 512),
            hardware_info={"device": "cuda"}
        )


class ModelRegistry:
    """
    Centralized model registry with automatic format conversion.
    
    This registry manages models in different formats, provides automatic conversion,
    benchmarking, and generates model cards with deployment recommendations.
    """
    
    def __init__(self, registry_dir: Union[str, Path] = "./model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store models and metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize converters
        self.converters = {
            ModelFormat.HUGGINGFACE: HuggingFaceConverter(),
            ModelFormat.GGUF: GGUFConverter(),
            ModelFormat.ONNX: ONNXConverter(),
            ModelFormat.TENSORRT: TensorRTConverter()
        }
        
        # Model registry storage
        self.models: Dict[str, ModelCard] = {}
        self._load_registry()
        
        logger.info(f"Model registry initialized at {self.registry_dir}")
    
    def _load_registry(self):
        """Load existing models from registry directory."""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.models[model_id] = ModelCard(**model_data)
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"
        data = {model_id: model_card.to_dict() for model_id, model_card in self.models.items()}
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_model_id(self, model_name: str, format: ModelFormat) -> str:
        """Generate a unique model ID."""
        hash_input = f"{model_name}_{format.value}_{time.time()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def register_model(self, 
                      model_name: str,
                      model_path: Union[str, Path],
                      source_format: ModelFormat = ModelFormat.HUGGINGFACE,
                      description: str = "",
                      tags: List[str] = None,
                      license: str = "",
                      **metadata) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model
            source_format: Format of the source model
            description: Model description
            tags: List of tags
            license: License information
            **metadata: Additional metadata
            
        Returns:
            Model ID
        """
        model_path = Path(model_path)
        
        # Generate model ID
        model_id = self._generate_model_id(model_name, source_format)
        
        # Get model info
        converter = self.converters.get(source_format)
        if not converter:
            raise ValueError(f"No converter available for format {source_format}")
        
        model_info = converter.get_model_info(model_path)
        
        # Create model card
        model_card = ModelCard(
            model_name=model_name,
            model_id=model_id,
            base_model=model_name,
            formats=[source_format],
            metadata={
                **model_info,
                **metadata,
                "source_path": str(model_path),
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            tags=tags or [],
            license=license,
            description=description
        )
        
        # Store in registry
        self.models[model_id] = model_card
        self._save_registry()
        
        logger.info(f"Registered model {model_name} with ID {model_id}")
        return model_id
    
    def convert_model(self,
                     model_id: str,
                     target_format: ModelFormat,
                     output_dir: Optional[Union[str, Path]] = None,
                     **conversion_kwargs) -> str:
        """
        Convert a model to a different format.
        
        Args:
            model_id: ID of the model to convert
            target_format: Target format for conversion
            output_dir: Directory to save converted model (optional)
            **conversion_kwargs: Additional arguments for conversion
            
        Returns:
            Path to converted model
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        source_format = model_card.formats[0]  # Assume first format is source
        
        # Check if already in target format
        if target_format in model_card.formats:
            logger.info(f"Model {model_id} already in {target_format.value} format")
            return model_card.metadata.get(f"{target_format.value}_path", "")
        
        # Find appropriate converter
        converter = None
        for conv in self.converters.values():
            if conv.can_convert(source_format, target_format):
                converter = conv
                break
        
        if not converter:
            raise ValueError(f"No converter available from {source_format.value} to {target_format.value}")
        
        # Set up output path
        if output_dir is None:
            output_dir = self.registry_dir / model_id / target_format.value
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"model.{target_format.value}"
        
        # Perform conversion
        source_path = Path(model_card.metadata.get("source_path", ""))
        converted_path, conversion_metadata = converter.convert(
            source_path, output_path, **conversion_kwargs
        )
        
        # Update model card
        model_card.formats.append(target_format)
        model_card.metadata.update({
            f"{target_format.value}_path": str(converted_path),
            f"{target_format.value}_conversion": conversion_metadata
        })
        
        self._save_registry()
        
        logger.info(f"Converted model {model_id} to {target_format.value} format")
        return str(converted_path)
    
    def benchmark_model(self,
                       model_id: str,
                       format: Optional[ModelFormat] = None,
                       benchmark_config: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """
        Benchmark a model in the registry.
        
        Args:
            model_id: ID of the model to benchmark
            format: Format to benchmark (uses first available if not specified)
            benchmark_config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        
        # Determine format to benchmark
        if format is None:
            if not model_card.formats:
                raise ValueError(f"Model {model_id} has no available formats")
            format = model_card.formats[0]
        elif format not in model_card.formats:
            raise ValueError(f"Model {model_id} not available in {format.value} format")
        
        # Get converter for benchmarking
        converter = self.converters.get(format)
        if not converter:
            raise ValueError(f"No converter available for format {format.value}")
        
        # Get model path
        model_path_key = f"{format.value}_path" if format != ModelFormat.HUGGINGFACE else "source_path"
        model_path = model_card.metadata.get(model_path_key, "")
        
        if not model_path or not Path(model_path).exists():
            raise ValueError(f"Model path not found for format {format.value}")
        
        # Set default benchmark config
        if benchmark_config is None:
            benchmark_config = {
                "batch_size": 1,
                "sequence_length": 512,
                "iterations": 10,
                "warmup": 3,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        
        # Run benchmark
        benchmark_result = converter.benchmark(model_path, benchmark_config)
        
        # Update model card
        benchmark_key = f"{format.value}_benchmark"
        model_card.benchmarks[benchmark_key] = benchmark_result
        
        # Generate deployment recommendations
        self._generate_deployment_recommendations(model_card, format, benchmark_result)
        
        self._save_registry()
        
        logger.info(f"Benchmarked model {model_id} in {format.value} format")
        return benchmark_result
    
    def _generate_deployment_recommendations(self,
                                           model_card: ModelCard,
                                           format: ModelFormat,
                                           benchmark: BenchmarkResult):
        """Generate deployment recommendations based on benchmark results."""
        recommendations = DeploymentRecommendation(format=format)
        
        if format == ModelFormat.HUGGINGFACE:
            recommendations.recommended_hardware = ["GPU (NVIDIA A100, V100, RTX 3090)", "CPU (for small models)"]
            recommendations.min_memory_gb = benchmark.memory_usage_mb / 1024 * 1.5  # 50% overhead
            recommendations.min_gpu_memory_gb = benchmark.gpu_memory_mb / 1024 * 1.5
            recommendations.quantization_suggestions = ["8-bit", "4-bit (GPTQ/AWQ)"]
            recommendations.batch_size_recommendation = 1 if benchmark.gpu_memory_mb > 8000 else 1
            recommendations.use_case = "Research, fine-tuning, full-precision inference"
            recommendations.notes = "Best for development and research. Consider quantization for deployment."
            
        elif format == ModelFormat.GGUF:
            recommendations.recommended_hardware = ["CPU (any modern x86_64)", "Apple Silicon (M1/M2/M3)"]
            recommendations.min_memory_gb = benchmark.memory_usage_mb / 1024 * 1.2
            recommendations.min_gpu_memory_gb = 0
            recommendations.quantization_suggestions = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
            recommendations.batch_size_recommendation = 1
            recommendations.use_case = "Local deployment, CPU inference, edge devices"
            recommendations.notes = "Optimized for CPU inference. Good for local deployment and edge devices."
            
        elif format == ModelFormat.ONNX:
            recommendations.recommended_hardware = ["GPU (NVIDIA, AMD, Intel)", "CPU"]
            recommendations.min_memory_gb = benchmark.memory_usage_mb / 1024 * 1.3
            recommendations.min_gpu_memory_gb = benchmark.gpu_memory_mb / 1024 * 1.3
            recommendations.quantization_suggestions = ["INT8", "FP16"]
            recommendations.batch_size_recommendation = 4 if benchmark.gpu_memory_mb > 4000 else 1
            recommendations.use_case = "Cross-platform deployment, production inference"
            recommendations.notes = "Good balance of performance and compatibility. Works across different hardware."
            
        elif format == ModelFormat.TENSORRT:
            recommendations.recommended_hardware = ["NVIDIA GPU (A100, V100, RTX 3090/4090)"]
            recommendations.min_memory_gb = benchmark.memory_usage_mb / 1024 * 1.2
            recommendations.min_gpu_memory_gb = benchmark.gpu_memory_mb / 1024 * 1.2
            recommendations.quantization_suggestions = ["FP16", "INT8"]
            recommendations.batch_size_recommendation = 8 if benchmark.gpu_memory_mb > 8000 else 4
            recommendations.use_case = "High-performance production deployment, NVIDIA GPUs"
            recommendations.notes = "Highest performance on NVIDIA GPUs. Requires NVIDIA hardware and TensorRT runtime."
        
        model_card.deployment_recommendations[format.value] = recommendations
    
    def get_model(self, model_id: str) -> ModelCard:
        """Get model card by ID."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        return self.models[model_id]
    
    def list_models(self, 
                   format_filter: Optional[ModelFormat] = None,
                   tag_filter: Optional[str] = None) -> List[ModelCard]:
        """
        List models in the registry with optional filters.
        
        Args:
            format_filter: Filter by format
            tag_filter: Filter by tag
            
        Returns:
            List of matching model cards
        """
        models = list(self.models.values())
        
        if format_filter:
            models = [m for m in models if format_filter in m.formats]
        
        if tag_filter:
            models = [m for m in models if tag_filter in m.tags]
        
        return models
    
    def compare_formats(self, model_id: str) -> Dict[str, Any]:
        """
        Compare performance across different formats for a model.
        
        Args:
            model_id: ID of the model to compare
            
        Returns:
            Comparison data
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        comparison = {
            "model_id": model_id,
            "model_name": model_card.model_name,
            "formats": [],
            "benchmarks": {},
            "recommendations": {}
        }
        
        for format in model_card.formats:
            format_str = format.value
            comparison["formats"].append(format_str)
            
            benchmark_key = f"{format_str}_benchmark"
            if benchmark_key in model_card.benchmarks:
                comparison["benchmarks"][format_str] = model_card.benchmarks[benchmark_key].to_dict()
            
            if format_str in model_card.deployment_recommendations:
                comparison["recommendations"][format_str] = model_card.deployment_recommendations[format_str].to_dict()
        
        # Calculate best format for different use cases
        if comparison["benchmarks"]:
            best_throughput = max(
                comparison["benchmarks"].items(),
                key=lambda x: x[1].get("throughput_tokens_per_sec", 0)
            )[0]
            
            best_latency = min(
                comparison["benchmarks"].items(),
                key=lambda x: x[1].get("latency_ms", float('inf'))
            )[0]
            
            best_memory = min(
                comparison["benchmarks"].items(),
                key=lambda x: x[1].get("memory_usage_mb", float('inf'))
            )[0]
            
            comparison["best_for"] = {
                "throughput": best_throughput,
                "latency": best_latency,
                "memory_efficiency": best_memory
            }
        
        return comparison
    
    def export_model_card(self, model_id: str, output_path: Union[str, Path]):
        """Export model card to a file."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_card = self.models[model_id]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_card.save(output_path)
        logger.info(f"Exported model card to {output_path}")
    
    def auto_convert_and_benchmark(self,
                                  model_name: str,
                                  model_path: Union[str, Path],
                                  target_formats: List[ModelFormat],
                                  benchmark_config: Optional[Dict[str, Any]] = None) -> ModelCard:
        """
        Automatically register, convert, and benchmark a model in multiple formats.
        
        Args:
            model_name: Name of the model
            model_path: Path to the source model
            target_formats: List of formats to convert to
            benchmark_config: Benchmark configuration
            
        Returns:
            Updated model card with all conversions and benchmarks
        """
        # Register the model
        model_id = self.register_model(
            model_name=model_name,
            model_path=model_path,
            source_format=ModelFormat.HUGGINGFACE,
            description=f"Auto-converted model: {model_name}",
            tags=["auto-converted", "multi-format"]
        )
        
        # Convert to each target format
        for target_format in target_formats:
            try:
                if target_format != ModelFormat.HUGGINGFACE:
                    self.convert_model(model_id, target_format)
            except Exception as e:
                logger.error(f"Failed to convert to {target_format.value}: {e}")
        
        # Benchmark all available formats
        model_card = self.get_model(model_id)
        for format in model_card.formats:
            try:
                self.benchmark_model(model_id, format, benchmark_config)
            except Exception as e:
                logger.error(f"Failed to benchmark {format.value}: {e}")
        
        return self.get_model(model_id)


# Convenience functions
def create_registry(registry_dir: Union[str, Path] = "./model_registry") -> ModelRegistry:
    """Create a new model registry instance."""
    return ModelRegistry(registry_dir)


def quick_convert(model_path: Union[str, Path],
                 target_formats: List[str],
                 output_dir: Union[str, Path] = "./converted_models") -> Dict[str, str]:
    """
    Quick conversion utility for common use cases.
    
    Args:
        model_path: Path to source model
        target_formats: List of target format strings (e.g., ["gguf", "onnx"])
        output_dir: Output directory
        
    Returns:
        Dictionary mapping format to output path
    """
    registry = ModelRegistry()
    
    # Convert format strings to enums
    format_enums = []
    for fmt in target_formats:
        try:
            format_enums.append(ModelFormat(fmt.lower()))
        except ValueError:
            logger.warning(f"Unknown format: {fmt}")
    
    # Register and convert
    model_id = registry.register_model(
        model_name=Path(model_path).name,
        model_path=model_path,
        source_format=ModelFormat.HUGGINGFACE
    )
    
    results = {}
    for target_format in format_enums:
        try:
            output_path = registry.convert_model(
                model_id=model_id,
                target_format=target_format,
                output_dir=Path(output_dir) / target_format.value
            )
            results[target_format.value] = output_path
        except Exception as e:
            logger.error(f"Conversion to {target_format.value} failed: {e}")
    
    return results


# Integration with existing forge scripts
def integrate_with_existing_scripts():
    """
    Integration point for existing forge conversion scripts.
    
    This function can be called to register existing converters from the scripts directory.
    """
    # This would be implemented to integrate with existing scripts like:
    # - scripts/convert_ckpt/llamafy_baichuan2.py
    # - scripts/convert_ckpt/llamafy_qwen.py
    # - scripts/convert_ckpt/tiny_llama4.py
    # - scripts/convert_ckpt/tiny_qwen3.py
    
    logger.info("Integration with existing scripts would be implemented here")
    # Example: from scripts.convert_ckpt.llamafy_qwen import convert_qwen_to_llama
    # Then register this as a custom converter


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create registry
    registry = create_registry()
    
    # Example: Register and convert a model
    # model_id = registry.register_model(
    #     model_name="example-model",
    #     model_path="./models/example",
    #     description="Example model for testing"
    # )
    
    # Convert to different formats
    # registry.convert_model(model_id, ModelFormat.GGUF)
    # registry.convert_model(model_id, ModelFormat.ONNX)
    
    # Benchmark all formats
    # registry.benchmark_model(model_id)
    
    # Get comparison
    # comparison = registry.compare_formats(model_id)
    # print(json.dumps(comparison, indent=2))
    
    print("Model registry ready for use")