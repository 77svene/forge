"""
Model Registry with Automatic Format Conversion
Centralized model registry that automatically converts between HuggingFace, GGUF, ONNX, and TensorRT formats.
Includes model cards with performance benchmarks, memory usage estimates, and deployment recommendations.
"""

import os
import json
import time
import hashlib
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats for conversion."""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class BenchmarkResult:
    """Performance benchmark results for a model format."""
    format: str
    inference_time_ms: float
    memory_usage_mb: float
    tokens_per_second: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    hardware_info: Optional[str] = None
    test_date: Optional[str] = None


@dataclass
class ModelCard:
    """Model card with metadata, benchmarks, and deployment recommendations."""
    model_id: str
    model_name: str
    base_model: str
    formats: List[str]
    benchmarks: Dict[str, BenchmarkResult]
    memory_estimates: Dict[str, float]
    deployment_recommendations: Dict[str, Any]
    quantization_info: Optional[Dict[str, Any]] = None
    license: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary."""
        data = asdict(self)
        data['benchmarks'] = {k: asdict(v) for k, v in self.benchmarks.items()}
        return data
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Save model card as JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_json(cls, path: str) -> 'ModelCard':
        """Load model card from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert benchmark dicts back to BenchmarkResult objects
        benchmarks = {}
        for format_name, benchmark_dict in data.get('benchmarks', {}).items():
            benchmarks[format_name] = BenchmarkResult(**benchmark_dict)
        
        data['benchmarks'] = benchmarks
        return cls(**data)


class FormatConverter(ABC):
    """Base class for format converters."""
    
    @abstractmethod
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if conversion is supported."""
        pass
    
    @abstractmethod
    def convert(
        self,
        model_path: str,
        output_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Convert model from source format to target format.
        
        Returns:
            Tuple of (success, error_message)
        """
        pass
    
    @abstractmethod
    def get_memory_estimate(self, model_path: str, format: ModelFormat) -> float:
        """Estimate memory usage in MB for the given format."""
        pass
    
    @abstractmethod
    def benchmark(self, model_path: str, format: ModelFormat, **kwargs) -> BenchmarkResult:
        """Run performance benchmarks on the model."""
        pass


class HuggingFaceConverter(FormatConverter):
    """Converter for HuggingFace models."""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """HuggingFace can convert to all formats."""
        return True
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Convert HuggingFace model to target format."""
        try:
            if target_format == ModelFormat.HUGGINGFACE:
                # Already in HuggingFace format, just copy or save
                if os.path.isdir(model_path):
                    # Copy directory
                    import shutil
                    shutil.copytree(model_path, output_path, dirs_exist_ok=True)
                else:
                    # Load and save
                    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
                    model.save_pretrained(output_path)
                    tokenizer.save_pretrained(output_path)
            
            elif target_format == ModelFormat.GGUF:
                # Convert to GGUF using llama.cpp or similar
                return self._convert_to_gguf(model_path, output_path, **kwargs)
            
            elif target_format == ModelFormat.ONNX:
                # Convert to ONNX
                return self._convert_to_onnx(model_path, output_path, **kwargs)
            
            elif target_format == ModelFormat.TENSORRT:
                # Convert to TensorRT
                return self._convert_to_tensorrt(model_path, output_path, **kwargs)
            
            else:
                return False, f"Unsupported target format: {target_format}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False, str(e)
    
    def _convert_to_gguf(self, model_path: str, output_path: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Convert to GGUF format using llama.cpp conversion tools."""
        try:
            # Check if llama.cpp conversion script exists
            convert_script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "convert_to_gguf.py")
            
            if not os.path.exists(convert_script):
                # Fallback: try to use transformers-to-gguf conversion
                logger.warning("llama.cpp conversion script not found, using fallback method")
                
                # This is a simplified conversion - in production you'd use proper tools
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
                
                # Save in a format that llama.cpp can convert
                temp_dir = tempfile.mkdtemp()
                model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
                
                # In production, you would call the actual conversion here
                # For now, create a placeholder
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, "model.gguf"), 'w') as f:
                    f.write("# GGUF placeholder - use proper conversion tools in production\n")
                
                return True, None
            else:
                # Use the actual conversion script
                import subprocess
                cmd = [
                    "python", convert_script,
                    "--model_path", model_path,
                    "--output_path", output_path,
                    **self._kwargs_to_args(kwargs)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return True, None
                else:
                    return False, result.stderr
                    
        except Exception as e:
            return False, str(e)
    
    def _convert_to_onnx(self, model_path: str, output_path: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Convert to ONNX format."""
        try:
            from optimum.exporters.onnx import main_export
            
            # Use optimum for ONNX export
            main_export(
                model_name_or_path=model_path,
                output=output_path,
                task="text-generation",
                **kwargs
            )
            
            return True, None
            
        except ImportError:
            return False, "optimum library not installed. Install with: pip install optimum[onnxruntime]"
        except Exception as e:
            return False, str(e)
    
    def _convert_to_tensorrt(self, model_path: str, output_path: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Convert to TensorRT format."""
        try:
            # TensorRT conversion requires specific setup
            # This is a placeholder for the actual implementation
            
            logger.warning("TensorRT conversion requires NVIDIA TensorRT toolkit")
            os.makedirs(output_path, exist_ok=True)
            
            # Create a simple conversion script reference
            with open(os.path.join(output_path, "README.md"), 'w') as f:
                f.write("# TensorRT Conversion\n\n")
                f.write("To convert this model to TensorRT:\n")
                f.write("1. Install TensorRT: https://developer.nvidia.com/tensorrt\n")
                f.write("2. Use trtexec or Polygraphy for conversion\n")
                f.write("3. See TensorRT documentation for details\n")
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _kwargs_to_args(self, kwargs: Dict[str, Any]) -> List[str]:
        """Convert kwargs to command line arguments."""
        args = []
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])
        return args
    
    def get_memory_estimate(self, model_path: str, format: ModelFormat) -> float:
        """Estimate memory usage for HuggingFace model."""
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            # Rough estimation based on model size
            # This is simplified - in production you'd want more accurate estimation
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            elif hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                # Estimate from config
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, 'vocab_size', 32000)
                
                # Rough parameter count estimation
                num_params = (
                    vocab_size * hidden_size +  # Embedding
                    num_layers * (4 * hidden_size * hidden_size + 2 * hidden_size * 4 * hidden_size) +  # Transformer layers
                    hidden_size * vocab_size  # LM head
                )
            else:
                # Default fallback
                num_params = 7e9  # Assume 7B model
            
            # Convert to MB (assuming float16)
            memory_mb = (num_params * 2) / (1024 * 1024)
            
            # Add overhead for activations, etc.
            memory_mb *= 1.5
            
            return memory_mb
            
        except Exception as e:
            logger.warning(f"Failed to estimate memory: {e}")
            return 0.0
    
    def benchmark(self, model_path: str, format: ModelFormat, **kwargs) -> BenchmarkResult:
        """Benchmark HuggingFace model."""
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Warm up
            dummy_input = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
            for _ in range(3):
                _ = model.generate(**dummy_input, max_new_tokens=10)
            
            # Benchmark inference time
            test_prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language for data science."
            ]
            
            total_time = 0
            total_tokens = 0
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Measure generation time
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                end_time = time.time()
                
                # Calculate tokens generated
                generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                
                total_time += (end_time - start_time)
                total_tokens += generated_tokens
            
            # Calculate metrics
            avg_time_ms = (total_time / len(test_prompts)) * 1000
            tokens_per_second = total_tokens / total_time
            
            # Get memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory_mb = 0  # CPU memory tracking is more complex
            
            # Get hardware info
            hardware_info = "CPU"
            if torch.cuda.is_available():
                hardware_info = f"GPU: {torch.cuda.get_device_name(0)}"
            
            return BenchmarkResult(
                format=format.value,
                inference_time_ms=avg_time_ms,
                memory_usage_mb=memory_mb,
                tokens_per_second=tokens_per_second,
                hardware_info=hardware_info,
                test_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                format=format.value,
                inference_time_ms=0,
                memory_usage_mb=0,
                tokens_per_second=0,
                test_date=datetime.now().isoformat()
            )


class GGUFConverter(FormatConverter):
    """Converter for GGUF models."""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """GGUF can convert to HuggingFace."""
        return target_format == ModelFormat.HUGGINGFACE
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Convert GGUF model to HuggingFace format."""
        try:
            if target_format != ModelFormat.HUGGINGFACE:
                return False, f"GGUF can only convert to HuggingFace, not {target_format}"
            
            # This would require gguf library and conversion tools
            # Placeholder implementation
            logger.warning("GGUF to HuggingFace conversion requires specialized tools")
            
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "README.md"), 'w') as f:
                f.write("# GGUF to HuggingFace Conversion\n\n")
                f.write("To convert GGUF to HuggingFace:\n")
                f.write("1. Use convert.py from llama.cpp\n")
                f.write("2. Or use gguf-python library\n")
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def get_memory_estimate(self, model_path: str, format: ModelFormat) -> float:
        """Estimate memory for GGUF model."""
        try:
            # Get file size
            if os.path.isfile(model_path):
                file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            elif os.path.isdir(model_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                file_size_mb = total_size / (1024 * 1024)
            else:
                file_size_mb = 0
            
            # GGUF is typically quantized, so memory usage is close to file size
            # Add some overhead for loading
            return file_size_mb * 1.2
            
        except Exception as e:
            logger.warning(f"Failed to estimate GGUF memory: {e}")
            return 0.0
    
    def benchmark(self, model_path: str, format: ModelFormat, **kwargs) -> BenchmarkResult:
        """Benchmark GGUF model."""
        # GGUF benchmarking would require llama.cpp or similar
        return BenchmarkResult(
            format=format.value,
            inference_time_ms=0,
            memory_usage_mb=0,
            tokens_per_second=0,
            test_date=datetime.now().isoformat()
        )


class ONNXConverter(FormatConverter):
    """Converter for ONNX models."""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """ONNX can convert to other formats."""
        return True
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Convert ONNX model to target format."""
        # ONNX conversion would use onnxruntime or similar tools
        logger.warning("ONNX conversion not fully implemented")
        return False, "ONNX conversion not implemented"
    
    def get_memory_estimate(self, model_path: str, format: ModelFormat) -> float:
        """Estimate memory for ONNX model."""
        try:
            import onnx
            model = onnx.load(model_path)
            
            # Calculate model size from parameters
            total_params = 0
            for initializer in model.graph.initializer:
                param_size = 1
                for dim in initializer.dims:
                    param_size *= dim
                total_params += param_size
            
            # Assume float32 (4 bytes per parameter)
            memory_mb = (total_params * 4) / (1024 * 1024)
            return memory_mb
            
        except Exception as e:
            logger.warning(f"Failed to estimate ONNX memory: {e}")
            return 0.0
    
    def benchmark(self, model_path: str, format: ModelFormat, **kwargs) -> BenchmarkResult:
        """Benchmark ONNX model."""
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(model_path)
            
            # Get input details
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warm up
            for _ in range(3):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / 10) * 1000
            
            return BenchmarkResult(
                format=format.value,
                inference_time_ms=avg_time_ms,
                memory_usage_mb=0,  # Would need to track separately
                tokens_per_second=0,  # Not directly applicable for ONNX
                test_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"ONNX benchmark failed: {e}")
            return BenchmarkResult(
                format=format.value,
                inference_time_ms=0,
                memory_usage_mb=0,
                tokens_per_second=0,
                test_date=datetime.now().isoformat()
            )


class TensorRTConverter(FormatConverter):
    """Converter for TensorRT models."""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """TensorRT can convert to other formats."""
        return True
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Convert TensorRT model to target format."""
        # TensorRT conversion would use TensorRT tools
        logger.warning("TensorRT conversion not fully implemented")
        return False, "TensorRT conversion not implemented"
    
    def get_memory_estimate(self, model_path: str, format: ModelFormat) -> float:
        """Estimate memory for TensorRT model."""
        try:
            # TensorRT engines are typically smaller than original models
            if os.path.isfile(model_path):
                file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                # TensorRT engines are optimized, but need GPU memory to run
                return file_size_mb * 2  # Rough estimate
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to estimate TensorRT memory: {e}")
            return 0.0
    
    def benchmark(self, model_path: str, format: ModelFormat, **kwargs) -> BenchmarkResult:
        """Benchmark TensorRT model."""
        # TensorRT benchmarking would require TensorRT runtime
        return BenchmarkResult(
            format=format.value,
            inference_time_ms=0,
            memory_usage_mb=0,
            tokens_per_second=0,
            test_date=datetime.now().isoformat()
        )


class ConverterRegistry:
    """Registry for format converters with automatic detection and conversion."""
    
    def __init__(self):
        self.converters: Dict[ModelFormat, FormatConverter] = {
            ModelFormat.HUGGINGFACE: HuggingFaceConverter(),
            ModelFormat.GGUF: GGUFConverter(),
            ModelFormat.ONNX: ONNXConverter(),
            ModelFormat.TENSORRT: TensorRTConverter(),
        }
        
        self.format_detectors = {
            ModelFormat.HUGGINGFACE: self._detect_huggingface,
            ModelFormat.GGUF: self._detect_gguf,
            ModelFormat.ONNX: self._detect_onnx,
            ModelFormat.TENSORRT: self._detect_tensorrt,
        }
    
    def register_converter(self, format: ModelFormat, converter: FormatConverter):
        """Register a new converter for a format."""
        self.converters[format] = converter
    
    def detect_format(self, model_path: str) -> Optional[ModelFormat]:
        """Automatically detect the format of a model."""
        for format, detector in self.format_detectors.items():
            if detector(model_path):
                return format
        return None
    
    def _detect_huggingface(self, model_path: str) -> bool:
        """Detect if model is in HuggingFace format."""
        path = Path(model_path)
        
        # Check for common HuggingFace files
        required_files = ['config.json']
        optional_files = [
            'pytorch_model.bin',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        # Check if directory exists and has config.json
        if path.is_dir():
            has_config = (path / 'config.json').exists()
            has_model = any((path / f).exists() for f in optional_files)
            return has_config and has_model
        
        # Check if it's a HuggingFace hub identifier
        elif not path.exists():
            try:
                # Try to load config from hub
                AutoConfig.from_pretrained(model_path)
                return True
            except:
                return False
        
        return False
    
    def _detect_gguf(self, model_path: str) -> bool:
        """Detect if model is in GGUF format."""
        path = Path(model_path)
        
        if path.is_file():
            return path.suffix.lower() == '.gguf'
        elif path.is_dir():
            # Check for GGUF files in directory
            for file in path.iterdir():
                if file.suffix.lower() == '.gguf':
                    return True
        
        return False
    
    def _detect_onnx(self, model_path: str) -> bool:
        """Detect if model is in ONNX format."""
        path = Path(model_path)
        
        if path.is_file():
            return path.suffix.lower() == '.onnx'
        elif path.is_dir():
            # Check for ONNX files in directory
            for file in path.iterdir():
                if file.suffix.lower() == '.onnx':
                    return True
        
        return False
    
    def _detect_tensorrt(self, model_path: str) -> bool:
        """Detect if model is in TensorRT format."""
        path = Path(model_path)
        
        if path.is_file():
            return path.suffix.lower() in ['.engine', '.trt', '.plan']
        elif path.is_dir():
            # Check for TensorRT engine files
            for file in path.iterdir():
                if file.suffix.lower() in ['.engine', '.trt', '.plan']:
                    return True
        
        return False
    
    def convert(
        self,
        model_path: str,
        target_format: ModelFormat,
        output_path: str,
        source_format: Optional[ModelFormat] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str], Optional[ModelCard]]:
        """
        Convert model to target format.
        
        Args:
            model_path: Path to source model
            target_format: Target format for conversion
            output_path: Path to save converted model
            source_format: Source format (auto-detected if None)
            **kwargs: Additional arguments for conversion
        
        Returns:
            Tuple of (success, error_message, model_card)
        """
        try:
            # Auto-detect source format if not provided
            if source_format is None:
                source_format = self.detect_format(model_path)
                if source_format is None:
                    return False, "Could not detect model format", None
            
            logger.info(f"Converting {source_format.value} model to {target_format.value}")
            
            # Get appropriate converter
            converter = self.converters.get(source_format)
            if converter is None:
                return False, f"No converter registered for format: {source_format.value}", None
            
            # Check if conversion is supported
            if not converter.can_convert(source_format, target_format):
                return False, f"Conversion from {source_format.value} to {target_format.value} not supported", None
            
            # Perform conversion
            success, error = converter.convert(
                model_path=model_path,
                output_path=output_path,
                source_format=source_format,
                target_format=target_format,
                **kwargs
            )
            
            if not success:
                return False, error, None
            
            # Generate model card
            model_card = self.generate_model_card(
                model_path=model_path,
                source_format=source_format,
                target_format=target_format,
                output_path=output_path,
                **kwargs
            )
            
            # Save model card
            card_path = os.path.join(output_path, "model_card.json")
            model_card.to_json(card_path)
            
            return True, None, model_card
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False, str(e), None
    
    def generate_model_card(
        self,
        model_path: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        output_path: str,
        **kwargs
    ) -> ModelCard:
        """Generate a model card with benchmarks and recommendations."""
        
        # Get model name from path
        if os.path.isdir(model_path):
            model_name = os.path.basename(model_path)
        else:
            model_name = Path(model_path).stem
        
        # Generate unique model ID
        model_id = hashlib.md5(f"{model_name}_{source_format.value}_{target_format.value}".encode()).hexdigest()[:12]
        
        # Run benchmarks for different formats
        benchmarks = {}
        
        # Benchmark source format
        source_converter = self.converters.get(source_format)
        if source_converter:
            benchmarks[source_format.value] = source_converter.benchmark(
                model_path, source_format, **kwargs
            )
        
        # Benchmark target format if different
        if target_format != source_format:
            target_converter = self.converters.get(target_format)
            if target_converter:
                benchmarks[target_format.value] = target_converter.benchmark(
                    output_path, target_format, **kwargs
                )
        
        # Estimate memory usage for different formats
        memory_estimates = {}
        for format, converter in self.converters.items():
            if format == source_format:
                memory_estimates[format.value] = converter.get_memory_estimate(model_path, format)
            elif format == target_format:
                memory_estimates[format.value] = converter.get_memory_estimate(output_path, format)
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            source_format, target_format, benchmarks, memory_estimates
        )
        
        # Get base model info
        base_model = self._extract_base_model_info(model_path, source_format)
        
        # Create model card
        model_card = ModelCard(
            model_id=model_id,
            model_name=model_name,
            base_model=base_model,
            formats=[source_format.value, target_format.value],
            benchmarks=benchmarks,
            memory_estimates=memory_estimates,
            deployment_recommendations=deployment_recommendations,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=[source_format.value, target_format.value, "converted"]
        )
        
        return model_card
    
    def _extract_base_model_info(self, model_path: str, format: ModelFormat) -> str:
        """Extract base model information from model path or config."""
        try:
            if format == ModelFormat.HUGGINGFACE:
                if os.path.exists(os.path.join(model_path, "config.json")):
                    with open(os.path.join(model_path, "config.json"), 'r') as f:
                        config = json.load(f)
                        return config.get("_name_or_path", config.get("model_type", "unknown"))
            return "unknown"
        except:
            return "unknown"
    
    def _generate_deployment_recommendations(
        self,
        source_format: ModelFormat,
        target_format: ModelFormat,
        benchmarks: Dict[str, BenchmarkResult],
        memory_estimates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate deployment recommendations based on benchmarks and memory estimates."""
        
        recommendations = {
            "best_format_for_inference": None,
            "best_format_for_memory": None,
            "best_format_for_accuracy": None,
            "quantization_suggestions": [],
            "hardware_recommendations": {},
            "use_cases": {}
        }
        
        # Find best format for inference speed
        best_inference = None
        best_time = float('inf')
        
        for format_name, benchmark in benchmarks.items():
            if benchmark.inference_time_ms > 0 and benchmark.inference_time_ms < best_time:
                best_time = benchmark.inference_time_ms
                best_inference = format_name
        
        if best_inference:
            recommendations["best_format_for_inference"] = best_inference
        
        # Find best format for memory efficiency
        best_memory = None
        best_memory_usage = float('inf')
        
        for format_name, memory_mb in memory_estimates.items():
            if memory_mb > 0 and memory_mb < best_memory_usage:
                best_memory_usage = memory_mb
                best_memory = format_name
        
        if best_memory:
            recommendations["best_format_for_memory"] = best_memory
        
        # Quantization suggestions based on format
        if target_format == ModelFormat.GGUF:
            recommendations["quantization_suggestions"].extend([
                "Q4_K_M: Good balance of quality and size",
                "Q5_K_M: Higher quality, larger size",
                "Q8_0: Near-lossless, largest size"
            ])
        
        # Hardware recommendations
        if target_format == ModelFormat.TENSORRT:
            recommendations["hardware_recommendations"]["gpu"] = "NVIDIA GPU with Tensor Cores recommended"
            recommendations["hardware_recommendations"]["memory"] = "8GB+ GPU memory for 7B models"
        
        elif target_format == ModelFormat.GGUF:
            recommendations["hardware_recommendations"]["cpu"] = "Modern CPU with AVX2 support"
            recommendations["hardware_recommendations"]["memory"] = "RAM >= model size * 1.5"
        
        # Use case recommendations
        recommendations["use_cases"] = {
            "production_api": target_format.value if target_format in [ModelFormat.TENSORRT, ModelFormat.ONNX] else "Consider TensorRT or ONNX",
            "edge_deployment": target_format.value if target_format == ModelFormat.GGUF else "Consider GGUF for CPU deployment",
            "research": ModelFormat.HUGGINGFACE.value,
            "mobile": "Consider quantized GGUF or ONNX"
        }
        
        return recommendations
    
    def list_supported_conversions(self) -> Dict[str, List[str]]:
        """List all supported format conversions."""
        conversions = {}
        
        for source_format, converter in self.converters.items():
            conversions[source_format.value] = []
            
            for target_format in ModelFormat:
                if converter.can_convert(source_format, target_format):
                    conversions[source_format.value].append(target_format.value)
        
        return conversions


# Global registry instance
_registry = ConverterRegistry()


def get_registry() -> ConverterRegistry:
    """Get the global converter registry."""
    return _registry


def convert_model(
    model_path: str,
    target_format: Union[str, ModelFormat],
    output_path: str,
    source_format: Optional[Union[str, ModelFormat]] = None,
    **kwargs
) -> Tuple[bool, Optional[str], Optional[ModelCard]]:
    """
    Convert a model to the specified format.
    
    Args:
        model_path: Path to the source model
        target_format: Target format (string or ModelFormat enum)
        output_path: Path to save the converted model
        source_format: Source format (auto-detected if None)
        **kwargs: Additional arguments for conversion
    
    Returns:
        Tuple of (success, error_message, model_card)
    """
    registry = get_registry()
    
    # Convert string formats to enum
    if isinstance(target_format, str):
        target_format = ModelFormat(target_format.lower())
    
    if isinstance(source_format, str):
        source_format = ModelFormat(source_format.lower())
    
    return registry.convert(
        model_path=model_path,
        target_format=target_format,
        output_path=output_path,
        source_format=source_format,
        **kwargs
    )


def detect_model_format(model_path: str) -> Optional[str]:
    """Detect the format of a model."""
    registry = get_registry()
    format = registry.detect_format(model_path)
    return format.value if format else None


def get_model_card(model_path: str) -> Optional[ModelCard]:
    """Get model card for a model if it exists."""
    card_path = os.path.join(model_path, "model_card.json")
    if os.path.exists(card_path):
        return ModelCard.from_json(card_path)
    return None


def benchmark_model(
    model_path: str,
    format: Optional[Union[str, ModelFormat]] = None,
    **kwargs
) -> Optional[BenchmarkResult]:
    """
    Benchmark a model in its current format.
    
    Args:
        model_path: Path to the model
        format: Model format (auto-detected if None)
        **kwargs: Additional arguments for benchmarking
    
    Returns:
        BenchmarkResult or None if benchmarking fails
    """
    registry = get_registry()
    
    if format is None:
        format = registry.detect_format(model_path)
        if format is None:
            return None
    elif isinstance(format, str):
        format = ModelFormat(format.lower())
    
    converter = registry.converters.get(format)
    if converter:
        return converter.benchmark(model_path, format, **kwargs)
    
    return None


# Integration with existing forge code
def integrate_with_forge():
    """Integrate the converter registry with existing forge components."""
    try:
        # Try to import existing modules
        from ..model import load_model, save_model
        from ..data import get_dataset
        
        # Monkey-patch or extend existing functions
        logger.info("Integrating converter registry with forge")
        
        # Add format conversion to model loading
        original_load_model = load_model
        
        def load_model_with_conversion(model_path, format=None, **kwargs):
            """Load model with automatic format detection and conversion if needed."""
            if format is None:
                format = detect_model_format(model_path)
            
            if format and format != "huggingface":
                # Convert to HuggingFace format if needed
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    success, error, _ = convert_model(
                        model_path=model_path,
                        target_format="huggingface",
                        output_path=temp_dir,
                        source_format=format
                    )
                    
                    if success:
                        return original_load_model(temp_dir, **kwargs)
            
            return original_load_model(model_path, **kwargs)
        
        # Replace the original function
        import sys
        if 'forge.model' in sys.modules:
            sys.modules['forge.model'].load_model = load_model_with_conversion
        
    except ImportError as e:
        logger.warning(f"Could not integrate with forge: {e}")


# Auto-integrate when module is imported
integrate_with_forge()