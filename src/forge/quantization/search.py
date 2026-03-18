# src/forge/quantization/search.py

"""
Adaptive Quantization Engine for forge
Dynamic quantization system that automatically selects optimal quantization methods
based on hardware constraints and accuracy requirements.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import GPUtil
from pathlib import Path
from collections import defaultdict
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods"""
    GPTQ = "gptq"
    AWQ = "awq"
    EXL2 = "exl2"
    HQQ = "hqq"
    BITSANDBYTES = "bitsandbytes"


@dataclass
class HardwareConstraints:
    """Hardware constraints for quantization selection"""
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: Tuple[int, int] = (0, 0)
    cpu_memory_gb: float = 0.0
    num_gpus: int = 1
    available_disk_gb: float = 0.0
    use_cpu_only: bool = False
    target_device: str = "auto"
    
    @classmethod
    def detect_hardware(cls) -> "HardwareConstraints":
        """Auto-detect hardware constraints"""
        constraints = cls()
        
        # Detect GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                constraints.gpu_memory_gb = sum(gpu.memoryTotal for gpu in gpus) / 1024.0
                constraints.num_gpus = len(gpus)
                # Get compute capability from first GPU
                if gpus[0].id is not None:
                    device = torch.device(f"cuda:{gpus[0].id}")
                    constraints.gpu_compute_capability = torch.cuda.get_device_capability(device)
        except:
            logger.warning("Could not detect GPU information, assuming CPU-only")
            constraints.use_cpu_only = True
        
        # Detect CPU memory
        constraints.cpu_memory_gb = psutil.virtual_memory().total / (1032**3)
        
        # Detect disk space
        disk_usage = psutil.disk_usage('/')
        constraints.available_disk_gb = disk_usage.free / (1032**3)
        
        # Determine target device
        if constraints.use_cpu_only or constraints.gpu_memory_gb < 2.0:
            constraints.target_device = "cpu"
        else:
            constraints.target_device = "cuda"
        
        return constraints


@dataclass
class AccuracyRequirements:
    """Accuracy requirements for quantization"""
    max_accuracy_loss: float = 0.01  # Maximum allowed accuracy loss (0-1)
    evaluation_metric: str = "perplexity"  # perplexity, accuracy, etc.
    calibration_samples: int = 128
    validation_samples: int = 512
    min_layer_coverage: float = 0.95  # Minimum percentage of layers to quantize


@dataclass
class QuantizationConfig:
    """Configuration for a quantization method"""
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    damp_percent: float = 0.01
    true_sequential: bool = True
    static_groups: bool = False
    device_map: str = "auto"
    memory_budget_gb: float = 4.0
    target_modules: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "method": self.method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "damp_percent": self.damp_percent,
            "true_sequential": self.true_sequential,
            "static_groups": self.static_groups,
            "device_map": self.device_map,
            "memory_budget_gb": self.memory_budget_gb,
            "target_modules": self.target_modules
        }


@dataclass
class QuantizationResult:
    """Result of a quantization evaluation"""
    config: QuantizationConfig
    memory_usage_gb: float
    inference_speed: float  # tokens/sec
    accuracy_score: float
    calibration_time: float
    quantization_time: float
    model_size_gb: float
    success: bool = True
    error_message: Optional[str] = None


class CalibrationDatasetSelector:
    """Selects optimal calibration datasets for quantization"""
    
    # Default calibration datasets by model family
    DEFAULT_CALIBRATION_DATASETS = {
        "llama": ["wikitext", "c4", "ptb"],
        "mistral": ["wikitext", "c4"],
        "qwen": ["wikitext", "c4", "pile"],
        "baichuan": ["wikitext", "c4"],
        "chatglm": ["wikitext", "c4"],
        "default": ["wikitext", "c4"]
    }
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.max_length = getattr(tokenizer, "model_max_length", 2048)
    
    def select_dataset(
        self, 
        model_name: str, 
        num_samples: int = 128,
        custom_dataset: Optional[Dataset] = None
    ) -> Dataset:
        """Select calibration dataset based on model architecture"""
        
        if custom_dataset is not None:
            return self._prepare_dataset(custom_dataset, num_samples)
        
        # Determine model family
        model_family = self._detect_model_family(model_name)
        
        # Select dataset based on model family
        dataset_names = self.DEFAULT_CALIBRATION_DATASETS.get(
            model_family, 
            self.DEFAULT_CALIBRATION_DATASETS["default"]
        )
        
        # Try datasets in order until one works
        for dataset_name in dataset_names:
            try:
                dataset = self._load_calibration_dataset(dataset_name, num_samples)
                if dataset is not None:
                    logger.info(f"Selected calibration dataset: {dataset_name}")
                    return dataset
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Fallback to synthetic dataset
        logger.warning("Using synthetic calibration dataset")
        return self._create_synthetic_dataset(num_samples)
    
    def _detect_model_family(self, model_name: str) -> str:
        """Detect model family from model name"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "baichuan" in model_name_lower:
            return "baichuan"
        elif "chatglm" in model_name_lower:
            return "chatglm"
        else:
            return "default"
    
    def _load_calibration_dataset(self, dataset_name: str, num_samples: int) -> Dataset:
        """Load and prepare calibration dataset"""
        
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            dataset = dataset.take(num_samples * 10)  # Take more than needed
        elif dataset_name == "ptb":
            dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
        elif dataset_name == "pile":
            dataset = load_dataset("pile", split="train", streaming=True)
            dataset = dataset.take(num_samples * 10)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self._prepare_dataset(dataset, num_samples)
    
    def _prepare_dataset(self, dataset: Dataset, num_samples: int) -> Dataset:
        """Tokenize and prepare dataset"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # Take subset if needed
        if len(dataset) > num_samples:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def _create_synthetic_dataset(self, num_samples: int) -> Dataset:
        """Create synthetic calibration dataset"""
        # Create random text samples
        vocab_size = self.tokenizer.vocab_size
        samples = []
        
        for _ in range(num_samples):
            # Generate random token sequence
            seq_length = np.random.randint(128, min(512, self.max_length))
            tokens = torch.randint(0, vocab_size, (seq_length,))
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append({"text": text})
        
        return Dataset.from_list(samples)


class AccuracyEvaluator:
    """Evaluates model accuracy after quantization"""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def evaluate_perplexity(
        self, 
        model: torch.nn.Module, 
        dataset: Dataset,
        max_samples: int = 512
    ) -> float:
        """Evaluate model perplexity on dataset"""
        
        model.eval()
        device = next(model.parameters()).device
        
        total_loss = 0.0
        total_tokens = 0
        
        # Limit samples
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                
                try:
                    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
                    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_accuracy(
        self,
        model: torch.nn.Module,
        task: str = "hellaswag",
        num_samples: int = 100
    ) -> float:
        """Evaluate model accuracy on specific task"""
        
        # This is a simplified version - in production, you'd use lm-evaluation-harness
        try:
            # Placeholder for actual evaluation
            # In practice, integrate with lm-evaluation-harness
            logger.info(f"Evaluating on {task} with {num_samples} samples")
            return 0.85  # Placeholder score
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0


class QuantizationMethodSelector:
    """Selects optimal quantization method based on constraints"""
    
    # Method compatibility matrix
    METHOD_COMPATIBILITY = {
        QuantizationMethod.GPTQ: {
            "min_gpu_memory_gb": 4.0,
            "supports_cpu": False,
            "best_for": ["nvidia_gpu", "high_accuracy"],
            "memory_efficiency": 0.7,
            "speed": 0.8,
            "accuracy": 0.9
        },
        QuantizationMethod.AWQ: {
            "min_gpu_memory_gb": 6.0,
            "supports_cpu": False,
            "best_for": ["nvidia_gpu", "inference_speed"],
            "memory_efficiency": 0.8,
            "speed": 0.9,
            "accuracy": 0.85
        },
        QuantizationMethod.EXL2: {
            "min_gpu_memory_gb": 8.0,
            "supports_cpu": False,
            "best_for": ["nvidia_gpu", "exllama_backend"],
            "memory_efficiency": 0.9,
            "speed": 0.95,
            "accuracy": 0.8
        },
        QuantizationMethod.HQQ: {
            "min_gpu_memory_gb": 2.0,
            "supports_cpu": True,
            "best_for": ["cpu", "fast_calibration"],
            "memory_efficiency": 0.6,
            "speed": 0.7,
            "accuracy": 0.75
        },
        QuantizationMethod.BITSANDBYTES: {
            "min_gpu_memory_gb": 4.0,
            "supports_cpu": True,
            "best_for": ["ease_of_use", "integration"],
            "memory_efficiency": 0.65,
            "speed": 0.75,
            "accuracy": 0.8
        }
    }
    
    @classmethod
    def select_methods(
        cls,
        hardware: HardwareConstraints,
        accuracy_req: AccuracyRequirements,
        model_config: Optional[Dict] = None
    ) -> List[QuantizationMethod]:
        """Select compatible quantization methods"""
        
        compatible_methods = []
        
        for method, specs in cls.METHOD_COMPATIBILITY.items():
            # Check GPU memory requirement
            if hardware.target_device == "cuda":
                if hardware.gpu_memory_gb < specs["min_gpu_memory_gb"]:
                    continue
            
            # Check CPU support
            if hardware.target_device == "cpu" and not specs["supports_cpu"]:
                continue
            
            # Check accuracy requirement
            if specs["accuracy"] < (1.0 - accuracy_req.max_accuracy_loss):
                # Method might not meet accuracy requirements
                # Still include but with lower priority
                pass
            
            compatible_methods.append(method)
        
        # Sort by expected performance
        compatible_methods.sort(
            key=lambda m: (
                cls.METHOD_COMPATIBILITY[m]["accuracy"] * 0.4 +
                cls.METHOD_COMPATIBILITY[m]["speed"] * 0.3 +
                cls.METHOD_COMPATIBILITY[m]["memory_efficiency"] * 0.3
            ),
            reverse=True
        )
        
        return compatible_methods
    
    @classmethod
    def get_default_config(
        cls,
        method: QuantizationMethod,
        hardware: HardwareConstraints,
        model_size_gb: float
    ) -> QuantizationConfig:
        """Get default configuration for a method"""
        
        # Calculate bits based on available memory
        if hardware.target_device == "cuda":
            available_memory = hardware.gpu_memory_gb * 0.8  # Leave 20% buffer
        else:
            available_memory = hardware.cpu_memory_gb * 0.5
        
        # Estimate required bits
        if model_size_gb > available_memory * 2:
            bits = 2
        elif model_size_gb > available_memory:
            bits = 4
        else:
            bits = 8
        
        # Method-specific defaults
        if method == QuantizationMethod.GPTQ:
            return QuantizationConfig(
                method=method,
                bits=bits,
                group_size=128,
                desc_act=False,
                sym=True,
                damp_percent=0.01,
                true_sequential=True
            )
        elif method == QuantizationMethod.AWQ:
            return QuantizationConfig(
                method=method,
                bits=bits,
                group_size=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        elif method == QuantizationMethod.EXL2:
            return QuantizationConfig(
                method=method,
                bits=bits,
                group_size=32
            )
        elif method == QuantizationMethod.HQQ:
            return QuantizationConfig(
                method=method,
                bits=bits,
                group_size=64
            )
        elif method == QuantizationMethod.BITSANDBYTES:
            return QuantizationConfig(
                method=method,
                bits=bits,
                group_size=128
            )
        else:
            return QuantizationConfig(method=method, bits=bits)


class AdaptiveQuantizationEngine:
    """Main adaptive quantization engine"""
    
    def __init__(
        self,
        model_name_or_path: str,
        hardware_constraints: Optional[HardwareConstraints] = None,
        accuracy_requirements: Optional[AccuracyRequirements] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.hardware = hardware_constraints or HardwareConstraints.detect_hardware()
        self.accuracy_req = accuracy_requirements or AccuracyRequirements()
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache/forge/quantization")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.model_config = None
        self.model_size_gb = 0.0
        
        # Load model info
        self._load_model_info()
        
        # Initialize calibration selector
        self.calibration_selector = None
        
        # Results storage
        self.search_results: List[QuantizationResult] = []
        self.best_config: Optional[QuantizationConfig] = None
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_model_info(self):
        """Load model configuration and estimate size"""
        
        try:
            # Load config
            self.model_config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Estimate model size
            self.model_size_gb = self._estimate_model_size()
            
            logger.info(f"Model size estimated: {self.model_size_gb:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
            raise
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB"""
        
        # Get number of parameters from config
        if hasattr(self.model_config, "num_parameters"):
            num_params = self.model_config.num_parameters
        elif hasattr(self.model_config, "hidden_size") and hasattr(self.model_config, "num_hidden_layers"):
            # Estimate for transformer models
            hidden_size = self.model_config.hidden_size
            num_layers = self.model_config.num_hidden_layers
            vocab_size = getattr(self.model_config, "vocab_size", 32000)
            intermediate_size = getattr(self.model_config, "intermediate_size", hidden_size * 4)
            
            # Rough estimation
            num_params = (
                vocab_size * hidden_size +  # Embedding
                num_layers * (
                    3 * hidden_size * hidden_size +  # QKV projections
                    hidden_size * hidden_size +  # Output projection
                    2 * hidden_size * intermediate_size  # FFN
                ) +
                hidden_size * vocab_size  # LM head
            )
        else:
            # Default to 7B parameters
            num_params = 7e9
        
        # Convert to GB (assuming float32)
        size_gb = num_params * 4 / (1024**3)
        
        return size_gb
    
    def run_quantization_search(
        self,
        calibration_dataset: Optional[Dataset] = None,
        max_search_time: float = 3600,  # 1 hour max
        target_memory_gb: Optional[float] = None
    ) -> QuantizationConfig:
        """Run quantization search to find optimal configuration"""
        
        logger.info("Starting quantization search...")
        start_time = time.time()
        
        # Select compatible methods
        compatible_methods = QuantizationMethodSelector.select_methods(
            self.hardware,
            self.accuracy_req,
            self.model_config.to_dict() if self.model_config else None
        )
        
        if not compatible_methods:
            raise ValueError("No compatible quantization methods found for given hardware constraints")
        
        logger.info(f"Compatible methods: {[m.value for m in compatible_methods]}")
        
        # Initialize calibration dataset selector
        if self.calibration_selector is None:
            self.calibration_selector = CalibrationDatasetSelector(self.tokenizer)
        
        # Select calibration dataset
        if calibration_dataset is None:
            calibration_dataset = self.calibration_selector.select_dataset(
                self.model_name_or_path,
                num_samples=self.accuracy_req.calibration_samples
            )
        
        # Initialize accuracy evaluator
        accuracy_evaluator = AccuracyEvaluator(self.tokenizer)
        
        # Try each method
        best_result = None
        best_score = -float('inf')
        
        for method in compatible_methods:
            if time.time() - start_time > max_search_time:
                logger.warning("Search time limit reached")
                break
            
            logger.info(f"Trying method: {method.value}")
            
            # Get default config for method
            config = QuantizationMethodSelector.get_default_config(
                method,
                self.hardware,
                self.model_size_gb
            )
            
            # Adjust bits if target memory specified
            if target_memory_gb:
                config.bits = self._calculate_optimal_bits(target_memory_gb, method)
            
            # Try quantization with this config
            result = self._try_quantization(
                config,
                calibration_dataset,
                accuracy_evaluator
            )
            
            if result.success:
                self.search_results.append(result)
                
                # Calculate score (weighted combination of metrics)
                score = self._calculate_config_score(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    self.best_config = config
                    
                logger.info(
                    f"Method {method.value}: "
                    f"Memory={result.memory_usage_gb:.2f}GB, "
                    f"Accuracy={result.accuracy_score:.3f}, "
                    f"Speed={result.inference_speed:.1f} tok/s"
                )
        
        if best_result is None:
            raise RuntimeError("No successful quantization configuration found")
        
        # Generate accuracy-memory tradeoff curve
        self._generate_tradeoff_curve()
        
        # Save search results
        self._save_search_results()
        
        logger.info(f"Search completed. Best method: {self.best_config.method.value}")
        
        return self.best_config
    
    def _calculate_optimal_bits(self, target_memory_gb: float, method: QuantizationMethod) -> int:
        """Calculate optimal bit width for target memory"""
        
        # Rough estimation: model_size * (bits/32) * overhead_factor
        overhead_factors = {
            QuantizationMethod.GPTQ: 1.2,
            QuantizationMethod.AWQ: 1.15,
            QuantizationMethod.EXL2: 1.1,
            QuantizationMethod.HQQ: 1.25,
            QuantizationMethod.BITSANDBYTES: 1.3
        }
        
        overhead = overhead_factors.get(method, 1.2)
        
        # Calculate required bits
        required_bits = (target_memory_gb / (self.model_size_gb * overhead)) * 32
        
        # Round to standard bit widths
        if required_bits <= 2:
            return 2
        elif required_bits <= 3:
            return 3
        elif required_bits <= 4:
            return 4
        elif required_bits <= 8:
            return 8
        else:
            return 16
    
    def _try_quantization(
        self,
        config: QuantizationConfig,
        calibration_dataset: Dataset,
        accuracy_evaluator: AccuracyEvaluator
    ) -> QuantizationResult:
        """Try quantization with given configuration"""
        
        start_time = time.time()
        
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.hardware.target_device == "cuda" else "cpu",
                trust_remote_code=True
            )
            
            # Apply quantization based on method
            quantized_model = self._apply_quantization(model, config, calibration_dataset)
            
            # Measure memory usage
            memory_usage = self._measure_memory_usage(quantized_model)
            
            # Evaluate accuracy
            accuracy_score = accuracy_evaluator.evaluate_perplexity(
                quantized_model,
                calibration_dataset,
                max_samples=self.accuracy_req.validation_samples
            )
            
            # Measure inference speed
            inference_speed = self._measure_inference_speed(quantized_model)
            
            # Calculate model size
            model_size = self._calculate_model_size(quantized_model)
            
            quantization_time = time.time() - start_time
            
            return QuantizationResult(
                config=config,
                memory_usage_gb=memory_usage,
                inference_speed=inference_speed,
                accuracy_score=accuracy_score,
                calibration_time=0.0,  # Would be measured during actual calibration
                quantization_time=quantization_time,
                model_size_gb=model_size,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Quantization failed for {config.method.value}: {e}")
            return QuantizationResult(
                config=config,
                memory_usage_gb=0.0,
                inference_speed=0.0,
                accuracy_score=0.0,
                calibration_time=0.0,
                quantization_time=time.time() - start_time,
                model_size_gb=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _apply_quantization(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        calibration_dataset: Dataset
    ) -> torch.nn.Module:
        """Apply quantization to model"""
        
        # This is a simplified version - in production, you'd use the actual quantization libraries
        # Each method would have its own implementation
        
        method = config.method
        
        if method == QuantizationMethod.GPTQ:
            # Placeholder for GPTQ quantization
            # In practice: from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            logger.info("Applying GPTQ quantization (placeholder)")
            return model
            
        elif method == QuantizationMethod.AWQ:
            # Placeholder for AWQ quantization
            # In practice: from awq import AutoAWQForCausalLM
            logger.info("Applying AWQ quantization (placeholder)")
            return model
            
        elif method == QuantizationMethod.EXL2:
            # Placeholder for EXL2 quantization
            logger.info("Applying EXL2 quantization (placeholder)")
            return model
            
        elif method == QuantizationMethod.HQQ:
            # Placeholder for HQQ quantization
            # In practice: from hqq.core.quantize import HQQBackend
            logger.info("Applying HQQ quantization (placeholder)")
            return model
            
        elif method == QuantizationMethod.BITSANDBYTES:
            # Placeholder for BitsAndBytes quantization
            # In practice: from transformers import BitsAndBytesConfig
            logger.info("Applying BitsAndBytes quantization (placeholder)")
            return model
            
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
    
    def _measure_memory_usage(self, model: torch.nn.Module) -> float:
        """Measure GPU memory usage of model"""
        
        if self.hardware.target_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            # Forward pass to measure memory
            dummy_input = torch.randint(0, 1000, (1, 128)).to("cuda")
            with torch.no_grad():
                _ = model(dummy_input)
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            return memory_mb / 1024.0  # Convert to GB
        else:
            # Estimate CPU memory usage
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            total_size = (param_size + buffer_size) / (1024**3)
            return total_size
    
    def _measure_inference_speed(self, model: torch.nn.Module) -> float:
        """Measure inference speed in tokens/second"""
        
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (1, 128)).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Measure
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - start_time
        
        tokens_per_second = (10 * 128) / elapsed
        return tokens_per_second
    
    def _calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in GB"""
        
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024**3)
        return total_size
    
    def _calculate_config_score(self, result: QuantizationResult) -> float:
        """Calculate score for quantization configuration"""
        
        # Normalize metrics
        memory_score = 1.0 / (1.0 + result.memory_usage_gb)  # Lower is better
        speed_score = result.inference_speed / 100.0  # Normalize to ~1.0
        accuracy_score = 1.0 / (1.0 + result.accuracy_score)  # Lower perplexity is better
        
        # Weighted combination
        score = (
            memory_score * 0.4 +
            speed_score * 0.3 +
            accuracy_score * 0.3
        )
        
        return score
    
    def _generate_tradeoff_curve(self):
        """Generate accuracy-memory tradeoff curve"""
        
        if not self.search_results:
            return
        
        # Sort by memory usage
        sorted_results = sorted(self.search_results, key=lambda x: x.memory_usage_gb)
        
        # Extract data for curve
        memory_values = [r.memory_usage_gb for r in sorted_results]
        accuracy_values = [r.accuracy_score for r in sorted_results]
        method_names = [r.config.method.value for r in sorted_results]
        
        # Save tradeoff data
        tradeoff_data = {
            "memory_gb": memory_values,
            "accuracy": accuracy_values,
            "methods": method_names,
            "model": self.model_name_or_path,
            "hardware": {
                "gpu_memory_gb": self.hardware.gpu_memory_gb,
                "cpu_memory_gb": self.hardware.cpu_memory_gb,
                "target_device": self.hardware.target_device
            }
        }
        
        tradeoff_path = os.path.join(self.cache_dir, "tradeoff_curve.json")
        with open(tradeoff_path, "w") as f:
            json.dump(tradeoff_data, f, indent=2)
        
        logger.info(f"Tradeoff curve saved to {tradeoff_path}")
    
    def _save_search_results(self):
        """Save search results to file"""
        
        results_data = {
            "model": self.model_name_or_path,
            "hardware": {
                "gpu_memory_gb": self.hardware.gpu_memory_gb,
                "cpu_memory_gb": self.hardware.cpu_memory_gb,
                "target_device": self.hardware.target_device
            },
            "accuracy_requirements": {
                "max_accuracy_loss": self.accuracy_req.max_accuracy_loss,
                "calibration_samples": self.accuracy_req.calibration_samples
            },
            "best_config": self.best_config.to_dict() if self.best_config else None,
            "search_results": [
                {
                    "config": r.config.to_dict(),
                    "memory_usage_gb": r.memory_usage_gb,
                    "inference_speed": r.inference_speed,
                    "accuracy_score": r.accuracy_score,
                    "quantization_time": r.quantization_time,
                    "model_size_gb": r.model_size_gb,
                    "success": r.success
                }
                for r in self.search_results
            ]
        }
        
        results_path = os.path.join(self.cache_dir, "search_results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Search results saved to {results_path}")
    
    def quantize_model(
        self,
        config: Optional[QuantizationConfig] = None,
        output_dir: Optional[str] = None,
        save_format: str = "safetensors"
    ) -> torch.nn.Module:
        """Quantize model using best or specified configuration"""
        
        if config is None:
            if self.best_config is None:
                raise ValueError("No quantization configuration available. Run search first.")
            config = self.best_config
        
        logger.info(f"Quantizing model with {config.method.value}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto" if self.hardware.target_device == "cuda" else "cpu",
            trust_remote_code=True
        )
        
        # Select calibration dataset
        calibration_dataset = self.calibration_selector.select_dataset(
            self.model_name_or_path,
            num_samples=self.accuracy_req.calibration_samples
        )
        
        # Apply quantization
        quantized_model = self._apply_quantization(model, config, calibration_dataset)
        
        # Save model if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            quantized_model.save_pretrained(
                output_dir,
                save_format=save_format
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save config
            config_path = os.path.join(output_dir, "quantization_config.json")
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info(f"Quantized model saved to {output_dir}")
        
        return quantized_model


def create_quantization_engine(
    model_name_or_path: str,
    hardware_constraints: Optional[Dict] = None,
    accuracy_requirements: Optional[Dict] = None,
    **kwargs
) -> AdaptiveQuantizationEngine:
    """Factory function to create quantization engine"""
    
    # Convert dicts to dataclasses if provided
    hardware = None
    if hardware_constraints:
        hardware = HardwareConstraints(**hardware_constraints)
    
    accuracy = None
    if accuracy_requirements:
        accuracy = AccuracyRequirements(**accuracy_requirements)
    
    return AdaptiveQuantizationEngine(
        model_name_or_path=model_name_or_path,
        hardware_constraints=hardware,
        accuracy_requirements=accuracy,
        **kwargs
    )


# Integration with existing forge modules
def integrate_with_forge():
    """Integration points with existing forge modules"""
    
    # This function would be called during forge initialization
    # to register the quantization search functionality
    
    # Example integration points:
    # 1. Add quantization options to training scripts
    # 2. Add quantization search to model conversion scripts
    # 3. Add quantization benchmarking to evaluation scripts
    
    pass


# CLI interface for quantization search
def quantization_search_cli():
    """Command-line interface for quantization search"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Quantization Search")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./quantized_model", help="Output directory")
    parser.add_argument("--max-accuracy-loss", type=float, default=0.01, help="Maximum accuracy loss")
    parser.add_argument("--target-memory-gb", type=float, help="Target memory in GB")
    parser.add_argument("--max-search-time", type=int, default=3600, help="Maximum search time in seconds")
    
    args = parser.parse_args()
    
    # Create accuracy requirements
    accuracy_req = AccuracyRequirements(
        max_accuracy_loss=args.max_accuracy_loss,
        calibration_samples=128,
        validation_samples=512
    )
    
    # Create quantization engine
    engine = create_quantization_engine(
        model_name_or_path=args.model,
        accuracy_requirements=accuracy_req.__dict__
    )
    
    # Run search
    try:
        best_config = engine.run_quantization_search(
            max_search_time=args.max_search_time,
            target_memory_gb=args.target_memory_gb
        )
        
        print(f"\nBest quantization configuration:")
        print(f"Method: {best_config.method.value}")
        print(f"Bits: {best_config.bits}")
        print(f"Group size: {best_config.group_size}")
        
        # Quantize and save model
        if args.output_dir:
            print(f"\nQuantizing and saving model to {args.output_dir}...")
            engine.quantize_model(
                config=best_config,
                output_dir=args.output_dir
            )
            print("Done!")
        
    except Exception as e:
        print(f"Error during quantization search: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run CLI
    exit(quantization_search_cli())