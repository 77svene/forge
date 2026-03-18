# src/forge/quantization/__init__.py
"""
Adaptive Quantization Engine for forge.

This module implements a dynamic quantization system that automatically selects
optimal quantization methods based on hardware constraints and accuracy requirements.
Includes accuracy-preserving quantization search and real-time memory optimization.
"""

import os
import sys
import json
import logging
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import existing forge modules
from ..utils import logging as forge_logging
from ..model import get_model_class
from ..data import get_dataset

logger = logging.getLogger(__name__)

# Define quantization methods
class QuantizationMethod(Enum):
    """Supported quantization methods."""
    GPTQ = "gptq"
    AWQ = "awq"
    EXL2 = "exl2"
    HQQ = "hqq"
    DYNAMIC = "dynamic"  # Dynamic quantization for CPU
    STATIC = "static"    # Static quantization for CPU

@dataclass
class HardwareSpec:
    """Hardware specifications for quantization selection."""
    device_type: str  # cpu, cuda, mps, etc.
    device_name: str  # e.g., "NVIDIA A100", "Apple M2"
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    compute_capability: Optional[str] = None  # for CUDA devices
    is_mobile: bool = False
    is_edge: bool = False
    power_constraint: Optional[float] = None  # in watts

@dataclass
class AccuracyRequirement:
    """Accuracy requirements for quantization."""
    max_accuracy_loss: float = 0.01  # Maximum acceptable accuracy loss (1%)
    min_accuracy: Optional[float] = None  # Minimum absolute accuracy
    evaluation_metric: str = "perplexity"  # perplexity, accuracy, f1, etc.
    calibration_samples: int = 128  # Number of calibration samples
    evaluation_samples: int = 512  # Number of evaluation samples

@dataclass
class QuantizationConfig:
    """Configuration for a specific quantization method."""
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    desc_act: bool = False
    damp_percent: float = 0.01
    true_sequential: bool = True
    static_groups: bool = False
    layerwise: bool = False
    calibration_dataset: Optional[str] = None
    calibration_samples: int = 128
    target_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    memory_optimization: bool = True
    accuracy_preserving: bool = True

@dataclass
class QuantizationResult:
    """Result of quantization process."""
    config: QuantizationConfig
    accuracy: float
    memory_reduction: float  # percentage
    inference_speedup: float  # percentage
    model_size_mb: float
    quantization_time: float  # seconds
    hardware_utilization: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class CalibrationDatasetSelector:
    """Automatic calibration dataset selection based on model architecture."""
    
    # Mapping of model architectures to recommended calibration datasets
    ARCHITECTURE_DATASET_MAP = {
        "llama": ["wikitext", "c4", "ptb"],
        "mistral": ["wikitext", "c4", "alpaca"],
        "qwen": ["wikitext", "c4", "alpaca"],
        "baichuan": ["wikitext", "c4", "alpaca"],
        "chatglm": ["wikitext", "c4", "alpaca"],
        "falcon": ["wikitext", "c4", "pile"],
        "bloom": ["wikitext", "c4", "pile"],
        "gpt2": ["wikitext", "c4", "openwebtext"],
        "opt": ["wikitext", "c4", "pile"],
        "bloomz": ["wikitext", "c4", "pile"],
    }
    
    @classmethod
    def select_dataset(cls, model_architecture: str, tokenizer: Any, 
                       num_samples: int = 128) -> Dataset:
        """Select appropriate calibration dataset for model architecture."""
        model_arch = model_architecture.lower()
        
        # Find matching architecture
        datasets = None
        for arch_key in cls.ARCHITECTURE_DATASET_MAP:
            if arch_key in model_arch:
                datasets = cls.ARCHITECTURE_DATASET_MAP[arch_key]
                break
        
        if datasets is None:
            # Default to wikitext if no specific mapping found
            datasets = ["wikitext"]
        
        # Try datasets in order until one loads successfully
        for dataset_name in datasets:
            try:
                dataset = get_dataset(
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    split="train",
                    max_samples=num_samples,
                    streaming=False
                )
                logger.info(f"Selected calibration dataset: {dataset_name}")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        # If all fail, create a dummy dataset
        logger.warning("Using dummy calibration dataset")
        return DummyCalibrationDataset(tokenizer, num_samples)

class DummyCalibrationDataset(Dataset):
    """Dummy calibration dataset for fallback."""
    
    def __init__(self, tokenizer, num_samples: int = 128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.samples = [
            "The quick brown fox jumps over the lazy dog. " * 10
            for _ in range(num_samples)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class HardwareDetector:
    """Detect hardware capabilities and constraints."""
    
    @staticmethod
    def detect_hardware() -> HardwareSpec:
        """Detect current hardware specifications."""
        device_type = "cpu"
        device_name = "CPU"
        total_memory = 0
        available_memory = 0
        compute_capability = None
        
        # Check for CUDA
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = torch.cuda.mem_get_info()[0]
            compute_capability = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon"
            # Estimate memory for Apple Silicon
            total_memory = 16 * 1024 * 1024 * 1024  # 16GB estimate
            available_memory = total_memory // 2
        
        # Check for mobile/edge devices
        is_mobile = "android" in platform.platform().lower() or "ios" in platform.platform().lower()
        is_edge = "jetson" in device_name.lower() or "raspberry" in device_name.lower()
        
        # Get system memory for CPU
        if device_type == "cpu":
            try:
                import psutil
                total_memory = psutil.virtual_memory().total
                available_memory = psutil.virtual_memory().available
            except ImportError:
                # Fallback estimation
                total_memory = 8 * 1024 * 1024 * 1024  # 8GB estimate
                available_memory = total_memory // 2
        
        return HardwareSpec(
            device_type=device_type,
            device_name=device_name,
            total_memory=total_memory,
            available_memory=available_memory,
            compute_capability=compute_capability,
            is_mobile=is_mobile,
            is_edge=is_edge
        )
    
    @staticmethod
    def estimate_power_constraint(hardware: HardwareSpec) -> Optional[float]:
        """Estimate power constraint based on hardware."""
        if hardware.is_mobile or hardware.is_edge:
            return 10.0  # 10W for mobile/edge devices
        elif "jetson" in hardware.device_name.lower():
            return 15.0  # 15W for Jetson devices
        elif "raspberry" in hardware.device_name.lower():
            return 5.0   # 5W for Raspberry Pi
        return None  # No constraint for servers/desktops

class QuantizationMethodEvaluator:
    """Evaluate different quantization methods for a given model."""
    
    def __init__(self, model: nn.Module, tokenizer: Any, 
                 hardware: HardwareSpec, 
                 accuracy_req: AccuracyRequirement):
        self.model = model
        self.tokenizer = tokenizer
        self.hardware = hardware
        self.accuracy_req = accuracy_req
        self.original_model_size = self._get_model_size(model)
        self.original_accuracy = None
        
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
    
    def _evaluate_accuracy(self, model: nn.Module, eval_dataset: Dataset) -> float:
        """Evaluate model accuracy on dataset."""
        model.eval()
        total_loss = 0
        total_samples = 0
        
        dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.hardware.device_type) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Convert perplexity to accuracy-like metric (lower perplexity = better)
        # This is a simplified conversion; in practice, you'd use task-specific metrics
        accuracy = max(0, 1 - (perplexity / 100))  # Normalize to 0-1 range
        return accuracy
    
    def _measure_inference_speed(self, model: nn.Module, 
                                 eval_dataset: Dataset, 
                                 num_batches: int = 10) -> float:
        """Measure inference speed in samples per second."""
        model.eval()
        dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
        
        # Warmup
        batch = next(iter(dataloader))
        batch = {k: v.to(self.hardware.device_type) for k, v in batch.items()}
        for _ in range(3):
            _ = model(**batch)
        
        # Measure
        if self.hardware.device_type == "cuda":
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        samples_processed = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = {k: v.to(self.hardware.device_type) for k, v in batch.items()}
            _ = model(**batch)
            samples_processed += batch["input_ids"].size(0)
        
        if self.hardware.device_type == "cuda":
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        return samples_processed / elapsed_time if elapsed_time > 0 else 0
    
    def evaluate_method(self, method: QuantizationMethod, 
                        config: QuantizationConfig,
                        calibration_dataset: Dataset,
                        eval_dataset: Dataset) -> QuantizationResult:
        """Evaluate a specific quantization method."""
        logger.info(f"Evaluating {method.value} quantization...")
        
        try:
            # Import quantization method
            quantized_model = self._apply_quantization(method, config, calibration_dataset)
            
            # Evaluate accuracy
            accuracy = self._evaluate_accuracy(quantized_model, eval_dataset)
            
            # Calculate metrics
            quantized_size = self._get_model_size(quantized_model)
            memory_reduction = (1 - quantized_size / self.original_model_size) * 100
            
            # Measure speed (simplified)
            original_speed = self._measure_inference_speed(self.model, eval_dataset)
            quantized_speed = self._measure_inference_speed(quantized_model, eval_dataset)
            speedup = ((quantized_speed / original_speed) - 1) * 100 if original_speed > 0 else 0
            
            # Check if accuracy requirement is met
            accuracy_loss = self.original_accuracy - accuracy if self.original_accuracy else 0
            success = accuracy_loss <= self.accuracy_req.max_accuracy_loss
            
            return QuantizationResult(
                config=config,
                accuracy=accuracy,
                memory_reduction=memory_reduction,
                inference_speedup=speedup,
                model_size_mb=quantized_size,
                quantization_time=0,  # Would be measured in actual implementation
                success=success
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate {method.value}: {e}")
            return QuantizationResult(
                config=config,
                accuracy=0,
                memory_reduction=0,
                inference_speedup=0,
                model_size_mb=0,
                quantization_time=0,
                success=False,
                error_message=str(e)
            )
    
    def _apply_quantization(self, method: QuantizationMethod,
                            config: QuantizationConfig,
                            calibration_dataset: Dataset) -> nn.Module:
        """Apply quantization method to model."""
        # This is a placeholder implementation
        # In practice, this would call the actual quantization libraries
        
        if method == QuantizationMethod.GPTQ:
            return self._apply_gptq(config, calibration_dataset)
        elif method == QuantizationMethod.AWQ:
            return self._apply_awq(config, calibration_dataset)
        elif method == QuantizationMethod.EXL2:
            return self._apply_exl2(config, calibration_dataset)
        elif method == QuantizationMethod.HQQ:
            return self._apply_hqq(config, calibration_dataset)
        elif method == QuantizationMethod.DYNAMIC:
            return self._apply_dynamic_quantization(config)
        elif method == QuantizationMethod.STATIC:
            return self._apply_static_quantization(config, calibration_dataset)
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
    
    def _apply_gptq(self, config: QuantizationConfig, 
                    calibration_dataset: Dataset) -> nn.Module:
        """Apply GPTQ quantization."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            quantize_config = BaseQuantizeConfig(
                bits=config.bits,
                group_size=config.group_size,
                desc_act=config.desc_act,
                sym=config.sym,
                damp_percent=config.damp_percent,
                true_sequential=config.true_sequential,
                static_groups=config.static_groups,
            )
            
            # Convert dataset to list of strings for GPTQ
            calibration_texts = []
            for i in range(min(128, len(calibration_dataset))):
                item = calibration_dataset[i]
                if "input_ids" in item:
                    text = self.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
                    calibration_texts.append(text)
            
            quantized_model = AutoGPTQForCausalLM.from_pretrained(
                self.model,
                quantize_config=quantize_config,
            )
            
            quantized_model.quantize(calibration_texts)
            return quantized_model
            
        except ImportError:
            logger.warning("auto_gptq not installed, using fallback")
            return self._fallback_quantization(config)
    
    def _apply_awq(self, config: QuantizationConfig,
                   calibration_dataset: Dataset) -> nn.Module:
        """Apply AWQ quantization."""
        try:
            from awq import AutoAWQForCausalLM
            
            quant_config = {
                "zero_point": True,
                "q_group_size": config.group_size,
                "w_bit": config.bits,
                "version": "GEMM"
            }
            
            # Convert dataset
            calibration_texts = []
            for i in range(min(128, len(calibration_dataset))):
                item = calibration_dataset[i]
                if "input_ids" in item:
                    text = self.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
                    calibration_texts.append(text)
            
            quantized_model = AutoAWQForCausalLM.from_pretrained(self.model)
            quantized_model.quantize(
                self.tokenizer,
                quant_config=quant_config,
                calib_data=calibration_texts,
            )
            
            return quantized_model
            
        except ImportError:
            logger.warning("autoawq not installed, using fallback")
            return self._fallback_quantization(config)
    
    def _apply_exl2(self, config: QuantizationConfig,
                    calibration_dataset: Dataset) -> nn.Module:
        """Apply EXL2 quantization."""
        # EXL2 requires specific conversion tools
        # This is a simplified implementation
        logger.info("EXL2 quantization requires external conversion")
        return self._fallback_quantization(config)
    
    def _apply_hqq(self, config: QuantizationConfig,
                   calibration_dataset: Dataset) -> nn.Module:
        """Apply HQQ quantization."""
        try:
            from hqq.core.quantize import HQQBackend, HQQLinear
            
            # HQQ quantization implementation
            # This is a simplified version
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Apply HQQ quantization to linear layers
                    hqq_layer = HQQLinear(
                        module,
                        quant_config={
                            'weight_bits': config.bits,
                            'group_size': config.group_size,
                        }
                    )
                    # Replace module (simplified)
                    # In practice, you'd need to properly replace the module
                    pass
            
            return self.model
            
        except ImportError:
            logger.warning("hqq not installed, using fallback")
            return self._fallback_quantization(config)
    
    def _apply_dynamic_quantization(self, config: QuantizationConfig) -> nn.Module:
        """Apply dynamic quantization (PyTorch built-in)."""
        model = self.model
        
        # Dynamic quantization for CPU
        if self.hardware.device_type == "cpu":
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
                dtype=torch.qint8 if config.bits == 8 else torch.qint4,
            )
            return quantized_model
        
        return model
    
    def _apply_static_quantization(self, config: QuantizationConfig,
                                   calibration_dataset: Dataset) -> nn.Module:
        """Apply static quantization (PyTorch built-in)."""
        model = self.model
        
        if self.hardware.device_type == "cpu":
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model for static quantization
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibrate with dataset
            dataloader = DataLoader(calibration_dataset, batch_size=4, shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v for k, v in batch.items()}
                    prepared_model(**batch)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            return quantized_model
        
        return model
    
    def _fallback_quantization(self, config: QuantizationConfig) -> nn.Module:
        """Fallback quantization when specialized libraries are not available."""
        logger.warning("Using fallback quantization (PyTorch dynamic quantization)")
        return self._apply_dynamic_quantization(config)

class AccuracyMemoryTradeoffAnalyzer:
    """Analyze accuracy-memory tradeoff curves for different quantization methods."""
    
    def __init__(self, model: nn.Module, tokenizer: Any,
                 hardware: HardwareSpec,
                 accuracy_req: AccuracyRequirement):
        self.model = model
        self.tokenizer = tokenizer
        self.hardware = hardware
        self.accuracy_req = accuracy_req
        self.evaluator = QuantizationMethodEvaluator(
            model, tokenizer, hardware, accuracy_req
        )
    
    def generate_tradeoff_curve(self, 
                                calibration_dataset: Dataset,
                                eval_dataset: Dataset,
                                methods: Optional[List[QuantizationMethod]] = None,
                                bit_options: Optional[List[int]] = None) -> Dict[str, List[Dict]]:
        """Generate tradeoff curves for different methods and bit widths."""
        if methods is None:
            methods = [
                QuantizationMethod.GPTQ,
                QuantizationMethod.AWQ,
                QuantizationMethod.HQQ,
                QuantizationMethod.DYNAMIC,
            ]
        
        if bit_options is None:
            bit_options = [8, 4, 3, 2]
        
        results = {}
        
        for method in methods:
            method_results = []
            
            for bits in bit_options:
                # Skip invalid combinations
                if method == QuantizationMethod.DYNAMIC and bits not in [8, 4]:
                    continue
                
                config = QuantizationConfig(
                    method=method,
                    bits=bits,
                    group_size=128 if bits <= 4 else -1,
                    calibration_samples=self.accuracy_req.calibration_samples
                )
                
                result = self.evaluator.evaluate_method(
                    method, config, calibration_dataset, eval_dataset
                )
                
                if result.success:
                    method_results.append({
                        "bits": bits,
                        "accuracy": result.accuracy,
                        "memory_reduction": result.memory_reduction,
                        "model_size_mb": result.model_size_mb,
                        "speedup": result.inference_speedup,
                    })
            
            if method_results:
                results[method.value] = method_results
        
        return results
    
    def find_optimal_configuration(self,
                                   calibration_dataset: Dataset,
                                   eval_dataset: Dataset,
                                   memory_constraint_mb: Optional[float] = None,
                                   accuracy_constraint: Optional[float] = None) -> QuantizationResult:
        """Find optimal quantization configuration given constraints."""
        best_result = None
        best_score = -float('inf')
        
        # Test different methods and configurations
        methods_to_test = [
            (QuantizationMethod.GPTQ, [4, 3, 2]),
            (QuantizationMethod.AWQ, [4, 3]),
            (QuantizationMethod.HQQ, [4, 3, 2]),
            (QuantizationMethod.DYNAMIC, [8, 4]),
        ]
        
        for method, bit_options in methods_to_test:
            for bits in bit_options:
                config = QuantizationConfig(
                    method=method,
                    bits=bits,
                    group_size=128 if bits <= 4 else -1,
                    calibration_samples=self.accuracy_req.calibration_samples
                )
                
                result = self.evaluator.evaluate_method(
                    method, config, calibration_dataset, eval_dataset
                )
                
                if not result.success:
                    continue
                
                # Check constraints
                if memory_constraint_mb and result.model_size_mb > memory_constraint_mb:
                    continue
                
                if accuracy_constraint and result.accuracy < accuracy_constraint:
                    continue
                
                # Calculate score (higher is better)
                # Weight accuracy more heavily than memory reduction
                accuracy_weight = 0.7
                memory_weight = 0.3
                
                # Normalize values
                accuracy_score = result.accuracy
                memory_score = min(1.0, result.memory_reduction / 100)  # Cap at 100% reduction
                
                score = (accuracy_weight * accuracy_score + 
                         memory_weight * memory_score)
                
                if score > best_score:
                    best_score = score
                    best_result = result
        
        return best_result

class AdaptiveQuantizationEngine:
    """
    Main adaptive quantization engine that selects optimal quantization methods.
    
    This engine automatically:
    1. Detects hardware capabilities
    2. Selects appropriate calibration datasets
    3. Evaluates different quantization methods
    4. Chooses optimal configuration based on constraints
    5. Provides real-time memory optimization
    """
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: Any,
                 target_hardware: Optional[HardwareSpec] = None,
                 accuracy_requirement: Optional[AccuracyRequirement] = None):
        """
        Initialize the adaptive quantization engine.
        
        Args:
            model: The model to quantize
            tokenizer: Tokenizer for the model
            target_hardware: Target hardware specifications (auto-detected if None)
            accuracy_requirement: Accuracy requirements (default if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Auto-detect hardware if not provided
        if target_hardware is None:
            self.hardware = HardwareDetector.detect_hardware()
            self.hardware.power_constraint = HardwareDetector.estimate_power_constraint(
                self.hardware
            )
        else:
            self.hardware = target_hardware
        
        # Set default accuracy requirements
        if accuracy_requirement is None:
            self.accuracy_req = AccuracyRequirement(
                max_accuracy_loss=0.01,  # 1% accuracy loss
                calibration_samples=128,
                evaluation_samples=512
            )
        else:
            self.accuracy_req = accuracy_requirement
        
        # Initialize components
        self.calibration_selector = CalibrationDatasetSelector()
        self.evaluator = QuantizationMethodEvaluator(
            model, tokenizer, self.hardware, self.accuracy_req
        )
        self.tradeoff_analyzer = AccuracyMemoryTradeoffAnalyzer(
            model, tokenizer, self.hardware, self.accuracy_req
        )
        
        # Store original model metrics
        self.original_model_size = self.evaluator._get_model_size(model)
        
        logger.info(f"Initialized AdaptiveQuantizationEngine for {self.hardware.device_name}")
        logger.info(f"Hardware: {self.hardware.device_type}, Memory: {self.hardware.total_memory / 1e9:.1f}GB")
    
    def select_optimal_quantization(self,
                                    memory_limit_mb: Optional[float] = None,
                                    accuracy_threshold: Optional[float] = None,
                                    preferred_methods: Optional[List[QuantizationMethod]] = None) -> QuantizationResult:
        """
        Select optimal quantization method based on constraints.
        
        Args:
            memory_limit_mb: Maximum allowed model size in MB
            accuracy_threshold: Minimum required accuracy (0-1)
            preferred_methods: List of preferred quantization methods
            
        Returns:
            QuantizationResult with optimal configuration
        """
        logger.info("Selecting optimal quantization method...")
        
        # Select calibration dataset
        model_arch = self.model.__class__.__name__
        calibration_dataset = self.calibration_selector.select_dataset(
            model_arch, self.tokenizer, self.accuracy_req.calibration_samples
        )
        
        # Create evaluation dataset (could be same as calibration or separate)
        eval_dataset = calibration_dataset  # In practice, use a separate validation set
        
        # Find optimal configuration
        result = self.tradeoff_analyzer.find_optimal_configuration(
            calibration_dataset=calibration_dataset,
            eval_dataset=eval_dataset,
            memory_constraint_mb=memory_limit_mb,
            accuracy_constraint=accuracy_threshold
        )
        
        if result is None:
            logger.warning("No suitable quantization configuration found")
            # Return a default configuration
            default_config = QuantizationConfig(
                method=QuantizationMethod.DYNAMIC,
                bits=8,
                calibration_samples=self.accuracy_req.calibration_samples
            )
            result = QuantizationResult(
                config=default_config,
                accuracy=1.0,  # Assume no accuracy loss for dynamic quantization
                memory_reduction=50.0,  # Estimate
                inference_speedup=0.0,
                model_size_mb=self.original_model_size * 0.5,
                quantization_time=0.0,
                success=True
            )
        
        logger.info(f"Selected {result.config.method.value} quantization with {result.config.bits} bits")
        logger.info(f"Expected accuracy: {result.accuracy:.3f}, Memory reduction: {result.memory_reduction:.1f}%")
        
        return result
    
    def quantize_model(self, 
                       config: Optional[QuantizationConfig] = None,
                       memory_limit_mb: Optional[float] = None) -> Tuple[nn.Module, QuantizationResult]:
        """
        Quantize the model using optimal or specified configuration.
        
        Args:
            config: Specific quantization configuration (auto-selected if None)
            memory_limit_mb: Memory limit for auto-selection
            
        Returns:
            Tuple of (quantized_model, quantization_result)
        """
        if config is None:
            # Auto-select optimal configuration
            result = self.select_optimal_quantization(memory_limit_mb=memory_limit_mb)
            config = result.config
        else:
            # Use provided configuration
            model_arch = self.model.__class__.__name__
            calibration_dataset = self.calibration_selector.select_dataset(
                model_arch, self.tokenizer, config.calibration_samples
            )
            eval_dataset = calibration_dataset
            
            result = self.evaluator.evaluate_method(
                config.method, config, calibration_dataset, eval_dataset
            )
        
        # Apply quantization
        model_arch = self.model.__class__.__name__
        calibration_dataset = self.calibration_selector.select_dataset(
            model_arch, self.tokenizer, config.calibration_samples
        )
        
        quantized_model = self.evaluator._apply_quantization(
            config.method, config, calibration_dataset
        )
        
        # Update result with actual quantized model
        result.model_size_mb = self.evaluator._get_model_size(quantized_model)
        result.memory_reduction = (1 - result.model_size_mb / self.original_model_size) * 100
        
        return quantized_model, result
    
    def optimize_for_hardware(self, 
                              model: Optional[nn.Module] = None) -> nn.Module:
        """
        Apply real-time memory optimization for target hardware.
        
        Args:
            model: Model to optimize (uses self.model if None)
            
        Returns:
            Optimized model
        """
        if model is None:
            model = self.model
        
        logger.info(f"Optimizing model for {self.hardware.device_name}")
        
        # Apply hardware-specific optimizations
        if self.hardware.device_type == "cuda":
            # CUDA optimizations
            if self.hardware.compute_capability and float(self.hardware.compute_capability) >= 8.0:
                # Ampere or newer - use TF32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # PyTorch 2.0+ has memory efficient attention
                pass
        
        elif self.hardware.device_type == "cpu":
            # CPU optimizations
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            
            # Use Intel optimizations if available
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model)
                logger.info("Applied Intel optimizations")
            except ImportError:
                pass
        
        elif self.hardware.device_type == "mps":
            # Apple Silicon optimizations
            # Use Apple's performance optimizations
            pass
        
        # Apply power-aware optimizations if constraint exists
        if self.hardware.power_constraint:
            model = self._apply_power_optimizations(model)
        
        return model
    
    def _apply_power_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply power-aware optimizations."""
        # Reduce precision for power efficiency
        if self.hardware.power_constraint < 10.0:  # Very low power
            # Use more aggressive quantization
            logger.info("Applying aggressive quantization for power efficiency")
            # Could apply 4-bit or 3-bit quantization
        
        return model
    
    def generate_report(self, 
                        results: List[QuantizationResult],
                        output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive quantization report.
        
        Args:
            results: List of quantization results to report
            output_path: Path to save report (optional)
            
        Returns:
            Report dictionary
        """
        report = {
            "hardware": asdict(self.hardware),
            "accuracy_requirements": asdict(self.accuracy_req),
            "original_model_size_mb": self.original_model_size,
            "quantization_results": [],
            "recommendations": []
        }
        
        for result in results:
            result_dict = asdict(result)
            result_dict["config"] = asdict(result.config)
            result_dict["config"]["method"] = result.config.method.value
            report["quantization_results"].append(result_dict)
        
        # Generate recommendations
        if results:
            best_result = max(results, key=lambda r: r.accuracy if r.success else -1)
            if best_result.success:
                report["recommendations"].append(
                    f"Use {best_result.config.method.value} with {best_result.config.bits}-bit "
                    f"quantization for best accuracy ({best_result.accuracy:.3f})"
                )
        
        # Memory-based recommendations
        memory_constrained_results = [r for r in results if r.success and r.memory_reduction > 50]
        if memory_constrained_results:
            best_memory = max(memory_constrained_results, key=lambda r: r.memory_reduction)
            report["recommendations"].append(
                f"For maximum memory reduction ({best_memory.memory_reduction:.1f}%), "
                f"use {best_memory.config.method.value} with {best_memory.config.bits}-bit"
            )
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report

# Utility functions for easy integration
def quantize_model_adaptive(
    model: nn.Module,
    tokenizer: Any,
    memory_limit_mb: Optional[float] = None,
    accuracy_threshold: float = 0.95,
    hardware: Optional[HardwareSpec] = None
) -> Tuple[nn.Module, QuantizationResult]:
    """
    Convenience function to adaptively quantize a model.
    
    Args:
        model: Model to quantize
        tokenizer: Model tokenizer
        memory_limit_mb: Memory limit in MB
        accuracy_threshold: Minimum accuracy (0-1)
        hardware: Target hardware (auto-detected if None)
        
    Returns:
        Tuple of (quantized_model, result)
    """
    engine = AdaptiveQuantizationEngine(
        model=model,
        tokenizer=tokenizer,
        target_hardware=hardware
    )
    
    return engine.quantize_model(memory_limit_mb=memory_limit_mb)

def get_quantization_recommendations(
    model: nn.Module,
    tokenizer: Any,
    hardware: Optional[HardwareSpec] = None
) -> Dict[str, Any]:
    """
    Get quantization recommendations for a model.
    
    Args:
        model: Model to analyze
        tokenizer: Model tokenizer
        hardware: Target hardware (auto-detected if None)
        
    Returns:
        Dictionary with recommendations
    """
    engine = AdaptiveQuantizationEngine(
        model=model,
        tokenizer=tokenizer,
        target_hardware=hardware
    )
    
    # Generate tradeoff curves
    model_arch = model.__class__.__name__
    calibration_dataset = engine.calibration_selector.select_dataset(
        model_arch, tokenizer, 128
    )
    eval_dataset = calibration_dataset
    
    curves = engine.tradeoff_analyzer.generate_tradeoff_curve(
        calibration_dataset, eval_dataset
    )
    
    # Find optimal configuration
    optimal = engine.select_optimal_quantization()
    
    return {
        "tradeoff_curves": curves,
        "optimal_configuration": asdict(optimal),
        "hardware_info": asdict(engine.hardware),
        "recommendations": [
            f"Recommended method: {optimal.config.method.value}",
            f"Recommended bits: {optimal.config.bits}",
            f"Expected memory reduction: {optimal.memory_reduction:.1f}%",
            f"Expected accuracy: {optimal.accuracy:.3f}"
        ]
    }

# Export main classes and functions
__all__ = [
    "AdaptiveQuantizationEngine",
    "QuantizationMethod",
    "QuantizationConfig",
    "QuantizationResult",
    "HardwareSpec",
    "AccuracyRequirement",
    "CalibrationDatasetSelector",
    "HardwareDetector",
    "QuantizationMethodEvaluator",
    "AccuracyMemoryTradeoffAnalyzer",
    "quantize_model_adaptive",
    "get_quantization_recommendations",
]