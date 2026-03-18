"""
Adaptive Quantization Engine for forge
Dynamic quantization system that automatically selects optimal quantization methods
based on hardware constraints and accuracy requirements.
"""

import os
import time
import json
import logging
import platform
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from forge.hparams import QuantizationArguments, ModelArguments, DataArguments
from forge.model.loader import load_model_and_tokenizer
from forge.data.loader import load_dataset
from forge.extras.logging import get_logger

logger = get_logger(__name__)


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    GPTQ = "gptq"
    AWQ = "awq"
    EXL2 = "exl2"
    HQQ = "hqq"
    DYNAMIC = "dynamic"
    AUTO = "auto"


@dataclass
class HardwareConstraints:
    """Hardware constraints for quantization selection."""
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    has_cuda: bool = False
    has_mps: bool = False
    has_cpu_only: bool = True
    target_device: str = "auto"
    max_model_memory_gb: Optional[float] = None
    min_inference_speed: Optional[float] = None  # tokens/sec


@dataclass
class AccuracyRequirements:
    """Accuracy requirements for quantization."""
    max_accuracy_drop: float = 0.05  # 5% maximum drop
    min_perplexity: Optional[float] = None
    min_task_score: Optional[float] = None
    evaluation_dataset: Optional[str] = None
    evaluation_samples: int = 1000


@dataclass
class QuantizationResult:
    """Result of quantization process."""
    method: QuantizationMethod
    config: Dict[str, Any]
    model_size_mb: float
    inference_speed: float
    accuracy_metrics: Dict[str, float]
    memory_footprint_mb: float
    quantization_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class QuantizationStrategy:
    """Selected quantization strategy."""
    primary_method: QuantizationMethod
    fallback_methods: List[QuantizationMethod]
    config: Dict[str, Any]
    expected_memory_mb: float
    expected_accuracy_drop: float
    confidence_score: float


class HardwareDetector:
    """Detects and analyzes hardware capabilities."""
    
    @staticmethod
    def detect_hardware() -> HardwareConstraints:
        """Detect current hardware constraints."""
        constraints = HardwareConstraints()
        
        # Detect CUDA
        constraints.has_cuda = torch.cuda.is_available()
        if constraints.has_cuda:
            constraints.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            constraints.target_device = "cuda"
        
        # Detect MPS (Apple Silicon)
        constraints.has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if constraints.has_mps and not constraints.has_cuda:
            constraints.target_device = "mps"
        
        # CPU detection
        constraints.has_cpu_only = not (constraints.has_cuda or constraints.has_mps)
        if constraints.has_cpu_only:
            constraints.target_device = "cpu"
        
        constraints.cpu_cores = os.cpu_count() or 1
        
        # Platform detection
        system_info = platform.uname()
        logger.info(f"Detected hardware: {system_info.system} {system_info.machine}")
        logger.info(f"CUDA: {constraints.has_cuda}, MPS: {constraints.has_mps}, CPU cores: {constraints.cpu_cores}")
        
        return constraints
    
    @staticmethod
    def estimate_model_memory(model: PreTrainedModel, method: QuantizationMethod) -> float:
        """Estimate model memory footprint after quantization in MB."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Bits per parameter for different methods
        bits_map = {
            QuantizationMethod.GPTQ: 4,  # Typically 4-bit
            QuantizationMethod.AWQ: 4,
            QuantizationMethod.EXL2: 4,
            QuantizationMethod.HQQ: 4,
            QuantizationMethod.DYNAMIC: 8,  # Dynamic usually 8-bit
            QuantizationMethod.AUTO: 4,
        }
        
        bits_per_param = bits_map.get(method, 4)
        memory_bytes = param_count * (bits_per_param / 8)
        
        # Add overhead for quantization metadata (10-20%)
        overhead_factor = 1.15
        
        return (memory_bytes * overhead_factor) / (1024**2)  # Convert to MB


class QuantizationMethodWrapper:
    """Wrapper for different quantization methods."""
    
    def __init__(self, method: QuantizationMethod):
        self.method = method
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if the quantization method is available."""
        try:
            if self.method == QuantizationMethod.GPTQ:
                import auto_gptq
                return True
            elif self.method == QuantizationMethod.AWQ:
                import awq
                return True
            elif self.method == QuantizationMethod.EXL2:
                import exllamav2
                return True
            elif self.method == QuantizationMethod.HQQ:
                import hqq
                return True
            elif self.method == QuantizationMethod.DYNAMIC:
                # Dynamic quantization is built into PyTorch
                return True
            return False
        except ImportError:
            return False
    
    def quantize(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 calibration_data: Optional[Dataset] = None,
                 config: Optional[Dict] = None) -> Tuple[PreTrainedModel, Dict]:
        """Quantize model using the specific method."""
        config = config or {}
        
        if self.method == QuantizationMethod.GPTQ:
            return self._quantize_gptq(model, tokenizer, calibration_data, config)
        elif self.method == QuantizationMethod.AWQ:
            return self._quantize_awq(model, tokenizer, calibration_data, config)
        elif self.method == QuantizationMethod.EXL2:
            return self._quantize_exl2(model, tokenizer, calibration_data, config)
        elif self.method == QuantizationMethod.HQQ:
            return self._quantize_hqq(model, tokenizer, calibration_data, config)
        elif self.method == QuantizationMethod.DYNAMIC:
            return self._quantize_dynamic(model, config)
        else:
            raise ValueError(f"Unsupported quantization method: {self.method}")
    
    def _quantize_gptq(self, model, tokenizer, calibration_data, config):
        """GPTQ quantization implementation."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            quantize_config = BaseQuantizeConfig(
                bits=config.get("bits", 4),
                group_size=config.get("group_size", 128),
                desc_act=config.get("desc_act", False),
            )
            
            # Prepare calibration dataset
            if calibration_data is None:
                calibration_data = self._create_default_calibration_dataset(tokenizer)
            
            # Convert dataset to GPTQ format
            calibration_dataset = []
            for item in calibration_data:
                if "text" in item:
                    calibration_dataset.append(tokenizer(item["text"]))
            
            # Perform quantization
            quantized_model = AutoGPTQForCausalLM.from_pretrained(
                model, quantize_config=quantize_config
            )
            quantized_model.quantize(calibration_dataset)
            
            return quantized_model.model, {"method": "gptq", "config": quantize_config.to_dict()}
            
        except Exception as e:
            logger.error(f"GPTQ quantization failed: {e}")
            raise
    
    def _quantize_awq(self, model, tokenizer, calibration_data, config):
        """AWQ quantization implementation."""
        try:
            from awq import AutoAWQForCausalLM
            
            quant_config = {
                "zero_point": True,
                "q_group_size": config.get("group_size", 128),
                "w_bit": config.get("bits", 4),
                "version": "GEMM"
            }
            
            # Prepare calibration dataset
            if calibration_data is None:
                calibration_data = self._create_default_calibration_dataset(tokenizer)
            
            # Convert dataset
            calibration_texts = [item.get("text", "") for item in calibration_data if "text" in item]
            
            # Perform quantization
            awq_model = AutoAWQForCausalLM.from_pretrained(model)
            awq_model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)
            
            return awq_model.model, {"method": "awq", "config": quant_config}
            
        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            raise
    
    def _quantize_exl2(self, model, tokenizer, calibration_data, config):
        """EXL2 quantization implementation."""
        try:
            from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
            from exllamav2.generator import ExLlamaV2DynamicGenerator
            
            # EXL2 requires specific conversion process
            # This is a simplified version
            logger.warning("EXL2 quantization requires manual conversion steps")
            
            # Return original model with note
            return model, {"method": "exl2", "note": "Requires manual conversion"}
            
        except Exception as e:
            logger.error(f"EXL2 quantization failed: {e}")
            raise
    
    def _quantize_hqq(self, model, tokenizer, calibration_data, config):
        """HQQ quantization implementation."""
        try:
            from hqq.core.quantize import HQQBackend, HQQLinear
            from hqq.models.hqq import HQQModelForCausalLM
            
            quant_config = {
                "nbits": config.get("bits", 4),
                "group_size": config.get("group_size", 64),
                "quant_zero": True,
                "quant_scale": True,
            }
            
            # Perform quantization
            hqq_model = HQQModelForCausalLM(model, quant_config=quant_config)
            
            return hqq_model, {"method": "hqq", "config": quant_config}
            
        except Exception as e:
            logger.error(f"HQQ quantization failed: {e}")
            raise
    
    def _quantize_dynamic(self, model, config):
        """Dynamic quantization implementation."""
        try:
            # Dynamic quantization for CPU
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8 if config.get("bits", 8) == 8 else torch.qint4
            )
            
            return quantized_model, {"method": "dynamic", "config": config}
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            raise
    
    def _create_default_calibration_dataset(self, tokenizer, num_samples=128) -> Dataset:
        """Create a default calibration dataset."""
        # Use a mix of different text sources for calibration
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Quantization reduces model size while maintaining accuracy.",
            "Large language models require significant computational resources.",
            "Optimization techniques improve inference speed and efficiency.",
        ] * (num_samples // 5 + 1)
        
        texts = texts[:num_samples]
        
        return Dataset.from_dict({"text": texts})


class AccuracyEvaluator:
    """Evaluates model accuracy after quantization."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 eval_dataset: Optional[Dataset] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = next(model.parameters()).device
    
    def evaluate_perplexity(self, num_samples: int = 100) -> float:
        """Evaluate model perplexity on evaluation dataset."""
        if self.eval_dataset is None:
            self.eval_dataset = self._create_default_eval_dataset()
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i, item in enumerate(self.eval_dataset):
                if i >= num_samples:
                    break
                
                text = item.get("text", "")
                if not text:
                    continue
                
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_memory_footprint(self) -> float:
        """Evaluate model memory footprint in MB."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
        return total_size
    
    def evaluate_inference_speed(self, 
                                 sequence_length: int = 128,
                                 num_iterations: int = 10) -> float:
        """Evaluate inference speed in tokens/sec."""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, sequence_length)).to(self.device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                self.model(dummy_input)
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        tokens_per_second = (sequence_length * num_iterations) / total_time
        
        return tokens_per_second
    
    def _create_default_eval_dataset(self) -> Dataset:
        """Create a default evaluation dataset."""
        texts = [
            "The transformer architecture revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Pre-training on large corpora enables models to learn general language patterns.",
            "Fine-tuning adapts pre-trained models to specific downstream tasks.",
            "Quantization reduces model size while preserving most of the accuracy.",
        ]
        
        return Dataset.from_dict({"text": texts})


class AdaptiveQuantizationEngine:
    """Main adaptive quantization engine."""
    
    def __init__(self,
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 quant_args: QuantizationArguments,
                 hardware_constraints: Optional[HardwareConstraints] = None,
                 accuracy_requirements: Optional[AccuracyRequirements] = None):
        
        self.model_args = model_args
        self.data_args = data_args
        self.quant_args = quant_args
        
        # Auto-detect hardware if not provided
        self.hardware_constraints = hardware_constraints or HardwareDetector.detect_hardware()
        self.accuracy_requirements = accuracy_requirements or AccuracyRequirements()
        
        # Initialize quantization method wrappers
        self.quantization_methods = {
            method: QuantizationMethodWrapper(method) 
            for method in QuantizationMethod 
            if method != QuantizationMethod.AUTO
        }
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(model_args)
        self.model = self.model.to(self.hardware_constraints.target_device)
        
        # Load calibration dataset
        self.calibration_dataset = self._load_calibration_dataset()
        
        # Results storage
        self.quantization_results: Dict[QuantizationMethod, QuantizationResult] = {}
        self.selected_strategy: Optional[QuantizationStrategy] = None
    
    def _load_calibration_dataset(self) -> Optional[Dataset]:
        """Load calibration dataset for quantization."""
        try:
            if self.data_args.dataset:
                dataset = load_dataset(self.data_args, self.model_args, self.tokenizer)
                # Use a subset for calibration
                if len(dataset) > self.quant_args.calibration_samples:
                    indices = np.random.choice(len(dataset), 
                                              self.quant_args.calibration_samples, 
                                              replace=False)
                    calibration_data = dataset.select(indices)
                else:
                    calibration_data = dataset
                return calibration_data
        except Exception as e:
            logger.warning(f"Failed to load calibration dataset: {e}")
        
        return None
    
    def _select_candidate_methods(self) -> List[QuantizationMethod]:
        """Select candidate quantization methods based on hardware and model."""
        candidates = []
        
        # Filter methods based on hardware
        for method, wrapper in self.quantization_methods.items():
            if not wrapper.available:
                logger.info(f"Method {method.value} not available, skipping")
                continue
            
            # Check hardware compatibility
            if method == QuantizationMethod.DYNAMIC and not self.hardware_constraints.has_cpu_only:
                # Dynamic quantization is mainly for CPU
                if self.hardware_constraints.has_cuda:
                    logger.info(f"Skipping dynamic quantization for CUDA device")
                    continue
            
            if method == QuantizationMethod.EXL2 and not self.hardware_constraints.has_cuda:
                # EXL2 requires CUDA
                logger.info(f"Skipping EXL2 for non-CUDA device")
                continue
            
            candidates.append(method)
        
        # If no specific method requested, try all available
        if self.quant_args.quantization_method == QuantizationMethod.AUTO:
            return candidates
        else:
            # Use requested method if available
            requested = self.quant_args.quantization_method
            if requested in candidates:
                return [requested]
            else:
                logger.warning(f"Requested method {requested} not available, using auto selection")
                return candidates
    
    def _evaluate_method(self, 
                        method: QuantizationMethod,
                        config: Dict[str, Any]) -> QuantizationResult:
        """Evaluate a specific quantization method."""
        logger.info(f"Evaluating quantization method: {method.value}")
        
        try:
            start_time = time.time()
            
            # Get quantization wrapper
            wrapper = self.quantization_methods[method]
            
            # Quantize model
            quantized_model, quant_config = wrapper.quantize(
                self.model, 
                self.tokenizer, 
                self.calibration_dataset,
                config
            )
            
            # Create evaluator
            evaluator = AccuracyEvaluator(quantized_model, self.tokenizer, self.calibration_dataset)
            
            # Evaluate metrics
            perplexity = evaluator.evaluate_perplexity(
                num_samples=min(100, self.accuracy_requirements.evaluation_samples)
            )
            memory_footprint = evaluator.evaluate_memory_footprint()
            inference_speed = evaluator.evaluate_inference_speed()
            
            # Estimate model size
            model_size = HardwareDetector.estimate_model_memory(quantized_model, method)
            
            quantization_time = time.time() - start_time
            
            # Calculate accuracy metrics
            accuracy_metrics = {
                "perplexity": perplexity,
                "memory_footprint_mb": memory_footprint,
                "inference_speed_tokens_per_sec": inference_speed
            }
            
            result = QuantizationResult(
                method=method,
                config=quant_config,
                model_size_mb=model_size,
                inference_speed=inference_speed,
                accuracy_metrics=accuracy_metrics,
                memory_footprint_mb=memory_footprint,
                quantization_time=quantization_time,
                success=True
            )
            
            logger.info(f"Method {method.value} evaluation completed: "
                       f"perplexity={perplexity:.2f}, "
                       f"memory={memory_footprint:.2f}MB, "
                       f"speed={inference_speed:.2f} tokens/sec")
            
            return result
            
        except Exception as e:
            logger.error(f"Method {method.value} evaluation failed: {e}")
            return QuantizationResult(
                method=method,
                config={},
                model_size_mb=0.0,
                inference_speed=0.0,
                accuracy_metrics={},
                memory_footprint_mb=0.0,
                quantization_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _select_optimal_strategy(self) -> QuantizationStrategy:
        """Select optimal quantization strategy based on evaluation results."""
        successful_results = {
            method: result for method, result in self.quantization_results.items()
            if result.success
        }
        
        if not successful_results:
            # Fallback to dynamic quantization
            logger.warning("No successful quantization methods, using dynamic quantization")
            return QuantizationStrategy(
                primary_method=QuantizationMethod.DYNAMIC,
                fallback_methods=[],
                config={"bits": 8},
                expected_memory_mb=HardwareDetector.estimate_model_memory(
                    self.model, QuantizationMethod.DYNAMIC
                ),
                expected_accuracy_drop=0.1,
                confidence_score=0.5
            )
        
        # Score each method based on constraints
        scored_methods = []
        
        for method, result in successful_results.items():
            score = self._calculate_method_score(result)
            scored_methods.append((method, result, score))
        
        # Sort by score (higher is better)
        scored_methods.sort(key=lambda x: x[2], reverse=True)
        
        best_method, best_result, best_score = scored_methods[0]
        
        # Select fallback methods (top 2 alternatives)
        fallback_methods = [method for method, _, _ in scored_methods[1:3]]
        
        strategy = QuantizationStrategy(
            primary_method=best_method,
            fallback_methods=fallback_methods,
            config=best_result.config,
            expected_memory_mb=best_result.memory_footprint_mb,
            expected_accuracy_drop=self._estimate_accuracy_drop(best_result),
            confidence_score=best_score
        )
        
        return strategy
    
    def _calculate_method_score(self, result: QuantizationResult) -> float:
        """Calculate score for a quantization method."""
        score = 0.0
        
        # Memory efficiency (lower is better, so invert)
        if self.hardware_constraints.max_model_memory_gb:
            max_memory_mb = self.hardware_constraints.max_model_memory_gb * 1024
            memory_ratio = result.memory_footprint_mb / max_memory_mb
            if memory_ratio <= 1.0:
                score += (1.0 - memory_ratio) * 40  # Up to 40 points for memory
            else:
                score -= (memory_ratio - 1.0) * 100  # Penalty for exceeding
        
        # Accuracy preservation
        if "perplexity" in result.accuracy_metrics:
            perplexity = result.accuracy_metrics["perplexity"]
            # Lower perplexity is better
            perplexity_score = max(0, 100 - perplexity) / 100
            score += perplexity_score * 30  # Up to 30 points for accuracy
        
        # Inference speed
        if result.inference_speed > 0:
            speed_score = min(1.0, result.inference_speed / 100)  # Normalize to 100 tokens/sec
            score += speed_score * 20  # Up to 20 points for speed
        
        # Quantization time (faster is better)
        time_score = max(0, 1.0 - (result.quantization_time / 3600))  # Normalize to 1 hour
        score += time_score * 10  # Up to 10 points for speed
        
        return score
    
    def _estimate_accuracy_drop(self, result: QuantizationResult) -> float:
        """Estimate accuracy drop compared to original model."""
        # This is a simplified estimation
        # In practice, you would compare with original model's metrics
        
        if "perplexity" in result.accuracy_metrics:
            # Assume original perplexity around 10-20 for typical models
            original_perplexity = 15.0
            quantized_perplexity = result.accuracy_metrics["perplexity"]
            
            if original_perplexity > 0:
                accuracy_drop = (quantized_perplexity - original_perplexity) / original_perplexity
                return max(0.0, accuracy_drop)
        
        return 0.05  # Default 5% drop assumption
    
    def run_quantization_search(self) -> QuantizationStrategy:
        """Run the quantization search to find optimal method."""
        logger.info("Starting adaptive quantization search")
        
        # Select candidate methods
        candidates = self._select_candidate_methods()
        
        if not candidates:
            logger.error("No quantization methods available")
            raise RuntimeError("No quantization methods available")
        
        logger.info(f"Testing {len(candidates)} quantization methods: "
                   f"{[m.value for m in candidates]}")
        
        # Test each method with different configurations
        for method in candidates:
            # Generate configurations to test
            configs = self._generate_configurations(method)
            
            for config in configs:
                result = self._evaluate_method(method, config)
                self.quantization_results[method] = result
                
                # Early stopping if we find a good enough method
                if (result.success and 
                    self.hardware_constraints.max_model_memory_gb and
                    result.memory_footprint_mb <= self.hardware_constraints.max_model_memory_gb * 1024):
                    logger.info(f"Found suitable method {method.value} within memory constraints")
                    break
        
        # Select optimal strategy
        self.selected_strategy = self._select_optimal_strategy()
        
        logger.info(f"Selected quantization strategy: {self.selected_strategy.primary_method.value}")
        logger.info(f"Expected memory: {self.selected_strategy.expected_memory_mb:.2f} MB")
        logger.info(f"Expected accuracy drop: {self.selected_strategy.expected_accuracy_drop:.2%}")
        
        return self.selected_strategy
    
    def _generate_configurations(self, method: QuantizationMethod) -> List[Dict[str, Any]]:
        """Generate different configurations to test for a method."""
        base_configs = []
        
        if method == QuantizationMethod.GPTQ:
            base_configs = [
                {"bits": 4, "group_size": 128, "desc_act": False},
                {"bits": 4, "group_size": 64, "desc_act": False},
                {"bits": 8, "group_size": 128, "desc_act": False},
            ]
        elif method == QuantizationMethod.AWQ:
            base_configs = [
                {"bits": 4, "group_size": 128},
                {"bits": 4, "group_size": 64},
                {"bits": 8, "group_size": 128},
            ]
        elif method == QuantizationMethod.EXL2:
            base_configs = [
                {"bits": 4, "group_size": 128},
                {"bits": 4, "group_size": 64},
            ]
        elif method == QuantizationMethod.HQQ:
            base_configs = [
                {"bits": 4, "group_size": 64},
                {"bits": 8, "group_size": 64},
            ]
        elif method == QuantizationMethod.DYNAMIC:
            base_configs = [
                {"bits": 8},
                {"bits": 4},
            ]
        
        return base_configs
    
    def apply_quantization(self, 
                          strategy: Optional[QuantizationStrategy] = None) -> PreTrainedModel:
        """Apply the selected quantization strategy to the model."""
        if strategy is None:
            strategy = self.selected_strategy
        
        if strategy is None:
            raise ValueError("No quantization strategy selected. Run quantization search first.")
        
        logger.info(f"Applying quantization with method: {strategy.primary_method.value}")
        
        wrapper = self.quantization_methods[strategy.primary_method]
        quantized_model, _ = wrapper.quantize(
            self.model,
            self.tokenizer,
            self.calibration_dataset,
            strategy.config
        )
        
        # Update model in place
        self.model = quantized_model
        
        return quantized_model
    
    def save_quantized_model(self, 
                            output_dir: str,
                            strategy: Optional[QuantizationStrategy] = None):
        """Save the quantized model and configuration."""
        if strategy is None:
            strategy = self.selected_strategy
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "model"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save quantization configuration
        config_path = output_path / "quantization_config.json"
        config_data = {
            "method": strategy.primary_method.value,
            "config": strategy.config,
            "hardware_constraints": asdict(self.hardware_constraints),
            "accuracy_requirements": asdict(self.accuracy_requirements),
            "expected_memory_mb": strategy.expected_memory_mb,
            "expected_accuracy_drop": strategy.expected_accuracy_drop,
            "confidence_score": strategy.confidence_score
        }
        
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Save evaluation results
        results_path = output_path / "evaluation_results.json"
        results_data = {
            method.value: asdict(result) 
            for method, result in self.quantization_results.items()
        }
        
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Quantized model saved to {output_dir}")
    
    def get_quantization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quantization report."""
        report = {
            "hardware": asdict(self.hardware_constraints),
            "accuracy_requirements": asdict(self.accuracy_requirements),
            "model_info": {
                "name": self.model_args.model_name_or_path,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "original_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
            },
            "quantization_results": {
                method.value: asdict(result) 
                for method, result in self.quantization_results.items()
            },
            "selected_strategy": asdict(self.selected_strategy) if self.selected_strategy else None
        }
        
        return report


def create_adaptive_quantization_engine(
    model_args: ModelArguments,
    data_args: DataArguments,
    quant_args: QuantizationArguments,
    **kwargs
) -> AdaptiveQuantizationEngine:
    """Factory function to create adaptive quantization engine."""
    return AdaptiveQuantizationEngine(
        model_args=model_args,
        data_args=data_args,
        quant_args=quant_args,
        **kwargs
    )


# Integration with existing forge workflow
def quantize_model_adaptive(
    model_args: ModelArguments,
    data_args: DataArguments,
    quant_args: QuantizationArguments,
    output_dir: str,
    **kwargs
) -> Tuple[PreTrainedModel, QuantizationStrategy]:
    """
    Adaptive quantization function for integration with forge.
    
    Args:
        model_args: Model arguments
        data_args: Data arguments
        quant_args: Quantization arguments
        output_dir: Output directory for quantized model
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (quantized_model, quantization_strategy)
    """
    # Create engine
    engine = create_adaptive_quantization_engine(
        model_args=model_args,
        data_args=data_args,
        quant_args=quant_args,
        **kwargs
    )
    
    # Run quantization search
    strategy = engine.run_quantization_search()
    
    # Apply quantization
    quantized_model = engine.apply_quantization(strategy)
    
    # Save model
    engine.save_quantized_model(output_dir, strategy)
    
    # Generate and save report
    report = engine.get_quantization_report()
    report_path = Path(output_dir) / "quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return quantized_model, strategy


# Example usage and testing
if __name__ == "__main__":
    # This would typically be called from the main training script
    from forge.hparams import get_train_args
    
    # Example configuration
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        quantization_method=QuantizationMethod.AUTO
    )
    
    data_args = DataArguments(
        dataset="alpaca_en_demo",
        template="llama2"
    )
    
    quant_args = QuantizationArguments(
        quantization_method=QuantizationMethod.AUTO,
        calibration_samples=128
    )
    
    # Run adaptive quantization
    try:
        quantized_model, strategy = quantize_model_adaptive(
            model_args=model_args,
            data_args=data_args,
            quant_args=quant_args,
            output_dir="./quantized_model"
        )
        
        print(f"Quantization completed successfully!")
        print(f"Selected method: {strategy.primary_method.value}")
        print(f"Expected memory: {strategy.expected_memory_mb:.2f} MB")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        raise