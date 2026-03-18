# coding=utf-8
# Copyright 2024 forge Team. All rights reserved.
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

"""
Advanced Memory Optimization Engine for forge
Implements cutting-edge optimization techniques for efficient large model training
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    logger.warning("bitsandbytes not installed. 8-bit/4-bit optimization disabled.")

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False
    logger.warning("CUDA AMP not available. Mixed precision training disabled.")


class QuantizationConfig:
    """Configuration for quantization-aware training"""
    
    def __init__(
        self,
        quantization_bit: Optional[int] = None,
        quantization_type: str = "nf4",
        double_quantization: bool = True,
        quantization_compute_dtype: torch.dtype = torch.float16,
        quantization_has_fp16_weights: bool = False,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    ):
        self.quantization_bit = quantization_bit
        self.quantization_type = quantization_type
        self.double_quantization = double_quantization
        self.quantization_compute_dtype = quantization_compute_dtype
        self.quantization_has_fp16_weights = quantization_has_fp16_weights
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype


class GradientCheckpointingConfig:
    """Configuration for gradient checkpointing"""
    
    def __init__(
        self,
        gradient_checkpointing_ratio: float = 0.5,
        selective_checkpointing: bool = True,
        checkpoint_every_n_layers: int = 1,
        checkpoint_attention_layers: bool = True,
        checkpoint_mlp_layers: bool = True,
        use_reentrant: bool = False,
    ):
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.selective_checkpointing = selective_checkpointing
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpoint_attention_layers = checkpoint_attention_layers
        self.checkpoint_mlp_layers = checkpoint_mlp_layers
        self.use_reentrant = use_reentrant


class AdaptiveSchedulerConfig:
    """Configuration for adaptive learning rate scheduling"""
    
    def __init__(
        self,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
        cosine_restarts: bool = True,
        restart_period: int = 1000,
        loss_plateau_patience: int = 100,
        loss_plateau_threshold: float = 1e-4,
        gradient_norm_threshold: float = 1.0,
        adaptive_warmup: bool = True,
        dynamic_batch_size: bool = False,
    ):
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.cosine_restarts = cosine_restarts
        self.restart_period = restart_period
        self.loss_plateau_patience = loss_plateau_patience
        self.loss_plateau_threshold = loss_plateau_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.adaptive_warmup = adaptive_warmup
        self.dynamic_batch_size = dynamic_batch_size


class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    
    def __init__(
        self,
        fp16: bool = True,
        bf16: bool = False,
        loss_scale: str = "dynamic",
        initial_loss_scale: float = 2**16,
        loss_scale_window: float = 1000,
        hysteresis: int = 2,
        min_loss_scale: float = 1.0,
        max_loss_scale: float = 2**24,
    ):
        self.fp16 = fp16
        self.bf16 = bf16
        self.loss_scale = loss_scale
        self.initial_loss_scale = initial_loss_scale
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale


class QLoRAModelWrapper:
    """Wrapper for QLoRA with dynamic quantization"""
    
    def __init__(self, model: PreTrainedModel, quantization_config: QuantizationConfig):
        if not HAS_BNB:
            raise ImportError("bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes")
        
        self.model = model
        self.quantization_config = quantization_config
        self._original_modules = {}
        
    def quantize_model(self) -> PreTrainedModel:
        """Apply QLoRA quantization to the model"""
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Store original module for potential dequantization
                self._original_modules[name] = module
                
                # Determine quantization parameters
                if self.quantization_config.quantization_bit == 4:
                    # 4-bit quantization
                    quantized_module = Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=self.quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=self.quantization_config.bnb_4bit_use_double_quant,
                        quant_type=self.quantization_config.bnb_4bit_quant_type,
                    )
                elif self.quantization_config.quantization_bit == 8:
                    # 8-bit quantization
                    quantized_module = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=self.quantization_config.quantization_has_fp16_weights,
                        threshold=6.0,
                    )
                else:
                    continue
                
                # Copy weights and biases
                quantized_module.weight = module.weight
                if module.bias is not None:
                    quantized_module.bias = module.bias
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, child_name, quantized_module)
                else:
                    setattr(self.model, child_name, quantized_module)
        
        return self.model
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters (only LoRA parameters should be trainable)"""
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def dequantize_model(self) -> PreTrainedModel:
        """Restore original model (for inference)"""
        for name, module in self._original_modules.items():
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(self.model.named_modules())[parent_name]
                setattr(parent, child_name, module)
            else:
                setattr(self.model, child_name, module)
        
        return self.model


class SelectiveGradientCheckpointing:
    """Gradient checkpointing with selective activation checkpointing"""
    
    def __init__(self, model: PreTrainedModel, config: GradientCheckpointingConfig):
        self.model = model
        self.config = config
        self._checkpointed_modules = set()
        
    def enable_checkpointing(self) -> None:
        """Enable gradient checkpointing with selective activation"""
        if not self.config.selective_checkpointing:
            # Standard gradient checkpointing
            self.model.gradient_checkpointing_enable()
            return
        
        # Selective checkpointing based on layer type and position
        for name, module in self.model.named_modules():
            if self._should_checkpoint(name, module):
                self._checkpoint_module(name, module)
    
    def _should_checkpoint(self, name: str, module: nn.Module) -> bool:
        """Determine if a module should be checkpointed"""
        # Check if module is attention or MLP layer
        is_attention = any(attn_name in name.lower() for attn_name in ["attention", "attn", "self_attn"])
        is_mlp = any(mlp_name in name.lower() for mlp_name in ["mlp", "ffn", "feed_forward"])
        
        if is_attention and not self.config.checkpoint_attention_layers:
            return False
        if is_mlp and not self.config.checkpoint_mlp_layers:
            return False
        
        # Check layer position (every n layers)
        if "layers" in name or "blocks" in name:
            try:
                # Extract layer number
                parts = name.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_num = int(part)
                        if layer_num % self.config.checkpoint_every_n_layers != 0:
                            return False
            except (ValueError, IndexError):
                pass
        
        # Check gradient checkpointing ratio
        if self.config.gradient_checkpointing_ratio < 1.0:
            # Randomly select modules based on ratio
            import hashlib
            hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
            if hash_val / (2**128) > self.config.gradient_checkpointing_ratio:
                return False
        
        return True
    
    def _checkpoint_module(self, name: str, module: nn.Module) -> None:
        """Apply checkpointing to a specific module"""
        from torch.utils.checkpoint import checkpoint
        
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            return checkpoint(
                original_forward,
                *args,
                use_reentrant=self.config.use_reentrant,
                **kwargs
            )
        
        module.forward = checkpointed_forward
        self._checkpointed_modules.add(name)
    
    def disable_checkpointing(self) -> None:
        """Disable gradient checkpointing"""
        for name, module in self.model.named_modules():
            if name in self._checkpointed_modules:
                # Restore original forward method
                if hasattr(module, '_original_forward'):
                    module.forward = module._original_forward
        
        self._checkpointed_modules.clear()


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler based on loss landscape analysis"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        config: AdaptiveSchedulerConfig,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.config = config
        self.last_epoch = last_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        
        # Loss tracking
        self.loss_history = []
        self.gradient_norm_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Warmup tracking
        self.warmup_steps = int(num_training_steps * config.warmup_ratio)
        self.current_step = 0
        
    def step(self, loss: Optional[float] = None, gradient_norm: Optional[float] = None) -> None:
        """Update learning rate based on training dynamics"""
        self.current_step += 1
        
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > 1000:
                self.loss_history.pop(0)
        
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
            if len(self.gradient_norm_history) > 1000:
                self.gradient_norm_history.pop(0)
        
        # Calculate new learning rate
        new_lrs = self._calculate_lrs()
        
        # Apply new learning rates
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = lr
        
        self.last_epoch += 1
    
    def _calculate_lrs(self) -> List[float]:
        """Calculate new learning rates based on training dynamics"""
        lrs = []
        
        for base_lr in self.base_lrs:
            # Warmup phase
            if self.current_step < self.warmup_steps:
                if self.config.adaptive_warmup:
                    # Adaptive warmup based on gradient norms
                    if self.gradient_norm_history:
                        avg_grad_norm = sum(self.gradient_norm_history[-10:]) / min(10, len(self.gradient_norm_history))
                        warmup_factor = min(1.0, self.config.gradient_norm_threshold / max(avg_grad_norm, 1e-8))
                    else:
                        warmup_factor = 1.0
                else:
                    warmup_factor = 1.0
                
                lr = base_lr * (self.current_step / self.warmup_steps) * warmup_factor
                lrs.append(lr)
                continue
            
            # Cosine annealing with restarts
            if self.config.cosine_restarts:
                cycle_step = (self.current_step - self.warmup_steps) % self.config.restart_period
                cycle_length = self.config.restart_period
                cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_step / cycle_length))
            else:
                progress = (self.current_step - self.warmup_steps) / (self.num_training_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Loss plateau detection
            if len(self.loss_history) >= self.config.loss_plateau_patience:
                recent_losses = self.loss_history[-self.config.loss_plateau_patience:]
                loss_change = abs(max(recent_losses) - min(recent_losses))
                
                if loss_change < self.config.loss_plateau_threshold:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.loss_plateau_patience:
                        # Reduce learning rate on plateau
                        cosine_decay *= 0.5
                        self.patience_counter = 0
                else:
                    self.patience_counter = 0
            
            # Gradient norm adjustment
            if self.gradient_norm_history and self.config.dynamic_batch_size:
                avg_grad_norm = sum(self.gradient_norm_history[-10:]) / min(10, len(self.gradient_norm_history))
                if avg_grad_norm > self.config.gradient_norm_threshold * 2:
                    # Reduce learning rate if gradients are too large
                    cosine_decay *= 0.8
                elif avg_grad_norm < self.config.gradient_norm_threshold * 0.5:
                    # Increase learning rate if gradients are too small
                    cosine_decay *= 1.2
            
            # Final learning rate calculation
            lr = base_lr * (
                self.config.min_lr_ratio +
                (1 - self.config.min_lr_ratio) * cosine_decay
            )
            lrs.append(lr)
        
        return lrs
    
    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate"""
        return [group["lr"] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state"""
        return {
            "last_epoch": self.last_epoch,
            "current_step": self.current_step,
            "loss_history": self.loss_history,
            "gradient_norm_history": self.gradient_norm_history,
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state"""
        self.last_epoch = state_dict["last_epoch"]
        self.current_step = state_dict["current_step"]
        self.loss_history = state_dict["loss_history"]
        self.gradient_norm_history = state_dict["gradient_norm_history"]
        self.best_loss = state_dict["best_loss"]
        self.patience_counter = state_dict["patience_counter"]


class MixedPrecisionTrainer:
    """Mixed precision training with automatic loss scaling"""
    
    def __init__(self, model: PreTrainedModel, config: MixedPrecisionConfig):
        if not HAS_AMP:
            raise RuntimeError("CUDA AMP is not available")
        
        self.model = model
        self.config = config
        self.scaler = None
        
        if config.fp16 and config.loss_scale == "dynamic":
            self.scaler = GradScaler(
                init_scale=config.initial_loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=config.loss_scale_window,
                enabled=True,
            )
    
    def training_step(
        self,
        model_forward_fn,
        inputs: Dict[str, torch.Tensor],
        optimizer: Optimizer,
        gradient_accumulation_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a mixed precision training step"""
        
        # Determine compute dtype
        if self.config.bf16 and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        elif self.config.fp16:
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32
        
        # Forward pass with autocast
        with autocast(dtype=compute_dtype):
            loss = model_forward_fn(inputs)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step with gradient unscaling
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Return loss and metrics
        metrics = {
            "loss": loss.item() * gradient_accumulation_steps,
            "grad_norm": grad_norm.item(),
            "scale": self.scaler.get_scale() if self.scaler else 1.0,
        }
        
        return loss, metrics


class MemoryOptimizer:
    """Main memory optimization engine combining all techniques"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        quantization_config: Optional[QuantizationConfig] = None,
        gradient_checkpointing_config: Optional[GradientCheckpointingConfig] = None,
        adaptive_scheduler_config: Optional[AdaptiveSchedulerConfig] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None,
    ):
        self.model = model
        self.quantization_config = quantization_config
        self.gradient_checkpointing_config = gradient_checkpointing_config
        self.adaptive_scheduler_config = adaptive_scheduler_config
        self.mixed_precision_config = mixed_precision_config
        
        # Initialize components
        self.qlora_wrapper = None
        self.gradient_checkpointing = None
        self.mixed_precision_trainer = None
        
        # Apply optimizations
        self._apply_optimizations()
    
    def _apply_optimizations(self) -> None:
        """Apply all configured optimizations"""
        
        # Apply quantization if configured
        if self.quantization_config and self.quantization_config.quantization_bit:
            logger.info(f"Applying {self.quantization_config.quantization_bit}-bit quantization")
            self.qlora_wrapper = QLoRAModelWrapper(self.model, self.quantization_config)
            self.model = self.qlora_wrapper.quantize_model()
        
        # Apply gradient checkpointing if configured
        if self.gradient_checkpointing_config:
            logger.info("Applying gradient checkpointing")
            self.gradient_checkpointing = SelectiveGradientCheckpointing(
                self.model, self.gradient_checkpointing_config
            )
            self.gradient_checkpointing.enable_checkpointing()
        
        # Initialize mixed precision trainer if configured
        if self.mixed_precision_config:
            logger.info("Initializing mixed precision training")
            self.mixed_precision_trainer = MixedPrecisionTrainer(
                self.model, self.mixed_precision_config
            )
    
    def create_optimizer(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        **kwargs,
    ) -> Optimizer:
        """Create optimizer with quantization support"""
        
        # Get trainable parameters
        if self.qlora_wrapper:
            trainable_params = self.qlora_wrapper.get_trainable_parameters()
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer based on type and quantization
        if optimizer_type.lower() == "adamw":
            if self.quantization_config and self.quantization_config.quantization_bit == 8 and HAS_BNB:
                optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs,
                )
            elif self.quantization_config and self.quantization_config.quantization_bit == 4 and HAS_BNB:
                optimizer = bnb.optim.AdamW4bit(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs,
                )
            else:
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs,
                )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
    ) -> Optional[AdaptiveLRScheduler]:
        """Create adaptive learning rate scheduler"""
        
        if self.adaptive_scheduler_config:
            return AdaptiveLRScheduler(
                optimizer=optimizer,
                num_training_steps=num_training_steps,
                config=self.adaptive_scheduler_config,
            )
        
        return None
    
    def training_step(
        self,
        model_forward_fn,
        inputs: Dict[str, torch.Tensor],
        optimizer: Optimizer,
        gradient_accumulation_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform an optimized training step"""
        
        if self.mixed_precision_trainer:
            return self.mixed_precision_trainer.training_step(
                model_forward_fn, inputs, optimizer, gradient_accumulation_steps
            )
        else:
            # Standard training step
            loss = model_forward_fn(inputs)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            metrics = {
                "loss": loss.item() * gradient_accumulation_steps,
                "grad_norm": grad_norm.item(),
                "scale": 1.0,
            }
            
            return loss, metrics
    
    def get_model(self) -> PreTrainedModel:
        """Get the optimized model"""
        return self.model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up optimizations"""
        if self.gradient_checkpointing:
            self.gradient_checkpointing.disable_checkpointing()
        
        if self.qlora_wrapper:
            self.model = self.qlora_wrapper.dequantize_model()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_memory_optimizer(
    model: PreTrainedModel,
    quantization_bit: Optional[int] = None,
    gradient_checkpointing: bool = True,
    adaptive_lr: bool = True,
    mixed_precision: bool = True,
    **kwargs,
) -> MemoryOptimizer:
    """Factory function to create memory optimizer with default configurations"""
    
    # Create configurations
    quantization_config = None
    if quantization_bit:
        quantization_config = QuantizationConfig(
            quantization_bit=quantization_bit,
            **{k: v for k, v in kwargs.items() if k.startswith("quantization_")},
        )
    
    gradient_checkpointing_config = None
    if gradient_checkpointing:
        gradient_checkpointing_config = GradientCheckpointingConfig(
            **{k: v for k, v in kwargs.items() if k.startswith("gradient_checkpointing_")},
        )
    
    adaptive_scheduler_config = None
    if adaptive_lr:
        adaptive_scheduler_config = AdaptiveSchedulerConfig(
            **{k: v for k, v in kwargs.items() if k.startswith("adaptive_")},
        )
    
    mixed_precision_config = None
    if mixed_precision:
        mixed_precision_config = MixedPrecisionConfig(
            **{k: v for k, v in kwargs.items() if k.startswith("mixed_precision_")},
        )
    
    return MemoryOptimizer(
        model=model,
        quantization_config=quantization_config,
        gradient_checkpointing_config=gradient_checkpointing_config,
        adaptive_scheduler_config=adaptive_scheduler_config,
        mixed_precision_config=mixed_precision_config,
    )


# Export public API
__all__ = [
    "MemoryOptimizer",
    "QuantizationConfig",
    "GradientCheckpointingConfig",
    "AdaptiveSchedulerConfig",
    "MixedPrecisionConfig",
    "QLoRAModelWrapper",
    "SelectiveGradientCheckpointing",
    "AdaptiveLRScheduler",
    "MixedPrecisionTrainer",
    "create_memory_optimizer",
]