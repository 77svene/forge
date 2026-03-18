"""
Advanced Optimization Engine for forge
Implements cutting-edge optimization techniques including QLoRA with dynamic quantization,
gradient checkpointing with selective activation, adaptive learning rate scheduling,
and mixed-precision training with automatic loss scaling.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint

try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import AdamW8bit, AdamW32bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from transformers import PreTrainedModel
    from transformers.utils import is_accelerate_available
    if is_accelerate_available():
        from accelerate import Accelerator
        from accelerate.utils import DistributedType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class QuantizationConfig:
    """Configuration for quantization-aware training."""
    
    def __init__(
        self,
        quantization_method: str = "qlora",
        bits: int = 4,
        double_quant: bool = True,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.float16,
        dynamic_quantization: bool = True,
        quantization_threshold: float = 0.1,
    ):
        self.quantization_method = quantization_method
        self.bits = bits
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.dynamic_quantization = dynamic_quantization
        self.quantization_threshold = quantization_threshold
        
        if bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")
        
        if quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"Quantization type must be 'fp4' or 'nf4', got {quant_type}")


class GradientCheckpointingConfig:
    """Configuration for gradient checkpointing with selective activation."""
    
    def __init__(
        self,
        enable: bool = True,
        checkpoint_ratio: float = 0.5,
        selective_layers: Optional[List[str]] = None,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ):
        self.enable = enable
        self.checkpoint_ratio = checkpoint_ratio
        self.selective_layers = selective_layers or []
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state


class AdaptiveSchedulerConfig:
    """Configuration for adaptive learning rate scheduling."""
    
    def __init__(
        self,
        scheduler_type: str = "cosine_with_warmup",
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
        cycle_length: Optional[int] = None,
        loss_monitor_window: int = 100,
        loss_plateau_threshold: float = 0.001,
        gradient_noise_scale_window: int = 50,
        adaptive_momentum: bool = True,
        lookahead_steps: int = 5,
        lookahead_alpha: float = 0.5,
    ):
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.cycle_length = cycle_length
        self.loss_monitor_window = loss_monitor_window
        self.loss_plateau_threshold = loss_plateau_threshold
        self.gradient_noise_scale_window = gradient_noise_scale_window
        self.adaptive_momentum = adaptive_momentum
        self.lookahead_steps = lookahead_steps
        self.lookahead_alpha = lookahead_alpha


class MixedPrecisionConfig:
    """Configuration for mixed-precision training."""
    
    def __init__(
        self,
        enable: bool = True,
        dtype: torch.dtype = torch.float16,
        loss_scaling: str = "dynamic",
        init_scale: float = 2.0**16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
    ):
        self.enable = enable
        self.dtype = dtype
        self.loss_scaling = loss_scaling
        self.init_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale


class DynamicQuantizer:
    """Dynamic quantization manager for QLoRA."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantization_cache = {}
        
    def quantize_weights(self, weight: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Apply dynamic quantization based on weight statistics."""
        if not self.config.dynamic_quantization:
            return weight
            
        # Calculate quantization threshold based on weight distribution
        weight_std = weight.std().item()
        weight_mean = weight.abs().mean().item()
        
        # Adaptive quantization: only quantize if weights are within threshold
        if weight_std / (weight_mean + 1e-8) < self.config.quantization_threshold:
            # Use simpler quantization for stable weights
            return self._quantize_static(weight)
        else:
            # Use more aggressive quantization for volatile weights
            return self._quantize_dynamic(weight)
    
    def _quantize_static(self, weight: torch.Tensor) -> torch.Tensor:
        """Static quantization with fixed parameters."""
        if not HAS_BNB:
            warnings.warn("bitsandbytes not available, skipping quantization")
            return weight
            
        if self.config.bits == 4:
            return bnb.nn.Params4bit(
                weight.data,
                requires_grad=False,
                compress_statistics=self.config.double_quant,
                quant_type=self.config.quant_type,
                blocksize=64,
            ).to(weight.device)
        else:  # 8-bit
            return bnb.nn.Int8Params(
                weight.data,
                requires_grad=False,
                has_fp16_weights=False,
            ).to(weight.device)
    
    def _quantize_dynamic(self, weight: torch.Tensor) -> torch.Tensor:
        """Dynamic quantization with adaptive parameters."""
        # Calculate optimal blocksize based on weight shape
        blocksize = 64 if weight.numel() > 1024 else 32
        
        if self.config.bits == 4:
            return bnb.nn.Params4bit(
                weight.data,
                requires_grad=False,
                compress_statistics=self.config.double_quant,
                quant_type=self.config.quant_type,
                blocksize=blocksize,
            ).to(weight.device)
        else:  # 8-bit
            return bnb.nn.Int8Params(
                weight.data,
                requires_grad=False,
                has_fp16_weights=False,
            ).to(weight.device)


class SelectiveCheckpointing:
    """Gradient checkpointing with selective layer activation."""
    
    def __init__(self, config: GradientCheckpointingConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.checkpointed_layers = set()
        self._setup_checkpointing()
        
    def _setup_checkpointing(self):
        """Setup gradient checkpointing for selected layers."""
        if not self.config.enable:
            return
            
        # Get all checkpointable layers
        checkpointable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)):
                checkpointable_layers.append((name, module))
        
        # Select layers based on checkpoint ratio
        num_to_checkpoint = int(len(checkpointable_layers) * self.config.checkpoint_ratio)
        
        # Prioritize layers with high computational cost
        prioritized_layers = self._prioritize_layers(checkpointable_layers)
        
        for i, (name, module) in enumerate(prioritized_layers):
            if i < num_to_checkpoint or name in self.config.selective_layers:
                self._wrap_layer_with_checkpointing(name, module)
                self.checkpointed_layers.add(name)
    
    def _prioritize_layers(self, layers: List[Tuple[str, nn.Module]]) -> List[Tuple[str, nn.Module]]:
        """Prioritize layers for checkpointing based on computational cost."""
        # Sort by parameter count (higher count = higher priority for checkpointing)
        return sorted(
            layers,
            key=lambda x: sum(p.numel() for p in x[1].parameters()),
            reverse=True
        )
    
    def _wrap_layer_with_checkpointing(self, name: str, module: nn.Module):
        """Wrap a layer with gradient checkpointing."""
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            return checkpoint(
                original_forward,
                *args,
                use_reentrant=self.config.use_reentrant,
                preserve_rng_state=self.config.preserve_rng_state,
                **kwargs
            )
        
        module.forward = checkpointed_forward
    
    def get_checkpointed_layers(self) -> List[str]:
        """Get list of checkpointed layer names."""
        return list(self.checkpointed_layers)


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler based on loss landscape analysis."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: AdaptiveSchedulerConfig,
        num_training_steps: int,
        num_warmup_steps: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.config = config
        self.num_training_steps = num_training_steps
        
        if num_warmup_steps is None:
            self.num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        else:
            self.num_warmup_steps = num_warmup_steps
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
        # Loss monitoring
        self.loss_history = []
        self.gradient_history = []
        
        # Lookahead optimizer components
        if config.adaptive_momentum:
            self._setup_lookahead()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
    
    def _setup_lookahead(self):
        """Setup lookahead optimization components."""
        self.slow_weights = []
        for group in self.optimizer.param_groups:
            slow_group = []
            for param in group['params']:
                if param.requires_grad:
                    slow_group.append(param.data.clone())
            self.slow_weights.append(slow_group)
        
        self.lookahead_step = 0
    
    def _create_scheduler(self) -> LambdaLR:
        """Create the base learning rate scheduler."""
        if self.config.scheduler_type == "cosine_with_warmup":
            return self._cosine_with_warmup_scheduler()
        elif self.config.scheduler_type == "polynomial_with_warmup":
            return self._polynomial_with_warmup_scheduler()
        elif self.config.scheduler_type == "cyclic":
            return self._cyclic_scheduler()
        elif self.config.scheduler_type == "adaptive_cosine":
            return self._adaptive_cosine_scheduler()
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def _cosine_with_warmup_scheduler(self) -> LambdaLR:
        """Cosine scheduler with linear warmup."""
        def lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            return max(
                self.config.min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _polynomial_with_warmup_scheduler(self) -> LambdaLR:
        """Polynomial scheduler with warmup."""
        def lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            return max(
                self.config.min_lr_ratio,
                (1.0 - progress) ** 2.0  # Quadratic decay
            )
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _cyclic_scheduler(self) -> LambdaLR:
        """Cyclic learning rate scheduler."""
        cycle_length = self.config.cycle_length or (self.num_training_steps // 3)
        
        def lr_lambda(current_step: int):
            cycle = current_step // cycle_length
            x = (current_step % cycle_length) / cycle_length
            
            # Triangular cycle
            if x <= 0.5:
                return 1.0 - 2.0 * abs(x - 0.25)
            else:
                return 1.0 - 2.0 * abs(x - 0.75)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _adaptive_cosine_scheduler(self) -> LambdaLR:
        """Adaptive cosine scheduler that adjusts based on loss landscape."""
        def lr_lambda(current_step: int):
            base_lr = self._cosine_with_warmup_scheduler().get_last_lr()[0]
            
            # Adjust based on loss history
            if len(self.loss_history) >= self.config.loss_monitor_window:
                recent_losses = self.loss_history[-self.config.loss_monitor_window:]
                loss_std = torch.std(torch.tensor(recent_losses)).item()
                
                # If loss is plateauing, reduce learning rate more aggressively
                if loss_std < self.config.loss_plateau_threshold:
                    return base_lr * 0.5
                # If loss is oscillating, reduce learning rate
                elif self._detect_oscillation(recent_losses):
                    return base_lr * 0.8
            
            return base_lr
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _detect_oscillation(self, losses: List[float]) -> bool:
        """Detect if loss is oscillating."""
        if len(losses) < 10:
            return False
        
        # Check for alternating increases and decreases
        changes = []
        for i in range(1, len(losses)):
            if losses[i] > losses[i-1]:
                changes.append(1)
            elif losses[i] < losses[i-1]:
                changes.append(-1)
            else:
                changes.append(0)
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(changes)):
            if changes[i] != 0 and changes[i-1] != 0 and changes[i] != changes[i-1]:
                sign_changes += 1
        
        return sign_changes / len(changes) > 0.6
    
    def step(self, loss: Optional[float] = None, gradients: Optional[List[torch.Tensor]] = None):
        """Update learning rate and apply lookahead if configured."""
        # Update loss history
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.config.loss_monitor_window * 2:
                self.loss_history = self.loss_history[-self.config.loss_monitor_window * 2:]
        
        # Update gradient history for noise scale estimation
        if gradients is not None:
            grad_norm = torch.norm(
                torch.stack([torch.norm(g) for g in gradients if g is not None])
            ).item()
            self.gradient_history.append(grad_norm)
            if len(self.gradient_history) > self.config.gradient_noise_scale_window * 2:
                self.gradient_history = self.gradient_history[-self.config.gradient_noise_scale_window * 2:]
        
        # Step the base scheduler
        self.scheduler.step()
        self.current_step += 1
        
        # Apply lookahead update
        if self.config.adaptive_momentum and self.current_step % self.config.lookahead_steps == 0:
            self._apply_lookahead()
    
    def _apply_lookahead(self):
        """Apply lookahead optimization step."""
        self.lookahead_step += 1
        
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param.requires_grad:
                    # Update slow weights
                    slow_weight = self.slow_weights[group_idx][param_idx]
                    slow_weight.add_(
                        self.config.lookahead_alpha * (param.data - slow_weight)
                    )
                    # Update fast weights with slow weights
                    param.data.copy_(slow_weight)
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return self.scheduler.get_last_lr()
    
    def get_lr(self) -> List[float]:
        """Get current learning rates (alias for compatibility)."""
        return self.get_last_lr()


class AdvancedOptimizer:
    """
    Advanced optimizer combining quantization-aware training, gradient checkpointing,
    adaptive learning rate scheduling, and mixed-precision training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        quantization_config: Optional[QuantizationConfig] = None,
        checkpointing_config: Optional[GradientCheckpointingConfig] = None,
        scheduler_config: Optional[AdaptiveSchedulerConfig] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None,
        num_training_steps: int = 10000,
        accelerator: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        
        # Setup quantization
        self.quantizer = None
        if quantization_config and quantization_config.quantization_method == "qlora":
            if not HAS_BNB:
                warnings.warn("bitsandbytes not available, QLoRA disabled")
            else:
                self.quantizer = DynamicQuantizer(quantization_config)
                self._setup_qlora()
        
        # Setup gradient checkpointing
        self.checkpointing = None
        if checkpointing_config and checkpointing_config.enable:
            self.checkpointing = SelectiveCheckpointing(checkpointing_config, model)
        
        # Setup adaptive scheduler
        self.scheduler = None
        if scheduler_config:
            self.scheduler = AdaptiveLRScheduler(
                optimizer, scheduler_config, num_training_steps
            )
        
        # Setup mixed precision
        self.mixed_precision = None
        if mixed_precision_config and mixed_precision_config.enable:
            self.mixed_precision = MixedPrecisionManager(mixed_precision_config)
        
        # Training state
        self.step_count = 0
        self.loss_scale = 1.0
        self.last_loss = None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 1
        self.current_accumulation_step = 0
    
    def _setup_qlora(self):
        """Setup QLoRA quantization for the model."""
        if not HAS_TRANSFORMERS or not isinstance(self.model, PreTrainedModel):
            return
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable gradient for LoRA parameters
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        
        # Apply dynamic quantization to non-LoRA weights
        self._apply_dynamic_quantization()
    
    def _apply_dynamic_quantization(self):
        """Apply dynamic quantization to model weights."""
        if not self.quantizer:
            return
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Skip LoRA layers (they are trainable)
                if 'lora' in name.lower():
                    continue
                
                # Apply quantization
                module.weight = self.quantizer.quantize_weights(
                    module.weight, name
                )
    
    def set_gradient_accumulation_steps(self, steps: int):
        """Set gradient accumulation steps."""
        self.gradient_accumulation_steps = steps
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed-precision and gradient accumulation."""
        # Scale loss for mixed precision
        if self.mixed_precision:
            loss = self.mixed_precision.scale_loss(loss)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.current_accumulation_step += 1
        
        # Store loss for scheduler
        if self.last_loss is None:
            self.last_loss = loss.item() * self.gradient_accumulation_steps
        else:
            self.last_loss = 0.9 * self.last_loss + 0.1 * loss.item() * self.gradient_accumulation_steps
    
    def step(self):
        """Optimization step with gradient clipping and adaptive scheduling."""
        if self.current_accumulation_step < self.gradient_accumulation_steps:
            return
        
        # Get gradients for adaptive scheduling
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad)
        
        # Gradient clipping
        if self.accelerator:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step(loss=self.last_loss, gradients=gradients)
        
        # Update mixed precision scaling
        if self.mixed_precision:
            self.mixed_precision.update()
        
        # Reset accumulation
        self.current_accumulation_step = 0
        self.step_count += 1
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.scheduler:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'last_loss': self.last_loss,
            'current_accumulation_step': self.current_accumulation_step,
        }
        
        if self.scheduler:
            state['scheduler'] = self.scheduler.scheduler.state_dict()
        
        if self.mixed_precision:
            state['mixed_precision'] = self.mixed_precision.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step_count = state_dict['step_count']
        self.last_loss = state_dict['last_loss']
        self.current_accumulation_step = state_dict['current_accumulation_step']
        
        if self.scheduler and 'scheduler' in state_dict:
            self.scheduler.scheduler.load_state_dict(state_dict['scheduler'])
        
        if self.mixed_precision and 'mixed_precision' in state_dict:
            self.mixed_precision.load_state_dict(state_dict['mixed_precision'])


class MixedPrecisionManager:
    """Manager for mixed-precision training with automatic loss scaling."""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=config.init_scale,
            growth_factor=config.scale_factor,
            backoff_factor=1.0 / config.scale_factor,
            growth_interval=config.scale_window,
            enabled=config.enable,
        )
        self.autocast = torch.cuda.amp.autocast(
            dtype=config.dtype,
            enabled=config.enable,
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed-precision training."""
        return self.scaler.scale(loss)
    
    def update(self):
        """Update loss scaling."""
        self.scaler.update()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state dict."""
        return self.scaler.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scaler state dict."""
        self.scaler.load_state_dict(state_dict)


def create_advanced_optimizer(
    model: nn.Module,
    optimizer: Optimizer,
    training_args: Any,
    num_training_steps: int,
    accelerator: Optional[Any] = None,
) -> AdvancedOptimizer:
    """
    Factory function to create an advanced optimizer from training arguments.
    
    Args:
        model: The model to optimize
        optimizer: The base optimizer
        training_args: Training arguments containing optimization configs
        num_training_steps: Total number of training steps
        accelerator: Optional accelerator for distributed training
        
    Returns:
        Configured AdvancedOptimizer instance
    """
    # Extract configs from training_args
    quantization_config = None
    if hasattr(training_args, 'quantization_method'):
        quantization_config = QuantizationConfig(
            quantization_method=training_args.quantization_method,
            bits=getattr(training_args, 'quantization_bits', 4),
            double_quant=getattr(training_args, 'double_quant', True),
            quant_type=getattr(training_args, 'quant_type', 'nf4'),
            compute_dtype=getattr(training_args, 'compute_dtype', torch.float16),
            dynamic_quantization=getattr(training_args, 'dynamic_quantization', True),
        )
    
    checkpointing_config = None
    if hasattr(training_args, 'gradient_checkpointing'):
        checkpointing_config = GradientCheckpointingConfig(
            enable=training_args.gradient_checkpointing,
            checkpoint_ratio=getattr(training_args, 'checkpoint_ratio', 0.5),
            selective_layers=getattr(training_args, 'selective_checkpoint_layers', []),
        )
    
    scheduler_config = None
    if hasattr(training_args, 'lr_scheduler_type'):
        scheduler_config = AdaptiveSchedulerConfig(
            scheduler_type=training_args.lr_scheduler_type,
            warmup_ratio=getattr(training_args, 'warmup_ratio', 0.1),
            min_lr_ratio=getattr(training_args, 'min_lr_ratio', 0.1),
            adaptive_momentum=getattr(training_args, 'adaptive_momentum', True),
        )
    
    mixed_precision_config = None
    if hasattr(training_args, 'fp16') or hasattr(training_args, 'bf16'):
        dtype = torch.float16 if getattr(training_args, 'fp16', False) else torch.bfloat16
        mixed_precision_config = MixedPrecisionConfig(
            enable=getattr(training_args, 'fp16', False) or getattr(training_args, 'bf16', False),
            dtype=dtype,
            loss_scaling=getattr(training_args, 'loss_scaling', 'dynamic'),
        )
    
    return AdvancedOptimizer(
        model=model,
        optimizer=optimizer,
        quantization_config=quantization_config,
        checkpointing_config=checkpointing_config,
        scheduler_config=scheduler_config,
        mixed_precision_config=mixed_precision_config,
        num_training_steps=num_training_steps,
        accelerator=accelerator,
    )


# Export main classes and functions
__all__ = [
    'QuantizationConfig',
    'GradientCheckpointingConfig',
    'AdaptiveSchedulerConfig',
    'MixedPrecisionConfig',
    'DynamicQuantizer',
    'SelectiveCheckpointing',
    'AdaptiveLRScheduler',
    'AdvancedOptimizer',
    'MixedPrecisionManager',
    'create_advanced_optimizer',
]