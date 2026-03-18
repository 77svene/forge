import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
from collections import deque

logger = logging.getLogger(__name__)


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    DYNAMIC = "dynamic"
    QLORA = "qlora"


@dataclass
class QuantizationConfig:
    """Configuration for quantization-aware training and inference."""
    method: QuantizationMethod = QuantizationMethod.NONE
    bits: int = 4
    group_size: int = 128
    use_double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.float16
    compress_statistics: bool = True
    dynamic_threshold: float = 0.99
    enable_gradient_checkpointing: bool = True
    checkpoint_ratio: float = 0.5
    selective_checkpoint_layers: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.method == QuantizationMethod.QLORA:
            self.bits = 4
            self.quant_type = "nf4"
            self.use_double_quant = True


class DynamicQuantizer:
    """Dynamic quantization with adaptive thresholds."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.activation_stats = {}
        self.weight_stats = {}
        
    def collect_statistics(self, model: nn.Module):
        """Collect activation and weight statistics for dynamic quantization."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.weight_stats[name] = {
                    'abs_max': module.weight.data.abs().max().item(),
                    'mean': module.weight.data.mean().item(),
                    'std': module.weight.data.std().item()
                }
    
    def quantize_weight(self, weight: torch.Tensor, name: str) -> Tuple[torch.Tensor, Dict]:
        """Quantize weight tensor dynamically."""
        if self.config.method == QuantizationMethod.DYNAMIC:
            # Dynamic quantization with percentile-based threshold
            threshold = torch.quantile(weight.abs().flatten(), self.config.dynamic_threshold)
            scale = threshold / (2 ** (self.config.bits - 1) - 1)
            quantized = torch.clamp(torch.round(weight / scale), 
                                   -(2 ** (self.config.bits - 1)), 
                                   2 ** (self.config.bits - 1) - 1)
            return quantized, {'scale': scale, 'threshold': threshold}
        return weight, {}
    
    def quantize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """Quantize activation tensor for forward pass."""
        if self.config.method in [QuantizationMethod.INT8, QuantizationMethod.DYNAMIC]:
            # Symmetric quantization for activations
            max_val = activation.abs().max()
            scale = max_val / 127  # 8-bit symmetric
            return torch.clamp(torch.round(activation / scale), -128, 127) * scale
        return activation


class QLoRALayer(nn.Module):
    """QLoRA layer with 4-bit quantization and low-rank adaptation."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 config: QuantizationConfig,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Quantized weights
        self.register_buffer('weight_quantized', 
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('weight_zero', torch.zeros(out_features))
        
        # Low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, config.group_size))
        self.lora_B = nn.Parameter(torch.zeros(config.group_size, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to 4-bit with double quantization."""
        # Reshape for group-wise quantization
        weight_reshaped = weight.reshape(-1, self.config.group_size)
        
        # Compute scales and zero points per group
        min_val = weight_reshaped.min(dim=1)[0]
        max_val = weight_reshaped.max(dim=1)[0]
        
        # Symmetric quantization
        abs_max = torch.max(min_val.abs(), max_val.abs())
        scale = abs_max / (2 ** (self.config.bits - 1) - 1)
        zero = torch.zeros_like(scale)
        
        # Quantize
        weight_scaled = weight_reshaped / scale.unsqueeze(1)
        weight_quant = torch.clamp(torch.round(weight_scaled), 
                                  -(2 ** (self.config.bits - 1)), 
                                  2 ** (self.config.bits - 1) - 1)
        
        # Reshape back
        self.weight_quantized.copy_(weight_quant.reshape(weight.shape).to(torch.int8))
        self.weight_scale.copy_(scale)
        self.weight_zero.copy_(zero)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight_dequant = self.weight_quantized.to(x.dtype) * self.weight_scale.unsqueeze(1)
        
        # Apply LoRA adaptation
        lora_weight = self.lora_A @ self.lora_B
        weight = weight_dequant + lora_weight
        
        # Forward pass
        output = F.linear(x, weight, self.bias)
        return output


class SelectiveGradientCheckpoint:
    """Gradient checkpointing with selective layer activation."""
    
    def __init__(self, 
                 model: nn.Module, 
                 config: QuantizationConfig):
        self.model = model
        self.config = config
        self.checkpoint_layers = set()
        self._setup_checkpointing()
        
    def _setup_checkpointing(self):
        """Setup gradient checkpointing for selected layers."""
        if not self.config.enable_gradient_checkpointing:
            return
            
        # Determine which layers to checkpoint
        all_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, 
                                  nn.TransformerDecoderLayer,
                                  nn.ModuleList)):
                all_layers.append((name, module))
        
        # Select layers based on ratio or specific indices
        if self.config.selective_checkpoint_layers:
            selected_indices = self.config.selective_checkpoint_layers
        else:
            num_to_checkpoint = int(len(all_layers) * self.config.checkpoint_ratio)
            selected_indices = torch.randperm(len(all_layers))[:num_to_checkpoint].tolist()
        
        for idx in selected_indices:
            if idx < len(all_layers):
                name, module = all_layers[idx]
                self.checkpoint_layers.add(name)
                self._wrap_layer(module, name)
                
    def _wrap_layer(self, layer: nn.Module, name: str):
        """Wrap layer with gradient checkpointing."""
        original_forward = layer.forward
        
        def checkpointed_forward(*args, **kwargs):
            if self.training:
                # Use gradient checkpointing for this layer
                def custom_forward(*inputs):
                    return original_forward(*inputs, **kwargs)
                
                return torch.utils.checkpoint.checkpoint(
                    custom_forward, 
                    *args,
                    use_reentrant=False
                )
            else:
                return original_forward(*args, **kwargs)
        
        layer.forward = checkpointed_forward
        
    def enable(self):
        """Enable gradient checkpointing."""
        self.config.enable_gradient_checkpointing = True
        self._setup_checkpointing()
        
    def disable(self):
        """Disable gradient checkpointing."""
        self.config.enable_gradient_checkpointing = False
        # Restore original forward methods
        for name, module in self.model.named_modules():
            if name in self.checkpoint_layers:
                if hasattr(module, '_original_forward'):
                    module.forward = module._original_forward


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler based on loss landscape analysis."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 initial_lr: float = 1e-4,
                 warmup_steps: int = 1000,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-3,
                 patience: int = 10,
                 factor: float = 0.5,
                 monitor: str = 'loss',
                 window_size: int = 100):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.window_size = window_size
        
        self.step_count = 0
        self.best_value = float('inf')
        self.patience_counter = 0
        self.loss_history = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        self.lr_history = []
        
        # Initialize learning rate
        self._set_lr(initial_lr)
        
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr_history.append(lr)
        
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def compute_gradient_norm(self) -> float:
        """Compute gradient norm for adaptive scheduling."""
        total_norm = 0.0
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    
    def analyze_loss_landscape(self) -> Dict[str, float]:
        """Analyze loss landscape characteristics."""
        if len(self.loss_history) < 2:
            return {'curvature': 0.0, 'smoothness': 1.0}
        
        # Compute loss curvature (second derivative approximation)
        losses = list(self.loss_history)
        curvature = abs(losses[-1] - 2 * losses[-2] + losses[-3]) if len(losses) >= 3 else 0.0
        
        # Compute loss smoothness (variance)
        smoothness = 1.0 / (1.0 + torch.var(torch.tensor(losses)).item())
        
        return {
            'curvature': curvature,
            'smoothness': smoothness,
            'trend': self._compute_trend()
        }
    
    def _compute_trend(self) -> float:
        """Compute loss trend over window."""
        if len(self.loss_history) < 10:
            return 0.0
        
        losses = list(self.loss_history)
        x = torch.arange(len(losses), dtype=torch.float32)
        y = torch.tensor(losses, dtype=torch.float32)
        
        # Linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope.item()
    
    def step(self, loss: float, metrics: Optional[Dict] = None):
        """Update learning rate based on training dynamics."""
        self.step_count += 1
        self.loss_history.append(loss)
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            lr = self.initial_lr * warmup_factor
            self._set_lr(lr)
            return
        
        # Compute gradient norm
        grad_norm = self.compute_gradient_norm()
        self.gradient_norms.append(grad_norm)
        
        # Analyze landscape
        landscape = self.analyze_loss_landscape()
        
        # Adaptive scheduling logic
        current_lr = self.get_lr()
        
        # Check for plateau
        if loss < self.best_value:
            self.best_value = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Adjust learning rate based on multiple factors
        new_lr = current_lr
        
        # Factor 1: Plateau detection
        if self.patience_counter >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            self.patience_counter = 0
            logger.info(f"Reducing LR to {new_lr:.2e} due to plateau")
        
        # Factor 2: Gradient norm
        if grad_norm > 10.0:  # Gradient explosion
            new_lr = max(current_lr * 0.8, self.min_lr)
            logger.info(f"Reducing LR to {new_lr:.2e} due to large gradient norm: {grad_norm:.2f}")
        elif grad_norm < 0.1 and landscape['curvature'] < 0.01:  # Flat region
            new_lr = min(current_lr * 1.1, self.max_lr)
            logger.info(f"Increasing LR to {new_lr:.2e} due to flat region")
        
        # Factor 3: Loss trend
        if landscape['trend'] > 0.01 and self.step_count > self.warmup_steps * 2:  # Loss increasing
            new_lr = max(current_lr * 0.9, self.min_lr)
            logger.info(f"Reducing LR to {new_lr:.2e} due to increasing loss trend")
        
        # Factor 4: Cosine annealing component
        progress = (self.step_count - self.warmup_steps) / max(1, 100000 - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        cosine_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        
        # Blend adaptive and cosine schedules
        if self.step_count % 100 == 0:  # Every 100 steps
            new_lr = 0.7 * new_lr + 0.3 * cosine_lr
        
        # Apply new learning rate
        if abs(new_lr - current_lr) > current_lr * 0.01:  # Only if change is significant
            self._set_lr(new_lr)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {
            'step_count': self.step_count,
            'best_value': self.best_value,
            'patience_counter': self.patience_counter,
            'loss_history': list(self.loss_history),
            'gradient_norms': list(self.gradient_norms),
            'lr_history': self.lr_history,
            'current_lr': self.get_lr()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.step_count = state_dict['step_count']
        self.best_value = state_dict['best_value']
        self.patience_counter = state_dict['patience_counter']
        self.loss_history = deque(state_dict['loss_history'], maxlen=self.window_size)
        self.gradient_norms = deque(state_dict['gradient_norms'], maxlen=self.window_size)
        self.lr_history = state_dict['lr_history']
        self._set_lr(state_dict['current_lr'])


class MixedPrecisionTrainer:
    """Mixed precision trainer with automatic loss scaling."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: QuantizationConfig,
                 init_scale: float = 2.0 ** 16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.enabled = enabled and (config.compute_dtype == torch.float16)
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            self.autocast_enabled = True
        else:
            self.scaler = None
            self.autocast_enabled = False
            
        self.loss_scale_history = []
        
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.autocast_enabled:
            with autocast(dtype=self.config.compute_dtype):
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """Optimizer step with gradient unscaling."""
        if self.enabled:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Record scale
            self.loss_scale_history.append(self.scaler.get_scale())
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Return trainer state."""
        state = {'enabled': self.enabled}
        if self.enabled:
            state['scaler'] = self.scaler.state_dict()
            state['loss_scale_history'] = self.loss_scale_history
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state."""
        self.enabled = state_dict['enabled']
        if self.enabled and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
            self.loss_scale_history = state_dict.get('loss_scale_history', [])


class OptimizationEngine:
    """Main optimization engine combining all techniques."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 quantization_config: Optional[QuantizationConfig] = None):
        self.model = model
        self.optimizer = optimizer
        self.config = quantization_config or QuantizationConfig()
        
        # Initialize components
        self.quantizer = DynamicQuantizer(self.config) if self.config.method != QuantizationMethod.NONE else None
        self.gradient_checkpoint = SelectiveGradientCheckpoint(model, self.config)
        self.lr_scheduler = AdaptiveLRScheduler(optimizer)
        self.mixed_precision = MixedPrecisionTrainer(model, optimizer, self.config)
        
        # Apply quantization if needed
        if self.config.method != QuantizationMethod.NONE:
            self._apply_quantization()
        
        logger.info(f"OptimizationEngine initialized with method: {self.config.method}")
    
    def _apply_quantization(self):
        """Apply quantization to model."""
        if self.config.method == QuantizationMethod.QLORA:
            self._convert_to_qlora()
        elif self.config.method in [QuantizationMethod.INT8, QuantizationMethod.INT4, QuantizationMethod.DYNAMIC]:
            self.quantizer.collect_statistics(self.model)
    
    def _convert_to_qlora(self):
        """Convert linear layers to QLoRA layers."""
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                # Replace with QLoRA layer
                qlora_layer = QLoRALayer(
                    module.in_features,
                    module.out_features,
                    self.config,
                    bias=module.bias is not None
                )
                
                # Copy and quantize weights
                qlora_layer.quantize_weight(module.weight.data)
                if module.bias is not None:
                    qlora_layer.bias.data.copy_(module.bias.data)
                
                setattr(self.model, name, qlora_layer)
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   criterion: nn.Module) -> Dict[str, float]:
        """Perform a single training step with all optimizations."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        with self.mixed_precision.autocast():
            outputs = self.model(**batch)
            loss = criterion(outputs, batch.get('labels'))
        
        # Backward pass with gradient scaling
        self.mixed_precision.backward(loss)
        
        # Optimizer step with gradient clipping
        self.mixed_precision.step()
        
        # Update learning rate
        self.lr_scheduler.step(loss.item())
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.lr_scheduler.get_lr(),
            'grad_norm': self.lr_scheduler.compute_gradient_norm(),
            'loss_scale': self.mixed_precision.get_scale()
        }
        
        return metrics
    
    def evaluate_step(self, 
                      batch: Dict[str, torch.Tensor],
                      criterion: nn.Module) -> Dict[str, float]:
        """Perform evaluation step."""
        self.model.eval()
        
        with torch.no_grad():
            with self.mixed_precision.autocast():
                outputs = self.model(**batch)
                loss = criterion(outputs, batch.get('labels'))
        
        return {'eval_loss': loss.item()}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization process."""
        stats = {
            'quantization_method': self.config.method.value,
            'gradient_checkpointing': self.config.enable_gradient_checkpointing,
            'current_lr': self.lr_scheduler.get_lr(),
            'loss_scale': self.mixed_precision.get_scale(),
            'landscape_analysis': self.lr_scheduler.analyze_loss_landscape()
        }
        
        if self.quantizer:
            stats['quantization_stats'] = {
                'weight_stats': self.quantizer.weight_stats,
                'activation_stats': self.quantizer.activation_stats
            }
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Save optimization state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
            'mixed_precision_state': self.mixed_precision.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load optimization state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        self.mixed_precision.load_state_dict(checkpoint['mixed_precision_state'])
        logger.info(f"Checkpoint loaded from {path}")


# Factory function for easy integration
def create_optimization_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    quantization_method: str = "none",
    **kwargs
) -> OptimizationEngine:
    """Create optimization engine with specified configuration."""
    
    method = QuantizationMethod(quantization_method.lower())
    config = QuantizationConfig(method=method, **kwargs)
    
    return OptimizationEngine(model, optimizer, config)


# Utility functions for integration with existing training scripts
def apply_quantization_to_model(
    model: nn.Module,
    method: str = "qlora",
    bits: int = 4,
    **kwargs
) -> nn.Module:
    """Apply quantization to an existing model."""
    
    config = QuantizationConfig(
        method=QuantizationMethod(method.lower()),
        bits=bits,
        **kwargs
    )
    
    engine = OptimizationEngine(model, torch.optim.Adam(model.parameters()), config)
    return engine.model


def setup_adaptive_scheduler(
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> AdaptiveLRScheduler:
    """Setup adaptive learning rate scheduler."""
    return AdaptiveLRScheduler(optimizer, **kwargs)


def enable_mixed_precision(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> MixedPrecisionTrainer:
    """Enable mixed precision training."""
    config = QuantizationConfig(compute_dtype=torch.float16)
    return MixedPrecisionTrainer(model, optimizer, config, **kwargs)