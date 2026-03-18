"""
forge Model Distillation Pipeline
Automated knowledge distillation system with architecture search, progressive distillation, and deployment optimization.
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from forge.extras.logging import get_logger
from forge.hparams import DataArguments, ModelArguments, TrainingArguments
from forge.model import load_model, load_tokenizer

logger = get_logger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation pipeline."""
    
    # Architecture search parameters
    target_model_size: str = "small"  # tiny, small, medium, base
    target_params_ratio: float = 0.25  # Target parameter ratio relative to teacher
    target_layers_ratio: float = 0.5  # Target layer ratio relative to teacher
    target_hidden_ratio: float = 0.75  # Target hidden dimension ratio
    
    # Distillation parameters
    temperature: float = 2.0
    distillation_weight: float = 0.7
    student_weight: float = 0.3
    progressive_distillation: bool = True
    progressive_stages: int = 3
    layer_mapping_strategy: str = "uniform"  # uniform, attention, gradient
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Deployment optimization
    quantization_bits: Optional[int] = None  # 4, 8, or None
    pruning_ratio: float = 0.0
    export_format: str = "huggingface"  # huggingface, onnx, tflite
    
    # Search constraints
    max_search_time: int = 3600  # Maximum search time in seconds
    search_trials: int = 10  # Number of architecture search trials
    hardware_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_sizes = ["tiny", "small", "medium", "base"]
        if self.target_model_size not in valid_sizes:
            raise ValueError(f"target_model_size must be one of {valid_sizes}")
        
        if not 0 < self.target_params_ratio <= 1:
            raise ValueError("target_params_ratio must be between 0 and 1")
        
        if not 0 < self.target_layers_ratio <= 1:
            raise ValueError("target_layers_ratio must be between 0 and 1")
        
        if not 0 < self.target_hidden_ratio <= 1:
            raise ValueError("target_hidden_ratio must be between 0 and 1")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if not 0 <= self.distillation_weight <= 1:
            raise ValueError("distillation_weight must be between 0 and 1")
        
        if self.quantization_bits and self.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4, 8, or None")


class StudentArchitectureSearch:
    """Automatic student architecture search based on teacher model and constraints."""
    
    def __init__(self, teacher_model: PreTrainedModel, config: DistillationConfig):
        self.teacher_model = teacher_model
        self.config = config
        self.teacher_config = teacher_model.config
        
        # Extract teacher architecture information
        self.teacher_params = sum(p.numel() for p in teacher_model.parameters())
        self.teacher_layers = self._get_num_layers()
        self.teacher_hidden = self.teacher_config.hidden_size
        self.teacher_intermediate = getattr(
            self.teacher_config, "intermediate_size", self.teacher_hidden * 4
        )
        
        logger.info(f"Teacher model: {self.teacher_params:,} parameters")
        logger.info(f"Teacher architecture: {self.teacher_layers} layers, "
                   f"hidden={self.teacher_hidden}, intermediate={self.teacher_intermediate}")
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers in teacher model."""
        if hasattr(self.teacher_config, "num_hidden_layers"):
            return self.teacher_config.num_hidden_layers
        elif hasattr(self.teacher_config, "n_layer"):
            return self.teacher_config.n_layer
        else:
            # Try to infer from model structure
            for name, module in self.teacher_model.named_modules():
                if "layers" in name or "h" in name:
                    # Count layer modules
                    layer_count = 0
                    for child_name, _ in module.named_children():
                        if child_name.isdigit():
                            layer_count += 1
                    if layer_count > 0:
                        return layer_count
            raise ValueError("Could not determine number of layers in teacher model")
    
    def search_architectures(self) -> List[Dict[str, Any]]:
        """Search for optimal student architectures."""
        architectures = []
        
        # Size-based architecture generation
        size_configs = {
            "tiny": {"layers_ratio": 0.25, "hidden_ratio": 0.5, "params_ratio": 0.1},
            "small": {"layers_ratio": 0.5, "hidden_ratio": 0.75, "params_ratio": 0.25},
            "medium": {"layers_ratio": 0.75, "hidden_ratio": 0.875, "params_ratio": 0.5},
            "base": {"layers_ratio": 1.0, "hidden_ratio": 1.0, "params_ratio": 1.0},
        }
        
        target_config = size_configs[self.config.target_model_size]
        
        # Generate candidate architectures
        for trial in range(self.config.search_trials):
            # Add some randomness to search
            layer_ratio = target_config["layers_ratio"] * (0.9 + 0.2 * torch.rand(1).item())
            hidden_ratio = target_config["hidden_ratio"] * (0.95 + 0.1 * torch.rand(1).item())
            
            # Calculate student dimensions
            student_layers = max(1, int(self.teacher_layers * layer_ratio))
            student_hidden = max(64, int(self.teacher_hidden * hidden_ratio))
            
            # Ensure hidden size is multiple of 64 for efficiency
            student_hidden = (student_hidden + 63) // 64 * 64
            
            # Calculate intermediate size (typically 4x hidden)
            student_intermediate = student_hidden * 4
            
            # Estimate parameter count
            student_params = self._estimate_parameters(
                student_layers, student_hidden, student_intermediate
            )
            
            architecture = {
                "trial": trial,
                "num_layers": student_layers,
                "hidden_size": student_hidden,
                "intermediate_size": student_intermediate,
                "estimated_params": student_params,
                "params_ratio": student_params / self.teacher_params,
                "layer_ratio": layer_ratio,
                "hidden_ratio": hidden_ratio,
            }
            
            architectures.append(architecture)
            logger.info(f"Trial {trial}: {student_layers} layers, "
                       f"hidden={student_hidden}, params≈{student_params:,}")
        
        # Sort by parameter efficiency (closest to target)
        target_params = self.teacher_params * self.config.target_params_ratio
        architectures.sort(key=lambda x: abs(x["estimated_params"] - target_params))
        
        return architectures
    
    def _estimate_parameters(self, num_layers: int, hidden_size: int, 
                           intermediate_size: int) -> int:
        """Estimate parameter count for student architecture."""
        # Simplified parameter estimation for transformer model
        # This is an approximation - actual count depends on specific architecture
        
        # Embedding layer
        vocab_size = getattr(self.teacher_config, "vocab_size", 32000)
        embedding_params = vocab_size * hidden_size
        
        # Transformer layers
        # Each layer has: attention (Q, K, V, O), FFN (up, down), layer norms
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        ffn_params = 2 * hidden_size * intermediate_size  # up and down projections
        layer_norm_params = 2 * hidden_size  # Two layer norms per layer
        
        layer_params = attention_params + ffn_params + layer_norm_params
        total_layer_params = num_layers * layer_params
        
        # Output layer
        output_params = vocab_size * hidden_size
        
        total_params = embedding_params + total_layer_params + output_params
        
        return total_params
    
    def create_student_config(self, architecture: Dict[str, Any]) -> PretrainedConfig:
        """Create student model configuration based on selected architecture."""
        # Start with teacher config and modify
        student_config = self.teacher_config.__class__.from_dict(
            self.teacher_config.to_dict()
        )
        
        # Update architecture parameters
        if hasattr(student_config, "num_hidden_layers"):
            student_config.num_hidden_layers = architecture["num_layers"]
        elif hasattr(student_config, "n_layer"):
            student_config.n_layer = architecture["num_layers"]
        
        if hasattr(student_config, "hidden_size"):
            student_config.hidden_size = architecture["hidden_size"]
        elif hasattr(student_config, "n_embd"):
            student_config.n_embd = architecture["hidden_size"]
        
        if hasattr(student_config, "intermediate_size"):
            student_config.intermediate_size = architecture["intermediate_size"]
        
        # Update attention heads if needed (ensure divisible by hidden size)
        if hasattr(student_config, "num_attention_heads"):
            # Keep proportional to hidden size
            head_ratio = student_config.num_attention_heads / self.teacher_hidden
            new_heads = max(1, int(architecture["hidden_size"] * head_ratio))
            # Ensure hidden size is divisible by number of heads
            while architecture["hidden_size"] % new_heads != 0 and new_heads > 1:
                new_heads -= 1
            student_config.num_attention_heads = new_heads
        
        logger.info(f"Student config: {architecture['num_layers']} layers, "
                   f"hidden={architecture['hidden_size']}, "
                   f"intermediate={architecture['intermediate_size']}")
        
        return student_config


class LayerWiseDistillationLoss(nn.Module):
    """Layer-wise knowledge distillation loss with progressive strategy."""
    
    def __init__(self, temperature: float = 2.0, 
                 layer_mapping: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.temperature = temperature
        self.layer_mapping = layer_mapping
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                student_hidden_states: List[torch.Tensor],
                teacher_hidden_states: List[torch.Tensor],
                stage: int = 0) -> torch.Tensor:
        """Calculate distillation loss."""
        total_loss = 0.0
        
        # Logit distillation loss
        if "logits" in student_outputs and "logits" in teacher_outputs:
            student_logits = student_outputs["logits"] / self.temperature
            teacher_logits = teacher_outputs["logits"] / self.temperature
            
            # Soft target loss
            logit_loss = self.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1)
            ) * (self.temperature ** 2)
            
            total_loss += logit_loss
        
        # Hidden state distillation loss (layer-wise)
        if student_hidden_states and teacher_hidden_states:
            if self.layer_mapping:
                # Use predefined layer mapping
                for student_idx, teacher_idx in self.layer_mapping:
                    if student_idx < len(student_hidden_states) and \
                       teacher_idx < len(teacher_hidden_states):
                        student_hidden = student_hidden_states[student_idx]
                        teacher_hidden = teacher_hidden_states[teacher_idx]
                        
                        # Project teacher hidden states to student dimension if needed
                        if student_hidden.size(-1) != teacher_hidden.size(-1):
                            # Simple linear projection (in practice, use learned projection)
                            teacher_hidden = teacher_hidden[..., :student_hidden.size(-1)]
                        
                        hidden_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
                        total_loss += hidden_loss
            else:
                # Uniform mapping: map student layers to evenly spaced teacher layers
                num_student_layers = len(student_hidden_states)
                num_teacher_layers = len(teacher_hidden_states)
                
                for i, student_hidden in enumerate(student_hidden_states):
                    # Map student layer i to teacher layer j
                    teacher_idx = int(i * num_teacher_layers / num_student_layers)
                    teacher_hidden = teacher_hidden_states[teacher_idx]
                    
                    # Project if needed
                    if student_hidden.size(-1) != teacher_hidden.size(-1):
                        teacher_hidden = teacher_hidden[..., :student_hidden.size(-1)]
                    
                    hidden_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
                    total_loss += hidden_loss
        
        return total_loss


class ProgressiveDistiller:
    """Progressive distillation with layer-wise training strategy."""
    
    def __init__(self, teacher_model: PreTrainedModel, student_model: PreTrainedModel,
                 config: DistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.teacher_model.eval()
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Create layer mapping
        self.layer_mapping = self._create_layer_mapping()
        
        # Distillation loss
        self.distill_loss_fn = LayerWiseDistillationLoss(
            temperature=config.temperature,
            layer_mapping=self.layer_mapping
        )
    
    def _create_layer_mapping(self) -> List[Tuple[int, int]]:
        """Create mapping between student and teacher layers."""
        teacher_layers = self._get_model_layers(self.teacher_model)
        student_layers = self._get_model_layers(self.student_model)
        
        num_teacher = len(teacher_layers)
        num_student = len(student_layers)
        
        mapping = []
        
        if self.config.layer_mapping_strategy == "uniform":
            # Uniform mapping: map student layers to evenly spaced teacher layers
            for i in range(num_student):
                teacher_idx = int(i * num_teacher / num_student)
                mapping.append((i, teacher_idx))
        
        elif self.config.layer_mapping_strategy == "attention":
            # Attention-based mapping: map based on attention pattern similarity
            # This would require computing attention patterns - simplified here
            for i in range(num_student):
                teacher_idx = int(i * num_teacher / num_student)
                mapping.append((i, teacher_idx))
        
        elif self.config.layer_mapping_strategy == "gradient":
            # Gradient-based mapping: map based on gradient similarity
            # This would require gradient computation - simplified here
            for i in range(num_student):
                teacher_idx = int(i * num_teacher / num_student)
                mapping.append((i, teacher_idx))
        
        logger.info(f"Layer mapping ({self.config.layer_mapping_strategy}): {mapping}")
        return mapping
    
    def _get_model_layers(self, model: PreTrainedModel) -> List[nn.Module]:
        """Get transformer layers from model."""
        layers = []
        
        # Common patterns for transformer layers
        for name, module in model.named_modules():
            if any(layer_name in name.lower() for layer_name in 
                   ["layer", "block", "h", "layers"]):
                if "attention" not in name and "ffn" not in name:
                    # This is likely a transformer layer
                    if hasattr(module, "attention") or hasattr(module, "mlp"):
                        layers.append(module)
        
        if not layers:
            # Try to find layers by looking at model structure
            for name, module in model.named_modules():
                if name.count(".") == 1:  # Top-level modules
                    if isinstance(module, nn.ModuleList):
                        layers.extend(module)
        
        return layers
    
    def distill_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                     stage: int = 0) -> float:
        """Perform one epoch of progressive distillation."""
        self.student_model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.student_model.device) for k, v in batch.items()}
            
            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    **batch, 
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Forward pass through student
            student_outputs = self.student_model(
                **batch,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Calculate loss
            loss = self.distill_loss_fn(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                student_hidden_states=student_outputs.hidden_states,
                teacher_hidden_states=teacher_outputs.hidden_states,
                stage=stage
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Stage {stage}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)


class DeploymentOptimizer:
    """Optimize distilled model for deployment."""
    
    def __init__(self, model: PreTrainedModel, config: DistillationConfig):
        self.model = model
        self.config = config
    
    def quantize_model(self) -> PreTrainedModel:
        """Apply quantization to the model."""
        if not self.config.quantization_bits:
            return self.model
        
        logger.info(f"Applying {self.config.quantization_bits}-bit quantization")
        
        if self.config.quantization_bits == 8:
            # 8-bit quantization using bitsandbytes
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                # Re-load model with quantization
                model_path = self.model.config._name_or_path
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                return quantized_model
                
            except ImportError:
                logger.warning("bitsandbytes not installed. Skipping quantization.")
                return self.model
        
        elif self.config.quantization_bits == 4:
            # 4-bit quantization
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                model_path = self.model.config._name_or_path
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                return quantized_model
                
            except ImportError:
                logger.warning("bitsandbytes not installed. Skipping quantization.")
                return self.model
        
        return self.model
    
    def prune_model(self) -> PreTrainedModel:
        """Apply pruning to the model."""
        if self.config.pruning_ratio <= 0:
            return self.model
        
        logger.info(f"Applying pruning with ratio {self.config.pruning_ratio}")
        
        # Simple magnitude-based pruning
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune weights based on magnitude
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), self.config.pruning_ratio)
                mask = torch.abs(weight) > threshold
                module.weight.data = weight * mask.float()
        
        return self.model
    
    def export_model(self, output_dir: str) -> str:
        """Export optimized model in specified format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.export_format == "huggingface":
            # Save in Hugging Face format
            self.model.save_pretrained(output_path)
            
            # Save tokenizer if available
            if hasattr(self.model, "tokenizer"):
                self.model.tokenizer.save_pretrained(output_path)
            
            logger.info(f"Model exported to {output_path} in Hugging Face format")
            return str(output_path)
        
        elif self.config.export_format == "onnx":
            # Export to ONNX format
            try:
                import onnx
                from transformers.onnx import export
                
                # Create dummy input
                dummy_input = torch.randint(0, 1000, (1, 128))
                
                onnx_path = output_path / "model.onnx"
                export(
                    preprocessor=self.model.tokenizer if hasattr(self.model, "tokenizer") else None,
                    model=self.model,
                    output=onnx_path,
                    opset=13
                )
                
                logger.info(f"Model exported to {onnx_path} in ONNX format")
                return str(onnx_path)
                
            except ImportError:
                logger.warning("ONNX export dependencies not installed.")
                return self.export_model(output_dir)  # Fallback to Hugging Face
        
        elif self.config.export_format == "tflite":
            # Export to TensorFlow Lite format
            logger.warning("TFLite export not yet implemented. Using Hugging Face format.")
            return self.export_model(output_dir)
        
        else:
            raise ValueError(f"Unsupported export format: {self.config.export_format}")


class DistillationPipeline:
    """Main pipeline for automated knowledge distillation."""
    
    def __init__(self, teacher_model_name: str, distillation_config: DistillationConfig,
                 training_args: Optional[TrainingArguments] = None):
        self.teacher_model_name = teacher_model_name
        self.config = distillation_config
        self.training_args = training_args or TrainingArguments(
            output_dir="./distilled_model",
            num_train_epochs=distillation_config.num_train_epochs,
            per_device_train_batch_size=distillation_config.per_device_train_batch_size,
            learning_rate=distillation_config.learning_rate,
            warmup_ratio=distillation_config.warmup_ratio,
            weight_decay=distillation_config.weight_decay,
            logging_steps=10,
            save_strategy="epoch",
        )
        
        # Load teacher model and tokenizer
        logger.info(f"Loading teacher model: {teacher_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Initialize components
        self.arch_search = StudentArchitectureSearch(self.teacher_model, self.config)
        self.student_model = None
        self.distiller = None
        self.optimizer = None
    
    def run(self, train_dataloader: DataLoader, 
            output_dir: str = "./distilled_model") -> str:
        """Run the complete distillation pipeline."""
        logger.info("Starting distillation pipeline")
        
        # Step 1: Architecture Search
        logger.info("Step 1: Searching for optimal student architecture")
        architectures = self.arch_search.search_architectures()
        
        if not architectures:
            raise RuntimeError("No suitable student architectures found")
        
        # Select best architecture
        best_arch = architectures[0]
        logger.info(f"Selected architecture: {best_arch}")
        
        # Step 2: Create Student Model
        logger.info("Step 2: Creating student model")
        student_config = self.arch_search.create_student_config(best_arch)
        self.student_model = AutoModelForCausalLM.from_config(student_config)
        
        # Initialize student with teacher weights where possible
        self._initialize_student_weights()
        
        # Step 3: Progressive Distillation
        if self.config.progressive_distillation:
            logger.info("Step 3: Running progressive distillation")
            self.distiller = ProgressiveDistiller(
                self.teacher_model, self.student_model, self.config
            )
            
            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Progressive training in stages
            for stage in range(self.config.progressive_stages):
                logger.info(f"Distillation stage {stage + 1}/{self.config.progressive_stages}")
                loss = self.distiller.distill_epoch(train_dataloader, optimizer, stage)
                logger.info(f"Stage {stage + 1} completed. Average loss: {loss:.4f}")
        else:
            # Standard distillation
            logger.info("Step 3: Running standard distillation")
            self._run_standard_distillation(train_dataloader)
        
        # Step 4: Deployment Optimization
        logger.info("Step 4: Optimizing for deployment")
        self.optimizer = DeploymentOptimizer(self.student_model, self.config)
        
        # Apply quantization
        if self.config.quantization_bits:
            self.student_model = self.optimizer.quantize_model()
        
        # Apply pruning
        if self.config.pruning_ratio > 0:
            self.student_model = self.optimizer.prune_model()
        
        # Step 5: Export Model
        logger.info("Step 5: Exporting optimized model")
        export_path = self.optimizer.export_model(output_dir)
        
        logger.info(f"Distillation pipeline completed successfully!")
        logger.info(f"Student model saved to: {export_path}")
        
        return export_path
    
    def _initialize_student_weights(self):
        """Initialize student model with teacher weights where dimensions match."""
        teacher_state = self.teacher_model.state_dict()
        student_state = self.student_model.state_dict()
        
        initialized = 0
        total = len(student_state)
        
        for name, param in student_state.items():
            if name in teacher_state:
                teacher_param = teacher_state[name]
                
                # Check if dimensions match
                if param.shape == teacher_param.shape:
                    student_state[name] = teacher_param.clone()
                    initialized += 1
                else:
                    # Try to initialize with subset of teacher weights
                    if len(param.shape) == len(teacher_param.shape):
                        # Create slice objects for each dimension
                        slices = tuple(slice(0, min(s, t)) for s, t in 
                                     zip(param.shape, teacher_param.shape))
                        student_state[name][slices] = teacher_param[slices].clone()
                        initialized += 1
        
        self.student_model.load_state_dict(student_state)
        logger.info(f"Initialized {initialized}/{total} student parameters from teacher")
    
    def _run_standard_distillation(self, dataloader: DataLoader):
        """Run standard (non-progressive) distillation."""
        # Create a simple trainer for distillation
        class DistillationTrainer(Trainer):
            def __init__(self, teacher_model, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.teacher_model.eval()
                self.distill_loss_fn = LayerWiseDistillationLoss(
                    temperature=self.args.temperature
                )
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # Forward pass through teacher
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                # Forward pass through student
                student_outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Calculate distillation loss
                loss = self.distill_loss_fn(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    student_hidden_states=student_outputs.hidden_states,
                    teacher_hidden_states=teacher_outputs.hidden_states
                )
                
                return (loss, student_outputs) if return_outputs else loss
        
        # Update training args with distillation-specific parameters
        self.training_args.temperature = self.config.temperature
        
        # Create trainer
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            model=self.student_model,
            args=self.training_args,
            train_dataset=dataloader.dataset,
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        logger.info("Standard distillation completed")


def create_distillation_pipeline(
    teacher_model_name: str,
    target_model_size: str = "small",
    target_params_ratio: float = 0.25,
    output_dir: str = "./distilled_model",
    **kwargs
) -> DistillationPipeline:
    """Factory function to create a distillation pipeline with common configurations."""
    
    # Create distillation config
    distillation_config = DistillationConfig(
        target_model_size=target_model_size,
        target_params_ratio=target_params_ratio,
        **kwargs
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=distillation_config.num_train_epochs,
        per_device_train_batch_size=distillation_config.per_device_train_batch_size,
        learning_rate=distillation_config.learning_rate,
        warmup_ratio=distillation_config.warmup_ratio,
        weight_decay=distillation_config.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )
    
    # Create pipeline
    pipeline = DistillationPipeline(
        teacher_model_name=teacher_model_name,
        distillation_config=distillation_config,
        training_args=training_args
    )
    
    return pipeline


# Example usage
if __name__ == "__main__":
    # Example: Distill a large model into a smaller one
    pipeline = create_distillation_pipeline(
        teacher_model_name="meta-llama/Llama-2-7b-hf",
        target_model_size="small",
        target_params_ratio=0.25,
        output_dir="./distilled_llama_small",
        progressive_distillation=True,
        progressive_stages=3,
        quantization_bits=8,
        export_format="huggingface"
    )
    
    # Note: In practice, you would provide a proper DataLoader
    # This is just a demonstration of the API
    print("Distillation pipeline created successfully!")
    print("To run: pipeline.run(train_dataloader, output_dir)")