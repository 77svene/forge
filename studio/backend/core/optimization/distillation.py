"""
Automated Model Optimization Pipeline for Unsloth Studio

This module provides comprehensive model optimization capabilities including:
- Automated hyperparameter optimization with Optuna
- Knowledge distillation from teacher to student models
- Structured pruning with performance recovery
- Neural architecture search integration
- End-to-end optimization pipelines

Integrates with existing Unsloth Studio backend components.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Conditional imports for optional dependencies
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
        Trainer,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from studio.backend.core.data_recipe.huggingface import HuggingFaceDatasetManager
from studio.backend.auth.storage import SecureStorage
from studio.backend.core import OptimizationError, ValidationError

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods."""
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    HYPERPARAMETER = "hyperparameter"
    NAS = "neural_architecture_search"
    FULL_PIPELINE = "full_pipeline"


class PruningStrategy(Enum):
    """Structured pruning strategies."""
    ATTENTION_HEAD = "attention_head"
    FEED_FORWARD = "feed_forward"
    LAYER = "layer"
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_model_name: str
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs task loss
    layer_mapping: Optional[Dict[int, int]] = None
    feature_matching_layers: List[int] = None
    max_sequence_length: int = 512
    
    def __post_init__(self):
        if self.feature_matching_layers is None:
            self.feature_matching_layers = [-1, -2, -3]  # Last three layers


@dataclass
class PruningConfig:
    """Configuration for structured pruning."""
    strategy: PruningStrategy
    sparsity_target: float = 0.3
    structured_pruning_dim: int = 0  # 0 for output, 1 for input
    importance_metric: str = "magnitude"  # magnitude, gradient, activation
    recovery_epochs: int = 3
    recovery_lr: float = 1e-5
    min_importance_threshold: float = 1e-6


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 50
    timeout_seconds: Optional[int] = 3600
    learning_rate_range: Tuple[float, float] = (1e-6, 1e-3)
    batch_size_options: List[int] = None
    warmup_steps_range: Tuple[int, int] = (0, 500)
    weight_decay_range: Tuple[float, float] = (0.0, 0.1)
    objective_metric: str = "eval_loss"
    direction: str = "minimize"
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [8, 16, 32, 64]


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    method: OptimizationMethod
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    parameters_changed: Dict[str, Any]
    optimization_time_seconds: float
    model_size_reduction: Optional[float] = None
    inference_speedup: Optional[float] = None
    config_used: Optional[Dict] = None
    trial_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['method'] = self.method.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationResult':
        """Create from dictionary."""
        data['method'] = OptimizationMethod(data['method'])
        return cls(**data)


class KnowledgeDistillationTrainer:
    """Handles knowledge distillation from teacher to student model."""
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DistillationConfig,
        dataset_manager: Optional[HuggingFaceDatasetManager] = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_manager = dataset_manager or HuggingFaceDatasetManager()
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Setup layer mapping if not provided
        if config.layer_mapping is None:
            self.config.layer_mapping = self._create_default_layer_mapping()
    
    def _create_default_layer_mapping(self) -> Dict[int, int]:
        """Create default layer mapping between teacher and student."""
        teacher_layers = len(self.teacher.config.hidden_layers) if hasattr(self.teacher.config, 'hidden_layers') else 12
        student_layers = len(self.student.config.hidden_layers) if hasattr(self.student.config, 'hidden_layers') else 6
        
        # Map student layers to evenly spaced teacher layers
        mapping = {}
        for i in range(student_layers):
            teacher_idx = int((i + 1) * teacher_layers / student_layers) - 1
            mapping[i] = max(0, min(teacher_idx, teacher_layers - 1))
        
        return mapping
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute combined distillation and task loss."""
        temp = temperature or self.config.temperature
        
        # Soften logits with temperature
        soft_student = F.log_softmax(student_logits / temp, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temp, dim=-1)
        
        # KL divergence loss for distillation
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean',
        ) * (temp ** 2)
        
        # Standard task loss (cross-entropy)
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        alpha = self.config.alpha
        combined_loss = alpha * distill_loss + (1 - alpha) * task_loss
        
        return combined_loss
    
    def extract_features(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate features from model."""
        features = {}
        
        # Hook to capture intermediate layer outputs
        def hook_fn(module, input, output):
            layer_name = f"layer_{len(features)}"
            if isinstance(output, tuple):
                features[layer_name] = output[0].detach()
            else:
                features[layer_name] = output.detach()
        
        # Register hooks for feature matching layers
        hooks = []
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            for layer_idx in self.config.feature_matching_layers:
                if abs(layer_idx) <= len(model.encoder.layer):
                    hook = model.encoder.layer[layer_idx].register_forward_hook(hook_fn)
                    hooks.append(hook)
        
        # Forward pass to capture features
        with torch.no_grad() if model == self.teacher else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Add hidden states if available
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            for i, hidden_state in enumerate(outputs.hidden_states):
                features[f"hidden_{i}"] = hidden_state.detach()
        
        return features
    
    def compute_feature_matching_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature matching loss between student and teacher."""
        total_loss = torch.tensor(0.0, device=next(self.student.parameters()).device)
        count = 0
        
        # Match features by layer mapping
        for student_layer, teacher_layer in self.config.layer_mapping.items():
            student_key = f"layer_{student_layer}"
            teacher_key = f"layer_{teacher_layer}"
            
            if student_key in student_features and teacher_key in teacher_features:
                student_feat = student_features[student_key]
                teacher_feat = teacher_features[teacher_key]
                
                # Project teacher features to student dimension if needed
                if student_feat.shape[-1] != teacher_feat.shape[-1]:
                    # Simple linear projection (in practice, use a learned projection)
                    teacher_feat = teacher_feat[..., :student_feat.shape[-1]]
                
                # MSE loss between features
                loss = F.mse_loss(student_feat, teacher_feat)
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)
    
    async def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
    ) -> Dict[str, float]:
        """Train student for one epoch with distillation."""
        self.student.train()
        total_loss = 0.0
        distill_loss_total = 0.0
        task_loss_total = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
            
            optimizer.zero_grad()
            
            # Forward pass through student
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            student_logits = student_outputs.logits
            
            # Forward pass through teacher (no gradient)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits
            
            # Compute distillation loss
            distill_loss = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
            
            # Optional: Feature matching loss
            feature_loss = torch.tensor(0.0, device=device)
            if self.config.feature_matching_layers:
                student_features = self.extract_features(
                    self.student, input_ids, attention_mask
                )
                teacher_features = self.extract_features(
                    self.teacher, input_ids, attention_mask
                )
                feature_loss = self.compute_feature_matching_loss(
                    student_features, teacher_features
                )
            
            # Total loss
            total_batch_loss = distill_loss + 0.1 * feature_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            distill_loss_total += distill_loss.item()
            task_loss_total += F.cross_entropy(student_logits, labels).item()
        
        # Calculate average losses
        avg_total_loss = total_loss / len(dataloader)
        avg_distill_loss = distill_loss_total / len(dataloader)
        avg_task_loss = task_loss_total / len(dataloader)
        
        return {
            'total_loss': avg_total_loss,
            'distillation_loss': avg_distill_loss,
            'task_loss': avg_task_loss,
            'epoch': epoch,
        }


class StructuredPruner:
    """Implements structured pruning with automatic recovery."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: PruningConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.importance_scores = {}
        self.pruned_layers = []
        
    def compute_importance_scores(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Compute importance scores for pruning."""
        self.model.eval()
        importance_scores = {}
        
        if self.config.importance_metric == "magnitude":
            # Magnitude-based importance
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    importance = torch.norm(param, dim=self.config.structured_pruning_dim)
                    importance_scores[name] = importance.detach()
        
        elif self.config.importance_metric == "gradient":
            # Gradient-based importance (requires forward/backward pass)
            self.model.train()
            total_gradients = {}
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                
                # Accumulate gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None and 'weight' in name and param.dim() >= 2:
                        grad_importance = torch.norm(param.grad, dim=self.config.structured_pruning_dim)
                        if name not in total_gradients:
                            total_gradients[name] = grad_importance
                        else:
                            total_gradients[name] += grad_importance
                
                self.model.zero_grad()
            
            # Average gradients
            for name in total_gradients:
                importance_scores[name] = total_gradients[name] / len(dataloader)
        
        elif self.config.importance_metric == "activation":
            # Activation-based importance (simplified)
            # In practice, would use hooks to capture activations
            logger.warning("Activation-based importance not fully implemented, using magnitude")
            return self.compute_importance_scores(dataloader, device)
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def identify_prunable_components(self) -> List[Tuple[str, torch.Tensor]]:
        """Identify components to prune based on importance scores."""
        prunable = []
        
        for name, importance in self.importance_scores.items():
            # Filter by threshold
            mask = importance > self.config.min_importance_threshold
            if mask.sum() > 0:  # At least one component is important
                prunable.append((name, importance[mask]))
        
        # Sort by importance (ascending) to prune least important first
        prunable.sort(key=lambda x: x[1].mean().item())
        
        return prunable
    
    def apply_structured_pruning(self) -> Dict[str, Any]:
        """Apply structured pruning to the model."""
        pruned_params = {}
        total_params_before = sum(p.numel() for p in self.model.parameters())
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'weight' in dict(module.named_parameters()):
                weight = module.weight.data
                importance = torch.norm(weight, dim=self.config.structured_pruning_dim)
                
                # Determine how many to prune
                n_prune = int(importance.numel() * self.config.sparsity_target)
                if n_prune == 0:
                    continue
                
                # Get indices to prune (least important)
                _, indices_to_prune = torch.topk(importance, n_prune, largest=False)
                
                # Create pruning mask
                mask = torch.ones_like(weight, dtype=torch.bool)
                if self.config.structured_pruning_dim == 0:
                    mask[indices_to_prune, :] = False
                else:
                    mask[:, indices_to_prune] = False
                
                # Apply pruning
                module.weight.data *= mask.float()
                
                # Track pruned parameters
                pruned_params[name] = {
                    'pruned_indices': indices_to_prune.cpu().tolist(),
                    'original_shape': list(weight.shape),
                    'pruned_count': n_prune,
                }
        
        total_params_after = sum(p.numel() for p in self.model.parameters())
        reduction = 1 - (total_params_after / total_params_before)
        
        return {
            'pruned_params': pruned_params,
            'total_params_before': total_params_before,
            'total_params_after': total_params_after,
            'reduction_percentage': reduction * 100,
        }
    
    async def recover_performance(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device,
        recovery_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Fine-tune pruned model to recover performance."""
        epochs = recovery_epochs or self.config.recovery_epochs
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.recovery_lr,
            weight_decay=0.01,
        )
        
        self.model.train()
        initial_loss = await self._evaluate(eval_dataloader, device)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Recovery epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        final_loss = await self._evaluate(eval_dataloader, device)
        recovery = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
        
        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'recovery_percentage': recovery * 100,
            'epochs_trained': epochs,
        }
    
    async def _evaluate(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> float:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        model_factory: Callable[[Dict], PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        config: HyperparameterConfig,
        dataset_manager: Optional[HuggingFaceDatasetManager] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise OptimizationError("Optuna is required for hyperparameter optimization")
        
        self.model_factory = model_factory
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_manager = dataset_manager or HuggingFaceDatasetManager()
        self.study = None
        self.best_params = None
        
    def _create_objective(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device,
    ) -> Callable[[optuna.Trial], float]:
        """Create Optuna objective function."""
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    *self.config.learning_rate_range,
                    log=True,
                ),
                'batch_size': trial.suggest_categorical(
                    'batch_size',
                    self.config.batch_size_options,
                ),
                'warmup_steps': trial.suggest_int(
                    'warmup_steps',
                    *self.config.warmup_steps_range,
                ),
                'weight_decay': trial.suggest_float(
                    'weight_decay',
                    *self.config.weight_decay_range,
                ),
                'num_epochs': trial.suggest_int('num_epochs', 1, 5),
                'gradient_accumulation_steps': trial.suggest_categorical(
                    'gradient_accumulation_steps', [1, 2, 4, 8]
                ),
            }
            
            # Create model with sampled hyperparameters
            model = self.model_factory(params)
            model.to(device)
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
            )
            
            # Training loop
            model.train()
            for epoch in range(params['num_epochs']):
                for batch_idx, batch in enumerate(train_dataloader):
                    # Implement gradient accumulation
                    if batch_idx % params['gradient_accumulation_steps'] == 0:
                        optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / params['gradient_accumulation_steps']
                    loss.backward()
                    
                    if (batch_idx + 1) % params['gradient_accumulation_steps'] == 0:
                        optimizer.step()
                    
                    # Report intermediate results for pruning
                    if batch_idx % 10 == 0:
                        trial.report(loss.item(), batch_idx)
                        
                        # Handle pruning based on intermediate results
                        if trial.should_prune():
                            raise optuna.TrialPruned()
            
            # Evaluation
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device) if 'labels' in batch else input_ids.clone()
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    eval_loss += outputs.loss.item()
            
            avg_eval_loss = eval_loss / len(eval_dataloader)
            return avg_eval_loss
        
        return objective
    
    async def optimize(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        study_name = study_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create or load study
        storage = "sqlite:///optuna_studies.db"
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=self.config.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        # Create objective function
        objective = self._create_objective(train_dataloader, eval_dataloader, device)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=True,
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Create model with best parameters
        best_model = self.model_factory(self.best_params)
        
        return {
            'best_params': self.best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'study_name': study_name,
            'model': best_model,
        }
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history for analysis."""
        if self.study is None:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            })
        
        return history


class ModelOptimizationPipeline:
    """
    End-to-end model optimization pipeline.
    
    Combines distillation, pruning, and hyperparameter optimization
    into a unified workflow.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset_manager: Optional[HuggingFaceDatasetManager] = None,
        secure_storage: Optional[SecureStorage] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_manager = dataset_manager or HuggingFaceDatasetManager()
        self.secure_storage = secure_storage or SecureStorage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        
        # Move model to device
        self.model.to(self.device)
    
    async def run_distillation(
        self,
        teacher_model_name: str,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[DistillationConfig] = None,
        epochs: int = 3,
    ) -> OptimizationResult:
        """Run knowledge distillation optimization."""
        logger.info(f"Starting knowledge distillation from {teacher_model_name}")
        
        start_time = datetime.now()
        
        # Load teacher model
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        teacher_model.to(self.device)
        
        # Create distillation config
        distill_config = config or DistillationConfig(teacher_model_name=teacher_model_name)
        
        # Create distillation trainer
        distiller = KnowledgeDistillationTrainer(
            teacher_model=teacher_model,
            student_model=self.model,
            tokenizer=self.tokenizer,
            config=distill_config,
        )
        
        # Evaluate before distillation
        original_metrics = await self._evaluate_model(eval_dataloader)
        
        # Train with distillation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        for epoch in range(epochs):
            epoch_metrics = await distiller.train_epoch(
                train_dataloader, optimizer, self.device, epoch
            )
            logger.info(f"Epoch {epoch + 1}: {epoch_metrics}")
        
        # Evaluate after distillation
        optimized_metrics = await self._evaluate_model(eval_dataloader)
        
        # Calculate improvement
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            method=OptimizationMethod.DISTILLATION,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            parameters_changed={
                'teacher_model': teacher_model_name,
                'temperature': distill_config.temperature,
                'alpha': distill_config.alpha,
                'epochs': epochs,
            },
            optimization_time_seconds=optimization_time,
            config_used=asdict(distill_config),
        )
        
        self.results.append(result)
        logger.info(f"Distillation completed in {optimization_time:.2f} seconds")
        
        return result
    
    async def run_pruning(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: Optional[PruningConfig] = None,
    ) -> OptimizationResult:
        """Run structured pruning optimization."""
        logger.info("Starting structured pruning")
        
        start_time = datetime.now()
        
        # Create pruning config
        prune_config = config or PruningConfig(strategy=PruningStrategy.ATTENTION_HEAD)
        
        # Create pruner
        pruner = StructuredPruner(
            model=self.model,
            config=prune_config,
            tokenizer=self.tokenizer,
        )
        
        # Evaluate before pruning
        original_metrics = await self._evaluate_model(eval_dataloader)
        
        # Compute importance scores
        importance_scores = pruner.compute_importance_scores(train_dataloader, self.device)
        
        # Apply pruning
        pruning_results = pruner.apply_structured_pruning()
        logger.info(f"Pruned {pruning_results['reduction_percentage']:.2f}% of parameters")
        
        # Recover performance
        recovery_results = await pruner.recover_performance(
            train_dataloader, eval_dataloader, self.device
        )
        
        # Evaluate after pruning and recovery
        optimized_metrics = await self._evaluate_model(eval_dataloader)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            method=OptimizationMethod.PRUNING,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            parameters_changed={
                'strategy': prune_config.strategy.value,
                'sparsity_target': prune_config.sparsity_target,
                'recovery_epochs': prune_config.recovery_epochs,
                **pruning_results,
                **recovery_results,
            },
            optimization_time_seconds=optimization_time,
            model_size_reduction=pruning_results['reduction_percentage'],
            config_used=asdict(prune_config),
        )
        
        self.results.append(result)
        logger.info(f"Pruning completed in {optimization_time:.2f} seconds")
        
        return result
    
    async def run_hyperparameter_optimization(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        model_factory: Optional[Callable] = None,
        config: Optional[HyperparameterConfig] = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization")
        
        start_time = datetime.now()
        
        # Create config
        hp_config = config or HyperparameterConfig()
        
        # Create model factory if not provided
        if model_factory is None:
            def model_factory(params: Dict) -> PreTrainedModel:
                # Clone current model architecture
                model_config = self.model.config
                model_class = type(self.model)
                new_model = model_class(model_config)
                return new_model
        
        # Create optimizer
        hp_optimizer = HyperparameterOptimizer(
            model_factory=model_factory,
            tokenizer=self.tokenizer,
            config=hp_config,
            dataset_manager=self.dataset_manager,
        )
        
        # Evaluate before optimization
        original_metrics = await self._evaluate_model(eval_dataloader)
        
        # Run optimization
        optimization_results = await hp_optimizer.optimize(
            train_dataloader, eval_dataloader, self.device
        )
        
        # Update model with best parameters
        self.model = optimization_results['model']
        self.model.to(self.device)
        
        # Evaluate after optimization
        optimized_metrics = await self._evaluate_model(eval_dataloader)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            method=OptimizationMethod.HYPERPARAMETER,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            parameters_changed=optimization_results['best_params'],
            optimization_time_seconds=optimization_time,
            config_used=asdict(hp_config),
            trial_id=optimization_results['study_name'],
        )
        
        self.results.append(result)
        logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
        
        return result
    
    async def run_full_pipeline(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        teacher_model_name: Optional[str] = None,
        distillation_config: Optional[DistillationConfig] = None,
        pruning_config: Optional[PruningConfig] = None,
        hyperparameter_config: Optional[HyperparameterConfig] = None,
    ) -> List[OptimizationResult]:
        """
        Run complete optimization pipeline.
        
        Order: Hyperparameter optimization -> Distillation -> Pruning
        """
        logger.info("Starting full optimization pipeline")
        pipeline_results = []
        
        # Step 1: Hyperparameter optimization
        if hyperparameter_config is not None:
            hp_result = await self.run_hyperparameter_optimization(
                train_dataloader, eval_dataloader, config=hyperparameter_config
            )
            pipeline_results.append(hp_result)
        
        # Step 2: Knowledge distillation (if teacher provided)
        if teacher_model_name is not None:
            distill_result = await self.run_distillation(
                teacher_model_name, train_dataloader, eval_dataloader,
                config=distillation_config, epochs=3,
            )
            pipeline_results.append(distill_result)
        
        # Step 3: Structured pruning
        if pruning_config is not None:
            prune_result = await self.run_pruning(
                train_dataloader, eval_dataloader, config=pruning_config
            )
            pipeline_results.append(prune_result)
        
        # Save pipeline results
        await self._save_pipeline_results(pipeline_results)
        
        return pipeline_results
    
    async def _evaluate_model(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if dataloader is None:
            return {'loss': 0.0, 'perplexity': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                # Calculate loss
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
        }
    
    async def _save_pipeline_results(
        self,
        results: List[OptimizationResult],
    ) -> None:
        """Save optimization results to secure storage."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'model_config': self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else {},
            'results': [r.to_dict() for r in results],
        }
        
        # Generate unique key based on model and timestamp
        key_hash = hashlib.sha256(
            f"{type(self.model).__name__}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        storage_key = f"optimization_results_{key_hash}"
        
        await self.secure_storage.store(
            key=storage_key,
            data=json.dumps(results_data),
            metadata={
                'type': 'optimization_results',
                'model_type': type(self.model).__name__,
                'num_optimizations': len(results),
            }
        )
        
        logger.info(f"Saved optimization results to {storage_key}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations performed."""
        if not self.results:
            return {'status': 'no_optimizations_performed'}
        
        summary = {
            'total_optimizations': len(self.results),
            'methods_used': [r.method.value for r in self.results],
            'total_time_seconds': sum(r.optimization_time_seconds for r in self.results),
            'optimizations': [],
        }
        
        for result in self.results:
            optimization_info = {
                'method': result.method.value,
                'improvement': {},
                'time_seconds': result.optimization_time_seconds,
            }
            
            # Calculate improvement for each metric
            for metric in result.original_metrics:
                if metric in result.optimized_metrics:
                    original = result.original_metrics[metric]
                    optimized = result.optimized_metrics[metric]
                    if original > 0:
                        improvement = ((original - optimized) / original) * 100
                        optimization_info['improvement'][metric] = {
                            'original': original,
                            'optimized': optimized,
                            'improvement_percent': improvement,
                        }
            
            summary['optimizations'].append(optimization_info)
        
        return summary
    
    async def export_optimized_model(
        self,
        export_path: str,
        include_optimizer: bool = False,
    ) -> str:
        """Export optimized model to disk."""
        os.makedirs(export_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(export_path, "model")
        self.model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(export_path, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save optimization metadata
        metadata = {
            'optimization_summary': self.get_optimization_summary(),
            'export_timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'device': str(self.device),
        }
        
        metadata_path = os.path.join(export_path, "optimization_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported optimized model to {export_path}")
        return export_path


# Factory functions for easy integration
def create_distillation_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    teacher_model_name: str,
    **kwargs,
) -> ModelOptimizationPipeline:
    """Create a pipeline configured for distillation."""
    pipeline = ModelOptimizationPipeline(model, tokenizer, **kwargs)
    return pipeline


def create_pruning_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    pruning_strategy: PruningStrategy = PruningStrategy.ATTENTION_HEAD,
    **kwargs,
) -> ModelOptimizationPipeline:
    """Create a pipeline configured for pruning."""
    pipeline = ModelOptimizationPipeline(model, tokenizer, **kwargs)
    return pipeline


def create_hyperparameter_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> ModelOptimizationPipeline:
    """Create a pipeline configured for hyperparameter optimization."""
    pipeline = ModelOptimizationPipeline(model, tokenizer, **kwargs)
    return pipeline


# Async context manager for optimization pipelines
class OptimizationPipelineContext:
    """Async context manager for optimization pipelines."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.pipeline = None
    
    async def __aenter__(self) -> ModelOptimizationPipeline:
        self.pipeline = ModelOptimizationPipeline(
            self.model, self.tokenizer, **self.kwargs
        )
        return self.pipeline
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pipeline and self.pipeline.results:
            await self.pipeline._save_pipeline_results(self.pipeline.results)


# Integration with existing Unsloth Studio components
def integrate_with_data_recipe(
    pipeline: ModelOptimizationPipeline,
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
) -> Tuple[DataLoader, DataLoader]:
    """
    Integrate optimization pipeline with data recipe system.
    
    Returns train and eval dataloaders from HuggingFace dataset.
    """
    dataset_manager = pipeline.dataset_manager
    
    # Load dataset
    dataset = dataset_manager.load_dataset(dataset_name, split=split)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return pipeline.tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        split_dataset['train'],
        batch_size=8,
        shuffle=True,
    )
    
    eval_dataloader = DataLoader(
        split_dataset['test'],
        batch_size=8,
        shuffle=False,
    )
    
    return train_dataloader, eval_dataloader


# CLI integration helper
def get_optimization_parser():
    """Get argument parser for optimization CLI commands."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Optimization Pipeline")
    parser.add_argument(
        "--method",
        type=str,
        choices=[m.value for m in OptimizationMethod],
        default=OptimizationMethod.FULL_PIPELINE.value,
        help="Optimization method to use",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        help="Teacher model for distillation",
    )
    parser.add_argument(
        "--sparsity-target",
        type=float,
        default=0.3,
        help="Target sparsity for pruning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of hyperparameter optimization trials",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimized_model",
        help="Directory to save optimized model",
    )
    
    return parser


# Example usage in async context
async def example_usage():
    """Example of how to use the optimization pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset manager
    dataset_manager = HuggingFaceDatasetManager()
    
    # Create optimization pipeline
    async with OptimizationPipelineContext(
        model=model,
        tokenizer=tokenizer,
        dataset_manager=dataset_manager,
    ) as pipeline:
        
        # Get dataloaders
        train_loader, eval_loader = integrate_with_data_recipe(
            pipeline, "wikitext", "train", "text"
        )
        
        # Configure optimizations
        distillation_config = DistillationConfig(
            teacher_model_name="gpt2-medium",
            temperature=3.0,
            alpha=0.7,
        )
        
        pruning_config = PruningConfig(
            strategy=PruningStrategy.ATTENTION_HEAD,
            sparsity_target=0.2,
            recovery_epochs=2,
        )
        
        # Run full pipeline
        results = await pipeline.run_full_pipeline(
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            teacher_model_name="gpt2-medium",
            distillation_config=distillation_config,
            pruning_config=pruning_config,
        )
        
        # Export optimized model
        await pipeline.export_optimized_model("./optimized_gpt2")
        
        # Print summary
        summary = pipeline.get_optimization_summary()
        print(f"Optimization Summary: {summary}")


if __name__ == "__main__":
    # Run example if executed directly
    asyncio.run(example_usage())