"""
studio/backend/core/optimization/hpo_engine.py

Automated Model Optimization Pipeline for Unsloth Studio
Provides hyperparameter optimization, neural architecture search,
knowledge distillation, and structured pruning with performance recovery.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Conditional imports for optional dependencies
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from forge import FastLanguageModel
    from forge.models import get_peft_model
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from studio.backend.core.data_recipe.huggingface import HuggingFaceDataset
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Supported optimization methods."""
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class OptimizationConfig:
    """Configuration for optimization pipeline."""
    method: OptimizationMethod = OptimizationMethod.HYPERPARAMETER
    n_trials: int = 50
    timeout_seconds: Optional[int] = 3600
    study_name: str = "forge_optimization"
    storage: Optional[str] = None  # Optuna storage URL
    direction: str = "maximize"  # maximize or minimize
    metric_name: str = "eval_accuracy"
    pruning_enabled: bool = True
    early_stopping_patience: int = 5
    
    # Distillation settings
    teacher_model_name: Optional[str] = None
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # Pruning settings
    pruning_ratio: float = 0.2
    structured_pruning: bool = True
    recovery_epochs: int = 3
    
    # Search space definitions
    search_space: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        config_dict['method'] = self.method.value
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create config from dictionary."""
        if 'method' in data and isinstance(data['method'], str):
            data['method'] = OptimizationMethod(data['method'])
        return cls(**data)


@dataclass
class OptimizationResult:
    """Results from optimization pipeline."""
    best_params: Dict[str, Any]
    best_value: float
    study_stats: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class HyperparameterSpace:
    """Defines hyperparameter search space for Optuna."""
    
    DEFAULT_SPACE = {
        "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
        "warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.2},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-2},
        "lora_r": {"type": "int", "low": 8, "high": 64, "step": 8},
        "lora_alpha": {"type": "int", "low": 16, "high": 128, "step": 16},
        "lora_dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
        "num_train_epochs": {"type": "int", "low": 1, "high": 10},
        "gradient_accumulation_steps": {"type": "int", "low": 1, "high": 8},
    }
    
    @staticmethod
    def sample_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from defined space."""
        params = {}
        for param_name, param_config in space.items():
            param_type = param_config.get("type", "uniform")
            
            if param_type == "uniform":
                params[param_name] = trial.suggest_uniform(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_loguniform(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return params


class KnowledgeDistiller:
    """Knowledge distillation from teacher to student model."""
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 2.0, alpha: float = 0.5):
        """
        Initialize knowledge distiller.
        
        Args:
            teacher_model: Pre-trained teacher model
            temperature: Temperature for softmax
            alpha: Weight for distillation loss (1-alpha for student loss)
        """
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                         labels: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Soften probabilities
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        distill_loss = distill_loss * (self.temperature ** 2)
        
        # Student loss (cross-entropy)
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        combined_loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss
        
        return combined_loss
    
    def train_step(self, student_model: nn.Module, batch: Dict[str, torch.Tensor], 
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one distillation training step.
        
        Args:
            student_model: Student model to train
            batch: Input batch
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        student_model.train()
        self.teacher.eval()
        
        # Forward pass with teacher
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
        
        # Forward pass with student
        student_outputs = student_model(**batch)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
        
        # Compute loss
        labels = batch.get('labels', batch.get('input_ids'))
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class StructuredPruner:
    """Structured pruning with performance recovery."""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.2, structured: bool = True):
        """
        Initialize pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.structured = structured
        self.masks = {}
    
    def compute_importance_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for pruning.
        
        Returns:
            Dictionary of importance scores per parameter
        """
        scores = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.structured:
                    # For structured pruning, compute L2 norm across output channels
                    if len(param.shape) > 1:
                        scores[name] = torch.norm(param, p=2, dim=tuple(range(1, len(param.shape))))
                else:
                    # For unstructured pruning, use magnitude
                    scores[name] = torch.abs(param)
        
        return scores
    
    def apply_pruning(self) -> Dict[str, torch.Tensor]:
        """
        Apply pruning masks to model.
        
        Returns:
            Dictionary of pruning masks
        """
        scores = self.compute_importance_scores()
        masks = {}
        
        for name, score in scores.items():
            # Flatten scores and get threshold
            flat_scores = score.flatten()
            k = int(len(flat_scores) * self.pruning_ratio)
            if k > 0:
                threshold = torch.topk(flat_scores, k, largest=False).values.max()
                mask = (score > threshold).float()
                
                # Expand mask to full parameter shape if structured
                if self.structured and len(mask.shape) < len(self.model.state_dict()[name].shape):
                    for _ in range(len(self.model.state_dict()[name].shape) - len(mask.shape)):
                        mask = mask.unsqueeze(-1)
                
                masks[name] = mask
                
                # Apply mask
                with torch.no_grad():
                    param = self.model.state_dict()[name]
                    param.mul_(mask)
        
        self.masks = masks
        return masks
    
    def recovery_training(self, train_loader: DataLoader, epochs: int = 3, 
                         lr: float = 1e-4) -> List[float]:
        """
        Fine-tune pruned model to recover performance.
        
        Args:
            train_loader: Training data loader
            epochs: Number of recovery epochs
            lr: Learning rate
            
        Returns:
            List of losses
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
                
                # Re-apply pruning masks to gradients
                for name, param in self.model.named_parameters():
                    if name in self.masks and param.grad is not None:
                        param.grad.mul_(self.masks[name])
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            logger.info(f"Recovery epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses


class HPOEngine:
    """Main engine for automated model optimization."""
    
    def __init__(self, job_manager: Optional[JobManager] = None):
        """
        Initialize HPO engine.
        
        Args:
            job_manager: Optional job manager for tracking optimization jobs
        """
        self.job_manager = job_manager
        self.current_study = None
        self.best_model = None
        self.results_history = []
        
        # Check dependencies
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed. Hyperparameter optimization will be limited.")
        if not UNSLOTH_AVAILABLE:
            logger.warning("Unsloth not installed. Some features may not work.")
    
    async def run_optimization(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: OptimizationConfig,
        training_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run optimization pipeline.
        
        Args:
            model: Base model to optimize
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Optimization configuration
            training_func: Function that trains model with given params
            progress_callback: Optional callback for progress updates
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting optimization with method: {config.method.value}")
        
        if config.method == OptimizationMethod.HYPERPARAMETER:
            return await self._run_hyperparameter_optimization(
                model, train_dataset, eval_dataset, config, training_func, progress_callback
            )
        elif config.method == OptimizationMethod.DISTILLATION:
            return await self._run_distillation(
                model, train_dataset, eval_dataset, config, training_func, progress_callback
            )
        elif config.method == OptimizationMethod.PRUNING:
            return await self._run_pruning(
                model, train_dataset, eval_dataset, config, training_func, progress_callback
            )
        elif config.method == OptimizationMethod.FULL_PIPELINE:
            return await self._run_full_pipeline(
                model, train_dataset, eval_dataset, config, training_func, progress_callback
            )
        else:
            raise ValueError(f"Unsupported optimization method: {config.method}")
    
    async def _run_hyperparameter_optimization(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: OptimizationConfig,
        training_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """Run hyperparameter optimization with Optuna."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for hyperparameter optimization")
        
        # Create or load study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if config.pruning_enabled else None
        
        study = optuna.create_study(
            study_name=config.study_name,
            storage=config.storage,
            sampler=sampler,
            pruner=pruner,
            direction=config.direction,
            load_if_exists=True
        )
        
        self.current_study = study
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            search_space = config.search_space or HyperparameterSpace.DEFAULT_SPACE
            params = HyperparameterSpace.sample_params(trial, search_space)
            
            # Create a copy of the model for this trial
            trial_model = self._clone_model(model)
            
            try:
                # Train model with sampled parameters
                metrics = training_func(
                    model=trial_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    params=params,
                    trial=trial
                )
                
                # Report intermediate values for pruning
                if hasattr(trial, 'report') and config.pruning_enabled:
                    for step, value in enumerate(metrics.get('intermediate_values', [])):
                        trial.report(value, step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                
                # Get final metric
                metric_value = metrics.get(config.metric_name, 0.0)
                
                # Update progress
                if progress_callback:
                    progress_callback({
                        'trial': trial.number,
                        'params': params,
                        'value': metric_value,
                        'state': 'completed'
                    })
                
                return metric_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                if progress_callback:
                    progress_callback({
                        'trial': trial.number,
                        'params': params,
                        'error': str(e),
                        'state': 'failed'
                    })
                raise
        
        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout_seconds,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        
        # Get best results
        best_trial = study.best_trial
        result = OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            study_stats={
                'n_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            },
            optimization_history=[
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in study.trials
            ]
        )
        
        # Store best model
        self.best_model = self._clone_model(model)
        training_func(
            model=self.best_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            params=best_trial.params
        )
        
        self.results_history.append(result)
        return result
    
    async def _run_distillation(
        self,
        student_model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: OptimizationConfig,
        training_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """Run knowledge distillation."""
        logger.info("Starting knowledge distillation")
        
        # Load teacher model
        teacher_model = self._load_teacher_model(config)
        if teacher_model is None:
            raise ValueError("Teacher model not specified or could not be loaded")
        
        # Initialize distiller
        distiller = KnowledgeDistiller(
            teacher_model=teacher_model,
            temperature=config.distillation_temperature,
            alpha=config.distillation_alpha
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=8)
        
        # Training loop with distillation
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        best_accuracy = 0.0
        best_state = None
        history = []
        
        for epoch in range(config.recovery_epochs):
            # Training
            student_model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                loss = distiller.train_step(student_model, batch, optimizer)
                train_losses.append(loss)
                
                if progress_callback and batch_idx % 10 == 0:
                    progress_callback({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss,
                        'phase': 'training'
                    })
            
            # Evaluation
            student_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in eval_loader:
                    outputs = student_model(**batch)
                    predictions = torch.argmax(outputs.logits if hasattr(outputs, 'logits') else outputs, dim=-1)
                    labels = batch.get('labels', batch.get('input_ids'))
                    
                    if labels is not None:
                        correct += (predictions == labels).sum().item()
                        total += labels.numel()
            
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = np.mean(train_losses)
            
            history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'eval_accuracy': accuracy
            })
            
            logger.info(f"Distillation epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}
            
            if progress_callback:
                progress_callback({
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'phase': 'evaluation'
                })
        
        # Restore best model
        if best_state:
            student_model.load_state_dict(best_state)
        
        result = OptimizationResult(
            best_params={'distillation_temperature': config.distillation_temperature,
                        'distillation_alpha': config.distillation_alpha},
            best_value=best_accuracy,
            study_stats={'epochs': config.recovery_epochs},
            optimization_history=history
        )
        
        self.best_model = student_model
        self.results_history.append(result)
        return result
    
    async def _run_pruning(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: OptimizationConfig,
        training_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """Run structured pruning with recovery."""
        logger.info(f"Starting pruning with ratio: {config.pruning_ratio}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=8)
        
        # Evaluate baseline
        baseline_metrics = self._evaluate_model(model, eval_loader)
        logger.info(f"Baseline accuracy: {baseline_metrics.get('accuracy', 0):.4f}")
        
        # Apply pruning
        pruner = StructuredPruner(
            model=model,
            pruning_ratio=config.pruning_ratio,
            structured=config.structured_pruning
        )
        
        masks = pruner.apply_pruning()
        pruned_params = sum(mask.sum().item() for mask in masks.values())
        total_params = sum(mask.numel() for mask in masks.values())
        actual_ratio = 1 - (pruned_params / total_params) if total_params > 0 else 0
        
        logger.info(f"Pruned {pruned_params}/{total_params} parameters (ratio: {actual_ratio:.2f})")
        
        # Recovery training
        recovery_losses = pruner.recovery_training(
            train_loader=train_loader,
            epochs=config.recovery_epochs
        )
        
        # Evaluate after recovery
        recovery_metrics = self._evaluate_model(model, eval_loader)
        
        result = OptimizationResult(
            best_params={
                'pruning_ratio': config.pruning_ratio,
                'actual_ratio': actual_ratio,
                'structured_pruning': config.structured_pruning,
                'recovery_epochs': config.recovery_epochs
            },
            best_value=recovery_metrics.get('accuracy', 0),
            study_stats={
                'pruned_params': int(pruned_params),
                'total_params': int(total_params),
                'recovery_losses': recovery_losses,
                'baseline_accuracy': baseline_metrics.get('accuracy', 0),
                'recovery_accuracy': recovery_metrics.get('accuracy', 0)
            },
            optimization_history=[
                {'epoch': i, 'loss': loss} for i, loss in enumerate(recovery_losses)
            ]
        )
        
        self.best_model = model
        self.results_history.append(result)
        return result
    
    async def _run_full_pipeline(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Any,
        config: OptimizationConfig,
        training_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """Run full optimization pipeline: HPO -> Distillation -> Pruning."""
        logger.info("Starting full optimization pipeline")
        
        # Step 1: Hyperparameter optimization
        hpo_config = OptimizationConfig(
            method=OptimizationMethod.HYPERPARAMETER,
            n_trials=max(10, config.n_trials // 3),
            study_name=f"{config.study_name}_hpo",
            metric_name=config.metric_name
        )
        
        hpo_result = await self._run_hyperparameter_optimization(
            model, train_dataset, eval_dataset, hpo_config, training_func, progress_callback
        )
        
        # Step 2: Distillation (if teacher specified)
        distillation_result = None
        if config.teacher_model_name or config.teacher_model_path:
            distillation_config = OptimizationConfig(
                method=OptimizationMethod.DISTILLATION,
                teacher_model_name=config.teacher_model_name,
                teacher_model_path=config.teacher_model_path,
                distillation_temperature=config.distillation_temperature,
                distillation_alpha=config.distillation_alpha,
                recovery_epochs=max(3, config.recovery_epochs // 2)
            )
            
            distillation_result = await self._run_distillation(
                self.best_model, train_dataset, eval_dataset, 
                distillation_config, training_func, progress_callback
            )
        
        # Step 3: Pruning
        pruning_config = OptimizationConfig(
            method=OptimizationMethod.PRUNING,
            pruning_ratio=config.pruning_ratio,
            structured_pruning=config.structured_pruning,
            recovery_epochs=config.recovery_epochs
        )
        
        pruning_result = await self._run_pruning(
            self.best_model, train_dataset, eval_dataset,
            pruning_config, training_func, progress_callback
        )
        
        # Combine results
        combined_result = OptimizationResult(
            best_params={
                'hpo': hpo_result.best_params,
                'distillation': distillation_result.best_params if distillation_result else None,
                'pruning': pruning_result.best_params
            },
            best_value=pruning_result.best_value,
            study_stats={
                'hpo': hpo_result.study_stats,
                'distillation': distillation_result.study_stats if distillation_result else None,
                'pruning': pruning_result.study_stats
            },
            optimization_history=(
                hpo_result.optimization_history +
                (distillation_result.optimization_history if distillation_result else []) +
                pruning_result.optimization_history
            )
        )
        
        self.results_history.append(combined_result)
        return combined_result
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def _load_teacher_model(self, config: OptimizationConfig) -> Optional[nn.Module]:
        """Load teacher model for distillation."""
        if config.teacher_model_name:
            try:
                # Try to load from Hugging Face
                from transformers import AutoModelForCausalLM
                teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model_name)
                return teacher
            except Exception as e:
                logger.error(f"Failed to load teacher model {config.teacher_model_name}: {e}")
        
        if config.teacher_model_path:
            try:
                # Load from local path
                from transformers import AutoModelForCausalLM
                teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model_path)
                return teacher
            except Exception as e:
                logger.error(f"Failed to load teacher model from {config.teacher_model_path}: {e}")
        
        return None
    
    def _evaluate_model(self, model: nn.Module, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on evaluation dataset."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                
                # Get predictions
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = torch.argmax(logits, dim=-1)
                labels = batch.get('labels', batch.get('input_ids'))
                
                if labels is not None:
                    correct += (predictions == labels).sum().item()
                    total += labels.numel()
                
                # Get loss if available
                if hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total
        }
    
    def save_results(self, output_dir: Union[str, Path]) -> Path:
        """Save optimization results to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results history
        results_file = output_path / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results_history], f, indent=2)
        
        # Save best model if available
        if self.best_model:
            model_path = output_path / "best_model"
            model_path.mkdir(exist_ok=True)
            
            if hasattr(self.best_model, 'save_pretrained'):
                self.best_model.save_pretrained(model_path)
            else:
                torch.save(self.best_model.state_dict(), model_path / "model.pt")
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def load_results(self, results_path: Union[str, Path]) -> List[OptimizationResult]:
        """Load optimization results from directory."""
        results_path = Path(results_path)
        results_file = results_path / "optimization_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        self.results_history = [OptimizationResult.from_dict(r) for r in results_data]
        return self.results_history
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from most recent optimization."""
        if not self.results_history:
            return None
        return self.results_history[-1].best_params
    
    def plot_optimization_history(self, save_path: Optional[Union[str, Path]] = None):
        """Plot optimization history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot history.")
            return
        
        if not self.results_history:
            logger.warning("No optimization history to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, result in enumerate(self.results_history[-3:]):  # Last 3 results
            if 'optimization_history' in result.to_dict():
                history = result.optimization_history
                
                # Plot metric values
                if 'value' in history[0]:
                    ax = axes[i // 2, i % 2]
                    values = [h.get('value', 0) for h in history]
                    ax.plot(values, marker='o')
                    ax.set_title(f"Optimization {i+1}")
                    ax.set_xlabel("Trial")
                    ax.set_ylabel("Metric Value")
                    ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


# Factory function for easy instantiation
def create_hpo_engine(job_manager: Optional[JobManager] = None) -> HPOEngine:
    """Create and return an HPOEngine instance."""
    return HPOEngine(job_manager=job_manager)


# Integration with existing job system
class OptimizationJob:
    """Wrapper for running optimization as a job."""
    
    def __init__(self, engine: HPOEngine, config: OptimizationConfig):
        self.engine = engine
        self.config = config
        self.job_id = None
        self.status = JobStatus.PENDING
    
    async def run(self, model, train_dataset, eval_dataset, training_func):
        """Run optimization job."""
        self.status = JobStatus.RUNNING
        
        try:
            result = await self.engine.run_optimization(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=self.config,
                training_func=training_func,
                progress_callback=self._update_progress
            )
            
            self.status = JobStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = JobStatus.FAILED
            logger.error(f"Optimization job failed: {e}")
            raise
    
    def _update_progress(self, progress_data: Dict[str, Any]):
        """Update job progress."""
        if self.engine.job_manager and self.job_id:
            self.engine.job_manager.update_job_progress(
                self.job_id,
                progress=progress_data.get('progress', 0),
                status_message=json.dumps(progress_data)
            )


# Example training function for demonstration
def example_training_function(model, train_dataset, eval_dataset, params, trial=None):
    """
    Example training function that can be used with HPOEngine.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        params: Hyperparameters
        trial: Optuna trial (optional)
        
    Returns:
        Dictionary of metrics
    """
    # This is a simplified example - in practice, you would implement
    # actual training logic here
    
    # Simulate training
    metrics = {
        'eval_accuracy': np.random.random(),
        'eval_loss': np.random.random(),
        'train_loss': np.random.random(),
    }
    
    # Report intermediate values if trial is provided
    if trial:
        for step in range(5):
            intermediate_value = np.random.random()
            trial.report(intermediate_value, step)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create engine
        engine = create_hpo_engine()
        
        # Create dummy data and model
        from torch.utils.data import TensorDataset
        
        # Dummy dataset
        train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        eval_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
        
        # Dummy model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        # Configuration
        config = OptimizationConfig(
            method=OptimizationMethod.HYPERPARAMETER,
            n_trials=10,
            study_name="example_study"
        )
        
        # Run optimization
        result = await engine.run_optimization(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            config=config,
            training_func=example_training_function
        )
        
        print(f"Best parameters: {result.best_params}")
        print(f"Best value: {result.best_value}")
        
        # Save results
        engine.save_results("./optimization_results")
    
    # Run example
    asyncio.run(main())