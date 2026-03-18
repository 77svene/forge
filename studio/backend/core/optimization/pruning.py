"""
Automated Model Optimization Pipeline for Unsloth Studio
Provides hyperparameter optimization, neural architecture search, knowledge distillation,
and structured pruning with automatic performance recovery.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Conditional imports for optional dependencies
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not installed. Hyperparameter optimization will be limited.")

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import existing modules
from studio.backend.core.data_recipe.huggingface import HuggingFaceDatasetManager
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.auth.storage import StorageManager

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
    method: OptimizationMethod = OptimizationMethod.FULL_PIPELINE
    target_metric: str = "eval_loss"
    target_direction: str = "minimize"  # "minimize" or "maximize"
    max_trials: int = 50
    timeout_hours: float = 24.0
    pruning_ratio: float = 0.3
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    recovery_epochs: int = 3
    save_dir: str = "optimization_results"
    device: str = "auto"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        config_dict['method'] = self.method.value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'OptimizationConfig':
        """Create config from dictionary."""
        config_dict['method'] = OptimizationMethod(config_dict['method'])
        return cls(**config_dict)


@dataclass
class OptimizationResult:
    """Results from optimization pipeline."""
    best_config: Dict
    best_score: float
    optimization_history: List[Dict]
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, result_dict: Dict) -> 'OptimizationResult':
        """Create result from dictionary."""
        return cls(**result_dict)


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        config: OptimizationConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.save_dir}/trial_{trial.number}",
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_dir=f"{self.config.save_dir}/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.target_metric,
            greater_is_better=self.config.target_direction == "maximize",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        # Train and evaluate
        try:
            trainer.train()
            metrics = trainer.evaluate()
            score = metrics.get(self.config.target_metric, float('inf'))
            
            # Report intermediate values for pruning
            trial.report(score, trainer.state.global_step)
            
            # Handle pruning based on intermediate values
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def optimize(self) -> OptimizationResult:
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=100)
        
        study = optuna.create_study(
            direction=self.config.target_direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"forge_opt_{int(time.time())}",
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout_hours * 3600,
            show_progress_bar=True,
        )
        
        # Get best trial
        best_trial = study.best_trial
        best_config = {
            "hyperparameters": best_trial.params,
            "score": best_trial.value,
            "trial_number": best_trial.number,
        }
        
        # Create result
        result = OptimizationResult(
            best_config=best_config,
            best_score=best_trial.value,
            optimization_history=[
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                }
                for t in study.trials
            ],
        )
        
        logger.info(f"Hyperparameter optimization completed. Best score: {best_trial.value}")
        return result


class KnowledgeDistiller:
    """Knowledge distillation from teacher to student model."""
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        config: OptimizationConfig,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Calculate distillation loss combining soft and hard targets."""
        # Soft target loss (KL divergence)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(student_logits / temperature, dim=-1),
            nn.functional.softmax(teacher_logits / temperature, dim=-1),
        ) * (temperature ** 2)
        
        # Hard target loss (cross-entropy)
        hard_loss = nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        
        # Combined loss
        return alpha * soft_loss + (1 - alpha) * hard_loss
    
    def distill(
        self,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
    ) -> Dict:
        """Perform knowledge distillation."""
        logger.info("Starting knowledge distillation")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Training loop
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.student_model.to(device)
        self.teacher_model.to(device)
        
        history = []
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.student_model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with teacher
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits
                
                # Forward pass with student
                student_outputs = self.student_model(**batch)
                student_logits = student_outputs.logits
                
                # Calculate loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    batch["labels"],
                    temperature=self.config.distillation_temperature,
                    alpha=self.config.distillation_alpha,
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Evaluation
            self.student_model.eval()
            eval_loss = 0
            
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.student_model(**batch)
                    eval_loss += outputs.loss.item()
            
            avg_eval_loss = eval_loss / len(eval_loader)
            
            # Save best model
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
            }
            history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        
        result = {
            "final_eval_loss": best_eval_loss,
            "training_history": history,
            "distillation_config": {
                "temperature": self.config.distillation_temperature,
                "alpha": self.config.distillation_alpha,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        }
        
        logger.info(f"Knowledge distillation completed. Final eval loss: {best_eval_loss:.4f}")
        return result


class StructuredPruner:
    """Structured pruning with automatic performance recovery."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        config: OptimizationConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
    
    def get_prunable_modules(self) -> List[Tuple[nn.Module, str]]:
        """Identify modules that can be pruned."""
        prunable_modules = []
        
        for name, module in self.model.named_modules():
            # Look for linear layers (common in transformers)
            if isinstance(module, nn.Linear):
                # Skip output layer and embeddings
                if "lm_head" not in name and "embeddings" not in name:
                    prunable_modules.append((module, name))
        
        return prunable_modules
    
    def apply_structured_pruning(
        self,
        pruning_ratio: float = 0.3,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """Apply structured pruning to the model."""
        logger.info(f"Applying structured pruning with ratio {pruning_ratio}")
        
        prunable_modules = self.get_prunable_modules()
        if not prunable_modules:
            logger.warning("No prunable modules found")
            return {"status": "no_modules_pruned"}
        
        pruning_stats = {
            "modules_pruned": 0,
            "total_parameters_before": 0,
            "total_parameters_after": 0,
        }
        
        # Calculate total parameters before pruning
        for module, _ in prunable_modules:
            pruning_stats["total_parameters_before"] += module.weight.numel()
        
        # Apply pruning to each module
        for module, name in prunable_modules:
            # Calculate number of neurons to prune
            num_neurons = module.weight.shape[0]
            n_prune = int(num_neurons * pruning_ratio)
            
            if n_prune > 0:
                # Use L1 norm for structured pruning
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=n_prune,
                    n=1,
                    dim=0,  # Prune entire output neurons
                )
                pruning_stats["modules_pruned"] += 1
        
        # Calculate parameters after pruning
        for module, _ in prunable_modules:
            # Count non-zero weights
            pruning_stats["total_parameters_after"] += torch.sum(module.weight != 0).item()
        
        pruning_stats["pruning_ratio_actual"] = (
            1 - pruning_stats["total_parameters_after"] / pruning_stats["total_parameters_before"]
        )
        
        logger.info(
            f"Pruned {pruning_stats['modules_pruned']} modules. "
            f"Actual pruning ratio: {pruning_stats['pruning_ratio_actual']:.2%}"
        )
        
        return pruning_stats
    
    def recover_performance(
        self,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
    ) -> Dict:
        """Fine-tune pruned model to recover performance."""
        logger.info("Starting performance recovery fine-tuning")
        
        # Remove pruning rehooks to make pruning permanent
        for module, _ in self.get_prunable_modules():
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        
        # Setup training
        training_args = TrainingArguments(
            output_dir=f"{self.config.save_dir}/recovery",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        # Train
        train_result = trainer.train()
        eval_results = trainer.evaluate()
        
        recovery_metrics = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "perplexity": np.exp(eval_results["eval_loss"]),
            "recovery_epochs": num_epochs,
        }
        
        logger.info(f"Performance recovery completed. Eval loss: {eval_results['eval_loss']:.4f}")
        return recovery_metrics
    
    def prune_and_recover(
        self,
        pruning_ratio: float = 0.3,
        recovery_epochs: int = 3,
    ) -> Dict:
        """Full pruning pipeline with performance recovery."""
        # Get baseline performance
        baseline_metrics = self.evaluate_model()
        
        # Apply pruning
        pruning_stats = self.apply_structured_pruning(pruning_ratio)
        
        # Evaluate after pruning
        pruned_metrics = self.evaluate_model()
        
        # Recover performance
        recovery_metrics = self.recover_performance(num_epochs=recovery_epochs)
        
        # Final evaluation
        final_metrics = self.evaluate_model()
        
        result = {
            "baseline": baseline_metrics,
            "after_pruning": pruned_metrics,
            "recovery": recovery_metrics,
            "final": final_metrics,
            "pruning_stats": pruning_stats,
            "performance_change": {
                "loss_change": final_metrics["eval_loss"] - baseline_metrics["eval_loss"],
                "perplexity_change": final_metrics["perplexity"] - baseline_metrics["perplexity"],
            },
        }
        
        return result
    
    def evaluate_model(self) -> Dict:
        """Evaluate model performance."""
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"{self.config.save_dir}/eval",
                per_device_eval_batch_size=8,
                logging_dir=f"{self.config.save_dir}/logs",
            ),
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )
        
        results = trainer.evaluate()
        results["perplexity"] = np.exp(results["eval_loss"])
        
        return results


class ModelOptimizer:
    """Main model optimization pipeline orchestrator."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        config: Optional[OptimizationConfig] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or OptimizationConfig()
        self.storage_manager = storage_manager or StorageManager()
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Initialize components
        self.hyperparameter_optimizer = None
        self.distiller = None
        self.pruner = None
        
        # Results storage
        self.results = {}
    
    def run_hyperparameter_optimization(self) -> OptimizationResult:
        """Run hyperparameter optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            config=self.config,
        )
        
        result = self.hyperparameter_optimizer.optimize()
        self.results['hyperparameter_optimization'] = result.to_dict()
        
        # Save results
        self._save_results('hyperparameter_optimization', result.to_dict())
        
        return result
    
    def run_knowledge_distillation(
        self,
        teacher_model: PreTrainedModel,
    ) -> Dict:
        """Run knowledge distillation."""
        self.distiller = KnowledgeDistiller(
            teacher_model=teacher_model,
            student_model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            config=self.config,
        )
        
        result = self.distiller.distill()
        self.results['knowledge_distillation'] = result
        
        # Save results
        self._save_results('knowledge_distillation', result)
        
        return result
    
    def run_structured_pruning(self) -> Dict:
        """Run structured pruning with performance recovery."""
        self.pruner = StructuredPruner(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            config=self.config,
        )
        
        result = self.pruner.prune_and_recover(
            pruning_ratio=self.config.pruning_ratio,
            recovery_epochs=self.config.recovery_epochs,
        )
        self.results['structured_pruning'] = result
        
        # Save results
        self._save_results('structured_pruning', result)
        
        return result
    
    def run_full_pipeline(
        self,
        teacher_model: Optional[PreTrainedModel] = None,
    ) -> Dict:
        """Run complete optimization pipeline."""
        logger.info("Starting full optimization pipeline")
        
        pipeline_results = {}
        
        # Step 1: Hyperparameter optimization
        if self.config.method in [OptimizationMethod.HYPERPARAMETER, OptimizationMethod.FULL_PIPELINE]:
            logger.info("Step 1: Hyperparameter optimization")
            try:
                hp_result = self.run_hyperparameter_optimization()
                pipeline_results['hyperparameter_optimization'] = hp_result.to_dict()
            except Exception as e:
                logger.error(f"Hyperparameter optimization failed: {e}")
                pipeline_results['hyperparameter_optimization'] = {"error": str(e)}
        
        # Step 2: Knowledge distillation (if teacher provided)
        if teacher_model and self.config.method in [OptimizationMethod.DISTILLATION, OptimizationMethod.FULL_PIPELINE]:
            logger.info("Step 2: Knowledge distillation")
            try:
                distill_result = self.run_knowledge_distillation(teacher_model)
                pipeline_results['knowledge_distillation'] = distill_result
            except Exception as e:
                logger.error(f"Knowledge distillation failed: {e}")
                pipeline_results['knowledge_distillation'] = {"error": str(e)}
        
        # Step 3: Structured pruning
        if self.config.method in [OptimizationMethod.PRUNING, OptimizationMethod.FULL_PIPELINE]:
            logger.info("Step 3: Structured pruning")
            try:
                prune_result = self.run_structured_pruning()
                pipeline_results['structured_pruning'] = prune_result
            except Exception as e:
                logger.error(f"Structured pruning failed: {e}")
                pipeline_results['structured_pruning'] = {"error": str(e)}
        
        # Save final model
        final_model_path = self._save_final_model()
        pipeline_results['final_model_path'] = final_model_path
        
        # Save complete pipeline results
        self._save_results('full_pipeline', pipeline_results)
        
        logger.info("Full optimization pipeline completed")
        return pipeline_results
    
    def _save_results(self, stage: str, results: Dict):
        """Save optimization results to storage."""
        timestamp = int(time.time())
        filename = f"{stage}_{timestamp}.json"
        filepath = os.path.join(self.config.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save to storage manager if available
        if self.storage_manager:
            try:
                self.storage_manager.save_file(
                    filepath,
                    f"optimization/{filename}",
                )
            except Exception as e:
                logger.warning(f"Could not save to storage manager: {e}")
    
    def _save_final_model(self) -> str:
        """Save the final optimized model."""
        model_path = os.path.join(self.config.save_dir, "final_model")
        os.makedirs(model_path, exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save optimization config
        config_path = os.path.join(model_path, "optimization_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Final model saved to {model_path}")
        return model_path
    
    def load_results(self, results_path: str) -> Dict:
        """Load optimization results from file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of all optimization results."""
        summary = {
            "config": self.config.to_dict(),
            "stages_completed": list(self.results.keys()),
            "timestamp": time.time(),
        }
        
        # Add metrics from each stage
        for stage, results in self.results.items():
            if isinstance(results, dict):
                if 'best_score' in results:
                    summary[f"{stage}_best_score"] = results['best_score']
                if 'final_eval_loss' in results:
                    summary[f"{stage}_eval_loss"] = results['final_eval_loss']
                if 'performance_change' in results:
                    summary[f"{stage}_performance_change"] = results['performance_change']
        
        return summary


def create_optimization_pipeline(
    model_name_or_path: str,
    dataset_name: str,
    optimization_config: Optional[Dict] = None,
    **kwargs,
) -> ModelOptimizer:
    """Factory function to create an optimization pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load dataset
    if "/" in dataset_name:
        # Hugging Face dataset
        dataset = load_dataset(dataset_name, split="train")
        # Simple train/eval split
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        # Local dataset or custom loading
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create config
    config = OptimizationConfig(**optimization_config) if optimization_config else OptimizationConfig()
    
    # Create optimizer
    optimizer = ModelOptimizer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        **kwargs,
    )
    
    return optimizer


# CLI integration
def optimize_model_cli(args):
    """CLI entry point for model optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model optimization pipeline")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--method", default="full_pipeline", 
                       choices=[m.value for m in OptimizationMethod],
                       help="Optimization method")
    parser.add_argument("--target-metric", default="eval_loss", help="Target metric to optimize")
    parser.add_argument("--target-direction", default="minimize", 
                       choices=["minimize", "maximize"],
                       help="Optimization direction")
    parser.add_argument("--max-trials", type=int, default=50, help="Maximum optimization trials")
    parser.add_argument("--pruning-ratio", type=float, default=0.3, help="Pruning ratio")
    parser.add_argument("--save-dir", default="optimization_results", help="Directory to save results")
    
    parsed_args = parser.parse_args(args)
    
    # Create optimization config
    config = OptimizationConfig(
        method=OptimizationMethod(parsed_args.method),
        target_metric=parsed_args.target_metric,
        target_direction=parsed_args.target_direction,
        max_trials=parsed_args.max_trials,
        pruning_ratio=parsed_args.pruning_ratio,
        save_dir=parsed_args.save_dir,
    )
    
    # Create and run pipeline
    optimizer = create_optimization_pipeline(
        model_name_or_path=parsed_args.model,
        dataset_name=parsed_args.dataset,
        optimization_config=config.to_dict(),
    )
    
    results = optimizer.run_full_pipeline()
    
    # Print summary
    summary = optimizer.get_optimization_summary()
    print("\n=== Optimization Summary ===")
    print(json.dumps(summary, indent=2))
    
    return results


# Integration with existing CLI
if __name__ == "__main__":
    import sys
    optimize_model_cli(sys.argv[1:])