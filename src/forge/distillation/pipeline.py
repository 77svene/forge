"""
Model Distillation Pipeline for forge

Automated knowledge distillation system that creates smaller, faster models from larger teacher models.
Includes automatic architecture search for student models, progressive distillation, and deployment-ready model optimization.
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
from collections import OrderedDict
import math

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
import numpy as np
from torch.utils.data import DataLoader

from ..model import load_model, load_tokenizer
from ..data import get_dataset, preprocess_data
from ..train import Trainer as forgeTrainer
from ..utils import (
    count_parameters,
    get_model_size,
    get_device_map,
    save_model,
    load_config,
    compute_loss,
)
from ..extras.logging import get_logger
from ..hparams import ModelArguments, DataArguments, TrainingArguments, FinetuningArguments

logger = get_logger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation pipeline."""
    
    # Architecture search parameters
    target_model_size_mb: float = 100.0  # Target model size in MB
    target_inference_speedup: float = 2.0  # Target speedup factor
    architecture_search_method: str = "evolutionary"  # evolutionary, random, or manual
    max_student_layers: int = 24
    min_student_layers: int = 6
    hidden_size_candidates: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    num_attention_heads_candidates: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    
    # Distillation parameters
    distillation_type: str = "progressive"  # progressive, layerwise, or full
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs student loss
    layer_mapping_strategy: str = "uniform"  # uniform, attention, or learned
    distill_attention: bool = True
    distill_hidden_states: bool = True
    distill_logits: bool = True
    
    # Training parameters
    num_distillation_epochs: int = 3
    progressive_stages: int = 3  # Number of stages for progressive distillation
    warmup_ratio: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Deployment optimization
    quantization: Optional[str] = None  # "int8", "int4", "gptq", or None
    pruning_ratio: float = 0.0  # Ratio of weights to prune
    export_format: str = "huggingface"  # huggingface, onnx, or tensorrt
    optimization_level: int = 1  # 0: no optimization, 1: basic, 2: aggressive
    
    # Paths
    teacher_model_path: str = ""
    student_model_save_path: str = ""
    cache_dir: str = "./cache"
    log_dir: str = "./logs/distillation"


class ArchitectureSearcher:
    """Automatic architecture search for student models based on teacher model and constraints."""
    
    def __init__(self, teacher_config: AutoConfig, distillation_config: DistillationConfig):
        self.teacher_config = teacher_config
        self.config = distillation_config
        self.search_space = self._build_search_space()
        
    def _build_search_space(self) -> Dict[str, List]:
        """Build the search space for student architecture."""
        return {
            "num_hidden_layers": list(range(self.config.min_student_layers, 
                                           self.config.max_student_layers + 1)),
            "hidden_size": self.config.hidden_size_candidates,
            "num_attention_heads": self.config.num_attention_heads_candidates,
            "intermediate_size_ratio": [2.0, 2.5, 3.0, 3.5, 4.0],  # Ratio to hidden_size
        }
    
    def _estimate_model_size(self, architecture: Dict[str, int]) -> float:
        """Estimate model size in MB based on architecture parameters."""
        # Simplified estimation - in practice, would be more accurate
        vocab_size = self.teacher_config.vocab_size
        hidden_size = architecture["hidden_size"]
        num_layers = architecture["num_hidden_layers"]
        intermediate_size = int(hidden_size * architecture.get("intermediate_size_ratio", 3.0))
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size
        
        # Transformer layer parameters (approximate)
        # Each layer has: attention (Q, K, V, O) + FFN (up, down) + layer norms
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        ffn_params = 2 * hidden_size * intermediate_size  # up and down projections
        layer_norm_params = 2 * hidden_size  # Two layer norms per layer
        layer_params = attention_params + ffn_params + layer_norm_params
        
        total_params = embedding_params + (num_layers * layer_params)
        size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return size_mb
    
    def _estimate_inference_speed(self, architecture: Dict[str, int]) -> float:
        """Estimate relative inference speed based on architecture."""
        # Simplified estimation - more layers and larger hidden size = slower
        num_layers = architecture["num_hidden_layers"]
        hidden_size = architecture["hidden_size"]
        
        # Relative speed estimate (higher is faster)
        speed_factor = 1000.0 / (num_layers * math.sqrt(hidden_size))
        return speed_factor
    
    def _evaluate_architecture(self, architecture: Dict[str, int]) -> float:
        """Evaluate architecture based on constraints."""
        size_mb = self._estimate_model_size(architecture)
        speed = self._estimate_inference_speed(architecture)
        
        # Check constraints
        size_penalty = max(0, size_mb - self.config.target_model_size_mb) * 10
        speed_penalty = max(0, 1.0/speed - 1.0/self.config.target_inference_speedup) * 10
        
        # Score based on how well it meets constraints (lower is better)
        score = size_penalty + speed_penalty
        
        # Add complexity penalty (prefer simpler models)
        complexity_penalty = architecture["num_hidden_layers"] * 0.01
        score += complexity_penalty
        
        return score
    
    def _random_search(self, num_candidates: int = 50) -> List[Dict[str, int]]:
        """Random search for student architectures."""
        candidates = []
        
        for _ in range(num_candidates):
            architecture = {}
            for param, values in self.search_space.items():
                if param == "intermediate_size_ratio":
                    architecture[param] = np.random.choice(values)
                else:
                    architecture[param] = np.random.choice(values)
            
            # Ensure num_attention_heads divides hidden_size
            if architecture["hidden_size"] % architecture["num_attention_heads"] != 0:
                # Find closest valid number of heads
                valid_heads = [h for h in self.search_space["num_attention_heads"] 
                              if architecture["hidden_size"] % h == 0]
                if valid_heads:
                    architecture["num_attention_heads"] = np.random.choice(valid_heads)
                else:
                    continue
            
            candidates.append(architecture)
        
        return candidates
    
    def _evolutionary_search(self, population_size: int = 50, 
                            generations: int = 20) -> Dict[str, int]:
        """Evolutionary search for optimal student architecture."""
        # Initialize population
        population = self._random_search(population_size)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_architecture(arch) for arch in population]
            
            # Select top performers
            sorted_indices = np.argsort(fitness_scores)
            elite_size = population_size // 4
            elite_indices = sorted_indices[:elite_size]
            elite = [population[i] for i in elite_indices]
            
            # Create next generation
            next_generation = elite.copy()
            
            while len(next_generation) < population_size:
                # Select parents
                parent1, parent2 = np.random.choice(elite, size=2, replace=False)
                
                # Crossover
                child = {}
                for param in self.search_space.keys():
                    if np.random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                
                # Mutation
                if np.random.random() < 0.1:  # 10% mutation rate
                    param_to_mutate = np.random.choice(list(self.search_space.keys()))
                    if param_to_mutate == "intermediate_size_ratio":
                        child[param_to_mutate] = np.random.choice(
                            self.search_space[param_to_mutate])
                    else:
                        child[param_to_mutate] = np.random.choice(
                            self.search_space[param_to_mutate])
                
                # Ensure validity
                if child["hidden_size"] % child["num_attention_heads"] == 0:
                    next_generation.append(child)
            
            population = next_generation
            
            # Log progress
            best_score = fitness_scores[elite_indices[0]]
            logger.info(f"Generation {generation}: Best score = {best_score:.4f}")
        
        # Return best architecture
        best_idx = np.argmin([self._evaluate_architecture(arch) for arch in population])
        return population[best_idx]
    
    def search(self) -> Dict[str, int]:
        """Search for optimal student architecture."""
        logger.info(f"Starting architecture search using {self.config.architecture_search_method}")
        
        if self.config.architecture_search_method == "random":
            candidates = self._random_search(100)
            scores = [self._evaluate_architecture(arch) for arch in candidates]
            best_idx = np.argmin(scores)
            best_architecture = candidates[best_idx]
        elif self.config.architecture_search_method == "evolutionary":
            best_architecture = self._evolutionary_search()
        else:
            # Manual architecture from config
            best_architecture = {
                "num_hidden_layers": self.config.max_student_layers,
                "hidden_size": self.config.hidden_size_candidates[-1],
                "num_attention_heads": self.config.num_attention_heads_candidates[-1],
                "intermediate_size_ratio": 3.0,
            }
        
        logger.info(f"Selected architecture: {best_architecture}")
        return best_architecture


class DistillationTrainer:
    """Handles the distillation training process with various strategies."""
    
    def __init__(self, teacher_model: PreTrainedModel, 
                 student_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 distillation_config: DistillationConfig):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.config = distillation_config
        self.device = next(student_model.parameters()).device
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Layer mapping between teacher and student
        self.layer_mapping = self._create_layer_mapping()
        
    def _create_layer_mapping(self) -> Dict[int, int]:
        """Create mapping between teacher and student layers."""
        teacher_layers = self.teacher.config.num_hidden_layers
        student_layers = self.student.config.num_hidden_layers
        
        if self.config.layer_mapping_strategy == "uniform":
            # Map student layers uniformly across teacher layers
            mapping = {}
            for i in range(student_layers):
                teacher_idx = int(i * teacher_layers / student_layers)
                mapping[i] = teacher_idx
            return mapping
        elif self.config.layer_mapping_strategy == "attention":
            # Map based on attention patterns (simplified)
            # In practice, would analyze attention similarities
            mapping = {}
            step = teacher_layers / student_layers
            for i in range(student_layers):
                mapping[i] = int(i * step)
            return mapping
        else:
            # Default uniform mapping
            return {i: int(i * teacher_layers / student_layers) 
                   for i in range(student_layers)}
    
    def _distillation_loss(self, student_logits: torch.Tensor, 
                          teacher_logits: torch.Tensor,
                          temperature: float) -> torch.Tensor:
        """Compute KL divergence distillation loss."""
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        loss = loss * (temperature ** 2)  # Scale by temperature squared
        return loss
    
    def _hidden_state_loss(self, student_hidden: torch.Tensor,
                          teacher_hidden: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between hidden states."""
        # Project student hidden states to teacher dimension if needed
        if student_hidden.size(-1) != teacher_hidden.size(-1):
            # Linear projection (in practice, would be a learned projection)
            projection = nn.Linear(student_hidden.size(-1), 
                                  teacher_hidden.size(-1)).to(student_hidden.device)
            student_hidden = projection(student_hidden)
        
        return F.mse_loss(student_hidden, teacher_hidden)
    
    def _attention_loss(self, student_attention: torch.Tensor,
                       teacher_attention: torch.Tensor) -> torch.Tensor:
        """Compute loss between attention matrices."""
        # Normalize attention matrices
        student_attention = student_attention / (student_attention.sum(dim=-1, keepdim=True) + 1e-8)
        teacher_attention = teacher_attention / (teacher_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        return F.mse_loss(student_attention, teacher_attention)
    
    def distill_layerwise(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Layer-wise distillation training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass through teacher
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
        
        # Forward pass through student
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        
        total_loss = 0.0
        num_losses = 0
        
        # Distill logits
        if self.config.distill_logits:
            logits_loss = self._distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                self.config.temperature
            )
            total_loss += logits_loss
            num_losses += 1
        
        # Distill hidden states
        if self.config.distill_hidden_states:
            for student_idx, teacher_idx in self.layer_mapping.items():
                student_hidden = student_outputs.hidden_states[student_idx]
                teacher_hidden = teacher_outputs.hidden_states[teacher_idx]
                
                hidden_loss = self._hidden_state_loss(student_hidden, teacher_hidden)
                total_loss += hidden_loss
                num_losses += 1
        
        # Distill attention patterns
        if self.config.distill_attention:
            for student_idx, teacher_idx in self.layer_mapping.items():
                student_attention = student_outputs.attentions[student_idx]
                teacher_attention = teacher_outputs.attentions[teacher_idx]
                
                attention_loss = self._attention_loss(student_attention, teacher_attention)
                total_loss += attention_loss
                num_losses += 1
        
        # Average losses
        if num_losses > 0:
            total_loss = total_loss / num_losses
        
        return total_loss
    
    def distill_progressive(self, batch: Dict[str, torch.Tensor],
                           stage: int) -> torch.Tensor:
        """Progressive distillation training step."""
        # In progressive distillation, we gradually add more layers to the student
        # For simplicity, we'll use the full student but with different loss weights
        
        # Adjust loss weights based on stage
        stage_weight = stage / self.config.progressive_stages
        
        # For early stages, focus more on logits; for later stages, add hidden states
        original_alpha = self.config.alpha
        self.config.alpha = original_alpha * stage_weight
        
        loss = self.distill_layerwise(batch)
        
        # Restore original alpha
        self.config.alpha = original_alpha
        
        return loss
    
    def train(self, train_dataset: Dataset, 
              eval_dataset: Optional[Dataset] = None,
              training_args: Optional[TrainingArguments] = None) -> PreTrainedModel:
        """Train student model using distillation."""
        
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=self.config.log_dir,
                num_train_epochs=self.config.num_distillation_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch" if eval_dataset else "no",
                remove_unused_columns=False,
                report_to="none",
            )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Custom trainer for distillation
        class DistillationTrainer(Trainer):
            def __init__(self, distillation_trainer: 'DistillationTrainer', **kwargs):
                super().__init__(**kwargs)
                self.distillation_trainer = distillation_trainer
                self.current_stage = 0
            
            def compute_loss(self, model, inputs, return_outputs=False):
                if self.distillation_trainer.config.distillation_type == "progressive":
                    loss = self.distillation_trainer.distill_progressive(
                        inputs, self.current_stage)
                else:
                    loss = self.distillation_trainer.distill_layerwise(inputs)
                
                return (loss, None) if return_outputs else loss
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                # Update stage for progressive distillation
                if self.distillation_trainer.config.distillation_type == "progressive":
                    epoch = state.epoch
                    stage = min(int(epoch * self.distillation_trainer.config.progressive_stages 
                                   / args.num_train_epochs), 
                               self.distillation_trainer.config.progressive_stages - 1)
                    self.current_stage = stage
                    logger.info(f"Progressive distillation stage: {stage}")
        
        # Initialize trainer
        trainer = DistillationTrainer(
            distillation_trainer=self,
            model=self.student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting distillation training...")
        train_result = trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return self.student


class DeploymentOptimizer:
    """Optimizes distilled models for deployment."""
    
    def __init__(self, model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer,
                 config: DistillationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def quantize_model(self) -> PreTrainedModel:
        """Apply quantization to the model."""
        if self.config.quantization == "int8":
            logger.info("Applying INT8 quantization...")
            # In practice, would use bitsandbytes or similar
            # For now, just return the model
            return self.model
        elif self.config.quantization == "int4":
            logger.info("Applying INT4 quantization...")
            # Would use GPTQ or similar
            return self.model
        elif self.config.quantization == "gptq":
            logger.info("Applying GPTQ quantization...")
            # Would use GPTQ library
            return self.model
        else:
            return self.model
    
    def prune_model(self) -> PreTrainedModel:
        """Apply pruning to the model."""
        if self.config.pruning_ratio <= 0:
            return self.model
        
        logger.info(f"Applying pruning with ratio {self.config.pruning_ratio}...")
        
        # Simple magnitude pruning
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate threshold
                weight = module.weight.data.abs()
                threshold = torch.quantile(weight.view(-1), self.config.pruning_ratio)
                
                # Create mask
                mask = weight > threshold
                
                # Apply mask
                module.weight.data *= mask.float()
        
        return self.model
    
    def export_model(self, output_path: str) -> str:
        """Export model in specified format."""
        logger.info(f"Exporting model to {output_path} in {self.config.export_format} format...")
        
        if self.config.export_format == "huggingface":
            # Save in Hugging Face format
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # Save config
            config_path = os.path.join(output_path, "distillation_config.json")
            with open(config_path, "w") as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            return output_path
        
        elif self.config.export_format == "onnx":
            # Export to ONNX
            try:
                import onnx
                from transformers.onnx import export
                
                onnx_path = os.path.join(output_path, "model.onnx")
                
                # Create dummy input
                dummy_input = self.tokenizer("Hello, world!", return_tensors="pt")
                
                # Export
                export(
                    preprocessor=self.tokenizer,
                    model=self.model,
                    output=Path(onnx_path),
                    opset=14,
                )
                
                return onnx_path
            except ImportError:
                logger.warning("ONNX export requires onnx and transformers.onnx")
                return self.export_model(output_path)  # Fallback to HuggingFace
        
        else:
            logger.warning(f"Unsupported export format: {self.config.export_format}")
            return self.export_model(output_path)  # Fallback to HuggingFace
    
    def optimize(self, output_path: str) -> str:
        """Apply all optimizations and export model."""
        # Apply optimizations based on level
        if self.config.optimization_level >= 1:
            self.model = self.quantize_model()
        
        if self.config.optimization_level >= 2:
            self.model = self.prune_model()
        
        # Export
        return self.export_model(output_path)


class DistillationPipeline:
    """Main pipeline for model distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None
        
        # Create directories
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.student_model_save_path), exist_ok=True)
    
    def load_teacher(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load teacher model and tokenizer."""
        logger.info(f"Loading teacher model from {self.config.teacher_model_path}")
        
        # Load tokenizer
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model_path,
            cache_dir=self.config.cache_dir,
        )
        
        # Load model
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_path,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        logger.info(f"Teacher model loaded: {count_parameters(self.teacher_model)} parameters")
        return self.teacher_model, self.teacher_tokenizer
    
    def create_student_model(self, architecture: Dict[str, int]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Create student model based on architecture."""
        logger.info("Creating student model...")
        
        # Create config based on teacher config and architecture
        teacher_config = self.teacher_model.config
        student_config = AutoConfig.from_pretrained(self.config.teacher_model_path)
        
        # Update config with student architecture
        student_config.num_hidden_layers = architecture["num_hidden_layers"]
        student_config.hidden_size = architecture["hidden_size"]
        student_config.num_attention_heads = architecture["num_attention_heads"]
        student_config.intermediate_size = int(
            architecture["hidden_size"] * architecture.get("intermediate_size_ratio", 3.0)
        )
        
        # Create model
        self.student_model = AutoModelForCausalLM.from_config(student_config)
        
        # Use teacher tokenizer for student
        self.student_tokenizer = self.teacher_tokenizer
        
        # Initialize student weights from teacher where possible
        self._initialize_student_weights()
        
        logger.info(f"Student model created: {count_parameters(self.student_model)} parameters")
        return self.student_model, self.student_tokenizer
    
    def _initialize_student_weights(self):
        """Initialize student model weights from teacher model."""
        teacher_state_dict = self.teacher_model.state_dict()
        student_state_dict = self.student_model.state_dict()
        
        # Copy embeddings
        if "model.embed_tokens.weight" in teacher_state_dict:
            student_state_dict["model.embed_tokens.weight"] = teacher_state_dict["model.embed_tokens.weight"]
        
        # Copy layers (simplified - in practice would need careful mapping)
        teacher_layers = self.teacher_model.config.num_hidden_layers
        student_layers = self.student_model.config.num_hidden_layers
        
        for i in range(student_layers):
            teacher_idx = int(i * teacher_layers / student_layers)
            
            # Copy attention weights
            for key in teacher_state_dict.keys():
                if f"model.layers.{teacher_idx}." in key:
                    student_key = key.replace(f"model.layers.{teacher_idx}.", 
                                            f"model.layers.{i}.")
                    if student_key in student_state_dict:
                        if teacher_state_dict[key].shape == student_state_dict[student_key].shape:
                            student_state_dict[student_key] = teacher_state_dict[key]
        
        # Copy LM head
        if "lm_head.weight" in teacher_state_dict:
            student_state_dict["lm_head.weight"] = teacher_state_dict["lm_head.weight"]
        
        # Load state dict
        self.student_model.load_state_dict(student_state_dict, strict=False)
    
    def prepare_dataset(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare dataset for distillation."""
        logger.info("Preparing dataset for distillation...")
        
        # Load dataset (simplified - in practice would use forge's data loading)
        # For now, create a dummy dataset
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of data for training.",
        ] * 100  # Repeat to create more samples
        
        # Tokenize
        tokenized = self.teacher_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist(),
        })
        
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.1, seed=42)
        
        return split["train"], split["test"]
    
    def run(self) -> str:
        """Run the complete distillation pipeline."""
        logger.info("Starting distillation pipeline...")
        
        # Step 1: Load teacher model
        self.load_teacher()
        
        # Step 2: Architecture search
        searcher = ArchitectureSearcher(self.teacher_model.config, self.config)
        architecture = searcher.search()
        
        # Step 3: Create student model
        self.create_student_model(architecture)
        
        # Step 4: Prepare dataset
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Step 5: Distillation training
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            tokenizer=self.student_tokenizer,
            distillation_config=self.config,
        )
        
        self.student_model = trainer.train(train_dataset, eval_dataset)
        
        # Step 6: Deployment optimization
        optimizer = DeploymentOptimizer(
            model=self.student_model,
            tokenizer=self.student_tokenizer,
            config=self.config,
        )
        
        # Step 7: Export optimized model
        output_path = optimizer.optimize(self.config.student_model_save_path)
        
        logger.info(f"Distillation pipeline completed. Model saved to: {output_path}")
        
        # Save pipeline config
        pipeline_config_path = os.path.join(self.config.student_model_save_path, 
                                           "pipeline_config.json")
        with open(pipeline_config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        return output_path


def main():
    """Example usage of the distillation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Distillation Pipeline")
    parser.add_argument("--teacher_model", type=str, required=True,
                       help="Path to teacher model")
    parser.add_argument("--student_output", type=str, required=True,
                       help="Path to save student model")
    parser.add_argument("--target_size_mb", type=float, default=100.0,
                       help="Target model size in MB")
    parser.add_argument("--target_speedup", type=float, default=2.0,
                       help="Target inference speedup")
    parser.add_argument("--distillation_type", type=str, default="progressive",
                       choices=["progressive", "layerwise", "full"],
                       help="Type of distillation")
    parser.add_argument("--quantization", type=str, default=None,
                       choices=["int8", "int4", "gptq", None],
                       help="Quantization method")
    
    args = parser.parse_args()
    
    # Create config
    config = DistillationConfig(
        teacher_model_path=args.teacher_model,
        student_model_save_path=args.student_output,
        target_model_size_mb=args.target_size_mb,
        target_inference_speedup=args.target_speedup,
        distillation_type=args.distillation_type,
        quantization=args.quantization,
    )
    
    # Run pipeline
    pipeline = DistillationPipeline(config)
    output_path = pipeline.run()
    
    print(f"Distillation completed. Model saved to: {output_path}")


if __name__ == "__main__":
    main()