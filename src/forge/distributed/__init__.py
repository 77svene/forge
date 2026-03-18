"""Distributed Training Optimizer for forge.

Automatic distributed training configuration that selects optimal parallelism strategy
(FSDP, DeepSpeed, Megatron-LM) based on model size and cluster topology. Includes
automatic gradient checkpointing, activation recomputation, and communication optimization.
"""

import os
import json
import math
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Available parallelism strategies."""
    DDP = "ddp"  # Data Parallel
    FSDP = "fsdp"  # Fully Sharded Data Parallel
    DEEPSPEED = "deepspeed"  # DeepSpeed ZeRO
    MEGATRON = "megatron"  # Megatron-LM tensor/pipeline parallel


@dataclass
class ClusterTopology:
    """Cluster topology information."""
    num_nodes: int = 1
    gpus_per_node: int = 1
    gpu_memory_gb: float = 80.0  # GB
    interconnect_bandwidth_gbps: float = 100.0  # Gbps
    has_nvlink: bool = True
    has_infiniband: bool = False
    cpu_cores_per_node: int = 64
    cpu_memory_gb: float = 256.0  # GB

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def is_multinode(self) -> bool:
        return self.num_nodes > 1


@dataclass
class ModelSpec:
    """Model specifications for optimization."""
    num_parameters: int  # Total parameters
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_sequence_length: int = 2048
    dtype: torch.dtype = torch.float16
    is_moe: bool = False  # Mixture of Experts
    num_experts: int = 1

    @property
    def parameter_memory_gb(self) -> float:
        """Estimate memory for parameters in GB."""
        bytes_per_param = 2 if self.dtype == torch.float16 else 4
        return (self.num_parameters * bytes_per_param) / (1024 ** 3)

    @property
    def activation_memory_gb(self) -> float:
        """Estimate activation memory in GB for batch size 1."""
        # Simplified estimation based on hidden size and sequence length
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        # Rough estimate: hidden_size * seq_len * num_layers * some factor
        return (self.hidden_size * self.max_sequence_length * 
                self.num_layers * bytes_per_element * 8) / (1024 ** 3)


@dataclass
class OptimizationConfig:
    """Configuration for distributed training optimization."""
    strategy: ParallelStrategy = ParallelStrategy.DDP
    gradient_checkpointing: bool = False
    activation_recomputation: bool = False
    communication_optimization: bool = True
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    zero_stage: int = 0  # DeepSpeed ZeRO stage
    cpu_offload: bool = False
    nvlink_optimization: bool = True
    overlap_communication: bool = True
    bucket_size_mb: int = 25
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['strategy'] = self.strategy.value
        return result


class DistributedTrainingOptimizer:
    """Automatic distributed training configuration optimizer.
    
    Analyzes model specifications and cluster topology to select optimal
    parallelism strategy and configuration for distributed training.
    """
    
    # Thresholds for strategy selection (in billions of parameters)
    SMALL_MODEL_THRESHOLD = 1.0
    MEDIUM_MODEL_THRESHOLD = 7.0
    LARGE_MODEL_THRESHOLD = 70.0
    
    def __init__(
        self,
        model_spec: ModelSpec,
        cluster_topology: Optional[ClusterTopology] = None,
        target_batch_size: int = 32,
        target_global_batch_size: Optional[int] = None,
    ):
        """Initialize the optimizer.
        
        Args:
            model_spec: Model specifications
            cluster_topology: Cluster topology information (auto-detected if None)
            target_batch_size: Target micro batch size per GPU
            target_global_batch_size: Target global batch size (optional)
        """
        self.model_spec = model_spec
        self.cluster_topology = cluster_topology or self._detect_cluster_topology()
        self.target_batch_size = target_batch_size
        self.target_global_batch_size = target_global_batch_size
        
        # Validate configurations
        self._validate_configuration()
    
    def _detect_cluster_topology(self) -> ClusterTopology:
        """Auto-detect cluster topology from environment."""
        try:
            # Try to detect from SLURM environment
            if 'SLURM_JOB_NUM_NODES' in os.environ:
                num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
                gpus_per_node = torch.cuda.device_count()
                return ClusterTopology(
                    num_nodes=num_nodes,
                    gpus_per_node=gpus_per_node,
                    has_infiniband=True  # Assume InfiniBand in HPC clusters
                )
            
            # Try to detect from PyTorch distributed environment
            if dist.is_initialized():
                world_size = dist.get_world_size()
                # Assume single node if not specified
                return ClusterTopology(
                    num_nodes=1,
                    gpus_per_node=world_size
                )
            
            # Default to single GPU
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            return ClusterTopology(
                num_nodes=1,
                gpus_per_node=gpu_count,
                gpu_memory_gb=self._get_gpu_memory_gb()
            )
            
        except Exception as e:
            logger.warning(f"Failed to auto-detect cluster topology: {e}")
            return ClusterTopology()
    
    def _get_gpu_memory_gb(self) -> float:
        """Get GPU memory in GB."""
        if torch.cuda.is_available():
            try:
                # Try to get memory from first GPU
                gpu_properties = torch.cuda.get_device_properties(0)
                return gpu_properties.total_memory / (1024 ** 3)
            except:
                pass
        return 80.0  # Default to A100 memory
    
    def _validate_configuration(self):
        """Validate the configuration."""
        if self.model_spec.num_parameters <= 0:
            raise ValueError("Model must have positive number of parameters")
        
        if self.cluster_topology.total_gpus <= 0:
            raise ValueError("Cluster must have at least one GPU")
    
    def estimate_memory_requirements(self, batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory requirements for training.
        
        Args:
            batch_size: Batch size per GPU
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Parameter memory
        param_memory = self.model_spec.parameter_memory_gb
        
        # Optimizer states (Adam: 8 bytes per parameter for fp16 training)
        optimizer_memory = param_memory * 8
        
        # Gradient memory
        gradient_memory = param_memory * 2  # Same as parameters for fp16
        
        # Activation memory (scales with batch size)
        activation_memory = self.model_spec.activation_memory_gb * batch_size
        
        # Total memory needed
        total_memory = param_memory + optimizer_memory + gradient_memory + activation_memory
        
        return {
            "parameters_gb": param_memory,
            "optimizer_states_gb": optimizer_memory,
            "gradients_gb": gradient_memory,
            "activations_gb": activation_memory,
            "total_gb": total_memory,
        }
    
    def select_optimal_strategy(self) -> OptimizationConfig:
        """Select optimal parallelism strategy based on model and cluster.
        
        Returns:
            OptimizationConfig with selected strategy and parameters
        """
        model_size_b = self.model_spec.num_parameters / 1e9
        gpu_memory_gb = self.cluster_topology.gpu_memory_gb
        total_gpus = self.cluster_topology.total_gpus
        
        logger.info(f"Model size: {model_size_b:.2f}B parameters")
        logger.info(f"Cluster: {self.cluster_topology.num_nodes} nodes, "
                   f"{self.cluster_topology.gpus_per_node} GPUs per node, "
                   f"{gpu_memory_gb:.1f}GB GPU memory")
        
        # Check if model fits in single GPU memory
        memory_req = self.estimate_memory_requirements(self.target_batch_size)
        fits_single_gpu = memory_req["total_gb"] < gpu_memory_gb * 0.8  # 80% threshold
        
        # Strategy selection logic
        if model_size_b < self.SMALL_MODEL_THRESHOLD:
            # Small models: DDP is sufficient
            config = self._configure_ddp()
            
        elif model_size_b < self.MEDIUM_MODEL_THRESHOLD:
            # Medium models: FSDP or DeepSpeed ZeRO
            if fits_single_gpu:
                config = self._configure_ddp()
            else:
                config = self._configure_fsdp()
                
        elif model_size_b < self.LARGE_MODEL_THRESHOLD:
            # Large models: DeepSpeed ZeRO or tensor parallelism
            if total_gpus >= 8:
                config = self._configure_megatron_tensor_parallel()
            else:
                config = self._configure_deepspeed()
                
        else:
            # Very large models: Megatron-LM with tensor and pipeline parallelism
            config = self._configure_megatron_full_parallel()
        
        # Apply automatic optimizations
        self._apply_automatic_optimizations(config)
        
        # Adjust batch sizes based on memory
        self._adjust_batch_sizes(config)
        
        logger.info(f"Selected strategy: {config.strategy.value}")
        logger.info(f"Configuration: {config.to_dict()}")
        
        return config
    
    def _configure_ddp(self) -> OptimizationConfig:
        """Configure for Distributed Data Parallel."""
        return OptimizationConfig(
            strategy=ParallelStrategy.DDP,
            data_parallel_size=self.cluster_topology.total_gpus,
            communication_optimization=True,
            overlap_communication=True,
        )
    
    def _configure_fsdp(self) -> OptimizationConfig:
        """Configure for Fully Sharded Data Parallel."""
        # Determine if CPU offloading is needed
        memory_req = self.estimate_memory_requirements(self.target_batch_size)
        cpu_offload = memory_req["total_gb"] > self.cluster_topology.gpu_memory_gb * 0.7
        
        return OptimizationConfig(
            strategy=ParallelStrategy.FSDP,
            data_parallel_size=self.cluster_topology.total_gpus,
            cpu_offload=cpu_offload,
            communication_optimization=True,
            overlap_communication=True,
            bucket_size_mb=25,
        )
    
    def _configure_deepspeed(self) -> OptimizationConfig:
        """Configure for DeepSpeed ZeRO."""
        # Determine ZeRO stage based on model size and GPU memory
        memory_req = self.estimate_memory_requirements(self.target_batch_size)
        
        if memory_req["total_gb"] < self.cluster_topology.gpu_memory_gb * 0.5:
            zero_stage = 1  # Optimizer state partitioning
        elif memory_req["total_gb"] < self.cluster_topology.gpu_memory_gb * 0.8:
            zero_stage = 2  # + Gradient partitioning
        else:
            zero_stage = 3  # + Parameter partitioning
        
        # Enable CPU offload for very large models
        cpu_offload = (zero_stage == 3 and 
                      memory_req["total_gb"] > self.cluster_topology.gpu_memory_gb * 0.9)
        
        return OptimizationConfig(
            strategy=ParallelStrategy.DEEPSPEED,
            data_parallel_size=self.cluster_topology.total_gpus,
            zero_stage=zero_stage,
            cpu_offload=cpu_offload,
            communication_optimization=True,
            overlap_communication=True,
            bucket_size_mb=500,  # Larger buckets for DeepSpeed
        )
    
    def _configure_megatron_tensor_parallel(self) -> OptimizationConfig:
        """Configure for Megatron-LM tensor parallelism."""
        # Determine optimal tensor parallel size
        # Typically, TP size should divide hidden_size evenly
        max_tp_size = min(
            self.cluster_topology.gpus_per_node,  # Within node
            self.model_spec.hidden_size // 64,  # Reasonable chunk size
            8  # Maximum practical TP size
        )
        
        # Find largest power of 2 that divides hidden_size
        tp_size = 1
        for i in range(int(math.log2(max_tp_size)) + 1, 0, -1):
            candidate = 2 ** i
            if self.model_spec.hidden_size % candidate == 0:
                tp_size = candidate
                break
        
        # Calculate data parallel size
        dp_size = self.cluster_topology.total_gpus // tp_size
        
        return OptimizationConfig(
            strategy=ParallelStrategy.MEGATRON,
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            pipeline_parallel_size=1,
            communication_optimization=True,
            nvlink_optimization=self.cluster_topology.has_nvlink,
        )
    
    def _configure_megatron_full_parallel(self) -> OptimizationConfig:
        """Configure for Megatron-LM with both tensor and pipeline parallelism."""
        # For very large models, use both tensor and pipeline parallelism
        
        # Start with tensor parallel within node
        tp_size = min(self.cluster_topology.gpus_per_node, 8)
        
        # Ensure TP divides hidden_size
        while tp_size > 1 and self.model_spec.hidden_size % tp_size != 0:
            tp_size -= 1
        
        # Calculate pipeline parallel size
        # PP size should divide number of layers
        max_pp_size = self.cluster_topology.num_nodes
        pp_size = 1
        for i in range(1, max_pp_size + 1):
            if self.model_spec.num_layers % i == 0:
                pp_size = i
        
        # Remaining GPUs go to data parallel
        dp_size = self.cluster_topology.total_gpus // (tp_size * pp_size)
        
        return OptimizationConfig(
            strategy=ParallelStrategy.MEGATRON,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=max(1, dp_size),
            communication_optimization=True,
            nvlink_optimization=self.cluster_topology.has_nvlink,
            overlap_communication=True,
        )
    
    def _apply_automatic_optimizations(self, config: OptimizationConfig):
        """Apply automatic optimizations based on model and cluster."""
        
        # Gradient checkpointing for large models
        model_size_b = self.model_spec.num_parameters / 1e9
        if model_size_b > self.MEDIUM_MODEL_THRESHOLD:
            config.gradient_checkpointing = True
            config.activation_recomputation = True
            logger.info("Enabled gradient checkpointing for large model")
        
        # Communication optimization based on interconnect
        if self.cluster_topology.has_nvlink:
            config.nvlink_optimization = True
            config.bucket_size_mb = 50  # Larger buckets for NVLink
        elif self.cluster_topology.has_infiniband:
            config.bucket_size_mb = 25
        else:
            # Ethernet or slower interconnect
            config.bucket_size_mb = 10
            config.overlap_communication = True
        
        # CPU offload for memory-constrained scenarios
        memory_req = self.estimate_memory_requirements(self.target_batch_size)
        if memory_req["total_gb"] > self.cluster_topology.gpu_memory_gb * 0.9:
            config.cpu_offload = True
            logger.info("Enabled CPU offload due to memory constraints")
    
    def _adjust_batch_sizes(self, config: OptimizationConfig):
        """Adjust batch sizes based on memory constraints."""
        # Calculate maximum batch size that fits in memory
        memory_per_sample = self.model_spec.activation_memory_gb
        
        # Account for gradient accumulation
        effective_batch_size = self.target_batch_size
        
        # Adjust based on parallelism strategy
        if config.strategy == ParallelStrategy.MEGATRON:
            # Tensor parallelism reduces activation memory
            memory_per_sample /= config.tensor_parallel_size
        
        # Calculate memory budget (leave 20% headroom)
        memory_budget = self.cluster_topology.gpu_memory_gb * 0.8
        
        # Subtract parameter and optimizer memory
        param_memory = self.model_spec.parameter_memory_gb
        if config.strategy in [ParallelStrategy.FSDP, ParallelStrategy.DEEPSPEED]:
            # These strategies partition parameters
            param_memory /= config.data_parallel_size
        
        optimizer_memory = param_memory * 8  # Adam optimizer
        gradient_memory = param_memory * 2
        
        available_memory = memory_budget - param_memory - optimizer_memory - gradient_memory
        
        # Calculate maximum batch size
        max_batch_size = int(available_memory / memory_per_sample)
        
        if max_batch_size < effective_batch_size:
            logger.warning(f"Reducing batch size from {effective_batch_size} to {max_batch_size} "
                          f"due to memory constraints")
            config.micro_batch_size = max(1, max_batch_size)
            
            # Adjust gradient accumulation to maintain global batch size
            if self.target_global_batch_size:
                config.gradient_accumulation_steps = max(
                    1,
                    self.target_global_batch_size // 
                    (config.micro_batch_size * config.data_parallel_size)
                )
    
    def generate_deepspeed_config(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate DeepSpeed configuration JSON.
        
        Args:
            config: Optimization configuration
            
        Returns:
            DeepSpeed configuration dictionary
        """
        if config.strategy != ParallelStrategy.DEEPSPEED:
            raise ValueError("DeepSpeed config only available for DeepSpeed strategy")
        
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": config.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 500000000,
                "overlap_comm": config.overlap_communication,
                "reduce_scatter": True,
                "reduce_bucket_size": 500000000,
                "contiguous_gradients": True,
            },
            "fp16": {
                "enabled": self.model_spec.dtype == torch.float16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {
                "enabled": self.model_spec.dtype == torch.bfloat16,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                },
            },
            "communication_data_type": "fp16" if self.model_spec.dtype == torch.float16 else "bf32",
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        
        # Add CPU offload if enabled
        if config.cpu_offload:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            if config.zero_stage == 3:
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
        
        return ds_config
    
    def generate_fsdp_config(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate FSDP configuration.
        
        Args:
            config: Optimization configuration
            
        Returns:
            FSDP configuration dictionary
        """
        if config.strategy != ParallelStrategy.FSDP:
            raise ValueError("FSDP config only available for FSDP strategy")
        
        fsdp_config = {
            "sharding_strategy": "FULL_SHARD",  # Equivalent to ZeRO-3
            "cpu_offload": config.cpu_offload,
            "mixed_precision": {
                "param_dtype": self.model_spec.dtype,
                "reduce_dtype": self.model_spec.dtype,
                "buffer_dtype": self.model_spec.dtype,
            },
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": False,
            "limit_all_gathers": True,
            "use_orig_params": True,
            "sync_module_states": True,
            "activation_checkpointing": config.gradient_checkpointing,
            "activation_checkpointing_reentrant": False,
        }
        
        return fsdp_config
    
    def generate_megatron_args(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate Megatron-LM arguments.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Megatron-LM arguments dictionary
        """
        if config.strategy != ParallelStrategy.MEGATRON:
            raise ValueError("Megatron args only available for Megatron strategy")
        
        megatron_args = {
            "--tensor-model-parallel-size": config.tensor_parallel_size,
            "--pipeline-model-parallel-size": config.pipeline_parallel_size,
            "--num-layers": self.model_spec.num_layers,
            "--hidden-size": self.model_spec.hidden_size,
            "--num-attention-heads": self.model_spec.num_attention_heads,
            "--seq-length": self.model_spec.max_sequence_length,
            "--max-position-embeddings": self.model_spec.max_sequence_length,
            "--micro-batch-size": config.micro_batch_size,
            "--global-batch-size": (config.micro_batch_size * 
                                   config.data_parallel_size * 
                                   config.gradient_accumulation_steps),
            "--lr": 1e-4,
            "--train-iters": 500000,
            "--lr-decay-iters": 320000,
            "--lr-decay-style": "cosine",
            "--min-lr": 1.0e-5,
            "--weight-decay": 1e-2,
            "--clip-grad": 1.0,
            "--lr-warmup-fraction": 0.01,
            "--vocab-size": self.model_spec.vocab_size,
            "--attention-dropout": 0.0,
            "--hidden-dropout": 0.0,
            "--optimizer": "adam",
            "--adam-beta1": 0.9,
            "--adam-beta2": 0.95,
            "--adam-eps": 1e-8,
            "--fp16": self.model_spec.dtype == torch.float16,
            "--bf16": self.model_spec.dtype == torch.bfloat16,
            "--recompute-activations": config.activation_recomputation,
            "--recompute-granularity": "full",
            "--distribute-saved-activations": False,
            "--use-flash-attn": True,
            "--no-masked-softmax-fusion": False,
            "--no-gradient-accumulation-fusion": False,
        }
        
        # Add MoE arguments if applicable
        if self.model_spec.is_moe:
            megatron_args.update({
                "--num-experts": self.model_spec.num_experts,
                "--moe-router-topk": 2,
                "--moe-grouped-gemm": True,
            })
        
        return megatron_args
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """Get complete recommended configuration.
        
        Returns:
            Dictionary with strategy, config, and additional recommendations
        """
        config = self.select_optimal_strategy()
        
        result = {
            "strategy": config.strategy.value,
            "optimization_config": config.to_dict(),
            "cluster_topology": asdict(self.cluster_topology),
            "model_spec": {
                "num_parameters": self.model_spec.num_parameters,
                "num_layers": self.model_spec.num_layers,
                "hidden_size": self.model_spec.hidden_size,
                "memory_estimate_gb": self.estimate_memory_requirements(),
            },
            "recommendations": [],
        }
        
        # Add strategy-specific configurations
        if config.strategy == ParallelStrategy.DEEPSPEED:
            result["deepspeed_config"] = self.generate_deepspeed_config(config)
        elif config.strategy == ParallelStrategy.FSDP:
            result["fsdp_config"] = self.generate_fsdp_config(config)
        elif config.strategy == ParallelStrategy.MEGATRON:
            result["megatron_args"] = self.generate_megatron_args(config)
        
        # Add general recommendations
        if config.gradient_checkpointing:
            result["recommendations"].append(
                "Enable gradient checkpointing to reduce activation memory at cost of ~20% compute"
            )
        
        if config.cpu_offload:
            result["recommendations"].append(
                "CPU offload enabled - ensure sufficient CPU memory and fast PCIe/NVLink connection"
            )
        
        if not self.cluster_topology.has_nvlink and config.tensor_parallel_size > 1:
            result["recommendations"].append(
                "Tensor parallelism without NVLink may have high communication overhead"
            )
        
        return result


def auto_configure_distributed_training(
    model: nn.Module,
    model_config: Optional[Dict[str, Any]] = None,
    cluster_info: Optional[Dict[str, Any]] = None,
    target_batch_size: int = 32,
) -> Dict[str, Any]:
    """Automatic distributed training configuration for a PyTorch model.
    
    Args:
        model: PyTorch model
        model_config: Optional model configuration (if not inferrable from model)
        cluster_info: Optional cluster information
        target_batch_size: Target micro batch size per GPU
        
    Returns:
        Complete configuration dictionary
    """
    # Extract model specifications
    if model_config:
        model_spec = ModelSpec(
            num_parameters=model_config.get("num_parameters", sum(p.numel() for p in model.parameters())),
            num_layers=model_config.get("num_layers", 32),
            hidden_size=model_config.get("hidden_size", 4096),
            num_attention_heads=model_config.get("num_attention_heads", 32),
            vocab_size=model_config.get("vocab_size", 32000),
            max_sequence_length=model_config.get("max_sequence_length", 2048),
            dtype=next(model.parameters()).dtype,
            is_moe=model_config.get("is_moe", False),
            num_experts=model_config.get("num_experts", 1),
        )
    else:
        # Try to infer from model (simplified)
        total_params = sum(p.numel() for p in model.parameters())
        model_spec = ModelSpec(
            num_parameters=total_params,
            num_layers=len([m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]) or 32,
            hidden_size=getattr(model, "hidden_size", 4096),
            num_attention_heads=getattr(model, "num_attention_heads", 32),
            vocab_size=getattr(model, "vocab_size", 32000),
            max_sequence_length=getattr(model, "max_position_embeddings", 2048),
            dtype=next(model.parameters()).dtype,
        )
    
    # Parse cluster information
    cluster_topology = None
    if cluster_info:
        cluster_topology = ClusterTopology(**cluster_info)
    
    # Create optimizer and get configuration
    optimizer = DistributedTrainingOptimizer(
        model_spec=model_spec,
        cluster_topology=cluster_topology,
        target_batch_size=target_batch_size,
    )
    
    return optimizer.get_recommended_config()


# Integration with existing forge utilities
def get_distributed_config_for_model(
    model_name_or_path: str,
    model_type: str = "llama",
    cluster_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Get distributed training configuration for a specific model.
    
    Args:
        model_name_or_path: Model name or path
        model_type: Model type (llama, qwen, etc.)
        cluster_config_path: Path to cluster configuration JSON
        
    Returns:
        Distributed training configuration
    """
    # Model size mapping (common models)
    MODEL_SIZES = {
        "llama-7b": {"num_parameters": 7e9, "num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
        "llama-13b": {"num_parameters": 13e9, "num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
        "llama-70b": {"num_parameters": 70e9, "num_layers": 80, "hidden_size": 8192, "num_attention_heads": 64},
        "qwen-7b": {"num_parameters": 7e9, "num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
        "qwen-14b": {"num_parameters": 14e9, "num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
        "qwen-72b": {"num_parameters": 72e9, "num_layers": 80, "hidden_size": 8192, "num_attention_heads": 64},
    }
    
    # Try to match model name
    model_key = None
    for key in MODEL_SIZES:
        if key in model_name_or_path.lower():
            model_key = key
            break
    
    if model_key:
        model_config = MODEL_SIZES[model_key].copy()
        model_config["vocab_size"] = 32000  # Default
    else:
        # Default to 7B model
        model_config = MODEL_SIZES["llama-7b"].copy()
    
    # Load cluster configuration if provided
    cluster_info = None
    if cluster_config_path and os.path.exists(cluster_config_path):
        with open(cluster_config_path, "r") as f:
            cluster_info = json.load(f)
    
    return auto_configure_distributed_training(
        model=None,  # We don't have the actual model here
        model_config=model_config,
        cluster_info=cluster_info,
    )


# Example usage and testing
if __name__ == "__main__":
    # Example: Configure for Llama-70B on 8-node cluster
    model_spec = ModelSpec(
        num_parameters=70e9,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        vocab_size=32000,
        max_sequence_length=4096,
        dtype=torch.float16,
    )
    
    cluster_topology = ClusterTopology(
        num_nodes=8,
        gpus_per_node=8,
        gpu_memory_gb=80.0,
        interconnect_bandwidth_gbps=400.0,
        has_nvlink=True,
        has_infiniband=True,
    )
    
    optimizer = DistributedTrainingOptimizer(
        model_spec=model_spec,
        cluster_topology=cluster_topology,
        target_batch_size=4,
    )
    
    config = optimizer.get_recommended_config()
    
    print("Recommended Distributed Training Configuration:")
    print(json.dumps(config, indent=2, default=str))