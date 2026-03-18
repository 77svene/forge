"""
Distributed Training Optimizer for forge

Automatic distributed training configuration that selects optimal parallelism
strategy (FSDP, DeepSpeed, Megatron-LM) based on model size and cluster topology.
Includes automatic gradient checkpointing, activation recomputation, and communication optimization.
"""

import os
import math
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp._init_utils import ProcessGroupType

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
    from deepspeed.runtime.config import DeepSpeedConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from megatron import mpu, get_args
    from megatron.model import get_language_model
    from megatron.optimizer import get_megatron_optimizer
    from megatron.training import get_optimizer_param_scheduler
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParallelismStrategy(Enum):
    """Supported parallelism strategies."""
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    MEGATRON = "megatron"
    DDP = "ddp"
    HYBRID = "hybrid"


class CommunicationBackend(Enum):
    """Supported communication backends."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


@dataclass
class ClusterTopology:
    """Cluster topology information."""
    num_nodes: int = 1
    gpus_per_node: int = 1
    gpu_memory_gb: float = 80.0
    cpu_memory_gb: float = 512.0
    interconnect_type: str = "nvlink"  # nvlink, infiniband, ethernet
    interconnect_bandwidth_gbps: float = 300.0
    has_nvlink: bool = True
    has_infiniband: bool = False
    numa_nodes: int = 1
    cpu_cores_per_gpu: int = 8

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def is_multi_node(self) -> bool:
        return self.num_nodes > 1


@dataclass
class ModelProfile:
    """Model profiling information."""
    num_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    sequence_length: int = 2048
    activation_memory_per_layer_gb: float = 0.0
    parameter_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_state_memory_gb: float = 0.0
    total_memory_gb: float = 0.0

    def estimate_memory(self, dtype: torch.dtype = torch.float16):
        """Estimate memory requirements for model."""
        bytes_per_param = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        self.parameter_memory_gb = (self.num_parameters * bytes_per_param) / (1024**3)
        self.gradient_memory_gb = self.parameter_memory_gb
        self.optimizer_state_memory_gb = self.parameter_memory_gb * 2  # Adam states
        self.total_memory_gb = (
            self.parameter_memory_gb + 
            self.gradient_memory_gb + 
            self.optimizer_state_memory_gb
        )
        
        # Estimate activation memory per layer
        # This is a simplified estimation
        activation_bytes = (
            self.sequence_length * 
            self.hidden_size * 
            bytes_per_param * 
            4  # Approximate factor for intermediate activations
        )
        self.activation_memory_per_layer_gb = (activation_bytes * self.num_layers) / (1024**3)


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    strategy: ParallelismStrategy = ParallelismStrategy.FSDP
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    gradient_checkpointing: bool = True
    activation_recomputation: bool = True
    communication_backend: CommunicationBackend = CommunicationBackend.NCCL
    bucket_size_mb: int = 25
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    cpu_offload: bool = False
    zero_stage: int = 2
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    sub_group_size: int = 1000000000000
    reduce_bucket_size: int = 500000000
    allgather_bucket_size: int = 500000000
    use_megatron: bool = False
    use_te: bool = False  # Transformer Engine
    sequence_parallel: bool = False
    async_tensor_model_parallel_allreduce: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "strategy": self.strategy.value,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "activation_recomputation": self.activation_recomputation,
            "communication_backend": self.communication_backend.value,
            "mixed_precision": self.mixed_precision,
            "zero_stage": self.zero_stage,
            "use_megatron": self.use_megatron,
        }


class DistributedOptimizer:
    """
    Distributed Training Optimizer
    
    Automatically selects optimal parallelism strategy and configuration
    based on model size and cluster topology.
    """
    
    def __init__(
        self,
        model: nn.Module,
        cluster_topology: Optional[ClusterTopology] = None,
        model_profile: Optional[ModelProfile] = None,
        target_memory_utilization: float = 0.85,
        min_tensor_parallel_size: int = 1,
        max_tensor_parallel_size: Optional[int] = None,
    ):
        """
        Initialize Distributed Optimizer.
        
        Args:
            model: PyTorch model to optimize
            cluster_topology: Cluster topology information
            model_profile: Model profiling information
            target_memory_utilization: Target GPU memory utilization (0-1)
            min_tensor_parallel_size: Minimum tensor parallel size
            max_tensor_parallel_size: Maximum tensor parallel size
        """
        self.model = model
        self.cluster_topology = cluster_topology or self._detect_cluster_topology()
        self.model_profile = model_profile or self._profile_model(model)
        self.target_memory_utilization = target_memory_utilization
        self.min_tensor_parallel_size = min_tensor_parallel_size
        self.max_tensor_parallel_size = max_tensor_parallel_size or self.cluster_topology.gpus_per_node
        
        # Estimate memory requirements
        self.model_profile.estimate_memory()
        
        logger.info(f"Model profile: {self.model_profile.num_parameters:,} parameters")
        logger.info(f"Cluster: {self.cluster_topology.total_gpus} GPUs, "
                   f"{self.cluster_topology.gpu_memory_gb}GB per GPU")
    
    def _detect_cluster_topology(self) -> ClusterTopology:
        """Auto-detect cluster topology from environment."""
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Try to detect GPU memory
        gpu_memory_gb = 80.0  # Default A100
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Try to detect interconnect
        has_nvlink = False
        has_infiniband = False
        
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi", "topo", "-m"], 
                                  capture_output=True, text=True)
            if "NV" in result.stdout:
                has_nvlink = True
            if "IB" in result.stdout or "InfiniBand" in result.stdout:
                has_infiniband = True
        except:
            pass
        
        interconnect_type = "nvlink" if has_nvlink else "infiniband" if has_infiniband else "ethernet"
        
        return ClusterTopology(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            gpu_memory_gb=gpu_memory_gb,
            interconnect_type=interconnect_type,
            has_nvlink=has_nvlink,
            has_infiniband=has_infiniband,
        )
    
    def _profile_model(self, model: nn.Module) -> ModelProfile:
        """Profile model to get memory requirements."""
        num_params = sum(p.numel() for p in model.parameters())
        
        # Try to extract model architecture info
        hidden_size = 4096  # Default
        num_layers = 32
        num_attention_heads = 32
        vocab_size = 32000
        sequence_length = 2048
        
        # Attempt to extract from model config
        if hasattr(model, "config"):
            config = model.config
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_layers = getattr(config, "num_hidden_layers", num_layers)
            num_attention_heads = getattr(config, "num_attention_heads", num_attention_heads)
            vocab_size = getattr(config, "vocab_size", vocab_size)
            sequence_length = getattr(config, "max_position_embeddings", sequence_length)
        
        # Count transformer layers
        transformer_layers = 0
        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                transformer_layers += 1
        if transformer_layers > 0:
            num_layers = transformer_layers
        
        return ModelProfile(
            num_parameters=num_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
        )
    
    def _calculate_optimal_parallelism(self) -> Tuple[int, int, int]:
        """
        Calculate optimal tensor, pipeline, and data parallelism sizes.
        
        Returns:
            Tuple of (tensor_parallel_size, pipeline_parallel_size, data_parallel_size)
        """
        total_gpus = self.cluster_topology.total_gpus
        gpu_memory_gb = self.cluster_topology.gpu_memory_gb
        available_memory_gb = gpu_memory_gb * self.target_memory_utilization
        
        # Memory required per GPU for model parameters, gradients, and optimizer states
        memory_per_gpu = self.model_profile.total_memory_gb
        
        # Add activation memory if not using gradient checkpointing
        if not self.gradient_checkpointing:
            memory_per_gpu += self.model_profile.activation_memory_per_layer_gb
        
        # Calculate minimum tensor parallel size needed
        min_tp_for_memory = max(
            1,
            math.ceil(memory_per_gpu / available_memory_gb)
        )
        
        # Consider model architecture constraints
        # Tensor parallel size should divide hidden size and attention heads
        hidden_size = self.model_profile.hidden_size
        num_heads = self.model_profile.num_attention_heads
        
        # Find valid tensor parallel sizes
        valid_tp_sizes = []
        for tp in range(self.min_tensor_parallel_size, 
                       min(self.max_tensor_parallel_size + 1, total_gpus + 1)):
            if hidden_size % tp == 0 and num_heads % tp == 0:
                valid_tp_sizes.append(tp)
        
        if not valid_tp_sizes:
            valid_tp_sizes = [1]
        
        # Select tensor parallel size
        tensor_parallel_size = min(
            max(valid_tp_sizes),
            max(min_tp_for_memory, self.min_tensor_parallel_size)
        )
        
        # For multi-node, prefer pipeline parallelism
        if self.cluster_topology.is_multi_node:
            # Use tensor parallel within nodes, pipeline parallel across nodes
            tensor_parallel_size = min(tensor_parallel_size, self.cluster_topology.gpus_per_node)
            pipeline_parallel_size = min(
                self.cluster_topology.num_nodes,
                total_gpus // tensor_parallel_size
            )
        else:
            pipeline_parallel_size = 1
        
        # Calculate data parallel size
        data_parallel_size = total_gpus // (tensor_parallel_size * pipeline_parallel_size)
        
        # Ensure we have at least 1 GPU for data parallel
        if data_parallel_size < 1:
            data_parallel_size = 1
            tensor_parallel_size = total_gpus
            pipeline_parallel_size = 1
        
        return tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    
    def _select_strategy(self) -> ParallelismStrategy:
        """Select optimal parallelism strategy."""
        total_params = self.model_profile.num_parameters
        total_gpus = self.cluster_topology.total_gpus
        gpu_memory_gb = self.cluster_topology.gpu_memory_gb
        
        # Memory required per GPU
        memory_required_gb = self.model_profile.total_memory_gb / total_gpus
        
        # Decision logic
        if total_params < 1e9:  # < 1B parameters
            if total_gpus <= 8:
                return ParallelismStrategy.DDP
            else:
                return ParallelismStrategy.FSDP
        
        elif total_params < 10e9:  # 1B - 10B parameters
            if memory_required_gb > gpu_memory_gb * 0.8:
                # Need memory optimization
                if DEEPSPEED_AVAILABLE:
                    return ParallelismStrategy.DEEPSPEED
                else:
                    return ParallelismStrategy.FSDP
            else:
                return ParallelismStrategy.FSDP
        
        elif total_params < 100e9:  # 10B - 100B parameters
            if MEGATRON_AVAILABLE and total_gpus >= 8:
                # Use Megatron for large models with enough GPUs
                return ParallelismStrategy.MEGATRON
            elif DEEPSPEED_AVAILABLE:
                return ParallelismStrategy.DEEPSPEED
            else:
                return ParallelismStrategy.FSDP
        
        else:  # > 100B parameters
            if MEGATRON_AVAILABLE:
                return ParallelismStrategy.MEGATRON
            elif DEEPSPEED_AVAILABLE:
                return ParallelismStrategy.DEEPSPEED
            else:
                # Fall back to FSDP with aggressive memory optimization
                return ParallelismStrategy.FSDP
    
    def _optimize_communication(self, config: DistributedConfig):
        """Optimize communication patterns based on cluster topology."""
        if self.cluster_topology.interconnect_type == "nvlink":
            config.overlap_comm = True
            config.bucket_size_mb = 50
            config.reduce_bucket_size = 500000000
            config.allgather_bucket_size = 500000000
        elif self.cluster_topology.interconnect_type == "infiniband":
            config.overlap_comm = True
            config.bucket_size_mb = 25
            config.reduce_bucket_size = 250000000
            config.allgather_bucket_size = 250000000
        else:  # Ethernet
            config.overlap_comm = False
            config.bucket_size_mb = 10
            config.reduce_bucket_size = 100000000
            config.allgather_bucket_size = 100000000
    
    def configure(self) -> DistributedConfig:
        """
        Configure distributed training.
        
        Returns:
            DistributedConfig with optimal configuration
        """
        # Calculate optimal parallelism
        tp_size, pp_size, dp_size = self._calculate_optimal_parallelism()
        
        # Select strategy
        strategy = self._select_strategy()
        
        # Determine gradient checkpointing
        memory_per_gpu = self.model_profile.total_memory_gb / (tp_size * pp_size)
        available_memory = self.cluster_topology.gpu_memory_gb * self.target_memory_utilization
        gradient_checkpointing = memory_per_gpu > available_memory * 0.7
        
        # Create configuration
        config = DistributedConfig(
            strategy=strategy,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            gradient_checkpointing=gradient_checkpointing,
            activation_recomputation=gradient_checkpointing,
            mixed_precision=True,
        )
        
        # Optimize based on strategy
        if strategy == ParallelismStrategy.DEEPSPEED and DEEPSPEED_AVAILABLE:
            config.zero_stage = 3 if self.model_profile.num_parameters > 10e9 else 2
            config.cpu_offload = memory_per_gpu > available_memory * 0.9
        
        elif strategy == ParallelismStrategy.MEGATRON and MEGATRON_AVAILABLE:
            config.use_megatron = True
            config.sequence_parallel = tp_size > 1
            config.async_tensor_model_parallel_allreduce = True
        
        elif strategy == ParallelismStrategy.FSDP:
            # FSDP-specific optimizations
            config.contiguous_gradients = True
            config.sub_group_size = min(1000000000000, self.model_profile.num_parameters // 100)
        
        # Optimize communication
        self._optimize_communication(config)
        
        logger.info(f"Selected strategy: {strategy.value}")
        logger.info(f"Parallelism: TP={tp_size}, PP={pp_size}, DP={dp_size}")
        logger.info(f"Gradient checkpointing: {gradient_checkpointing}")
        
        return config
    
    def apply_fsdp(self, model: nn.Module, config: DistributedConfig) -> FSDP:
        """Apply FSDP to model with given configuration."""
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed process group not initialized")
        
        # Mixed precision policy
        if config.mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None
        
        # Auto wrap policy
        if hasattr(model, "layers") or hasattr(model, "encoder") or hasattr(model, "decoder"):
            # Transformer-based model
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                auto_wrap_policy = transformer_auto_wrap_policy
                transformer_layer_cls = {LlamaDecoderLayer}
            except ImportError:
                auto_wrap_policy = size_based_auto_wrap_policy
                transformer_layer_cls = None
        else:
            auto_wrap_policy = size_based_auto_wrap_policy
            transformer_layer_cls = None
        
        # Sharding strategy
        if config.tensor_parallel_size > 1:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        
        # Apply gradient checkpointing wrapper
        if config.gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        # Apply FSDP
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=None,  # Could add CPU offload here
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            forward_prefetch=True,
            limit_all_gathers=True,
        )
        
        return fsdp_model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model."""
        # Try to use HuggingFace's gradient checkpointing if available
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            return model
        
        # Manual implementation for custom models
        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Wrap transformer layers with checkpoint
                module = checkpoint_wrapper(
                    module,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
        
        return model
    
    def get_deepspeed_config(self, config: DistributedConfig) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed not available")
        
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "zero_optimization": {
                "stage": config.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": config.allgather_bucket_size,
                "overlap_comm": config.overlap_comm,
                "reduce_scatter": True,
                "reduce_bucket_size": config.reduce_bucket_size,
                "contiguous_gradients": config.contiguous_gradients,
                "sub_group_size": config.sub_group_size,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                }
            },
            "fp16": {
                "enabled": config.mixed_precision,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "wall_clock_breakdown": False,
        }
        
        if config.cpu_offload:
            ds_config["zero_optimization"]["cpu_offload"] = True
            ds_config["zero_optimization"]["cpu_offload_params"] = True
            ds_config["zero_optimization"]["cpu_offload_use_pin_memory"] = True
        
        return ds_config
    
    def get_megatron_args(self, config: DistributedConfig) -> Dict[str, Any]:
        """Generate Megatron-LM arguments."""
        if not MEGATRON_AVAILABLE:
            raise RuntimeError("Megatron-LM not available")
        
        megatron_args = {
            "tensor_model_parallel_size": config.tensor_parallel_size,
            "pipeline_model_parallel_size": config.pipeline_parallel_size,
            "sequence_parallel": config.sequence_parallel,
            "async_tensor_model_parallel_allreduce": config.async_tensor_model_parallel_allreduce,
            "use_cpu_initialization": False,
            "use_flash_attn": True,
            "recompute_granularity": "full" if config.activation_recomputation else None,
            "recompute_method": "uniform" if config.activation_recomputation else None,
            "recompute_num_layers": 1 if config.activation_recomputation else None,
            "distribute_saved_activations": config.activation_recomputation,
            "overlap_p2p_comm": True,
            "pipeline_dtype": torch.float16 if config.mixed_precision else torch.float32,
        }
        
        return megatron_args
    
    def estimate_throughput(self, config: DistributedConfig) -> Dict[str, float]:
        """
        Estimate training throughput.
        
        Returns:
            Dictionary with throughput estimates
        """
        # Simplified throughput estimation
        total_gpus = self.cluster_topology.total_gpus
        gpu_tflops = 312  # A100 tensor TFLOPS for FP16
        
        # Model FLOPs per iteration
        model_flops = 6 * self.model_profile.num_parameters * self.model_profile.sequence_length
        
        # Parallelism efficiency
        tp_efficiency = 0.9 if config.tensor_parallel_size <= 4 else 0.7
        pp_efficiency = 0.85 if config.pipeline_parallel_size <= 4 else 0.6
        dp_efficiency = 0.95
        
        total_efficiency = tp_efficiency * pp_efficiency * dp_efficiency
        
        # Estimate samples per second
        samples_per_second = (
            (total_gpus * gpu_tflops * 1e12 * total_efficiency) / 
            (model_flops * config.gradient_accumulation_steps)
        )
        
        # Estimate tokens per second
        tokens_per_second = samples_per_second * self.model_profile.sequence_length
        
        return {
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "gpu_utilization": total_efficiency,
            "estimated_tflops_per_gpu": gpu_tflops * total_efficiency,
        }
    
    def save_config(self, config: DistributedConfig, path: str):
        """Save configuration to file."""
        config_dict = config.to_dict()
        config_dict.update({
            "cluster_topology": {
                "num_nodes": self.cluster_topology.num_nodes,
                "gpus_per_node": self.cluster_topology.gpus_per_node,
                "total_gpus": self.cluster_topology.total_gpus,
                "gpu_memory_gb": self.cluster_topology.gpu_memory_gb,
                "interconnect_type": self.cluster_topology.interconnect_type,
            },
            "model_profile": {
                "num_parameters": self.model_profile.num_parameters,
                "num_layers": self.model_profile.num_layers,
                "hidden_size": self.model_profile.hidden_size,
                "parameter_memory_gb": self.model_profile.parameter_memory_gb,
                "total_memory_gb": self.model_profile.total_memory_gb,
            }
        })
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")


def auto_distributed_setup(
    model: nn.Module,
    cluster_info: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> Tuple[nn.Module, DistributedConfig]:
    """
    Automatic distributed training setup.
    
    Args:
        model: PyTorch model
        cluster_info: Cluster information dictionary
        config_path: Path to save configuration
        
    Returns:
        Tuple of (distributed_model, config)
    """
    # Create cluster topology
    if cluster_info:
        topology = ClusterTopology(**cluster_info)
    else:
        topology = ClusterTopology()
    
    # Create optimizer
    optimizer = DistributedOptimizer(
        model=model,
        cluster_topology=topology,
    )
    
    # Configure
    config = optimizer.configure()
    
    # Save configuration if path provided
    if config_path:
        optimizer.save_config(config, config_path)
    
    # Apply distributed training
    if config.strategy == ParallelismStrategy.FSDP:
        distributed_model = optimizer.apply_fsdp(model, config)
    elif config.strategy == ParallelismStrategy.DEEPSPEED and DEEPSPEED_AVAILABLE:
        ds_config = optimizer.get_deepspeed_config(config)
        distributed_model, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
    elif config.strategy == ParallelismStrategy.MEGATRON and MEGATRON_AVAILABLE:
        # Megatron requires special initialization
        megatron_args = optimizer.get_megatron_args(config)
        # This would need to integrate with Megatron's initialization
        distributed_model = model  # Placeholder
    else:
        # Fall back to DDP
        distributed_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )
    
    return distributed_model, config


# Integration with existing forge scripts
def setup_distributed_for_training(
    model: nn.Module,
    training_args: Any,
    model_args: Any,
) -> Tuple[nn.Module, DistributedConfig]:
    """
    Setup distributed training for forge training scripts.
    
    Args:
        model: Model to train
        training_args: Training arguments
        model_args: Model arguments
        
    Returns:
        Tuple of (distributed_model, config)
    """
    # Extract cluster info from environment
    cluster_info = {
        "num_nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
        "gpus_per_node": torch.cuda.device_count() if torch.cuda.is_available() else 1,
        "gpu_memory_gb": 80.0,  # Default, could be detected
    }
    
    # Create optimizer
    optimizer = DistributedOptimizer(
        model=model,
        cluster_topology=ClusterTopology(**cluster_info),
    )
    
    # Configure with training args influence
    config = optimizer.configure()
    
    # Override with training args if specified
    if hasattr(training_args, "gradient_checkpointing"):
        config.gradient_checkpointing = training_args.gradient_checkpointing
    if hasattr(training_args, "fp16") or hasattr(training_args, "bf16"):
        config.mixed_precision = True
    
    # Apply distributed training
    if config.strategy == ParallelismStrategy.FSDP:
        distributed_model = optimizer.apply_fsdp(model, config)
    elif config.strategy == ParallelismStrategy.DEEPSPEED and DEEPSPEED_AVAILABLE:
        ds_config = optimizer.get_deepspeed_config(config)
        distributed_model, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
    else:
        distributed_model = model
    
    return distributed_model, config


# Utility functions for existing scripts
def get_optimal_parallel_config(
    num_parameters: int,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
) -> Dict[str, int]:
    """
    Get optimal parallel configuration for given constraints.
    
    Args:
        num_parameters: Number of model parameters
        num_gpus: Number of GPUs available
        gpu_memory_gb: GPU memory in GB
        
    Returns:
        Dictionary with optimal configuration
    """
    # Simplified heuristic
    memory_per_param_gb = 12 / (1024**3)  # ~12 bytes per parameter for training
    memory_required_gb = num_parameters * memory_per_param_gb
    memory_per_gpu = memory_required_gb / num_gpus
    
    if memory_per_gpu < gpu_memory_gb * 0.5:
        # Fits comfortably, use data parallelism
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "strategy": "ddp",
        }
    elif memory_per_gpu < gpu_memory_gb * 0.8:
        # Needs some memory optimization
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "strategy": "fsdp",
            "gradient_checkpointing": True,
        }
    else:
        # Needs model parallelism
        tp_size = min(8, math.ceil(memory_per_gpu / (gpu_memory_gb * 0.7)))
        dp_size = num_gpus // tp_size
        return {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": 1,
            "data_parallel_size": max(1, dp_size),
            "strategy": "fsdp",
            "gradient_checkpointing": True,
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Optimizer")
    parser.add_argument("--model-size", type=float, default=7.0, help="Model size in billions")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=float, default=80.0, help="GPU memory in GB")
    args = parser.parse_args()
    
    # Get optimal configuration
    config = get_optimal_parallel_config(
        num_parameters=int(args.model_size * 1e9),
        num_gpus=args.num_gpus,
        gpu_memory_gb=args.gpu_memory,
    )
    
    print("Optimal Distributed Training Configuration:")
    print(json.dumps(config, indent=2))