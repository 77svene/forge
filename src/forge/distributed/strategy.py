"""
src/forge/distributed/strategy.py
Distributed Training Optimizer for automatic parallelism strategy selection.
"""

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Supported parallelism strategies."""
    DDP = "ddp"  # Data Parallelism only
    FSDP = "fsdp"  # Fully Sharded Data Parallelism
    DEEPSPEED = "deepspeed"  # DeepSpeed ZeRO
    MEGATRON = "megatron"  # Megatron-LM style tensor/pipeline parallelism
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class ClusterTopology:
    """Hardware cluster information."""
    num_nodes: int
    gpus_per_node: int
    gpu_memory_gb: float
    gpu_flops: float  # TFLOPS per GPU
    inter_node_bandwidth_gbps: float
    intra_node_bandwidth_gbps: float
    nvlink_available: bool = False
    infiniband_available: bool = False

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def total_memory_gb(self) -> float:
        return self.total_gpus * self.gpu_memory_gb


@dataclass
class ModelCharacteristics:
    """Model architecture characteristics."""
    num_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    intermediate_size: Optional[int] = None
    max_sequence_length: int = 2048
    activation_memory_per_token_mb: float = 0.0  # Estimated activation memory

    @property
    def parameter_memory_gb(self) -> float:
        """Estimate memory for parameters in GB (assuming fp16)."""
        return self.num_parameters * 2 / (1024 ** 3)

    @property
    def gradient_memory_gb(self) -> float:
        """Estimate memory for gradients (same as parameters)."""
        return self.parameter_memory_gb

    @property
    def optimizer_state_memory_gb(self) -> float:
        """Estimate memory for optimizer states (Adam: 2 states per parameter)."""
        return self.num_parameters * 8 / (1024 ** 3)  # 8 bytes per param for fp32 states

    def estimate_training_memory_gb(self, batch_size: int, sequence_length: int) -> float:
        """Estimate total training memory including activations."""
        # Rough estimation: parameters + gradients + optimizer states + activations
        base_memory = self.parameter_memory_gb + self.gradient_memory_gb + self.optimizer_state_memory_gb
        
        # Activation memory estimation (simplified)
        # For transformer: ~ 2 * batch_size * seq_len * hidden_size * num_layers * bytes_per_element
        activation_memory = (
            2 * batch_size * sequence_length * self.hidden_size * self.num_layers * 2 / (1024 ** 3)
        )
        
        return base_memory + activation_memory


class CommunicationPatternAnalyzer:
    """Analyzes communication patterns for different parallelism strategies."""
    
    @staticmethod
    def estimate_allreduce_time(
        data_size_gb: float,
        num_gpus: int,
        bandwidth_gbps: float,
        latency_ms: float = 0.1
    ) -> float:
        """Estimate AllReduce time in seconds using ring algorithm."""
        # Ring AllReduce: 2*(n-1)/n * data_size / bandwidth
        if num_gpus <= 1:
            return 0.0
        
        # Convert GB to Gb (1 byte = 8 bits)
        data_size_gb_bits = data_size_gb * 8
        communication_time = 2 * (num_gpus - 1) / num_gpus * data_size_gb_bits / bandwidth_gbps
        return communication_time + latency_ms / 1000
    
    @staticmethod
    def estimate_tensor_parallel_communication(
        model: ModelCharacteristics,
        tensor_parallel_size: int,
        sequence_length: int,
        batch_size: int,
        bandwidth_gbps: float
    ) -> float:
        """Estimate communication overhead for tensor parallelism."""
        # Tensor parallelism requires all-reduce of activations
        # Each layer communicates activations of size: batch_size * seq_len * hidden_size
        activation_size_gb = (
            batch_size * sequence_length * model.hidden_size * 2 / (1024 ** 3)
        )
        
        # Two communications per transformer layer (attention + MLP)
        total_communication = activation_size_gb * 2 * model.num_layers
        
        return CommunicationPatternAnalyzer.estimate_allreduce_time(
            total_communication, tensor_parallel_size, bandwidth_gbps
        )
    
    @staticmethod
    def estimate_pipeline_parallel_communication(
        model: ModelCharacteristics,
        pipeline_parallel_size: int,
        micro_batch_size: int,
        sequence_length: int,
        bandwidth_gbps: float
    ) -> float:
        """Estimate communication overhead for pipeline parallelism."""
        # Pipeline parallelism sends activations between pipeline stages
        # Each stage boundary communicates: micro_batch_size * seq_len * hidden_size
        activation_size_gb = (
            micro_batch_size * sequence_length * model.hidden_size * 2 / (1024 ** 3)
        )
        
        # Number of pipeline boundaries
        num_boundaries = pipeline_parallel_size - 1
        
        return CommunicationPatternAnalyzer.estimate_allreduce_time(
            activation_size_gb * num_boundaries, 2, bandwidth_gbps  # Point-to-point between 2 GPUs
        )


class DistributedStrategySelector:
    """Automatic distributed training strategy selector."""
    
    # Memory thresholds (in GB)
    MEMORY_SAFETY_FACTOR = 0.85  # Use 85% of available memory
    MIN_EFFICIENCY_THRESHOLD = 0.7  # Minimum parallel efficiency
    
    # Model size categories (in billions of parameters)
    SMALL_MODEL_THRESHOLD = 1e9  # 1B
    MEDIUM_MODEL_THRESHOLD = 7e9  # 7B
    LARGE_MODEL_THRESHOLD = 70e9  # 70B
    
    def __init__(
        self,
        model: nn.Module,
        cluster: ClusterTopology,
        target_batch_size: int = 32,
        target_sequence_length: int = 2048,
        precision: str = "bf16",
        enable_gradient_checkpointing: bool = True,
        enable_activation_recomputation: bool = False,
    ):
        self.model = model
        self.cluster = cluster
        self.target_batch_size = target_batch_size
        self.target_sequence_length = target_sequence_length
        self.precision = precision
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_activation_recomputation = enable_activation_recomputation
        
        # Analyze model
        self.model_characteristics = self._analyze_model()
        
        # Communication analyzer
        self.comm_analyzer = CommunicationPatternAnalyzer()
        
        logger.info(f"Initialized DistributedStrategySelector for model with "
                    f"{self.model_characteristics.num_parameters / 1e9:.2f}B parameters")
        logger.info(f"Cluster: {cluster.num_nodes} nodes, {cluster.gpus_per_node} GPUs/node, "
                    f"{cluster.gpu_memory_gb}GB GPU memory")
    
    def _analyze_model(self) -> ModelCharacteristics:
        """Analyze model architecture and characteristics."""
        # Count parameters
        num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # Try to extract architecture information
        # This is model-specific and might need adjustment
        num_layers = 0
        hidden_size = 0
        num_attention_heads = 0
        vocab_size = 0
        intermediate_size = None
        
        # Common attribute names in transformer models
        for name, module in self.model.named_modules():
            if hasattr(module, 'num_layers'):
                num_layers = module.num_layers
            elif hasattr(module, 'n_layers'):
                num_layers = module.n_layers
            elif hasattr(module, 'num_hidden_layers'):
                num_layers = module.num_hidden_layers
            
            if hasattr(module, 'hidden_size'):
                hidden_size = module.hidden_size
            elif hasattr(module, 'd_model'):
                hidden_size = module.d_model
            elif hasattr(module, 'n_embd'):
                hidden_size = module.n_embd
            
            if hasattr(module, 'num_attention_heads'):
                num_attention_heads = module.num_attention_heads
            elif hasattr(module, 'n_head'):
                num_attention_heads = module.n_head
            elif hasattr(module, 'num_heads'):
                num_attention_heads = module.num_heads
            
            if hasattr(module, 'vocab_size'):
                vocab_size = module.vocab_size
            
            if hasattr(module, 'intermediate_size'):
                intermediate_size = module.intermediate_size
            elif hasattr(module, 'n_inner'):
                intermediate_size = module.n_inner
        
        # Fallbacks if not found
        if num_layers == 0:
            # Estimate from number of parameters
            num_layers = max(1, int(math.log10(num_parameters / 1e6)))
        
        if hidden_size == 0:
            # Rough estimate based on parameter count
            hidden_size = int(math.sqrt(num_parameters / (6 * num_layers)))
        
        if num_attention_heads == 0:
            # Common ratio: hidden_size / 64
            num_attention_heads = max(1, hidden_size // 64)
        
        # Estimate activation memory per token
        # For transformer: ~ 2 * hidden_size * num_layers * bytes_per_element
        bytes_per_element = 2 if self.precision in ["fp16", "bf16"] else 4
        activation_memory_per_token_mb = (
            2 * hidden_size * num_layers * bytes_per_element / (1024 ** 2)
        )
        
        return ModelCharacteristics(
            num_parameters=num_parameters,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            max_sequence_length=self.target_sequence_length,
            activation_memory_per_token_mb=activation_memory_per_token_mb,
        )
    
    def _calculate_memory_requirements(self, batch_size: int) -> Dict[str, float]:
        """Calculate memory requirements for different components."""
        seq_len = self.target_sequence_length
        
        # Parameter memory
        param_memory = self.model_characteristics.parameter_memory_gb
        
        # Gradient memory (same as parameters)
        grad_memory = self.model_characteristics.gradient_memory_gb
        
        # Optimizer state memory
        optimizer_memory = self.model_characteristics.optimizer_state_memory_gb
        
        # Activation memory (with gradient checkpointing consideration)
        base_activation_memory = (
            batch_size * seq_len * 
            self.model_characteristics.activation_memory_per_token_mb / 1024
        )
        
        if self.enable_gradient_checkpointing:
            # Gradient checkpointing reduces activation memory by ~60-70%
            activation_memory = base_activation_memory * 0.35
        else:
            activation_memory = base_activation_memory
        
        # Total memory
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
        
        return {
            "parameters_gb": param_memory,
            "gradients_gb": grad_memory,
            "optimizer_states_gb": optimizer_memory,
            "activations_gb": activation_memory,
            "total_gb": total_memory,
        }
    
    def _select_data_parallelism_strategy(self) -> Tuple[ParallelStrategy, Dict[str, Any]]:
        """Select optimal data parallelism strategy."""
        memory_req = self._calculate_memory_requirements(self.target_batch_size)
        total_memory_needed = memory_req["total_gb"]
        available_memory = self.cluster.gpu_memory_gb * self.MEMORY_SAFETY_FACTOR
        
        if total_memory_needed <= available_memory:
            # Model fits in single GPU memory
            if self.cluster.total_gpus > 1:
                # Use DDP for simplicity and good performance
                logger.info("Selected DDP: Model fits in single GPU memory")
                return ParallelStrategy.DDP, {
                    "find_unused_parameters": False,
                    "gradient_as_bucket_view": True,
                    "static_graph": True,
                }
            else:
                # Single GPU, no parallelism needed
                return ParallelStrategy.DDP, {}
        else:
            # Need memory optimization
            if self.model_characteristics.num_parameters <= self.MEDIUM_MODEL_THRESHOLD:
                # Medium models: FSDP is good balance
                logger.info("Selected FSDP: Medium model requiring memory optimization")
                return ParallelStrategy.FSDP, {
                    "sharding_strategy": "FULL_SHARD",
                    "cpu_offload": False,
                    "mixed_precision": True,
                    "activation_checkpointing": self.enable_gradient_checkpointing,
                    "sync_module_states": True,
                }
            else:
                # Large models: DeepSpeed ZeRO for better memory optimization
                logger.info("Selected DeepSpeed: Large model requiring advanced memory optimization")
                return ParallelStrategy.DEEPSPEED, {
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {"device": "cpu"},
                        "offload_param": {"device": "cpu"},
                        "overlap_comm": True,
                        "contiguous_gradients": True,
                        "sub_group_size": 1e9,
                        "reduce_bucket_size": 5e8,
                        "stage3_prefetch_bucket_size": 5e8,
                        "stage3_param_persistence_threshold": 1e6,
                    },
                    "gradient_accumulation_steps": 1,
                    "gradient_clipping": 1.0,
                    "steps_per_print": 100,
                }
    
    def _select_model_parallelism_strategy(self) -> Tuple[ParallelStrategy, Dict[str, Any]]:
        """Select optimal model parallelism strategy for very large models."""
        # For models that don't fit in single GPU even with FSDP/DeepSpeed
        memory_req = self._calculate_memory_requirements(1)  # Per GPU memory
        available_memory = self.cluster.gpu_memory_gb * self.MEMORY_SAFETY_FACTOR
        
        # Calculate minimum tensor parallelism needed
        min_tensor_parallel = math.ceil(memory_req["parameters_gb"] / available_memory)
        
        # Consider cluster topology
        if self.cluster.num_nodes == 1:
            # Single node: use tensor parallelism within node
            tensor_parallel_size = min(min_tensor_parallel, self.cluster.gpus_per_node)
            pipeline_parallel_size = 1
            
            # Align tensor parallel size with attention heads
            if self.model_characteristics.num_attention_heads % tensor_parallel_size != 0:
                # Find largest divisor of attention heads <= tensor_parallel_size
                for tp in range(tensor_parallel_size, 0, -1):
                    if self.model_characteristics.num_attention_heads % tp == 0:
                        tensor_parallel_size = tp
                        break
            
            logger.info(f"Selected Megatron-LM: Single node with tensor parallel size {tensor_parallel_size}")
            return ParallelStrategy.MEGATRON, {
                "tensor_model_parallel_size": tensor_parallel_size,
                "pipeline_model_parallel_size": pipeline_parallel_size,
                "virtual_pipeline_model_parallel_size": None,
                "sequence_parallel": True,
                "async_tensor_model_parallel_allreduce": True,
                "gradient_accumulation_fusion": True,
                "use_flash_attn": True,
            }
        else:
            # Multi-node: use hybrid parallelism
            # Tensor parallel within node, pipeline parallel across nodes
            tensor_parallel_size = min(min_tensor_parallel, self.cluster.gpus_per_node)
            pipeline_parallel_size = min(
                self.cluster.num_nodes,
                math.ceil(self.model_characteristics.num_layers / 8)  # Reasonable pipeline stages
            )
            
            # Calculate micro-batch size for pipeline parallelism
            global_batch_size = self.target_batch_size
            micro_batch_size = max(1, global_batch_size // (tensor_parallel_size * pipeline_parallel_size))
            
            logger.info(f"Selected Hybrid: Tensor parallel {tensor_parallel_size}, "
                       f"Pipeline parallel {pipeline_parallel_size}")
            
            return ParallelStrategy.HYBRID, {
                "tensor_model_parallel_size": tensor_parallel_size,
                "pipeline_model_parallel_size": pipeline_parallel_size,
                "data_parallel_size": self.cluster.total_gpus // (tensor_parallel_size * pipeline_parallel_size),
                "micro_batch_size": micro_batch_size,
                "global_batch_size": global_batch_size,
                "use_distributed_optimizer": True,
                "overlap_grad_reduce": True,
                "overlap_param_gather": True,
            }
    
    def _estimate_parallel_efficiency(self, strategy: ParallelStrategy, config: Dict[str, Any]) -> float:
        """Estimate parallel efficiency for given strategy."""
        if strategy == ParallelStrategy.DDP:
            # DDP efficiency: 1 - communication_overhead
            data_size_gb = self.model_characteristics.parameter_memory_gb
            comm_time = self.comm_analyzer.estimate_allreduce_time(
                data_size_gb,
                self.cluster.total_gpus,
                self.cluster.inter_node_bandwidth_gbps
            )
            compute_time = self.model_characteristics.num_parameters * 2 / (self.cluster.gpu_flops * 1e12)
            efficiency = compute_time / (compute_time + comm_time)
            
        elif strategy == ParallelStrategy.FSDP:
            # FSDP has additional overhead from sharding
            efficiency = 0.85  # Typical FSDP efficiency
            
        elif strategy == ParallelStrategy.DEEPSPEED:
            # DeepSpeed ZeRO efficiency depends on stage
            stage = config.get("zero_optimization", {}).get("stage", 3)
            efficiency = {1: 0.9, 2: 0.85, 3: 0.8}.get(stage, 0.8)
            
        elif strategy == ParallelStrategy.MEGATRON:
            # Tensor parallelism efficiency
            tp_size = config.get("tensor_model_parallel_size", 1)
            if tp_size > 1:
                comm_time = self.comm_analyzer.estimate_tensor_parallel_communication(
                    self.model_characteristics,
                    tp_size,
                    self.target_sequence_length,
                    self.target_batch_size,
                    self.cluster.intra_node_bandwidth_gbps
                )
                compute_time = self.model_characteristics.num_parameters * 2 / (self.cluster.gpu_flops * 1e12)
                efficiency = compute_time / (compute_time + comm_time * tp_size)
            else:
                efficiency = 0.95
                
        elif strategy == ParallelStrategy.HYBRID:
            # Hybrid parallelism efficiency
            tp_size = config.get("tensor_model_parallel_size", 1)
            pp_size = config.get("pipeline_model_parallel_size", 1)
            
            # Tensor parallel efficiency
            tp_comm_time = self.comm_analyzer.estimate_tensor_parallel_communication(
                self.model_characteristics,
                tp_size,
                self.target_sequence_length,
                self.target_batch_size // pp_size,
                self.cluster.intra_node_bandwidth_gbps
            )
            
            # Pipeline parallel efficiency (bubble overhead)
            num_micro_batches = config.get("global_batch_size", 32) // config.get("micro_batch_size", 1)
            pipeline_efficiency = (num_micro_batches) / (num_micro_batches + pp_size - 1)
            
            compute_time = self.model_characteristics.num_parameters * 2 / (self.cluster.gpu_flops * 1e12)
            total_comm_time = tp_comm_time * pp_size
            
            efficiency = pipeline_efficiency * (compute_time / (compute_time + total_comm_time))
            
        else:
            efficiency = 0.5
        
        return min(max(efficiency, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _optimize_communication_patterns(self, strategy: ParallelStrategy, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize communication patterns based on cluster topology."""
        optimized_config = config.copy()
        
        if strategy in [ParallelStrategy.DDP, ParallelStrategy.FSDP]:
            # Optimize bucket sizes for AllReduce
            if self.cluster.infiniband_available:
                # Larger buckets for high-bandwidth networks
                optimized_config["bucket_cap_mb"] = 25
                optimized_config["gradient_as_bucket_view"] = True
            else:
                # Smaller buckets for Ethernet
                optimized_config["bucket_cap_mb"] = 10
                optimized_config["gradient_as_bucket_view"] = False
            
            # Enable gradient compression if available
            if hasattr(torch.distributed, "GradBucket"):
                optimized_config["gradient_compression"] = True
        
        elif strategy == ParallelStrategy.DEEPSPEED:
            # Optimize DeepSpeed communication
            if "zero_optimization" in optimized_config:
                zero_config = optimized_config["zero_optimization"]
                
                # Adjust bucket sizes based on GPU memory
                gpu_memory_gb = self.cluster.gpu_memory_gb
                if gpu_memory_gb >= 80:
                    zero_config["reduce_bucket_size"] = 1e9
                    zero_config["stage3_prefetch_bucket_size"] = 1e9
                elif gpu_memory_gb >= 40:
                    zero_config["reduce_bucket_size"] = 5e8
                    zero_config["stage3_prefetch_bucket_size"] = 5e8
                else:
                    zero_config["reduce_bucket_size"] = 2e8
                    zero_config["stage3_prefetch_bucket_size"] = 2e8
                
                # Enable communication overlap
                zero_config["overlap_comm"] = True
                zero_config["overlap_grad_reduce"] = True
                zero_config["overlap_param_gather"] = True
        
        elif strategy in [ParallelStrategy.MEGATRON, ParallelStrategy.HYBRID]:
            # Optimize Megatron-LM communication
            if self.cluster.nvlink_available:
                optimized_config["use_nvlink"] = True
                optimized_config["sequence_parallel"] = True
            else:
                optimized_config["use_nvlink"] = False
                optimized_config["sequence_parallel"] = False
            
            # Enable async communication
            optimized_config["async_tensor_model_parallel_allreduce"] = True
            optimized_config["gradient_accumulation_fusion"] = True
        
        return optimized_config
    
    def select_optimal_strategy(self) -> Dict[str, Any]:
        """
        Select optimal distributed training strategy.
        
        Returns:
            Dictionary containing strategy configuration.
        """
        logger.info("Analyzing model and cluster for optimal strategy selection...")
        
        # Check if model fits in single GPU
        memory_req = self._calculate_memory_requirements(self.target_batch_size)
        available_memory = self.cluster.gpu_memory_gb * self.MEMORY_SAFETY_FACTOR
        
        if memory_req["total_gb"] <= available_memory:
            # Simple data parallelism
            strategy, config = self._select_data_parallelism_strategy()
        elif memory_req["parameters_gb"] <= available_memory * 2:
            # Model fits with memory optimization
            if self.model_characteristics.num_parameters <= self.LARGE_MODEL_THRESHOLD:
                strategy, config = self._select_data_parallelism_strategy()
            else:
                strategy, config = self._select_model_parallelism_strategy()
        else:
            # Need model parallelism
            strategy, config = self._select_model_parallelism_strategy()
        
        # Optimize communication patterns
        config = self._optimize_communication_patterns(strategy, config)
        
        # Estimate efficiency
        efficiency = self._estimate_parallel_efficiency(strategy, config)
        
        # Add gradient checkpointing and activation recomputation
        if self.enable_gradient_checkpointing:
            config["gradient_checkpointing"] = True
        
        if self.enable_activation_recomputation:
            config["activation_recomputation"] = True
        
        # Add precision settings
        config["precision"] = self.precision
        
        # Final configuration
        final_config = {
            "strategy": strategy.value,
            "model_parallelism": config,
            "estimated_efficiency": efficiency,
            "memory_requirements_gb": memory_req,
            "cluster_topology": {
                "num_nodes": self.cluster.num_nodes,
                "gpus_per_node": self.cluster.gpus_per_node,
                "total_gpus": self.cluster.total_gpus,
                "gpu_memory_gb": self.cluster.gpu_memory_gb,
            },
            "model_characteristics": {
                "num_parameters": self.model_characteristics.num_parameters,
                "num_layers": self.model_characteristics.num_layers,
                "hidden_size": self.model_characteristics.hidden_size,
                "num_attention_heads": self.model_characteristics.num_attention_heads,
            },
            "recommendations": self._generate_recommendations(strategy, efficiency, memory_req),
        }
        
        logger.info(f"Selected strategy: {strategy.value} with estimated efficiency: {efficiency:.2%}")
        return final_config
    
    def _generate_recommendations(
        self, 
        strategy: ParallelStrategy, 
        efficiency: float,
        memory_req: Dict[str, float]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if efficiency < self.MIN_EFFICIENCY_THRESHOLD:
            recommendations.append(
                f"Low parallel efficiency ({efficiency:.1%}). Consider adjusting batch size or sequence length."
            )
        
        if memory_req["activations_gb"] > memory_req["total_gb"] * 0.5:
            recommendations.append(
                "High activation memory usage. Enable gradient checkpointing or reduce sequence length."
            )
        
        if strategy == ParallelStrategy.DDP and self.cluster.total_gpus > 8:
            recommendations.append(
                "Large DDP job may benefit from gradient compression or bucket size tuning."
            )
        
        if strategy == ParallelStrategy.HYBRID:
            recommendations.append(
                "Hybrid parallelism detected. Ensure pipeline stages are balanced for optimal performance."
            )
        
        if not self.cluster.nvlink_available and strategy in [ParallelStrategy.MEGATRON, ParallelStrategy.HYBRID]:
            recommendations.append(
                "NVLink not available. Tensor parallelism may have high communication overhead."
            )
        
        return recommendations


def auto_configure_distributed_training(
    model: nn.Module,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    gpu_memory_gb: float = 80.0,
    gpu_flops: float = 312.0,  # A100 TFLOPS
    inter_node_bandwidth_gbps: float = 200.0,
    intra_node_bandwidth_gbps: float = 600.0,
    nvlink_available: bool = True,
    infiniband_available: bool = True,
    target_batch_size: int = 32,
    target_sequence_length: int = 2048,
    precision: str = "bf16",
    enable_gradient_checkpointing: bool = True,
    enable_activation_recomputation: bool = False,
) -> Dict[str, Any]:
    """
    Automatically configure distributed training for a model.
    
    Args:
        model: PyTorch model
        num_nodes: Number of nodes in cluster
        gpus_per_node: Number of GPUs per node
        gpu_memory_gb: GPU memory in GB
        gpu_flops: GPU performance in TFLOPS
        inter_node_bandwidth_gbps: Inter-node bandwidth in Gbps
        intra_node_bandwidth_gbps: Intra-node bandwidth in Gbps
        nvlink_available: Whether NVLink is available
        infiniband_available: Whether InfiniBand is available
        target_batch_size: Target global batch size
        target_sequence_length: Target sequence length
        precision: Training precision (fp16, bf16, fp32)
        enable_gradient_checkpointing: Enable gradient checkpointing
        enable_activation_recomputation: Enable activation recomputation
        
    Returns:
        Configuration dictionary for distributed training
    """
    cluster = ClusterTopology(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        gpu_memory_gb=gpu_memory_gb,
        gpu_flops=gpu_flops,
        inter_node_bandwidth_gbps=inter_node_bandwidth_gbps,
        intra_node_bandwidth_gbps=intra_node_bandwidth_gbps,
        nvlink_available=nvlink_available,
        infiniband_available=infiniband_available,
    )
    
    selector = DistributedStrategySelector(
        model=model,
        cluster=cluster,
        target_batch_size=target_batch_size,
        target_sequence_length=target_sequence_length,
        precision=precision,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        enable_activation_recomputation=enable_activation_recomputation,
    )
    
    return selector.select_optimal_strategy()


def apply_distributed_strategy(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Apply distributed strategy to model based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration from auto_configure_distributed_training
        
    Returns:
        Distributed model
    """
    strategy = config["strategy"]
    model_config = config["model_parallelism"]
    
    if strategy == "ddp":
        # Wrap with DistributedDataParallel
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
                find_unused_parameters=model_config.get("find_unused_parameters", False),
                gradient_as_bucket_view=model_config.get("gradient_as_bucket_view", True),
            )
    
    elif strategy == "fsdp":
        # Wrap with FullyShardedDataParallel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            CPUOffload,
            MixedPrecision,
        )
        
        # Configure mixed precision
        if model_config.get("mixed_precision", True):
            if config.get("precision") == "bf16":
                dtype = torch.bfloat16
            elif config.get("precision") == "fp16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )
        else:
            mixed_precision_policy = None
        
        # Configure CPU offload
        cpu_offload = CPUOffload(offload_params=True) if model_config.get("cpu_offload", False) else None
        
        model = FSDP(
            model,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload,
            sharding_strategy=model_config.get("sharding_strategy", "FULL_SHARD"),
            sync_module_states=model_config.get("sync_module_states", True),
        )
    
    elif strategy == "deepspeed":
        # DeepSpeed integration would require DeepSpeed library
        logger.warning("DeepSpeed strategy selected but not implemented in this example")
    
    elif strategy in ["megatron", "hybrid"]:
        # Megatron-LM integration would require Megatron-LM library
        logger.warning("Megatron-LM strategy selected but not implemented in this example")
    
    # Apply gradient checkpointing if enabled
    if model_config.get("gradient_checkpointing", False):
        model = _apply_gradient_checkpointing(model)
    
    return model


def _apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Apply gradient checkpointing to model."""
    from torch.utils.checkpoint import checkpoint
    
    # Find transformer layers and wrap with checkpoint
    for name, module in model.named_modules():
        if any(layer_type in name.lower() for layer_type in ["layer", "block", "transformer"]):
            if hasattr(module, "forward"):
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                
                module.forward = checkpointed_forward
    
    return model


# Integration with existing forge code
def integrate_with_forge():
    """
    Integration point for forge training pipeline.
    This function should be called during training setup.
    """
    # This would be integrated into the training script
    pass


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Strategy Selector")
    parser.add_argument("--model-size", type=float, default=7.0, help="Model size in billions")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")
    parser.add_argument("--batch-size", type=int, default=32, help="Target batch size")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    
    args = parser.parse_args()
    
    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self, num_params):
            super().__init__()
            self.num_params = num_params
            # Create a simple model with approximately num_params parameters
            hidden_size = int(math.sqrt(num_params / 6))
            self.linear = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel(int(args.model_size * 1e9))
    
    config = auto_configure_distributed_training(
        model=model,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        target_batch_size=args.batch_size,
        target_sequence_length=args.seq_length,
    )
    
    print("Distributed Training Configuration:")
    print(f"Strategy: {config['strategy']}")
    print(f"Estimated Efficiency: {config['estimated_efficiency']:.2%}")
    print(f"Memory Requirements: {config['memory_requirements_gb']['total_gb']:.2f} GB")
    print("\nRecommendations:")
    for rec in config['recommendations']:
        print(f"  - {rec}")