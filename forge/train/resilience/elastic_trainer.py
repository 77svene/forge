"""
Distributed Training Resilience Layer for forge
Provides automatic checkpointing, fault tolerance, and elastic scaling
for distributed training across heterogeneous hardware.
"""

import os
import time
import json
import logging
import signal
import threading
import queue
import psutil
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Supported distributed training backends."""
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    MEGATRON = "megatron"
    PYTORCH = "pytorch"
    HOROVOD = "horovod"


class AcceleratorType(Enum):
    """Supported accelerator types."""
    GPU = "gpu"
    TPU = "tpu"
    CUSTOM = "custom"
    CPU = "cpu"


class NodeState(Enum):
    """Node health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class CheckpointConfig:
    """Configuration for automatic checkpointing."""
    save_dir: str = "checkpoints"
    save_interval: int = 1000  # Steps
    save_on_interrupt: bool = True
    save_on_failure: bool = True
    keep_last_n: int = 5
    async_save: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    compress: bool = False


@dataclass
class ElasticConfig:
    """Configuration for elastic scaling."""
    enabled: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 1024
    adjustment_factor: float = 0.8
    memory_threshold: float = 0.9  # 90% memory usage
    check_interval: int = 100  # Steps
    warmup_steps: int = 100
    cooldown_steps: int = 50
    enable_auto_batch: bool = True
    enable_auto_workers: bool = True


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance."""
    enabled: bool = True
    heartbeat_interval: int = 30  # Seconds
    failure_timeout: int = 300  # Seconds
    max_restarts: int = 3
    restart_delay: int = 60  # Seconds
    enable_preemption: bool = True
    preemption_signal: int = signal.SIGTERM
    checkpoint_on_failure: bool = True
    resume_from_checkpoint: bool = True


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    rank: int
    local_rank: int
    world_size: int
    accelerator_type: AcceleratorType
    accelerator_count: int
    memory_total: float  # GB
    memory_available: float  # GB
    state: NodeState = NodeState.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages automatic checkpointing with fault tolerance."""
    
    def __init__(self, config: CheckpointConfig, model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 backend: DistributedBackend = DistributedBackend.PYTORCH):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.backend = backend
        self.save_queue = queue.Queue() if config.async_save else None
        self.save_thread = None
        self.current_step = 0
        self.checkpoint_history = []
        
        # Create checkpoint directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        if config.async_save:
            self._start_async_saver()
    
    def _start_async_saver(self):
        """Start background thread for asynchronous checkpoint saving."""
        self.save_thread = threading.Thread(target=self._async_save_worker, daemon=True)
        self.save_thread.start()
    
    def _async_save_worker(self):
        """Worker thread for asynchronous checkpoint saving."""
        while True:
            try:
                checkpoint_data = self.save_queue.get(timeout=1)
                if checkpoint_data is None:  # Poison pill
                    break
                self._save_checkpoint_sync(**checkpoint_data)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Async checkpoint save failed: {e}")
    
    def save_checkpoint(self, step: int, epoch: int = 0, 
                       metrics: Optional[Dict[str, float]] = None,
                       force: bool = False):
        """Save checkpoint with automatic frequency control."""
        self.current_step = step
        
        # Check if we should save
        should_save = force or (step % self.config.save_interval == 0)
        
        if should_save:
            checkpoint_data = {
                'step': step,
                'epoch': epoch,
                'model_state_dict': self._get_model_state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer and self.config.save_optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler and self.config.save_scheduler else None,
                'metrics': metrics or {},
                'rng_state': self._get_rng_state() if self.config.save_rng_state else None,
                'timestamp': time.time(),
                'backend': self.backend.value
            }
            
            if self.config.async_save and self.save_queue:
                self.save_queue.put(checkpoint_data)
            else:
                self._save_checkpoint_sync(**checkpoint_data)
    
    def _save_checkpoint_sync(self, step: int, epoch: int, **kwargs):
        """Synchronously save checkpoint to disk."""
        try:
            checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_step_{step}.pt")
            
            # Save with compression if enabled
            if self.config.compress:
                checkpoint_path += ".gz"
                torch.save(kwargs, checkpoint_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(kwargs, checkpoint_path)
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'step': step,
                'epoch': epoch,
                'path': checkpoint_path,
                'timestamp': time.time()
            })
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
    
    def _get_model_state_dict(self):
        """Get model state dict handling different backends."""
        if self.backend == DistributedBackend.DEEPSPEED:
            # DeepSpeed model saving
            return self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        elif self.backend == DistributedBackend.FSDP:
            # FSDP model saving
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(self.model, FSDP):
                return self.model.state_dict()
            return self.model.state_dict()
        else:
            # Standard PyTorch DDP or single GPU
            return self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
    
    def _get_rng_state(self):
        """Get RNG state for reproducibility."""
        rng_state = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': None  # Python's random state is not easily serializable
        }
        if torch.cuda.is_available():
            rng_state['cuda'] = torch.cuda.get_rng_state_all()
        return rng_state
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N."""
        if len(self.checkpoint_history) > self.config.keep_last_n:
            # Sort by step (descending)
            self.checkpoint_history.sort(key=lambda x: x['step'], reverse=True)
            
            # Remove old checkpoints
            for checkpoint in self.checkpoint_history[self.config.keep_last_n:]:
                try:
                    if os.path.exists(checkpoint['path']):
                        os.remove(checkpoint['path'])
                        logger.debug(f"Removed old checkpoint: {checkpoint['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
            
            # Update history
            self.checkpoint_history = self.checkpoint_history[:self.config.keep_last_n]
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, 
                       load_optimizer: bool = True, load_scheduler: bool = True):
        """Load checkpoint from disk."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            model_state = checkpoint.get('model_state_dict')
            if model_state:
                if hasattr(self.model, 'module'):
                    self.model.module.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
            
            # Load optimizer state
            if load_optimizer and self.optimizer and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if load_scheduler and self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore RNG state
            if checkpoint.get('rng_state'):
                self._restore_rng_state(checkpoint['rng_state'])
            
            logger.info(f"Checkpoint loaded successfully from step {checkpoint.get('step', 0)}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the save directory."""
        try:
            checkpoints = []
            for f in os.listdir(self.config.save_dir):
                if f.startswith("checkpoint_step_") and (f.endswith(".pt") or f.endswith(".pt.gz")):
                    step = int(f.split("_")[2].split(".")[0])
                    checkpoints.append((step, os.path.join(self.config.save_dir, f)))
            
            if checkpoints:
                checkpoints.sort(reverse=True)
                return checkpoints[0][1]
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
        
        return None
    
    def _restore_rng_state(self, rng_state: Dict):
        """Restore RNG state from checkpoint."""
        if 'torch' in rng_state:
            torch.set_rng_state(rng_state['torch'])
        if 'numpy' in rng_state:
            np.random.set_state(rng_state['numpy'])
        if 'cuda' in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])
    
    def shutdown(self):
        """Shutdown checkpoint manager."""
        if self.config.async_save and self.save_queue:
            self.save_queue.put(None)  # Poison pill
            if self.save_thread:
                self.save_thread.join(timeout=30)


class NodeMonitor:
    """Monitors node health and detects failures."""
    
    def __init__(self, config: FaultToleranceConfig, node_info: NodeInfo):
        self.config = config
        self.node_info = node_info
        self.nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_thread = None
        self.monitoring = False
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Register signal handlers for graceful shutdown
        if config.enable_preemption:
            signal.signal(config.preemption_signal, self._handle_preemption)
    
    def start_monitoring(self):
        """Start node monitoring."""
        if not self.config.enabled:
            return
        
        self.monitoring = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Node monitoring started")
    
    def _heartbeat_worker(self):
        """Worker thread for sending heartbeats and checking node health."""
        while self.monitoring:
            try:
                # Send heartbeat
                self._send_heartbeat()
                
                # Check other nodes
                self._check_node_health()
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat worker error: {e}")
                time.sleep(5)
    
    def _send_heartbeat(self):
        """Send heartbeat to other nodes."""
        if dist.is_initialized():
            # Update last heartbeat time
            self.node_info.last_heartbeat = time.time()
            
            # Broadcast heartbeat to all nodes
            heartbeat_data = {
                'node_id': self.node_info.node_id,
                'timestamp': self.node_info.last_heartbeat,
                'state': self.node_info.state.value,
                'memory_available': self.node_info.memory_available
            }
            
            # In a real implementation, this would use distributed communication
            # For now, we'll simulate it
            pass
    
    def _check_node_health(self):
        """Check health of all nodes."""
        current_time = time.time()
        
        for node_id, node_info in self.nodes.items():
            if node_id == self.node_info.node_id:
                continue
            
            time_since_heartbeat = current_time - node_info.last_heartbeat
            
            if time_since_heartbeat > self.config.failure_timeout:
                if node_info.state != NodeState.FAILED:
                    logger.warning(f"Node {node_id} failed (no heartbeat for {time_since_heartbeat:.1f}s)")
                    node_info.state = NodeState.FAILED
                    node_info.failure_count += 1
                    self._trigger_failure_callbacks(node_id)
            
            elif time_since_heartbeat > self.config.failure_timeout * 0.7:
                if node_info.state == NodeState.HEALTHY:
                    logger.warning(f"Node {node_id} degraded (heartbeat delay: {time_since_heartbeat:.1f}s)")
                    node_info.state = NodeState.DEGRADED
    
    def _trigger_failure_callbacks(self, failed_node_id: str):
        """Trigger callbacks when a node fails."""
        for callback in self.failure_callbacks:
            try:
                callback(failed_node_id)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def register_failure_callback(self, callback: Callable):
        """Register callback for node failures."""
        self.failure_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """Register callback for node recovery."""
        self.recovery_callbacks.append(callback)
    
    def _handle_preemption(self, signum, frame):
        """Handle preemption signal (e.g., from cloud scheduler)."""
        logger.warning(f"Received preemption signal {signum}")
        self.node_info.state = NodeState.RECOVERING
        
        # Trigger checkpoint before shutdown
        for callback in self.failure_callbacks:
            try:
                callback(self.node_info.node_id, preemption=True)
            except Exception as e:
                logger.error(f"Preemption callback error: {e}")
    
    def get_healthy_nodes(self) -> List[NodeInfo]:
        """Get list of healthy nodes."""
        return [node for node in self.nodes.values() if node.state == NodeState.HEALTHY]
    
    def shutdown(self):
        """Shutdown node monitoring."""
        self.monitoring = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=10)


class ElasticScaler:
    """Manages elastic scaling of batch size and workers."""
    
    def __init__(self, config: ElasticConfig, model: torch.nn.Module,
                 train_dataloader: Any, device: torch.device):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = device
        self.current_batch_size = getattr(train_dataloader, 'batch_size', 1)
        self.current_workers = getattr(train_dataloader, 'num_workers', 0)
        self.memory_history = []
        self.adjustment_history = []
        self.last_adjustment_step = 0
        
        # Monitor memory
        self._update_memory_stats()
    
    def _update_memory_stats(self):
        """Update memory statistics."""
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            memory_available = memory_total - memory_allocated
            memory_usage = memory_allocated / memory_total if memory_total > 0 else 0
        else:
            # For CPU or other devices
            memory = psutil.virtual_memory()
            memory_available = memory.available / 1024**3  # GB
            memory_total = memory.total / 1024**3  # GB
            memory_usage = memory.percent / 100
        
        return {
            'available': memory_available,
            'total': memory_total,
            'usage': memory_usage,
            'timestamp': time.time()
        }
    
    def should_adjust(self, step: int) -> bool:
        """Check if we should adjust batch size/workers."""
        if not self.config.enabled:
            return False
        
        # Check cooldown
        if step - self.last_adjustment_step < self.config.cooldown_steps:
            return False
        
        # Check interval
        if step % self.config.check_interval != 0:
            return False
        
        # Check warmup
        if step < self.config.warmup_steps:
            return False
        
        return True
    
    def adjust_batch_size(self, step: int, current_loss: Optional[float] = None) -> bool:
        """Adjust batch size based on memory usage and training dynamics."""
        if not self.should_adjust(step):
            return False
        
        memory_stats = self._update_memory_stats()
        self.memory_history.append(memory_stats)
        
        # Keep only recent history
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        should_adjust = False
        new_batch_size = self.current_batch_size
        
        # Check memory threshold
        if memory_stats['usage'] > self.config.memory_threshold:
            # Reduce batch size
            reduction_factor = self.config.adjustment_factor
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * reduction_factor)
            )
            should_adjust = new_batch_size != self.current_batch_size
            reason = f"Memory usage {memory_stats['usage']:.1%} > threshold {self.config.memory_threshold:.1%}"
        
        elif memory_stats['usage'] < self.config.memory_threshold * 0.7:
            # Increase batch size if we have headroom
            increase_factor = 1.0 / self.config.adjustment_factor
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * increase_factor)
            )
            should_adjust = new_batch_size != self.current_batch_size
            reason = f"Memory usage {memory_stats['usage']:.1%} has headroom"
        
        if should_adjust:
            logger.info(f"Adjusting batch size from {self.current_batch_size} to {new_batch_size} ({reason})")
            
            # Update dataloader
            if hasattr(self.train_dataloader, 'batch_sampler'):
                # For PyTorch DataLoader
                self.train_dataloader.batch_sampler.batch_size = new_batch_size
            elif hasattr(self.train_dataloader, 'batch_size'):
                self.train_dataloader.batch_size = new_batch_size
            
            self.current_batch_size = new_batch_size
            self.last_adjustment_step = step
            
            self.adjustment_history.append({
                'step': step,
                'old_batch_size': self.current_batch_size,
                'new_batch_size': new_batch_size,
                'reason': reason,
                'memory_usage': memory_stats['usage']
            })
        
        return should_adjust
    
    def adjust_workers(self, step: int) -> bool:
        """Adjust number of data loading workers."""
        if not self.config.enabled or not self.config.enable_auto_workers:
            return False
        
        # This is a simplified implementation
        # In practice, you'd monitor data loading bottlenecks
        cpu_percent = psutil.cpu_percent(interval=1)
        
        should_adjust = False
        new_workers = self.current_workers
        
        if cpu_percent > 80 and self.current_workers > 0:
            # Reduce workers if CPU is overloaded
            new_workers = max(0, self.current_workers - 1)
            should_adjust = True
            reason = f"CPU usage {cpu_percent}% > 80%"
        
        elif cpu_percent < 50 and self.current_workers < os.cpu_count():
            # Increase workers if CPU has capacity
            new_workers = min(os.cpu_count(), self.current_workers + 1)
            should_adjust = True
            reason = f"CPU usage {cpu_percent}% < 50%"
        
        if should_adjust:
            logger.info(f"Adjusting workers from {self.current_workers} to {new_workers} ({reason})")
            
            # Note: Changing num_workers requires recreating the DataLoader
            # This would typically be handled by the training loop
            self.current_workers = new_workers
        
        return should_adjust
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on historical data."""
        if not self.memory_history:
            return self.current_batch_size
        
        # Find batch size that keeps memory usage below threshold
        recent_memory = self.memory_history[-10:] if len(self.memory_history) >= 10 else self.memory_history
        avg_usage = np.mean([m['usage'] for m in recent_memory])
        
        if avg_usage > self.config.memory_threshold:
            # Reduce batch size
            optimal = int(self.current_batch_size * self.config.adjustment_factor)
        elif avg_usage < self.config.memory_threshold * 0.7:
            # Increase batch size
            optimal = int(self.current_batch_size / self.config.adjustment_factor)
        else:
            optimal = self.current_batch_size
        
        return max(self.config.min_batch_size, min(self.config.max_batch_size, optimal))


class ElasticTrainer:
    """
    Distributed Training Resilience Layer with automatic checkpointing,
    fault tolerance, and elastic scaling.
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: Any,
                 val_dataloader: Optional[Any] = None,
                 scheduler: Optional[Any] = None,
                 device: Optional[torch.device] = None,
                 backend: DistributedBackend = DistributedBackend.PYTORCH,
                 accelerator_type: AcceleratorType = AcceleratorType.GPU,
                 checkpoint_config: Optional[CheckpointConfig] = None,
                 elastic_config: Optional[ElasticConfig] = None,
                 fault_tolerance_config: Optional[FaultToleranceConfig] = None,
                 distributed_config: Optional[Dict[str, Any]] = None):
        
        # Initialize configurations
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.elastic_config = elastic_config or ElasticConfig()
        self.fault_tolerance_config = fault_tolerance_config or FaultToleranceConfig()
        self.distributed_config = distributed_config or {}
        
        # Store components
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.backend = backend
        self.accelerator_type = accelerator_type
        
        # Setup device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize distributed if needed
        self._init_distributed()
        
        # Get node information
        self.node_info = self._get_node_info()
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            config=self.checkpoint_config,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            backend=self.backend
        )
        
        self.node_monitor = NodeMonitor(
            config=self.fault_tolerance_config,
            node_info=self.node_info
        )
        
        self.elastic_scaler = ElasticScaler(
            config=self.elastic_config,
            model=self.model,
            train_dataloader=self.train_dataloader,
            device=self.device
        )
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.training_active = False
        self.interrupted = False
        
        # Register callbacks
        self.node_monitor.register_failure_callback(self._on_node_failure)
        self.node_monitor.register_recovery_callback(self._on_node_recovery)
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"ElasticTrainer initialized on node {self.node_info.node_id}")
        logger.info(f"Backend: {self.backend.value}, Accelerator: {self.accelerator_type.value}")
        logger.info(f"Checkpoint dir: {self.checkpoint_config.save_dir}")
    
    def _init_distributed(self):
        """Initialize distributed training backend."""
        if not dist.is_initialized():
            if self.backend == DistributedBackend.DEEPSPEED:
                self._init_deepspeed()
            elif self.backend == DistributedBackend.FSDP:
                self._init_fsdp()
            elif self.backend == DistributedBackend.MEGATRON:
                self._init_megatron()
            elif self.backend == DistributedBackend.HOROVOD:
                self._init_horovod()
            else:
                # Default PyTorch distributed
                if 'RANK' in os.environ:
                    dist.init_process_group(
                        backend=self.distributed_config.get('backend', 'nccl'),
                        init_method=self.distributed_config.get('init_method', 'env://')
                    )
        
        # Wrap model with DDP if needed
        if dist.is_initialized() and self.backend == DistributedBackend.PYTORCH:
            self.model = DDP(
                self.model,
                device_ids=[self.node_info.local_rank] if self.accelerator_type == AcceleratorType.GPU else None,
                output_device=self.node_info.local_rank if self.accelerator_type == AcceleratorType.GPU else None
            )
    
    def _init_deepspeed(self):
        """Initialize DeepSpeed backend."""
        try:
            import deepspeed
            # DeepSpeed initialization would be handled by the training script
            # This is a placeholder for integration
            logger.info("DeepSpeed backend initialized")
        except ImportError:
            logger.warning("DeepSpeed not installed, falling back to PyTorch distributed")
            self.backend = DistributedBackend.PYTORCH
    
    def _init_fsdp(self):
        """Initialize FSDP backend."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                CPUOffload,
                MixedPrecision,
            )
            # FSDP initialization would be handled by the training script
            logger.info("FSDP backend initialized")
        except ImportError:
            logger.warning("FSDP not available, falling back to PyTorch distributed")
            self.backend = DistributedBackend.PYTORCH
    
    def _init_megatron(self):
        """Initialize Megatron-LM backend."""
        try:
            # Megatron initialization would be handled by the training script
            logger.info("Megatron backend initialized")
        except ImportError:
            logger.warning("Megatron not available, falling back to PyTorch distributed")
            self.backend = DistributedBackend.PYTORCH
    
    def _init_horovod(self):
        """Initialize Horovod backend."""
        try:
            import horovod.torch as hvd
            hvd.init()
            logger.info("Horovod backend initialized")
        except ImportError:
            logger.warning("Horovod not installed, falling back to PyTorch distributed")
            self.backend = DistributedBackend.PYTORCH
    
    def _get_node_info(self) -> NodeInfo:
        """Get information about the current node."""
        hostname = os.uname().nodename
        node_id = f"{hostname}_{os.getpid()}"
        
        # Get distributed info
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Get accelerator info
        if self.accelerator_type == AcceleratorType.GPU:
            accelerator_count = torch.cuda.device_count()
            if accelerator_count > 0:
                memory_total = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
                memory_available = memory_total - torch.cuda.memory_allocated(local_rank) / 1024**3
            else:
                memory_total = 0
                memory_available = 0
        else:
            accelerator_count = 1
            memory = psutil.virtual_memory()
            memory_total = memory.total / 1024**3
            memory_available = memory.available / 1024**3
        
        return NodeInfo(
            node_id=node_id,
            hostname=hostname,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            accelerator_type=self.accelerator_type,
            accelerator_count=accelerator_count,
            memory_total=memory_total,
            memory_available=memory_available
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        self.interrupted = True
        
        if self.checkpoint_config.save_on_interrupt:
            logger.info("Saving checkpoint due to interrupt")
            self.checkpoint_manager.save_checkpoint(
                step=self.current_step,
                epoch=self.current_epoch,
                force=True
            )
        
        self.shutdown()
    
    def _on_node_failure(self, failed_node_id: str, preemption: bool = False):
        """Handle node failure."""
        logger.warning(f"Node failure detected: {failed_node_id}")
        
        if self.checkpoint_config.save_on_failure:
            logger.info("Saving checkpoint due to node failure")
            self.checkpoint_manager.save_checkpoint(
                step=self.current_step,
                epoch=self.current_epoch,
                force=True
            )
        
        # In a real implementation, this would trigger recovery procedures
        # For now, we just log the event
    
    def _on_node_recovery(self, recovered_node_id: str):
        """Handle node recovery."""
        logger.info(f"Node recovered: {recovered_node_id}")
    
    def train_step(self, batch: Any, step: int) -> Dict[str, float]:
        """
        Execute a single training step with resilience features.
        
        Args:
            batch: Training batch
            step: Current step number
            
        Returns:
            Dictionary of metrics
        """
        self.current_step = step
        
        try:
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            self.model.train()
            self.optimizer.zero_grad()
            
            # Handle different model interfaces
            if hasattr(self.model, 'forward'):
                outputs = self.model(batch)
            else:
                outputs = self.model(*batch) if isinstance(batch, (list, tuple)) else self.model(batch)
            
            # Calculate loss
            if isinstance(outputs, torch.Tensor):
                loss = outputs
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                loss = outputs[0]
            else:
                raise ValueError("Model output must contain loss")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (if configured)
            if self.distributed_config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.distributed_config['grad_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate metrics
            metrics = {
                'loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.elastic_scaler.current_batch_size,
                'memory_usage': self.elastic_scaler._update_memory_stats()['usage']
            }
            
            # Add any additional metrics from model output
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if k != 'loss' and isinstance(v, torch.Tensor):
                        metrics[k] = v.item()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training step {step} failed: {e}")
            
            # Save checkpoint on failure
            if self.checkpoint_config.save_on_failure:
                logger.info("Saving checkpoint due to training step failure")
                self.checkpoint_manager.save_checkpoint(
                    step=self.current_step,
                    epoch=self.current_epoch,
                    force=True
                )
            
            raise
    
    def train(self,
              train_step_fn: Optional[Callable] = None,
              num_epochs: int = 1,
              num_steps: Optional[int] = None,
              resume_from_checkpoint: Optional[str] = None,
              callback: Optional[Callable] = None) -> Dict[str, List[float]]:
        """
        Main training loop with resilience features.
        
        Args:
            train_step_fn: Custom training step function (optional)
            num_epochs: Number of epochs to train
            num_steps: Total number of steps (overrides num_epochs)
            resume_from_checkpoint: Path to checkpoint to resume from
            callback: Callback function called after each step
            
        Returns:
            Dictionary of training history
        """
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.resume_training(resume_from_checkpoint)
        elif self.fault_tolerance_config.resume_from_checkpoint:
            # Try to resume from latest checkpoint
            latest_checkpoint = self.checkpoint_manager._find_latest_checkpoint()
            if latest_checkpoint:
                self.resume_training(latest_checkpoint)
        
        # Start node monitoring
        self.node_monitor.start_monitoring()
        
        # Training history
        history = {
            'loss': [],
            'learning_rate': [],
            'batch_size': [],
            'memory_usage': [],
            'step': [],
            'epoch': []
        }
        
        self.training_active = True
        step = self.current_step
        
        try:
            logger.info(f"Starting training from step {step}")
            
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_step = step
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    if not self.training_active or self.interrupted:
                        break
                    
                    # Check if we've reached total steps
                    if num_steps and step >= num_steps:
                        break
                    
                    # Custom training step function
                    if train_step_fn:
                        metrics = train_step_fn(self.model, batch, step, self.device)
                    else:
                        metrics = self.train_step(batch, step)
                    
                    # Update history
                    for k, v in metrics.items():
                        if k in history:
                            history[k].append(v)
                    history['step'].append(step)
                    history['epoch'].append(epoch)
                    
                    # Elastic scaling adjustments
                    if self.elastic_config.enabled:
                        self.elastic_scaler.adjust_batch_size(step, metrics.get('loss'))
                        self.elastic_scaler.adjust_workers(step)
                    
                    # Automatic checkpointing
                    self.checkpoint_manager.save_checkpoint(
                        step=step,
                        epoch=epoch,
                        metrics=metrics
                    )
                    
                    # Callback
                    if callback:
                        callback(step, epoch, metrics)
                    
                    # Log progress
                    if step % 100 == 0:
                        logger.info(
                            f"Step {step}, Epoch {epoch}, "
                            f"Loss: {metrics['loss']:.4f}, "
                            f"LR: {metrics['learning_rate']:.2e}, "
                            f"Batch: {metrics['batch_size']}, "
                            f"Memory: {metrics['memory_usage']:.1%}"
                        )
                    
                    step += 1
                
                # Epoch completed
                logger.info(f"Epoch {epoch} completed, steps: {step - epoch_start_step}")
                
                # Validation
                if self.val_dataloader:
                    val_metrics = self.validate()
                    logger.info(f"Validation metrics: {val_metrics}")
                
                # Check if we should stop
                if not self.training_active or self.interrupted:
                    break
            
            logger.info(f"Training completed. Total steps: {step}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Save checkpoint on failure
            if self.checkpoint_config.save_on_failure:
                logger.info("Saving checkpoint due to training failure")
                self.checkpoint_manager.save_checkpoint(
                    step=self.current_step,
                    epoch=self.current_epoch,
                    force=True
                )
            
            raise
        
        finally:
            # Shutdown components
            self.shutdown()
        
        return history
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        val_metrics = {}
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(batch)
                else:
                    outputs = self.model(*batch) if isinstance(batch, (list, tuple)) else self.model(batch)
                
                # Calculate loss
                if isinstance(outputs, torch.Tensor):
                    loss = outputs
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    loss = outputs[0]
                else:
                    continue
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            val_metrics['val_loss'] = total_loss / num_batches
        
        self.model.train()
        return val_metrics
    
    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if checkpoint:
            self.current_step = checkpoint.get('step', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            
            logger.info(f"Resumed from step {self.current_step}, epoch {self.current_epoch}")
            
            return checkpoint
        
        return {}
    
    def shutdown(self):
        """Shutdown all components."""
        logger.info("Shutting down ElasticTrainer")
        
        self.training_active = False
        
        # Shutdown components
        self.checkpoint_manager.shutdown()
        self.node_monitor.shutdown()
        
        # Final checkpoint
        if self.current_step > 0:
            self.checkpoint_manager.save_checkpoint(
                step=self.current_step,
                epoch=self.current_epoch,
                force=True
            )
        
        logger.info("ElasticTrainer shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


# Factory function for easy instantiation
def create_elastic_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: Any,
    **kwargs
) -> ElasticTrainer:
    """
    Factory function to create an ElasticTrainer with sensible defaults.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        train_dataloader: Training data loader
        **kwargs: Additional arguments for ElasticTrainer
        
    Returns:
        Configured ElasticTrainer instance
    """
    # Set default configurations if not provided
    checkpoint_config = kwargs.pop('checkpoint_config', None) or CheckpointConfig()
    elastic_config = kwargs.pop('elastic_config', None) or ElasticConfig()
    fault_tolerance_config = kwargs.pop('fault_tolerance_config', None) or FaultToleranceConfig()
    
    return ElasticTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        checkpoint_config=checkpoint_config,
        elastic_config=elastic_config,
        fault_tolerance_config=fault_tolerance_config,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # This is example code showing how to use the ElasticTrainer
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create trainer
    trainer = create_elastic_trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        checkpoint_config=CheckpointConfig(save_interval=100),
        elastic_config=ElasticConfig(enabled=True),
        fault_tolerance_config=FaultToleranceConfig(enabled=True)
    )
    
    # Training loop
    def custom_train_step(model, batch, step, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        return {'loss': loss.item()}
    
    # Train
    history = trainer.train(
        train_step_fn=custom_train_step,
        num_epochs=5,
        callback=lambda step, epoch, metrics: print(f"Step {step}: {metrics['loss']:.4f}")
    )
    
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")