#!/usr/bin/env python3
"""
Distributed Training Resilience Layer for forge

Provides automatic checkpointing, fault tolerance, and elastic scaling
for distributed training across heterogeneous hardware.
"""

import os
import sys
import time
import json
import logging
import signal
import threading
import socket
import pickle
import hashlib
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np

# Import torch and distributed modules
try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not found, some features will be disabled")

# Import optional distributed backends
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False

try:
    from megatron.core import parallel_state as mpu
    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Supported distributed training backends."""
    DDP = "ddp"
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    MEGATRON = "megatron"
    UNKNOWN = "unknown"


class CheckpointFormat(Enum):
    """Supported checkpoint formats."""
    PYTORCH = "pytorch"
    DEEPSPEED = "deepspeed"
    SAFETENSORS = "safetensors"
    HUGGINGFACE = "huggingface"


class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 1000  # Save every N steps
    save_on_epoch_end: bool = True
    keep_last_n: int = 5  # Keep last N checkpoints
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    async_save: bool = True  # Use background thread for saving
    compression: Optional[str] = None  # "gzip", "zstd", etc.
    format: CheckpointFormat = CheckpointFormat.PYTORCH
    remote_storage: Optional[str] = None  # S3/GCS path for backup
    upload_async: bool = True


@dataclass
class ResilienceConfig:
    """Configuration for resilience features."""
    enable_fault_tolerance: bool = True
    heartbeat_interval: int = 30  # seconds
    failure_timeout: int = 120  # seconds before declaring node failed
    enable_elastic_scaling: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 128
    batch_size_step: int = 2  # Multiply/divide batch size by this factor
    memory_threshold: float = 0.85  # Memory usage threshold for scaling
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_delay: int = 10  # seconds between recovery attempts


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    rank: int
    local_rank: int
    world_size: int
    device_type: str  # "cuda", "tpu", "cpu", etc.
    device_count: int
    memory_total: float  # GB
    memory_available: float  # GB
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0


@dataclass
class TrainingState:
    """Current training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    batch_size: int = 1
    learning_rate: float = 0.0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    rng_states: Dict[str, Any] = field(default_factory=dict)
    custom_state: Dict[str, Any] = field(default_factory=dict)


class MemoryMonitor:
    """Monitor system and device memory usage."""
    
    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        self.history = deque(maxlen=100)
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            "system_memory_percent": psutil.virtual_memory().percent / 100.0,
            "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        if self.device_type == "cuda" and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"gpu_{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                stats[f"gpu_{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
                stats[f"gpu_{i}_memory_total_gb"] = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        self.history.append(stats)
        return stats
    
    def should_adjust_batch_size(self, current_batch_size: int, config: ResilienceConfig) -> Tuple[bool, int]:
        """Determine if batch size should be adjusted based on memory usage."""
        stats = self.get_memory_stats()
        
        # Check system memory
        if stats["system_memory_percent"] > config.memory_threshold:
            logger.warning(f"System memory usage high: {stats['system_memory_percent']:.1%}")
            return True, max(config.min_batch_size, current_batch_size // config.batch_size_step)
        
        # Check GPU memory if available
        if self.device_type == "cuda" and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated_key = f"gpu_{i}_memory_allocated_gb"
                total_key = f"gpu_{i}_memory_total_gb"
                if allocated_key in stats and total_key in stats:
                    usage = stats[allocated_key] / stats[total_key]
                    if usage > config.memory_threshold:
                        logger.warning(f"GPU {i} memory usage high: {usage:.1%}")
                        return True, max(config.min_batch_size, current_batch_size // config.batch_size_step)
        
        # Check if we can increase batch size
        if stats["system_memory_percent"] < config.memory_threshold * 0.5:
            # Memory usage is low, consider increasing batch size
            if self.device_type == "cuda" and torch.cuda.is_available():
                can_increase = True
                for i in range(torch.cuda.device_count()):
                    allocated_key = f"gpu_{i}_memory_allocated_gb"
                    total_key = f"gpu_{i}_memory_total_gb"
                    if allocated_key in stats and total_key in stats:
                        usage = stats[allocated_key] / stats[total_key]
                        if usage > config.memory_threshold * 0.7:
                            can_increase = False
                            break
                if can_increase:
                    new_batch_size = min(config.max_batch_size, current_batch_size * config.batch_size_step)
                    if new_batch_size != current_batch_size:
                        return True, new_batch_size
        
        return False, current_batch_size


class DistributedBackendDetector:
    """Detect and interface with different distributed backends."""
    
    @staticmethod
    def detect_backend() -> DistributedBackend:
        """Detect which distributed backend is being used."""
        if not HAS_TORCH or not dist.is_initialized():
            return DistributedBackend.UNKNOWN
        
        # Check for DeepSpeed
        if HAS_DEEPSPEED:
            try:
                import deepspeed
                if hasattr(deepspeed, 'initialized'):
                    return DistributedBackend.DEEPSPEED
            except:
                pass
        
        # Check for FSDP
        if HAS_FSDP:
            try:
                # Check if any module is wrapped with FSDP
                for obj in gc.get_objects():
                    if isinstance(obj, FSDP):
                        return DistributedBackend.FSDP
            except:
                pass
        
        # Check for Megatron
        if HAS_MEGATRON:
            try:
                if mpu.model_parallel_is_initialized():
                    return DistributedBackend.MEGATRON
            except:
                pass
        
        # Default to DDP
        return DistributedBackend.DDP
    
    @staticmethod
    def get_world_size() -> int:
        """Get world size for current distributed backend."""
        if not HAS_TORCH or not dist.is_initialized():
            return 1
        
        backend = DistributedBackendDetector.detect_backend()
        
        if backend == DistributedBackend.MEGATRON and HAS_MEGATRON:
            return mpu.get_data_parallel_world_size()
        else:
            return dist.get_world_size()
    
    @staticmethod
    def get_rank() -> int:
        """Get rank for current distributed backend."""
        if not HAS_TORCH or not dist.is_initialized():
            return 0
        
        backend = DistributedBackendDetector.detect_backend()
        
        if backend == DistributedBackend.MEGATRON and HAS_MEGATRON:
            return mpu.get_data_parallel_rank()
        else:
            return dist.get_rank()
    
    @staticmethod
    def get_local_rank() -> int:
        """Get local rank for current distributed backend."""
        if not HAS_TORCH or not dist.is_initialized():
            return 0
        
        backend = DistributedBackendDetector.detect_backend()
        
        if backend == DistributedBackend.MEGATRON and HAS_MEGATRON:
            return mpu.get_tensor_model_parallel_rank()
        else:
            return int(os.environ.get("LOCAL_RANK", 0))


class CheckpointManager:
    """
    Manages checkpointing for distributed training with fault tolerance
    and elastic scaling capabilities.
    """
    
    def __init__(
        self,
        checkpoint_config: CheckpointConfig,
        resilience_config: ResilienceConfig,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        training_state: Optional[TrainingState] = None,
        backend: Optional[DistributedBackend] = None
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_config: Configuration for checkpointing
            resilience_config: Configuration for resilience features
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint
            scheduler: The learning rate scheduler to checkpoint
            training_state: Current training state
            backend: Distributed backend to use (auto-detected if None)
        """
        self.checkpoint_config = checkpoint_config
        self.resilience_config = resilience_config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_state = training_state or TrainingState()
        
        # Auto-detect backend if not specified
        self.backend = backend or DistributedBackendDetector.detect_backend()
        
        # Initialize node information
        self.node_info = self._initialize_node_info()
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.node_info.device_type)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread for async operations
        self.save_thread = None
        self.stop_event = threading.Event()
        
        # Heartbeat thread for fault tolerance
        self.heartbeat_thread = None
        self.node_status = {}
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.last_recovery_time = 0
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info(f"CheckpointManager initialized with backend: {self.backend.value}")
        logger.info(f"Node info: {self.node_info}")
    
    def _initialize_node_info(self) -> NodeInfo:
        """Initialize node information."""
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "unknown"
        
        device_type = "cpu"
        device_count = 0
        
        if HAS_TORCH and torch.cuda.is_available():
            device_type = "cuda"
            device_count = torch.cuda.device_count()
        elif "TPU_NAME" in os.environ:
            device_type = "tpu"
            device_count = 8  # Typical TPU core count
        
        memory = psutil.virtual_memory()
        
        return NodeInfo(
            node_id=f"{hostname}_{ip_address}_{os.getpid()}",
            hostname=hostname,
            ip_address=ip_address,
            rank=DistributedBackendDetector.get_rank(),
            local_rank=DistributedBackendDetector.get_local_rank(),
            world_size=DistributedBackendDetector.get_world_size(),
            device_type=device_type,
            device_count=device_count,
            memory_total=memory.total / (1024**3),
            memory_available=memory.available / (1024**3)
        )
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, saving checkpoint before exit...")
            self.save_checkpoint(emergency=True)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def save_checkpoint(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        emergency: bool = False
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step (uses training_state.step if None)
            epoch: Current epoch (uses training_state.epoch if None)
            emergency: If True, save synchronously even if async_save is enabled
            
        Returns:
            Path to the saved checkpoint
        """
        if step is None:
            step = self.training_state.step
        if epoch is None:
            epoch = self.training_state.epoch
        
        # Update training state
        self.training_state.step = step
        self.training_state.epoch = epoch
        self.training_state.global_step = step
        
        # Determine checkpoint path
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Check if we should save
        should_save = self._should_save_checkpoint(step, emergency)
        if not should_save:
            return ""
        
        # Save checkpoint
        if self.checkpoint_config.async_save and not emergency:
            if self.save_thread and self.save_thread.is_alive():
                logger.warning("Previous save operation still in progress, skipping")
                return ""
            
            self.save_thread = threading.Thread(
                target=self._save_checkpoint_async,
                args=(checkpoint_path,),
                daemon=True
            )
            self.save_thread.start()
            return str(checkpoint_path)
        else:
            return self._save_checkpoint_sync(checkpoint_path)
    
    def _should_save_checkpoint(self, step: int, emergency: bool) -> bool:
        """Determine if a checkpoint should be saved."""
        if emergency:
            return True
        
        if step % self.checkpoint_config.save_frequency == 0:
            return True
        
        if self.checkpoint_config.save_on_epoch_end and step == 0:
            return True
        
        return False
    
    def _save_checkpoint_sync(self, checkpoint_path: Path) -> str:
        """Save checkpoint synchronously."""
        try:
            # Create checkpoint directory
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save based on backend
            if self.backend == DistributedBackend.DEEPSPEED and HAS_DEEPSPEED:
                self._save_deepspeed_checkpoint(checkpoint_path)
            elif self.backend == DistributedBackend.FSDP and HAS_FSDP:
                self._save_fsdp_checkpoint(checkpoint_path)
            elif self.backend == DistributedBackend.MEGATRON and HAS_MEGATRON:
                self._save_megatron_checkpoint(checkpoint_path)
            else:
                self._save_pytorch_checkpoint(checkpoint_path)
            
            # Save training state
            self._save_training_state(checkpoint_path)
            
            # Save metadata
            self._save_metadata(checkpoint_path)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Upload to remote storage if configured
            if self.checkpoint_config.remote_storage:
                self._upload_checkpoint(checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def _save_checkpoint_async(self, checkpoint_path: Path):
        """Save checkpoint in background thread."""
        try:
            self._save_checkpoint_sync(checkpoint_path)
        except Exception as e:
            logger.error(f"Async checkpoint save failed: {e}")
    
    def _save_pytorch_checkpoint(self, checkpoint_path: Path):
        """Save checkpoint in PyTorch format."""
        checkpoint_data = {}
        
        # Save model state
        if self.model is not None:
            if hasattr(self.model, 'module'):  # DDP wrapped
                checkpoint_data['model_state_dict'] = self.model.module.state_dict()
            else:
                checkpoint_data['model_state_dict'] = self.model.state_dict()
        
        # Save optimizer state
        if self.optimizer is not None and self.checkpoint_config.save_optimizer:
            checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Save scheduler state
        if self.scheduler is not None and self.checkpoint_config.save_scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save RNG states
        if self.checkpoint_config.save_rng_state:
            checkpoint_data['rng_states'] = self._get_rng_states()
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_file)
    
    def _save_deepspeed_checkpoint(self, checkpoint_path: Path):
        """Save checkpoint for DeepSpeed."""
        if not HAS_DEEPSPEED or self.model is None:
            return
        
        # DeepSpeed handles its own checkpointing
        self.model.save_checkpoint(str(checkpoint_path))
    
    def _save_fsdp_checkpoint(self, checkpoint_path: Path):
        """Save checkpoint for FSDP."""
        if not HAS_FSDP or self.model is None:
            return
        
        # Save with FSDP's full state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.model.state_dict()
            
        if DistributedBackendDetector.get_rank() == 0:
            checkpoint_file = checkpoint_path / "model_state_dict.pt"
            torch.save(state_dict, checkpoint_file)
    
    def _save_megatron_checkpoint(self, checkpoint_path: Path):
        """Save checkpoint for Megatron."""
        if not HAS_MEGATRON or self.model is None:
            return
        
        # Megatron has its own checkpointing mechanism
        # This is a simplified version
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        
        if DistributedBackendDetector.get_rank() == 0:
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'training_state': asdict(self.training_state)
            }
            torch.save(checkpoint_data, checkpoint_file)
    
    def _save_training_state(self, checkpoint_path: Path):
        """Save training state."""
        state_file = checkpoint_path / "training_state.json"
        
        # Convert training state to serializable format
        state_dict = asdict(self.training_state)
        
        # Convert non-serializable items
        for key, value in state_dict.items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                state_dict[key] = value.item() if hasattr(value, 'item') else value.tolist()
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def _save_metadata(self, checkpoint_path: Path):
        """Save checkpoint metadata."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'step': self.training_state.step,
            'epoch': self.training_state.epoch,
            'backend': self.backend.value,
            'node_info': asdict(self.node_info),
            'checkpoint_config': asdict(self.checkpoint_config),
            'resilience_config': asdict(self.resilience_config),
            'memory_stats': self.memory_monitor.get_memory_stats()
        }
        
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_rng_states(self) -> Dict[str, Any]:
        """Get RNG states for reproducibility."""
        states = {}
        
        if HAS_TORCH:
            states['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                states['cuda_rng_state'] = torch.cuda.get_rng_state()
                states['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        
        states['numpy_rng_state'] = np.random.get_state()
        states['python_rng_state'] = __import__('random').getstate()
        
        return states
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        while len(checkpoints) > self.checkpoint_config.keep_last_n:
            old_checkpoint = checkpoints.pop(0)
            try:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def _upload_checkpoint(self, checkpoint_path: Path):
        """Upload checkpoint to remote storage."""
        # This is a placeholder for actual remote storage implementation
        # In production, you would use boto3 for S3, google-cloud-storage for GCS, etc.
        logger.info(f"Would upload {checkpoint_path} to {self.checkpoint_config.remote_storage}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_training_state: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (loads latest if None)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            load_training_state: Whether to load training state
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        # Find checkpoint to load
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                logger.info("No checkpoint found to load")
                return {}
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load based on backend
        if self.backend == DistributedBackend.DEEPSPEED and HAS_DEEPSPEED:
            return self._load_deepspeed_checkpoint(
                checkpoint_path, load_optimizer, load_scheduler, load_training_state
            )
        elif self.backend == DistributedBackend.FSDP and HAS_FSDP:
            return self._load_fsdp_checkpoint(
                checkpoint_path, load_optimizer, load_scheduler, load_training_state
            )
        elif self.backend == DistributedBackend.MEGATRON and HAS_MEGATRON:
            return self._load_megatron_checkpoint(
                checkpoint_path, load_optimizer, load_scheduler, load_training_state
            )
        else:
            return self._load_pytorch_checkpoint(
                checkpoint_path, load_optimizer, load_scheduler, load_training_state
            )
    
    def _load_pytorch_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool,
        load_scheduler: bool,
        load_training_state: bool
    ) -> Dict[str, Any]:
        """Load checkpoint in PyTorch format."""
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
        
        # Load model state
        if 'model_state_dict' in checkpoint_data and self.model is not None:
            if hasattr(self.model, 'module'):  # DDP wrapped
                self.model.module.load_state_dict(checkpoint_data['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint_data and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and 'scheduler_state_dict' in checkpoint_data and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Load RNG states
        if 'rng_states' in checkpoint_data and self.checkpoint_config.save_rng_state:
            self._set_rng_states(checkpoint_data['rng_states'])
        
        # Load training state
        if load_training_state:
            self._load_training_state(checkpoint_path)
        
        return {
            'step': self.training_state.step,
            'epoch': self.training_state.epoch,
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _load_deepspeed_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool,
        load_scheduler: bool,
        load_training_state: bool
    ) -> Dict[str, Any]:
        """Load checkpoint for DeepSpeed."""
        if not HAS_DEEPSPEED or self.model is None:
            return {}
        
        # DeepSpeed handles its own checkpoint loading
        self.model.load_checkpoint(str(checkpoint_path))
        
        if load_training_state:
            self._load_training_state(checkpoint_path)
        
        return {
            'step': self.training_state.step,
            'epoch': self.training_state.epoch,
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _load_fsdp_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool,
        load_scheduler: bool,
        load_training_state: bool
    ) -> Dict[str, Any]:
        """Load checkpoint for FSDP."""
        if not HAS_FSDP or self.model is None:
            return {}
        
        checkpoint_file = checkpoint_path / "model_state_dict.pt"
        
        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            
            # Load with FSDP
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(state_dict)
        
        if load_training_state:
            self._load_training_state(checkpoint_path)
        
        return {
            'step': self.training_state.step,
            'epoch': self.training_state.epoch,
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _load_megatron_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool,
        load_scheduler: bool,
        load_training_state: bool
    ) -> Dict[str, Any]:
        """Load checkpoint for Megatron."""
        if not HAS_MEGATRON or self.model is None:
            return {}
        
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        
        if checkpoint_file.exists():
            checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
            
            if 'model_state_dict' in checkpoint_data:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            if load_training_state and 'training_state' in checkpoint_data:
                for key, value in checkpoint_data['training_state'].items():
                    if hasattr(self.training_state, key):
                        setattr(self.training_state, key, value)
        
        return {
            'step': self.training_state.step,
            'epoch': self.training_state.epoch,
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _load_training_state(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        state_file = checkpoint_path / "training_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Update training state
            for key, value in state_dict.items():
                if hasattr(self.training_state, key):
                    setattr(self.training_state, key, value)
    
    def _set_rng_states(self, rng_states: Dict[str, Any]):
        """Restore RNG states."""
        if HAS_TORCH and 'torch_rng_state' in rng_states:
            torch.set_rng_state(rng_states['torch_rng_state'])
            
            if torch.cuda.is_available() and 'cuda_rng_state' in rng_states:
                torch.cuda.set_rng_state(rng_states['cuda_rng_state'])
            
            if 'cuda_rng_state_all' in rng_states:
                torch.cuda.set_rng_state_all(rng_states['cuda_rng_state_all'])
        
        if 'numpy_rng_state' in rng_states:
            np.random.set_state(rng_states['numpy_rng_state'])
        
        if 'python_rng_state' in rng_states:
            __import__('random').setstate(rng_states['python_rng_state'])
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
            reverse=True
        )
        
        return str(checkpoints[0]) if checkpoints else None
    
    def start_heartbeat(self):
        """Start heartbeat monitoring for fault tolerance."""
        if not self.resilience_config.enable_fault_tolerance:
            return
        
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.info("Heartbeat monitoring started")
    
    def _heartbeat_loop(self):
        """Heartbeat loop for node health monitoring."""
        while not self.stop_event.is_set():
            try:
                # Update own heartbeat
                self.node_info.last_heartbeat = time.time()
                self.node_info.status = NodeStatus.HEALTHY
                
                # Check other nodes (in a real implementation, this would involve
                # communication with other nodes via the distributed backend)
                self._check_node_health()
                
                # Sleep until next heartbeat
                time.sleep(self.resilience_config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _check_node_health(self):
        """Check health of all nodes in the cluster."""
        # This is a simplified implementation
        # In production, you would use the distributed backend to check all nodes
        
        current_time = time.time()
        
        # Check if any nodes have missed heartbeats
        for node_id, last_heartbeat in list(self.node_status.items()):
            if current_time - last_heartbeat > self.resilience_config.failure_timeout:
                logger.warning(f"Node {node_id} appears to have failed")
                self._handle_node_failure(node_id)
    
    def _handle_node_failure(self, failed_node_id: str):
        """Handle node failure."""
        if not self.resilience_config.enable_auto_recovery:
            logger.error(f"Node {failed_node_id} failed, auto-recovery disabled")
            return
        
        if self.recovery_attempts >= self.resilience_config.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for node {failed_node_id}")
            return
        
        current_time = time.time()
        if current_time - self.last_recovery_time < self.resilience_config.recovery_delay:
            logger.info(f"Waiting before recovery attempt for node {failed_node_id}")
            return
        
        logger.info(f"Attempting recovery for node {failed_node_id}")
        self.recovery_attempts += 1
        self.last_recovery_time = current_time
        
        # Save emergency checkpoint
        self.save_checkpoint(emergency=True)
        
        # In a real implementation, you would:
        # 1. Notify the distributed backend about the failure
        # 2. Respawn the failed node if possible
        # 3. Redistribute workload
        # 4. Resume training from the last checkpoint
    
    def adjust_batch_size(self) -> int:
        """
        Adjust batch size based on available memory.
        
        Returns:
            New batch size
        """
        if not self.resilience_config.enable_elastic_scaling:
            return self.training_state.batch_size
        
        should_adjust, new_batch_size = self.memory_monitor.should_adjust_batch_size(
            self.training_state.batch_size,
            self.resilience_config
        )
        
        if should_adjust and new_batch_size != self.training_state.batch_size:
            logger.info(
                f"Adjusting batch size from {self.training_state.batch_size} "
                f"to {new_batch_size}"
            )
            self.training_state.batch_size = new_batch_size
            
            # In a real implementation, you would also need to:
            # 1. Adjust gradient accumulation steps
            # 2. Adjust learning rate if using linear scaling
            # 3. Reconfigure data loader
        
        return self.training_state.batch_size
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring."""
        return {
            "node_info": asdict(self.node_info),
            "training_state": asdict(self.training_state),
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "checkpoint_config": asdict(self.checkpoint_config),
            "resilience_config": asdict(self.resilience_config),
            "backend": self.backend.value,
            "recovery_attempts": self.recovery_attempts,
            "last_recovery_time": self.last_recovery_time
        }
    
    def shutdown(self):
        """Gracefully shutdown the checkpoint manager."""
        logger.info("Shutting down CheckpointManager")
        self.stop_event.set()
        
        # Save final checkpoint
        self.save_checkpoint(emergency=True)
        
        # Wait for threads to finish
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=10)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        logger.info("CheckpointManager shutdown complete")


class ResilienceManager:
    """
    High-level manager for distributed training resilience.
    Integrates checkpointing, fault tolerance, and elastic scaling.
    """
    
    def __init__(
        self,
        checkpoint_config: Optional[CheckpointConfig] = None,
        resilience_config: Optional[ResilienceConfig] = None,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ):
        """
        Initialize the resilience manager.
        
        Args:
            checkpoint_config: Checkpoint configuration
            resilience_config: Resilience configuration
            model: Model to manage
            optimizer: Optimizer to manage
            scheduler: Scheduler to manage
        """
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.resilience_config = resilience_config or ResilienceConfig()
        
        # Initialize training state
        self.training_state = TrainingState()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config=self.checkpoint_config,
            resilience_config=self.resilience_config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=self.training_state
        )
        
        # Start heartbeat if fault tolerance is enabled
        if self.resilience_config.enable_fault_tolerance:
            self.checkpoint_manager.start_heartbeat()
        
        logger.info("ResilienceManager initialized")
    
    def on_training_step(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """
        Called at each training step.
        
        Args:
            step: Current step
            loss: Current loss
            metrics: Additional metrics
        """
        self.training_state.step = step
        self.training_state.loss = loss
        
        if metrics:
            self.training_state.metrics.update(metrics)
        
        # Check if we should save checkpoint
        if step % self.checkpoint_config.save_frequency == 0:
            self.checkpoint_manager.save_checkpoint(step=step)
        
        # Adjust batch size if needed
        if self.resilience_config.enable_elastic_scaling:
            new_batch_size = self.checkpoint_manager.adjust_batch_size()
            if new_batch_size != self.training_state.batch_size:
                # In a real implementation, you would adjust the data loader here
                pass
    
    def on_epoch_end(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch
            metrics: Epoch metrics
        """
        self.training_state.epoch = epoch
        
        if metrics:
            self.training_state.metrics.update(metrics)
        
        # Save checkpoint at epoch end if configured
        if self.checkpoint_config.save_on_epoch_end:
            self.checkpoint_manager.save_checkpoint(epoch=epoch)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint information
        """
        return self.checkpoint_manager.load_checkpoint(checkpoint_path)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current resilience status."""
        return self.checkpoint_manager.get_system_status()
    
    def shutdown(self):
        """Shutdown the resilience manager."""
        self.checkpoint_manager.shutdown()


# Factory function for easy creation
def create_resilience_manager(
    checkpoint_dir: str = "./checkpoints",
    save_frequency: int = 1000,
    enable_fault_tolerance: bool = True,
    enable_elastic_scaling: bool = True,
    **kwargs
) -> ResilienceManager:
    """
    Factory function to create a ResilienceManager with common configurations.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        save_frequency: Save checkpoint every N steps
        enable_fault_tolerance: Enable fault tolerance features
        enable_elastic_scaling: Enable elastic scaling features
        **kwargs: Additional arguments for ResilienceManager
        
    Returns:
        Configured ResilienceManager instance
    """
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency
    )
    
    resilience_config = ResilienceConfig(
        enable_fault_tolerance=enable_fault_tolerance,
        enable_elastic_scaling=enable_elastic_scaling
    )
    
    return ResilienceManager(
        checkpoint_config=checkpoint_config,
        resilience_config=resilience_config,
        **kwargs
    )


# Integration with existing forge training scripts
def integrate_with_forge_trainer(trainer, config: Optional[Dict[str, Any]] = None):
    """
    Integrate resilience features with forge's trainer.
    
    Args:
        trainer: forge trainer instance
        config: Configuration dictionary
        
    Returns:
        ResilienceManager instance
    """
    config = config or {}
    
    # Extract configuration
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    save_frequency = config.get('save_frequency', 1000)
    
    # Create resilience manager
    resilience_manager = create_resilience_manager(
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.lr_scheduler
    )
    
    # Hook into trainer's training loop
    original_training_step = trainer.training_step
    
    def wrapped_training_step(*args, **kwargs):
        result = original_training_step(*args, **kwargs)
        
        # Update resilience manager
        step = trainer.state.global_step
        loss = result.loss.item() if hasattr(result, 'loss') else 0.0
        
        resilience_manager.on_training_step(
            step=step,
            loss=loss,
            metrics={'learning_rate': trainer.get_lr()}
        )
        
        return result
    
    trainer.training_step = wrapped_training_step
    
    # Hook into epoch end
    if hasattr(trainer, 'on_epoch_end'):
        original_on_epoch_end = trainer.on_epoch_end
        
        def wrapped_on_epoch_end(*args, **kwargs):
            result = original_on_epoch_end(*args, **kwargs)
            
            resilience_manager.on_epoch_end(
                epoch=trainer.state.epoch,
                metrics=trainer.state.log_history[-1] if trainer.state.log_history else {}
            )
            
            return result
        
        trainer.on_epoch_end = wrapped_on_epoch_end
    
    logger.info("Resilience features integrated with forge trainer")
    return resilience_manager


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Manager Example")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--save_frequency", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--enable_fault_tolerance", action="store_true",
                        help="Enable fault tolerance")
    parser.add_argument("--enable_elastic_scaling", action="store_true",
                        help="Enable elastic scaling")
    
    args = parser.parse_args()
    
    # Create resilience manager
    manager = create_resilience_manager(
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        enable_fault_tolerance=args.enable_fault_tolerance,
        enable_elastic_scaling=args.enable_elastic_scaling
    )
    
    # Simulate training
    try:
        for step in range(1000):
            # Simulate training step
            loss = 1.0 / (step + 1)
            
            manager.on_training_step(
                step=step,
                loss=loss,
                metrics={"accuracy": min(1.0, step / 1000)}
            )
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
                print(f"System status: {manager.get_status()}")
            
            time.sleep(0.1)  # Simulate training time
            
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        manager.shutdown()
        print("Shutdown complete")