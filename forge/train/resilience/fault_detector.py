import os
import time
import torch
import torch.distributed as dist
import logging
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Supported distributed backends."""
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    MEGATRON = "megatron"
    PYTORCH = "pytorch"

class FailureType(Enum):
    """Types of failures that can be detected."""
    NODE_FAILURE = "node_failure"
    GPU_OOM = "gpu_oom"
    NETWORK_FAILURE = "network_failure"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    MEMORY_OVERFLOW = "memory_overflow"

@dataclass
class ResilienceConfig:
    """Configuration for the resilience layer."""
    # Checkpointing settings
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 1000  # steps
    checkpoint_keep_last: int = 3
    checkpoint_async: bool = True
    checkpoint_compression: bool = True
    
    # Fault detection settings
    heartbeat_interval: float = 30.0  # seconds
    failure_timeout: float = 300.0  # seconds
    max_retries: int = 3
    retry_delay: float = 60.0  # seconds
    
    # Elastic scaling settings
    enable_elastic_scaling: bool = True
    batch_size_adjustment_interval: int = 100  # steps
    min_batch_size: int = 1
    max_batch_size: int = 1024
    memory_threshold: float = 0.85  # 85% memory usage threshold
    
    # Backend settings
    backend: BackendType = BackendType.PYTORCH
    backend_config: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery settings
    auto_recovery: bool = True
    recovery_strategy: str = "latest"  # "latest", "best", "specific"
    recovery_checkpoint_path: Optional[str] = None

class FaultDetector:
    """
    Distributed Training Resilience Layer for forge.
    
    Provides automatic checkpointing, fault detection, recovery, and elastic scaling
    for distributed training across heterogeneous hardware.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        config: ResilienceConfig,
        train_state: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.train_state = train_state or {}
        
        self.current_step = self.train_state.get("global_step", 0)
        self.current_epoch = self.train_state.get("epoch", 0)
        self.best_metric = self.train_state.get("best_metric", float("inf"))
        
        self._setup_distributed()
        self._setup_checkpointing()
        self._setup_fault_detection()
        self._setup_elastic_scaling()
        
        self._failure_history = []
        self._recovery_attempts = 0
        self._last_heartbeat = time.time()
        self._shutdown_flag = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"FaultDetector initialized with backend: {self.config.backend.value}")
    
    def _setup_distributed(self):
        """Setup distributed backend and detect hardware."""
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = torch.device("cpu")
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.device)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device(f"xpu:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
        
        # Detect hardware capabilities
        self.hardware_info = self._detect_hardware()
        
        # Initialize backend-specific components
        if self.config.backend == BackendType.DEEPSPEED:
            self._setup_deepspeed()
        elif self.config.backend == BackendType.FSDP:
            self._setup_fsdp()
        elif self.config.backend == BackendType.MEGATRON:
            self._setup_megatron()
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware capabilities and memory."""
        info = {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device_type": self.device.type,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "memory_info": {}
        }
        
        # Get memory information
        if self.device.type == "cuda":
            for i in range(torch.cuda.device_count()):
                mem_info = {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "allocated": torch.cuda.memory_allocated(i),
                    "cached": torch.cuda.memory_reserved(i)
                }
                info["memory_info"][f"cuda:{i}"] = mem_info
        elif self.device.type == "xpu":
            for i in range(torch.xpu.device_count()):
                mem_info = {
                    "total": torch.xpu.get_device_properties(i).total_memory,
                    "allocated": torch.xpu.memory_allocated(i),
                    "cached": torch.xpu.memory_reserved(i)
                }
                info["memory_info"][f"xpu:{i}"] = mem_info
        
        # System memory
        info["system_memory"] = {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        }
        
        return info
    
    def _setup_checkpointing(self):
        """Setup checkpointing infrastructure."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different checkpoint types
        (self.checkpoint_dir / "regular").mkdir(exist_ok=True)
        (self.checkpoint_dir / "emergency").mkdir(exist_ok=True)
        (self.checkpoint_dir / "best").mkdir(exist_ok=True)
        
        # Thread pool for async operations
        if self.config.checkpoint_async:
            self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Checkpoint metadata
        self._checkpoint_metadata = self._load_checkpoint_metadata()
    
    def _setup_fault_detection(self):
        """Setup fault detection mechanisms."""
        self._node_status = {i: True for i in range(self.world_size)}
        self._heartbeat_thread = None
        self._monitoring_active = False
        
        if self.world_size > 1:
            self._start_heartbeat_monitor()
    
    def _setup_elastic_scaling(self):
        """Setup elastic scaling mechanisms."""
        self.current_batch_size = self.train_state.get("batch_size", 1)
        self._batch_size_history = []
        self._memory_monitor_thread = None
        
        if self.config.enable_elastic_scaling and self.world_size > 1:
            self._start_memory_monitor()
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed-specific components."""
        try:
            import deepspeed
            self.deepspeed = deepspeed
            self._deepspeed_initialized = True
            
            # DeepSpeed checkpointing
            self.ds_checkpoint = deepspeed.checkpointing
            self.ds_checkpoint.configure(
                self.model,
                partition_activations=True,
                contiguous_checkpointing=True,
                checkpoint_in_cpu=True,
                profile_backward=False
            )
            
            logger.info("DeepSpeed checkpointing configured")
        except ImportError:
            logger.warning("DeepSpeed not available, falling back to PyTorch")
            self.config.backend = BackendType.PYTORCH
            self._deepspeed_initialized = False
    
    def _setup_fsdp(self):
        """Setup FSDP-specific components."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
                ShardedStateDictConfig
            )
            self.fsdp = FSDP
            self._fsdp_initialized = True
            
            # FSDP state dict configs
            self.full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
            self.sharded_state_dict_config = ShardedStateDictConfig()
            
            logger.info("FSDP checkpointing configured")
        except ImportError:
            logger.warning("FSDP not available, falling back to PyTorch")
            self.config.backend = BackendType.PYTORCH
            self._fsdp_initialized = False
    
    def _setup_megatron(self):
        """Setup Megatron-specific components."""
        try:
            from megatron.core import parallel_state
            from megatron.checkpointing import load_checkpoint, save_checkpoint
            self.megatron_parallel_state = parallel_state
            self.megatron_load_checkpoint = load_checkpoint
            self.megatron_save_checkpoint = save_checkpoint
            self._megatron_initialized = True
            
            logger.info("Megatron checkpointing configured")
        except ImportError:
            logger.warning("Megatron not available, falling back to PyTorch")
            self.config.backend = BackendType.PYTORCH
            self._megatron_initialized = False
    
    def _load_checkpoint_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from disk."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
        
        return {
            "checkpoints": [],
            "last_checkpoint_step": -1,
            "best_checkpoint_step": -1,
            "best_metric": float("inf")
        }
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to disk."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(self._checkpoint_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    def _start_heartbeat_monitor(self):
        """Start heartbeat monitoring thread."""
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._monitoring_active = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor_loop,
                daemon=True
            )
            self._heartbeat_thread.start()
    
    def _heartbeat_monitor_loop(self):
        """Heartbeat monitoring loop."""
        while self._monitoring_active and not self._shutdown_flag:
            try:
                self._check_node_health()
                self._send_heartbeat()
                time.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(5)  # Backoff on error
    
    def _check_node_health(self):
        """Check health of all nodes in the distributed setup."""
        if self.world_size <= 1:
            return
        
        # Create a heartbeat tensor
        heartbeat = torch.tensor([self.rank], device=self.device)
        
        # Gather heartbeats from all ranks
        if dist.is_initialized():
            try:
                gathered = [torch.zeros_like(heartbeat) for _ in range(self.world_size)]
                dist.all_gather(gathered, heartbeat)
                
                current_time = time.time()
                for i, hb in enumerate(gathered):
                    if hb.item() != i:
                        if self._node_status[i]:
                            logger.warning(f"Node {i} heartbeat missing")
                            self._node_status[i] = False
                            self._handle_node_failure(i)
                    else:
                        self._node_status[i] = True
            except Exception as e:
                logger.error(f"Failed to check node health: {e}")
    
    def _send_heartbeat(self):
        """Send heartbeat to indicate this node is alive."""
        self._last_heartbeat = time.time()
    
    def _handle_node_failure(self, node_rank: int):
        """Handle node failure detection."""
        failure_info = {
            "type": FailureType.NODE_FAILURE,
            "node_rank": node_rank,
            "timestamp": time.time(),
            "step": self.current_step
        }
        
        self._failure_history.append(failure_info)
        logger.error(f"Node {node_rank} failure detected at step {self.current_step}")
        
        if self.config.auto_recovery and self._recovery_attempts < self.config.max_retries:
            self._initiate_recovery(failure_info)
    
    def _start_memory_monitor(self):
        """Start memory monitoring thread for elastic scaling."""
        if self._memory_monitor_thread is None or not self._memory_monitor_thread.is_alive():
            self._memory_monitor_thread = threading.Thread(
                target=self._memory_monitor_loop,
                daemon=True
            )
            self._memory_monitor_thread.start()
    
    def _memory_monitor_loop(self):
        """Memory monitoring loop for elastic scaling."""
        while not self._shutdown_flag:
            try:
                if self.current_step % self.config.batch_size_adjustment_interval == 0:
                    self._adjust_batch_size()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(30)  # Backoff on error
    
    def _adjust_batch_size(self):
        """Adjust batch size based on available memory."""
        if not self.config.enable_elastic_scaling:
            return
        
        # Get current memory usage
        memory_usage = self._get_memory_usage()
        
        # Determine if we need to adjust batch size
        if memory_usage > self.config.memory_threshold:
            # Memory pressure, reduce batch size
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            if new_batch_size != self.current_batch_size:
                logger.info(
                    f"Reducing batch size from {self.current_batch_size} to {new_batch_size} "
                    f"(memory usage: {memory_usage:.1%})"
                )
                self.current_batch_size = new_batch_size
                self._update_train_state()
        elif memory_usage < self.config.memory_threshold * 0.7:
            # Memory available, increase batch size
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            if new_batch_size != self.current_batch_size:
                logger.info(
                    f"Increasing batch size from {self.current_batch_size} to {new_batch_size} "
                    f"(memory usage: {memory_usage:.1%})"
                )
                self.current_batch_size = new_batch_size
                self._update_train_state()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as a fraction."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        elif self.device.type == "xpu":
            allocated = torch.xpu.memory_allocated(self.device)
            total = torch.xpu.get_device_properties(self.device).total_memory
            return allocated / total
        else:
            return psutil.virtual_memory().percent / 100.0
    
    def _update_train_state(self):
        """Update training state with current values."""
        self.train_state.update({
            "global_step": self.current_step,
            "epoch": self.current_epoch,
            "batch_size": self.current_batch_size,
            "best_metric": self.best_metric
        })
    
    def on_train_begin(self):
        """Called at the beginning of training."""
        logger.info("Starting training with resilience layer")
        
        # Try to resume from checkpoint if configured
        if self.config.auto_recovery:
            self._try_resume_from_checkpoint()
        
        # Start monitoring
        if self.world_size > 1:
            self._start_heartbeat_monitor()
        
        if self.config.enable_elastic_scaling:
            self._start_memory_monitor()
    
    def on_step_end(self, step: int, loss: Optional[float] = None, metrics: Optional[Dict[str, float]] = None):
        """Called at the end of each training step."""
        self.current_step = step
        
        # Update best metric if provided
        if metrics and "eval_loss" in metrics:
            if metrics["eval_loss"] < self.best_metric:
                self.best_metric = metrics["eval_loss"]
                self._save_checkpoint("best")
        
        # Check if it's time for regular checkpoint
        if step > 0 and step % self.config.checkpoint_frequency == 0:
            self._save_checkpoint("regular")
        
        # Check for GPU OOM or other issues
        self._check_for_failures()
    
    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""
        self.current_epoch = epoch
        self._save_checkpoint("epoch")
    
    def on_train_end(self):
        """Called at the end of training."""
        logger.info("Training completed, saving final checkpoint")
        self._save_checkpoint("final")
        self._cleanup()
    
    def on_exception(self, exception: Exception):
        """Called when an exception occurs during training."""
        logger.error(f"Training exception: {exception}")
        
        # Determine failure type
        failure_type = self._classify_exception(exception)
        
        if failure_type == FailureType.GPU_OOM:
            self._handle_oom()
        elif failure_type == FailureType.CHECKPOINT_CORRUPTION:
            self._handle_checkpoint_corruption()
        else:
            self._handle_general_failure(exception)
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """Classify exception into failure type."""
        exception_str = str(exception).lower()
        
        if "out of memory" in exception_str or "oom" in exception_str:
            return FailureType.GPU_OOM
        elif "checkpoint" in exception_str and ("corrupt" in exception_str or "invalid" in exception_str):
            return FailureType.CHECKPOINT_CORRUPTION
        elif "network" in exception_str or "connection" in exception_str:
            return FailureType.NETWORK_FAILURE
        else:
            return FailureType.NODE_FAILURE
    
    def _handle_oom(self):
        """Handle GPU OOM error."""
        logger.warning("GPU OOM detected, attempting recovery")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size
        if self.config.enable_elastic_scaling:
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size // 2
            )
            logger.info(f"Reduced batch size to {self.current_batch_size} due to OOM")
        
        # Try to resume from last checkpoint
        if self.config.auto_recovery:
            self._try_resume_from_checkpoint()
    
    def _handle_checkpoint_corruption(self):
        """Handle checkpoint corruption."""
        logger.warning("Checkpoint corruption detected")
        
        # Try to load from previous checkpoint
        checkpoints = self._checkpoint_metadata.get("checkpoints", [])
        if len(checkpoints) > 1:
            # Remove corrupted checkpoint
            corrupted = checkpoints[-1]
            checkpoints = checkpoints[:-1]
            self._checkpoint_metadata["checkpoints"] = checkpoints
            
            # Try to load previous checkpoint
            if checkpoints:
                self._load_checkpoint(checkpoints[-1]["path"])
    
    def _handle_general_failure(self, exception: Exception):
        """Handle general training failure."""
        failure_info = {
            "type": FailureType.NODE_FAILURE,
            "exception": str(exception),
            "timestamp": time.time(),
            "step": self.current_step
        }
        
        self._failure_history.append(failure_info)
        
        if self.config.auto_recovery and self._recovery_attempts < self.config.max_retries:
            self._initiate_recovery(failure_info)
    
    def _initiate_recovery(self, failure_info: Dict[str, Any]):
        """Initiate recovery from failure."""
        self._recovery_attempts += 1
        logger.info(
            f"Initiating recovery (attempt {self._recovery_attempts}/{self.config.max_retries})"
        )
        
        # Wait before retry
        time.sleep(self.config.retry_delay)
        
        # Try to resume from checkpoint
        self._try_resume_from_checkpoint()
    
    def _try_resume_from_checkpoint(self):
        """Try to resume training from checkpoint."""
        checkpoint_path = self._find_recovery_checkpoint()
        
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path)
        else:
            logger.warning("No valid checkpoint found for recovery")
    
    def _find_recovery_checkpoint(self) -> Optional[Path]:
        """Find checkpoint for recovery based on strategy."""
        if self.config.recovery_strategy == "specific" and self.config.recovery_checkpoint_path:
            return Path(self.config.recovery_checkpoint_path)
        
        checkpoints = self._checkpoint_metadata.get("checkpoints", [])
        if not checkpoints:
            return None
        
        if self.config.recovery_strategy == "best":
            # Find checkpoint with best metric
            best_step = self._checkpoint_metadata.get("best_checkpoint_step", -1)
            for cp in checkpoints:
                if cp.get("step") == best_step:
                    return Path(cp["path"])
        
        # Default: latest checkpoint
        return Path(checkpoints[-1]["path"])
    
    def _save_checkpoint(self, checkpoint_type: str = "regular"):
        """Save checkpoint with resilience features."""
        if self._shutdown_flag:
            return
        
        try:
            checkpoint_name = f"checkpoint-{checkpoint_type}-{self.current_step}"
            checkpoint_path = self.checkpoint_dir / checkpoint_type / checkpoint_name
            
            # Prepare checkpoint data
            checkpoint_data = {
                "model_state_dict": self._get_model_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "step": self.current_step,
                "epoch": self.current_epoch,
                "batch_size": self.current_batch_size,
                "best_metric": self.best_metric,
                "config": self.config.__dict__,
                "timestamp": time.time()
            }
            
            # Save based on backend
            if self.config.backend == BackendType.DEEPSPEED and self._deepspeed_initialized:
                self._save_deepspeed_checkpoint(checkpoint_path, checkpoint_data)
            elif self.config.backend == BackendType.FSDP and self._fsdp_initialized:
                self._save_fsdp_checkpoint(checkpoint_path, checkpoint_data)
            elif self.config.backend == BackendType.MEGATRON and self._megatron_initialized:
                self._save_megatron_checkpoint(checkpoint_path, checkpoint_data)
            else:
                self._save_pytorch_checkpoint(checkpoint_path, checkpoint_data)
            
            # Update metadata
            self._update_checkpoint_metadata(checkpoint_path, checkpoint_type)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(checkpoint_type)
            
            logger.info(f"Saved {checkpoint_type} checkpoint at step {self.current_step}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Try emergency save
            self._emergency_save()
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get model state dict based on backend."""
        if hasattr(self.model, "module"):  # DDP wrapped
            return self.model.module.state_dict()
        elif self.config.backend == BackendType.FSDP and self._fsdp_initialized:
            with self.fsdp.state_dict_type(self.model, self.full_state_dict_config):
                return self.model.state_dict()
        else:
            return self.model.state_dict()
    
    def _save_deepspeed_checkpoint(self, path: Path, data: Dict[str, Any]):
        """Save checkpoint using DeepSpeed."""
        self.deepspeed.save_checkpoint(
            str(path),
            tag=f"global_step{self.current_step}",
            client_state=data
        )
    
    def _save_fsdp_checkpoint(self, path: Path, data: Dict[str, Any]):
        """Save checkpoint using FSDP."""
        with self.fsdp.state_dict_type(self.model, self.full_state_dict_config):
            model_state_dict = self.model.state_dict()
        
        if self.rank == 0:
            # Only rank 0 saves full state dict
            data["model_state_dict"] = model_state_dict
            torch.save(data, path)
        else:
            # Other ranks save sharded state dict
            with self.fsdp.state_dict_type(self.model, self.sharded_state_dict_config):
                sharded_state_dict = self.model.state_dict()
            torch.save(sharded_state_dict, f"{path}.shard{self.rank}")
    
    def _save_megatron_checkpoint(self, path: Path, data: Dict[str, Any]):
        """Save checkpoint using Megatron."""
        self.megatron_save_checkpoint(
            self.current_step,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            client_state=data
        )
    
    def _save_pytorch_checkpoint(self, path: Path, data: Dict[str, Any]):
        """Save checkpoint using PyTorch."""
        if self.config.checkpoint_compression:
            torch.save(data, path, _use_new_zipfile_serialization=True)
        else:
            torch.save(data, path)
    
    def _emergency_save(self):
        """Emergency checkpoint save when normal save fails."""
        try:
            emergency_path = self.checkpoint_dir / "emergency" / f"emergency-{self.current_step}"
            emergency_data = {
                "step": self.current_step,
                "epoch": self.current_epoch,
                "model_state_dict": self._get_model_state_dict(),
                "timestamp": time.time()
            }
            
            torch.save(emergency_data, emergency_path)
            logger.info(f"Emergency checkpoint saved to {emergency_path}")
            
        except Exception as e:
            logger.error(f"Emergency save also failed: {e}")
    
    def _update_checkpoint_metadata(self, path: Path, checkpoint_type: str):
        """Update checkpoint metadata."""
        checkpoint_info = {
            "path": str(path),
            "step": self.current_step,
            "epoch": self.current_epoch,
            "type": checkpoint_type,
            "timestamp": time.time(),
            "batch_size": self.current_batch_size
        }
        
        self._checkpoint_metadata["checkpoints"].append(checkpoint_info)
        self._checkpoint_metadata["last_checkpoint_step"] = self.current_step
        
        if checkpoint_type == "best":
            self._checkpoint_metadata["best_checkpoint_step"] = self.current_step
            self._checkpoint_metadata["best_metric"] = self.best_metric
        
        self._save_checkpoint_metadata()
    
    def _cleanup_old_checkpoints(self, checkpoint_type: str):
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self._checkpoint_metadata.get("checkpoints", [])
        type_checkpoints = [cp for cp in checkpoints if cp.get("type") == checkpoint_type]
        
        if len(type_checkpoints) > self.config.checkpoint_keep_last:
            # Sort by step (descending)
            type_checkpoints.sort(key=lambda x: x["step"], reverse=True)
            
            # Remove oldest checkpoints
            for cp in type_checkpoints[self.config.checkpoint_keep_last:]:
                try:
                    cp_path = Path(cp["path"])
                    if cp_path.exists():
                        if cp_path.is_file():
                            cp_path.unlink()
                        else:
                            # Directory checkpoint (DeepSpeed)
                            import shutil
                            shutil.rmtree(cp_path)
                    
                    # Remove from metadata
                    self._checkpoint_metadata["checkpoints"].remove(cp)
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {cp['path']}: {e}")
            
            self._save_checkpoint_metadata()
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint for recovery."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            # Load based on backend
            if self.config.backend == BackendType.DEEPSPEED and self._deepspeed_initialized:
                self._load_deepspeed_checkpoint(checkpoint_path)
            elif self.config.backend == BackendType.FSDP and self._fsdp_initialized:
                self._load_fsdp_checkpoint(checkpoint_path)
            elif self.config.backend == BackendType.MEGATRON and self._megatron_initialized:
                self._load_megatron_checkpoint(checkpoint_path)
            else:
                self._load_pytorch_checkpoint(checkpoint_path)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def _load_deepspeed_checkpoint(self, path: Path):
        """Load checkpoint using DeepSpeed."""
        _, client_state = self.deepspeed.load_checkpoint(
            str(path),
            tag=f"global_step{self.current_step}"
        )
        
        # Update training state
        if client_state:
            self.current_step = client_state.get("step", self.current_step)
            self.current_epoch = client_state.get("epoch", self.current_epoch)
            self.current_batch_size = client_state.get("batch_size", self.current_batch_size)
            self.best_metric = client_state.get("best_metric", self.best_metric)
    
    def _load_fsdp_checkpoint(self, path: Path):
        """Load checkpoint using FSDP."""
        if self.rank == 0:
            # Load full state dict on rank 0
            checkpoint = torch.load(path, map_location="cpu")
            
            # Load model state
            with self.fsdp.state_dict_type(self.model, self.full_state_dict_config):
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state
            if self.lr_scheduler and checkpoint.get("lr_scheduler_state_dict"):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            
            # Update training state
            self.current_step = checkpoint.get("step", self.current_step)
            self.current_epoch = checkpoint.get("epoch", self.current_epoch)
            self.current_batch_size = checkpoint.get("batch_size", self.current_batch_size)
            self.best_metric = checkpoint.get("best_metric", self.best_metric)
        else:
            # Load sharded state dict on other ranks
            shard_path = f"{path}.shard{self.rank}"
            if Path(shard_path).exists():
                sharded_state_dict = torch.load(shard_path, map_location="cpu")
                with self.fsdp.state_dict_type(self.model, self.sharded_state_dict_config):
                    self.model.load_state_dict(sharded_state_dict)
    
    def _load_megatron_checkpoint(self, path: Path):
        """Load checkpoint using Megatron."""
        self.megatron_load_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            load_arg="load",
            strict=True
        )
    
    def _load_pytorch_checkpoint(self, path: Path):
        """Load checkpoint using PyTorch."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, "module"):  # DDP wrapped
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.lr_scheduler and checkpoint.get("lr_scheduler_state_dict"):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        # Update training state
        self.current_step = checkpoint.get("step", self.current_step)
        self.current_epoch = checkpoint.get("epoch", self.current_epoch)
        self.current_batch_size = checkpoint.get("batch_size", self.current_batch_size)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
    
    def _check_for_failures(self):
        """Check for potential failures during training."""
        # Check memory usage
        memory_usage = self._get_memory_usage()
        if memory_usage > 0.95:  # 95% memory usage
            logger.warning(f"High memory usage detected: {memory_usage:.1%}")
            if self.config.enable_elastic_scaling:
                self._adjust_batch_size()
        
        # Check for NaN/Inf in model parameters
        self._check_model_health()
    
    def _check_model_health(self):
        """Check model parameters for NaN/Inf values."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.warning(f"NaN gradient detected in {name}")
                    self._handle_nan_gradient(name, param)
                if torch.isinf(param.grad).any():
                    logger.warning(f"Inf gradient detected in {name}")
                    self._handle_inf_gradient(name, param)
    
    def _handle_nan_gradient(self, param_name: str, param: torch.Tensor):
        """Handle NaN gradient in parameter."""
        # Zero out NaN gradients
        param.grad[torch.isnan(param.grad)] = 0.0
        
        # Log for debugging
        logger.debug(f"Zeroed NaN gradients in {param_name}")
    
    def _handle_inf_gradient(self, param_name: str, param: torch.Tensor):
        """Handle Inf gradient in parameter."""
        # Clip inf gradients
        param.grad[torch.isinf(param.grad)] = torch.sign(
            param.grad[torch.isinf(param.grad)]
        ) * 1e6
        
        # Log for debugging
        logger.debug(f"Clipped Inf gradients in {param_name}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self._shutdown_flag = True
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources."""
        self._monitoring_active = False
        
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
        
        if self._memory_monitor_thread and self._memory_monitor_thread.is_alive():
            self._memory_monitor_thread.join(timeout=5.0)
        
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
        
        # Final checkpoint save
        self._save_checkpoint("shutdown")
        
        logger.info("Resilience layer cleanup completed")
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "current_batch_size": self.current_batch_size,
            "best_metric": self.best_metric,
            "recovery_attempts": self._recovery_attempts,
            "failure_history": self._failure_history,
            "node_status": self._node_status,
            "memory_usage": self._get_memory_usage(),
            "hardware_info": self.hardware_info,
            "checkpoint_count": len(self._checkpoint_metadata.get("checkpoints", []))
        }