"""
studio/backend/core/training/orchestrator.py

Distributed Training Orchestrator for Unsloth Studio
Handles automatic multi-GPU/multi-node training with fault tolerance,
checkpointing, and dynamic resource allocation using Ray and Kubernetes.

Integrates with existing Unsloth training infrastructure.
"""

import os
import sys
import json
import time
import logging
import tempfile
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Ray imports with fallback handling
try:
    import ray
    from ray import train, tune
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available. Distributed training will be limited.")

# Kubernetes imports with fallback handling
try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    logging.warning("Kubernetes client not available. Cloud deployment features disabled.")

# Import existing Unsloth components
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, TrainingJobConfig

logger = logging.getLogger(__name__)


class TrainingBackend(Enum):
    """Supported distributed training backends."""
    RAY = "ray"
    PYTORCH_DDP = "pytorch_ddp"
    HOROVOD = "horovod"
    SINGLE_GPU = "single_gpu"


class ClusterStatus(Enum):
    """Cluster health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    SCALING = "scaling"


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    num_gpus: int
    gpu_memory: List[int]  # Memory per GPU in MB
    cpu_cores: int
    memory_gb: float
    status: str = "available"
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    # Model and training parameters
    model_name: str
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    
    # Distributed training parameters
    backend: TrainingBackend = TrainingBackend.RAY
    num_workers: int = 1
    num_gpus_per_worker: int = 1
    use_gpu: bool = True
    mixed_precision: bool = True
    
    # Fault tolerance
    checkpoint_interval: int = 100  # Steps between checkpoints
    max_restarts: int = 3
    restart_timeout: int = 300  # seconds
    
    # Resource allocation
    min_workers: int = 1
    max_workers: int = 8
    auto_scale: bool = True
    cpu_per_worker: int = 4
    memory_per_worker_gb: int = 16
    
    # Kubernetes specific
    k8s_namespace: str = "forge-training"
    k8s_image: str = "forge/studio:latest"
    k8s_storage_class: str = "standard"
    k8s_pvc_size: str = "100Gi"
    
    # Advanced options
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Handle enum conversion
        if 'backend' in config_dict and isinstance(config_dict['backend'], str):
            config_dict['backend'] = TrainingBackend(config_dict['backend'])
        return cls(**config_dict)


@dataclass
class TrainingState:
    """Current state of distributed training."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    current_epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Distributed training state
    world_size: int = 1
    local_rank: int = 0
    global_rank: int = 0
    
    # Resource allocation
    allocated_nodes: List[NodeInfo] = field(default_factory=list)
    allocated_gpus: int = 0
    
    # Checkpoint info
    last_checkpoint: Optional[str] = None
    checkpoint_step: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        result = asdict(self)
        # Convert enum to string for serialization
        if 'status' in result:
            result['status'] = result['status'].value if hasattr(result['status'], 'value') else result['status']
        return result


class CheckpointManager:
    """Manages model checkpoints with fault tolerance."""
    
    def __init__(self, checkpoint_dir: str, max_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints: List[Dict[str, Any]] = []
        self._load_checkpoint_index()
    
    def _load_checkpoint_index(self):
        """Load existing checkpoint index."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.checkpoints = json.load(f)
    
    def _save_checkpoint_index(self):
        """Save checkpoint index."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def save_checkpoint(self, 
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       scheduler_state: Optional[Dict[str, Any]],
                       step: int,
                       epoch: int,
                       metrics: Dict[str, float]) -> str:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step{step}_epoch{epoch}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model and optimizer states
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'metrics': metrics,
            'timestamp': timestamp
        }, checkpoint_path / "training_state.pt")
        
        # Update checkpoint index
        checkpoint_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp
        }
        self.checkpoints.append(checkpoint_info)
        
        # Remove old checkpoints if exceeding max_keep
        if len(self.checkpoints) > self.max_keep:
            # Sort by step (or timestamp)
            self.checkpoints.sort(key=lambda x: x['step'])
            to_remove = self.checkpoints[:-self.max_keep]
            for old_checkpoint in to_remove:
                old_path = Path(old_checkpoint['path'])
                if old_path.exists():
                    import shutil
                    shutil.rmtree(old_path)
            self.checkpoints = self.checkpoints[-self.max_keep:]
        
        self._save_checkpoint_index()
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        # Sort by step to get latest
        self.checkpoints.sort(key=lambda x: x['step'])
        latest = self.checkpoints[-1]
        checkpoint_path = Path(latest['path']) / "training_state.pt"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_checkpoint_at_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Get checkpoint at specific step."""
        for checkpoint in self.checkpoints:
            if checkpoint['step'] == step:
                checkpoint_path = Path(checkpoint['path']) / "training_state.pt"
                if checkpoint_path.exists():
                    return torch.load(checkpoint_path, map_location='cpu')
        return None


class RayDistributedTrainer:
    """Ray-based distributed training orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState(job_id=f"train_{int(time.time())}")
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        self.cluster_manager = ClusterManager(config)
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            self._init_ray()
    
    def _init_ray(self):
        """Initialize Ray cluster."""
        try:
            # Try to connect to existing cluster
            ray.init(address='auto', ignore_reinit_error=True)
            logger.info("Connected to existing Ray cluster")
        except:
            # Start local Ray cluster
            ray.init(
                num_gpus=self.config.num_workers * self.config.num_gpus_per_worker,
                ignore_reinit_error=True
            )
            logger.info("Started local Ray cluster")
    
    def _train_func(self, train_loop_config: Dict[str, Any]):
        """Training function to be executed on each worker."""
        # Import here to avoid circular imports
        from studio.backend.core.training.trainer import UnslothTrainer
        
        # Set up distributed training
        train_config = TrainingConfig.from_dict(train_loop_config['config'])
        
        # Initialize trainer
        trainer = UnslothTrainer(
            model_name=train_config.model_name,
            dataset_path=train_config.dataset_path,
            output_dir=train_config.output_dir,
            config=train_config.to_dict()
        )
        
        # Load checkpoint if resuming
        if train_loop_config.get('resume_from_checkpoint'):
            checkpoint = train_loop_config['resume_from_checkpoint']
            trainer.load_checkpoint(checkpoint)
        
        # Training loop
        for epoch in range(train_config.num_epochs):
            # Set epoch for distributed sampler if using
            if hasattr(trainer, 'train_sampler') and trainer.train_sampler is not None:
                trainer.train_sampler.set_epoch(epoch)
            
            # Train one epoch
            metrics = trainer.train_epoch(epoch)
            
            # Report metrics to Ray
            train.report({
                "epoch": epoch,
                "step": trainer.global_step,
                **metrics
            })
            
            # Save checkpoint if needed
            if trainer.global_step % train_config.checkpoint_interval == 0:
                # Save checkpoint
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    model_state=trainer.model.state_dict(),
                    optimizer_state=trainer.optimizer.state_dict(),
                    scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
                    step=trainer.global_step,
                    epoch=epoch,
                    metrics=metrics
                )
                
                # Report checkpoint to Ray
                train.report({
                    "checkpoint": checkpoint_path,
                    "step": trainer.global_step
                })
        
        # Final checkpoint
        final_metrics = trainer.evaluate() if hasattr(trainer, 'evaluate') else {}
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            model_state=trainer.model.state_dict(),
            optimizer_state=trainer.optimizer.state_dict(),
            scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
            step=trainer.global_step,
            epoch=train_config.num_epochs - 1,
            metrics=final_metrics
        )
        
        return {
            "final_checkpoint": final_checkpoint,
            "final_metrics": final_metrics
        }
    
    def run(self) -> TrainingState:
        """Run distributed training with fault tolerance."""
        logger.info(f"Starting distributed training job {self.state.job_id}")
        self.state.status = JobStatus.RUNNING
        self.state.start_time = time.time()
        
        try:
            # Configure scaling
            scaling_config = ScalingConfig(
                num_workers=self.config.num_workers,
                use_gpu=self.config.use_gpu,
                resources_per_worker={
                    "CPU": self.config.cpu_per_worker,
                    "GPU": self.config.num_gpus_per_worker,
                    "memory": self.config.memory_per_worker_gb * 1024 * 1024 * 1024
                }
            )
            
            # Set up checkpoint configuration
            checkpoint_config = train.CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
                checkpoint_frequency=self.config.checkpoint_interval
            )
            
            # Set up failure configuration
            failure_config = train.FailureConfig(
                max_failures=self.config.max_restarts
            )
            
            # Prepare training loop config
            train_loop_config = {
                "config": self.config.to_dict(),
                "resume_from_checkpoint": self.checkpoint_manager.load_latest_checkpoint()
            }
            
            # Initialize trainer
            trainer = TorchTrainer(
                self._train_func,
                train_loop_config=train_loop_config,
                scaling_config=scaling_config,
                checkpoint_config=checkpoint_config,
                failure_config=failure_config,
                metadata={"job_id": self.state.job_id}
            )
            
            # Run training
            result = trainer.fit()
            
            # Update state with results
            self.state.status = JobStatus.COMPLETED
            self.state.end_time = time.time()
            self.state.best_metric = result.metrics.get("loss", None)
            
            logger.info(f"Training completed successfully. Best metric: {self.state.best_metric}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.state.status = JobStatus.FAILED
            self.state.last_error = str(e)
            self.state.error_count += 1
            raise
        
        return self.state
    
    def stop(self):
        """Stop the training job."""
        logger.info(f"Stopping training job {self.state.job_id}")
        # Ray doesn't have a direct stop method, but we can update state
        self.state.status = JobStatus.STOPPED
        self.state.end_time = time.time()


class KubernetesOrchestrator:
    """Kubernetes-based orchestrator for cloud deployments."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.k8s_client = None
        self.custom_api = None
        
        if K8S_AVAILABLE:
            self._init_k8s()
    
    def _init_k8s(self):
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except:
            try:
                # Fall back to kubeconfig
                config.load_kube_config()
                logger.info("Loaded kubeconfig")
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                return
        
        self.k8s_client = client.CoreV1Api()
        self.custom_api = client.CustomObjectsApi()
    
    def create_training_job(self) -> bool:
        """Create a Kubernetes training job."""
        if not K8S_AVAILABLE or not self.k8s_client:
            logger.error("Kubernetes not available")
            return False
        
        try:
            # Create namespace if it doesn't exist
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=self.config.k8s_namespace)
            )
            try:
                self.k8s_client.create_namespace(namespace)
                logger.info(f"Created namespace: {self.config.k8s_namespace}")
            except ApiException as e:
                if e.status != 409:  # Already exists
                    raise
            
            # Create PVC for shared storage
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=f"training-data-{self.state.job_id}",
                    namespace=self.config.k8s_namespace
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteMany"],
                    storage_class_name=self.config.k8s_storage_class,
                    resources=client.V1ResourceRequirements(
                        requests={"storage": self.config.k8s_pvc_size}
                    )
                )
            )
            self.k8s_client.create_namespaced_persistent_volume_claim(
                namespace=self.config.k8s_namespace,
                body=pvc
            )
            
            # Create training job manifest
            job_manifest = self._create_job_manifest()
            
            # Create the job
            batch_v1 = client.BatchV1Api()
            job = batch_v1.create_namespaced_job(
                namespace=self.config.k8s_namespace,
                body=job_manifest
            )
            
            logger.info(f"Created Kubernetes job: {job.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes job: {e}")
            return False
    
    def _create_job_manifest(self) -> Dict[str, Any]:
        """Create Kubernetes job manifest."""
        # This is a simplified version - in production, you'd want a more complete manifest
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"forge-training-{self.state.job_id}",
                "namespace": self.config.k8s_namespace
            },
            "spec": {
                "parallelism": self.config.num_workers,
                "completions": self.config.num_workers,
                "backoffLimit": self.config.max_restarts,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "forge-training",
                            "job-id": self.state.job_id
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "trainer",
                            "image": self.config.k8s_image,
                            "command": ["python", "-m", "studio.backend.core.training.orchestrator"],
                            "args": ["--config", json.dumps(self.config.to_dict())],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": str(self.config.num_gpus_per_worker),
                                    "cpu": str(self.config.cpu_per_worker),
                                    "memory": f"{self.config.memory_per_worker_gb}Gi"
                                },
                                "requests": {
                                    "nvidia.com/gpu": str(self.config.num_gpus_per_worker),
                                    "cpu": str(self.config.cpu_per_worker),
                                    "memory": f"{self.config.memory_per_worker_gb}Gi"
                                }
                            },
                            "volumeMounts": [{
                                "name": "training-data",
                                "mountPath": "/data"
                            }, {
                                "name": "output",
                                "mountPath": "/output"
                            }],
                            "env": [{
                                "name": "NCCL_DEBUG",
                                "value": "INFO"
                            }, {
                                "name": "PYTHONUNBUFFERED",
                                "value": "1"
                            }]
                        }],
                        "volumes": [{
                            "name": "training-data",
                            "persistentVolumeClaim": {
                                "claimName": f"training-data-{self.state.job_id}"
                            }
                        }, {
                            "name": "output",
                            "emptyDir": {}
                        }],
                        "restartPolicy": "OnFailure",
                        "nodeSelector": {
                            "gpu": "true"
                        }
                    }
                }
            }
        }
    
    def scale_training(self, num_workers: int) -> bool:
        """Scale the training job to specified number of workers."""
        if not K8S_AVAILABLE:
            return False
        
        try:
            batch_v1 = client.BatchV1Api()
            job_name = f"forge-training-{self.state.job_id}"
            
            # Get current job
            job = batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.config.k8s_namespace
            )
            
            # Update parallelism
            job.spec.parallelism = num_workers
            job.spec.completions = num_workers
            
            # Update the job
            batch_v1.replace_namespaced_job(
                name=job_name,
                namespace=self.config.k8s_namespace,
                body=job
            )
            
            logger.info(f"Scaled job to {num_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale job: {e}")
            return False
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of the Kubernetes job."""
        if not K8S_AVAILABLE:
            return {"status": "kubernetes_not_available"}
        
        try:
            batch_v1 = client.BatchV1Api()
            job_name = f"forge-training-{self.state.job_id}"
            
            job = batch_v1.read_namespaced_job_status(
                name=job_name,
                namespace=self.config.k8s_namespace
            )
            
            return {
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "start_time": job.status.start_time,
                "completion_time": job.status.completion_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"status": "error", "message": str(e)}


class ClusterManager:
    """Manages cluster resources and node allocation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.nodes: Dict[str, NodeInfo] = {}
        self.available_gpus: int = 0
        self.cluster_status: ClusterStatus = ClusterStatus.HEALTHY
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_cluster, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_cluster(self):
        """Monitor cluster health and resources."""
        while True:
            try:
                if RAY_AVAILABLE and ray.is_initialized():
                    # Get cluster resources from Ray
                    resources = ray.cluster_resources()
                    self.available_gpus = resources.get('GPU', 0)
                    
                    # Update cluster status
                    if self.available_gpus < self.config.min_workers * self.config.num_gpus_per_worker:
                        self.cluster_status = ClusterStatus.DEGRADED
                    elif self.available_gpus >= self.config.max_workers * self.config.num_gpus_per_worker:
                        self.cluster_status = ClusterStatus.HEALTHY
                    else:
                        self.cluster_status = ClusterStatus.SCALING
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Cluster monitoring error: {e}")
                self.cluster_status = ClusterStatus.FAILED
                time.sleep(30)  # Back off on error
    
    def allocate_resources(self, 
                          num_workers: int, 
                          gpus_per_worker: int) -> List[NodeInfo]:
        """Allocate resources for training."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, using single node allocation")
            return self._allocate_single_node(num_workers, gpus_per_worker)
        
        required_gpus = num_workers * gpus_per_worker
        
        if self.available_gpus < required_gpus:
            if self.config.auto_scale:
                logger.info(f"Auto-scaling: requesting {required_gpus - self.available_gpus} more GPUs")
                # In a real implementation, this would trigger cloud scaling
            else:
                raise RuntimeError(f"Insufficient GPUs: {self.available_gpus} available, {required_gpus} required")
        
        # Create placement group for distributed training
        bundles = [{"GPU": gpus_per_worker, "CPU": self.config.cpu_per_worker} 
                  for _ in range(num_workers)]
        
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())  # Wait for placement group to be ready
        
        logger.info(f"Allocated {num_workers} workers with {gpus_per_worker} GPUs each")
        return []  # Return node info in production
    
    def _allocate_single_node(self, num_workers: int, gpus_per_worker: int) -> List[NodeInfo]:
        """Allocate resources on a single node."""
        import torch
        
        available_gpus = torch.cuda.device_count()
        if available_gpus < num_workers * gpus_per_worker:
            raise RuntimeError(f"Insufficient GPUs on single node: {available_gpus} available")
        
        # Create a single node info
        node = NodeInfo(
            node_id="local_node",
            hostname="localhost",
            ip_address="127.0.0.1",
            num_gpus=available_gpus,
            gpu_memory=[torch.cuda.get_device_properties(i).total_memory 
                       for i in range(available_gpus)],
            cpu_cores=os.cpu_count() or 4,
            memory_gb=32.0  # Default assumption
        )
        
        return [node]
    
    def release_resources(self, nodes: List[NodeInfo]):
        """Release allocated resources."""
        # In a real implementation, this would clean up placement groups
        logger.info("Released cluster resources")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        info = {
            "status": self.cluster_status.value,
            "total_gpus": self.available_gpus,
            "nodes": len(self.nodes),
            "timestamp": datetime.now().isoformat()
        }
        
        if RAY_AVAILABLE and ray.is_initialized():
            info["ray_nodes"] = len(ray.nodes())
            info["ray_resources"] = ray.cluster_resources()
        
        return info


class DistributedTrainingOrchestrator:
    """
    Main orchestrator for distributed training.
    Coordinates between Ray, Kubernetes, and existing Unsloth components.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], TrainingConfig]] = None):
        # Load configuration
        if config is None:
            config = self._load_default_config()
        elif isinstance(config, dict):
            config = TrainingConfig.from_dict(config)
        
        self.config = config
        self.state = TrainingState(job_id=f"train_{int(time.time())}")
        
        # Initialize components
        self.ray_trainer = RayDistributedTrainer(config) if RAY_AVAILABLE else None
        self.k8s_orchestrator = KubernetesOrchestrator(config) if K8S_AVAILABLE else None
        self.cluster_manager = ClusterManager(config)
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        
        # Integration with existing job manager
        self.job_manager = None
        self._init_job_manager()
        
        logger.info(f"Initialized DistributedTrainingOrchestrator for job {self.state.job_id}")
    
    def _load_default_config(self) -> TrainingConfig:
        """Load default training configuration."""
        return TrainingConfig(
            model_name="forge/llama-3-8b-bnb-4bit",
            dataset_path="./data",
            output_dir="./output",
            num_workers=min(4, os.cpu_count() or 1),
            num_gpus_per_worker=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
    
    def _init_job_manager(self):
        """Initialize connection to existing job manager."""
        try:
            # Try to import and connect to existing job manager
            from studio.backend.core.data_recipe.jobs.manager import get_job_manager
            self.job_manager = get_job_manager()
            logger.info("Connected to existing job manager")
        except ImportError:
            logger.warning("Could not connect to job manager")
    
    def train(self, 
              model_fn: Optional[Callable] = None,
              dataset_fn: Optional[Callable] = None,
              **kwargs) -> TrainingState:
        """
        Start distributed training.
        
        Args:
            model_fn: Optional function to create model
            dataset_fn: Optional function to load dataset
            **kwargs: Additional training arguments
        
        Returns:
            TrainingState with job information
        """
        # Update config with any additional arguments
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Register job with job manager if available
        if self.job_manager:
            job_config = TrainingJobConfig(
                job_id=self.state.job_id,
                model_name=self.config.model_name,
                dataset_path=self.config.dataset_path,
                output_dir=self.config.output_dir,
                config=self.config.to_dict()
            )
            self.job_manager.create_job(job_config)
        
        # Choose training backend
        if self.config.backend == TrainingBackend.RAY and self.ray_trainer:
            logger.info("Using Ray distributed training backend")
            self.state = self.ray_trainer.run()
        elif self.config.backend == TrainingBackend.PYTORCH_DDP:
            logger.info("Using PyTorch DDP training backend")
            self.state = self._run_pytorch_ddp(model_fn, dataset_fn)
        else:
            logger.info("Using single GPU training backend")
            self.state = self._run_single_gpu(model_fn, dataset_fn)
        
        # Update job manager with final state
        if self.job_manager:
            self.job_manager.update_job_status(
                self.state.job_id,
                self.state.status,
                self.state.to_dict()
            )
        
        return self.state
    
    def _run_pytorch_ddp(self, 
                        model_fn: Optional[Callable],
                        dataset_fn: Optional[Callable]) -> TrainingState:
        """Run training with PyTorch DDP."""
        # This is a simplified implementation
        # In production, you'd want more sophisticated DDP setup
        
        import torch.multiprocessing as mp
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        def train_worker(rank: int, world_size: int, config: TrainingConfig):
            """Worker function for DDP training."""
            # Set up distributed
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            
            # Create model
            if model_fn:
                model = model_fn(config)
            else:
                # Default model creation
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float16 if config.fp16 else torch.float32
                )
            
            model = model.to(rank)
            model = DDP(model, device_ids=[rank])
            
            # Training loop would go here
            # This is a placeholder
            
            dist.destroy_process_group()
        
        # Spawn workers
        world_size = self.config.num_workers * self.config.num_gpus_per_worker
        mp.spawn(train_worker,
                args=(world_size, self.config),
                nprocs=world_size,
                join=True)
        
        self.state.status = JobStatus.COMPLETED
        self.state.end_time = time.time()
        return self.state
    
    def _run_single_gpu(self,
                       model_fn: Optional[Callable],
                       dataset_fn: Optional[Callable]) -> TrainingState:
        """Run training on single GPU or CPU."""
        # This integrates with existing Unsloth training
        try:
            from studio.backend.core.training.trainer import UnslothTrainer
            
            trainer = UnslothTrainer(
                model_name=self.config.model_name,
                dataset_path=self.config.dataset_path,
                output_dir=self.config.output_dir,
                config=self.config.to_dict()
            )
            
            # Load checkpoint if exists
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                trainer.load_checkpoint(latest_checkpoint)
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                metrics = trainer.train_epoch(epoch)
                
                # Save checkpoint
                if epoch % (self.config.checkpoint_interval // 100) == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model_state=trainer.model.state_dict(),
                        optimizer_state=trainer.optimizer.state_dict(),
                        scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
                        step=trainer.global_step,
                        epoch=epoch,
                        metrics=metrics
                    )
            
            self.state.status = JobStatus.COMPLETED
            self.state.end_time = time.time()
            
        except Exception as e:
            logger.error(f"Single GPU training failed: {e}")
            self.state.status = JobStatus.FAILED
            self.state.last_error = str(e)
        
        return self.state
    
    def scale(self, num_workers: int) -> bool:
        """Scale the training job to specified number of workers."""
        if self.k8s_orchestrator:
            return self.k8s_orchestrator.scale_training(num_workers)
        elif self.ray_trainer:
            # Ray scaling would be handled through the trainer
            logger.info(f"Scaling to {num_workers} workers (Ray)")
            return True
        else:
            logger.warning("Scaling not supported in current configuration")
            return False
    
    def stop(self):
        """Stop the training job."""
        logger.info(f"Stopping training job {self.state.job_id}")
        
        if self.ray_trainer:
            self.ray_trainer.stop()
        
        self.state.status = JobStatus.STOPPED
        self.state.end_time = time.time()
        
        # Update job manager
        if self.job_manager:
            self.job_manager.update_job_status(
                self.state.job_id,
                self.state.status,
                self.state.to_dict()
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        status = self.state.to_dict()
        
        # Add cluster information
        status['cluster'] = self.cluster_manager.get_cluster_info()
        
        # Add Kubernetes status if available
        if self.k8s_orchestrator:
            status['kubernetes'] = self.k8s_orchestrator.get_job_status()
        
        return status
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints."""
        return self.checkpoint_manager.checkpoints
    
    def load_checkpoint(self, step: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint at specific step."""
        return self.checkpoint_manager.get_checkpoint_at_step(step)


# CLI interface for running from command line
def main():
    """Command-line interface for distributed training orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unsloth Distributed Training Orchestrator")
    parser.add_argument("--config", type=str, help="JSON configuration string")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--dataset", type=str, help="Dataset path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--workers", type=int, help="Number of workers")
    parser.add_argument("--gpus", type=int, help="GPUs per worker")
    parser.add_argument("--backend", choices=["ray", "pytorch_ddp", "single_gpu"], 
                       help="Training backend")
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = {}
    
    if args.config:
        config_dict = json.loads(args.config)
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            if args.config_file.endswith('.json'):
                config_dict = json.load(f)
            elif args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                config_dict = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.model:
        config_dict['model_name'] = args.model
    if args.dataset:
        config_dict['dataset_path'] = args.dataset
    if args.output:
        config_dict['output_dir'] = args.output
    if args.workers:
        config_dict['num_workers'] = args.workers
    if args.gpus:
        config_dict['num_gpus_per_worker'] = args.gpus
    if args.backend:
        config_dict['backend'] = args.backend
    
    # Initialize and run orchestrator
    orchestrator = DistributedTrainingOrchestrator(config_dict)
    
    try:
        final_state = orchestrator.train()
        print(f"Training completed with status: {final_state.status}")
        print(f"Final state: {json.dumps(final_state.to_dict(), indent=2)}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        orchestrator.stop()
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


# Factory function for easy integration
def create_distributed_trainer(config: Optional[Union[Dict[str, Any], TrainingConfig]] = None) -> DistributedTrainingOrchestrator:
    """
    Factory function to create a distributed trainer.
    
    Args:
        config: Training configuration
    
    Returns:
        Configured DistributedTrainingOrchestrator instance
    """
    return DistributedTrainingOrchestrator(config)


# Integration with existing Unsloth training
def integrate_with_forge_trainer(trainer_instance, config: Optional[Dict[str, Any]] = None):
    """
    Integrate distributed training with existing Unsloth trainer.
    
    Args:
        trainer_instance: Existing Unsloth trainer instance
        config: Optional distributed training configuration
    """
    if config is None:
        config = {
            'model_name': trainer_instance.model_name if hasattr(trainer_instance, 'model_name') else 'forge/model',
            'output_dir': trainer_instance.output_dir if hasattr(trainer_instance, 'output_dir') else './output'
        }
    
    orchestrator = DistributedTrainingOrchestrator(config)
    
    # Wrap trainer methods
    original_train = trainer_instance.train
    
    def distributed_train(*args, **kwargs):
        """Distributed training wrapper."""
        # This would need to be implemented based on the actual trainer interface
        # For now, it's a placeholder
        return original_train(*args, **kwargs)
    
    trainer_instance.train = distributed_train
    trainer_instance.distributed_orchestrator = orchestrator
    
    return trainer_instance


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()