"""Distributed Training Orchestration with Ray and Kubernetes

This module provides automatic multi-GPU/multi-node training orchestration
with fault tolerance, checkpointing, and dynamic resource allocation.
"""

import os
import time
import json
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import signal

import ray
from ray import train, tune
from ray.train import Checkpoint, ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    # Training parameters
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    
    # Distributed training parameters
    num_workers: int = 1
    use_gpu: bool = True
    resources_per_worker: Dict[str, float] = None
    max_retries: int = 3
    checkpoint_frequency: int = 100  # steps
    keep_checkpoints_max: int = 5
    
    # Kubernetes parameters
    use_kubernetes: bool = False
    namespace: str = "forge"
    cpu_per_worker: str = "4"
    memory_per_worker: str = "16Gi"
    gpu_per_worker: int = 1
    
    # Fault tolerance
    enable_auto_recovery: bool = True
    recovery_timeout: int = 300  # seconds
    
    def __post_init__(self):
        if self.resources_per_worker is None:
            self.resources_per_worker = {"CPU": 2, "GPU": 1} if self.use_gpu else {"CPU": 2}


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints."""
    epoch: int
    global_step: int
    loss: float
    timestamp: str
    worker_rank: int
    is_best: bool = False
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class CheckpointManager:
    """Manages checkpointing in distributed training environment."""
    
    def __init__(self, checkpoint_dir: str, config: TrainingConfig):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.best_loss = float('inf')
        self.checkpoint_queue = queue.Queue()
        self._setup_checkpoint_dir()
        
    def _setup_checkpoint_dir(self):
        """Create checkpoint directory structure."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "best").mkdir(exist_ok=True)
        (self.checkpoint_dir / "latest").mkdir(exist_ok=True)
        (self.checkpoint_dir / "epoch").mkdir(exist_ok=True)
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        loss: float,
        metrics: Dict[str, Any] = None,
        is_best: bool = False
    ) -> str:
        """Save a checkpoint with metadata."""
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint_epoch{epoch}_step{global_step}_{timestamp.replace(':', '-')}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        model_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': timestamp,
            'config': asdict(self.config)
        }
        
        torch.save(model_state, checkpoint_path / "model_state.pt")
        
        # Save metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            loss=loss,
            timestamp=timestamp,
            worker_rank=ray.train.get_context().get_world_rank() if ray.is_initialized() else 0,
            is_best=is_best,
            metrics=metrics or {}
        )
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update symlinks
        self._update_symlinks(checkpoint_path, is_best)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Notify listeners
        self.checkpoint_queue.put(str(checkpoint_path))
        
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)
    
    def _update_symlinks(self, checkpoint_path: Path, is_best: bool):
        """Update latest and best checkpoint symlinks."""
        latest_link = self.checkpoint_dir / "latest"
        best_link = self.checkpoint_dir / "best"
        
        # Remove old symlinks
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        
        # Create new symlink for latest
        latest_link.symlink_to(checkpoint_path)
        
        # Update best if this is the best checkpoint
        if is_best:
            if best_link.exists():
                if best_link.is_symlink():
                    best_link.unlink()
                else:
                    shutil.rmtree(best_link)
            best_link.symlink_to(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_dirs = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_epoch"):
                checkpoint_dirs.append(item)
        
        # Sort by modification time
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_dirs[self.config.keep_checkpoints_max:]:
            if old_checkpoint.is_symlink():
                continue
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest"
        if not latest_link.exists():
            return None
        
        checkpoint_path = latest_link
        if checkpoint_path.is_symlink():
            checkpoint_path = checkpoint_path.resolve()
        
        model_file = checkpoint_path / "model_state.pt"
        if not model_file.exists():
            return None
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        return torch.load(model_file, map_location="cpu")
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        model_file = checkpoint_path / "model_state.pt"
        
        if not model_file.exists():
            return None
        
        return torch.load(model_file, map_location="cpu")
    
    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        """Get metadata for all checkpoints."""
        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_epoch"):
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
        
        return sorted(checkpoints, key=lambda x: x["global_step"])


class DistributedTrainingOrchestrator:
    """Orchestrates distributed training with Ray and Kubernetes support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_manager = None
        self.trainer = None
        self._ray_initialized = False
        self._kubernetes_initialized = False
        
    def initialize_ray(self):
        """Initialize Ray with appropriate resources."""
        if self._ray_initialized:
            return
        
        if self.config.use_kubernetes:
            self._initialize_kubernetes_ray()
        else:
            self._initialize_local_ray()
        
        self._ray_initialized = True
        logger.info("Ray initialized successfully")
    
    def _initialize_local_ray(self):
        """Initialize Ray for local multi-GPU training."""
        ray.init(
            ignore_reinit_error=True,
            num_gpus=torch.cuda.device_count() if self.config.use_gpu else 0,
            logging_level=logging.INFO
        )
    
    def _initialize_kubernetes_ray(self):
        """Initialize Ray on Kubernetes cluster."""
        try:
            from ray.util.cluster_dump import ClusterDump
            from kubernetes import client, config
            
            # Load kube config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            v1 = client.CoreV1Api()
            
            # Check if Ray cluster exists
            namespace = self.config.namespace
            pods = v1.list_namespaced_pod(
                namespace=namespace,
                label_selector="app=ray-head"
            )
            
            if not pods.items:
                self._create_kubernetes_ray_cluster()
            
            # Connect to Ray cluster
            ray.init(address="auto", ignore_reinit_error=True)
            self._kubernetes_initialized = True
            
        except ImportError:
            logger.warning("Kubernetes packages not installed. Falling back to local Ray.")
            self._initialize_local_ray()
    
    def _create_kubernetes_ray_cluster(self):
        """Create a Ray cluster on Kubernetes."""
        from kubernetes import client
        
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
        
        namespace = self.config.namespace
        
        # Create namespace if it doesn't exist
        try:
            v1.create_namespace(
                client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
            )
        except client.exceptions.ApiException as e:
            if e.status != 409:  # 409 = Already Exists
                raise
        
        # Create Ray head service
        head_service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name="ray-head",
                namespace=namespace
            ),
            spec=client.V1ServiceSpec(
                selector={"app": "ray-head"},
                ports=[
                    client.V1ServicePort(port=6379, name="redis"),
                    client.V1ServicePort(port=8265, name="dashboard"),
                    client.V1ServicePort(port=10001, name="client")
                ],
                type="ClusterIP"
            )
        )
        
        try:
            v1.create_namespaced_service(namespace=namespace, body=head_service)
        except client.exceptions.ApiException as e:
            if e.status != 409:
                raise
        
        # Create Ray head deployment
        head_deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name="ray-head",
                namespace=namespace
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": "ray-head"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "ray-head"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="ray-head",
                                image="rayproject/ray:latest",
                                command=["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0"],
                                ports=[
                                    client.V1ContainerPort(container_port=6379),
                                    client.V1ContainerPort(container_port=8265),
                                    client.V1ContainerPort(container_port=10001)
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "cpu": self.config.cpu_per_worker,
                                        "memory": self.config.memory_per_worker
                                    },
                                    limits={
                                        "cpu": self.config.cpu_per_worker,
                                        "memory": self.config.memory_per_worker
                                    }
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        try:
            apps_v1.create_namespaced_deployment(namespace=namespace, body=head_deployment)
        except client.exceptions.ApiException as e:
            if e.status != 409:
                raise
        
        # Wait for head node to be ready
        self._wait_for_ray_head(namespace)
    
    def _wait_for_ray_head(self, namespace: str, timeout: int = 300):
        """Wait for Ray head node to be ready."""
        from kubernetes import client, watch
        
        v1 = client.CoreV1Api()
        w = watch.Watch()
        
        start_time = time.time()
        for event in w.stream(v1.list_namespaced_pod, namespace=namespace, timeout_seconds=timeout):
            pod = event['object']
            if pod.metadata.labels.get('app') == 'ray-head':
                if pod.status.phase == 'Running':
                    # Check if all containers are ready
                    all_ready = all(
                        container.ready 
                        for container in pod.status.container_statuses or []
                    )
                    if all_ready:
                        logger.info("Ray head node is ready")
                        return
            
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for Ray head node")
    
    def create_trainer(
        self,
        train_loop_per_worker: Callable,
        train_loop_config: Dict[str, Any] = None,
        datasets: Dict[str, Any] = None
    ) -> TorchTrainer:
        """Create a Ray TorchTrainer with the given configuration."""
        if not self._ray_initialized:
            self.initialize_ray()
        
        if train_loop_config is None:
            train_loop_config = {}
        
        # Add checkpoint manager to config
        train_loop_config['checkpoint_manager'] = self.checkpoint_manager
        train_loop_config['training_config'] = asdict(self.config)
        
        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker
        )
        
        # Configure checkpointing
        checkpoint_config = train.CheckpointConfig(
            num_to_keep=self.config.keep_checkpoints_max,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
            checkpoint_frequency=self.config.checkpoint_frequency
        )
        
        # Configure run
        run_config = RunConfig(
            name=f"forge_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            checkpoint_config=checkpoint_config,
            failure_config=FailureConfig(
                max_failures=self.config.max_retries
            )
        )
        
        # Create trainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets or {}
        )
        
        return self.trainer
    
    def train(
        self,
        train_loop_per_worker: Callable,
        train_loop_config: Dict[str, Any] = None,
        datasets: Dict[str, Any] = None,
        resume_from_checkpoint: str = None
    ) -> ray.train.Result:
        """Run distributed training."""
        if self.trainer is None:
            self.create_trainer(train_loop_per_worker, train_loop_config, datasets)
        
        # Handle checkpoint resumption
        resume_config = None
        if resume_from_checkpoint:
            checkpoint = Checkpoint.from_directory(resume_from_checkpoint)
            resume_config = ray.train.CheckpointConfig(
                checkpoint=checkpoint
            )
        
        # Run training
        try:
            result = self.trainer.fit()
            logger.info(f"Training completed. Results: {result}")
            return result
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.config.enable_auto_recovery:
                return self._handle_training_failure(e, train_loop_per_worker, train_loop_config, datasets)
            raise
    
    def _handle_training_failure(
        self,
        error: Exception,
        train_loop_per_worker: Callable,
        train_loop_config: Dict[str, Any],
        datasets: Dict[str, Any]
    ) -> ray.train.Result:
        """Handle training failure with auto-recovery."""
        logger.info("Attempting auto-recovery from training failure")
        
        # Wait for resources to be available
        time.sleep(10)
        
        # Try to resume from latest checkpoint
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_path = self.checkpoint_manager.checkpoint_dir / "latest"
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                return self.train(
                    train_loop_per_worker,
                    train_loop_config,
                    datasets,
                    resume_from_checkpoint=str(checkpoint_path)
                )
        
        # If no checkpoint, restart training
        logger.info("No checkpoint found, restarting training from scratch")
        return self.train(train_loop_per_worker, train_loop_config, datasets)
    
    def setup_checkpoint_manager(self, checkpoint_dir: str):
        """Setup checkpoint manager for the orchestrator."""
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.config)
        return self.checkpoint_manager
    
    def shutdown(self):
        """Shutdown Ray and cleanup resources."""
        if self._ray_initialized:
            ray.shutdown()
            self._ray_initialized = False
            logger.info("Ray shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize_ray()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class KubernetesOperator:
    """Kubernetes operator for managing distributed training jobs."""
    
    def __init__(self, namespace: str = "forge"):
        self.namespace = namespace
        self.k8s_client = None
        self._initialize_kubernetes()
    
    def _initialize_kubernetes(self):
        """Initialize Kubernetes client."""
        try:
            from kubernetes import client, config
            
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            self.k8s_client = client
            logger.info("Kubernetes client initialized")
            
        except ImportError:
            logger.error("Kubernetes packages not installed")
            raise
    
    def create_training_job(
        self,
        job_name: str,
        config: TrainingConfig,
        training_script: str,
        image: str = "forge/training:latest"
    ):
        """Create a Kubernetes job for distributed training."""
        batch_v1 = self.k8s_client.BatchV1Api()
        
        # Create job specification
        job = self.k8s_client.V1Job(
            metadata=self.k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace
            ),
            spec=self.k8s_client.V1JobSpec(
                template=self.k8s_client.V1PodTemplateSpec(
                    metadata=self.k8s_client.V1ObjectMeta(
                        labels={"app": "forge-training", "job": job_name}
                    ),
                    spec=self.k8s_client.V1PodSpec(
                        containers=[
                            self.k8s_client.V1Container(
                                name="training",
                                image=image,
                                command=["python", "-c", training_script],
                                env=[
                                    self.k8s_client.V1EnvVar(
                                        name="RAY_HEAD_HOST",
                                        value="ray-head"
                                    ),
                                    self.k8s_client.V1EnvVar(
                                        name="TRAINING_CONFIG",
                                        value=json.dumps(asdict(config))
                                    )
                                ],
                                resources=self.k8s_client.V1ResourceRequirements(
                                    requests={
                                        "cpu": config.cpu_per_worker,
                                        "memory": config.memory_per_worker,
                                        "nvidia.com/gpu": str(config.gpu_per_worker)
                                    },
                                    limits={
                                        "cpu": config.cpu_per_worker,
                                        "memory": config.memory_per_worker,
                                        "nvidia.com/gpu": str(config.gpu_per_worker)
                                    }
                                )
                            )
                        ],
                        restart_policy="Never"
                    )
                ),
                backoff_limit=config.max_retries
            )
        )
        
        try:
            api_response = batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            logger.info(f"Created training job: {job_name}")
            return api_response
        except self.k8s_client.exceptions.ApiException as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a training job."""
        batch_v1 = self.k8s_client.BatchV1Api()
        
        try:
            job = batch_v1.read_namespaced_job_status(
                name=job_name,
                namespace=self.namespace
            )
            
            status = {
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "start_time": job.status.start_time,
                "completion_time": job.status.completion_time
            }
            
            return status
        except self.k8s_client.exceptions.ApiException as e:
            logger.error(f"Failed to get job status: {e}")
            return {}
    
    def delete_job(self, job_name: str):
        """Delete a training job."""
        batch_v1 = self.k8s_client.BatchV1Api()
        
        try:
            batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=self.k8s_client.V1DeleteOptions(
                    propagation_policy="Foreground"
                )
            )
            logger.info(f"Deleted job: {job_name}")
        except self.k8s_client.exceptions.ApiException as e:
            logger.error(f"Failed to delete job: {e}")
            raise


def create_distributed_trainer(
    config: TrainingConfig,
    checkpoint_dir: str = "./checkpoints"
) -> Tuple[DistributedTrainingOrchestrator, CheckpointManager]:
    """Factory function to create a distributed trainer with checkpoint manager."""
    orchestrator = DistributedTrainingOrchestrator(config)
    checkpoint_manager = orchestrator.setup_checkpoint_manager(checkpoint_dir)
    
    return orchestrator, checkpoint_manager


def train_with_ray(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Any,
    config: TrainingConfig,
    checkpoint_dir: str = "./checkpoints",
    train_step_fn: Callable = None
):
    """High-level function for distributed training with Ray."""
    
    def train_loop_per_worker(config: Dict[str, Any]):
        """Training loop to be executed on each worker."""
        # Get distributed training context
        train_context = ray.train.get_context()
        world_rank = train_context.get_world_rank()
        world_size = train_context.get_world_size()
        
        # Setup device
        device = torch.device(f"cuda:{train_context.get_local_rank()}" 
                            if torch.cuda.is_available() and config['training_config']['use_gpu']
                            else "cpu")
        
        # Load checkpoint if exists
        checkpoint_manager = config.get('checkpoint_manager')
        start_epoch = 0
        global_step = 0
        
        if checkpoint_manager:
            latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                model.load_state_dict(latest_checkpoint['model_state_dict'])
                optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
                if scheduler and latest_checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(latest_checkpoint['scheduler_state_dict'])
                start_epoch = latest_checkpoint['epoch']
                global_step = latest_checkpoint['global_step']
                logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
        
        # Move model to device
        model.to(device)
        
        # Distributed training
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == 'cuda' else None
            )
        
        # Training loop
        training_config = TrainingConfig(**config['training_config'])
        
        for epoch in range(start_epoch, training_config.epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Custom training step or default
                if train_step_fn:
                    loss = train_step_fn(model, batch, optimizer, scheduler, config)
                else:
                    # Default training step
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Gradient accumulation
                    loss = loss / training_config.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                
                epoch_loss += loss.item()
                
                # Checkpointing
                if (global_step % training_config.checkpoint_frequency == 0 and 
                    checkpoint_manager and 
                    world_rank == 0):  # Only save on rank 0
                    
                    avg_loss = epoch_loss / (batch_idx + 1)
                    is_best = avg_loss < checkpoint_manager.best_loss
                    
                    if is_best:
                        checkpoint_manager.best_loss = avg_loss
                    
                    checkpoint_manager.save_checkpoint(
                        model=model.module if hasattr(model, 'module') else model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        loss=avg_loss,
                        metrics={"batch_idx": batch_idx},
                        is_best=is_best
                    )
                
                # Report metrics to Ray Train
                train.report({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            # End of epoch checkpoint
            if checkpoint_manager and world_rank == 0:
                avg_loss = epoch_loss / len(train_dataloader)
                checkpoint_manager.save_checkpoint(
                    model=model.module if hasattr(model, 'module') else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    loss=avg_loss,
                    metrics={"epoch_completed": True}
                )
    
    # Create orchestrator and run training
    with create_distributed_trainer(config, checkpoint_dir) as (orchestrator, checkpoint_manager):
        trainer = orchestrator.create_trainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                'training_config': asdict(config),
                'checkpoint_manager': checkpoint_manager
            }
        )
        
        result = orchestrator.train()
        return result


# Signal handling for graceful shutdown
def setup_signal_handlers(orchestrator: DistributedTrainingOrchestrator):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        orchestrator.shutdown()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = TrainingConfig(
        epochs=3,
        batch_size=32,
        learning_rate=5e-5,
        num_workers=4,
        use_gpu=True,
        checkpoint_frequency=100,
        use_kubernetes=False
    )
    
    # Example model and dataloader
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Create dummy dataloader
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    
    # Run distributed training
    result = train_with_ray(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        checkpoint_dir="./checkpoints"
    )
    
    print(f"Training completed: {result}")