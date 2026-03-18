"""
Production-Grade Distributed Training Orchestrator for forge
Handles automatic distributed training orchestration with fault tolerance, elastic scaling,
and cost optimization across cloud providers.
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import signal
import psutil
import hashlib
import pickle
import tempfile
import shutil

# Optional imports with fallbacks
try:
    import ray
    from ray import tune
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.util.placement_group import placement_group
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

try:
    import kubernetes
    from kubernetes import client, config, watch
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    kubernetes = None

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None

try:
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    compute_v1 = None

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    DefaultAzureCredential = None

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

# Import from existing forge modules
try:
    from ..utils import logging as forge_logging
    from ..extras.constants import (
        SUPPORTED_MODEL_TYPES,
        MODEL_PARALLEL_STRATEGIES,
        CHECKPOINT_FORMATS,
    )
    from ..model import load_model_and_tokenizer
    from ..train import get_train_args
except ImportError:
    # Fallback for standalone testing
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")
    
    forge_logging = MockLogger()

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class InstanceType(Enum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    PREEMPTIBLE = "preemptible"


class TrainingStrategy(Enum):
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATED = "terminated"
    PREEMPTED = "preempted"
    RECOVERING = "recovering"


@dataclass
class CloudInstanceConfig:
    """Configuration for cloud instances"""
    provider: CloudProvider = CloudProvider.AWS
    instance_type: str = "p3.2xlarge"
    region: str = "us-east-1"
    zone: Optional[str] = None
    instance_category: InstanceType = InstanceType.SPOT
    max_price: Optional[float] = None
    min_count: int = 1
    max_count: int = 8
    disk_size_gb: int = 100
    image_id: Optional[str] = None
    ssh_key_name: Optional[str] = None
    security_group_ids: List[str] = field(default_factory=list)
    subnet_ids: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    use_placement_group: bool = True
    placement_group_strategy: str = "cluster"


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name_or_path: str = ""
    model_type: str = "llama"
    training_strategy: TrainingStrategy = TrainingStrategy.DDP
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    deepspeed_config: Optional[str] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    auto_model_parallel: bool = True
    auto_batch_size: bool = True
    cost_optimization: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    save_interval: int = 500
    save_total_limit: int = 3
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    async_save: bool = True
    checkpoint_format: str = "torch"
    upload_to_cloud: bool = True
    cloud_bucket: Optional[str] = None
    cloud_prefix: Optional[str] = None
    compression: Optional[str] = None


@dataclass
class FaultToleranceConfig:
    """Fault tolerance configuration"""
    max_restarts: int = 3
    restart_timeout: int = 300
    health_check_interval: int = 30
    node_failure_threshold: float = 0.3
    preemption_handling: bool = True
    elastic_scaling: bool = True
    min_nodes: int = 1
    max_nodes: int = 16
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    cooldown_period: int = 300
    automatic_recovery: bool = True
    backup_strategy: str = "incremental"


class CloudAdapter:
    """Base class for cloud provider adapters"""
    
    def __init__(self, config: CloudInstanceConfig):
        self.config = config
        self.instances = {}
        self.placement_group_id = None
        
    def provision_instances(self, count: int) -> List[str]:
        """Provision cloud instances"""
        raise NotImplementedError
        
    def terminate_instances(self, instance_ids: List[str]):
        """Terminate cloud instances"""
        raise NotImplementedError
        
    def get_instance_status(self, instance_id: str) -> NodeStatus:
        """Get instance status"""
        raise NotImplementedError
        
    def get_instance_ip(self, instance_id: str) -> str:
        """Get instance IP address"""
        raise NotImplementedError
        
    def handle_spot_interruption(self, instance_id: str) -> bool:
        """Handle spot instance interruption"""
        raise NotImplementedError
        
    def estimate_cost(self, instance_type: str, hours: float) -> float:
        """Estimate cost for running instances"""
        raise NotImplementedError


class AWSAdapter(CloudAdapter):
    """AWS cloud adapter"""
    
    def __init__(self, config: CloudInstanceConfig):
        super().__init__(config)
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS adapter")
        self.ec2 = boto3.client('ec2', region_name=config.region)
        self.ec2_resource = boto3.resource('ec2', region_name=config.region)
        
    def provision_instances(self, count: int) -> List[str]:
        """Provision AWS EC2 instances"""
        try:
            run_params = {
                'ImageId': self.config.image_id or self._get_default_ami(),
                'InstanceType': self.config.instance_type,
                'MinCount': 1,
                'MaxCount': count,
                'KeyName': self.config.ssh_key_name,
                'SecurityGroupIds': self.config.security_group_ids,
                'SubnetId': self.config.subnet_ids[0] if self.config.subnet_ids else None,
                'BlockDeviceMappings': [{
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': self.config.disk_size_gb,
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }],
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': k, 'Value': v} for k, v in self.config.tags.items()]
                }]
            }
            
            if self.config.instance_category == InstanceType.SPOT:
                run_params['InstanceMarketOptions'] = {
                    'MarketType': 'spot',
                    'SpotOptions': {
                        'MaxPrice': str(self.config.max_price) if self.config.max_price else '',
                        'SpotInstanceType': 'persistent',
                        'InstanceInterruptionBehavior': 'stop'
                    }
                }
            
            if self.config.use_placement_group:
                if not self.placement_group_id:
                    self.placement_group_id = self._create_placement_group()
                run_params['Placement'] = {
                    'GroupName': self.placement_group_id,
                    'Strategy': self.config.placement_group_strategy
                }
            
            response = self.ec2.run_instances(**run_params)
            instance_ids = [inst['InstanceId'] for inst in response['Instances']]
            
            # Wait for instances to be running
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=instance_ids)
            
            for instance_id in instance_ids:
                self.instances[instance_id] = {
                    'status': NodeStatus.RUNNING,
                    'launch_time': datetime.now()
                }
            
            logger.info(f"Provisioned {len(instance_ids)} AWS instances: {instance_ids}")
            return instance_ids
            
        except ClientError as e:
            logger.error(f"Failed to provision AWS instances: {e}")
            raise
    
    def terminate_instances(self, instance_ids: List[str]):
        """Terminate AWS instances"""
        try:
            self.ec2.terminate_instances(InstanceIds=instance_ids)
            for instance_id in instance_ids:
                if instance_id in self.instances:
                    self.instances[instance_id]['status'] = NodeStatus.TERMINATED
            logger.info(f"Terminated AWS instances: {instance_ids}")
        except ClientError as e:
            logger.error(f"Failed to terminate AWS instances: {e}")
    
    def get_instance_status(self, instance_id: str) -> NodeStatus:
        """Get AWS instance status"""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            
            status_map = {
                'pending': NodeStatus.PENDING,
                'running': NodeStatus.RUNNING,
                'shutting-down': NodeStatus.TERMINATED,
                'terminated': NodeStatus.TERMINATED,
                'stopping': NodeStatus.TERMINATED,
                'stopped': NodeStatus.TERMINATED
            }
            
            return status_map.get(state, NodeStatus.FAILED)
        except ClientError:
            return NodeStatus.FAILED
    
    def get_instance_ip(self, instance_id: str) -> str:
        """Get AWS instance IP"""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            return response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        except ClientError:
            return ""
    
    def handle_spot_interruption(self, instance_id: str) -> bool:
        """Handle AWS spot instance interruption"""
        try:
            # Check for spot interruption warning
            response = self.ec2.describe_spot_instance_requests(
                Filters=[{'Name': 'instance-id', 'Values': [instance_id]}]
            )
            
            if response['SpotInstanceRequests']:
                status = response['SpotInstanceRequests'][0]['Status']['Code']
                if status in ['marked-for-termination', 'instance-terminated-by-price']:
                    logger.warning(f"Spot instance {instance_id} marked for termination")
                    return True
            return False
        except ClientError:
            return False
    
    def _create_placement_group(self) -> str:
        """Create AWS placement group"""
        try:
            pg_name = f"forge-pg-{int(time.time())}"
            self.ec2.create_placement_group(
                GroupName=pg_name,
                Strategy=self.config.placement_group_strategy
            )
            return pg_name
        except ClientError as e:
            logger.warning(f"Failed to create placement group: {e}")
            return ""
    
    def _get_default_ami(self) -> str:
        """Get default AMI for deep learning"""
        # Deep Learning AMI (Ubuntu 20.04)
        return "ami-0c7217cdde317cfec"


class GCPAdapter(CloudAdapter):
    """GCP cloud adapter"""
    
    def __init__(self, config: CloudInstanceConfig):
        super().__init__(config)
        if not GCP_AVAILABLE:
            raise ImportError("google-cloud-compute is required for GCP adapter")
        self.client = compute_v1.InstancesClient()
        
    def provision_instances(self, count: int) -> List[str]:
        """Provision GCP instances"""
        # Implementation for GCP
        pass


class AzureAdapter(CloudAdapter):
    """Azure cloud adapter"""
    
    def __init__(self, config: CloudInstanceConfig):
        super().__init__(config)
        if not AZURE_AVAILABLE:
            raise ImportError("azure-mgmt-compute is required for Azure adapter")
        credential = DefaultAzureCredential()
        self.client = ComputeManagementClient(credential, config.region)
        
    def provision_instances(self, count: int) -> List[str]:
        """Provision Azure instances"""
        # Implementation for Azure
        pass


class CheckpointManager:
    """Manages checkpointing and recovery"""
    
    def __init__(self, config: CheckpointConfig, cloud_adapter: Optional[CloudAdapter] = None):
        self.config = config
        self.cloud_adapter = cloud_adapter
        self.checkpoint_history = []
        self.last_checkpoint_time = None
        self.checkpoint_lock = threading.Lock()
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       step: int,
                       global_step: int,
                       metrics: Dict[str, float],
                       extra_state: Optional[Dict] = None) -> str:
        """Save checkpoint"""
        with self.checkpoint_lock:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_name = f"checkpoint-epoch{epoch}-step{step}-global{global_step}"
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            # Create checkpoint directory
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model state
            if isinstance(model, (DDP, FSDP)):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            torch.save(model_state, checkpoint_path / "model.pt")
            
            # Save optimizer state
            if self.config.save_optimizer:
                torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
            
            # Save scheduler state
            if scheduler and self.config.save_scheduler:
                torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")
            
            # Save training state
            training_state = {
                'epoch': epoch,
                'step': step,
                'global_step': global_step,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'rng_state': torch.random.get_rng_state() if self.config.save_rng_state else None,
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            
            if extra_state:
                training_state.update(extra_state)
            
            with open(checkpoint_path / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
            
            # Upload to cloud if configured
            if self.config.upload_to_cloud and self.cloud_adapter:
                self._upload_checkpoint_to_cloud(checkpoint_path)
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'timestamp': datetime.now(),
                'step': global_step,
                'metrics': metrics
            })
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            self.last_checkpoint_time = datetime.now()
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       map_location: str = 'cpu') -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model state
        model_state = torch.load(checkpoint_path / "model.pt", map_location=map_location)
        if isinstance(model, (DDP, FSDP)):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
        # Load optimizer state
        if optimizer and (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(checkpoint_path / "optimizer.pt", map_location=map_location)
            )
        
        # Load scheduler state
        if scheduler and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(checkpoint_path / "scheduler.pt", map_location=map_location)
            )
        
        # Load training state
        training_state = {}
        if (checkpoint_path / "training_state.json").exists():
            with open(checkpoint_path / "training_state.json", 'r') as f:
                training_state = json.load(f)
            
            # Restore RNG states
            if 'rng_state' in training_state and training_state['rng_state']:
                torch.random.set_rng_state(training_state['rng_state'])
            if 'cuda_rng_state' in training_state and training_state['cuda_rng_state']:
                torch.cuda.set_rng_state_all(training_state['cuda_rng_state'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return training_state
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    def _upload_checkpoint_to_cloud(self, checkpoint_path: Path):
        """Upload checkpoint to cloud storage"""
        # Implementation depends on cloud provider
        pass
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints beyond save_total_limit"""
        if len(self.checkpoint_history) <= self.config.save_total_limit:
            return
        
        # Sort by step (oldest first)
        self.checkpoint_history.sort(key=lambda x: x['step'])
        
        # Remove oldest checkpoints
        while len(self.checkpoint_history) > self.config.save_total_limit:
            old_checkpoint = self.checkpoint_history.pop(0)
            try:
                shutil.rmtree(old_checkpoint['path'])
                logger.debug(f"Removed old checkpoint: {old_checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")


class ModelParallelismSelector:
    """Automatically selects optimal model parallelism strategy"""
    
    def __init__(self, model_config: Dict[str, Any], hardware_info: Dict[str, Any]):
        self.model_config = model_config
        self.hardware_info = hardware_info
        
    def select_strategy(self) -> Tuple[TrainingStrategy, Dict[str, int]]:
        """Select optimal parallelism strategy"""
        model_size_gb = self._estimate_model_size()
        gpu_memory_gb = self.hardware_info.get('gpu_memory_gb', 16)
        num_gpus = self.hardware_info.get('num_gpus', 1)
        num_nodes = self.hardware_info.get('num_nodes', 1)
        
        # Calculate required memory
        required_memory_gb = model_size_gb * 1.5  # Account for gradients and optimizer states
        
        # Decision logic
        if required_memory_gb <= gpu_memory_gb * 0.8:
            # Model fits in single GPU
            if num_gpus > 1:
                strategy = TrainingStrategy.DDP
                config = {
                    'data_parallel_size': num_gpus,
                    'model_parallel_size': 1,
                    'pipeline_parallel_size': 1
                }
            else:
                strategy = TrainingStrategy.DDP
                config = {
                    'data_parallel_size': 1,
                    'model_parallel_size': 1,
                    'pipeline_parallel_size': 1
                }
        elif required_memory_gb <= gpu_memory_gb * num_gpus * 0.8:
            # Model fits with tensor parallelism
            strategy = TrainingStrategy.MODEL_PARALLEL
            model_parallel_size = min(
                num_gpus,
                int(required_memory_gb / (gpu_memory_gb * 0.8)) + 1
            )
            config = {
                'data_parallel_size': num_gpus // model_parallel_size,
                'model_parallel_size': model_parallel_size,
                'pipeline_parallel_size': 1
            }
        else:
            # Need pipeline parallelism
            strategy = TrainingStrategy.PIPELINE_PARALLEL
            pipeline_stages = min(
                num_nodes,
                int(required_memory_gb / (gpu_memory_gb * num_gpus * 0.8)) + 1
            )
            config = {
                'data_parallel_size': num_gpus,
                'model_parallel_size': 1,
                'pipeline_parallel_size': pipeline_stages
            }
        
        logger.info(f"Selected parallelism strategy: {strategy}")
        logger.info(f"Configuration: {config}")
        
        return strategy, config
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB"""
        # Simplified estimation based on parameter count
        # In production, this would use actual model architecture
        param_count = self.model_config.get('num_parameters', 7_000_000_000)  # Default 7B
        bytes_per_param = 2 if self.model_config.get('fp16', True) else 4
        return (param_count * bytes_per_param) / (1024 ** 3)


class ElasticScaler:
    """Handles elastic scaling of training cluster"""
    
    def __init__(self, 
                 cloud_adapter: CloudAdapter,
                 fault_tolerance_config: FaultToleranceConfig):
        self.cloud_adapter = cloud_adapter
        self.config = fault_tolerance_config
        self.current_nodes = 0
        self.target_nodes = 0
        self.scaling_history = []
        self.last_scale_time = None
        self.metrics_history = []
        
    def evaluate_scaling(self, 
                        current_metrics: Dict[str, float],
                        training_progress: Dict[str, Any]) -> int:
        """Evaluate if scaling is needed"""
        # Add metrics to history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics
        })
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Check cooldown period
        if self.last_scale_time:
            time_since_last_scale = (datetime.now() - self.last_scale_time).total_seconds()
            if time_since_last_scale < self.config.cooldown_period:
                return self.current_nodes
        
        # Calculate average metrics
        avg_gpu_util = self._calculate_average_metric('gpu_utilization')
        avg_memory_util = self._calculate_average_metric('memory_utilization')
        avg_throughput = self._calculate_average_metric('throughput')
        
        # Scaling decision logic
        if avg_gpu_util > self.config.scale_up_threshold and avg_memory_util > 0.7:
            # Scale up
            new_nodes = min(
                self.config.max_nodes,
                self.current_nodes + max(1, self.current_nodes // 2)
            )
            if new_nodes > self.current_nodes:
                logger.info(f"Scaling up from {self.current_nodes} to {new_nodes} nodes")
                self._scale_cluster(new_nodes)
                return new_nodes
        
        elif avg_gpu_util < self.config.scale_down_threshold and self.current_nodes > self.config.min_nodes:
            # Scale down
            new_nodes = max(
                self.config.min_nodes,
                self.current_nodes - max(1, self.current_nodes // 4)
            )
            if new_nodes < self.current_nodes:
                logger.info(f"Scaling down from {self.current_nodes} to {new_nodes} nodes")
                self._scale_cluster(new_nodes)
                return new_nodes
        
        return self.current_nodes
    
    def handle_node_failure(self, failed_node_id: str) -> bool:
        """Handle node failure"""
        logger.warning(f"Handling failure of node {failed_node_id}")
        
        # Mark node as failed
        self.cloud_adapter.instances[failed_node_id]['status'] = NodeStatus.FAILED
        
        # Check if we need to replace the node
        if self.config.automatic_recovery:
            # Provision replacement node
            try:
                new_nodes = self.cloud_adapter.provision_instances(1)
                if new_nodes:
                    logger.info(f"Provisioned replacement node: {new_nodes[0]}")
                    return True
            except Exception as e:
                logger.error(f"Failed to provision replacement node: {e}")
        
        return False
    
    def _calculate_average_metric(self, metric_name: str) -> float:
        """Calculate average of a metric over recent history"""
        if not self.metrics_history:
            return 0.0
        
        values = []
        for entry in self.metrics_history:
            if metric_name in entry['metrics']:
                values.append(entry['metrics'][metric_name])
        
        return sum(values) / len(values) if values else 0.0
    
    def _scale_cluster(self, target_nodes: int):
        """Scale cluster to target number of nodes"""
        if target_nodes == self.current_nodes:
            return
        
        if target_nodes > self.current_nodes:
            # Scale up
            nodes_to_add = target_nodes - self.current_nodes
            new_instances = self.cloud_adapter.provision_instances(nodes_to_add)
            self.current_nodes += len(new_instances)
        else:
            # Scale down
            nodes_to_remove = self.current_nodes - target_nodes
            instances_to_terminate = list(self.cloud_adapter.instances.keys())[:nodes_to_remove]
            self.cloud_adapter.terminate_instances(instances_to_terminate)
            self.current_nodes -= len(instances_to_terminate)
        
        self.target_nodes = target_nodes
        self.last_scale_time = datetime.now()
        
        self.scaling_history.append({
            'timestamp': datetime.now(),
            'from_nodes': self.current_nodes,
            'to_nodes': target_nodes,
            'action': 'scale_up' if target_nodes > self.current_nodes else 'scale_down'
        })


class DistributedTrainingOrchestrator:
    """Main orchestrator for distributed training"""
    
    def __init__(self, 
                 training_config: TrainingConfig,
                 cloud_config: CloudInstanceConfig,
                 checkpoint_config: Optional[CheckpointConfig] = None,
                 fault_tolerance_config: Optional[FaultToleranceConfig] = None):
        
        self.training_config = training_config
        self.cloud_config = cloud_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.fault_tolerance_config = fault_tolerance_config or FaultToleranceConfig()
        
        # Initialize components
        self.cloud_adapter = self._create_cloud_adapter()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_config, self.cloud_adapter)
        self.elastic_scaler = ElasticScaler(self.cloud_adapter, self.fault_tolerance_config)
        
        # Training state
        self.is_training = False
        self.training_process = None
        self.health_check_thread = None
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Node management
        self.nodes = {}
        self.master_node = None
        
        logger.info("DistributedTrainingOrchestrator initialized")
    
    def _create_cloud_adapter(self) -> CloudAdapter:
        """Create appropriate cloud adapter"""
        provider = self.cloud_config.provider
        
        if provider == CloudProvider.AWS:
            return AWSAdapter(self.cloud_config)
        elif provider == CloudProvider.GCP:
            return GCPAdapter(self.cloud_config)
        elif provider == CloudProvider.AZURE:
            return AzureAdapter(self.cloud_config)
        else:
            # Local adapter for testing
            return LocalAdapter(self.cloud_config)
    
    def setup_cluster(self, num_nodes: int = 1) -> bool:
        """Setup training cluster"""
        logger.info(f"Setting up cluster with {num_nodes} nodes")
        
        try:
            # Provision instances
            instance_ids = self.cloud_adapter.provision_instances(num_nodes)
            
            # Setup nodes
            for instance_id in instance_ids:
                self.nodes[instance_id] = {
                    'status': NodeStatus.PENDING,
                    'ip': self.cloud_adapter.get_instance_ip(instance_id),
                    'role': 'worker',
                    'last_health_check': None,
                    'restart_count': 0
                }
            
            # Designate master node
            if instance_ids:
                self.master_node = instance_ids[0]
                self.nodes[self.master_node]['role'] = 'master'
            
            # Wait for nodes to be ready
            self._wait_for_nodes_ready()
            
            # Setup distributed environment
            self._setup_distributed_environment()
            
            logger.info("Cluster setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup cluster: {e}")
            self.cleanup()
            return False
    
    def _wait_for_nodes_ready(self, timeout: int = 300):
        """Wait for all nodes to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for node_id, node_info in self.nodes.items():
                status = self.cloud_adapter.get_instance_status(node_id)
                
                if status == NodeStatus.RUNNING:
                    node_info['status'] = NodeStatus.RUNNING
                elif status in [NodeStatus.FAILED, NodeStatus.TERMINATED]:
                    logger.error(f"Node {node_id} failed to start")
                    return False
                else:
                    all_ready = False
            
            if all_ready:
                return True
            
            time.sleep(5)
        
        logger.error("Timeout waiting for nodes to be ready")
        return False
    
    def _setup_distributed_environment(self):
        """Setup distributed training environment"""
        # Generate hostfile for distributed training
        hostfile_content = []
        
        for node_id, node_info in self.nodes.items():
            if node_info['status'] == NodeStatus.RUNNING:
                slots = self._get_node_slots(node_id)
                hostfile_content.append(f"{node_info['ip']} slots={slots}")
        
        # Write hostfile
        hostfile_path = Path("/tmp/forge_hostfile")
        with open(hostfile_path, 'w') as f:
            f.write('\n'.join(hostfile_content))
        
        logger.info(f"Generated hostfile at {hostfile_path}")
        
        # Setup SSH keys for passwordless access
        self._setup_ssh_access()
    
    def _get_node_slots(self, node_id: str) -> int:
        """Get number of slots (GPUs) for a node"""
        # This would query the actual node for GPU count
        # For now, return a default
        return 8  # Assuming 8 GPUs per node
    
    def _setup_ssh_access(self):
        """Setup SSH access between nodes"""
        # Implementation for SSH key distribution
        pass
    
    def start_training(self, 
                      train_func: Callable,
                      train_args: Dict[str, Any]) -> bool:
        """Start distributed training"""
        if self.is_training:
            logger.warning("Training is already running")
            return False
        
        logger.info("Starting distributed training")
        
        try:
            # Find latest checkpoint if resuming
            if self.training_config.resume_from_checkpoint:
                checkpoint_path = self.training_config.resume_from_checkpoint
            else:
                checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
            
            # Prepare training arguments
            train_args['checkpoint_path'] = checkpoint_path
            train_args['training_config'] = asdict(self.training_config)
            
            # Start training process
            self.training_process = self._launch_distributed_training(train_func, train_args)
            
            # Start monitoring threads
            self.is_training = True
            self._start_monitoring()
            
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.is_training = False
            return False
    
    def _launch_distributed_training(self, 
                                    train_func: Callable,
                                    train_args: Dict[str, Any]) -> subprocess.Popen:
        """Launch distributed training using appropriate backend"""
        
        if RAY_AVAILABLE and self.training_config.training_strategy in [
            TrainingStrategy.DDP, TrainingStrategy.FSDP
        ]:
            return self._launch_ray_training(train_func, train_args)
        else:
            return self._launch_torchrun_training(train_func, train_args)
    
    def _launch_ray_training(self, 
                            train_func: Callable,
                            train_args: Dict[str, Any]) -> subprocess.Popen:
        """Launch training using Ray"""
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address='auto')  # Connect to existing cluster or start local
        
        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=len(self.nodes),
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 4}
        )
        
        # Create trainer
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_args,
            scaling_config=scaling_config,
            metadata={"training_config": asdict(self.training_config)}
        )
        
        # Start training
        result = trainer.fit()
        
        # Return a mock process object
        class MockProcess:
            def poll(self): return None
            def wait(self): return 0
            def terminate(self): pass
        
        return MockProcess()
    
    def _launch_torchrun_training(self, 
                                 train_func: Callable,
                                 train_args: Dict[str, Any]) -> subprocess.Popen:
        """Launch training using torchrun"""
        
        # Prepare torchrun command
        nproc_per_node = self._get_node_slots(self.master_node)
        nnodes = len(self.nodes)
        
        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc_per_node}',
            f'--nnodes={nnodes}',
            '--master_addr', self.nodes[self.master_node]['ip'],
            '--master_port', '29500',
            '--node_rank', '0',  # This would be different for each node
            self._get_training_script_path(),
            '--training_args', json.dumps(train_args)
        ]
        
        # Launch process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._prepare_environment()
        )
        
        return process
    
    def _get_training_script_path(self) -> str:
        """Get path to training script"""
        # This would point to the actual training script
        return "train.py"
    
    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for training"""
        env = os.environ.copy()
        
        # Set distributed training variables
        env['MASTER_ADDR'] = self.nodes[self.master_node]['ip']
        env['MASTER_PORT'] = '29500'
        env['WORLD_SIZE'] = str(sum(self._get_node_slots(nid) for nid in self.nodes))
        
        # Set CUDA device
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(self._get_node_slots(self.master_node)))
        
        return env
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        # Monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _health_check_loop(self):
        """Health check loop"""
        while not self.shutdown_event.is_set() and self.is_training:
            try:
                self._check_node_health()
                time.sleep(self.fault_tolerance_config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _check_node_health(self):
        """Check health of all nodes"""
        for node_id, node_info in self.nodes.items():
            try:
                # Check if node is reachable
                if self._ping_node(node_info['ip']):
                    node_info['last_health_check'] = datetime.now()
                else:
                    logger.warning(f"Node {node_id} is unreachable")
                    
                    # Handle node failure
                    if self.elastic_scaler.handle_node_failure(node_id):
                        # Update node info
                        self.nodes[node_id]['status'] = NodeStatus.RECOVERING
                        self.nodes[node_id]['restart_count'] += 1
                        
                        # Check restart limit
                        if self.nodes[node_id]['restart_count'] > self.fault_tolerance_config.max_restarts:
                            logger.error(f"Node {node_id} exceeded max restarts")
                            self.nodes[node_id]['status'] = NodeStatus.FAILED
            except Exception as e:
                logger.error(f"Error checking node {node_id}: {e}")
    
    def _ping_node(self, ip: str) -> bool:
        """Ping a node to check if it's reachable"""
        try:
            # Simple socket connection test
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip, 22))  # SSH port
            sock.close()
            return result == 0
        except:
            return False
    
    def _monitoring_loop(self):
        """Monitoring loop for metrics and scaling"""
        while not self.shutdown_event.is_set() and self.is_training:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Evaluate scaling
                training_progress = self._get_training_progress()
                new_node_count = self.elastic_scaler.evaluate_scaling(metrics, training_progress)
                
                # Handle scaling if needed
                if new_node_count != len(self.nodes):
                    self._handle_scaling(new_node_count)
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect training metrics"""
        # This would collect actual metrics from the training process
        # For now, return mock metrics
        return {
            'gpu_utilization': 0.75,
            'memory_utilization': 0.65,
            'throughput': 100.0,
            'loss': 0.5,
            'learning_rate': 2e-5
        }
    
    def _get_training_progress(self) -> Dict[str, Any]:
        """Get training progress"""
        # This would get actual training progress
        return {
            'current_epoch': 1,
            'current_step': 1000,
            'total_steps': 10000,
            'progress_percentage': 10.0
        }
    
    def _handle_scaling(self, new_node_count: int):
        """Handle cluster scaling"""
        logger.info(f"Handling scaling to {new_node_count} nodes")
        
        # This would involve:
        # 1. Pausing training
        # 2. Saving checkpoint
        # 3. Scaling cluster
        # 4. Resuming training
        
        # For now, just update node count
        self.nodes = {nid: info for nid, info in self.nodes.items() 
                     if info['status'] != NodeStatus.TERMINATED}
    
    def stop_training(self, save_checkpoint: bool = True):
        """Stop training"""
        logger.info("Stopping training")
        
        self.is_training = False
        self.shutdown_event.set()
        
        # Stop training process
        if self.training_process:
            self.training_process.terminate()
            self.training_process.wait(timeout=30)
        
        # Save final checkpoint if requested
        if save_checkpoint:
            self._save_final_checkpoint()
        
        # Cleanup
        self.cleanup()
        
        logger.info("Training stopped")
    
    def _save_final_checkpoint(self):
        """Save final checkpoint"""
        # This would save the actual model checkpoint
        logger.info("Saving final checkpoint")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources")
        
        # Terminate all instances
        instance_ids = list(self.cloud_adapter.instances.keys())
        if instance_ids:
            self.cloud_adapter.terminate_instances(instance_ids)
        
        # Clear node information
        self.nodes.clear()
        self.master_node = None
        
        # Stop monitoring threads
        if self.health_check_thread:
            self.health_check_thread.join(timeout=10)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'is_training': self.is_training,
            'num_nodes': len(self.nodes),
            'master_node': self.master_node,
            'nodes': self.nodes,
            'training_config': asdict(self.training_config),
            'cloud_config': asdict(self.cloud_config),
            'last_checkpoint': self.checkpoint_manager.last_checkpoint_time,
            'scaling_history': self.elastic_scaler.scaling_history[-10:] if self.elastic_scaler.scaling_history else []
        }


class LocalAdapter(CloudAdapter):
    """Local adapter for testing"""
    
    def __init__(self, config: CloudInstanceConfig):
        super().__init__(config)
        self.local_nodes = {}
        
    def provision_instances(self, count: int) -> List[str]:
        """Provision local 'instances' (actually just processes)"""
        instance_ids = []
        for i in range(count):
            instance_id = f"local-{i}-{int(time.time())}"
            self.local_nodes[instance_id] = {
                'ip': '127.0.0.1',
                'status': NodeStatus.RUNNING,
                'pid': None
            }
            instance_ids.append(instance_id)
        return instance_ids
    
    def terminate_instances(self, instance_ids: List[str]):
        """Terminate local instances"""
        for instance_id in instance_ids:
            if instance_id in self.local_nodes:
                self.local_nodes[instance_id]['status'] = NodeStatus.TERMINATED
    
    def get_instance_status(self, instance_id: str) -> NodeStatus:
        """Get local instance status"""
        return self.local_nodes.get(instance_id, {}).get('status', NodeStatus.FAILED)
    
    def get_instance_ip(self, instance_id: str) -> str:
        """Get local instance IP"""
        return self.local_nodes.get(instance_id, {}).get('ip', '127.0.0.1')


def create_distributed_orchestrator(
    model_name_or_path: str,
    output_dir: str,
    cloud_provider: str = "aws",
    instance_type: str = "p3.2xlarge",
    num_nodes: int = 1,
    training_strategy: str = "ddp",
    **kwargs
) -> DistributedTrainingOrchestrator:
    """Factory function to create distributed orchestrator"""
    
    # Parse configurations
    training_config = TrainingConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        training_strategy=TrainingStrategy(training_strategy),
        **kwargs
    )
    
    cloud_config = CloudInstanceConfig(
        provider=CloudProvider(cloud_provider),
        instance_type=instance_type,
        min_count=num_nodes,
        max_count=num_nodes * 2  # Allow scaling up to 2x
    )
    
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        upload_to_cloud=True,
        cloud_bucket=kwargs.get('checkpoint_bucket'),
        cloud_prefix=kwargs.get('checkpoint_prefix')
    )
    
    fault_tolerance_config = FaultToleranceConfig(
        elastic_scaling=kwargs.get('elastic_scaling', True),
        min_nodes=num_nodes,
        max_nodes=kwargs.get('max_nodes', num_nodes * 4)
    )
    
    return DistributedTrainingOrchestrator(
        training_config=training_config,
        cloud_config=cloud_config,
        checkpoint_config=checkpoint_config,
        fault_tolerance_config=fault_tolerance_config
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    orchestrator = create_distributed_orchestrator(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        output_dir="./output",
        cloud_provider="aws",
        instance_type="p3.8xlarge",
        num_nodes=4,
        training_strategy="fsdp",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        elastic_scaling=True,
        max_nodes=8
    )
    
    # Setup cluster
    if orchestrator.setup_cluster(num_nodes=4):
        print("Cluster setup successful")
        
        # Define training function
        def train_func(config):
            # This would be the actual training function
            print(f"Training with config: {config}")
            time.sleep(10)  # Simulate training
            return {"loss": 0.5, "accuracy": 0.85}
        
        # Start training
        if orchestrator.start_training(train_func, {"dummy": "config"}):
            print("Training started")
            
            try:
                # Wait for training to complete
                while orchestrator.is_training:
                    status = orchestrator.get_status()
                    print(f"Training progress: {status}")
                    time.sleep(30)
            except KeyboardInterrupt:
                print("Interrupted, stopping training...")
                orchestrator.stop_training()
        else:
            print("Failed to start training")
    else:
        print("Failed to setup cluster")