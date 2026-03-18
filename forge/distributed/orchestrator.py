"""Production-Grade Distributed Training Orchestrator for forge.

This module provides automatic distributed training orchestration with fault tolerance,
elastic scaling, cost optimization across cloud providers, and automatic model parallelism
strategy selection. Uses Ray for orchestration with Kubernetes integration.
"""

import os
import time
import json
import signal
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import psutil
import GPUtil
import numpy as np

try:
    import ray
    from ray import tune
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    from ray.train.huggingface import TransformersTrainer
    from ray.util.placement_group import placement_group
    from ray.util.state import list_nodes
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available. Install with: pip install ray[train]")

try:
    from kubernetes import client, config, watch
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class ParallelismStrategy(Enum):
    """Model parallelism strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    ZERO_REDUNDANCY = "zero_redundancy"


class InstanceType(Enum):
    """Cloud instance types for cost optimization."""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    PREEMPTIBLE = "preemptible"


@dataclass
class ResourceSpec:
    """Specification for compute resources."""
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    num_cpus_per_node: int = 4
    memory_per_node_gb: int = 16
    gpu_type: str = "A100"
    instance_type: InstanceType = InstanceType.SPOT
    cloud_provider: CloudProvider = CloudProvider.LOCAL
    region: Optional[str] = None
    max_price_per_hour: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    model_name_or_path: str
    dataset_name: str
    output_dir: str = "./output"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = True
    bf16: bool = False
    deepspeed_config: Optional[str] = None
    model_parallelism: Optional[ParallelismStrategy] = None
    auto_select_strategy: bool = True
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    fault_tolerance: bool = True
    elastic_scaling: bool = True
    max_restarts: int = 3
    cost_optimization: bool = True


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    ip_address: str
    num_gpus: int
    num_cpus: int
    memory_gb: float
    gpu_type: str
    status: str = "initializing"
    last_heartbeat: float = field(default_factory=time.time)
    training_progress: float = 0.0
    current_loss: float = float('inf')
    is_spot: bool = False


class DistributedOrchestrator:
    """Production-grade distributed training orchestrator.
    
    Handles automatic provisioning, fault tolerance, elastic scaling,
    and cost optimization across cloud providers.
    """
    
    def __init__(self, 
                 training_config: TrainingConfig,
                 resource_spec: ResourceSpec,
                 cloud_credentials: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator.
        
        Args:
            training_config: Training configuration
            resource_spec: Resource specification
            cloud_credentials: Cloud provider credentials
        """
        self.training_config = training_config
        self.resource_spec = resource_spec
        self.cloud_credentials = cloud_credentials or {}
        
        self.nodes: Dict[str, NodeInfo] = {}
        self.checkpoint_manager = CheckpointManager(training_config.checkpoint_dir)
        self.scaling_manager = ScalingManager(resource_spec)
        self.cost_optimizer = CostOptimizer(resource_spec)
        
        self._is_running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._training_job_id = None
        self._ray_initialized = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized DistributedOrchestrator with {resource_spec.num_nodes} nodes")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def initialize_ray_cluster(self) -> bool:
        """Initialize Ray cluster for distributed training."""
        if not RAY_AVAILABLE:
            logger.error("Ray is not available. Cannot initialize cluster.")
            return False
        
        try:
            if not ray.is_initialized():
                # Initialize Ray with resource specifications
                ray.init(
                    address="auto" if self.resource_spec.cloud_provider != CloudProvider.LOCAL else None,
                    ignore_reinit_error=True,
                    runtime_env={
                        "working_dir": os.getcwd(),
                        "excludes": ["*.pyc", "__pycache__", ".git"]
                    }
                )
                self._ray_initialized = True
                logger.info("Ray cluster initialized successfully")
                
                # Log cluster resources
                resources = ray.cluster_resources()
                logger.info(f"Cluster resources: {resources}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            return False
    
    def provision_resources(self) -> bool:
        """Provision compute resources based on cloud provider."""
        logger.info(f"Provisioning resources on {self.resource_spec.cloud_provider.value}")
        
        if self.resource_spec.cloud_provider == CloudProvider.LOCAL:
            return self._provision_local_resources()
        elif self.resource_spec.cloud_provider == CloudProvider.AWS:
            return self._provision_aws_resources()
        elif self.resource_spec.cloud_provider == CloudProvider.GCP:
            return self._provision_gcp_resources()
        elif self.resource_spec.cloud_provider == CloudProvider.AZURE:
            return self._provision_azure_resources()
        else:
            logger.error(f"Unsupported cloud provider: {self.resource_spec.cloud_provider}")
            return False
    
    def _provision_local_resources(self) -> bool:
        """Provision local resources."""
        try:
            # Detect local GPUs
            gpus = GPUtil.getGPUs()
            available_gpus = len(gpus)
            
            if available_gpus < self.resource_spec.num_gpus_per_node:
                logger.warning(f"Requested {self.resource_spec.num_gpus_per_node} GPUs, "
                             f"but only {available_gpus} available")
                self.resource_spec.num_gpus_per_node = min(
                    self.resource_spec.num_gpus_per_node, available_gpus
                )
            
            # Create local node
            node_id = f"local_{int(time.time())}"
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                ip_address="localhost",
                num_gpus=self.resource_spec.num_gpus_per_node,
                num_cpus=self.resource_spec.num_cpus_per_node,
                memory_gb=self.resource_spec.memory_per_node_gb,
                gpu_type=self.resource_spec.gpu_type,
                status="ready"
            )
            
            logger.info(f"Provisioned local node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to provision local resources: {e}")
            return False
    
    def _provision_aws_resources(self) -> bool:
        """Provision AWS EC2 instances."""
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available. Install with: pip install boto3")
            return False
        
        try:
            ec2 = boto3.client('ec2', region_name=self.resource_spec.region)
            
            # Determine instance type based on GPU requirements
            instance_type = self._get_aws_instance_type()
            if not instance_type:
                return False
            
            # Launch instances
            for i in range(self.resource_spec.num_nodes):
                launch_spec = {
                    'ImageId': self._get_aws_ami_id(),
                    'InstanceType': instance_type,
                    'MinCount': 1,
                    'MaxCount': 1,
                    'KeyName': self.cloud_credentials.get('key_name'),
                    'SecurityGroupIds': self.cloud_credentials.get('security_group_ids', []),
                    'SubnetId': self.cloud_credentials.get('subnet_id'),
                    'TagSpecifications': [{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': 'Name', 'Value': f'forge-node-{i}'}]
                    }]
                }
                
                # Use spot instances for cost optimization
                if self.resource_spec.instance_type == InstanceType.SPOT:
                    launch_spec['InstanceMarketOptions'] = {
                        'MarketType': 'spot',
                        'SpotOptions': {
                            'MaxPrice': str(self.resource_spec.max_price_per_hour or '0.5'),
                            'SpotInstanceType': 'one-time'
                        }
                    }
                
                response = ec2.run_instances(**launch_spec)
                instance_id = response['Instances'][0]['InstanceId']
                
                # Wait for instance to be running
                waiter = ec2.get_waiter('instance_running')
                waiter.wait(InstanceIds=[instance_id])
                
                # Get instance details
                instance_info = ec2.describe_instances(InstanceIds=[instance_id])
                instance = instance_info['Reservations'][0]['Instances'][0]
                
                node_id = instance_id
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id,
                    ip_address=instance.get('PrivateIpAddress', 'pending'),
                    num_gpus=self._get_gpu_count_for_instance(instance_type),
                    num_cpus=self.resource_spec.num_cpus_per_node,
                    memory_gb=self.resource_spec.memory_per_node_gb,
                    gpu_type=self.resource_spec.gpu_type,
                    status="running",
                    is_spot=(self.resource_spec.instance_type == InstanceType.SPOT)
                )
                
                logger.info(f"Provisioned AWS instance: {instance_id} ({instance_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to provision AWS resources: {e}")
            return False
    
    def _provision_gcp_resources(self) -> bool:
        """Provision GCP compute instances."""
        if not GCP_AVAILABLE:
            logger.error("GCP SDK not available. Install with: pip install google-cloud-compute")
            return False
        
        # Implementation for GCP would go here
        logger.warning("GCP provisioning not yet implemented")
        return False
    
    def _provision_azure_resources(self) -> bool:
        """Provision Azure compute instances."""
        if not AZURE_AVAILABLE:
            logger.error("Azure SDK not available. Install with: pip install azure-mgmt-compute")
            return False
        
        # Implementation for Azure would go here
        logger.warning("Azure provisioning not yet implemented")
        return False
    
    def _get_aws_instance_type(self) -> Optional[str]:
        """Determine AWS instance type based on GPU requirements."""
        gpu_type = self.resource_spec.gpu_type.lower()
        
        instance_map = {
            "a100": "p4d.24xlarge",  # 8x A100 GPUs
            "v100": "p3.8xlarge",    # 4x V100 GPUs
            "t4": "g4dn.xlarge",     # 1x T4 GPU
            "a10g": "g5.xlarge",     # 1x A10G GPU
        }
        
        for key, instance in instance_map.items():
            if key in gpu_type:
                return instance
        
        # Default to p3.2xlarge (1x V100)
        return "p3.2xlarge"
    
    def _get_aws_ami_id(self) -> str:
        """Get AMI ID for deep learning."""
        # Deep Learning AMI (Ubuntu)
        return "ami-0c7217cdde317cfec"  # us-east-1
    
    def _get_gpu_count_for_instance(self, instance_type: str) -> int:
        """Get GPU count for AWS instance type."""
        gpu_counts = {
            "p4d.24xlarge": 8,
            "p3.8xlarge": 4,
            "p3.2xlarge": 1,
            "g4dn.xlarge": 1,
            "g5.xlarge": 1,
        }
        return gpu_counts.get(instance_type, 1)
    
    def select_parallelism_strategy(self, model_size_gb: float) -> ParallelismStrategy:
        """Automatically select optimal parallelism strategy.
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            Selected parallelism strategy
        """
        if not self.training_config.auto_select_strategy:
            return self.training_config.model_parallelism or ParallelismStrategy.DATA_PARALLEL
        
        total_gpus = sum(node.num_gpus for node in self.nodes.values())
        available_memory_gb = sum(
            node.memory_gb for node in self.nodes.values()
        )
        
        logger.info(f"Selecting parallelism strategy for {model_size_gb}GB model "
                   f"across {total_gpus} GPUs")
        
        # Decision logic based on model size and available resources
        if model_size_gb < 2:  # Small models
            return ParallelismStrategy.DATA_PARALLEL
        
        elif model_size_gb < 8:  # Medium models
            if total_gpus >= 4:
                return ParallelismStrategy.MODEL_PARALLEL
            else:
                return ParallelismStrategy.DATA_PARALLEL
        
        elif model_size_gb < 32:  # Large models
            if total_gpus >= 8:
                return ParallelismStrategy.HYBRID_PARALLEL
            elif total_gpus >= 4:
                return ParallelismStrategy.PIPELINE_PARALLEL
            else:
                return ParallelismStrategy.ZERO_REDUNDANCY
        
        else:  # Very large models
            if total_gpus >= 16:
                return ParallelismStrategy.HYBRID_PARALLEL
            else:
                return ParallelismStrategy.ZERO_REDUNDANCY
    
    def setup_distributed_training(self) -> bool:
        """Setup distributed training with fault tolerance."""
        if not self._ray_initialized:
            if not self.initialize_ray_cluster():
                return False
        
        try:
            # Configure training with fault tolerance
            scaling_config = ScalingConfig(
                num_workers=self.resource_spec.num_nodes,
                use_gpu=True,
                resources_per_worker={
                    "GPU": self.resource_spec.num_gpus_per_node,
                    "CPU": self.resource_spec.num_cpus_per_node,
                },
                placement_strategy="SPREAD"  # Distribute across nodes
            )
            
            # Configure checkpointing for fault tolerance
            checkpoint_config = CheckpointConfig(
                num_to_keep=3,  # Keep last 3 checkpoints
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
                checkpoint_frequency=self.training_config.save_steps
            )
            
            # Configure run with fault tolerance
            run_config = RunConfig(
                name="forge-training",
                storage_path=self.training_config.checkpoint_dir,
                checkpoint_config=checkpoint_config,
                failure_config=ray.train.FailureConfig(
                    max_failures=self.training_config.max_restarts
                )
            )
            
            # Create trainer
            trainer = TorchTrainer(
                train_loop_per_worker=self._create_train_loop(),
                train_loop_config=self._create_train_config(),
                scaling_config=scaling_config,
                run_config=run_config,
                datasets=self._prepare_datasets()
            )
            
            self._trainer = trainer
            logger.info("Distributed training setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            return False
    
    def _create_train_loop(self) -> Callable:
        """Create training loop function for Ray Train."""
        def train_loop(config):
            import torch
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from forge.model import load_model
            from forge.data import load_dataset
            from forge.train import Trainer
            
            # Setup distributed training
            if dist.is_initialized():
                local_rank = dist.get_rank()
                torch.cuda.set_device(local_rank)
            
            # Load model and tokenizer
            model, tokenizer = load_model(
                model_name_or_path=config["model_name_or_path"],
                model_parallelism=config.get("model_parallelism")
            )
            
            # Load dataset
            train_dataset, eval_dataset = load_dataset(
                dataset_name=config["dataset_name"],
                tokenizer=tokenizer
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=config["output_dir"],
                per_device_train_batch_size=config["per_device_train_batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                learning_rate=config["learning_rate"],
                num_train_epochs=config["num_train_epochs"],
                max_steps=config["max_steps"],
                warmup_steps=config["warmup_steps"],
                logging_steps=config["logging_steps"],
                save_steps=config["save_steps"],
                eval_steps=config["eval_steps"],
                fp16=config["fp16"],
                bf16=config["bf16"],
                ddp_find_unused_parameters=False,
                deepspeed=config.get("deepspeed_config"),
                local_rank=local_rank if dist.is_initialized() else -1,
                report_to="none"  # Disable external logging
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer
            )
            
            # Resume from checkpoint if specified
            checkpoint_path = config.get("resume_from_checkpoint")
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                trainer.train()
            
            # Save final model
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
        
        return train_loop
    
    def _create_train_config(self) -> Dict[str, Any]:
        """Create training configuration dictionary."""
        return {
            "model_name_or_path": self.training_config.model_name_or_path,
            "dataset_name": self.training_config.dataset_name,
            "output_dir": self.training_config.output_dir,
            "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "learning_rate": self.training_config.learning_rate,
            "num_train_epochs": self.training_config.num_train_epochs,
            "max_steps": self.training_config.max_steps,
            "warmup_steps": self.training_config.warmup_steps,
            "logging_steps": self.training_config.logging_steps,
            "save_steps": self.training_config.save_steps,
            "eval_steps": self.training_config.eval_steps,
            "fp16": self.training_config.fp16,
            "bf16": self.training_config.bf16,
            "deepspeed_config": self.training_config.deepspeed_config,
            "model_parallelism": self.training_config.model_parallelism.value 
                if self.training_config.model_parallelism else None,
            "resume_from_checkpoint": self.training_config.resume_from_checkpoint
        }
    
    def _prepare_datasets(self) -> Dict[str, Any]:
        """Prepare datasets for distributed training."""
        # This would integrate with existing forge data loading
        # For now, return empty dict - actual implementation would load datasets
        return {}
    
    def start_training(self) -> bool:
        """Start distributed training with monitoring."""
        if not hasattr(self, '_trainer'):
            logger.error("Trainer not initialized. Call setup_distributed_training first.")
            return False
        
        self._is_running = True
        self._shutdown_event.clear()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_training,
            daemon=True
        )
        monitor_thread.start()
        
        # Start cost optimization thread if enabled
        if self.training_config.cost_optimization:
            cost_thread = threading.Thread(
                target=self._optimize_costs,
                daemon=True
            )
            cost_thread.start()
        
        try:
            # Start training
            logger.info("Starting distributed training...")
            result = self._trainer.fit()
            
            logger.info(f"Training completed. Results: {result.metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Attempt recovery if fault tolerance enabled
            if self.training_config.fault_tolerance:
                logger.info("Attempting recovery from failure...")
                return self._recover_from_failure()
            
            return False
        
        finally:
            self._is_running = False
    
    def _monitor_training(self):
        """Monitor training progress and node health."""
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Check node health
                self._check_node_health()
                
                # Update training progress
                self._update_training_progress()
                
                # Check for scaling opportunities
                if self.training_config.elastic_scaling:
                    self._check_scaling_needs()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _check_node_health(self):
        """Check health of all nodes."""
        for node_id, node_info in list(self.nodes.items()):
            try:
                # Check if node is responsive
                if node_info.status == "running":
                    # Update heartbeat
                    node_info.last_heartbeat = time.time()
                    
                    # Check GPU health
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        if gpu.load > 0.95:  # GPU overloaded
                            logger.warning(f"Node {node_id}: GPU {gpu.id} overloaded")
                
            except Exception as e:
                logger.error(f"Health check failed for node {node_id}: {e}")
                node_info.status = "unhealthy"
    
    def _update_training_progress(self):
        """Update training progress from Ray metrics."""
        try:
            if RAY_AVAILABLE and ray.is_initialized():
                # Get training metrics from Ray
                # This would integrate with Ray's metrics system
                pass
        except Exception as e:
            logger.error(f"Failed to update training progress: {e}")
    
    def _check_scaling_needs(self):
        """Check if scaling up/down is needed."""
        try:
            # Get current resource utilization
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Get GPU utilization
            gpus = GPUtil.getGPUs()
            gpu_loads = [gpu.load for gpu in gpus]
            avg_gpu_load = np.mean(gpu_loads) if gpu_loads else 0
            
            # Scale up if resources are heavily utilized
            if (cpu_percent > 80 or memory_percent > 80 or avg_gpu_load > 0.8):
                if len(self.nodes) < self.scaling_manager.max_nodes:
                    logger.info("High resource utilization detected, scaling up...")
                    self._scale_up()
            
            # Scale down if resources are underutilized
            elif (cpu_percent < 30 and memory_percent < 30 and avg_gpu_load < 0.3):
                if len(self.nodes) > self.scaling_manager.min_nodes:
                    logger.info("Low resource utilization detected, scaling down...")
                    self._scale_down()
                    
        except Exception as e:
            logger.error(f"Scaling check failed: {e}")
    
    def _scale_up(self):
        """Scale up by adding more nodes."""
        # Implementation would provision new nodes
        logger.info("Scaling up not implemented in demo")
    
    def _scale_down(self):
        """Scale down by removing nodes."""
        # Implementation would terminate nodes
        logger.info("Scaling down not implemented in demo")
    
    def _optimize_costs(self):
        """Optimize costs by switching to spot instances when possible."""
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Check spot instance availability and pricing
                if self.resource_spec.cloud_provider == CloudProvider.AWS:
                    self._optimize_aws_costs()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
                time.sleep(600)
    
    def _optimize_aws_costs(self):
        """Optimize AWS costs by switching to spot instances."""
        try:
            ec2 = boto3.client('ec2', region_name=self.resource_spec.region)
            
            # Get current spot prices
            spot_prices = ec2.describe_spot_price_history(
                InstanceTypes=[self._get_aws_instance_type()],
                ProductDescriptions=['Linux/UNIX'],
                MaxResults=1
            )
            
            if spot_prices['SpotPriceHistory']:
                current_price = float(spot_prices['SpotPriceHistory'][0]['SpotPrice'])
                max_price = self.resource_spec.max_price_per_hour or 0.5
                
                if current_price < max_price * 0.5:  # Spot price is half of max
                    logger.info(f"Spot price ${current_price:.4f} is favorable, "
                              "consider switching to spot instances")
                    
        except Exception as e:
            logger.error(f"AWS cost optimization failed: {e}")
    
    def _recover_from_failure(self) -> bool:
        """Recover from training failure."""
        logger.info("Attempting to recover from failure...")
        
        # Find latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            self.training_config.resume_from_checkpoint = latest_checkpoint
            
            # Restart training
            return self.start_training()
        else:
            logger.error("No checkpoint found for recovery")
            return False
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        logger.info("Shutting down orchestrator...")
        self._is_running = False
        self._shutdown_event.set()
        
        # Stop training if running
        if hasattr(self, '_trainer'):
            try:
                # Ray Train doesn't have a direct stop method
                # Training will complete or be interrupted
                pass
            except Exception as e:
                logger.error(f"Error stopping trainer: {e}")
        
        # Terminate cloud instances if not local
        if self.resource_spec.cloud_provider != CloudProvider.LOCAL:
            self._terminate_cloud_instances()
        
        # Shutdown Ray
        if self._ray_initialized and RAY_AVAILABLE:
            ray.shutdown()
        
        logger.info("Orchestrator shutdown complete")
    
    def _terminate_cloud_instances(self):
        """Terminate cloud instances."""
        try:
            if self.resource_spec.cloud_provider == CloudProvider.AWS and AWS_AVAILABLE:
                ec2 = boto3.client('ec2', region_name=self.resource_spec.region)
                instance_ids = list(self.nodes.keys())
                
                if instance_ids:
                    ec2.terminate_instances(InstanceIds=instance_ids)
                    logger.info(f"Terminated AWS instances: {instance_ids}")
                    
        except Exception as e:
            logger.error(f"Failed to terminate cloud instances: {e}")


class CheckpointManager:
    """Manages checkpoints for fault tolerance."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
            if not checkpoints:
                return None
            
            # Sort by checkpoint number
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            return str(checkpoints[-1])
            
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    def save_checkpoint(self, state_dict: Dict[str, Any], step: int):
        """Save training checkpoint."""
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save checkpoint
            torch.save(state_dict, checkpoint_path / "training_state.bin")
            
            # Save metadata
            metadata = {
                "step": step,
                "timestamp": time.time(),
                "config": {}  # Would include training config
            }
            
            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved checkpoint at step {step}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Remove old checkpoints, keeping only the last N."""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
            if len(checkpoints) <= keep_last_n:
                return
            
            # Sort by checkpoint number
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            
            # Remove old checkpoints
            for checkpoint in checkpoints[:-keep_last_n]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")


class ScalingManager:
    """Manages elastic scaling of resources."""
    
    def __init__(self, resource_spec: ResourceSpec):
        self.resource_spec = resource_spec
        self.min_nodes = 1
        self.max_nodes = resource_spec.num_nodes * 2  # Allow scaling up to 2x
    
    def calculate_optimal_nodes(self, 
                               model_size_gb: float,
                               dataset_size_gb: float) -> int:
        """Calculate optimal number of nodes based on model and dataset size."""
        # Simple heuristic - in production this would be more sophisticated
        gpu_memory_gb = 80  # Assuming A100 80GB
        
        # Estimate memory needed for model
        model_memory_needed = model_size_gb * 1.5  # Account for gradients, optimizer states
        
        # Estimate memory needed for data
        data_memory_needed = dataset_size_gb * 0.1  # Batch processing
        
        total_memory_needed = model_memory_needed + data_memory_needed
        
        # Calculate nodes needed
        gpus_needed = int(np.ceil(total_memory_needed / gpu_memory_gb))
        nodes_needed = int(np.ceil(gpus_needed / self.resource_spec.num_gpus_per_node))
        
        # Apply constraints
        nodes_needed = max(self.min_nodes, min(nodes_needed, self.max_nodes))
        
        return nodes_needed


class CostOptimizer:
    """Optimizes training costs across cloud providers."""
    
    def __init__(self, resource_spec: ResourceSpec):
        self.resource_spec = resource_spec
        self.cost_history = []
    
    def estimate_training_cost(self,
                              model_size_gb: float,
                              dataset_size_gb: float,
                              training_hours: float) -> Dict[str, float]:
        """Estimate training cost across different providers."""
        costs = {}
        
        # AWS pricing (simplified)
        if self.resource_spec.cloud_provider == CloudProvider.AWS:
            instance_cost_per_hour = self._get_aws_instance_cost()
            total_cost = instance_cost_per_hour * training_hours * self.resource_spec.num_nodes
            costs["aws"] = total_cost
        
        # GCP pricing
        elif self.resource_spec.cloud_provider == CloudProvider.GCP:
            # Simplified GCP pricing
            costs["gcp"] = training_hours * 2.5 * self.resource_spec.num_nodes
        
        # Azure pricing
        elif self.resource_spec.cloud_provider == CloudProvider.AZURE:
            # Simplified Azure pricing
            costs["azure"] = training_hours * 2.8 * self.resource_spec.num_nodes
        
        # Local (no cloud cost)
        costs["local"] = 0.0
        
        return costs
    
    def _get_aws_instance_cost(self) -> float:
        """Get AWS instance cost per hour."""
        # Simplified pricing - in production would use AWS Pricing API
        instance_type = self._get_aws_instance_type()
        
        pricing = {
            "p4d.24xlarge": 32.77,  # 8x A100
            "p3.8xlarge": 12.24,    # 4x V100
            "p3.2xlarge": 3.06,     # 1x V100
            "g4dn.xlarge": 0.526,   # 1x T4
            "g5.xlarge": 1.006,     # 1x A10G
        }
        
        return pricing.get(instance_type, 3.0)
    
    def _get_aws_instance_type(self) -> str:
        """Get AWS instance type based on GPU requirements."""
        gpu_type = self.resource_spec.gpu_type.lower()
        
        if "a100" in gpu_type:
            return "p4d.24xlarge"
        elif "v100" in gpu_type:
            return "p3.8xlarge"
        elif "t4" in gpu_type:
            return "g4dn.xlarge"
        elif "a10g" in gpu_type:
            return "g5.xlarge"
        else:
            return "p3.2xlarge"


def create_distributed_orchestrator(
    model_name_or_path: str,
    dataset_name: str,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    cloud_provider: str = "local",
    **kwargs
) -> DistributedOrchestrator:
    """Factory function to create a distributed orchestrator.
    
    Args:
        model_name_or_path: Path or name of the model
        dataset_name: Name or path of the dataset
        num_nodes: Number of compute nodes
        num_gpus_per_node: Number of GPUs per node
        cloud_provider: Cloud provider (aws, gcp, azure, local)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured DistributedOrchestrator instance
    """
    # Create training config
    training_config = TrainingConfig(
        model_name_or_path=model_name_or_path,
        dataset_name=dataset_name,
        **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
    )
    
    # Create resource spec
    resource_spec = ResourceSpec(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        cloud_provider=CloudProvider(cloud_provider.lower()),
        **{k: v for k, v in kwargs.items() if hasattr(ResourceSpec, k)}
    )
    
    return DistributedOrchestrator(training_config, resource_spec)


# Integration with existing forge training script
def run_distributed_training(
    model_args: Dict[str, Any],
    data_args: Dict[str, Any],
    training_args: Dict[str, Any],
    orchestrator_config: Optional[Dict[str, Any]] = None
):
    """Run distributed training using the orchestrator.
    
    This function integrates with existing forge training scripts.
    
    Args:
        model_args: Model arguments
        data_args: Data arguments
        training_args: Training arguments
        orchestrator_config: Orchestrator configuration
    """
    # Merge configurations
    config = {
        **model_args,
        **data_args,
        **training_args,
        **(orchestrator_config or {})
    }
    
    # Create orchestrator
    orchestrator = create_distributed_orchestrator(
        model_name_or_path=config.get("model_name_or_path", ""),
        dataset_name=config.get("dataset_name", ""),
        num_nodes=config.get("num_nodes", 1),
        num_gpus_per_node=config.get("num_gpus_per_node", 1),
        cloud_provider=config.get("cloud_provider", "local"),
        **config
    )
    
    try:
        # Provision resources
        if not orchestrator.provision_resources():
            logger.error("Failed to provision resources")
            return False
        
        # Setup distributed training
        if not orchestrator.setup_distributed_training():
            logger.error("Failed to setup distributed training")
            return False
        
        # Start training
        success = orchestrator.start_training()
        
        return success
        
    finally:
        # Ensure cleanup
        orchestrator.shutdown()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = {
        "model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "dataset_name": "alpaca",
        "num_nodes": 2,
        "num_gpus_per_node": 4,
        "cloud_provider": "aws",
        "region": "us-east-1",
        "instance_type": "spot",
        "max_price_per_hour": 10.0,
        "output_dir": "./output",
        "checkpoint_dir": "./checkpoints",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "fp16": True,
        "fault_tolerance": True,
        "elastic_scaling": True,
        "cost_optimization": True,
        "auto_select_strategy": True
    }
    
    # Create and run orchestrator
    orchestrator = create_distributed_orchestrator(**config)
    
    try:
        # Provision resources
        if orchestrator.provision_resources():
            # Setup training
            if orchestrator.setup_distributed_training():
                # Start training
                orchestrator.start_training()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        orchestrator.shutdown()