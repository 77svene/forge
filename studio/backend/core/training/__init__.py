# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training submodule - Training backends and trainer classes
"""

from .training import TrainingBackend, TrainingProgress, get_training_backend

# Distributed training imports - conditionally loaded
try:
    import ray
    from ray import train, tune
    from ray.train.torch import TorchTrainer
    from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
    from ray.air.integrations.wandb import WandbLoggerCallback
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Kubernetes imports - conditionally loaded  
try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

__all__ = [
    "TrainingProgress",
    "TrainingBackend",
    "get_training_backend",
    "DistributedTrainingOrchestrator",
    "KubernetesTrainingOperator",
    "get_distributed_training_orchestrator",
]


class DistributedTrainingOrchestrator:
    """Ray-based distributed training orchestrator with fault tolerance and dynamic resource allocation"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.trainer = None
        self.ray_initialized = False
        
    def initialize_ray(self, address="auto", **ray_kwargs):
        """Initialize Ray runtime if not already initialized"""
        if not self.ray_initialized:
            if not ray.is_initialized():
                ray.init(address=address, **ray_kwargs)
            self.ray_initialized = True
            
    def create_trainer(self, train_func, scaling_config=None, checkpoint_config=None):
        """Create Ray TorchTrainer with automatic resource allocation"""
        if not DISTRIBUTED_AVAILABLE:
            raise ImportError("Ray is required for distributed training. Install with: pip install ray[train]")
            
        self.initialize_ray()
        
        # Default scaling configuration
        default_scaling = {
            "num_workers": 2,
            "use_gpu": True,
            "resources_per_worker": {"CPU": 2, "GPU": 1}
        }
        scaling_config = {**default_scaling, **(scaling_config or {})}
        
        # Default checkpoint configuration
        default_checkpoint = {
            "checkpoint_frequency": 10,
            "checkpoint_at_end": True
        }
        checkpoint_config = {**default_checkpoint, **(checkpoint_config or {})}
        
        # Create trainer
        self.trainer = TorchTrainer(
            train_func,
            scaling_config=ScalingConfig(**scaling_config),
            run_config=RunConfig(
                checkpoint_config=CheckpointConfig(**checkpoint_config),
                callbacks=[WandbLoggerCallback(project="forge-distributed")] 
                if self.config.get("use_wandb") else []
            )
        )
        
        return self.trainer
    
    def train(self, train_func, scaling_config=None, checkpoint_config=None):
        """Execute distributed training with fault tolerance"""
        if self.trainer is None:
            self.create_trainer(train_func, scaling_config, checkpoint_config)
            
        # Start training with automatic recovery
        try:
            results = self.trainer.fit()
            return results
        except Exception as e:
            print(f"Training failed with error: {e}. Attempting recovery...")
            # Ray automatically handles checkpoint recovery
            self.trainer.restore_latest_checkpoint()
            results = self.trainer.fit()
            return results
    
    def dynamic_resource_allocation(self, min_workers=1, max_workers=8):
        """Dynamically adjust resources based on workload"""
        if not self.ray_initialized:
            self.initialize_ray()
            
        # Monitor cluster resources
        cluster_resources = ray.cluster_resources()
        available_gpus = cluster_resources.get("GPU", 0)
        available_cpus = cluster_resources.get("CPU", 0)
        
        # Calculate optimal workers
        optimal_workers = min(
            max_workers,
            int(available_gpus),
            int(available_cpus / 2)  # 2 CPUs per worker
        )
        optimal_workers = max(min_workers, optimal_workers)
        
        return optimal_workers
    
    def scale_training(self, num_workers):
        """Scale training to specified number of workers"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_trainer first.")
            
        # Update scaling config
        new_scaling = self.trainer.scaling_config.copy()
        new_scaling["num_workers"] = num_workers
        self.trainer.scaling_config = ScalingConfig(**new_scaling)
        
        return self.trainer
    
    def get_checkpoint_path(self, run_id=None):
        """Get latest checkpoint path for recovery"""
        if run_id is None:
            # Get latest run
            runs = ray.train.get_experiment_runs()
            if not runs:
                return None
            run_id = runs[-1].run_id
            
        checkpoints = ray.train.get_checkpoints(run_id)
        if not checkpoints:
            return None
            
        return checkpoints[-1].path
    
    def cleanup(self):
        """Clean up Ray resources"""
        if ray.is_initialized():
            ray.shutdown()
        self.ray_initialized = False


class KubernetesTrainingOperator:
    """Kubernetes operator for cloud-based distributed training deployments"""
    
    def __init__(self, namespace="forge-training"):
        self.namespace = namespace
        self.k8s_available = K8S_AVAILABLE
        
        if self.k8s_available:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                try:
                    config.load_kube_config()
                except config.ConfigException:
                    self.k8s_available = False
                    
        if self.k8s_available:
            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            self.custom_api = client.CustomObjectsApi()
    
    def create_training_job(self, job_name, image, command, 
                          num_workers=2, gpu_per_worker=1,
                          cpu_per_worker=4, memory_per_worker="16Gi"):
        """Create Kubernetes training job with specified resources"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes client not available. Install with: pip install kubernetes")
            
        # Define container
        container = client.V1Container(
            name="trainer",
            image=image,
            command=command,
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": str(cpu_per_worker),
                    "memory": memory_per_worker,
                    "nvidia.com/gpu": str(gpu_per_worker)
                },
                limits={
                    "cpu": str(cpu_per_worker * 2),
                    "memory": memory_per_worker,
                    "nvidia.com/gpu": str(gpu_per_worker)
                }
            ),
            env=[
                client.V1EnvVar(name="NCCL_DEBUG", value="INFO"),
                client.V1EnvVar(name="PYTHONUNBUFFERED", value="1")
            ]
        )
        
        # Define pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": job_name}),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="OnFailure"
            )
        )
        
        # Define job spec
        job_spec = client.V1JobSpec(
            template=template,
            parallelism=num_workers,
            completions=num_workers,
            backoff_limit=4
        )
        
        # Create job
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name=job_name),
            spec=job_spec
        )
        
        try:
            api_response = self.core_api.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            return api_response
        except ApiException as e:
            print(f"Exception when creating job: {e}")
            return None
    
    def monitor_training_job(self, job_name, timeout=3600):
        """Monitor training job status with timeout"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes client not available")
            
        w = watch.Watch()
        start_time = time.time()
        
        for event in w.stream(self.core_api.list_namespaced_pod, 
                             namespace=self.namespace,
                             label_selector=f"app={job_name}",
                             timeout_seconds=timeout):
            pod = event['object']
            print(f"Pod {pod.metadata.name} status: {pod.status.phase}")
            
            if pod.status.phase == "Succeeded":
                w.stop()
                return "Succeeded"
            elif pod.status.phase == "Failed":
                w.stop()
                return "Failed"
                
            if time.time() - start_time > timeout:
                w.stop()
                return "Timeout"
                
        return "Unknown"
    
    def delete_training_job(self, job_name):
        """Delete training job and associated resources"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes client not available")
            
        try:
            # Delete job
            api_response = self.core_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5
                )
            )
            
            # Delete associated pods
            self.core_api.delete_collection_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={job_name}"
            )
            
            return api_response
        except ApiException as e:
            print(f"Exception when deleting job: {e}")
            return None
    
    def create_ray_cluster(self, cluster_name, min_workers=2, max_workers=8):
        """Create Ray cluster on Kubernetes"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes client not available")
            
        # Ray cluster CRD
        ray_cluster = {
            "apiVersion": "ray.io/v1alpha1",
            "kind": "RayCluster",
            "metadata": {
                "name": cluster_name,
                "namespace": self.namespace
            },
            "spec": {
                "maxWorkers": max_workers,
                "headGroupSpec": {
                    "serviceType": "ClusterIP",
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "num-cpus": "2"
                    },
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "ray-head",
                                "image": "rayproject/ray:latest",
                                "ports": [
                                    {"containerPort": 6379, "name": "gcs-server"},
                                    {"containerPort": 8265, "name": "dashboard"},
                                    {"containerPort": 10001, "name": "client"}
                                ],
                                "resources": {
                                    "requests": {"cpu": "2", "memory": "4Gi"},
                                    "limits": {"cpu": "4", "memory": "8Gi"}
                                }
                            }]
                        }
                    }
                },
                "workerGroupSpecs": [{
                    "replicas": min_workers,
                    "minReplicas": min_workers,
                    "maxReplicas": max_workers,
                    "groupName": "worker-group",
                    "rayStartParams": {"num-cpus": "4"},
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "ray-worker",
                                "image": "rayproject/ray:latest",
                                "resources": {
                                    "requests": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
                                    "limits": {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": "1"}
                                }
                            }]
                        }
                    }
                }]
            }
        }
        
        try:
            api_response = self.custom_api.create_namespaced_custom_object(
                group="ray.io",
                version="v1alpha1",
                namespace=self.namespace,
                plural="rayclusters",
                body=ray_cluster
            )
            return api_response
        except ApiException as e:
            print(f"Exception when creating Ray cluster: {e}")
            return None


def get_distributed_training_orchestrator(config=None):
    """Factory function to get distributed training orchestrator"""
    return DistributedTrainingOrchestrator(config)


# Update __all__ with new distributed components
__all__.extend([
    "DistributedTrainingOrchestrator",
    "KubernetesTrainingOperator",
    "get_distributed_training_orchestrator",
    "DISTRIBUTED_AVAILABLE",
    "K8S_AVAILABLE"
])