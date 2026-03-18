"""
Experiment tracking module for forge with reproducibility guarantees.

This module provides complete experiment tracking that captures hyperparameters,
code versions, data samples, and environment details. Includes automatic Docker
image generation for exact reproduction and comparison tools for A/B testing.
"""

import os
import sys
import json
import hashlib
import subprocess
import platform
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings

# Optional dependencies with graceful fallback
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

import torch
import transformers
from datasets import Dataset

from forge.hparams import TrainingArguments


class TrackerBackend(Enum):
    """Supported experiment tracking backends."""
    MLFLOW = "mlflow"
    WANDB = "wandb"
    LOCAL = "local"
    NONE = "none"


@dataclass
class EnvironmentInfo:
    """Container for environment information."""
    python_version: str
    platform: str
    platform_release: str
    platform_version: str
    architecture: str
    processor: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: Optional[int]
    gpu_names: Optional[List[str]]
    torch_version: str
    transformers_version: str
    forge_version: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    pip_packages: Dict[str, str]
    environment_variables: Dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for tracking."""
    experiment_name: str
    run_name: str
    training_args: Dict[str, Any]
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    environment: EnvironmentInfo
    code_version: str
    data_samples: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class ExperimentTracker:
    """
    Complete experiment tracking system with reproducibility guarantees.
    
    Captures all hyperparameters, code versions, data samples, and environment
    details. Includes automatic Docker image generation for exact reproduction
    and comparison tools for A/B testing different configurations.
    """
    
    def __init__(
        self,
        backend: Union[str, TrackerBackend] = TrackerBackend.LOCAL,
        experiment_name: str = "forge_experiment",
        run_name: Optional[str] = None,
        output_dir: str = "./experiments",
        track_data_samples: int = 10,
        auto_generate_docker: bool = True,
        **kwargs
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            backend: Tracking backend to use (mlflow, wandb, local, none)
            experiment_name: Name of the experiment
            run_name: Name of this specific run (auto-generated if None)
            output_dir: Directory to store local tracking data
            track_data_samples: Number of data samples to track
            auto_generate_docker: Whether to auto-generate Dockerfile
            **kwargs: Additional backend-specific arguments
        """
        self.backend = TrackerBackend(backend) if isinstance(backend, str) else backend
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir)
        self.track_data_samples = track_data_samples
        self.auto_generate_docker = auto_generate_docker
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_dir / self.experiment_name / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking backend
        self._init_backend(**kwargs)
        
        # Capture environment
        self.environment = self._capture_environment()
        
        # Track git information
        self.git_info = self._get_git_info()
        
        # Store initial config
        self.config = None
        self.active = False
        
    def _init_backend(self, **kwargs):
        """Initialize the selected tracking backend."""
        self.backend_client = None
        
        if self.backend == TrackerBackend.MLFLOW:
            if not MLFLOW_AVAILABLE:
                warnings.warn("MLflow not installed. Falling back to local tracking.")
                self.backend = TrackerBackend.LOCAL
            else:
                mlflow.set_experiment(self.experiment_name)
                self.backend_client = mlflow
                
        elif self.backend == TrackerBackend.WANDB:
            if not WANDB_AVAILABLE:
                warnings.warn("Weights & Biases not installed. Falling back to local tracking.")
                self.backend = TrackerBackend.LOCAL
            else:
                self.backend_client = wandb.init(
                    project=self.experiment_name,
                    name=self.run_name,
                    **kwargs
                )
                
    def _capture_environment(self) -> EnvironmentInfo:
        """Capture complete environment information."""
        # Get CUDA info
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_count > 0 else []
        
        # Get pip packages
        try:
            import pkg_resources
            pip_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        except:
            pip_packages = {}
        
        # Get environment variables (filter sensitive ones)
        sensitive_keys = {'KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL'}
        env_vars = {}
        for key, value in os.environ.items():
            if not any(s in key.upper() for s in sensitive_keys):
                env_vars[key] = value
        
        return EnvironmentInfo(
            python_version=sys.version,
            platform=platform.system(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor(),
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            forge_version=self._get_forge_version(),
            git_commit=self.git_info.get('commit'),
            git_branch=self.git_info.get('branch'),
            pip_packages=pip_packages,
            environment_variables=env_vars
        )
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        git_info = {}
        
        try:
            # Get git commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                git_info['commit'] = result.stdout.strip()
                
            # Get git branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
            # Get git status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                git_info['dirty'] = bool(result.stdout.strip())
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return git_info
    
    def _get_forge_version(self) -> str:
        """Get forge version."""
        try:
            from forge import __version__
            return __version__
        except:
            return "unknown"
    
    def start_run(
        self,
        training_args: TrainingArguments,
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Start tracking an experiment run.
        
        Args:
            training_args: Training arguments/configuration
            model_config: Model configuration dictionary
            data_config: Data configuration dictionary
            dataset: Dataset for sampling data examples
            notes: Additional notes about the run
            tags: Tags for organizing experiments
        """
        if self.active:
            warnings.warn("Run already active. Ending previous run.")
            self.end_run()
        
        # Convert training args to dict
        training_args_dict = self._training_args_to_dict(training_args)
        
        # Sample data if provided
        data_samples = None
        if dataset is not None and self.track_data_samples > 0:
            data_samples = self._sample_dataset(dataset)
        
        # Create experiment config
        self.config = ExperimentConfig(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            training_args=training_args_dict,
            model_config=model_config or {},
            data_config=data_config or {},
            environment=self.environment,
            code_version=self.git_info.get('commit', 'unknown'),
            data_samples=data_samples,
            notes=notes,
            tags=tags
        )
        
        # Start backend tracking
        if self.backend == TrackerBackend.MLFLOW and MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(training_args_dict)
            if model_config:
                mlflow.log_params({f"model_{k}": v for k, v in model_config.items()})
            if data_config:
                mlflow.log_params({f"data_{k}": v for k, v in data_config.items()})
            if tags:
                mlflow.set_tags({f"tag_{i}": tag for i, tag in enumerate(tags)})
                
        elif self.backend == TrackerBackend.WANDB and WANDB_AVAILABLE:
            wandb.config.update({
                "training_args": training_args_dict,
                "model_config": model_config,
                "data_config": data_config
            })
            if tags:
                wandb.run.tags = tags
        
        # Save config locally
        self._save_config()
        
        # Generate Dockerfile if enabled
        if self.auto_generate_docker:
            self.generate_dockerfile()
        
        self.active = True
        
    def _training_args_to_dict(self, args: TrainingArguments) -> Dict[str, Any]:
        """Convert TrainingArguments to dictionary."""
        args_dict = {}
        for key in dir(args):
            if not key.startswith('_'):
                value = getattr(args, key)
                if not callable(value):
                    try:
                        # Try to serialize the value
                        json.dumps(value)
                        args_dict[key] = value
                    except (TypeError, ValueError):
                        # Convert non-serializable types to string
                        args_dict[key] = str(value)
        return args_dict
    
    def _sample_dataset(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Sample data points from dataset for tracking."""
        samples = []
        sample_size = min(self.track_data_samples, len(dataset))
        
        if sample_size > 0:
            indices = torch.randperm(len(dataset))[:sample_size].tolist()
            for idx in indices:
                sample = dataset[idx]
                # Convert tensors to lists for JSON serialization
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.tolist()
                samples.append(sample)
        
        return samples
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if not self.active:
            warnings.warn("No active run. Call start_run() first.")
            return
        
        # Log to backend
        if self.backend == TrackerBackend.MLFLOW and MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)
        elif self.backend == TrackerBackend.WANDB and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
        
        # Save locally
        metrics_file = self.run_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            entry = {"step": step, "metrics": metrics, "timestamp": datetime.now().isoformat()}
            f.write(json.dumps(entry) + '\n')
    
    def log_artifact(self, artifact_path: Union[str, Path], artifact_name: Optional[str] = None):
        """
        Log an artifact (file or directory) to the tracking system.
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Optional name for the artifact
        """
        if not self.active:
            warnings.warn("No active run. Call start_run() first.")
            return
        
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            warnings.warn(f"Artifact not found: {artifact_path}")
            return
        
        # Copy to run directory
        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        dest_name = artifact_name or artifact_path.name
        dest_path = artifacts_dir / dest_name
        
        if artifact_path.is_file():
            shutil.copy2(artifact_path, dest_path)
        else:
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        
        # Log to backend
        if self.backend == TrackerBackend.MLFLOW and MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(artifact_path), artifact_name)
        elif self.backend == TrackerBackend.WANDB and WANDB_AVAILABLE:
            wandb.save(str(artifact_path), base_path=str(artifact_path.parent))
    
    def _save_config(self):
        """Save experiment configuration to disk."""
        if self.config is None:
            return
        
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Also save environment info separately
        env_file = self.run_dir / "environment.json"
        with open(env_file, 'w') as f:
            json.dump(asdict(self.environment), f, indent=2, default=str)
    
    def generate_dockerfile(self, base_image: str = "python:3.10-slim") -> Path:
        """
        Generate a Dockerfile for reproducing the experiment environment.
        
        Args:
            base_image: Base Docker image to use
            
        Returns:
            Path to the generated Dockerfile
        """
        if self.config is None:
            raise ValueError("No experiment configuration available. Call start_run() first.")
        
        dockerfile_content = f"""# Auto-generated Dockerfile for forge experiment
# Experiment: {self.experiment_name}
# Run: {self.run_name}
# Generated: {datetime.now().isoformat()}

FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "train.py"]
"""
        
        # Generate requirements.txt from pip packages
        requirements_content = self._generate_requirements()
        
        # Write Dockerfile
        dockerfile_path = self.run_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Write requirements.txt
        requirements_path = self.run_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Write docker-compose.yml for easy reproduction
        self._generate_docker_compose()
        
        return dockerfile_path
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt from current environment."""
        requirements = []
        
        # Add key packages with versions
        key_packages = [
            'torch',
            'transformers',
            'datasets',
            'accelerate',
            'peft',
            'trl',
            'bitsandbytes',
            'scipy',
            'tensorboard',
            'deepspeed'
        ]
        
        for package in key_packages:
            if package in self.environment.pip_packages:
                requirements.append(f"{package}=={self.environment.pip_packages[package]}")
        
        # Add other packages from environment
        for package, version in self.environment.pip_packages.items():
            if package not in key_packages and not package.startswith('forge'):
                requirements.append(f"{package}=={version}")
        
        return '\n'.join(sorted(requirements))
    
    def _generate_docker_compose(self):
        """Generate docker-compose.yml for easy experiment reproduction."""
        compose_content = f"""version: '3.8'

services:
  training:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      python train.py
      --model_name_or_path {self.config.training_args.get('model_name_or_path', 'meta-llama/Llama-2-7b-hf')}
      --dataset_dir /app/data
      --output_dir /app/output
      --experiment_name {self.experiment_name}
      --run_name {self.run_name}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        compose_path = self.run_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current experiment run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.active:
            return
        
        # End backend tracking
        if self.backend == TrackerBackend.MLFLOW and MLFLOW_AVAILABLE:
            mlflow.end_run(status=status)
        elif self.backend == TrackerBackend.WANDB and WANDB_AVAILABLE:
            wandb.finish(exit_code=0 if status == "FINISHED" else 1)
        
        # Save final config
        self._save_config()
        
        # Generate summary report
        self._generate_summary_report()
        
        self.active = False
    
    def _generate_summary_report(self):
        """Generate a summary report of the experiment run."""
        if self.config is None:
            return
        
        summary = {
            "experiment": self.experiment_name,
            "run": self.run_name,
            "status": "completed",
            "start_time": self.environment.timestamp,
            "end_time": datetime.now().isoformat(),
            "git_commit": self.git_info.get('commit'),
            "git_branch": self.git_info.get('branch'),
            "git_dirty": self.git_info.get('dirty', False),
            "training_args": self.config.training_args,
            "model_config": self.config.model_config,
            "data_config": self.config.data_config,
            "environment": {
                "python": self.environment.python_version,
                "platform": self.environment.platform,
                "cuda": self.environment.cuda_version,
                "gpus": self.environment.gpu_names,
                "torch": self.environment.torch_version,
                "transformers": self.environment.transformers_version
            }
        }
        
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiment runs.
        
        Args:
            run_ids: List of run identifiers to compare
            
        Returns:
            Comparison results dictionary
        """
        comparison = {
            "runs": [],
            "differences": {},
            "summary": {}
        }
        
        runs_data = []
        
        for run_id in run_ids:
            run_dir = self.output_dir / self.experiment_name / run_id
            config_file = run_dir / "config.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    run_config = json.load(f)
                    runs_data.append(run_config)
                    comparison["runs"].append({
                        "run_id": run_id,
                        "config": run_config
                    })
        
        if len(runs_data) < 2:
            return comparison
        
        # Find differences in training args
        all_keys = set()
        for run in runs_data:
            all_keys.update(run.get("training_args", {}).keys())
        
        for key in all_keys:
            values = []
            for run in runs_data:
                values.append(run.get("training_args", {}).get(key))
            
            # Check if all values are the same
            if len(set(str(v) for v in values)) > 1:
                comparison["differences"][key] = values
        
        # Compare metrics if available
        metrics_comparison = self._compare_metrics(run_ids)
        comparison["metrics_comparison"] = metrics_comparison
        
        return comparison
    
    def _compare_metrics(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across runs."""
        metrics_data = {}
        
        for run_id in run_ids:
            metrics_file = self.output_dir / self.experiment_name / run_id / "metrics.jsonl"
            
            if metrics_file.exists():
                run_metrics = []
                with open(metrics_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            run_metrics.append(json.loads(line))
                
                metrics_data[run_id] = run_metrics
        
        # Analyze metrics
        analysis = {
            "final_metrics": {},
            "best_metrics": {},
            "convergence": {}
        }
        
        for run_id, metrics in metrics_data.items():
            if metrics:
                # Get final metrics
                final = metrics[-1].get("metrics", {})
                analysis["final_metrics"][run_id] = final
                
                # Find best metrics (assuming higher is better for most)
                for metric_name in final.keys():
                    values = [m["metrics"].get(metric_name) for m in metrics if metric_name in m["metrics"]]
                    if values:
                        best_value = max(values) if "loss" not in metric_name.lower() else min(values)
                        if metric_name not in analysis["best_metrics"]:
                            analysis["best_metrics"][metric_name] = {}
                        analysis["best_metrics"][metric_name][run_id] = best_value
        
        return analysis
    
    def get_reproduction_command(self, run_id: Optional[str] = None) -> str:
        """
        Get command to reproduce an experiment run.
        
        Args:
            run_id: Specific run ID (uses current run if None)
            
        Returns:
            Command string for reproduction
        """
        run_id = run_id or self.run_name
        run_dir = self.output_dir / self.experiment_name / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
        
        # Check if Dockerfile exists
        dockerfile = run_dir / "Dockerfile"
        if dockerfile.exists():
            return f"cd {run_dir} && docker-compose up --build"
        
        # Otherwise return python command
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            training_args = config.get("training_args", {})
            args_str = " ".join([f"--{k} {v}" for k, v in training_args.items() 
                                if not isinstance(v, (list, dict))])
            
            return f"python train.py {args_str}"
        
        return f"# No reproduction command available for run: {run_id}"
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all experiment runs.
        
        Returns:
            List of run information dictionaries
        """
        runs = []
        experiment_dir = self.output_dir / self.experiment_name
        
        if experiment_dir.exists():
            for run_dir in experiment_dir.iterdir():
                if run_dir.is_dir():
                    config_file = run_dir / "config.json"
                    summary_file = run_dir / "summary.json"
                    
                    run_info = {
                        "run_id": run_dir.name,
                        "path": str(run_dir),
                        "has_config": config_file.exists(),
                        "has_summary": summary_file.exists()
                    }
                    
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            run_info.update(summary)
                    
                    runs.append(run_info)
        
        return sorted(runs, key=lambda x: x.get("start_time", ""), reverse=True)


class AutoTracker:
    """
    Automatic experiment tracker that integrates with forge training.
    
    Automatically captures training progress, hyperparameters, and generates
    reproducible artifacts without manual intervention.
    """
    
    def __init__(self, training_args: TrainingArguments, **kwargs):
        """
        Initialize automatic tracker.
        
        Args:
            training_args: Training arguments from forge
            **kwargs: Additional arguments for ExperimentTracker
        """
        self.training_args = training_args
        
        # Determine experiment name from model/dataset
        model_name = getattr(training_args, 'model_name_or_path', 'unknown').split('/')[-1]
        dataset_name = getattr(training_args, 'dataset', 'unknown')
        experiment_name = f"{model_name}_{dataset_name}"
        
        # Initialize tracker
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            output_dir=getattr(training_args, 'output_dir', './experiments'),
            **kwargs
        )
        
        # Hook into training callbacks
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks for automatic tracking."""
        # This would integrate with Trainer callbacks
        # For now, we provide manual methods
        pass
    
    def start(self, **kwargs):
        """Start automatic tracking."""
        self.tracker.start_run(self.training_args, **kwargs)
    
    def log_step(self, logs: Dict[str, float], step: int):
        """Log training step metrics."""
        self.tracker.log_metrics(logs, step=step)
    
    def end(self, **kwargs):
        """End automatic tracking."""
        self.tracker.end_run(**kwargs)


# Convenience functions for easy import
def create_tracker(
    backend: str = "local",
    experiment_name: str = "forge_experiment",
    **kwargs
) -> ExperimentTracker:
    """
    Create an experiment tracker with specified backend.
    
    Args:
        backend: Tracking backend (mlflow, wandb, local, none)
        experiment_name: Name of the experiment
        **kwargs: Additional tracker arguments
        
    Returns:
        Configured ExperimentTracker instance
    """
    return ExperimentTracker(
        backend=backend,
        experiment_name=experiment_name,
        **kwargs
    )


def track_experiment(
    training_args: TrainingArguments,
    backend: str = "local",
    **kwargs
) -> AutoTracker:
    """
    Create an automatic experiment tracker for training.
    
    Args:
        training_args: Training arguments from forge
        backend: Tracking backend
        **kwargs: Additional tracker arguments
        
    Returns:
        Configured AutoTracker instance
    """
    return AutoTracker(training_args, backend=backend, **kwargs)


# Export main classes and functions
__all__ = [
    'ExperimentTracker',
    'AutoTracker',
    'TrackerBackend',
    'EnvironmentInfo',
    'ExperimentConfig',
    'create_tracker',
    'track_experiment'
]