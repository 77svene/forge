"""
Experiment Tracking with Reproducibility Guarantees for forge.

Complete experiment tracking system that captures all hyperparameters, code versions,
data samples, and environment details. Includes automatic Docker image generation
for exact reproduction and comparison tools for A/B testing different configurations.
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum

import yaml
import torch
import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

class TrackerBackend(Enum):
    """Supported tracking backends."""
    MLFLOW = "mlflow"
    WANDB = "wandb"
    LOCAL = "local"
    NONE = "none"

@dataclass
class EnvironmentInfo:
    """Captures environment details for reproducibility."""
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: platform.platform())
    architecture: str = field(default_factory=lambda: platform.machine())
    processor: str = field(default_factory=lambda: platform.processor())
    cuda_available: bool = field(default_factory=lambda: torch.cuda.is_available())
    cuda_version: Optional[str] = field(default_factory=lambda: torch.version.cuda if torch.cuda.is_available() else None)
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    gpu_count: int = field(default_factory=lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0)
    gpu_names: List[str] = field(default_factory=lambda: [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [])
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 0)
    hostname: str = field(default_factory=lambda: platform.node())
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_commit: Optional[str] = field(default=None)
    git_branch: Optional[str] = field(default=None)
    git_remote: Optional[str] = field(default=None)
    pip_packages: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Capture git information if available."""
        try:
            self.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
            
            self.git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
            
            self.git_remote = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git information not available")
        
        # Capture key package versions
        try:
            import importlib.metadata
            packages_to_check = [
                "transformers", "peft", "accelerate", "datasets",
                "trl", "bitsandbytes", "scipy", "tensorboard",
                "deepspeed", "flash-attn"
            ]
            for package in packages_to_check:
                try:
                    self.pip_packages[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
        except ImportError:
            pass

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    project_name: str = "forge"
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracker_backend: TrackerBackend = TrackerBackend.LOCAL
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    log_system_info: bool = True
    log_git_info: bool = True
    log_code_snapshot: bool = True
    log_data_samples: bool = True
    data_sample_count: int = 10
    generate_dockerfile: bool = True
    output_dir: str = "./experiments"

class ReproducibilityTracker:
    """
    Main class for experiment tracking with reproducibility guarantees.
    
    Captures all aspects of an experiment including hyperparameters,
    code versions, data samples, and environment details.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the reproducibility tracker.
        
        Args:
            config: Configuration for experiment tracking
        """
        self.config = config
        self.environment_info = EnvironmentInfo()
        self.experiment_id = self._generate_experiment_id()
        self.run_id = self._generate_run_id()
        self.output_path = Path(config.output_dir) / self.experiment_id / self.run_id
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking backend
        self.tracker = None
        self._init_tracker_backend()
        
        # Store all tracked data
        self.data = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "config": asdict(config),
            "environment": asdict(self.environment_info),
            "hyperparameters": {},
            "metrics": {},
            "data_samples": [],
            "code_snapshot": None,
            "dockerfile": None,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Initialized reproducibility tracker for experiment {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on project and experiment name."""
        base = f"{self.config.project_name}"
        if self.config.experiment_name:
            base += f"_{self.config.experiment_name}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{timestamp}"
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        if self.config.run_name:
            return self.config.run_name
        return f"run_{datetime.now().strftime('%H%M%S')}"
    
    def _init_tracker_backend(self):
        """Initialize the selected tracking backend."""
        if self.config.tracker_backend == TrackerBackend.MLFLOW:
            try:
                import mlflow
                if self.config.tracking_uri:
                    mlflow.set_tracking_uri(self.config.tracking_uri)
                mlflow.set_experiment(self.config.project_name)
                self.tracker = mlflow
                logger.info("Initialized MLflow tracker")
            except ImportError:
                logger.warning("MLflow not installed, falling back to local tracking")
                self.config.tracker_backend = TrackerBackend.LOCAL
        
        elif self.config.tracker_backend == TrackerBackend.WANDB:
            try:
                import wandb
                wandb.init(
                    project=self.config.project_name,
                    name=self.config.run_name,
                    config=self.config.tags,
                    dir=str(self.output_path)
                )
                self.tracker = wandb
                logger.info("Initialized Weights & Biases tracker")
            except ImportError:
                logger.warning("W&B not installed, falling back to local tracking")
                self.config.tracker_backend = TrackerBackend.LOCAL
        
        if self.config.tracker_backend == TrackerBackend.LOCAL:
            logger.info("Using local file-based tracking")
    
    def start_run(self, hyperparameters: Dict[str, Any]):
        """
        Start tracking a new experiment run.
        
        Args:
            hyperparameters: Dictionary of hyperparameters for this run
        """
        self.data["hyperparameters"] = hyperparameters
        
        # Log to tracking backend
        if self.config.tracker_backend == TrackerBackend.MLFLOW:
            self.tracker.start_run(run_name=self.run_id)
            self.tracker.log_params(hyperparameters)
            self.tracker.set_tags(self.config.tags)
        
        elif self.config.tracker_backend == TrackerBackend.WANDB:
            self.tracker.config.update(hyperparameters)
        
        # Save initial state
        self._save_state()
        
        logger.info(f"Started tracking run {self.run_id}")
    
    def log_metric(self, key: str, value: Union[float, int], step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for time series
        """
        if step is not None:
            if key not in self.data["metrics"]:
                self.data["metrics"][key] = []
            self.data["metrics"][key].append({"step": step, "value": value, "timestamp": datetime.now().isoformat()})
        else:
            self.data["metrics"][key] = value
        
        # Log to tracking backend
        if self.config.tracker_backend == TrackerBackend.MLFLOW:
            self.tracker.log_metric(key, value, step=step)
        
        elif self.config.tracker_backend == TrackerBackend.WANDB:
            self.tracker.log({key: value}, step=step)
        
        self._save_state()
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time series
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_data_samples(self, dataset: Union[Dataset, DatasetDict], 
                         split: str = "train", 
                         num_samples: Optional[int] = None):
        """
        Log samples from the dataset for reproducibility.
        
        Args:
            dataset: Hugging Face dataset
            split: Dataset split to sample from
            num_samples: Number of samples to log (defaults to config)
        """
        if not self.config.log_data_samples:
            return
        
        if num_samples is None:
            num_samples = self.config.data_sample_count
        
        try:
            if isinstance(dataset, DatasetDict):
                if split not in dataset:
                    logger.warning(f"Split {split} not found in dataset")
                    return
                ds = dataset[split]
            else:
                ds = dataset
            
            # Get random samples
            if len(ds) > num_samples:
                indices = np.random.choice(len(ds), num_samples, replace=False)
                samples = ds.select(indices)
            else:
                samples = ds[:num_samples]
            
            # Convert to serializable format
            sample_data = []
            for i in range(len(samples["input_ids"] if "input_ids" in samples else samples["text"])):
                sample = {key: samples[key][i] for key in samples.keys()}
                # Convert tensors to lists for JSON serialization
                for key in sample:
                    if hasattr(sample[key], 'tolist'):
                        sample[key] = sample[key].tolist()
                sample_data.append(sample)
            
            self.data["data_samples"] = {
                "split": split,
                "count": len(sample_data),
                "samples": sample_data,
                "dataset_hash": self._compute_dataset_hash(ds),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Logged {len(sample_data)} data samples from {split} split")
            
        except Exception as e:
            logger.warning(f"Failed to log data samples: {e}")
    
    def _compute_dataset_hash(self, dataset: Dataset) -> str:
        """Compute a hash of the dataset for versioning."""
        try:
            # Create a string representation of dataset features and size
            features_str = str(dataset.features)
            size_str = str(len(dataset))
            hash_input = f"{features_str}_{size_str}".encode()
            return hashlib.md5(hash_input).hexdigest()[:12]
        except:
            return "unknown"
    
    def log_code_snapshot(self, source_dirs: List[str] = None):
        """
        Create a snapshot of the code for reproducibility.
        
        Args:
            source_dirs: List of directories to include in snapshot
        """
        if not self.config.log_code_snapshot:
            return
        
        if source_dirs is None:
            # Default to main source directories
            source_dirs = [
                "src/forge",
                "scripts",
                "examples"
            ]
        
        try:
            snapshot_dir = self.output_path / "code_snapshot"
            snapshot_dir.mkdir(exist_ok=True)
            
            # Copy source files
            for source_dir in source_dirs:
                source_path = Path(source_dir)
                if source_path.exists():
                    dest_path = snapshot_dir / source_path.name
                    if source_path.is_dir():
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_path, dest_path)
            
            # Create manifest
            manifest = {
                "timestamp": datetime.now().isoformat(),
                "source_dirs": source_dirs,
                "files": []
            }
            
            for file_path in snapshot_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(snapshot_dir)
                    manifest["files"].append(str(rel_path))
            
            with open(snapshot_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            self.data["code_snapshot"] = {
                "path": str(snapshot_dir),
                "file_count": len(manifest["files"]),
                "timestamp": manifest["timestamp"]
            }
            
            logger.info(f"Created code snapshot with {len(manifest['files'])} files")
            
        except Exception as e:
            logger.warning(f"Failed to create code snapshot: {e}")
    
    def generate_dockerfile(self, base_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"):
        """
        Generate a Dockerfile for reproducing the experiment environment.
        
        Args:
            base_image: Base Docker image to use
        """
        if not self.config.generate_dockerfile:
            return
        
        try:
            dockerfile_content = self._create_dockerfile_content(base_image)
            dockerfile_path = self.output_path / "Dockerfile"
            
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            # Also save requirements.txt
            requirements = self._generate_requirements()
            requirements_path = self.output_path / "requirements.txt"
            
            with open(requirements_path, "w") as f:
                f.write(requirements)
            
            self.data["dockerfile"] = {
                "path": str(dockerfile_path),
                "base_image": base_image,
                "requirements_path": str(requirements_path),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated Dockerfile at {dockerfile_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate Dockerfile: {e}")
    
    def _create_dockerfile_content(self, base_image: str) -> str:
        """Create Dockerfile content based on captured environment."""
        env = self.environment_info
        
        dockerfile = f"""# Auto-generated Dockerfile for forge experiment reproduction
# Experiment: {self.experiment_id}
# Run: {self.run_id}
# Generated: {datetime.now().isoformat()}

FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    curl \\
    wget \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment configuration
COPY experiment_config.json .

# Copy code snapshot if available
"""
        
        if self.data.get("code_snapshot"):
            dockerfile += "COPY code_snapshot/ ./code_snapshot/\n"
        
        dockerfile += """
# Set entrypoint
ENTRYPOINT ["python"]
CMD ["--help"]
"""
        
        return dockerfile
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt from captured package versions."""
        requirements = []
        
        # Add core packages with versions from environment
        core_packages = {
            "torch": self.environment_info.pytorch_version,
            "transformers": None,
            "datasets": None,
            "accelerate": None,
            "peft": None,
            "trl": None,
            "bitsandbytes": None,
        }
        
        # Update with actual versions from pip_packages
        for package, version in self.environment_info.pip_packages.items():
            if package in core_packages:
                core_packages[package] = version
        
        # Add packages with versions
        for package, version in core_packages.items():
            if version:
                requirements.append(f"{package}=={version}")
            else:
                requirements.append(package)
        
        # Add additional packages from pip_packages that aren't in core
        for package, version in self.environment_info.pip_packages.items():
            if package not in core_packages:
                requirements.append(f"{package}=={version}")
        
        # Add forge itself
        requirements.append("-e .")  # Assuming editable install
        
        return "\n".join(sorted(requirements))
    
    def save_experiment_config(self, config: Dict[str, Any]):
        """
        Save the full experiment configuration.
        
        Args:
            config: Complete experiment configuration dictionary
        """
        config_path = self.output_path / "experiment_config.json"
        
        # Make config JSON serializable
        serializable_config = self._make_serializable(config)
        
        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2, default=str)
        
        self.data["experiment_config_path"] = str(config_path)
        logger.info(f"Saved experiment configuration to {config_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make an object JSON serializable."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)
    
    def _save_state(self):
        """Save current tracking state to disk."""
        state_path = self.output_path / "tracking_state.json"
        
        with open(state_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current tracking run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        self.data["status"] = status
        self.data["ended_at"] = datetime.now().isoformat()
        
        # End tracking backend run
        if self.config.tracker_backend == TrackerBackend.MLFLOW:
            self.tracker.end_run(status=status)
        elif self.config.tracker_backend == TrackerBackend.WANDB:
            self.tracker.finish()
        
        # Save final state
        self._save_state()
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info(f"Ended tracking run {self.run_id} with status {status}")
    
    def _generate_summary_report(self):
        """Generate a summary report of the experiment."""
        report_path = self.output_path / "summary_report.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Experiment Summary Report\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n")
            f.write(f"**Run ID:** {self.run_id}\n")
            f.write(f"**Status:** {self.data.get('status', 'UNKNOWN')}\n")
            f.write(f"**Created:** {self.data.get('created_at', 'UNKNOWN')}\n")
            f.write(f"**Ended:** {self.data.get('ended_at', 'UNKNOWN')}\n\n")
            
            f.write("## Environment\n\n")
            f.write(f"- **Python:** {self.environment_info.python_version}\n")
            f.write(f"- **Platform:** {self.environment_info.platform}\n")
            f.write(f"- **PyTorch:** {self.environment_info.pytorch_version}\n")
            f.write(f"- **CUDA:** {self.environment_info.cuda_version or 'Not available'}\n")
            f.write(f"- **GPUs:** {self.environment_info.gpu_count}\n\n")
            
            if self.data.get("hyperparameters"):
                f.write("## Hyperparameters\n\n")
                for key, value in self.data["hyperparameters"].items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            if self.data.get("metrics"):
                f.write("## Final Metrics\n\n")
                for key, value in self.data["metrics"].items():
                    if isinstance(value, list) and value:
                        final_value = value[-1].get("value", value[-1])
                    else:
                        final_value = value
                    f.write(f"- **{key}:** {final_value}\n")
                f.write("\n")
            
            f.write("## Reproduction Instructions\n\n")
            f.write("1. Build Docker image: `docker build -t experiment-reproduction .`\n")
            f.write("2. Run container: `docker run -v $(pwd)/data:/app/data experiment-reproduction`\n")
            f.write("3. Or use the provided experiment configuration directly.\n\n")
            
            f.write("## Files Generated\n\n")
            for file_path in self.output_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.output_path)
                    f.write(f"- `{rel_path}`\n")
        
        logger.info(f"Generated summary report at {report_path}")
    
    @classmethod
    def compare_experiments(cls, experiment_dirs: List[str], 
                           metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple experiments for A/B testing.
        
        Args:
            experiment_dirs: List of experiment directories to compare
            metrics: Specific metrics to compare (None for all)
        
        Returns:
            Comparison report dictionary
        """
        experiments = []
        
        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            state_file = exp_path / "tracking_state.json"
            
            if state_file.exists():
                with open(state_file, "r") as f:
                    exp_data = json.load(f)
                    experiments.append({
                        "path": str(exp_path),
                        "data": exp_data
                    })
        
        if not experiments:
            return {"error": "No valid experiments found"}
        
        # Extract metrics for comparison
        comparison = {
            "experiment_count": len(experiments),
            "experiments": [],
            "metric_comparison": {},
            "hyperparameter_comparison": {}
        }
        
        # Collect all metrics and hyperparameters
        all_metrics = set()
        all_hyperparams = set()
        
        for exp in experiments:
            exp_data = exp["data"]
            comparison["experiments"].append({
                "experiment_id": exp_data.get("experiment_id"),
                "run_id": exp_data.get("run_id"),
                "status": exp_data.get("status"),
                "created_at": exp_data.get("created_at")
            })
            
            # Collect metrics
            for metric_name in exp_data.get("metrics", {}).keys():
                all_metrics.add(metric_name)
            
            # Collect hyperparameters
            for param_name in exp_data.get("hyperparameters", {}).keys():
                all_hyperparams.add(param_name)
        
        # Compare metrics
        for metric in all_metrics:
            if metrics and metric not in metrics:
                continue
            
            metric_values = []
            for exp in experiments:
                exp_data = exp["data"]
                metric_data = exp_data.get("metrics", {}).get(metric)
                
                if metric_data:
                    if isinstance(metric_data, list) and metric_data:
                        # Time series metric - get final value
                        final_value = metric_data[-1].get("value", metric_data[-1])
                    else:
                        final_value = metric_data
                    
                    metric_values.append({
                        "experiment_id": exp_data.get("experiment_id"),
                        "value": final_value
                    })
            
            if metric_values:
                values = [mv["value"] for mv in metric_values if isinstance(mv["value"], (int, float))]
                if values:
                    comparison["metric_comparison"][metric] = {
                        "values": metric_values,
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
        
        # Compare hyperparameters
        for param in all_hyperparams:
            param_values = []
            for exp in experiments:
                exp_data = exp["data"]
                param_value = exp_data.get("hyperparameters", {}).get(param)
                
                if param_value is not None:
                    param_values.append({
                        "experiment_id": exp_data.get("experiment_id"),
                        "value": param_value
                    })
            
            if param_values:
                # Check if all values are the same
                unique_values = set(str(pv["value"]) for pv in param_values)
                comparison["hyperparameter_comparison"][param] = {
                    "values": param_values,
                    "is_constant": len(unique_values) == 1,
                    "unique_values": list(unique_values)
                }
        
        return comparison

def create_reproducibility_tracker(
    project_name: str = "forge",
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tracker_backend: Union[str, TrackerBackend] = TrackerBackend.LOCAL,
    **kwargs
) -> ReproducibilityTracker:
    """
    Factory function to create a reproducibility tracker.
    
    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        run_name: Name of the run
        tracker_backend: Tracking backend to use
        **kwargs: Additional configuration options
    
    Returns:
        Configured ReproducibilityTracker instance
    """
    if isinstance(tracker_backend, str):
        tracker_backend = TrackerBackend(tracker_backend)
    
    config = ExperimentConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        run_name=run_name,
        tracker_backend=tracker_backend,
        **kwargs
    )
    
    return ReproducibilityTracker(config)

# Integration with existing forge training
def integrate_with_training(
    trainer,
    training_args,
    model_args,
    data_args,
    callbacks: Optional[List] = None
) -> ReproducibilityTracker:
    """
    Integrate reproducibility tracking with forge training.
    
    Args:
        trainer: Trainer instance
        training_args: Training arguments
        model_args: Model arguments
        data_args: Data arguments
        callbacks: Optional list of callbacks
    
    Returns:
        Configured ReproducibilityTracker instance
    """
    # Extract configuration from arguments
    hyperparameters = {
        **vars(training_args),
        **vars(model_args),
        **vars(data_args)
    }
    
    # Create tracker
    tracker = create_reproducibility_tracker(
        project_name=getattr(training_args, "project_name", "forge"),
        experiment_name=getattr(training_args, "experiment_name", None),
        run_name=getattr(training_args, "run_name", None),
        tracker_backend=getattr(training_args, "tracker_backend", "local"),
        output_dir=getattr(training_args, "output_dir", "./experiments")
    )
    
    # Start tracking
    tracker.start_run(hyperparameters)
    
    # Log dataset samples if available
    if hasattr(trainer, "train_dataset"):
        tracker.log_data_samples(trainer.train_dataset, split="train")
    
    # Create code snapshot
    tracker.log_code_snapshot()
    
    # Generate Dockerfile
    tracker.generate_dockerfile()
    
    # Save full experiment configuration
    full_config = {
        "training_args": vars(training_args),
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "environment": asdict(tracker.environment_info)
    }
    tracker.save_experiment_config(full_config)
    
    # Add callback to trainer for metric logging
    class ReproducibilityCallback:
        def __init__(self, tracker: ReproducibilityTracker):
            self.tracker = tracker
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.tracker.log_metrics(logs, step=state.global_step)
        
        def on_train_end(self, args, state, control, **kwargs):
            self.tracker.end_run(status="FINISHED")
    
    if callbacks is None:
        callbacks = []
    callbacks.append(ReproducibilityCallback(tracker))
    
    return tracker

# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    tracker = create_reproducibility_tracker(
        project_name="llama-finetuning",
        experiment_name="qlora-7b",
        run_name="run_001",
        tracker_backend="local"
    )
    
    # Start tracking
    tracker.start_run({
        "model_name": "meta-llama/Llama-2-7b-hf",
        "learning_rate": 2e-4,
        "batch_size": 4,
        "num_epochs": 3
    })
    
    # Log metrics during training
    for step in range(100):
        tracker.log_metric("loss", 1.0 / (step + 1), step=step)
        tracker.log_metric("accuracy", min(0.99, step / 100), step=step)
    
    # End tracking
    tracker.end_run()
    
    # Example 2: Compare experiments
    comparison = ReproducibilityTracker.compare_experiments(
        experiment_dirs=[
            "./experiments/llama-finetuning_qlora-7b_20240101_120000/run_001",
            "./experiments/llama-finetuning_qlora-7b_20240101_130000/run_002"
        ],
        metrics=["loss", "accuracy"]
    )
    
    print(json.dumps(comparison, indent=2))