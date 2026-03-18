# Copyright 2025 the forge team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .parser import get_eval_args, get_infer_args, get_ray_args, get_train_args, read_args
from .training_args import RayArguments, TrainingArguments

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Complete experiment tracking system with reproducibility guarantees."""
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self.output_dir / self.experiment_id
        self.metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Trackers
        self.mlflow_run = None
        self.wandb_run = None
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.experiment_name.encode()).hexdigest()[:8]
        return f"{timestamp}_{name_hash}"
    
    def capture_environment(self) -> Dict[str, Any]:
        """Capture complete environment details for reproducibility."""
        env_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "git_commit": self._get_git_commit(),
            "git_status": self._get_git_status(),
            "pip_packages": self._get_installed_packages(),
            "environment_variables": dict(os.environ),
            "system_info": self._get_system_info()
        }
        
        # Save environment info
        env_file = self.experiment_dir / "environment.json"
        with open(env_file, "w") as f:
            json.dump(env_info, f, indent=2, default=str)
        
        self.metadata["environment"] = env_info
        return env_info
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def _get_git_status(self) -> Dict[str, Any]:
        """Get git repository status."""
        try:
            # Get modified files
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            modified_files = result.stdout.strip().split("\n") if result.stdout else []
            
            # Get remote URL
            remote_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else None
            
            return {
                "modified_files": modified_files,
                "remote_url": remote_url,
                "is_dirty": len(modified_files) > 0
            }
        except Exception:
            return {"modified_files": [], "remote_url": None, "is_dirty": False}
    
    def _get_installed_packages(self) -> List[str]:
        """Get list of installed Python packages."""
        try:
            import pkg_resources
            return [f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]
        except Exception:
            return []
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        import platform
        import psutil
        
        return {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
    
    def track_hyperparameters(self, 
                            model_args: ModelArguments,
                            data_args: DataArguments,
                            training_args: TrainingArguments,
                            finetuning_args: FinetuningArguments,
                            generating_args: Optional[GeneratingArguments] = None,
                            evaluation_args: Optional[EvaluationArguments] = None) -> None:
        """Track all hyperparameters and configuration."""
        hyperparams = {
            "model": model_args.to_dict(),
            "data": data_args.to_dict(),
            "training": training_args.to_dict(),
            "finetuning": finetuning_args.to_dict(),
        }
        
        if generating_args:
            hyperparams["generating"] = generating_args.to_dict()
        if evaluation_args:
            hyperparams["evaluation"] = evaluation_args.to_dict()
        
        # Save hyperparameters
        params_file = self.experiment_dir / "hyperparameters.json"
        with open(params_file, "w") as f:
            json.dump(hyperparams, f, indent=2, default=str)
        
        self.metadata["hyperparameters"] = hyperparams
        
        # Log to external trackers
        self._log_to_mlflow(hyperparams)
        self._log_to_wandb(hyperparams)
    
    def track_data_samples(self, 
                          dataset_path: str,
                          sample_size: int = 100) -> None:
        """Capture data samples for reproducibility."""
        try:
            import pandas as pd
            from datasets import load_dataset
            
            # Load a sample of the dataset
            dataset = load_dataset(dataset_path, split=f"train[:{sample_size}]")
            
            # Convert to pandas for easier handling
            df = pd.DataFrame(dataset)
            
            # Save sample
            sample_file = self.experiment_dir / "data_samples.json"
            df.to_json(sample_file, orient="records", lines=True)
            
            # Calculate and save data hash
            data_hash = hashlib.md5(str(dataset).encode()).hexdigest()
            
            self.metadata["data_samples"] = {
                "dataset_path": dataset_path,
                "sample_size": sample_size,
                "data_hash": data_hash,
                "columns": list(df.columns),
                "sample_file": str(sample_file)
            }
            
        except Exception as e:
            logger.warning(f"Failed to capture data samples: {e}")
    
    def track_code_version(self, 
                          code_dir: str = ".",
                          include_patterns: List[str] = None) -> None:
        """Track code version by creating a snapshot."""
        if include_patterns is None:
            include_patterns = ["*.py", "*.yaml", "*.yml", "*.json", "*.md"]
        
        import tarfile
        import tempfile
        
        # Create code snapshot
        snapshot_file = self.experiment_dir / "code_snapshot.tar.gz"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_snapshot = Path(tmpdir) / "snapshot.tar.gz"
            
            with tarfile.open(tmp_snapshot, "w:gz") as tar:
                for pattern in include_patterns:
                    for file_path in Path(code_dir).rglob(pattern):
                        if "experiments" not in str(file_path) and ".git" not in str(file_path):
                            tar.add(file_path, arcname=file_path.relative_to(code_dir))
            
            # Move to experiment directory
            tmp_snapshot.rename(snapshot_file)
        
        # Calculate code hash
        with open(snapshot_file, "rb") as f:
            code_hash = hashlib.md5(f.read()).hexdigest()
        
        self.metadata["code_version"] = {
            "snapshot_file": str(snapshot_file),
            "code_hash": code_hash,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_dockerfile(self, 
                           base_image: str = "python:3.10-slim",
                           requirements_files: List[str] = None) -> Path:
        """Generate Dockerfile for exact reproduction."""
        dockerfile_content = f"""# Auto-generated Dockerfile for experiment: {self.experiment_name}
# Generated at: {datetime.now().isoformat()}
FROM {base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
"""
        
        if requirements_files:
            for i, req_file in enumerate(requirements_files):
                dockerfile_content += f"COPY {req_file} /app/requirements_{i}.txt\n"
            
            dockerfile_content += "\n# Install Python dependencies\n"
            for i in range(len(requirements_files)):
                dockerfile_content += f"RUN pip install --no-cache-dir -r requirements_{i}.txt\n"
        
        # Add experiment-specific setup
        dockerfile_content += f"""
# Copy experiment configuration
COPY experiments/{self.experiment_id}/hyperparameters.json /app/config/
COPY experiments/{self.experiment_id}/environment.json /app/config/

# Copy code snapshot
COPY experiments/{self.experiment_id}/code_snapshot.tar.gz /app/
RUN tar -xzf code_snapshot.tar.gz && rm code_snapshot.tar.gz

# Set environment variables
ENV EXPERIMENT_ID={self.experiment_id}
ENV EXPERIMENT_NAME={self.experiment_name}

# Default command
CMD ["python", "train.py", "--config", "/app/config/hyperparameters.json"]
"""
        
        # Save Dockerfile
        dockerfile_path = self.experiment_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Generate docker-compose.yml for easy reproduction
        self._generate_docker_compose()
        
        self.metadata["dockerfile"] = {
            "path": str(dockerfile_path),
            "base_image": base_image,
            "generated_at": datetime.now().isoformat()
        }
        
        return dockerfile_path
    
    def _generate_docker_compose(self) -> None:
        """Generate docker-compose.yml for experiment reproduction."""
        compose_content = f"""version: '3.8'

services:
  experiment:
    build:
      context: ../..
      dockerfile: experiments/{self.experiment_id}/Dockerfile
    volumes:
      - ../../data:/app/data
      - ./outputs:/app/outputs
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""
        
        compose_file = self.experiment_dir / "docker-compose.yml"
        with open(compose_file, "w") as f:
            f.write(compose_content)
    
    def start_mlflow_tracking(self, 
                             tracking_uri: str = "http://localhost:5000",
                             experiment_name: Optional[str] = None) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(tracking_uri)
            
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            else:
                mlflow.set_experiment(self.experiment_name)
            
            self.mlflow_run = mlflow.start_run(run_name=self.experiment_id)
            
            # Log parameters
            mlflow.log_params(self.metadata.get("hyperparameters", {}))
            
            # Log tags
            mlflow.set_tags({
                "experiment_id": self.experiment_id,
                "git_commit": self.metadata.get("environment", {}).get("git_commit"),
                "platform": self.metadata.get("environment", {}).get("platform")
            })
            
            logger.info(f"MLflow tracking started: {tracking_uri}")
            
        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
        except Exception as e:
            logger.error(f"Failed to start MLflow tracking: {e}")
    
    def start_wandb_tracking(self,
                            project: str = "forge",
                            entity: Optional[str] = None,
                            config: Optional[Dict] = None) -> None:
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            wandb_config = config or {}
            wandb_config.update({
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name
            })
            
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=self.experiment_id,
                config=wandb_config,
                reinit=True
            )
            
            logger.info(f"W&B tracking started: {project}/{self.experiment_id}")
            
        except ImportError:
            logger.warning("W&B not installed. Install with: pip install wandb")
        except Exception as e:
            logger.error(f"Failed to start W&B tracking: {e}")
    
    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """Log metrics to all active trackers."""
        # Save locally
        metrics_file = self.experiment_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            entry = {"step": step, "metrics": metrics, "timestamp": datetime.now().isoformat()}
            f.write(json.dumps(entry) + "\n")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                import mlflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        # Log to W&B
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
    
    def log_artifact(self, 
                    artifact_path: str,
                    artifact_type: str = "model",
                    name: Optional[str] = None) -> None:
        """Log artifact (model, dataset, etc.) to trackers."""
        artifact_name = name or Path(artifact_path).name
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")
        
        # Log to W&B
        if self.wandb_run:
            try:
                import wandb
                artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
                artifact.add_file(artifact_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log artifact to W&B: {e}")
    
    def finish(self, status: str = "completed") -> None:
        """Finish experiment tracking and cleanup."""
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now().isoformat()
        
        # Save final metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Stop trackers
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
        
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        
        logger.info(f"Experiment {self.experiment_id} finished with status: {status}")
    
    def compare_with(self, other_experiment_id: str) -> Dict[str, Any]:
        """Compare this experiment with another."""
        other_dir = self.output_dir / other_experiment_id
        
        if not other_dir.exists():
            raise ValueError(f"Experiment {other_experiment_id} not found")
        
        # Load metadata
        with open(other_dir / "metadata.json", "r") as f:
            other_metadata = json.load(f)
        
        # Compare hyperparameters
        comparison = {
            "experiment_1": self.metadata,
            "experiment_2": other_metadata,
            "differences": {}
        }
        
        # Compare hyperparameters
        params1 = self.metadata.get("hyperparameters", {})
        params2 = other_metadata.get("hyperparameters", {})
        
        for key in set(params1.keys()) | set(params2.keys()):
            if key in params1 and key in params2:
                if params1[key] != params2[key]:
                    comparison["differences"][key] = {
                        "experiment_1": params1[key],
                        "experiment_2": params2[key]
                    }
            else:
                comparison["differences"][key] = {
                    "experiment_1": params1.get(key, "MISSING"),
                    "experiment_2": params2.get(key, "MISSING")
                }
        
        # Compare metrics if available
        metrics1_file = self.experiment_dir / "metrics.jsonl"
        metrics2_file = other_dir / "metrics.jsonl"
        
        if metrics1_file.exists() and metrics2_file.exists():
            comparison["metrics_comparison"] = self._compare_metrics(
                metrics1_file, metrics2_file
            )
        
        # Save comparison
        comparison_file = self.output_dir / f"comparison_{self.experiment_id}_{other_experiment_id}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        return comparison
    
    def _compare_metrics(self, file1: Path, file2: Path) -> Dict[str, Any]:
        """Compare metrics between two experiments."""
        metrics1 = []
        metrics2 = []
        
        with open(file1, "r") as f:
            for line in f:
                metrics1.append(json.loads(line))
        
        with open(file2, "r") as f:
            for line in f:
                metrics2.append(json.loads(line))
        
        # Find common metrics
        common_metrics = set()
        for entry in metrics1:
            common_metrics.update(entry["metrics"].keys())
        for entry in metrics2:
            common_metrics.update(entry["metrics"].keys())
        
        comparison = {}
        for metric in common_metrics:
            values1 = [entry["metrics"].get(metric) for entry in metrics1 if metric in entry["metrics"]]
            values2 = [entry["metrics"].get(metric) for entry in metrics2 if metric in entry["metrics"]]
            
            if values1 and values2:
                comparison[metric] = {
                    "experiment_1": {
                        "mean": sum(values1) / len(values1),
                        "min": min(values1),
                        "max": max(values1),
                        "final": values1[-1] if values1 else None
                    },
                    "experiment_2": {
                        "mean": sum(values2) / len(values2),
                        "min": min(values2),
                        "max": max(values2),
                        "final": values2[-1] if values2 else None
                    }
                }
        
        return comparison


class ABTestRunner:
    """Run and compare multiple experiment configurations."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.experiments = []
        self.results = []
    
    def add_variant(self, 
                   name: str,
                   modifications: Dict[str, Any],
                   tracker: Optional[ExperimentTracker] = None) -> None:
        """Add a variant configuration for A/B testing."""
        variant_config = self.base_config.copy()
        
        # Apply modifications
        for key, value in modifications.items():
            keys = key.split(".")
            config = variant_config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
        
        self.experiments.append({
            "name": name,
            "config": variant_config,
            "modifications": modifications,
            "tracker": tracker
        })
    
    def run_all(self, 
                train_func,
                num_seeds: int = 3) -> Dict[str, Any]:
        """Run all experiment variants with multiple seeds."""
        import concurrent.futures
        
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for exp in self.experiments:
                for seed in range(num_seeds):
                    future = executor.submit(
                        self._run_single_experiment,
                        exp, seed, train_func
                    )
                    futures.append((exp["name"], seed, future))
            
            # Collect results
            for name, seed, future in futures:
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    key = f"{name}_seed{seed}"
                    all_results[key] = result
                except Exception as e:
                    logger.error(f"Experiment {name} seed {seed} failed: {e}")
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Save analysis
        analysis_file = Path("ab_test_results.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _run_single_experiment(self, 
                              experiment: Dict,
                              seed: int,
                              train_func) -> Dict[str, Any]:
        """Run a single experiment variant."""
        import copy
        
        # Deep copy config to avoid mutation
        config = copy.deepcopy(experiment["config"])
        
        # Set random seed
        if "training" in config:
            config["training"]["seed"] = seed
        
        # Create tracker for this run
        if experiment["tracker"]:
            tracker = experiment["tracker"]
        else:
            tracker = ExperimentTracker(
                experiment_name=f"{experiment['name']}_seed{seed}",
                output_dir=f"./ab_tests/{experiment['name']}"
            )
        
        # Track configuration
        tracker.capture_environment()
        tracker.track_hyperparameters(
            ModelArguments(**config.get("model", {})),
            DataArguments(**config.get("data", {})),
            TrainingArguments(**config.get("training", {})),
            FinetuningArguments(**config.get("finetuning", {})),
            GeneratingArguments(**config.get("generating", {})) if "generating" in config else None,
            EvaluationArguments(**config.get("evaluation", {})) if "evaluation" in config else None
        )
        
        # Run training
        try:
            results = train_func(config)
            tracker.log_metrics(results.get("metrics", {}))
            tracker.finish("completed")
            
            return {
                "experiment": experiment["name"],
                "seed": seed,
                "config": config,
                "results": results,
                "tracker_id": tracker.experiment_id
            }
        except Exception as e:
            tracker.finish("failed")
            raise e
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze A/B test results with statistical significance."""
        from scipy import stats
        import numpy as np
        
        # Group results by experiment name
        grouped = {}
        for key, result in results.items():
            name = result["experiment"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(result)
        
        analysis = {
            "experiments": {},
            "comparisons": {},
            "summary": {}
        }
        
        # Analyze each experiment
        for name, exp_results in grouped.items():
            metrics_across_seeds = {}
            
            for result in exp_results:
                for metric_name, value in result["results"].get("metrics", {}).items():
                    if metric_name not in metrics_across_seeds:
                        metrics_across_seeds[metric_name] = []
                    metrics_across_seeds[metric_name].append(value)
            
            # Calculate statistics
            exp_analysis = {}
            for metric_name, values in metrics_across_seeds.items():
                exp_analysis[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values
                }
            
            analysis["experiments"][name] = exp_analysis
        
        # Perform pairwise comparisons
        experiment_names = list(grouped.keys())
        for i in range(len(experiment_names)):
            for j in range(i + 1, len(experiment_names)):
                name1 = experiment_names[i]
                name2 = experiment_names[j]
                
                comparison_key = f"{name1}_vs_{name2}"
                analysis["comparisons"][comparison_key] = {}
                
                # Compare common metrics
                metrics1 = set(analysis["experiments"][name1].keys())
                metrics2 = set(analysis["experiments"][name2].keys())
                common_metrics = metrics1 & metrics2
                
                for metric in common_metrics:
                    values1 = analysis["experiments"][name1][metric]["values"]
                    values2 = analysis["experiments"][name2][metric]["values"]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = np.mean(values1) - np.mean(values2)
                    pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    analysis["comparisons"][comparison_key][metric] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "cohens_d": float(cohens_d),
                        "mean_difference": float(mean_diff),
                        "relative_improvement": float(mean_diff / np.mean(values2)) if np.mean(values2) != 0 else 0
                    }
        
        # Generate summary
        if analysis["comparisons"]:
            best_experiment = None
            best_score = -float('inf')
            
            for name, metrics in analysis["experiments"].items():
                # Use first metric as primary score (could be made configurable)
                primary_metric = list(metrics.keys())[0]
                score = metrics[primary_metric]["mean"]
                
                if score > best_score:
                    best_score = score
                    best_experiment = name
            
            analysis["summary"] = {
                "best_experiment": best_experiment,
                "best_score": best_score,
                "num_experiments": len(experiment_names),
                "total_runs": len(results)
            }
        
        return analysis


def setup_experiment_tracking(
    experiment_name: str,
    output_dir: str = "./experiments",
    use_mlflow: bool = False,
    use_wandb: bool = False,
    mlflow_uri: Optional[str] = None,
    wandb_project: Optional[str] = None,
    capture_environment: bool = True,
    generate_dockerfile: bool = True,
    track_data_samples: bool = True,
    data_sample_size: int = 100
) -> ExperimentTracker:
    """Setup experiment tracking with specified configuration."""
    tracker = ExperimentTracker(experiment_name, output_dir)
    
    if capture_environment:
        tracker.capture_environment()
    
    if use_mlflow:
        tracker.start_mlflow_tracking(
            tracking_uri=mlflow_uri or "http://localhost:5000"
        )
    
    if use_wandb:
        tracker.start_wandb_tracking(
            project=wandb_project or "forge"
        )
    
    if generate_dockerfile:
        tracker.generate_dockerfile()
    
    if track_data_samples:
        # This would need to be called after data_args are available
        pass
    
    return tracker


def generate_reproduction_script(
    experiment_id: str,
    experiment_dir: str = "./experiments"
) -> Path:
    """Generate a script to reproduce an experiment."""
    exp_dir = Path(experiment_dir) / experiment_id
    
    if not exp_dir.exists():
        raise ValueError(f"Experiment {experiment_id} not found")
    
    script_content = f"""#!/bin/bash
# Auto-generated reproduction script for experiment: {experiment_id}
# Generated at: {datetime.now().isoformat()}

set -e

echo "Reproducing experiment: {experiment_id}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Build and run with Docker Compose
cd {exp_dir}
docker-compose up --build

echo "Experiment reproduction completed!"
"""
    
    script_path = exp_dir / "reproduce.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    return script_path


__all__ = [
    "DataArguments",
    "EvaluationArguments",
    "FinetuningArguments",
    "GeneratingArguments",
    "ModelArguments",
    "RayArguments",
    "TrainingArguments",
    "get_eval_args",
    "get_infer_args",
    "get_ray_args",
    "get_train_args",
    "read_args",
    "ExperimentTracker",
    "ABTestRunner",
    "setup_experiment_tracking",
    "generate_reproduction_script",
]