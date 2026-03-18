"""
Complete experiment tracking system with reproducibility guarantees for forge.
Captures hyperparameters, code versions, data samples, and environment details.
Includes automatic Docker image generation and A/B testing comparison tools.
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
import logging
import pkg_resources
import platform

# Try to import tracking integrations
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

from ..extras.logging import get_logger
from ..extras.misc import get_device_count

logger = get_logger(__name__)


@dataclass
class EnvironmentInfo:
    """Complete environment information for reproducibility."""
    python_version: str = ""
    platform_info: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    transformers_version: str = ""
    forge_version: str = ""
    git_commit: str = ""
    git_branch: str = ""
    installed_packages: Dict[str, str] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()
        self._capture_environment()
    
    def _capture_environment(self):
        """Capture complete environment details."""
        # Python and platform info
        self.python_version = sys.version
        self.platform_info = platform.platform()
        
        # Package versions
        self.torch_version = self._get_package_version("torch")
        self.transformers_version = self._get_package_version("transformers")
        self.forge_version = self._get_package_version("forge")
        
        # CUDA version
        try:
            import torch
            if torch.cuda.is_available():
                self.cuda_version = torch.version.cuda
        except:
            self.cuda_version = "N/A"
        
        # Git information
        self._capture_git_info()
        
        # Hardware information
        self._capture_hardware_info()
        
        # All installed packages
        self._capture_installed_packages()
    
    def _get_package_version(self, package_name: str) -> str:
        """Get version of a specific package."""
        try:
            return pkg_resources.get_distribution(package_name).version
        except:
            return "N/A"
    
    def _capture_git_info(self):
        """Capture git commit and branch information."""
        try:
            # Get git commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()
            
            # Get git branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                self.git_branch = result.stdout.strip()
        except:
            self.git_commit = "N/A"
            self.git_branch = "N/A"
    
    def _capture_hardware_info(self):
        """Capture hardware information."""
        self.hardware_info = {
            "cpu_count": os.cpu_count(),
            "gpu_count": get_device_count(),
            "platform_machine": platform.machine(),
            "platform_processor": platform.processor()
        }
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_info.append({
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory
                    })
                self.hardware_info["gpus"] = gpu_info
        except:
            pass
    
    def _capture_installed_packages(self):
        """Capture all installed Python packages."""
        try:
            installed_packages = {}
            for dist in pkg_resources.working_set:
                installed_packages[dist.project_name] = dist.version
            self.installed_packages = installed_packages
        except:
            self.installed_packages = {}


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiment_name: str
    run_name: str
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    log_system_metrics: bool = True
    log_artifacts: bool = True
    capture_code: bool = True
    capture_data_samples: bool = True
    generate_docker: bool = True
    docker_base_image: str = "python:3.10-slim"
    docker_system_packages: List[str] = field(default_factory=lambda: [
        "git", "wget", "curl", "build-essential"
    ])


class DockerGenerator:
    """Generates Dockerfiles for experiment reproducibility."""
    
    def __init__(self, base_image: str = "python:3.10-slim"):
        self.base_image = base_image
    
    def generate_dockerfile(
        self,
        requirements_file: Optional[str] = None,
        system_packages: Optional[List[str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        copy_files: Optional[List[Tuple[str, str]]] = None,
        entrypoint: Optional[str] = None,
        working_dir: str = "/app"
    ) -> str:
        """Generate a complete Dockerfile for experiment reproduction."""
        
        lines = [
            f"FROM {self.base_image}",
            "",
            "# Set environment variables",
            "ENV PYTHONUNBUFFERED=1 \\",
            "    PYTHONDONTWRITEBYTECODE=1 \\",
            "    PIP_NO_CACHE_DIR=1 \\",
            "    PIP_DISABLE_PIP_VERSION_CHECK=1"
        ]
        
        # Add custom environment variables
        if environment_vars:
            for key, value in environment_vars.items():
                lines.append(f"ENV {key}={value}")
        
        lines.extend(["", "# Install system packages"])
        
        # Install system packages
        if system_packages:
            lines.append("RUN apt-get update && apt-get install -y \\")
            for i, package in enumerate(system_packages):
                if i == len(system_packages) - 1:
                    lines.append(f"    {package} \\")
                else:
                    lines.append(f"    {package} \\")
            lines.extend([
                "    && rm -rf /var/lib/apt/lists/*",
                ""
            ])
        
        # Set working directory
        lines.extend([
            f"WORKDIR {working_dir}",
            ""
        ])
        
        # Copy requirements and install Python packages
        if requirements_file and os.path.exists(requirements_file):
            lines.extend([
                "# Copy and install Python requirements",
                f"COPY {os.path.basename(requirements_file)} ./",
                f"RUN pip install --no-cache-dir -r {os.path.basename(requirements_file)}",
                ""
            ])
        
        # Copy additional files
        if copy_files:
            lines.append("# Copy experiment files")
            for src, dst in copy_files:
                lines.append(f"COPY {src} {dst}")
            lines.append("")
        
        # Set entrypoint
        if entrypoint:
            lines.extend([
                "# Set entrypoint",
                f"ENTRYPOINT {entrypoint}"
            ])
        
        return "\n".join(lines)
    
    def generate_requirements_file(
        self,
        packages: Optional[List[str]] = None,
        output_path: str = "requirements.txt"
    ) -> str:
        """Generate a requirements.txt file with pinned versions."""
        
        if packages is None:
            # Capture current environment
            import pkg_resources
            packages = []
            for dist in pkg_resources.working_set:
                packages.append(f"{dist.project_name}=={dist.version}")
        
        # Write requirements file
        with open(output_path, 'w') as f:
            f.write("# Generated by forge Experiment Tracker\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
            for package in sorted(packages):
                f.write(f"{package}\n")
        
        return output_path
    
    def build_docker_image(
        self,
        dockerfile_path: str,
        tag: str,
        build_context: str = ".",
        no_cache: bool = False
    ) -> bool:
        """Build Docker image from generated Dockerfile."""
        
        if not HAS_DOCKER:
            logger.warning("Docker SDK not available. Cannot build image.")
            return False
        
        try:
            client = docker.from_env()
            
            # Build command
            build_args = {
                "path": build_context,
                "dockerfile": dockerfile_path,
                "tag": tag,
                "rm": True
            }
            
            if no_cache:
                build_args["nocache"] = True
            
            # Build the image
            image, logs = client.images.build(**build_args)
            
            # Log build output
            for log in logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Successfully built Docker image: {tag}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            return False


class ExperimentTracker:
    """Complete experiment tracking system with reproducibility guarantees."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.environment_info = EnvironmentInfo()
        self.docker_generator = DockerGenerator(config.docker_base_image)
        self.run = None
        self.metrics_history = []
        self.artifacts = []
        self.data_samples = []
        
        # Initialize tracking backends
        self._init_tracking_backends()
    
    def _init_tracking_backends(self):
        """Initialize tracking backends (MLflow, W&B)."""
        self.mlflow_run = None
        self.wandb_run = None
        
        # Initialize MLflow if available
        if HAS_MLFLOW and self.config.tracking_uri:
            try:
                mlflow.set_tracking_uri(self.config.tracking_uri)
                mlflow.set_experiment(self.config.experiment_name)
                self.mlflow_run = mlflow.start_run(run_name=self.config.run_name)
                
                # Log tags
                for key, value in self.config.tags.items():
                    mlflow.set_tag(key, value)
                
                logger.info(f"MLflow tracking initialized: {self.config.tracking_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
        
        # Initialize W&B if available
        if HAS_WANDB:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.experiment_name,
                    name=self.config.run_name,
                    config=self.config.tags,
                    reinit=True
                )
                logger.info("Weights & Biases tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")
        
        # Log to W&B
        if self.wandb_run:
            try:
                wandb.config.update(params)
            except Exception as e:
                logger.warning(f"Failed to log params to W&B: {e}")
        
        # Store locally
        self.metrics_history.append({
            "type": "parameters",
            "timestamp": datetime.now().isoformat(),
            "data": params
        })
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
        
        # Log to W&B
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to W&B: {e}")
        
        # Store locally
        self.metrics_history.append({
            "type": "metrics",
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": metrics
        })
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log an artifact (model, dataset, etc.)."""
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")
        
        # Log to W&B
        if self.wandb_run:
            try:
                wandb.save(artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to W&B: {e}")
        
        # Store reference
        self.artifacts.append({
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_data_sample(self, data: Any, sample_name: str, metadata: Optional[Dict] = None):
        """Log a sample of data for reproducibility."""
        sample_info = {
            "name": sample_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "data_type": str(type(data))
        }
        
        # Try to serialize data if possible
        try:
            if isinstance(data, (dict, list, str, int, float, bool)):
                sample_info["data"] = data
            elif hasattr(data, "tolist"):  # numpy arrays, torch tensors
                sample_info["data"] = data.tolist()
            else:
                sample_info["data"] = str(data)
        except:
            sample_info["data"] = "Unable to serialize"
        
        self.data_samples.append(sample_info)
        
        # Log to W&B as a table if possible
        if self.wandb_run and isinstance(data, (list, dict)):
            try:
                import pandas as pd
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    table = wandb.Table(dataframe=df)
                    wandb.log({f"sample_{sample_name}": table})
            except:
                pass
    
    def capture_code_snapshot(self, code_dir: str = "."):
        """Capture current code state for reproducibility."""
        if not self.config.capture_code:
            return
        
        try:
            # Create temporary directory for code snapshot
            with tempfile.TemporaryDirectory() as temp_dir:
                snapshot_dir = Path(temp_dir) / "code_snapshot"
                snapshot_dir.mkdir()
                
                # Copy Python files
                for py_file in Path(code_dir).rglob("*.py"):
                    if ".git" not in str(py_file) and "__pycache__" not in str(py_file):
                        rel_path = py_file.relative_to(code_dir)
                        dest_path = snapshot_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(py_file, dest_path)
                
                # Create archive
                archive_path = shutil.make_archive(
                    str(Path(temp_dir) / "code_snapshot"),
                    "zip",
                    root_dir=temp_dir,
                    base_dir="code_snapshot"
                )
                
                # Log as artifact
                self.log_artifact(archive_path, "code_snapshot")
                
        except Exception as e:
            logger.warning(f"Failed to capture code snapshot: {e}")
    
    def generate_docker_artifacts(self, output_dir: str = "docker_artifacts"):
        """Generate Docker artifacts for experiment reproduction."""
        if not self.config.generate_docker:
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate requirements.txt
            requirements_path = os.path.join(output_dir, "requirements.txt")
            self.docker_generator.generate_requirements_file(output_path=requirements_path)
            
            # Generate Dockerfile
            dockerfile_content = self.docker_generator.generate_dockerfile(
                requirements_file=requirements_path,
                system_packages=self.config.docker_system_packages,
                environment_vars={
                    "EXPERIMENT_NAME": self.config.experiment_name,
                    "RUN_NAME": self.config.run_name,
                    "GIT_COMMIT": self.environment_info.git_commit
                },
                copy_files=[
                    (requirements_path, "./requirements.txt"),
                    ("src", "./src"),
                    ("scripts", "./scripts")
                ],
                entrypoint='["python", "-m", "forge.train"]'
            )
            
            dockerfile_path = os.path.join(output_dir, "Dockerfile")
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Generate environment info JSON
            env_info_path = os.path.join(output_dir, "environment_info.json")
            with open(env_info_path, 'w') as f:
                json.dump(asdict(self.environment_info), f, indent=2)
            
            # Generate experiment config JSON
            config_path = os.path.join(output_dir, "experiment_config.json")
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            # Generate run script
            run_script = self._generate_run_script()
            run_script_path = os.path.join(output_dir, "run_experiment.sh")
            with open(run_script_path, 'w') as f:
                f.write(run_script)
            os.chmod(run_script_path, 0o755)
            
            logger.info(f"Docker artifacts generated in: {output_dir}")
            
            # Log artifacts directory
            self.log_artifact(output_dir, "docker_artifacts")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to generate Docker artifacts: {e}")
            return None
    
    def _generate_run_script(self) -> str:
        """Generate a run script for the experiment."""
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated experiment run script",
            f"# Experiment: {self.config.experiment_name}",
            f"# Run: {self.config.run_name}",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "set -e",
            "",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "",
            "# Run training",
            "python -m forge.train \\"
        ]
        
        # Add experiment-specific arguments
        for key, value in self.config.tags.items():
            if isinstance(value, bool):
                if value:
                    script_lines.append(f"    --{key} \\")
            else:
                script_lines.append(f"    --{key} {value} \\")
        
        # Remove trailing backslash from last line
        if script_lines[-1].endswith("\\"):
            script_lines[-1] = script_lines[-1][:-2]
        
        script_lines.extend([
            "",
            "echo 'Experiment completed successfully'",
            f"echo 'Run ID: {self.config.run_name}'"
        ])
        
        return "\n".join(script_lines)
    
    def save_tracking_data(self, output_path: str = "tracking_data.json"):
        """Save all tracking data to a JSON file."""
        tracking_data = {
            "experiment_config": asdict(self.config),
            "environment_info": asdict(self.environment_info),
            "metrics_history": self.metrics_history,
            "artifacts": self.artifacts,
            "data_samples": self.data_samples,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        logger.info(f"Tracking data saved to: {output_path}")
        return output_path
    
    def finish(self):
        """Finish the experiment run and clean up."""
        # Save tracking data
        tracking_file = self.save_tracking_data()
        self.log_artifact(tracking_file, "tracking_data")
        
        # Generate Docker artifacts
        if self.config.generate_docker:
            self.generate_docker_artifacts()
        
        # End MLflow run
        if self.mlflow_run:
            try:
                mlflow.end_run()
            except:
                pass
        
        # End W&B run
        if self.wandb_run:
            try:
                wandb.finish()
            except:
                pass
        
        logger.info(f"Experiment '{self.config.experiment_name}' completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class ExperimentComparator:
    """Tools for comparing experiments and A/B testing."""
    
    @staticmethod
    def compare_experiments(experiment_dirs: List[str], output_file: str = "comparison_report.html"):
        """Compare multiple experiments and generate a report."""
        
        experiments = []
        for exp_dir in experiment_dirs:
            tracking_file = Path(exp_dir) / "tracking_data.json"
            if tracking_file.exists():
                with open(tracking_file, 'r') as f:
                    experiments.append(json.load(f))
        
        if not experiments:
            logger.warning("No experiment data found for comparison")
            return
        
        # Generate comparison report
        report = ExperimentComparator._generate_comparison_report(experiments)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Comparison report generated: {output_file}")
        return output_file
    
    @staticmethod
    def _generate_comparison_report(experiments: List[Dict]) -> str:
        """Generate an HTML comparison report."""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>forge Experiment Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .experiment { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .metric { margin: 10px 0; }
                .metric-name { font-weight: bold; }
                .metric-value { color: #2196F3; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .best { background-color: #e8f5e9; }
            </style>
        </head>
        <body>
            <h1>forge Experiment Comparison Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        # Add experiment summaries
        html += "<h2>Experiment Summaries</h2>"
        
        for i, exp in enumerate(experiments):
            config = exp.get("experiment_config", {})
            env_info = exp.get("environment_info", {})
            metrics = exp.get("metrics_history", [])
            
            html += f"""
            <div class="experiment">
                <h3>Experiment {i+1}: {config.get('experiment_name', 'Unknown')}</h3>
                <p><strong>Run:</strong> {config.get('run_name', 'Unknown')}</p>
                <p><strong>Git Commit:</strong> {env_info.get('git_commit', 'N/A')}</p>
                <p><strong>Timestamp:</strong> {env_info.get('timestamp', 'N/A')}</p>
            """
            
            # Add metrics table
            if metrics:
                html += "<h4>Final Metrics</h4><table><tr><th>Metric</th><th>Value</th></tr>"
                
                # Get final metrics (last metrics entry)
                final_metrics = {}
                for entry in reversed(metrics):
                    if entry.get("type") == "metrics":
                        final_metrics = entry.get("data", {})
                        break
                
                for metric_name, metric_value in final_metrics.items():
                    html += f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
        
        # Add comparison table
        if len(experiments) > 1:
            html += "<h2>Side-by-Side Comparison</h2>"
            html += "<table><tr><th>Experiment</th>"
            
            # Get all unique metric names
            all_metrics = set()
            for exp in experiments:
                for entry in exp.get("metrics_history", []):
                    if entry.get("type") == "metrics":
                        all_metrics.update(entry.get("data", {}).keys())
            
            for metric in sorted(all_metrics):
                html += f"<th>{metric}</th>"
            
            html += "</tr>"
            
            # Add experiment rows
            for i, exp in enumerate(experiments):
                html += f"<tr><td>Experiment {i+1}</td>"
                
                # Get final metrics for this experiment
                final_metrics = {}
                for entry in reversed(exp.get("metrics_history", [])):
                    if entry.get("type") == "metrics":
                        final_metrics = entry.get("data", {})
                        break
                
                for metric in sorted(all_metrics):
                    value = final_metrics.get(metric, "N/A")
                    html += f"<td>{value}</td>"
                
                html += "</tr>"
            
            html += "</table>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def run_ab_test(
        experiment_configs: List[Dict[str, Any]],
        train_func,
        output_dir: str = "ab_test_results"
    ):
        """Run A/B test comparing different configurations."""
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, config in enumerate(experiment_configs):
            exp_name = f"ab_test_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create experiment tracker
            exp_config = ExperimentConfig(
                experiment_name="ab_test",
                run_name=exp_name,
                tags=config
            )
            
            with ExperimentTracker(exp_config) as tracker:
                # Log configuration
                tracker.log_parameters(config)
                
                # Run training
                try:
                    metrics = train_func(config, tracker)
                    tracker.log_metrics(metrics)
                    
                    # Store results
                    results.append({
                        "experiment_id": exp_name,
                        "config": config,
                        "metrics": metrics,
                        "status": "completed"
                    })
                    
                except Exception as e:
                    logger.error(f"Experiment {exp_name} failed: {e}")
                    results.append({
                        "experiment_id": exp_name,
                        "config": config,
                        "error": str(e),
                        "status": "failed"
                    })
        
        # Save results
        results_file = os.path.join(output_dir, "ab_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate comparison report
        ExperimentComparator.compare_experiments(
            [os.path.join(output_dir, r["experiment_id"]) for r in results if r["status"] == "completed"],
            os.path.join(output_dir, "ab_test_report.html")
        )
        
        return results


def create_experiment_tracker(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    **kwargs
) -> ExperimentTracker:
    """Convenience function to create an experiment tracker."""
    
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        **kwargs
    )
    
    return ExperimentTracker(config)


# Example usage
if __name__ == "__main__":
    # Example: Track a training experiment
    tracker = create_experiment_tracker(
        experiment_name="llama_finetuning",
        run_name="test_run_001",
        tracking_uri="http://localhost:5000"
    )
    
    with tracker:
        # Log hyperparameters
        tracker.log_parameters({
            "learning_rate": 2e-5,
            "batch_size": 4,
            "epochs": 3,
            "model_name": "llama-7b"
        })
        
        # Simulate training loop
        for epoch in range(3):
            # Log metrics
            tracker.log_metrics({
                "train_loss": 0.5 - epoch * 0.1,
                "eval_loss": 0.6 - epoch * 0.1,
                "accuracy": 0.7 + epoch * 0.05
            }, step=epoch)
            
            # Log data sample
            if epoch == 0:
                tracker.log_data_sample(
                    data={"input": "Hello world", "output": "Hola mundo"},
                    sample_name="translation_sample",
                    metadata={"language": "en-es"}
                )
        
        # Log final model
        tracker.log_artifact("final_model.bin", "model")
        
        # Capture code and generate Docker artifacts
        tracker.capture_code_snapshot()
        tracker.generate_docker_artifacts()
    
    print("Experiment tracking completed successfully!")