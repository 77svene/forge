"""
Edge Deployment Pipeline for Unsloth Studio
One-click deployment to edge devices with automatic hardware optimization
"""

import os
import json
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Third-party imports (gracefully handle missing dependencies)
try:
    import docker
    from docker.models.containers import Container
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

try:
    import torch
    import onnx
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.auth.storage import AuthStorage

logger = logging.getLogger(__name__)


class TargetDevice(Enum):
    """Supported edge devices"""
    JETSON_NANO = "jetson-nano"
    JETSON_XAVIER = "jetson-xavier"
    JETSON_ORIN = "jetson-orin"
    RASPBERRY_PI_4 = "raspberry-pi-4"
    RASPBERRY_PI_5 = "raspberry-pi-5"
    GENERIC_ARM64 = "generic-arm64"
    GENERIC_ARM32 = "generic-arm32"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    target_device: TargetDevice
    optimization_level: int = 3  # TVM optimization level (0-3)
    use_fp16: bool = False
    use_int8: bool = False
    batch_size: int = 1
    input_shapes: Dict[str, List[int]] = None
    tuning_records: Optional[str] = None
    
    def __post_init__(self):
        if self.input_shapes is None:
            self.input_shapes = {}


@dataclass
class DeploymentManifest:
    """Manifest for deployed model"""
    model_id: str
    model_name: str
    version: str
    target_device: TargetDevice
    optimization_config: Dict[str, Any]
    input_specs: Dict[str, Any]
    output_specs: Dict[str, Any]
    checksum: str
    created_at: str
    docker_image: Optional[str] = None
    ota_endpoint: Optional[str] = None


class EdgeOptimizer:
    """
    Main class for edge deployment pipeline
    Handles model optimization, container generation, and OTA updates
    """
    
    # Device-specific configurations
    DEVICE_CONFIGS = {
        TargetDevice.JETSON_NANO: {
            "tvm_target": "llvm -mcpu=cortex-a57 -mattr=+neon",
            "docker_base": "nvcr.io/nvidia/l4t-base:r32.7.1",
            "memory_limit_mb": 4096,
            "compute_capability": "5.3"
        },
        TargetDevice.JETSON_XAVIER: {
            "tvm_target": "llvm -mcpu=carmel -mattr=+neon",
            "docker_base": "nvcr.io/nvidia/l4t-base:r35.3.1",
            "memory_limit_mb": 16384,
            "compute_capability": "7.2"
        },
        TargetDevice.JETSON_ORIN: {
            "tvm_target": "llvm -mcpu=carmel -mattr=+neon",
            "docker_base": "nvcr.io/nvidia/l4t-base:r36.2.0",
            "memory_limit_mb": 32768,
            "compute_capability": "8.7"
        },
        TargetDevice.RASPBERRY_PI_4: {
            "tvm_target": "llvm -mcpu=cortex-a72 -mattr=+neon",
            "docker_base": "balenalib/raspberrypi4-64-python:3.9",
            "memory_limit_mb": 8192,
            "compute_capability": "0"
        },
        TargetDevice.RASPBERRY_PI_5: {
            "tvm_target": "llvm -mcpu=cortex-a76 -mattr=+neon",
            "docker_base": "balenalib/raspberrypi5-python:3.11",
            "memory_limit_mb": 16384,
            "compute_capability": "0"
        },
        TargetDevice.GENERIC_ARM64: {
            "tvm_target": "llvm -mtriple=aarch64-linux-gnu",
            "docker_base": "arm64v8/python:3.9-slim",
            "memory_limit_mb": 4096,
            "compute_capability": "0"
        },
        TargetDevice.GENERIC_ARM32: {
            "tvm_target": "llvm -mtriple=arm-linux-gnueabihf",
            "docker_base": "arm32v7/python:3.9-slim",
            "memory_limit_mb": 2048,
            "compute_capability": "0"
        }
    }
    
    def __init__(self, 
                 workspace_dir: Optional[Union[str, Path]] = None,
                 auth_storage: Optional[AuthStorage] = None):
        """
        Initialize Edge Optimizer
        
        Args:
            workspace_dir: Directory for temporary files and outputs
            auth_storage: Authentication storage for device credentials
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(tempfile.mkdtemp(prefix="forge_edge_"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.auth_storage = auth_storage or AuthStorage()
        self.job_manager = JobManager()
        
        # Initialize Docker client if available
        self.docker_client = None
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
        
        # Check dependencies
        self._check_dependencies()
        
        # Registry for deployed models
        self.deployment_registry_path = self.workspace_dir / "deployment_registry.json"
        self._load_deployment_registry()
    
    def _check_dependencies(self):
        """Check for required dependencies"""
        missing_deps = []
        
        if not HAS_TVM:
            missing_deps.append("Apache TVM (pip install apache-tvm)")
        if not HAS_TORCH:
            missing_deps.append("PyTorch (pip install torch)")
        if not HAS_DOCKER:
            missing_deps.append("Docker SDK (pip install docker)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("Some functionality may be limited")
    
    def _load_deployment_registry(self):
        """Load deployment registry from disk"""
        self.deployment_registry = {}
        if self.deployment_registry_path.exists():
            try:
                with open(self.deployment_registry_path, 'r') as f:
                    self.deployment_registry = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load deployment registry: {e}")
    
    def _save_deployment_registry(self):
        """Save deployment registry to disk"""
        try:
            with open(self.deployment_registry_path, 'w') as f:
                json.dump(self.deployment_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save deployment registry: {e}")
    
    def _get_device_config(self, target_device: TargetDevice) -> Dict[str, Any]:
        """Get configuration for target device"""
        if target_device not in self.DEVICE_CONFIGS:
            raise ValueError(f"Unsupported target device: {target_device}")
        return self.DEVICE_CONFIGS[target_device]
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def optimize_model(self,
                      model_path: Union[str, Path],
                      config: OptimizationConfig,
                      output_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Optimize model for target edge device using TVM
        
        Args:
            model_path: Path to PyTorch model (.pt) or ONNX model (.onnx)
            config: Optimization configuration
            output_dir: Output directory for optimized model
            
        Returns:
            Path to optimized model library
        """
        if not HAS_TVM:
            raise RuntimeError("Apache TVM is required for model optimization")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        output_dir = Path(output_dir) if output_dir else self.workspace_dir / "optimized_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique model ID
        model_id = f"{model_path.stem}_{config.target_device.value}_{hashlib.md5(str(config).encode()).hexdigest()[:8]}"
        optimized_lib_path = output_dir / f"{model_id}.tar"
        
        logger.info(f"Optimizing model {model_path.name} for {config.target_device.value}")
        
        try:
            # Load model based on file extension
            if model_path.suffix == '.pt':
                if not HAS_TORCH:
                    raise RuntimeError("PyTorch is required for .pt models")
                model = torch.jit.load(str(model_path))
                model.eval()
                
                # Get example input
                input_shapes = config.input_shapes or {"input": [1, 3, 224, 224]}
                example_input = torch.randn(*input_shapes.get("input", [1, 3, 224, 224]))
                
                # Convert to Relay IR
                mod, params = relay.frontend.from_pytorch(model, [("input", example_input.shape)])
                
            elif model_path.suffix == '.onnx':
                onnx_model = onnx.load(str(model_path))
                mod, params = relay.frontend.from_onnx(onnx_model)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
            # Get device-specific TVM target
            device_config = self._get_device_config(config.target_device)
            target = tvm.target.Target(device_config["tvm_target"])
            
            # Apply optimizations
            with tvm.transform.PassContext(opt_level=config.optimization_level):
                # Apply quantization if requested
                if config.use_fp16:
                    mod = relay.transform.ToMixedPrecision()(mod)
                
                if config.use_int8:
                    # Simple dynamic quantization example
                    # In production, you'd use TVM's quantization toolkit
                    logger.warning("INT8 quantization requires calibration dataset")
                
                # Optimize for target
                lib = relay.build(mod, target=target, params=params)
            
            # Export optimized library
            lib.export_library(str(optimized_lib_path))
            
            # Save optimization metadata
            metadata = {
                "model_id": model_id,
                "original_model": str(model_path),
                "target_device": config.target_device.value,
                "optimization_config": asdict(config),
                "tvm_version": tvm.__version__,
                "input_shapes": config.input_shapes or {"input": [1, 3, 224, 224]}
            }
            
            metadata_path = output_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model optimized successfully: {optimized_lib_path}")
            return optimized_lib_path
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
    
    def generate_docker_image(self,
                            optimized_model_path: Union[str, Path],
                            model_name: str,
                            version: str = "1.0.0",
                            include_server: bool = True,
                            server_port: int = 8080) -> str:
        """
        Generate Docker image for edge deployment
        
        Args:
            optimized_model_path: Path to optimized model library
            model_name: Name of the model
            version: Version tag
            include_server: Whether to include inference server
            server_port: Port for inference server
            
        Returns:
            Docker image tag
        """
        if not HAS_DOCKER:
            raise RuntimeError("Docker SDK is required for image generation")
        
        optimized_model_path = Path(optimized_model_path)
        if not optimized_model_path.exists():
            raise FileNotFoundError(f"Optimized model not found: {optimized_model_path}")
        
        # Load metadata
        metadata_path = optimized_model_path.parent / f"{optimized_model_path.stem}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        target_device = TargetDevice(metadata["target_device"])
        device_config = self._get_device_config(target_device)
        
        # Create build context
        build_dir = self.workspace_dir / f"docker_build_{model_name}_{version}"
        build_dir.mkdir(exist_ok=True)
        
        try:
            # Copy model files
            shutil.copy2(optimized_model_path, build_dir / "model.tar")
            shutil.copy2(metadata_path, build_dir / "metadata.json")
            
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(
                base_image=device_config["docker_base"],
                model_name=model_name,
                include_server=include_server,
                server_port=server_port,
                target_device=target_device
            )
            
            with open(build_dir / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # Generate server code if requested
            if include_server:
                self._generate_inference_server(build_dir, metadata, server_port)
            
            # Generate requirements.txt
            self._generate_requirements(build_dir, include_server)
            
            # Build Docker image
            image_tag = f"forge/{model_name}:{version}-{target_device.value}"
            
            logger.info(f"Building Docker image: {image_tag}")
            image, build_logs = self.docker_client.images.build(
                path=str(build_dir),
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Docker image built successfully: {image_tag}")
            return image_tag
            
        except Exception as e:
            logger.error(f"Docker image generation failed: {e}")
            raise
        finally:
            # Cleanup build directory
            shutil.rmtree(build_dir, ignore_errors=True)
    
    def _generate_dockerfile(self,
                           base_image: str,
                           model_name: str,
                           include_server: bool,
                           server_port: int,
                           target_device: TargetDevice) -> str:
        """Generate Dockerfile content"""
        
        dockerfile = f"""# Auto-generated Dockerfile for Unsloth Edge Deployment
# Target: {target_device.value}
# Generated by Unsloth Studio Edge Optimizer

FROM {base_image}

LABEL maintainer="Unsloth Studio"
LABEL model="{model_name}"
LABEL target="{target_device.value}"

# Set environment variables
ENV MODEL_NAME={model_name}
ENV DEPLOYMENT_TYPE=edge
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy model files
COPY model.tar /app/models/
COPY metadata.json /app/models/

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

"""
        
        if include_server:
            dockerfile += f"""
# Copy server code
COPY server.py /app/

# Expose port
EXPOSE {server_port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{server_port}/health || exit 1

# Run server
CMD ["python3", "server.py"]
"""
        else:
            dockerfile += """
# No server included - model library only
CMD ["python3", "-c", "print('Model container ready')"]
"""
        
        return dockerfile
    
    def _generate_inference_server(self,
                                 build_dir: Path,
                                 metadata: Dict[str, Any],
                                 server_port: int):
        """Generate Flask-based inference server"""
        
        server_code = f'''"""
Unsloth Edge Inference Server
Auto-generated by Unsloth Studio Edge Optimizer
"""

import os
import json
import logging
from flask import Flask, request, jsonify
import numpy as np

# TVM imports
import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelServer:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load optimized model"""
        try:
            # Load metadata
            with open("/app/models/metadata.json", "r") as f:
                self.metadata = json.load(f)
            
            # Load model library
            lib_path = "/app/models/model.tar"
            loaded_lib = tvm.runtime.load_module(lib_path)
            
            # Get device context
            device = tvm.cpu()  # Use tvm.cuda() for GPU devices
            
            # Create graph executor
            self.model = runtime.GraphModule(loaded_lib["default"](device))
            
            logger.info(f"Model loaded successfully: {{self.metadata['model_id']}}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {{e}}")
            raise
    
    def predict(self, input_data):
        """Run inference"""
        try:
            # Set input
            self.model.set_input("input", input_data)
            
            # Run inference
            self.model.run()
            
            # Get output
            output = self.model.get_output(0).numpy()
            
            return output
            
        except Exception as e:
            logger.error(f"Inference failed: {{e}}")
            raise

# Initialize model server
model_server = ModelServer()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({{"status": "healthy", "model": model_server.metadata["model_id"]}})

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        
        if "input" not in data:
            return jsonify({{"error": "Missing 'input' field"}}), 400
        
        # Convert to numpy array
        input_array = np.array(data["input"], dtype=np.float32)
        
        # Run inference
        output = model_server.predict(input_array)
        
        # Return result
        return jsonify({{
            "prediction": output.tolist(),
            "model_id": model_server.metadata["model_id"],
            "status": "success"
        }})
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return jsonify({{"error": str(e)}}), 500

@app.route("/metadata", methods=["GET"])
def get_metadata():
    """Get model metadata"""
    return jsonify(model_server.metadata)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port={server_port}, debug=False)
'''
        
        with open(build_dir / "server.py", 'w') as f:
            f.write(server_code)
    
    def _generate_requirements(self, build_dir: Path, include_server: bool):
        """Generate requirements.txt"""
        
        requirements = [
            "numpy>=1.21.0",
            "apache-tvm>=0.12.0",
        ]
        
        if include_server:
            requirements.extend([
                "flask>=2.0.0",
                "requests>=2.25.0",
            ])
        
        with open(build_dir / "requirements.txt", 'w') as f:
            f.write("\n".join(requirements))
    
    def deploy_to_edge(self,
                      image_tag: str,
                      device_host: str,
                      device_user: str = "root",
                      device_port: int = 22,
                      password: Optional[str] = None,
                      key_file: Optional[Union[str, Path]] = None,
                      container_name: Optional[str] = None,
                      environment_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Deploy Docker container to edge device via SSH
        
        Args:
            image_tag: Docker image tag to deploy
            device_host: Hostname/IP of edge device
            device_user: SSH username
            device_port: SSH port
            password: SSH password (if not using key)
            key_file: Path to SSH private key
            container_name: Name for the container
            environment_vars: Environment variables for container
            
        Returns:
            Deployment status dictionary
        """
        if not HAS_DOCKER:
            raise RuntimeError("Docker SDK is required for deployment")
        
        # Generate container name if not provided
        if not container_name:
            container_name = f"forge_{image_tag.replace(':', '_').replace('/', '_')}"
        
        # Prepare SSH command
        ssh_cmd = ["ssh", f"{device_user}@{device_host}", "-p", str(device_port)]
        
        if key_file:
            ssh_cmd.extend(["-i", str(key_file)])
        
        # Check if Docker is installed on device
        check_cmd = ssh_cmd + ["which docker"]
        try:
            subprocess.run(check_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Docker not found on device {device_host}")
        
        # Save image to tar file
        image_tar_path = self.workspace_dir / f"{container_name}.tar"
        logger.info(f"Saving Docker image to {image_tar_path}")
        
        image = self.docker_client.images.get(image_tag)
        with open(image_tar_path, 'wb') as f:
            for chunk in image.save(named=True):
                f.write(chunk)
        
        # Transfer image to device
        scp_cmd = ["scp", "-P", str(device_port)]
        if key_file:
            scp_cmd.extend(["-i", str(key_file)])
        scp_cmd.extend([str(image_tar_path), f"{device_user}@{device_host}:/tmp/{container_name}.tar"])
        
        logger.info(f"Transferring image to {device_host}")
        try:
            subprocess.run(scp_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to transfer image: {e}")
        
        # Load and run container on device
        env_vars = environment_vars or {}
        env_str = " ".join([f"-e {k}={v}" for k, v in env_vars.items()])
        
        deploy_commands = [
            f"docker load -i /tmp/{container_name}.tar",
            f"docker rm -f {container_name} 2>/dev/null || true",
            f"docker run -d --name {container_name} {env_str} --restart unless-stopped {image_tag}"
        ]
        
        for cmd in deploy_commands:
            full_cmd = ssh_cmd + [cmd]
            logger.info(f"Executing on device: {cmd}")
            try:
                result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    logger.debug(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed: {e.stderr}")
                raise RuntimeError(f"Deployment failed: {e.stderr}")
        
        # Cleanup
        image_tar_path.unlink(missing_ok=True)
        
        # Update deployment registry
        deployment_info = {
            "image_tag": image_tag,
            "device_host": device_host,
            "container_name": container_name,
            "deployed_at": str(datetime.now()),
            "status": "deployed"
        }
        
        self.deployment_registry[container_name] = deployment_info
        self._save_deployment_registry()
        
        logger.info(f"Successfully deployed {image_tag} to {device_host}")
        return deployment_info
    
    def setup_ota_updates(self,
                         container_name: str,
                         update_server_url: str,
                         check_interval_hours: int = 24) -> bool:
        """
        Setup Over-The-Air updates for deployed container
        
        Args:
            container_name: Name of deployed container
            update_server_url: URL to check for updates
            check_interval_hours: How often to check for updates
            
        Returns:
            True if setup successful
        """
        if container_name not in self.deployment_registry:
            raise ValueError(f"Container {container_name} not found in registry")
        
        deployment_info = self.deployment_registry[container_name]
        device_host = deployment_info["device_host"]
        
        # Generate OTA update script
        ota_script = f'''#!/bin/bash
# Unsloth OTA Update Script
# Generated by Unsloth Studio Edge Optimizer

UPDATE_SERVER="{update_server_url}"
CONTAINER_NAME="{container_name}"
CHECK_INTERVAL={check_interval_hours * 3600}  # Convert to seconds

log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}}

check_for_updates() {{
    log "Checking for updates..."
    
    # Get current version
    CURRENT_VERSION=$(docker inspect --format='{{{{.Config.Image}}}}' $CONTAINER_NAME 2>/dev/null)
    if [ -z "$CURRENT_VERSION" ]; then
        log "Container $CONTAINER_NAME not found"
        return 1
    fi
    
    # Check for new version
    RESPONSE=$(curl -s "$UPDATE_SERVER/check?container=$CONTAINER_NAME&current=$CURRENT_VERSION")
    if [ $? -ne 0 ]; then
        log "Failed to check for updates"
        return 1
    fi
    
    NEW_VERSION=$(echo $RESPONSE | grep -o '"latest_version":"[^"]*' | cut -d'"' -f4)
    if [ -z "$NEW_VERSION" ] || [ "$NEW_VERSION" == "$CURRENT_VERSION" ]; then
        log "No updates available"
        return 0
    fi
    
    log "Update available: $NEW_VERSION"
    
    # Pull new image
    log "Pulling new image..."
    docker pull $NEW_VERSION
    if [ $? -ne 0 ]; then
        log "Failed to pull new image"
        return 1
    fi
    
    # Stop and remove old container
    log "Stopping old container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    
    # Start new container
    log "Starting new container..."
    docker run -d --name $CONTAINER_NAME --restart unless-stopped $NEW_VERSION
    if [ $? -ne 0 ]; then
        log "Failed to start new container"
        return 1
    fi
    
    log "Update completed successfully"
    return 0
}}

# Main loop
while true; do
    check_for_updates
    sleep $CHECK_INTERVAL
done
'''
        
        # Save script locally
        script_path = self.workspace_dir / f"ota_update_{container_name}.sh"
        with open(script_path, 'w') as f:
            f.write(ota_script)
        script_path.chmod(0o755)
        
        # Transfer and setup on device
        ssh_cmd = ["ssh", f"root@{device_host}"]  # Assuming root for simplicity
        
        # Transfer script
        scp_cmd = ["scp", str(script_path), f"root@{device_host}:/usr/local/bin/forge_ota.sh"]
        try:
            subprocess.run(scp_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to transfer OTA script: {e}")
            return False
        
        # Create systemd service
        service_content = f"""[Unit]
Description=Unsloth OTA Update Service for {container_name}
After=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/forge_ota.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = self.workspace_dir / f"forge-ota-{container_name}.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        # Transfer and enable service
        commands = [
            f"scp {service_path} root@{device_host}:/etc/systemd/system/",
            f"ssh root@{device_host} 'systemctl daemon-reload'",
            f"ssh root@{device_host} 'systemctl enable forge-ota-{container_name}.service'",
            f"ssh root@{device_host} 'systemctl start forge-ota-{container_name}.service'"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd.split(), check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to setup OTA service: {e}")
                return False
        
        logger.info(f"OTA updates configured for {container_name}")
        return True
    
    def list_deployments(self) -> Dict[str, Dict[str, Any]]:
        """List all deployments"""
        return self.deployment_registry.copy()
    
    def get_deployment_status(self, container_name: str) -> Dict[str, Any]:
        """Get status of a deployment"""
        if container_name not in self.deployment_registry:
            raise ValueError(f"Container {container_name} not found in registry")
        
        deployment_info = self.deployment_registry[container_name]
        device_host = deployment_info["device_host"]
        
        # Check container status via SSH
        ssh_cmd = ["ssh", f"root@{device_host}", f"docker inspect --format='{{{{.State.Status}}}}' {container_name}"]
        
        try:
            result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
            status = result.stdout.strip()
            
            deployment_info["current_status"] = status
            deployment_info["status_checked_at"] = str(datetime.now())
            
            return deployment_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get status: {e}")
            deployment_info["current_status"] = "unknown"
            return deployment_info
    
    def remove_deployment(self, container_name: str, remove_image: bool = False) -> bool:
        """Remove deployment from edge device"""
        if container_name not in self.deployment_registry:
            raise ValueError(f"Container {container_name} not found in registry")
        
        deployment_info = self.deployment_registry[container_name]
        device_host = deployment_info["device_host"]
        image_tag = deployment_info["image_tag"]
        
        # Remove container and optionally image
        commands = [
            f"docker rm -f {container_name}",
        ]
        
        if remove_image:
            commands.append(f"docker rmi {image_tag}")
        
        ssh_cmd = ["ssh", f"root@{device_host}"]
        
        for cmd in commands:
            full_cmd = ssh_cmd + [cmd]
            try:
                subprocess.run(full_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Command failed: {e.stderr}")
        
        # Remove from registry
        del self.deployment_registry[container_name]
        self._save_deployment_registry()
        
        logger.info(f"Removed deployment {container_name}")
        return True


# Convenience functions for CLI/API integration
def quick_deploy(model_path: Union[str, Path],
                target_device: Union[str, TargetDevice],
                device_host: str,
                device_user: str = "root",
                **kwargs) -> Dict[str, Any]:
    """
    One-click deployment function
    
    Args:
        model_path: Path to model file
        target_device: Target device (string or enum)
        device_host: Hostname/IP of edge device
        device_user: SSH username
        **kwargs: Additional arguments for optimization and deployment
        
    Returns:
        Deployment information
    """
    # Convert string to enum if needed
    if isinstance(target_device, str):
        target_device = TargetDevice(target_device)
    
    # Initialize optimizer
    optimizer = EdgeOptimizer()
    
    # Create optimization config
    config = OptimizationConfig(
        target_device=target_device,
        optimization_level=kwargs.get("optimization_level", 3),
        use_fp16=kwargs.get("use_fp16", False),
        use_int8=kwargs.get("use_int8", False),
        batch_size=kwargs.get("batch_size", 1),
        input_shapes=kwargs.get("input_shapes")
    )
    
    # Optimize model
    optimized_path = optimizer.optimize_model(model_path, config)
    
    # Generate Docker image
    model_name = Path(model_path).stem
    version = kwargs.get("version", "1.0.0")
    image_tag = optimizer.generate_docker_image(
        optimized_path,
        model_name,
        version,
        include_server=kwargs.get("include_server", True),
        server_port=kwargs.get("server_port", 8080)
    )
    
    # Deploy to device
    deployment_info = optimizer.deploy_to_edge(
        image_tag,
        device_host,
        device_user,
        password=kwargs.get("password"),
        key_file=kwargs.get("key_file"),
        container_name=kwargs.get("container_name"),
        environment_vars=kwargs.get("environment_vars")
    )
    
    # Setup OTA updates if requested
    if kwargs.get("enable_ota", False):
        update_server_url = kwargs.get("update_server_url", "")
        if update_server_url:
            optimizer.setup_ota_updates(
                deployment_info["container_name"],
                update_server_url,
                kwargs.get("ota_check_interval_hours", 24)
            )
    
    return {
        "optimized_model": str(optimized_path),
        "docker_image": image_tag,
        "deployment": deployment_info,
        "status": "success"
    }


# Import datetime for timestamps
from datetime import datetime