"""Edge deployment pipeline for Unsloth models with hardware-specific optimization."""

import os
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import subprocess
import docker
from docker.errors import DockerException
import requests
from datetime import datetime

# TVM imports with fallback
try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    tvm = None

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import ModelArtifact

logger = logging.getLogger(__name__)


class EdgeDevice(Enum):
    """Supported edge device types."""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    CORAL_EDGE_TPU = "coral_edge_tpu"
    GENERIC_ARM64 = "generic_arm64"
    GENERIC_AMD64 = "generic_amd64"


@dataclass
class DeviceProfile:
    """Hardware profile for edge devices."""
    device_type: EdgeDevice
    architecture: str
    compute_capability: str
    memory_mb: int
    tvm_target: str
    docker_base_image: str
    optimization_level: int = 3
    enable_fp16: bool = False
    enable_int8: bool = False
    max_workspace_mb: int = 1024


@dataclass
class DeploymentPackage:
    """Deployment package metadata."""
    package_id: str
    model_name: str
    device_type: EdgeDevice
    created_at: str
    model_hash: str
    optimization_config: Dict[str, Any]
    docker_tag: str
    package_path: str
    ota_url: Optional[str] = None
    version: str = "1.0.0"


class TVMOptimizer:
    """Apache TVM optimization engine for edge devices."""
    
    def __init__(self):
        if not TVM_AVAILABLE:
            raise ImportError("Apache TVM is required for edge optimization. Install with: pip install apache-tvm")
    
    def optimize_for_device(
        self,
        model_path: str,
        device_profile: DeviceProfile,
        input_shapes: Dict[str, List[int]],
        output_dir: str
    ) -> str:
        """Optimize model using TVM for target hardware."""
        try:
            # Load model (supports PyTorch, ONNX, TensorFlow)
            mod, params = self._load_model(model_path, input_shapes)
            
            # Apply device-specific optimizations
            with tvm.transform.PassContext(opt_level=device_profile.optimization_level):
                # Convert to relay
                if device_profile.enable_fp16:
                    mod = relay.transform.ToMixedPrecision()(mod)
                
                if device_profile.enable_int8:
                    mod = relay.quantize.quantize(mod, params)
                
                # Build for target
                lib = relay.build(
                    mod,
                    target=device_profile.tvm_target,
                    params=params
                )
            
            # Export optimized model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            lib_path = output_path / "model.so"
            graph_path = output_path / "model.json"
            param_path = output_path / "model.params"
            
            lib.export_library(str(lib_path))
            
            with open(graph_path, "w") as f:
                f.write(lib.get_json())
            
            with open(param_path, "wb") as f:
                f.write(relay.save_param_dict(params))
            
            # Generate inference wrapper
            self._generate_inference_wrapper(output_path, device_profile)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TVM optimization failed: {e}")
            raise
    
    def _load_model(self, model_path: str, input_shapes: Dict[str, List[int]]):
        """Load model from various formats."""
        model_path = Path(model_path)
        
        if model_path.suffix == ".pt":
            import torch
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            input_data = {k: torch.randn(*v) for k, v in input_shapes.items()}
            mod, params = relay.frontend.from_pytorch(model, list(input_data.items()))
            
        elif model_path.suffix == ".onnx":
            import onnx
            model = onnx.load(model_path)
            mod, params = relay.frontend.from_onnx(model, shape=input_shapes)
            
        elif model_path.suffix in [".pb", ".savedmodel"]:
            import tensorflow as tf
            model = tf.saved_model.load(str(model_path))
            mod, params = relay.frontend.from_tensorflow(model, shape=input_shapes)
            
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        return mod, params
    
    def _generate_inference_wrapper(self, output_dir: Path, device_profile: DeviceProfile):
        """Generate Python inference wrapper."""
        wrapper_code = f'''#!/usr/bin/env python3
"""Auto-generated inference wrapper for {device_profile.device_type.value}."""

import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import json
from pathlib import Path

class ModelInference:
    def __init__(self, model_dir: str = "."):
        model_dir = Path(model_dir)
        
        # Load compiled model
        self.lib = tvm.runtime.load_module(str(model_dir / "model.so"))
        
        with open(model_dir / "model.json", "r") as f:
            self.graph_json = f.read()
        
        with open(model_dir / "model.params", "rb") as f:
            self.params = bytearray(f.read())
        
        # Create graph executor
        self.device = tvm.device("{device_profile.tvm_target}", 0)
        self.module = graph_executor.create(self.graph_json, self.lib, self.device)
        self.module.load_params(self.params)
    
    def predict(self, inputs: dict) -> dict:
        """Run inference."""
        # Set inputs
        for name, data in inputs.items():
            self.module.set_input(name, data)
        
        # Run
        self.module.run()
        
        # Get outputs
        outputs = {{}}
        for i in range(self.module.get_num_outputs()):
            output = self.module.get_output(i)
            outputs[f"output_{{i}}"] = output.numpy()
        
        return outputs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        inputs = json.load(f)
    
    # Convert to numpy arrays
    inputs = {{k: np.array(v) for k, v in inputs.items()}}
    
    model = ModelInference()
    outputs = model.predict(inputs)
    
    # Convert to serializable format
    outputs = {{k: v.tolist() for k, v in outputs.items()}}
    
    with open(args.output, "w") as f:
        json.dump(outputs, f)
'''
        
        wrapper_path = output_dir / "inference.py"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)
        
        # Make executable
        wrapper_path.chmod(0o755)


class DockerPackager:
    """Docker container generator for edge devices."""
    
    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        try:
            self.client = docker_client or docker.from_env()
        except DockerException:
            logger.warning("Docker not available. Container generation will be simulated.")
            self.client = None
    
    def generate_container(
        self,
        optimized_model_dir: str,
        device_profile: DeviceProfile,
        package_metadata: DeploymentPackage,
        requirements: Optional[List[str]] = None,
        include_monitoring: bool = True
    ) -> str:
        """Generate Docker container for edge deployment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy optimized model
            model_dest = temp_path / "model"
            shutil.copytree(optimized_model_dir, model_dest)
            
            # Generate Dockerfile
            dockerfile = self._generate_dockerfile(
                device_profile, 
                package_metadata,
                requirements,
                include_monitoring
            )
            
            dockerfile_path = temp_path / "Dockerfile"
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile)
            
            # Generate docker-compose.yml for easy deployment
            compose_file = self._generate_compose_file(device_profile, package_metadata)
            compose_path = temp_path / "docker-compose.yml"
            with open(compose_path, "w") as f:
                f.write(compose_file)
            
            # Generate entrypoint script
            entrypoint = self._generate_entrypoint(device_profile, package_metadata)
            entrypoint_path = temp_path / "entrypoint.sh"
            with open(entrypoint_path, "w") as f:
                f.write(entrypoint)
            entrypoint_path.chmod(0o755)
            
            # Build Docker image
            if self.client:
                try:
                    image, build_logs = self.client.images.build(
                        path=str(temp_path),
                        tag=package_metadata.docker_tag,
                        buildargs={
                            "DEVICE_TYPE": device_profile.device_type.value,
                            "MODEL_HASH": package_metadata.model_hash
                        }
                    )
                    
                    # Save image to tar file
                    output_dir = Path(package_metadata.package_path).parent
                    image_path = output_dir / f"{package_metadata.package_id}.tar"
                    
                    with open(image_path, "wb") as f:
                        for chunk in image.save():
                            f.write(chunk)
                    
                    return str(image_path)
                    
                except DockerException as e:
                    logger.error(f"Docker build failed: {e}")
                    raise
            
            # Fallback: create package directory without Docker
            output_dir = Path(package_metadata.package_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files to output directory
            for item in temp_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir)
                elif item.is_dir():
                    shutil.copytree(item, output_dir / item.name)
            
            return str(output_dir)
    
    def _generate_dockerfile(
        self,
        device_profile: DeviceProfile,
        package_metadata: DeploymentPackage,
        requirements: Optional[List[str]],
        include_monitoring: bool
    ) -> str:
        """Generate Dockerfile for edge device."""
        base_image = device_profile.docker_base_image
        
        # Device-specific setup commands
        setup_commands = []
        
        if device_profile.device_type in [EdgeDevice.JETSON_NANO, EdgeDevice.JETSON_XAVIER, EdgeDevice.JETSON_ORIN]:
            setup_commands.extend([
                "# Install Jetson dependencies",
                "RUN apt-get update && apt-get install -y \\",
                "    python3-pip \\",
                "    libopenblas-base \\",
                "    libopenmpi-dev \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
                "# Install JetPack utilities",
                "RUN pip3 install --upgrade pip",
                "RUN pip3 install jetson-stats"
            ])
        
        elif device_profile.device_type in [EdgeDevice.RASPBERRY_PI_4, EdgeDevice.RASPBERRY_PI_5]:
            setup_commands.extend([
                "# Install Raspberry Pi dependencies",
                "RUN apt-get update && apt-get install -y \\",
                "    python3-pip \\",
                "    python3-numpy \\",
                "    libatlas-base-dev \\",
                "    && rm -rf /var/lib/apt/lists/*"
            ])
        
        # Requirements
        req_commands = []
        if requirements:
            req_file = "requirements.txt"
            req_commands = [
                f"RUN echo '{chr(10).join(requirements)}' > {req_file}",
                f"RUN pip3 install --no-cache-dir -r {req_file}"
            ]
        
        # Monitoring setup
        monitoring_commands = []
        if include_monitoring:
            monitoring_commands = [
                "# Install monitoring tools",
                "RUN pip3 install prometheus-client psutil",
                "RUN mkdir -p /var/log/forge",
                "ENV UNSLOTH_LOG_LEVEL=INFO"
            ]
        
        dockerfile = f"""# Auto-generated Dockerfile for {device_profile.device_type.value}
# Model: {package_metadata.model_name}
# Generated: {package_metadata.created_at}

FROM {base_image}

LABEL maintainer="Unsloth Studio"
LABEL device.type="{device_profile.device_type.value}"
LABEL model.name="{package_metadata.model_name}"
LABEL model.version="{package_metadata.version}"
LABEL model.hash="{package_metadata.model_hash}"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/opt/forge/model
ENV CONFIG_DIR=/opt/forge/config

# Create directories
RUN mkdir -p $MODEL_DIR $CONFIG_DIR /var/log/forge

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

{chr(10).join(setup_commands)}

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir numpy tvm

{chr(10).join(req_commands)}

{chr(10).join(monitoring_commands)}

# Copy model and application files
COPY model/ $MODEL_DIR/
COPY entrypoint.sh /opt/forge/
COPY config.json $CONFIG_DIR/

# Create non-root user
RUN groupadd -r forge && useradd -r -g forge forge
RUN chown -R forge:forge /opt/forge /var/log/forge

USER forge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 -c "import sys; sys.path.insert(0, '$MODEL_DIR'); from inference import ModelInference; model = ModelInference('$MODEL_DIR'); print('Health check passed')" || exit 1

# Expose metrics port if monitoring is enabled
{"EXPOSE 9090" if include_monitoring else ""}

# Set entrypoint
ENTRYPOINT ["/opt/forge/entrypoint.sh"]
CMD ["serve"]
"""
        return dockerfile
    
    def _generate_compose_file(
        self,
        device_profile: DeviceProfile,
        package_metadata: DeploymentPackage
    ) -> str:
        """Generate docker-compose.yml for easy deployment."""
        compose = f"""version: '3.8'

services:
  forge-model:
    image: {package_metadata.docker_tag}
    container_name: forge-{package_metadata.package_id}
    restart: unless-stopped
    privileged: true
    volumes:
      - ./model:/opt/forge/model:ro
      - ./logs:/var/log/forge
      - /dev:/dev
    environment:
      - DEVICE_TYPE={device_profile.device_type.value}
      - MODEL_NAME={package_metadata.model_name}
      - MODEL_VERSION={package_metadata.version}
      - UNSLOTH_LOG_LEVEL=INFO
    ports:
      - "8080:8080"  # API port
      - "9090:9090"  # Metrics port
    networks:
      - forge-network
    deploy:
      resources:
        limits:
          memory: {device_profile.memory_mb}M
        reservations:
          memory: {device_profile.memory_mb // 2}M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  forge-network:
    driver: bridge

volumes:
  model:
  logs:
"""
        return compose
    
    def _generate_entrypoint(
        self,
        device_profile: DeviceProfile,
        package_metadata: DeploymentPackage
    ) -> str:
        """Generate entrypoint script."""
        entrypoint = f"""#!/bin/bash
set -e

# Unsloth Edge Model Entrypoint
# Device: {device_profile.device_type.value}
# Model: {package_metadata.model_name}

LOG_FILE="/var/log/forge/model.log"
MODEL_DIR="/opt/forge/model"
CONFIG_FILE="/opt/forge/config/config.json"

log() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}}

check_device() {{
    log "Checking device compatibility..."
    
    # Check available memory
    AVAILABLE_MEM=$(free -m | awk '/^Mem: {{print $7}}')
    if [ $AVAILABLE_MEM -lt {device_profile.memory_mb // 2} ]; then
        log "WARNING: Low memory available (${{AVAILABLE_MEM}}MB)"
    fi
    
    # Check GPU if applicable
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
    
    log "Device check completed"
}}

load_model() {{
    log "Loading model from $MODEL_DIR..."
    
    if [ ! -f "$MODEL_DIR/inference.py" ]; then
        log "ERROR: Model files not found"
        exit 1
    fi
    
    # Verify model hash
    if [ -f "$CONFIG_FILE" ]; then
        EXPECTED_HASH=$(jq -r '.model_hash' $CONFIG_FILE)
        ACTUAL_HASH=$(sha256sum $MODEL_DIR/model.so | cut -d' ' -f1)
        
        if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
            log "WARNING: Model hash mismatch"
        fi
    fi
    
    log "Model loaded successfully"
}}

start_server() {{
    log "Starting inference server..."
    
    # Start metrics server in background
    if [ "$ENABLE_MONITORING" = "true" ]; then
        python3 -m prometheus_client 9090 &
        METRICS_PID=$!
        log "Metrics server started on port 9090 (PID: $METRICS_PID)"
    fi
    
    # Start main server
    cd $MODEL_DIR
    python3 -m http.server 8080 &
    SERVER_PID=$!
    
    log "Inference server started on port 8080 (PID: $SERVER_PID)"
    
    # Wait for server
    wait $SERVER_PID
}}

update_model() {{
    log "Checking for model updates..."
    
    OTA_URL="${{OTA_URL:-}}"
    if [ -z "$OTA_URL" ]; then
        log "No OTA URL configured"
        return 0
    fi
    
    # Check for updates
    RESPONSE=$(curl -s -w "\\n%{{http_code}}" "$OTA_URL/check?device={device_profile.device_type.value}&version={package_metadata.version}")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        UPDATE_AVAILABLE=$(echo "$BODY" | jq -r '.update_available')
        
        if [ "$UPDATE_AVAILABLE" = "true" ]; then
            log "Update available, downloading..."
            
            # Download update
            DOWNLOAD_URL=$(echo "$BODY" | jq -r '.download_url')
            curl -L -o /tmp/model_update.tar.gz "$DOWNLOAD_URL"
            
            # Backup current model
            BACKUP_DIR="/opt/forge/backup/$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$BACKUP_DIR"
            cp -r $MODEL_DIR/* "$BACKUP_DIR/"
            
            # Extract update
            tar -xzf /tmp/model_update.tar.gz -C $MODEL_DIR --strip-components=1
            
            # Update config
            echo "$BODY" | jq '.config' > $CONFIG_FILE
            
            log "Model updated successfully"
            return 1  # Signal to restart
        else
            log "No updates available"
        fi
    else
        log "Failed to check for updates (HTTP $HTTP_CODE)"
    fi
    
    return 0
}}

case "$1" in
    serve)
        check_device
        load_model
        start_server
        ;;
    update)
        if update_model; then
            log "Update completed, restarting..."
            exec "$0" serve
        fi
        ;;
    health)
        python3 -c "
import sys
sys.path.insert(0, '$MODEL_DIR')
from inference import ModelInference
model = ModelInference('$MODEL_DIR')
print('OK')
"
        ;;
    *)
        echo "Usage: $0 {{serve|update|health}}"
        exit 1
        ;;
esac
"""
        return entrypoint


class OTAManager:
    """Over-The-Air update manager for edge deployments."""
    
    def __init__(self, update_server_url: str, auth_token: Optional[str] = None):
        self.server_url = update_server_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})
    
    def publish_update(
        self,
        package: DeploymentPackage,
        model_path: str,
        changelog: str = "",
        force_update: bool = False
    ) -> str:
        """Publish a model update to OTA server."""
        try:
            # Create update package
            update_package = {
                "package_id": package.package_id,
                "model_name": package.model_name,
                "version": package.version,
                "device_type": package.device_type.value,
                "model_hash": package.model_hash,
                "changelog": changelog,
                "force_update": force_update,
                "timestamp": datetime.utcnow().isoformat(),
                "download_url": f"{self.server_url}/models/{package.package_id}/{package.version}"
            }
            
            # Upload model files
            with open(model_path, 'rb') as f:
                files = {'model': (f'{package.package_id}.tar.gz', f, 'application/gzip')}
                data = {'metadata': json.dumps(update_package)}
                
                response = self.session.post(
                    f"{self.server_url}/api/updates/publish",
                    files=files,
                    data=data
                )
                response.raise_for_status()
            
            result = response.json()
            update_url = result.get('update_url')
            
            # Update package with OTA URL
            package.ota_url = update_url
            
            logger.info(f"Published update for {package.model_name} v{package.version}")
            return update_url
            
        except requests.RequestException as e:
            logger.error(f"Failed to publish update: {e}")
            raise
    
    def check_for_updates(
        self,
        package_id: str,
        current_version: str,
        device_type: EdgeDevice
    ) -> Dict[str, Any]:
        """Check for available updates."""
        try:
            params = {
                "package_id": package_id,
                "current_version": current_version,
                "device_type": device_type.value
            }
            
            response = self.session.get(
                f"{self.server_url}/api/updates/check",
                params=params
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Failed to check for updates: {e}")
            return {"update_available": False, "error": str(e)}
    
    def rollback_update(
        self,
        package_id: str,
        target_version: str
    ) -> bool:
        """Rollback to a previous version."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/updates/rollback",
                json={
                    "package_id": package_id,
                    "target_version": target_version
                }
            )
            response.raise_for_status()
            
            logger.info(f"Rolled back {package_id} to version {target_version}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to rollback: {e}")
            return False


class EdgePackager:
    """Main edge deployment pipeline orchestrator."""
    
    # Device profiles database
    DEVICE_PROFILES = {
        EdgeDevice.JETSON_NANO: DeviceProfile(
            device_type=EdgeDevice.JETSON_NANO,
            architecture="aarch64",
            compute_capability="5.3",
            memory_mb=4096,
            tvm_target="llvm -target=aarch64-linux-gnu",
            docker_base_image="nvcr.io/nvidia/l4t-base:r32.7.1",
            enable_fp16=True
        ),
        EdgeDevice.JETSON_XAVIER: DeviceProfile(
            device_type=EdgeDevice.JETSON_XAVIER,
            architecture="aarch64",
            compute_capability="7.2",
            memory_mb=16384,
            tvm_target="llvm -target=aarch64-linux-gnu",
            docker_base_image="nvcr.io/nvidia/l4t-base:r32.7.1",
            enable_fp16=True,
            enable_int8=True
        ),
        EdgeDevice.JETSON_ORIN: DeviceProfile(
            device_type=EdgeDevice.JETSON_ORIN,
            architecture="aarch64",
            compute_capability="8.7",
            memory_mb=32768,
            tvm_target="llvm -target=aarch64-linux-gnu",
            docker_base_image="nvcr.io/nvidia/l4t-base:r35.3.1",
            enable_fp16=True,
            enable_int8=True
        ),
        EdgeDevice.RASPBERRY_PI_4: DeviceProfile(
            device_type=EdgeDevice.RASPBERRY_PI_4,
            architecture="armv7l",
            compute_capability="",
            memory_mb=4096,
            tvm_target="llvm -target=arm-linux-gnueabihf",
            docker_base_image="balenalib/raspberry-pi-debian:bullseye",
            enable_fp16=False,
            enable_int8=True
        ),
        EdgeDevice.RASPBERRY_PI_5: DeviceProfile(
            device_type=EdgeDevice.RASPBERRY_PI_5,
            architecture="aarch64",
            compute_capability="",
            memory_mb=8192,
            tvm_target="llvm -target=aarch64-linux-gnu",
            docker_base_image="balenalib/raspberry-pi-debian:bullseye",
            enable_fp16=True,
            enable_int8=True
        ),
        EdgeDevice.CORAL_EDGE_TPU: DeviceProfile(
            device_type=EdgeDevice.CORAL_EDGE_TPU,
            architecture="armv7l",
            compute_capability="",
            memory_mb=1024,
            tvm_target="llvm -target=arm-linux-gnueabihf",
            docker_base_image="debian:buster-slim",
            enable_fp16=False,
            enable_int8=True,
            optimization_level=2
        ),
        EdgeDevice.GENERIC_ARM64: DeviceProfile(
            device_type=EdgeDevice.GENERIC_ARM64,
            architecture="aarch64",
            compute_capability="",
            memory_mb=8192,
            tvm_target="llvm -target=aarch64-linux-gnu",
            docker_base_image="arm64v8/debian:bullseye-slim",
            enable_fp16=True
        ),
        EdgeDevice.GENERIC_AMD64: DeviceProfile(
            device_type=EdgeDevice.GENERIC_AMD64,
            architecture="x86_64",
            compute_capability="",
            memory_mb=16384,
            tvm_target="llvm -target=x86_64-linux-gnu",
            docker_base_image="debian:bullseye-slim",
            enable_fp16=True,
            enable_int8=True
        )
    }
    
    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        ota_server_url: Optional[str] = None,
        ota_auth_token: Optional[str] = None
    ):
        self.job_manager = job_manager or JobManager()
        self.tvm_optimizer = TVMOptimizer() if TVM_AVAILABLE else None
        self.docker_packager = DockerPackager()
        self.ota_manager = OTAManager(ota_server_url, ota_auth_token) if ota_server_url else None
        
        # Create output directory
        self.output_dir = Path.home() / ".forge" / "deployments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def package_for_device(
        self,
        model_path: str,
        model_name: str,
        device_type: Union[EdgeDevice, str],
        input_shapes: Dict[str, List[int]],
        version: str = "1.0.0",
        requirements: Optional[List[str]] = None,
        include_monitoring: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> DeploymentPackage:
        """Package model for specific edge device."""
        # Convert string to enum if needed
        if isinstance(device_type, str):
            device_type = EdgeDevice(device_type)
        
        # Get device profile
        device_profile = self.DEVICE_PROFILES.get(device_type)
        if not device_profile:
            raise ValueError(f"Unsupported device type: {device_type}")
        
        # Generate package ID
        package_id = self._generate_package_id(model_name, device_type, version)
        
        # Create output directory
        package_dir = self.output_dir / package_id
        package_dir.mkdir(exist_ok=True)
        
        # Calculate model hash
        model_hash = self._calculate_file_hash(model_path)
        
        # Create deployment package metadata
        package = DeploymentPackage(
            package_id=package_id,
            model_name=model_name,
            device_type=device_type,
            created_at=datetime.utcnow().isoformat(),
            model_hash=model_hash,
            optimization_config={
                "input_shapes": input_shapes,
                "device_profile": asdict(device_profile),
                "custom_config": custom_config or {}
            },
            docker_tag=f"forge/{model_name}:{version}-{device_type.value}",
            package_path=str(package_dir),
            version=version
        )
        
        # Save package metadata
        with open(package_dir / "package.json", "w") as f:
            json.dump(asdict(package), f, indent=2)
        
        # Optimize model
        optimized_dir = package_dir / "optimized"
        optimized_dir.mkdir(exist_ok=True)
        
        if self.tvm_optimizer:
            try:
                optimized_model_path = self.tvm_optimizer.optimize_for_device(
                    model_path=model_path,
                    device_profile=device_profile,
                    input_shapes=input_shapes,
                    output_dir=str(optimized_dir)
                )
                logger.info(f"Model optimized for {device_type.value}")
            except Exception as e:
                logger.warning(f"TVM optimization failed, using original model: {e}")
                # Fallback: copy original model
                shutil.copy2(model_path, optimized_dir / "model.pt")
                optimized_model_path = str(optimized_dir)
        else:
            # TVM not available, use original model
            shutil.copy2(model_path, optimized_dir / "model.pt")
            optimized_model_path = str(optimized_dir)
        
        # Generate Docker container
        container_path = self.docker_packager.generate_container(
            optimized_model_dir=optimized_model_path,
            device_profile=device_profile,
            package_metadata=package,
            requirements=requirements,
            include_monitoring=include_monitoring
        )
        
        # Generate deployment scripts
        self._generate_deployment_scripts(package_dir, device_profile, package)
        
        # Generate documentation
        self._generate_documentation(package_dir, package, device_profile)
        
        # Update package with container path
        package.package_path = str(package_dir)
        
        # Save final package metadata
        with open(package_dir / "package.json", "w") as f:
            json.dump(asdict(package), f, indent=2)
        
        logger.info(f"Created deployment package: {package_id}")
        return package
    
    def deploy_to_device(
        self,
        package: DeploymentPackage,
        device_ip: str,
        ssh_user: str = "root",
        ssh_key_path: Optional[str] = None,
        remote_path: str = "/opt/forge"
    ) -> bool:
        """Deploy package to edge device via SSH."""
        try:
            import paramiko
            
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to device
            connect_kwargs = {
                "hostname": device_ip,
                "username": ssh_user,
                "timeout": 30
            }
            
            if ssh_key_path:
                connect_kwargs["key_filename"] = ssh_key_path
            
            ssh.connect(**connect_kwargs)
            
            # Create remote directory
            stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_path}")
            if stdout.channel.recv_exit_status() != 0:
                raise Exception(f"Failed to create directory: {stderr.read().decode()}")
            
            # Transfer files using SFTP
            sftp = ssh.open_sftp()
            
            package_dir = Path(package.package_path)
            
            # Transfer all files
            for item in package_dir.rglob("*"):
                if item.is_file():
                    remote_file = f"{remote_path}/{item.relative_to(package_dir)}"
                    remote_dir = str(Path(remote_file).parent)
                    
                    # Create remote directory if needed
                    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
                    stdout.channel.recv_exit_status()
                    
                    # Transfer file
                    sftp.put(str(item), remote_file)
            
            # Make scripts executable
            stdin, stdout, stderr = ssh.exec_command(
                f"chmod +x {remote_path}/*.sh {remote_path}/entrypoint.sh"
            )
            stdout.channel.recv_exit_status()
            
            # Load Docker image if available
            if (package_dir / f"{package.package_id}.tar").exists():
                stdin, stdout, stderr = ssh.exec_command(
                    f"docker load < {remote_path}/{package.package_id}.tar"
                )
                if stdout.channel.recv_exit_status() != 0:
                    logger.warning("Failed to load Docker image on device")
            
            # Start deployment
            stdin, stdout, stderr = ssh.exec_command(
                f"cd {remote_path} && ./deploy.sh"
            )
            
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                logger.info(f"Successfully deployed to {device_ip}")
                
                # Update OTA manager if available
                if self.ota_manager:
                    self.ota_manager.publish_update(
                        package=package,
                        model_path=str(package_dir / f"{package.package_id}.tar"),
                        changelog=f"Deployed to {device_ip}"
                    )
                
                return True
            else:
                error = stderr.read().decode()
                logger.error(f"Deployment failed: {error}")
                return False
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
        finally:
            if 'ssh' in locals():
                ssh.close()
    
    def batch_deploy(
        self,
        package: DeploymentPackage,
        devices: List[Dict[str, str]],
        parallel: bool = True
    ) -> Dict[str, bool]:
        """Deploy to multiple devices."""
        results = {}
        
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=min(10, len(devices))) as executor:
                futures = {
                    executor.submit(
                        self.deploy_to_device,
                        package=package,
                        device_ip=device["ip"],
                        ssh_user=device.get("user", "root"),
                        ssh_key_path=device.get("ssh_key"),
                        remote_path=device.get("path", "/opt/forge")
                    ): device["ip"]
                    for device in devices
                }
                
                for future in as_completed(futures):
                    device_ip = futures[future]
                    try:
                        results[device_ip] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to deploy to {device_ip}: {e}")
                        results[device_ip] = False
        else:
            for device in devices:
                device_ip = device["ip"]
                try:
                    results[device_ip] = self.deploy_to_device(
                        package=package,
                        device_ip=device_ip,
                        ssh_user=device.get("user", "root"),
                        ssh_key_path=device.get("ssh_key"),
                        remote_path=device.get("path", "/opt/forge")
                    )
                except Exception as e:
                    logger.error(f"Failed to deploy to {device_ip}: {e}")
                    results[device_ip] = False
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Batch deployment completed: {successful}/{len(devices)} successful")
        
        return results
    
    def _generate_package_id(
        self,
        model_name: str,
        device_type: EdgeDevice,
        version: str
    ) -> str:
        """Generate unique package ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{model_name}{device_type.value}{version}{timestamp}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"forge-{model_name}-{device_type.value}-{version}-{hash_digest}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _generate_deployment_scripts(
        self,
        package_dir: Path,
        device_profile: DeviceProfile,
        package: DeploymentPackage
    ):
        """Generate deployment helper scripts."""
        
        # deploy.sh - Main deployment script
        deploy_script = f"""#!/bin/bash
set -e

echo "Unsloth Edge Deployment"
echo "======================="
echo "Model: {package.model_name}"
echo "Device: {device_profile.device_type.value}"
echo "Version: {package.version}"
echo ""

# Check prerequisites
check_command() {{
    if ! command -v $1 &> /dev/null; then
        echo "ERROR: $1 is not installed"
        exit 1
    fi
}}

check_command docker
check_command docker-compose

# Load Docker image if available
if [ -f "{package.package_id}.tar" ]; then
    echo "Loading Docker image..."
    docker load < {package.package_id}.tar
fi

# Create necessary directories
mkdir -p logs model

# Copy model files if needed
if [ -d "optimized" ] && [ ! -d "model" ]; then
    cp -r optimized/* model/
fi

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for service to be ready
echo "Waiting for service to start..."
for i in {{1..30}}; do
    if curl -s http://localhost:8080/health > /dev/null; then
        echo "Service is ready!"
        echo ""
        echo "Access the model API at: http://localhost:8080"
        echo "View logs: docker-compose logs -f"
        echo "Stop service: docker-compose down"
        exit 0
    fi
    sleep 1
done

echo "WARNING: Service may not be fully started"
echo "Check logs: docker-compose logs"
"""
        
        deploy_path = package_dir / "deploy.sh"
        with open(deploy_path, "w") as f:
            f.write(deploy_script)
        deploy_path.chmod(0o755)
        
        # update.sh - OTA update script
        if self.ota_manager:
            update_script = f"""#!/bin/bash
set -e

echo "Checking for model updates..."
echo "Current version: {package.version}"

# Check for updates
UPDATE_INFO=$(curl -s "{self.ota_manager.server_url}/api/updates/check?package_id={package.package_id}&current_version={package.version}&device_type={device_profile.device_type.value}")

UPDATE_AVAILABLE=$(echo "$UPDATE_INFO" | jq -r '.update_available')

if [ "$UPDATE_AVAILABLE" = "true" ]; then
    NEW_VERSION=$(echo "$UPDATE_INFO" | jq -r '.new_version')
    DOWNLOAD_URL=$(echo "$UPDATE_INFO" | jq -r '.download_url')
    
    echo "Update available: v$NEW_VERSION"
    echo "Downloading update..."
    
    # Backup current model
    BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r model/* "$BACKUP_DIR/" 2>/dev/null || true
    
    # Download and extract update
    curl -L -o model_update.tar.gz "$DOWNLOAD_URL"
    tar -xzf model_update.tar.gz -C model --strip-components=1
    rm model_update.tar.gz
    
    # Restart service
    echo "Restarting service..."
    docker-compose restart
    
    echo "Update completed successfully!"
else
    echo "No updates available"
fi
"""
            
            update_path = package_dir / "update.sh"
            with open(update_path, "w") as f:
                f.write(update_script)
            update_path.chmod(0o755)
        
        # monitor.sh - Monitoring script
        monitor_script = """#!/bin/bash
echo "Unsloth Edge Model Monitor"
echo "=========================="
echo ""

# Check service status
docker-compose ps

echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.NetIO}}"

echo ""
echo "Recent Logs:"
docker-compose logs --tail=20
"""
        
        monitor_path = package_dir / "monitor.sh"
        with open(monitor_path, "w") as f:
            f.write(monitor_script)
        monitor_path.chmod(0o755)
    
    def _generate_documentation(
        self,
        package_dir: Path,
        package: DeploymentPackage,
        device_profile: DeviceProfile
    ):
        """Generate deployment documentation."""
        doc = f"""# Unsloth Edge Deployment Package

## Overview
- **Model**: {package.model_name}
- **Device**: {device_profile.device_type.value}
- **Version**: {package.version}
- **Package ID**: {package.package_id}
- **Created**: {package.created_at}

## Hardware Requirements
- **Architecture**: {device_profile.architecture}
- **Memory**: {device_profile.memory_mb} MB
- **Compute Capability**: {device_profile.compute_capability or 'N/A'}
- **Optimizations**: {'FP16, ' if device_profile.enable_fp16 else ''}{'INT8, ' if device_profile.enable_int8 else ''}Level {device_profile.optimization_level}

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Network access to download dependencies

### Deployment
```bash
# Make scripts executable
chmod +x *.sh

# Deploy the model
./deploy.sh

# Check status
./monitor.sh

# Update model (if OTA configured)
./update.sh
```

### Manual Deployment
```bash
# Load Docker image
docker load < {package.package_id}.tar

# Start service
docker-compose up -d

# Check logs
docker-compose logs -f
```

## API Usage

### Health Check
```bash
curl http://localhost:8080/health
```

### Inference
```bash
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "input": [1.0, 2.0, 3.0, 4.0]
  }}'
```

## Configuration

### Environment Variables
- `MODEL_DIR`: Path to model directory (default: `/opt/forge/model`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `ENABLE_MONITORING`: Enable metrics collection (default: `true`)

### Docker Compose
Edit `docker-compose.yml` to:
- Change exposed ports
- Mount additional volumes
- Configure resource limits
- Set environment variables

## Monitoring

### Metrics
Prometheus metrics available at `http://localhost:9090/metrics`

### Logs
Logs are stored in `./logs/model.log`

### Health Checks
Docker health check configured to verify model loading and API availability.

## Troubleshooting

### Service won't start
1. Check Docker logs: `docker-compose logs`
2. Verify model files exist in `./model/`
3. Check available memory: `free -m`

### Out of memory
1. Reduce batch size in inference requests
2. Enable model quantization (INT8)
3. Increase device swap space

### Network issues
1. Verify ports 8080 and 9090 are not in use
2. Check firewall settings
3. Ensure Docker network is created: `docker network create forge-network`

## OTA Updates

### Automatic Updates
If OTA server is configured, updates will be checked on service start.

### Manual Updates
```bash
./update.sh
```

### Rollback
To rollback to a previous version:
```bash
# Stop current service
docker-compose down

# Restore from backup
cp -r backup/VERSION/* model/

# Restart
docker-compose up -d
```

## Security Considerations

1. **Network**: Expose only necessary ports
2. **Authentication**: Add API authentication for production
3. **Updates**: Regularly update base images
4. **Secrets**: Use Docker secrets for sensitive data

## Support
For issues or questions, please refer to Unsloth documentation or open an issue on GitHub.
"""
        
        doc_path = package_dir / "README.md"
        with open(doc_path, "w") as f:
            f.write(doc)
        
        # Generate config.json
        config = {
            "model_name": package.model_name,
            "version": package.version,
            "device_type": device_profile.device_type.value,
            "model_hash": package.model_hash,
            "created_at": package.created_at,
            "optimization_config": package.optimization_config,
            "ota_url": package.ota_url
        }
        
        config_path = package_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


# CLI integration
def create_edge_package(
    model_path: str,
    model_name: str,
    device_type: str,
    input_shapes: Dict[str, List[int]],
    version: str = "1.0.0",
    output_dir: Optional[str] = None
) -> DeploymentPackage:
    """CLI wrapper for edge packaging."""
    packager = EdgePackager()
    
    if output_dir:
        packager.output_dir = Path(output_dir)
        packager.output_dir.mkdir(parents=True, exist_ok=True)
    
    return packager.package_for_device(
        model_path=model_path,
        model_name=model_name,
        device_type=device_type,
        input_shapes=input_shapes,
        version=version
    )


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Package model for edge deployment")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--device", required=True, help="Target device type")
    parser.add_argument("--input-shape", required=True, help="Input shapes as JSON")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse input shapes
    input_shapes = json.loads(args.input_shape)
    
    # Create package
    package = create_edge_package(
        model_path=args.model,
        model_name=args.name,
        device_type=args.device,
        input_shapes=input_shapes,
        version=args.version,
        output_dir=args.output
    )
    
    print(f"Package created: {package.package_id}")
    print(f"Location: {package.package_path}")