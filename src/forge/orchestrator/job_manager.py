"""
forge Unified Training Orchestrator with Real-time Dashboard
Manages training jobs, provides real-time metrics visualization, and enables distributed training
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"


class JobType(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    CONVERSION = "conversion"
    MERGING = "merging"


class GPUInfo(BaseModel):
    index: int
    name: str
    memory_total: float
    memory_used: float
    memory_free: float
    utilization: float
    temperature: Optional[float] = None


class MetricPoint(BaseModel):
    timestamp: float
    step: int
    value: float
    epoch: Optional[float] = None


class JobMetrics(BaseModel):
    loss: List[MetricPoint] = []
    learning_rate: List[MetricPoint] = []
    gpu_utilization: List[Dict[str, Any]] = []
    memory_usage: List[Dict[str, Any]] = []
    throughput: List[MetricPoint] = []
    custom_metrics: Dict[str, List[MetricPoint]] = {}


class JobConfig(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType = JobType.TRAINING
    name: str
    description: str = ""
    script_path: str
    script_args: List[str] = []
    working_dir: str = "."
    env_vars: Dict[str, str] = {}
    num_gpus: int = 1
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    max_retries: int = 3
    timeout: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    log_dir: str = "./logs"
    auto_resume: bool = True
    priority: int = 0
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)


class JobInfo(BaseModel):
    config: JobConfig
    status: JobStatus = JobStatus.PENDING
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    gpu_ids: List[int] = []
    node_info: Dict[str, Any] = {}


class TrainingCallback:
    """Callback interface for training loops to report metrics"""
    
    def __init__(self, job_id: str, orchestrator_url: str = "http://localhost:8000"):
        self.job_id = job_id
        self.orchestrator_url = orchestrator_url
        self.step = 0
        self.epoch = 0
        
    def report_loss(self, loss: float, step: Optional[int] = None):
        """Report training loss"""
        if step is not None:
            self.step = step
        self._send_metric("loss", loss)
        
    def report_learning_rate(self, lr: float, step: Optional[int] = None):
        """Report current learning rate"""
        if step is not None:
            self.step = step
        self._send_metric("learning_rate", lr)
        
    def report_custom_metric(self, name: str, value: float, step: Optional[int] = None):
        """Report custom metric"""
        if step is not None:
            self.step = step
        self._send_metric(f"custom_{name}", value)
        
    def report_gpu_stats(self, gpu_index: int, utilization: float, memory_used: float):
        """Report GPU statistics"""
        import requests
        try:
            requests.post(
                f"{self.orchestrator_url}/api/jobs/{self.job_id}/gpu_stats",
                json={
                    "gpu_index": gpu_index,
                    "utilization": utilization,
                    "memory_used": memory_used,
                    "timestamp": time.time()
                },
                timeout=1
            )
        except:
            pass  # Silently fail for metrics
            
    def _send_metric(self, metric_name: str, value: float):
        """Send metric to orchestrator"""
        import requests
        try:
            requests.post(
                f"{self.orchestrator_url}/api/jobs/{self.job_id}/metrics",
                json={
                    "metric_name": metric_name,
                    "value": value,
                    "step": self.step,
                    "epoch": self.epoch,
                    "timestamp": time.time()
                },
                timeout=1
            )
        except:
            pass  # Silently fail for metrics


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
                
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass
                    
    async def broadcast_all(self, message: dict):
        for job_id in list(self.active_connections.keys()):
            await self.broadcast(job_id, message)


class GPUManager:
    """Manages GPU allocation and monitoring"""
    
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.allocated_gpus: Dict[str, List[int]] = {}  # job_id -> gpu_ids
        
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except:
            pass
        return []
        
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get current GPU information"""
        gpu_info = []
        try:
            import torch
            for i in self.available_gpus:
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                gpu_info.append(GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total=memory_total / (1024 ** 3),  # Convert to GB
                    memory_used=memory_allocated / (1024 ** 3),
                    memory_free=(memory_total - memory_allocated) / (1024 ** 3),
                    utilization=self._get_gpu_utilization(i),
                    temperature=self._get_gpu_temperature(i)
                ))
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
        return gpu_info
        
    def _get_gpu_utilization(self, gpu_index: int) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
            
    def _get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """Get GPU temperature"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            return None
            
    def allocate_gpus(self, job_id: str, num_gpus: int) -> List[int]:
        """Allocate GPUs for a job"""
        if num_gpus > len(self.available_gpus):
            raise ValueError(f"Requested {num_gpus} GPUs but only {len(self.available_gpus)} available")
            
        # Find available GPUs
        allocated = []
        for gpu_id in self.available_gpus:
            if not any(gpu_id in gpus for gpus in self.allocated_gpus.values()):
                allocated.append(gpu_id)
                if len(allocated) == num_gpus:
                    break
                    
        if len(allocated) < num_gpus:
            raise ValueError(f"Could not allocate {num_gpus} GPUs")
            
        self.allocated_gpus[job_id] = allocated
        return allocated
        
    def release_gpus(self, job_id: str):
        """Release GPUs allocated to a job"""
        if job_id in self.allocated_gpus:
            del self.allocated_gpus[job_id]


class JobManager:
    """Manages training jobs lifecycle"""
    
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.gpu_manager = GPUManager()
        self.connection_manager = ConnectionManager()
        self._monitor_task = None
        self._shutdown = False
        
    async def start(self):
        """Start the job manager"""
        self._monitor_task = asyncio.create_task(self._monitor_jobs())
        logger.info("Job manager started")
        
    async def stop(self):
        """Stop the job manager"""
        self._shutdown = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
        # Stop all running jobs
        for job_id in list(self.jobs.keys()):
            if self.jobs[job_id].status == JobStatus.RUNNING:
                await self.stop_job(job_id)
                
        logger.info("Job manager stopped")
        
    async def create_job(self, config: JobConfig) -> JobInfo:
        """Create a new job"""
        if config.job_id in self.jobs:
            raise ValueError(f"Job {config.job_id} already exists")
            
        # Create log directory
        log_dir = Path(config.log_dir) / config.job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        job_info = JobInfo(config=config)
        self.jobs[config.job_id] = job_info
        
        logger.info(f"Created job {config.job_id}: {config.name}")
        await self.connection_manager.broadcast_all({
            "type": "job_created",
            "job_id": config.job_id,
            "job": job_info.dict()
        })
        
        return job_info
        
    async def start_job(self, job_id: str) -> JobInfo:
        """Start a job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job_info = self.jobs[job_id]
        if job_info.status == JobStatus.RUNNING:
            raise ValueError(f"Job {job_id} is already running")
            
        # Allocate GPUs
        try:
            gpu_ids = self.gpu_manager.allocate_gpus(job_id, job_info.config.num_gpus)
            job_info.gpu_ids = gpu_ids
        except ValueError as e:
            job_info.status = JobStatus.FAILED
            job_info.error_message = str(e)
            raise
            
        # Prepare environment
        env = os.environ.copy()
        env.update(job_info.config.env_vars)
        
        # Set CUDA_VISIBLE_DEVICES
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            
        # For distributed training
        if job_info.config.num_gpus > 1 or job_info.config.num_nodes > 1:
            env["MASTER_ADDR"] = job_info.config.master_addr
            env["MASTER_PORT"] = str(job_info.config.master_port)
            env["WORLD_SIZE"] = str(job_info.config.num_gpus * job_info.config.num_nodes)
            env["RANK"] = str(job_info.config.node_rank * job_info.config.num_gpus)
            env["LOCAL_RANK"] = "0"  # Will be overridden per process
            
        # Build command
        cmd = self._build_command(job_info.config)
        
        # Start process
        try:
            process = subprocess.Popen(
                cmd,
                cwd=job_info.config.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[job_id] = process
            job_info.status = JobStatus.RUNNING
            job_info.pid = process.pid
            job_info.start_time = datetime.now()
            
            # Start output monitoring
            asyncio.create_task(self._monitor_output(job_id, process))
            
            logger.info(f"Started job {job_id} with PID {process.pid}")
            await self.connection_manager.broadcast_all({
                "type": "job_started",
                "job_id": job_id,
                "pid": process.pid,
                "gpu_ids": gpu_ids
            })
            
        except Exception as e:
            job_info.status = JobStatus.FAILED
            job_info.error_message = f"Failed to start process: {e}"
            self.gpu_manager.release_gpus(job_id)
            raise
            
        return job_info
        
    def _build_command(self, config: JobConfig) -> List[str]:
        """Build command line for job execution"""
        cmd = []
        
        # Check if it's a distributed training job
        if config.num_gpus > 1 or config.num_nodes > 1:
            # Use torchrun for distributed training
            cmd = [
                "torchrun",
                f"--nproc_per_node={config.num_gpus}",
                f"--nnodes={config.num_nodes}",
                f"--node_rank={config.node_rank}",
                f"--master_addr={config.master_addr}",
                f"--master_port={config.master_port}",
                config.script_path
            ]
        else:
            # Single GPU/CPU training
            cmd = [sys.executable, config.script_path]
            
        # Add script arguments
        cmd.extend(config.script_args)
        
        return cmd
        
    async def stop_job(self, job_id: str, graceful: bool = True) -> JobInfo:
        """Stop a running job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job_info = self.jobs[job_id]
        if job_info.status != JobStatus.RUNNING:
            raise ValueError(f"Job {job_id} is not running")
            
        if job_id in self.processes:
            process = self.processes[job_id]
            
            if graceful:
                # Send SIGTERM first
                process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(process)),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    # Force kill if not terminated
                    process.kill()
                    await asyncio.create_task(self._wait_for_process(process))
            else:
                process.kill()
                await asyncio.create_task(self._wait_for_process(process))
                
            del self.processes[job_id]
            
        job_info.status = JobStatus.STOPPED
        job_info.end_time = datetime.now()
        self.gpu_manager.release_gpus(job_id)
        
        logger.info(f"Stopped job {job_id}")
        await self.connection_manager.broadcast_all({
            "type": "job_stopped",
            "job_id": job_id
        })
        
        return job_info
        
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for process to complete"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
            
    async def _monitor_output(self, job_id: str, process: subprocess.Popen):
        """Monitor job output and stream to connected clients"""
        job_info = self.jobs[job_id]
        log_file = Path(job_info.config.log_dir) / job_id / "output.log"
        
        with open(log_file, "w") as f:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    # Write to log file
                    f.write(line)
                    f.flush()
                    
                    # Broadcast to connected clients
                    await self.connection_manager.broadcast(job_id, {
                        "type": "output",
                        "job_id": job_id,
                        "line": line.strip(),
                        "timestamp": time.time()
                    })
                    
                    # Parse metrics from output
                    await self._parse_metrics(job_id, line)
                    
        # Process completed
        exit_code = process.returncode
        job_info.exit_code = exit_code
        job_info.end_time = datetime.now()
        
        if exit_code == 0:
            job_info.status = JobStatus.COMPLETED
        else:
            job_info.status = JobStatus.FAILED
            job_info.error_message = f"Process exited with code {exit_code}"
            
            # Auto-retry if configured
            if job_info.config.auto_resume and job_info.retry_count < job_info.config.max_retries:
                job_info.retry_count += 1
                logger.info(f"Auto-retrying job {job_id} (attempt {job_info.retry_count})")
                await asyncio.sleep(5)  # Wait before retry
                await self.start_job(job_id)
                return
                
        self.gpu_manager.release_gpus(job_id)
        
        await self.connection_manager.broadcast_all({
            "type": "job_completed",
            "job_id": job_id,
            "status": job_info.status,
            "exit_code": exit_code
        })
        
    async def _parse_metrics(self, job_id: str, line: str):
        """Parse metrics from output line"""
        job_info = self.jobs[job_id]
        
        # Common metric patterns
        import re
        
        # Loss pattern: "loss: 0.123" or "train/loss: 0.123"
        loss_match = re.search(r'(?:train/)?loss:\s*([\d.]+)', line, re.IGNORECASE)
        if loss_match:
            loss = float(loss_match.group(1))
            job_info.metrics.loss.append(MetricPoint(
                timestamp=time.time(),
                step=len(job_info.metrics.loss),
                value=loss
            ))
            
        # Learning rate pattern: "lr: 0.0001" or "learning_rate: 0.0001"
        lr_match = re.search(r'(?:learning_rate|lr):\s*([\d.e-]+)', line, re.IGNORECASE)
        if lr_match:
            lr = float(lr_match.group(1))
            job_info.metrics.learning_rate.append(MetricPoint(
                timestamp=time.time(),
                step=len(job_info.metrics.learning_rate),
                value=lr
            ))
            
        # Step/epoch pattern: "step: 100" or "epoch: 1.5"
        step_match = re.search(r'step:\s*(\d+)', line, re.IGNORECASE)
        epoch_match = re.search(r'epoch:\s*([\d.]+)', line, re.IGNORECASE)
        
        # Broadcast metrics update
        if loss_match or lr_match:
            await self.connection_manager.broadcast(job_id, {
                "type": "metrics_update",
                "job_id": job_id,
                "loss": job_info.metrics.loss[-1].value if job_info.metrics.loss else None,
                "learning_rate": job_info.metrics.learning_rate[-1].value if job_info.metrics.learning_rate else None,
                "timestamp": time.time()
            })
            
    async def _monitor_jobs(self):
        """Background task to monitor jobs and update GPU stats"""
        while not self._shutdown:
            try:
                # Update GPU stats for all running jobs
                gpu_info = self.gpu_manager.get_gpu_info()
                
                for job_id, job_info in self.jobs.items():
                    if job_info.status == JobStatus.RUNNING:
                        # Update GPU utilization
                        job_gpu_info = [g for g in gpu_info if g.index in job_info.gpu_ids]
                        if job_gpu_info:
                            job_info.metrics.gpu_utilization.append({
                                "timestamp": time.time(),
                                "gpus": [g.dict() for g in job_gpu_info]
                            })
                            
                        # Broadcast GPU update
                        await self.connection_manager.broadcast(job_id, {
                            "type": "gpu_update",
                            "job_id": job_id,
                            "gpus": [g.dict() for g in job_gpu_info],
                            "timestamp": time.time()
                        })
                        
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in job monitor: {e}")
                await asyncio.sleep(10)
                
    async def get_job(self, job_id: str) -> JobInfo:
        """Get job information"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]
        
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobInfo]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs
        
    async def delete_job(self, job_id: str):
        """Delete a job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job_info = self.jobs[job_id]
        if job_info.status == JobStatus.RUNNING:
            await self.stop_job(job_id)
            
        # Clean up log directory
        log_dir = Path(job_info.config.log_dir) / job_id
        if log_dir.exists():
            import shutil
            shutil.rmtree(log_dir)
            
        del self.jobs[job_id]
        logger.info(f"Deleted job {job_id}")
        
    async def update_job_metrics(self, job_id: str, metric_name: str, value: float, 
                                step: int, epoch: Optional[float] = None):
        """Update job metrics (called by training callback)"""
        if job_id not in self.jobs:
            return
            
        job_info = self.jobs[job_id]
        metric_point = MetricPoint(
            timestamp=time.time(),
            step=step,
            value=value,
            epoch=epoch
        )
        
        if metric_name == "loss":
            job_info.metrics.loss.append(metric_point)
        elif metric_name == "learning_rate":
            job_info.metrics.learning_rate.append(metric_point)
        elif metric_name.startswith("custom_"):
            custom_name = metric_name[7:]  # Remove "custom_" prefix
            if custom_name not in job_info.metrics.custom_metrics:
                job_info.metrics.custom_metrics[custom_name] = []
            job_info.metrics.custom_metrics[custom_name].append(metric_point)
            
        # Broadcast metrics update
        await self.connection_manager.broadcast(job_id, {
            "type": "metrics_update",
            "job_id": job_id,
            "metric_name": metric_name,
            "value": value,
            "step": step,
            "epoch": epoch,
            "timestamp": time.time()
        })


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    job_manager = JobManager()
    await job_manager.start()
    app.state.job_manager = job_manager
    
    yield
    
    # Shutdown
    await job_manager.stop()


app = FastAPI(
    title="forge Orchestrator",
    description="Unified Training Orchestrator with Real-time Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.post("/api/jobs", response_model=JobInfo)
async def create_job(config: JobConfig, background_tasks: BackgroundTasks):
    """Create a new training job"""
    try:
        job_manager = app.state.job_manager
        job_info = await job_manager.create_job(config)
        return job_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/jobs/{job_id}/start", response_model=JobInfo)
async def start_job(job_id: str):
    """Start a job"""
    try:
        job_manager = app.state.job_manager
        job_info = await job_manager.start_job(job_id)
        return job_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/jobs/{job_id}/stop", response_model=JobInfo)
async def stop_job(job_id: str, graceful: bool = True):
    """Stop a running job"""
    try:
        job_manager = app.state.job_manager
        job_info = await job_manager.stop_job(job_id, graceful)
        return job_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/jobs", response_model=List[JobInfo])
async def list_jobs(status: Optional[JobStatus] = None):
    """List all jobs"""
    job_manager = app.state.job_manager
    return await job_manager.list_jobs(status)


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """Get job details"""
    try:
        job_manager = app.state.job_manager
        return await job_manager.get_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    try:
        job_manager = app.state.job_manager
        await job_manager.delete_job(job_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gpus", response_model=List[GPUInfo])
async def get_gpu_info():
    """Get GPU information"""
    job_manager = app.state.job_manager
    return job_manager.gpu_manager.get_gpu_info()


@app.post("/api/jobs/{job_id}/metrics")
async def update_job_metrics(job_id: str, metric_data: dict):
    """Update job metrics (called by training callback)"""
    try:
        job_manager = app.state.job_manager
        await job_manager.update_job_metrics(
            job_id=job_id,
            metric_name=metric_data["metric_name"],
            value=metric_data["value"],
            step=metric_data["step"],
            epoch=metric_data.get("epoch")
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/jobs/{job_id}/gpu_stats")
async def update_gpu_stats(job_id: str, gpu_data: dict):
    """Update GPU statistics"""
    # This endpoint is for reporting GPU stats from training nodes
    # Implementation depends on distributed setup
    return {"status": "success"}


# WebSocket endpoints
@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    job_manager = app.state.job_manager
    
    try:
        await job_manager.connection_manager.connect(websocket, job_id)
        
        # Send initial job state
        job_info = await job_manager.get_job(job_id)
        await websocket.send_json({
            "type": "initial_state",
            "job": job_info.dict()
        })
        
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Could handle client messages here if needed
            
    except WebSocketDisconnect:
        job_manager.connection_manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        job_manager.connection_manager.disconnect(websocket, job_id)


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """WebSocket endpoint for dashboard updates"""
    job_manager = app.state.job_manager
    
    try:
        await websocket.accept()
        
        # Send initial state
        jobs = await job_manager.list_jobs()
        gpu_info = job_manager.gpu_manager.get_gpu_info()
        
        await websocket.send_json({
            "type": "initial_state",
            "jobs": [job.dict() for job in jobs],
            "gpus": [gpu.dict() for gpu in gpu_info]
        })
        
        # Register for all updates
        dashboard_id = str(uuid.uuid4())
        job_manager.connection_manager.active_connections[dashboard_id] = [websocket]
        
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        if dashboard_id in job_manager.connection_manager.active_connections:
            del job_manager.connection_manager.active_connections[dashboard_id]
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        if dashboard_id in job_manager.connection_manager.active_connections:
            del job_manager.connection_manager.active_connections[dashboard_id]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    job_manager = app.state.job_manager
    
    metrics = []
    
    # Job metrics
    for job_id, job_info in job_manager.jobs.items():
        metrics.append(f'forge_job_status{{job_id="{job_id}",status="{job_info.status}"}} 1')
        
        if job_info.metrics.loss:
            latest_loss = job_info.metrics.loss[-1].value
            metrics.append(f'forge_job_loss{{job_id="{job_id}"}} {latest_loss}')
            
    # GPU metrics
    gpu_info = job_manager.gpu_manager.get_gpu_info()
    for gpu in gpu_info:
        metrics.append(f'forge_gpu_utilization{{gpu="{gpu.index}"}} {gpu.utilization}')
        metrics.append(f'forge_gpu_memory_used{{gpu="{gpu.index}"}} {gpu.memory_used}')
        metrics.append(f'forge_gpu_memory_total{{gpu="{gpu.index}"}} {gpu.memory_total}')
        
    return "\n".join(metrics)


# Example usage and CLI
def main():
    """Main entry point for the orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Orchestrator")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "forge.orchestrator.job_manager:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()