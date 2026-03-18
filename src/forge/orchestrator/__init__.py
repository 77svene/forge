"""Unified Training Orchestrator with Real-time Dashboard

This module provides a centralized orchestrator for managing forge training jobs,
with real-time metrics visualization, distributed training support, and automatic fault recovery.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil
import torch
import torch.distributed as dist
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
METRICS_UPDATE_INTERVAL = 1.0  # seconds
MAX_JOB_HISTORY = 100
HEALTH_CHECK_INTERVAL = 30  # seconds


class JobStatus(str, Enum):
    """Training job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingPhase(str, Enum):
    """Training phase enumeration."""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    EVALUATION = "evaluation"
    CHECKPOINTING = "checkpointing"
    FINALIZING = "finalizing"


@dataclass
class GPUInfo:
    """GPU information for monitoring."""
    index: int
    name: str
    memory_total: float  # in MB
    memory_used: float
    memory_free: float
    utilization: float  # percentage
    temperature: float
    power_usage: float
    power_limit: float


@dataclass
class TrainingMetrics:
    """Training metrics for real-time monitoring."""
    timestamp: float
    epoch: int
    global_step: int
    phase: TrainingPhase
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    throughput: Optional[float] = None  # samples/sec
    gpu_info: List[GPUInfo] = field(default_factory=list)
    memory_allocated: Optional[float] = None  # in MB
    memory_reserved: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class JobConfig:
    """Configuration for a training job."""
    job_id: str
    name: str
    model_name_or_path: str
    dataset: str
    output_dir: str
    num_gpus: int = 1
    num_nodes: int = 1
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class TrainingJob:
    """Represents a training job."""
    config: JobConfig
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    current_metrics: Optional[TrainingMetrics] = None
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    error_message: Optional[str] = None
    process_id: Optional[int] = None
    worker_ids: List[int] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    best_metric: Optional[float] = None
    last_updated: float = field(default_factory=time.time)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.job_subscriptions: Dict[WebSocket, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id].add(websocket)
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        """Remove a WebSocket connection."""
        self.active_connections[client_id].discard(websocket)
        # Remove all subscriptions for this websocket
        for job_id in self.job_subscriptions.get(websocket, set()):
            self.unsubscribe(websocket, job_id)
        if websocket in self.job_subscriptions:
            del self.job_subscriptions[websocket]
        if not self.active_connections[client_id]:
            del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    def subscribe(self, websocket: WebSocket, job_id: str):
        """Subscribe a WebSocket to job updates."""
        self.job_subscriptions[websocket].add(job_id)
    
    def unsubscribe(self, websocket: WebSocket, job_id: str):
        """Unsubscribe a WebSocket from job updates."""
        if websocket in self.job_subscriptions:
            self.job_subscriptions[websocket].discard(job_id)
    
    async def broadcast_to_job(self, job_id: str, message: dict):
        """Broadcast a message to all subscribers of a job."""
        disconnected = set()
        for websocket, subscriptions in self.job_subscriptions.items():
            if job_id in subscriptions:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")
                    disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            for job_id in list(self.job_subscriptions.get(websocket, set())):
                self.unsubscribe(websocket, job_id)
    
    async def broadcast_global(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for client_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append((websocket, client_id))
        
        # Clean up disconnected websockets
        for websocket, client_id in disconnected:
            self.disconnect(websocket, client_id)


class MetricsCollector:
    """Collects and aggregates training metrics from distributed workers."""
    
    def __init__(self):
        self.gpu_stats_cache: Dict[int, GPUInfo] = {}
        self.last_update: float = 0
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Collect GPU information using nvidia-smi or torch."""
        gpu_info = []
        
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        # Get basic torch info
                        props = torch.cuda.get_device_properties(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                        memory_total = props.total_memory / 1024**2
                        
                        # Try to get more detailed info via pynvml
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                            
                            gpu_info.append(GPUInfo(
                                index=i,
                                name=props.name,
                                memory_total=memory_total,
                                memory_used=memory_allocated,
                                memory_free=memory_total - memory_allocated,
                                utilization=util.gpu,
                                temperature=temp,
                                power_usage=power,
                                power_limit=power_limit
                            ))
                        except (ImportError, Exception):
                            # Fallback to basic torch info
                            gpu_info.append(GPUInfo(
                                index=i,
                                name=props.name,
                                memory_total=memory_total,
                                memory_used=memory_allocated,
                                memory_free=memory_total - memory_allocated,
                                utilization=0.0,
                                temperature=0.0,
                                power_usage=0.0,
                                power_limit=0.0
                            ))
                    except Exception as e:
                        logger.warning(f"Failed to get GPU {i} info: {e}")
        except Exception as e:
            logger.error(f"Error collecting GPU info: {e}")
        
        return gpu_info
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system-level metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3,
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        }
        
        # Add disk usage for output directory
        try:
            disk = psutil.disk_usage("/")
            metrics["disk_percent"] = disk.percent
            metrics["disk_free_gb"] = disk.free / 1024**3
        except:
            pass
        
        return metrics


class TrainingWorker:
    """Manages a single training worker process."""
    
    def __init__(self, job: TrainingJob, metrics_queue: mp.Queue):
        self.job = job
        self.metrics_queue = metrics_queue
        self.process: Optional[mp.Process] = None
        self.should_stop = mp.Event()
    
    def start(self):
        """Start the training worker process."""
        self.process = mp.Process(
            target=self._run_training,
            args=(self.job.config, self.metrics_queue, self.should_stop),
            daemon=True
        )
        self.process.start()
        self.job.process_id = self.process.pid
        logger.info(f"Started training worker for job {self.job.config.job_id} (PID: {self.process.pid})")
    
    def stop(self):
        """Stop the training worker process."""
        if self.process and self.process.is_alive():
            self.should_stop.set()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
            logger.info(f"Stopped training worker for job {self.job.config.job_id}")
    
    def _run_training(self, config: JobConfig, metrics_queue: mp.Queue, should_stop: mp.Event):
        """Run the actual training loop (to be implemented by subclasses)."""
        try:
            # This is a placeholder - actual implementation would import and run
            # the appropriate training script from the existing codebase
            logger.info(f"Starting training for job {config.job_id}")
            
            # Simulate training with periodic metric updates
            total_steps = 1000
            for step in range(total_steps):
                if should_stop.is_set():
                    logger.info(f"Training stopped at step {step}")
                    break
                
                # Simulate training step
                time.sleep(0.1)
                
                # Collect metrics
                metrics = TrainingMetrics(
                    timestamp=time.time(),
                    epoch=step // 100,
                    global_step=step,
                    phase=TrainingPhase.TRAINING,
                    loss=1.0 / (step + 1) + 0.1 * (step % 10) / 10,
                    learning_rate=0.001 * (0.99 ** (step // 100)),
                    grad_norm=1.0 / (step + 1),
                    throughput=100.0 + step % 50,
                    gpu_info=[],  # Would be collected in real implementation
                    memory_allocated=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                    memory_reserved=torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
                )
                
                metrics_queue.put({
                    "type": "metrics",
                    "job_id": config.job_id,
                    "metrics": asdict(metrics)
                })
            
            # Training completed
            metrics_queue.put({
                "type": "status",
                "job_id": config.job_id,
                "status": JobStatus.COMPLETED,
                "progress": 1.0
            })
            
        except Exception as e:
            logger.error(f"Training failed for job {config.job_id}: {e}")
            traceback.print_exc()
            metrics_queue.put({
                "type": "status",
                "job_id": config.job_id,
                "status": JobStatus.FAILED,
                "error": str(e)
            })


class TrainingOrchestrator:
    """Central orchestrator for managing training jobs."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.jobs: Dict[str, TrainingJob] = {}
        self.workers: Dict[str, TrainingWorker] = {}
        self.connection_manager = ConnectionManager()
        self.metrics_collector = MetricsCollector()
        self.metrics_queue = mp.Queue()
        self.shutdown_event = asyncio.Event()
        self.background_tasks: List[asyncio.Task] = []
        
        # Create FastAPI app
        self.app = FastAPI(
            title="forge Training Orchestrator",
            description="Unified training orchestrator with real-time dashboard",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Start background tasks on startup."""
            self.background_tasks.append(
                asyncio.create_task(self._metrics_processor())
            )
            self.background_tasks.append(
                asyncio.create_task(self._health_checker())
            )
            logger.info("Orchestrator started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            self.shutdown_event.set()
            for task in self.background_tasks:
                task.cancel()
            # Stop all workers
            for job_id in list(self.workers.keys()):
                await self.stop_job(job_id)
            logger.info("Orchestrator stopped")
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "forge Training Orchestrator",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "active_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
                "total_jobs": len(self.jobs),
                "connected_clients": len(self.connection_manager.active_connections)
            }
        
        @self.app.get("/api/jobs")
        async def list_jobs():
            """List all training jobs."""
            jobs_list = []
            for job in self.jobs.values():
                job_dict = asdict(job)
                # Convert non-serializable fields
                job_dict["config"] = asdict(job.config)
                if job.current_metrics:
                    job_dict["current_metrics"] = asdict(job.current_metrics)
                jobs_list.append(job_dict)
            return {"jobs": jobs_list}
        
        @self.app.post("/api/jobs")
        async def create_job(job_config: Dict[str, Any]):
            """Create a new training job."""
            try:
                # Generate job ID
                job_id = str(uuid.uuid4())
                
                # Create job configuration
                config = JobConfig(
                    job_id=job_id,
                    name=job_config.get("name", f"job_{job_id[:8]}"),
                    model_name_or_path=job_config.get("model_name_or_path", ""),
                    dataset=job_config.get("dataset", ""),
                    output_dir=job_config.get("output_dir", f"./output/{job_id}"),
                    num_gpus=job_config.get("num_gpus", 1),
                    num_nodes=job_config.get("num_nodes", 1),
                    config_overrides=job_config.get("config_overrides", {})
                )
                
                # Create job
                job = TrainingJob(config=config)
                self.jobs[job_id] = job
                
                logger.info(f"Created job {job_id}: {config.name}")
                
                # Broadcast job creation
                await self.connection_manager.broadcast_global({
                    "type": "job_created",
                    "job_id": job_id,
                    "job": asdict(job)
                })
                
                return {"job_id": job_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create job: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/jobs/{job_id}/start")
        async def start_job(job_id: str, background_tasks: BackgroundTasks):
            """Start a training job."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            
            if job.status != JobStatus.PENDING:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job is not in pending state (current: {job.status})"
                )
            
            try:
                # Update job status
                job.status = JobStatus.RUNNING
                job.config.started_at = time.time()
                job.last_updated = time.time()
                
                # Create and start worker
                worker = TrainingWorker(job, self.metrics_queue)
                self.workers[job_id] = worker
                worker.start()
                
                logger.info(f"Started job {job_id}")
                
                # Broadcast status update
                await self.connection_manager.broadcast_global({
                    "type": "job_status_changed",
                    "job_id": job_id,
                    "status": job.status,
                    "timestamp": time.time()
                })
                
                return {"status": "started", "job_id": job_id}
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                logger.error(f"Failed to start job {job_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/jobs/{job_id}/stop")
        async def stop_job_endpoint(job_id: str):
            """Stop a training job."""
            return await self.stop_job(job_id)
        
        @self.app.get("/api/jobs/{job_id}")
        async def get_job(job_id: str):
            """Get job details."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            job_dict = asdict(job)
            job_dict["config"] = asdict(job.config)
            if job.current_metrics:
                job_dict["current_metrics"] = asdict(job.current_metrics)
            
            return job_dict
        
        @self.app.get("/api/jobs/{job_id}/metrics")
        async def get_job_metrics(job_id: str, limit: int = 100):
            """Get job metrics history."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            metrics_history = [asdict(m) for m in job.metrics_history[-limit:]]
            
            return {
                "job_id": job_id,
                "metrics": metrics_history,
                "current": asdict(job.current_metrics) if job.current_metrics else None
            }
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket, client_id)
            
            try:
                while True:
                    # Receive messages from client
                    data = await websocket.receive_json()
                    
                    if data.get("type") == "subscribe":
                        job_id = data.get("job_id")
                        if job_id:
                            self.connection_manager.subscribe(websocket, job_id)
                            # Send current job state
                            if job_id in self.jobs:
                                job = self.jobs[job_id]
                                await websocket.send_json({
                                    "type": "job_state",
                                    "job_id": job_id,
                                    "status": job.status,
                                    "progress": job.progress,
                                    "current_metrics": asdict(job.current_metrics) if job.current_metrics else None
                                })
                    
                    elif data.get("type") == "unsubscribe":
                        job_id = data.get("job_id")
                        if job_id:
                            self.connection_manager.unsubscribe(websocket, job_id)
                    
                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket, client_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket, client_id)
    
    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a training job."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = self.jobs[job_id]
        
        if job.status not in [JobStatus.RUNNING, JobStatus.PAUSED]:
            raise HTTPException(
                status_code=400,
                detail=f"Job is not running (current: {job.status})"
            )
        
        try:
            # Stop worker
            if job_id in self.workers:
                self.workers[job_id].stop()
                del self.workers[job_id]
            
            # Update job status
            job.status = JobStatus.STOPPED
            job.completed_at = time.time()
            job.last_updated = time.time()
            
            logger.info(f"Stopped job {job_id}")
            
            # Broadcast status update
            await self.connection_manager.broadcast_global({
                "type": "job_status_changed",
                "job_id": job_id,
                "status": job.status,
                "timestamp": time.time()
            })
            
            return {"status": "stopped", "job_id": job_id}
            
        except Exception as e:
            logger.error(f"Failed to stop job {job_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _metrics_processor(self):
        """Process metrics from worker processes."""
        while not self.shutdown_event.is_set():
            try:
                # Non-blocking queue get with timeout
                try:
                    message = self.metrics_queue.get(timeout=0.1)
                except:
                    await asyncio.sleep(0.1)
                    continue
                
                msg_type = message.get("type")
                job_id = message.get("job_id")
                
                if job_id not in self.jobs:
                    continue
                
                job = self.jobs[job_id]
                
                if msg_type == "metrics":
                    # Update job metrics
                    metrics_data = message.get("metrics", {})
                    metrics = TrainingMetrics(**metrics_data)
                    
                    # Add GPU info
                    metrics.gpu_info = self.metrics_collector.get_gpu_info()
                    
                    job.current_metrics = metrics
                    job.metrics_history.append(metrics)
                    
                    # Trim history
                    if len(job.metrics_history) > MAX_JOB_HISTORY:
                        job.metrics_history = job.metrics_history[-MAX_JOB_HISTORY:]
                    
                    # Update progress (simplified)
                    if metrics.global_step > 0:
                        # This would be calculated based on total steps in real implementation
                        job.progress = min(1.0, metrics.global_step / 1000)
                    
                    job.last_updated = time.time()
                    
                    # Broadcast metrics to subscribers
                    await self.connection_manager.broadcast_to_job(job_id, {
                        "type": "metrics_update",
                        "job_id": job_id,
                        "metrics": asdict(metrics),
                        "progress": job.progress,
                        "timestamp": time.time()
                    })
                    
                elif msg_type == "status":
                    # Update job status
                    new_status = message.get("status")
                    job.status = new_status
                    
                    if new_status == JobStatus.COMPLETED:
                        job.completed_at = time.time()
                        job.progress = 1.0
                    elif new_status == JobStatus.FAILED:
                        job.error_message = message.get("error")
                        job.completed_at = time.time()
                    
                    job.last_updated = time.time()
                    
                    # Broadcast status change
                    await self.connection_manager.broadcast_global({
                        "type": "job_status_changed",
                        "job_id": job_id,
                        "status": new_status,
                        "timestamp": time.time()
                    })
                    
                    # Clean up worker
                    if job_id in self.workers:
                        del self.workers[job_id]
                
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
                await asyncio.sleep(1)
    
    async def _health_checker(self):
        """Periodically check health of running jobs."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
                current_time = time.time()
                for job_id, job in list(self.jobs.items()):
                    if job.status == JobStatus.RUNNING:
                        # Check if job is stuck (no updates for too long)
                        if current_time - job.last_updated > 300:  # 5 minutes
                            logger.warning(f"Job {job_id} appears stuck, restarting...")
                            await self.stop_job(job_id)
                            job.status = JobStatus.PENDING
                            # Could implement automatic restart here
                
                # Clean up old completed jobs
                completed_jobs = [
                    jid for jid, job in self.jobs.items()
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED]
                    and current_time - job.last_updated > 86400  # 24 hours
                ]
                
                for job_id in completed_jobs:
                    del self.jobs[job_id]
                    logger.info(f"Cleaned up old job {job_id}")
                    
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def run(self):
        """Run the orchestrator."""
        import uvicorn
        
        logger.info(f"Starting forge Orchestrator on {self.host}:{self.port}")
        logger.info(f"Dashboard will be available at http://{self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )


# Integration with existing training scripts
class OrchestratorCallback:
    """Callback for integrating with existing training loops."""
    
    def __init__(self, job_id: str, metrics_queue: mp.Queue):
        self.job_id = job_id
        self.metrics_queue = metrics_queue
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_count = 0
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        self.metrics_queue.put({
            "type": "status",
            "job_id": self.job_id,
            "status": JobStatus.RUNNING,
            "phase": TrainingPhase.TRAINING
        })
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        self.metrics_queue.put({
            "type": "status",
            "job_id": self.job_id,
            "status": JobStatus.COMPLETED,
            "progress": 1.0
        })
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch."""
        self.step_count += 1
        
        # Only send metrics every N seconds to avoid flooding
        current_time = time.time()
        if current_time - self.last_log_time >= METRICS_UPDATE_INTERVAL:
            self.last_log_time = current_time
            
            metrics = TrainingMetrics(
                timestamp=current_time,
                epoch=logs.get("epoch", 0),
                global_step=self.step_count,
                phase=TrainingPhase.TRAINING,
                loss=logs.get("loss", 0.0),
                learning_rate=logs.get("learning_rate", 0.0),
                grad_norm=logs.get("grad_norm"),
                throughput=logs.get("throughput"),
                memory_allocated=logs.get("memory_allocated"),
                memory_reserved=logs.get("memory_reserved"),
                custom_metrics={
                    k: v for k, v in logs.items()
                    if k not in ["loss", "learning_rate", "grad_norm", "throughput", "epoch"]
                }
            )
            
            self.metrics_queue.put({
                "type": "metrics",
                "job_id": self.job_id,
                "metrics": asdict(metrics)
            })
    
    def on_log(self, logs: Dict[str, float]):
        """Called when logging metrics."""
        self.on_batch_end(self.step_count, logs)


# Factory function for easy integration
def create_orchestrator_callback(job_id: str, metrics_queue: mp.Queue) -> OrchestratorCallback:
    """Create an orchestrator callback for integration with training loops."""
    return OrchestratorCallback(job_id, metrics_queue)


# CLI entry point
def main():
    """Main entry point for the orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Training Orchestrator")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run orchestrator
    orchestrator = TrainingOrchestrator(host=args.host, port=args.port)
    orchestrator.run()


if __name__ == "__main__":
    main()