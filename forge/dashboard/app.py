import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

import psutil
import GPUtil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AnomalyType(str, Enum):
    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    GRADIENT_EXPLOSION = "gradient_explosion"
    MEMORY_OVERFLOW = "memory_overflow"
    GPU_UTILIZATION_LOW = "gpu_utilization_low"
    LEARNING_RATE_ISSUE = "learning_rate_issue"


@dataclass
class TrainingMetrics:
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    cpu_percent: float = 0.0
    throughput: float = 0.0  # samples per second
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingRun:
    run_id: str
    name: str
    model_name: str
    status: RunStatus = RunStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    early_stop_recommendation: Optional[Dict[str, Any]] = None
    last_update: float = field(default_factory=time.time)


class MetricCollector:
    """Collects and aggregates training metrics with configurable sampling."""
    
    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate  # seconds
        self._metrics_buffer: Dict[str, deque] = {}
        self._lock = threading.Lock()
        
    def add_metrics(self, run_id: str, metrics: TrainingMetrics):
        with self._lock:
            if run_id not in self._metrics_buffer:
                self._metrics_buffer[run_id] = deque(maxlen=1000)
            self._metrics_buffer[run_id].append(metrics)
    
    def get_recent_metrics(self, run_id: str, limit: int = 100) -> List[TrainingMetrics]:
        with self._lock:
            if run_id not in self._metrics_buffer:
                return []
            buffer = self._metrics_buffer[run_id]
            return list(buffer)[-limit:]
    
    def get_aggregated_metrics(self, run_id: str, window: int = 10) -> Dict[str, Any]:
        """Get aggregated metrics for visualization."""
        recent = self.get_recent_metrics(run_id, window)
        if not recent:
            return {}
        
        losses = [m.loss for m in recent]
        eval_losses = [m.eval_loss for m in recent if m.eval_loss is not None]
        
        return {
            "avg_loss": np.mean(losses) if losses else 0.0,
            "loss_std": np.std(losses) if len(losses) > 1 else 0.0,
            "avg_eval_loss": np.mean(eval_losses) if eval_losses else None,
            "avg_gpu_memory": np.mean([m.gpu_memory_used[0] if m.gpu_memory_used else 0 for m in recent]),
            "avg_gpu_util": np.mean([m.gpu_utilization[0] if m.gpu_utilization else 0 for m in recent]),
            "avg_throughput": np.mean([m.throughput for m in recent]),
            "trend": self._calculate_trend(losses)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return "stable"
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.01:
            return "improving"
        elif slope > 0.01:
            return "worsening"
        else:
            return "stable"


class AnomalyDetector:
    """Detects training anomalies and recommends early stopping."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anomaly_thresholds = {
            "loss_spike_threshold": config.get("loss_spike_threshold", 2.0),
            "loss_plateau_patience": config.get("loss_plateau_patience", 100),
            "grad_norm_threshold": config.get("grad_norm_threshold", 10.0),
            "gpu_memory_threshold": config.get("gpu_memory_threshold", 0.95),
            "gpu_util_threshold": config.get("gpu_util_threshold", 0.3),
        }
    
    def detect_anomalies(self, run: TrainingRun) -> List[Dict[str, Any]]:
        anomalies = []
        recent_metrics = run.metrics_history[-100:] if run.metrics_history else []
        
        if not recent_metrics:
            return anomalies
        
        # Loss spike detection
        if len(recent_metrics) >= 2:
            current_loss = recent_metrics[-1].loss
            avg_loss = np.mean([m.loss for m in recent_metrics[:-1]])
            if avg_loss > 0 and current_loss / avg_loss > self.anomaly_thresholds["loss_spike_threshold"]:
                anomalies.append({
                    "type": AnomalyType.LOSS_SPIKE,
                    "severity": "high",
                    "message": f"Loss spike detected: {current_loss:.4f} vs avg {avg_loss:.4f}",
                    "timestamp": time.time()
                })
        
        # Loss plateau detection
        if len(recent_metrics) >= self.anomaly_thresholds["loss_plateau_patience"]:
            window = recent_metrics[-self.anomaly_thresholds["loss_plateau_patience"]:]
            loss_std = np.std([m.loss for m in window])
            if loss_std < 0.001:  # Very small variation
                anomalies.append({
                    "type": AnomalyType.LOSS_PLATEAU,
                    "severity": "medium",
                    "message": f"Loss plateau detected over {len(window)} steps",
                    "timestamp": time.time()
                })
        
        # Gradient explosion detection
        if recent_metrics[-1].grad_norm > self.anomaly_thresholds["grad_norm_threshold"]:
            anomalies.append({
                "type": AnomalyType.GRADIENT_EXPLOSION,
                "severity": "high",
                "message": f"Gradient norm too high: {recent_metrics[-1].grad_norm:.2f}",
                "timestamp": time.time()
            })
        
        # GPU memory overflow detection
        if recent_metrics[-1].gpu_memory_used:
            max_gpu_mem = max(recent_metrics[-1].gpu_memory_used)
            if max_gpu_mem > self.anomaly_thresholds["gpu_memory_threshold"]:
                anomalies.append({
                    "type": AnomalyType.MEMORY_OVERFLOW,
                    "severity": "critical",
                    "message": f"GPU memory usage too high: {max_gpu_mem*100:.1f}%",
                    "timestamp": time.time()
                })
        
        # Low GPU utilization detection
        if recent_metrics[-1].gpu_utilization:
            avg_gpu_util = np.mean(recent_metrics[-1].gpu_utilization)
            if avg_gpu_util < self.anomaly_thresholds["gpu_util_threshold"]:
                anomalies.append({
                    "type": AnomalyType.GPU_UTILIZATION_LOW,
                    "severity": "low",
                    "message": f"Low GPU utilization: {avg_gpu_util*100:.1f}%",
                    "timestamp": time.time()
                })
        
        return anomalies
    
    def recommend_early_stopping(self, run: TrainingRun) -> Optional[Dict[str, Any]]:
        """Generate early stopping recommendation based on training progress."""
        if len(run.metrics_history) < 50:
            return None
        
        recent_metrics = run.metrics_history[-100:]
        losses = [m.loss for m in recent_metrics]
        
        # Calculate improvement rate
        if len(losses) >= 20:
            early_avg = np.mean(losses[:10])
            late_avg = np.mean(losses[-10:])
            improvement = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
            
            # Check for stagnation
            loss_std = np.std(losses[-20:])
            
            if improvement < 0.01 and loss_std < 0.005:
                return {
                    "recommendation": "stop",
                    "confidence": 0.8,
                    "reason": "Training has stagnated with minimal improvement",
                    "suggested_action": "Consider stopping or adjusting learning rate",
                    "metrics": {
                        "improvement_rate": improvement,
                        "loss_variation": loss_std
                    }
                }
        
        # Check for divergence
        if len(losses) >= 10:
            recent_trend = np.polyfit(range(10), losses[-10:], 1)[0]
            if recent_trend > 0.01:  # Loss increasing
                return {
                    "recommendation": "stop",
                    "confidence": 0.9,
                    "reason": "Training appears to be diverging",
                    "suggested_action": "Stop training immediately and check configuration",
                    "metrics": {
                        "loss_trend": recent_trend
                    }
                }
        
        return None


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, run_id: str = "global"):
        await websocket.accept()
        async with self._lock:
            if run_id not in self.active_connections:
                self.active_connections[run_id] = set()
            self.active_connections[run_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, run_id: str = "global"):
        async with self._lock:
            if run_id in self.active_connections:
                self.active_connections[run_id].discard(websocket)
                if not self.active_connections[run_id]:
                    del self.active_connections[run_id]
    
    async def broadcast(self, message: Dict[str, Any], run_id: str = "global"):
        async with self._lock:
            if run_id in self.active_connections:
                dead_connections = set()
                for connection in self.active_connections[run_id]:
                    try:
                        await connection.send_json(message)
                    except Exception:
                        dead_connections.add(connection)
                
                # Clean up dead connections
                for dead in dead_connections:
                    self.active_connections[run_id].discard(dead)


class DashboardApp:
    """Main dashboard application with FastAPI backend."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.app = FastAPI(
            title="forge Training Dashboard",
            description="Real-time monitoring for LLM training runs",
            version="1.0.0"
        )
        
        self.config = self._load_config(config_path)
        self.metric_collector = MetricCollector(
            sampling_rate=self.config.get("sampling_rate", 1.0)
        )
        self.anomaly_detector = AnomalyDetector(self.config)
        self.connection_manager = ConnectionManager()
        self.training_runs: Dict[str, TrainingRun] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_background_tasks()
        
        logger.info("Dashboard application initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "sampling_rate": 1.0,
            "max_runs": 50,
            "history_limit": 1000,
            "update_interval": 2.0,
            "loss_spike_threshold": 2.0,
            "loss_plateau_patience": 100,
            "grad_norm_threshold": 10.0,
            "gpu_memory_threshold": 0.95,
            "gpu_util_threshold": 0.3,
            "cors_origins": ["*"],
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the main dashboard HTML."""
            return self._generate_dashboard_html()
        
        @self.app.websocket("/ws/{run_id}")
        async def websocket_endpoint(websocket: WebSocket, run_id: str):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket, run_id)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    # Handle any client messages if needed
            except WebSocketDisconnect:
                await self.connection_manager.disconnect(websocket, run_id)
        
        @self.app.get("/api/runs")
        async def list_runs():
            """List all training runs."""
            return {
                "runs": [
                    {
                        "run_id": run.run_id,
                        "name": run.name,
                        "model_name": run.model_name,
                        "status": run.status.value,
                        "start_time": run.start_time,
                        "end_time": run.end_time,
                        "last_update": run.last_update
                    }
                    for run in self.training_runs.values()
                ]
            }
        
        @self.app.get("/api/runs/{run_id}")
        async def get_run(run_id: str):
            """Get details of a specific training run."""
            if run_id not in self.training_runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.training_runs[run_id]
            recent_metrics = self.metric_collector.get_recent_metrics(run_id, 50)
            aggregated = self.metric_collector.get_aggregated_metrics(run_id)
            
            return {
                "run": {
                    "run_id": run.run_id,
                    "name": run.name,
                    "model_name": run.model_name,
                    "status": run.status.value,
                    "config": run.config,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "metrics_count": len(run.metrics_history),
                    "anomalies_count": len(run.anomalies),
                    "early_stop_recommendation": run.early_stop_recommendation
                },
                "recent_metrics": [
                    {
                        "epoch": m.epoch,
                        "step": m.step,
                        "loss": m.loss,
                        "eval_loss": m.eval_loss,
                        "learning_rate": m.learning_rate,
                        "grad_norm": m.grad_norm,
                        "gpu_memory_used": m.gpu_memory_used,
                        "gpu_utilization": m.gpu_utilization,
                        "cpu_percent": m.cpu_percent,
                        "throughput": m.throughput,
                        "timestamp": m.timestamp
                    }
                    for m in recent_metrics
                ],
                "aggregated": aggregated
            }
        
        @self.app.post("/api/runs")
        async def create_run(run_data: Dict[str, Any]):
            """Create a new training run."""
            run_id = str(uuid.uuid4())
            run = TrainingRun(
                run_id=run_id,
                name=run_data.get("name", f"Run {run_id[:8]}"),
                model_name=run_data.get("model_name", "unknown"),
                config=run_data.get("config", {}),
                start_time=time.time()
            )
            self.training_runs[run_id] = run
            
            # Broadcast new run creation
            await self.connection_manager.broadcast({
                "type": "run_created",
                "run_id": run_id,
                "name": run.name,
                "timestamp": time.time()
            })
            
            return {"run_id": run_id, "status": "created"}
        
        @self.app.post("/api/runs/{run_id}/metrics")
        async def update_metrics(run_id: str, metrics_data: Dict[str, Any]):
            """Update metrics for a training run."""
            if run_id not in self.training_runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.training_runs[run_id]
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=metrics_data.get("epoch", 0),
                step=metrics_data.get("step", 0),
                loss=metrics_data.get("loss", 0.0),
                eval_loss=metrics_data.get("eval_loss"),
                learning_rate=metrics_data.get("learning_rate", 0.0),
                grad_norm=metrics_data.get("grad_norm", 0.0),
                gpu_memory_used=metrics_data.get("gpu_memory_used", []),
                gpu_utilization=metrics_data.get("gpu_utilization", []),
                cpu_percent=metrics_data.get("cpu_percent", 0.0),
                throughput=metrics_data.get("throughput", 0.0),
                timestamp=time.time()
            )
            
            # Add to run history
            run.metrics_history.append(metrics)
            if len(run.metrics_history) > self.config["history_limit"]:
                run.metrics_history = run.metrics_history[-self.config["history_limit"]:]
            
            # Add to metric collector
            self.metric_collector.add_metrics(run_id, metrics)
            
            # Update run status
            run.status = RunStatus.RUNNING
            run.last_update = time.time()
            
            # Detect anomalies
            new_anomalies = self.anomaly_detector.detect_anomalies(run)
            for anomaly in new_anomalies:
                run.anomalies.append(anomaly)
                # Broadcast anomaly
                await self.connection_manager.broadcast({
                    "type": "anomaly_detected",
                    "run_id": run_id,
                    "anomaly": anomaly,
                    "timestamp": time.time()
                })
            
            # Check for early stopping recommendation
            recommendation = self.anomaly_detector.recommend_early_stopping(run)
            if recommendation:
                run.early_stop_recommendation = recommendation
                await self.connection_manager.broadcast({
                    "type": "early_stop_recommendation",
                    "run_id": run_id,
                    "recommendation": recommendation,
                    "timestamp": time.time()
                })
            
            # Broadcast metrics update
            aggregated = self.metric_collector.get_aggregated_metrics(run_id)
            await self.connection_manager.broadcast({
                "type": "metrics_update",
                "run_id": run_id,
                "metrics": {
                    "loss": metrics.loss,
                    "eval_loss": metrics.eval_loss,
                    "learning_rate": metrics.learning_rate,
                    "gpu_memory_used": metrics.gpu_memory_used,
                    "gpu_utilization": metrics.gpu_utilization,
                    "throughput": metrics.throughput,
                    "aggregated": aggregated
                },
                "timestamp": time.time()
            }, run_id)
            
            return {"status": "updated", "anomalies_detected": len(new_anomalies)}
        
        @self.app.post("/api/runs/{run_id}/stop")
        async def stop_run(run_id: str):
            """Stop a training run."""
            if run_id not in self.training_runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.training_runs[run_id]
            run.status = RunStatus.STOPPED
            run.end_time = time.time()
            
            await self.connection_manager.broadcast({
                "type": "run_stopped",
                "run_id": run_id,
                "timestamp": time.time()
            }, run_id)
            
            return {"status": "stopped"}
        
        @self.app.get("/api/runs/{run_id}/visualization")
        async def get_visualization(run_id: str, metric: str = "loss", window: int = 100):
            """Get visualization data for a specific metric."""
            if run_id not in self.training_runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.training_runs[run_id]
            recent_metrics = self.metric_collector.get_recent_metrics(run_id, window)
            
            if not recent_metrics:
                return {"error": "No metrics available"}
            
            # Generate Plotly figure
            fig = self._create_visualization(run, recent_metrics, metric)
            
            return {
                "figure": fig.to_dict(),
                "run_id": run_id,
                "metric": metric,
                "window": window
            }
        
        @self.app.get("/api/system/resources")
        async def get_system_resources():
            """Get current system resource utilization."""
            try:
                gpus = GPUtil.getGPUs()
                gpu_info = [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
                    for gpu in gpus
                ]
            except:
                gpu_info = []
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "gpus": gpu_info,
                "timestamp": time.time()
            }
    
    def _setup_background_tasks(self):
        """Setup background tasks for periodic updates."""
        @self.app.on_event("startup")
        async def startup_event():
            # Start background task for system resource monitoring
            asyncio.create_task(self._system_monitor_task())
    
    async def _system_monitor_task(self):
        """Background task to monitor system resources."""
        while True:
            try:
                # Get system resources
                resources = await self.get_system_resources()
                
                # Broadcast to all connections
                await self.connection_manager.broadcast({
                    "type": "system_resources",
                    "resources": resources,
                    "timestamp": time.time()
                })
                
                await asyncio.sleep(self.config["update_interval"])
            except Exception as e:
                logger.error(f"Error in system monitor task: {e}")
                await asyncio.sleep(5)
    
    def _create_visualization(self, run: TrainingRun, metrics: List[TrainingMetrics], metric: str) -> go.Figure:
        """Create Plotly visualization for metrics."""
        if metric == "loss":
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Evaluation Loss', 'GPU Memory', 'Throughput'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Training loss
            steps = [m.step for m in metrics]
            losses = [m.loss for m in metrics]
            fig.add_trace(
                go.Scatter(x=steps, y=losses, mode='lines+markers', name='Training Loss'),
                row=1, col=1
            )
            
            # Evaluation loss
            eval_steps = [m.step for m in metrics if m.eval_loss is not None]
            eval_losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
            if eval_steps:
                fig.add_trace(
                    go.Scatter(x=eval_steps, y=eval_losses, mode='lines+markers', name='Eval Loss'),
                    row=1, col=2
                )
            
            # GPU memory
            gpu_steps = [m.step for m in metrics if m.gpu_memory_used]
            gpu_memory = [m.gpu_memory_used[0] if m.gpu_memory_used else 0 for m in metrics if m.gpu_memory_used]
            if gpu_steps:
                fig.add_trace(
                    go.Scatter(x=gpu_steps, y=gpu_memory, mode='lines', name='GPU Memory'),
                    row=2, col=1
                )
            
            # Throughput
            throughput_steps = [m.step for m in metrics if m.throughput > 0]
            throughputs = [m.throughput for m in metrics if m.throughput > 0]
            if throughput_steps:
                fig.add_trace(
                    go.Scatter(x=throughput_steps, y=throughputs, mode='lines', name='Throughput'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f"Training Metrics for {run.name}",
                height=800,
                showlegend=True
            )
            
            return fig
        
        elif metric == "resources":
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('GPU Utilization', 'GPU Memory'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # GPU utilization
            steps = [m.step for m in metrics if m.gpu_utilization]
            for i in range(len(metrics[0].gpu_utilization) if metrics and metrics[0].gpu_utilization else 0):
                util_values = [m.gpu_utilization[i] if m.gpu_utilization and len(m.gpu_utilization) > i else 0 
                             for m in metrics if m.gpu_utilization]
                fig.add_trace(
                    go.Scatter(x=steps[:len(util_values)], y=util_values, mode='lines', name=f'GPU {i} Util'),
                    row=1, col=1
                )
            
            # GPU memory
            for i in range(len(metrics[0].gpu_memory_used) if metrics and metrics[0].gpu_memory_used else 0):
                mem_values = [m.gpu_memory_used[i] if m.gpu_memory_used and len(m.gpu_memory_used) > i else 0 
                            for m in metrics if m.gpu_memory_used]
                fig.add_trace(
                    go.Scatter(x=steps[:len(mem_values)], y=mem_values, mode='lines', name=f'GPU {i} Mem'),
                    row=1, col=2
                )
            
            fig.update_layout(
                title=f"Resource Utilization for {run.name}",
                height=500,
                showlegend=True
            )
            
            return fig
        
        else:
            # Default: show loss
            return self._create_visualization(run, metrics, "loss")
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>forge Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .status-badge {
            font-size: 0.8em;
            padding: 0.25em 0.5em;
        }
        .anomaly-alert {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">forge Dashboard</a>
            <div class="navbar-text">
                <span class="badge bg-success" id="connection-status">Connected</span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <!-- System Resources -->
            <div class="col-md-3">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">System Resources</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">CPU Usage</label>
                            <div class="progress">
                                <div class="progress-bar" id="cpu-progress" role="progressbar" style="width: 0%"></div>
                            </div>
                            <small class="text-muted" id="cpu-text">0%</small>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Memory Usage</label>
                            <div class="progress">
                                <div class="progress-bar bg-info" id="memory-progress" role="progressbar" style="width: 0%"></div>
                            </div>
                            <small class="text-muted" id="memory-text">0%</small>
                        </div>
                        <div id="gpu-info">
                            <!-- GPU info will be inserted here -->
                        </div>
                    </div>
                </div>

                <!-- Training Runs List -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Training Runs</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="list-group list-group-flush" id="runs-list">
                            <!-- Runs will be inserted here -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="col-md-9">
                <!-- Metrics Visualization -->
                <div class="card mb-4">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Training Metrics</h5>
                        <div>
                            <select class="form-select form-select-sm" id="metric-selector">
                                <option value="loss">Loss Metrics</option>
                                <option value="resources">Resource Usage</option>
                            </select>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="metrics-chart" style="height: 500px;"></div>
                    </div>
                </div>

                <!-- Anomalies and Recommendations -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Anomalies</h5>
                            </div>
                            <div class="card-body">
                                <div id="anomalies-list">
                                    <div class="alert alert-info">No anomalies detected</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Early Stopping Recommendations</h5>
                            </div>
                            <div class="card-body">
                                <div id="recommendations-list">
                                    <div class="alert alert-info">No recommendations available</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentRunId = null;
        let ws = null;
        let metricsChart = null;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            loadRuns();
            setupEventListeners();
        });

        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/global`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('connection-status').className = 'badge bg-success';
                document.getElementById('connection-status').textContent = 'Connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').className = 'badge bg-danger';
                document.getElementById('connection-status').textContent = 'Disconnected';
                // Attempt to reconnect after 3 seconds
                setTimeout(initializeWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'metrics_update':
                    if (data.run_id === currentRunId) {
                        updateMetricsChart(data.metrics);
                    }
                    break;
                case 'anomaly_detected':
                    addAnomaly(data.anomaly);
                    break;
                case 'early_stop_recommendation':
                    addRecommendation(data.recommendation);
                    break;
                case 'system_resources':
                    updateSystemResources(data.resources);
                    break;
                case 'run_created':
                    addRunToList(data.run_id, data.name);
                    break;
                case 'run_stopped':
                    updateRunStatus(data.run_id, 'stopped');
                    break;
            }
        }

        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const data = await response.json();
                
                const runsList = document.getElementById('runs-list');
                runsList.innerHTML = '';
                
                if (data.runs.length === 0) {
                    runsList.innerHTML = '<div class="list-group-item text-muted">No training runs</div>';
                    return;
                }
                
                data.runs.forEach(run => {
                    addRunToList(run.run_id, run.name, run.status);
                });
                
                // Select first run by default
                if (data.runs.length > 0 && !currentRunId) {
                    selectRun(data.runs[0].run_id);
                }
            } catch (error) {
                console.error('Error loading runs:', error);
            }
        }

        function addRunToList(runId, name, status = 'pending') {
            const runsList = document.getElementById('runs-list');
            
            const statusColors = {
                'pending': 'bg-secondary',
                'running': 'bg-success',
                'paused': 'bg-warning',
                'completed': 'bg-primary',
                'failed': 'bg-danger',
                'stopped': 'bg-dark'
            };
            
            const item = document.createElement('a');
            item.href = '#';
            item.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
            item.id = `run-${runId}`;
            item.innerHTML = `
                <div>
                    <strong>${name}</strong>
                    <br>
                    <small class="text-muted">${runId.substring(0, 8)}...</small>
                </div>
                <span class="badge ${statusColors[status]} status-badge">${status}</span>
            `;
            
            item.onclick = function(e) {
                e.preventDefault();
                selectRun(runId);
            };
            
            runsList.appendChild(item);
        }

        function updateRunStatus(runId, status) {
            const item = document.getElementById(`run-${runId}`);
            if (item) {
                const badge = item.querySelector('.badge');
                badge.textContent = status;
                badge.className = `badge bg-${status === 'running' ? 'success' : status === 'stopped' ? 'dark' : 'secondary'} status-badge`;
            }
        }

        async function selectRun(runId) {
            currentRunId = runId;
            
            // Update UI to show selected run
            document.querySelectorAll('#runs-list .list-group-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById(`run-${runId}`).classList.add('active');
            
            // Load run details
            await loadRunDetails(runId);
            
            // Subscribe to run-specific WebSocket updates
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'subscribe', run_id: runId }));
            }
        }

        async function loadRunDetails(runId) {
            try {
                const response = await fetch(`/api/runs/${runId}`);
                const data = await response.json();
                
                // Update metrics chart
                if (data.recent_metrics && data.recent_metrics.length > 0) {
                    updateMetricsChart({
                        recent_metrics: data.recent_metrics,
                        aggregated: data.aggregated
                    });
                }
                
                // Update anomalies
                if (data.run.anomalies_count > 0) {
                    // Would need to fetch anomalies separately
                }
                
                // Update recommendations
                if (data.run.early_stop_recommendation) {
                    addRecommendation(data.run.early_stop_recommendation);
                }
            } catch (error) {
                console.error('Error loading run details:', error);
            }
        }

        function updateMetricsChart(metrics) {
            const chartDiv = document.getElementById('metrics-chart');
            
            if (!metricsChart) {
                // Initialize chart
                const trace1 = {
                    x: [],
                    y: [],
                    mode: 'lines+markers',
                    name: 'Training Loss',
                    line: { color: '#1f77b4' }
                };
                
                const trace2 = {
                    x: [],
                    y: [],
                    mode: 'lines+markers',
                    name: 'Eval Loss',
                    line: { color: '#ff7f0e' }
                };
                
                const layout = {
                    title: 'Training Progress',
                    xaxis: { title: 'Step' },
                    yaxis: { title: 'Loss' },
                    showlegend: true,
                    height: 450
                };
                
                Plotly.newPlot(chartDiv, [trace1, trace2], layout);
                metricsChart = true;
            }
            
            // Update chart with new data
            if (metrics.recent_metrics) {
                const steps = metrics.recent_metrics.map(m => m.step);
                const losses = metrics.recent_metrics.map(m => m.loss);
                const evalLosses = metrics.recent_metrics.map(m => m.eval_loss).filter(l => l !== null);
                const evalSteps = metrics.recent_metrics.filter(m => m.eval_loss !== null).map(m => m.step);
                
                Plotly.react(chartDiv, [
                    {
                        x: steps,
                        y: losses,
                        mode: 'lines+markers',
                        name: 'Training Loss'
                    },
                    {
                        x: evalSteps,
                        y: evalLosses,
                        mode: 'lines+markers',
                        name: 'Eval Loss'
                    }
                ], {
                    title: 'Training Progress',
                    xaxis: { title: 'Step' },
                    yaxis: { title: 'Loss' },
                    showlegend: true,
                    height: 450
                });
            }
        }

        function updateSystemResources(resources) {
            // Update CPU
            document.getElementById('cpu-progress').style.width = `${resources.cpu_percent}%`;
            document.getElementById('cpu-text').textContent = `${resources.cpu_percent}%`;
            
            // Update Memory
            document.getElementById('memory-progress').style.width = `${resources.memory_percent}%`;
            document.getElementById('memory-text').textContent = `${resources.memory_percent}%`;
            
            // Update GPU info
            const gpuInfo = document.getElementById('gpu-info');
            gpuInfo.innerHTML = '';
            
            if (resources.gpus && resources.gpus.length > 0) {
                resources.gpus.forEach((gpu, index) => {
                    const gpuDiv = document.createElement('div');
                    gpuDiv.className = 'mb-3';
                    gpuDiv.innerHTML = `
                        <label class="form-label">GPU ${index}: ${gpu.name}</label>
                        <div class="progress mb-1">
                            <div class="progress-bar bg-warning" style="width: ${gpu.load}%"></div>
                        </div>
                        <small class="text-muted">
                            ${gpu.load.toFixed(1)}% | 
                            ${gpu.memory_used}MB / ${gpu.memory_total}MB | 
                            ${gpu.temperature}°C
                        </small>
                    `;
                    gpuInfo.appendChild(gpuDiv);
                });
            } else {
                gpuInfo.innerHTML = '<div class="alert alert-info">No GPU detected</div>';
            }
        }

        function addAnomaly(anomaly) {
            const anomaliesList = document.getElementById('anomalies-list');
            
            // Clear "no anomalies" message if present
            if (anomaliesList.querySelector('.alert-info')) {
                anomaliesList.innerHTML = '';
            }
            
            const severityColors = {
                'low': 'alert-warning',
                'medium': 'alert-warning',
                'high': 'alert-danger',
                'critical': 'alert-danger'
            };
            
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${severityColors[anomaly.severity]} anomaly-alert`;
            alertDiv.innerHTML = `
                <strong>${anomaly.type.replace('_', ' ').toUpperCase()}</strong><br>
                ${anomaly.message}<br>
                <small class="text-muted">${new Date(anomaly.timestamp * 1000).toLocaleTimeString()}</small>
            `;
            
            anomaliesList.insertBefore(alertDiv, anomaliesList.firstChild);
            
            // Keep only last 10 anomalies
            while (anomaliesList.children.length > 10) {
                anomaliesList.removeChild(anomaliesList.lastChild);
            }
        }

        function addRecommendation(recommendation) {
            const recommendationsList = document.getElementById('recommendations-list');
            
            // Clear "no recommendations" message if present
            if (recommendationsList.querySelector('.alert-info')) {
                recommendationsList.innerHTML = '';
            }
            
            const confidenceColors = {
                0.8: 'alert-success',
                0.9: 'alert-warning',
                1.0: 'alert-danger'
            };
            
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${confidenceColors[recommendation.confidence] || 'alert-info'}`;
            alertDiv.innerHTML = `
                <strong>Recommendation: ${recommendation.recommendation.toUpperCase()}</strong><br>
                ${recommendation.reason}<br>
                <em>${recommendation.suggested_action}</em><br>
                <small class="text-muted">Confidence: ${(recommendation.confidence * 100).toFixed(0)}%</small>
            `;
            
            recommendationsList.innerHTML = '';
            recommendationsList.appendChild(alertDiv);
        }

        function setupEventListeners() {
            document.getElementById('metric-selector').addEventListener('change', function(e) {
                if (currentRunId) {
                    loadRunDetails(currentRunId);
                }
            });
        }
    </script>
</body>
</html>
        """


# Factory function to create the dashboard app
def create_dashboard_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure the dashboard application."""
    dashboard = DashboardApp(config_path)
    return dashboard.app


# CLI entry point for running the dashboard
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Training Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    app = create_dashboard_app(args.config)
    
    print(f"Starting forge Dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )