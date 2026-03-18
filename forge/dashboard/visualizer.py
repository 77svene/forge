"""
Real-time Training Dashboard for forge
Provides web-based monitoring with real-time metrics, resource utilization,
anomaly detection, and early stopping recommendations.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pydantic import BaseModel, Field
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLING_RATE = 5  # seconds
MAX_HISTORY_POINTS = 1000
ANOMALY_THRESHOLD = 2.0  # standard deviations
EARLY_STOP_PATIENCE = 10
MIN_IMPROVEMENT = 0.001


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    EARLY_STOPPED = "early_stopped"


class MetricType(str, Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    GPU_MEMORY = "gpu_memory"
    GPU_UTIL = "gpu_util"
    CPU_UTIL = "cpu_util"
    MEMORY_UTIL = "memory_util"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific timestamp."""
    timestamp: float
    step: int
    epoch: float
    metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "resource_usage": self.resource_usage
        }


@dataclass
class TrainingRun:
    """Represents a single training run."""
    run_id: str
    name: str
    model_name: str
    start_time: float
    status: RunStatus = RunStatus.RUNNING
    config: Dict[str, Any] = field(default_factory=dict)
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    last_improvement_step: int = 0
    anomaly_flags: List[str] = field(default_factory=list)
    early_stop_recommendation: bool = False
    early_stop_reason: Optional[str] = None
    
    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics to history with rolling window."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > MAX_HISTORY_POINTS:
            self.metrics_history = self.metrics_history[-MAX_HISTORY_POINTS:]
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metric_trend(self, metric_name: str, window: int = 20) -> Optional[float]:
        """Calculate trend for a specific metric (positive = improving)."""
        if len(self.metrics_history) < window:
            return None
        
        recent = self.metrics_history[-window:]
        values = [m.metrics.get(metric_name) for m in recent if metric_name in m.metrics]
        
        if len(values) < window // 2:
            return None
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # For loss, negative slope is good; for accuracy, positive is good
        if "loss" in metric_name.lower():
            return -slope
        else:
            return slope
    
    def check_anomalies(self) -> List[str]:
        """Detect anomalies in training metrics."""
        anomalies = []
        if len(self.metrics_history) < 10:
            return anomalies
        
        recent = self.metrics_history[-10:]
        
        # Check for NaN or infinite values
        for metrics in recent:
            for key, value in metrics.metrics.items():
                if np.isnan(value) or np.isinf(value):
                    anomalies.append(f"NaN/Inf detected in {key}")
        
        # Check for exploding gradients
        if "gradient_norm" in recent[-1].metrics:
            grad_norms = [m.metrics.get("gradient_norm", 0) for m in recent]
            mean_norm = np.mean(grad_norms)
            std_norm = np.std(grad_norms)
            if std_norm > 0 and grad_norms[-1] > mean_norm + 3 * std_norm:
                anomalies.append("Exploding gradients detected")
        
        # Check for loss spikes
        if "loss" in recent[-1].metrics:
            losses = [m.metrics.get("loss", float('inf')) for m in recent]
            mean_loss = np.mean(losses[:-1])
            std_loss = np.std(losses[:-1])
            if std_loss > 0 and losses[-1] > mean_loss + 2 * std_loss:
                anomalies.append("Loss spike detected")
        
        # Check for resource exhaustion
        if "gpu_memory" in recent[-1].resource_usage:
            gpu_mem = recent[-1].resource_usage["gpu_memory"]
            if gpu_mem > 0.95:  # 95% GPU memory usage
                anomalies.append("GPU memory near exhaustion")
        
        self.anomaly_flags = anomalies
        return anomalies
    
    def check_early_stopping(self, metric_name: str = "loss") -> Tuple[bool, Optional[str]]:
        """Determine if early stopping should be recommended."""
        if len(self.metrics_history) < EARLY_STOP_PATIENCE * 2:
            return False, None
        
        # Get the primary metric for early stopping
        recent_values = []
        for m in self.metrics_history[-EARLY_STOP_PATIENCE * 2:]:
            if metric_name in m.metrics:
                recent_values.append((m.step, m.metrics[metric_name]))
        
        if len(recent_values) < EARLY_STOP_PATIENCE:
            return False, None
        
        # Check if metric has plateaued
        steps, values = zip(*recent_values)
        values = np.array(values)
        
        # Calculate improvement over patience window
        window_size = min(EARLY_STOP_PATIENCE, len(values) // 2)
        if window_size < 5:
            return False, None
        
        first_half = values[:window_size]
        second_half = values[-window_size:]
        
        # For loss, we want decrease; for accuracy, we want increase
        if "loss" in metric_name.lower():
            improvement = np.mean(first_half) - np.mean(second_half)
            min_improvement = MIN_IMPROVEMENT * np.mean(first_half)
        else:
            improvement = np.mean(second_half) - np.mean(first_half)
            min_improvement = MIN_IMPROVEMENT * np.mean(first_half)
        
        # Check if improvement is below threshold
        if improvement < min_improvement:
            reason = f"No significant improvement in {metric_name} for {EARLY_STOP_PATIENCE} steps"
            self.early_stop_recommendation = True
            self.early_stop_reason = reason
            return True, reason
        
        # Check for divergence
        if "loss" in metric_name.lower():
            recent_trend = np.polyfit(range(len(values[-10:])), values[-10:], 1)[0]
            if recent_trend > 0:  # Loss increasing
                reason = f"{metric_name} is diverging (increasing trend)"
                self.early_stop_recommendation = True
                self.early_stop_reason = reason
                return True, reason
        
        self.early_stop_recommendation = False
        self.early_stop_reason = None
        return False, None


class ResourceMonitor:
    """Monitor system resource utilization."""
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU information and utilization."""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                "temperature": gpu.temperature
            } for gpu in gpus]
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return []
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU utilization."""
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory utilization."""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
            "free": mem.free
        }
    
    @staticmethod
    def get_disk_info() -> Dict[str, Any]:
        """Get disk utilization."""
        disk = psutil.disk_usage('/')
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }


class MetricCollector:
    """Collect and aggregate metrics from training runs."""
    
    def __init__(self, sampling_rate: int = DEFAULT_SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.runs: Dict[str, TrainingRun] = {}
        self.active_connections: Set[WebSocket] = set()
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def add_run(self, run: TrainingRun):
        """Add a new training run to monitor."""
        self.runs[run.run_id] = run
        logger.info(f"Added training run: {run.run_id} - {run.name}")
    
    def update_run_metrics(self, run_id: str, metrics: TrainingMetrics):
        """Update metrics for a specific run."""
        if run_id not in self.runs:
            logger.warning(f"Run {run_id} not found")
            return
        
        run = self.runs[run_id]
        run.add_metrics(metrics)
        
        # Update best metric
        if "loss" in metrics.metrics:
            current_loss = metrics.metrics["loss"]
            if run.best_metric is None or current_loss < run.best_metric:
                run.best_metric = current_loss
                run.best_step = metrics.step
                run.last_improvement_step = metrics.step
        
        # Check for anomalies
        anomalies = run.check_anomalies()
        
        # Check early stopping
        should_stop, reason = run.check_early_stopping()
        
        # Broadcast update to all connected clients
        asyncio.create_task(self._broadcast_update(run_id))
    
    async def _broadcast_update(self, run_id: str):
        """Broadcast metrics update to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        run = self.runs.get(run_id)
        if not run:
            return
        
        message = {
            "type": "metrics_update",
            "run_id": run_id,
            "timestamp": time.time(),
            "data": {
                "run": asdict(run),
                "latest_metrics": run.get_latest_metrics().to_dict() if run.get_latest_metrics() else None,
                "anomalies": run.anomaly_flags,
                "early_stop_recommendation": run.early_stop_recommendation,
                "early_stop_reason": run.early_stop_reason
            }
        }
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        # Clean up disconnected clients
        self.active_connections -= disconnected
    
    async def _monitor_resources(self):
        """Continuously monitor system resources."""
        while True:
            try:
                resource_data = {
                    "timestamp": time.time(),
                    "gpus": ResourceMonitor.get_gpu_info(),
                    "cpu": ResourceMonitor.get_cpu_info(),
                    "memory": ResourceMonitor.get_memory_info(),
                    "disk": ResourceMonitor.get_disk_info()
                }
                
                # Broadcast resource update
                if self.active_connections:
                    message = {
                        "type": "resource_update",
                        "data": resource_data
                    }
                    
                    disconnected = set()
                    for connection in self.active_connections:
                        try:
                            await connection.send_json(message)
                        except Exception:
                            disconnected.add(connection)
                    
                    self.active_connections -= disconnected
                
                await asyncio.sleep(self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.sampling_rate)
    
    def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_resources())
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("Stopped resource monitoring")


class DashboardVisualizer:
    """Main dashboard application with FastAPI and WebSocket support."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8050):
        self.host = host
        self.port = port
        self.app = FastAPI(title="forge Training Dashboard")
        self.metric_collector = MetricCollector()
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket()
        
        # Start monitoring
        self.metric_collector.start_monitoring()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        
        @self.app.get("/")
        async def dashboard():
            """Serve the main dashboard HTML."""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/runs")
        async def get_runs():
            """Get all training runs."""
            runs_data = []
            for run_id, run in self.metric_collector.runs.items():
                latest = run.get_latest_metrics()
                runs_data.append({
                    "run_id": run_id,
                    "name": run.name,
                    "model_name": run.model_name,
                    "status": run.status.value,
                    "start_time": run.start_time,
                    "step": latest.step if latest else 0,
                    "epoch": latest.epoch if latest else 0,
                    "best_metric": run.best_metric,
                    "anomaly_count": len(run.anomaly_flags),
                    "early_stop_recommendation": run.early_stop_recommendation
                })
            return {"runs": runs_data}
        
        @self.app.get("/api/runs/{run_id}")
        async def get_run(run_id: str):
            """Get detailed information for a specific run."""
            if run_id not in self.metric_collector.runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.metric_collector.runs[run_id]
            return {
                "run": asdict(run),
                "metrics_summary": self._get_metrics_summary(run)
            }
        
        @self.app.get("/api/runs/{run_id}/metrics")
        async def get_run_metrics(
            run_id: str,
            metric_name: Optional[str] = None,
            limit: int = Query(100, ge=1, le=1000)
        ):
            """Get metrics history for a specific run."""
            if run_id not in self.metric_collector.runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.metric_collector.runs[run_id]
            history = run.metrics_history[-limit:] if limit else run.metrics_history
            
            if metric_name:
                # Filter for specific metric
                filtered = []
                for m in history:
                    if metric_name in m.metrics:
                        filtered.append({
                            "timestamp": m.timestamp,
                            "step": m.step,
                            "value": m.metrics[metric_name]
                        })
                return {"metric": metric_name, "data": filtered}
            else:
                # Return all metrics
                return {"data": [m.to_dict() for m in history]}
        
        @self.app.get("/api/resources")
        async def get_resources():
            """Get current resource utilization."""
            return {
                "gpus": ResourceMonitor.get_gpu_info(),
                "cpu": ResourceMonitor.get_cpu_info(),
                "memory": ResourceMonitor.get_memory_info(),
                "disk": ResourceMonitor.get_disk_info()
            }
        
        @self.app.get("/api/visualizations/{run_id}")
        async def get_visualization(
            run_id: str,
            viz_type: str = Query("metrics", enum=["metrics", "resources", "comparison"])
        ):
            """Generate visualization for a run."""
            if run_id not in self.metric_collector.runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.metric_collector.runs[run_id]
            
            if viz_type == "metrics":
                fig = self._create_metrics_visualization(run)
            elif viz_type == "resources":
                fig = self._create_resources_visualization(run)
            elif viz_type == "comparison":
                fig = self._create_comparison_visualization()
            else:
                raise HTTPException(status_code=400, detail="Invalid visualization type")
            
            return {"plot": fig.to_json()}
        
        @self.app.post("/api/runs/{run_id}/stop")
        async def stop_run(run_id: str):
            """Stop a training run (simulated)."""
            if run_id not in self.metric_collector.runs:
                raise HTTPException(status_code=404, detail="Run not found")
            
            run = self.metric_collector.runs[run_id]
            run.status = RunStatus.STOPPED
            
            # Broadcast status update
            await self.metric_collector._broadcast_update(run_id)
            
            return {"status": "stopped", "run_id": run_id}
    
    def _setup_websocket(self):
        """Setup WebSocket endpoint for real-time updates."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.metric_collector.active_connections.add(websocket)
            
            try:
                # Send initial data
                initial_data = {
                    "type": "init",
                    "data": {
                        "runs": list(self.metric_collector.runs.keys()),
                        "timestamp": time.time()
                    }
                }
                await websocket.send_json(initial_data)
                
                # Keep connection alive and handle messages
                while True:
                    data = await websocket.receive_text()
                    # Handle client messages if needed
                    
            except WebSocketDisconnect:
                self.metric_collector.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.metric_collector.active_connections:
                    self.metric_collector.active_connections.remove(websocket)
    
    def _get_metrics_summary(self, run: TrainingRun) -> Dict[str, Any]:
        """Generate summary statistics for a run's metrics."""
        if not run.metrics_history:
            return {}
        
        summary = {}
        metric_names = set()
        
        # Collect all metric names
        for m in run.metrics_history:
            metric_names.update(m.metrics.keys())
        
        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = [m.metrics.get(metric_name) for m in run.metrics_history 
                     if metric_name in m.metrics]
            
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "trend": run.get_metric_trend(metric_name)
                }
        
        return summary
    
    def _create_metrics_visualization(self, run: TrainingRun) -> go.Figure:
        """Create interactive metrics visualization."""
        if not run.metrics_history:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(title="No metrics available")
            return fig
        
        # Extract data
        steps = [m.step for m in run.metrics_history]
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in run.metrics_history]
        
        # Get all metric names
        metric_names = set()
        for m in run.metrics_history:
            metric_names.update(m.metrics.keys())
        
        # Create subplots for different metric groups
        loss_metrics = [m for m in metric_names if "loss" in m.lower()]
        accuracy_metrics = [m for m in metric_names if "acc" in m.lower() or "f1" in m.lower()]
        other_metrics = metric_names - set(loss_metrics) - set(accuracy_metrics)
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Loss Metrics", "Accuracy Metrics", "Other Metrics"),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Plot loss metrics
        for metric in loss_metrics:
            values = [m.metrics.get(metric) for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=values,
                    name=metric,
                    mode='lines',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot accuracy metrics
        for metric in accuracy_metrics:
            values = [m.metrics.get(metric) for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=values,
                    name=metric,
                    mode='lines',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Plot other metrics
        for metric in list(other_metrics)[:5]:  # Limit to 5 metrics
            values = [m.metrics.get(metric) for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=values,
                    name=metric,
                    mode='lines',
                    line=dict(width=2)
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Training Metrics: {run.name}",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Add anomaly markers
        anomaly_steps = []
        anomaly_labels = []
        for i, m in enumerate(run.metrics_history):
            if i > 0 and run.anomaly_flags:  # Check for anomalies at this step
                # Simple heuristic: if there was a recent anomaly
                anomaly_steps.append(m.step)
                anomaly_labels.append("Anomaly")
        
        if anomaly_steps:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_steps,
                    y=[0] * len(anomaly_steps),  # Will be positioned on the plot
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='x'),
                    name='Anomalies',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        return fig
    
    def _create_resources_visualization(self, run: TrainingRun) -> go.Figure:
        """Create resource utilization visualization."""
        if not run.metrics_history:
            fig = go.Figure()
            fig.update_layout(title="No resource data available")
            return fig
        
        # Extract resource data
        steps = [m.step for m in run.metrics_history]
        
        # Create subplots for different resources
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("GPU Memory", "GPU Utilization", "CPU Utilization", "System Memory"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # GPU Memory
        if any("gpu_memory" in m.resource_usage for m in run.metrics_history):
            gpu_mem = [m.resource_usage.get("gpu_memory", 0) * 100 for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=gpu_mem,
                    name="GPU Memory %",
                    mode='lines',
                    fill='tozeroy'
                ),
                row=1, col=1
            )
        
        # GPU Utilization
        if any("gpu_util" in m.resource_usage for m in run.metrics_history):
            gpu_util = [m.resource_usage.get("gpu_util", 0) * 100 for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=gpu_util,
                    name="GPU Utilization %",
                    mode='lines',
                    fill='tozeroy'
                ),
                row=1, col=2
            )
        
        # CPU Utilization
        if any("cpu_util" in m.resource_usage for m in run.metrics_history):
            cpu_util = [m.resource_usage.get("cpu_util", 0) for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=cpu_util,
                    name="CPU Utilization %",
                    mode='lines',
                    fill='tozeroy'
                ),
                row=2, col=1
            )
        
        # System Memory
        if any("memory_util" in m.resource_usage for m in run.metrics_history):
            mem_util = [m.resource_usage.get("memory_util", 0) for m in run.metrics_history]
            fig.add_trace(
                go.Scatter(
                    x=steps, y=mem_util,
                    name="Memory Utilization %",
                    mode='lines',
                    fill='tozeroy'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"Resource Utilization: {run.name}",
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(range=[0, 100], title_text="Percentage")
        fig.update_xaxes(title_text="Training Step")
        
        return fig
    
    def _create_comparison_visualization(self) -> go.Figure:
        """Create comparison visualization across multiple runs."""
        runs = list(self.metric_collector.runs.values())
        
        if not runs:
            fig = go.Figure()
            fig.update_layout(title="No runs available for comparison")
            return fig
        
        # Find common metrics across runs
        common_metrics = set()
        for run in runs:
            if run.metrics_history:
                for m in run.metrics_history[:10]:  # Check first 10 entries
                    common_metrics.update(m.metrics.keys())
        
        # Filter to metrics present in at least 2 runs
        metric_counts = defaultdict(int)
        for run in runs:
            if run.metrics_history:
                for m in run.metrics_history[:10]:
                    for metric in m.metrics.keys():
                        metric_counts[metric] += 1
        
        common_metrics = [m for m, count in metric_counts.items() if count >= 2]
        
        if not common_metrics:
            fig = go.Figure()
            fig.update_layout(title="No common metrics found across runs")
            return fig
        
        # Select primary metric for comparison
        primary_metric = "loss" if "loss" in common_metrics else common_metrics[0]
        
        # Create comparison plot
        fig = go.Figure()
        
        for run in runs:
            if not run.metrics_history:
                continue
            
            # Extract metric values
            steps = []
            values = []
            for m in run.metrics_history:
                if primary_metric in m.metrics:
                    steps.append(m.step)
                    values.append(m.metrics[primary_metric])
            
            if steps and values:
                fig.add_trace(
                    go.Scatter(
                        x=steps, y=values,
                        name=f"{run.name} ({run.model_name})",
                        mode='lines',
                        line=dict(width=2)
                    )
                )
        
        fig.update_layout(
            title=f"Run Comparison: {primary_metric}",
            xaxis_title="Training Step",
            yaxis_title=primary_metric,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML with embedded Plotly."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>forge Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .anomaly-alert {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .early-stop-alert {
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-running {
            background-color: #d4edda;
            color: #155724;
        }
        .status-stopped {
            background-color: #f8d7da;
            color: #721c24;
        }
        .status-completed {
            background-color: #cce5ff;
            color: #004085;
        }
        #metrics-plot, #resources-plot, #comparison-plot {
            height: 400px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="dashboard-header">
            <h1>🚀 forge Training Dashboard</h1>
            <p class="mb-0">Real-time monitoring of training runs with anomaly detection and early stopping recommendations</p>
        </div>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Training Runs</h5>
                    </div>
                    <div class="card-body">
                        <div id="runs-list">
                            <div class="text-center py-4">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <p class="mt-2">Loading runs...</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">System Resources</h5>
                    </div>
                    <div class="card-body">
                        <div id="resources-summary">
                            <div class="metric-card mb-2">
                                <small class="text-muted">GPU Memory</small>
                                <div id="gpu-memory" class="h5 mb-0">--</div>
                            </div>
                            <div class="metric-card mb-2">
                                <small class="text-muted">CPU Usage</small>
                                <div id="cpu-usage" class="h5 mb-0">--</div>
                            </div>
                            <div class="metric-card">
                                <small class="text-muted">Memory Usage</small>
                                <div id="memory-usage" class="h5 mb-0">--</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="alerts-container">
                            <p class="text-muted">No alerts</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Metrics Visualization</h5>
                        <div>
                            <button class="btn btn-sm btn-outline-primary" onclick="refreshData()">Refresh</button>
                            <select id="run-selector" class="form-select form-select-sm d-inline-block w-auto ms-2">
                                <option value="">Select a run</option>
                            </select>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="metrics-plot"></div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Resource Utilization</h5>
                            </div>
                            <div class="card-body">
                                <div id="resources-plot"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Run Comparison</h5>
                            </div>
                            <div class="card-body">
                                <div id="comparison-plot"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Run Details</h5>
                    </div>
                    <div class="card-body">
                        <div id="run-details">
                            <p class="text-muted">Select a run to view details</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let ws = null;
        let currentRunId = null;
        let runsData = {};
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                refreshData();
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected, reconnecting in 5s...');
                setTimeout(initWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        // Handle incoming WebSocket messages
        function handleMessage(message) {
            switch(message.type) {
                case 'init':
                    // Initial data load
                    loadRuns();
                    break;
                    
                case 'metrics_update':
                    // Update metrics for a specific run
                    updateRunMetrics(message.run_id, message.data);
                    break;
                    
                case 'resource_update':
                    // Update resource utilization
                    updateResources(message.data);
                    break;
            }
        }
        
        // Load all training runs
        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const data = await response.json();
                
                runsData = {};
                data.runs.forEach(run => {
                    runsData[run.run_id] = run;
                });
                
                updateRunsList(data.runs);
                updateRunSelector(data.runs);
                
                // Auto-select first run if none selected
                if (!currentRunId && data.runs.length > 0) {
                    selectRun(data.runs[0].run_id);
                }
            } catch (error) {
                console.error('Error loading runs:', error);
            }
        }
        
        // Update runs list in sidebar
        function updateRunsList(runs) {
            const container = document.getElementById('runs-list');
            
            if (runs.length === 0) {
                container.innerHTML = '<p class="text-muted">No training runs</p>';
                return;
            }
            
            let html = '<div class="list-group">';
            runs.forEach(run => {
                const statusClass = `status-${run.status}`;
                const anomalyBadge = run.anomaly_count > 0 ? 
                    `<span class="badge bg-warning text-dark ms-2">${run.anomaly_count} anomalies</span>` : '';
                const earlyStopBadge = run.early_stop_recommendation ? 
                    '<span class="badge bg-info ms-2">Early Stop</span>' : '';
                
                html += `
                    <a href="#" class="list-group-item list-group-item-action ${currentRunId === run.run_id ? 'active' : ''}" 
                       onclick="selectRun('${run.run_id}')">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${run.name}</strong>
                                <div class="small text-muted">${run.model_name}</div>
                            </div>
                            <div>
                                <span class="status-badge ${statusClass}">${run.status}</span>
                                ${anomalyBadge}
                                ${earlyStopBadge}
                            </div>
                        </div>
                        <div class="small mt-1">
                            Step: ${run.step} | Epoch: ${run.epoch.toFixed(2)}
                            ${run.best_metric ? ` | Best: ${run.best_metric.toFixed(4)}` : ''}
                        </div>
                    </a>
                `;
            });
            html += '</div>';
            
            container.innerHTML = html;
        }
        
        // Update run selector dropdown
        function updateRunSelector(runs) {
            const selector = document.getElementById('run-selector');
            selector.innerHTML = '<option value="">Select a run</option>';
            
            runs.forEach(run => {
                const option = document.createElement('option');
                option.value = run.run_id;
                option.textContent = `${run.name} (${run.model_name})`;
                selector.appendChild(option);
            });
            
            selector.value = currentRunId || '';
        }
        
        // Select a run to view details
        async function selectRun(runId) {
            currentRunId = runId;
            
            // Update UI
            document.querySelectorAll('#runs-list .list-group-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Load run details
            try {
                const response = await fetch(`/api/runs/${runId}`);
                const data = await response.json();
                
                updateRunDetails(data.run, data.metrics_summary);
                loadVisualizations(runId);
                
                // Update selector
                document.getElementById('run-selector').value = runId;
                
            } catch (error) {
                console.error('Error loading run details:', error);
            }
        }
        
        // Update run details panel
        function updateRunDetails(run, metricsSummary) {
            const container = document.getElementById('run-details');
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Run Information</h6>
                        <table class="table table-sm">
                            <tr><th>Name:</th><td>${run.name}</td></tr>
                            <tr><th>Model:</th><td>${run.model_name}</td></tr>
                            <tr><th>Status:</th><td><span class="status-badge status-${run.status}">${run.status}</span></td></tr>
                            <tr><th>Start Time:</th><td>${new Date(run.start_time * 1000).toLocaleString()}</td></tr>
                            <tr><th>Current Step:</th><td>${run.metrics_history.length > 0 ? run.metrics_history[run.metrics_history.length - 1].step : 0}</td></tr>
                            <tr><th>Best Metric:</th><td>${run.best_metric ? run.best_metric.toFixed(6) : 'N/A'}</td></tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h6>Configuration</h6>
                        <pre class="bg-light p-2 rounded" style="max-height: 200px; overflow-y: auto;">${JSON.stringify(run.config, null, 2)}</pre>
                    </div>
                </div>
            `;
            
            // Add anomalies section
            if (run.anomaly_flags && run.anomaly_flags.length > 0) {
                html += `
                    <div class="anomaly-alert">
                        <h6>⚠️ Anomalies Detected</h6>
                        <ul class="mb-0">
                            ${run.anomaly_flags.map(flag => `<li>${flag}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            // Add early stopping recommendation
            if (run.early_stop_recommendation) {
                html += `
                    <div class="early-stop-alert">
                        <h6>🛑 Early Stopping Recommended</h6>
                        <p class="mb-0">${run.early_stop_reason}</p>
                        <button class="btn btn-sm btn-outline-danger mt-2" onclick="stopRun('${run.run_id}')">
                            Stop Training
                        </button>
                    </div>
                `;
            }
            
            // Add metrics summary
            if (metricsSummary && Object.keys(metricsSummary).length > 0) {
                html += '<h6 class="mt-3">Metrics Summary</h6><div class="row">';
                
                Object.entries(metricsSummary).forEach(([metric, stats]) => {
                    const trendIcon = stats.trend > 0 ? '↑' : stats.trend < 0 ? '↓' : '→';
                    const trendColor = (metric.includes('loss') && stats.trend > 0) || 
                                      (!metric.includes('loss') && stats.trend < 0) ? 'text-danger' : 'text-success';
                    
                    html += `
                        <div class="col-md-4 mb-2">
                            <div class="metric-card">
                                <small class="text-muted">${metric}</small>
                                <div class="h5 mb-0">${stats.current.toFixed(4)} 
                                    <span class="${trendColor}">${trendIcon}</span>
                                </div>
                                <small class="text-muted">
                                    Min: ${stats.min.toFixed(4)} | Max: ${stats.max.toFixed(4)}
                                </small>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            container.innerHTML = html;
        }
        
        // Load visualizations for a run
        async function loadVisualizations(runId) {
            try {
                // Load metrics visualization
                const metricsResponse = await fetch(`/api/visualizations/${runId}?viz_type=metrics`);
                const metricsData = await metricsResponse.json();
                Plotly.newPlot('metrics-plot', JSON.parse(metricsData.plot).data, JSON.parse(metricsData.plot).layout);
                
                // Load resources visualization
                const resourcesResponse = await fetch(`/api/visualizations/${runId}?viz_type=resources`);
                const resourcesData = await resourcesResponse.json();
                Plotly.newPlot('resources-plot', JSON.parse(resourcesData.plot).data, JSON.parse(resourcesData.plot).layout);
                
                // Load comparison visualization
                const comparisonResponse = await fetch(`/api/visualizations/${runId}?viz_type=comparison`);
                const comparisonData = await comparisonResponse.json();
                Plotly.newPlot('comparison-plot', JSON.parse(comparisonData.plot).data, JSON.parse(comparisonData.plot).layout);
                
            } catch (error) {
                console.error('Error loading visualizations:', error);
            }
        }
        
        // Update run metrics in real-time
        function updateRunMetrics(runId, data) {
            if (runsData[runId]) {
                // Update run data
                Object.assign(runsData[runId], data.run);
                
                // Update UI if this is the currently selected run
                if (currentRunId === runId) {
                    updateRunDetails(data.run, null);
                    
                    // Reload visualizations if we have new metrics
                    if (data.latest_metrics) {
                        loadVisualizations(runId);
                    }
                }
                
                // Update runs list
                updateRunsList(Object.values(runsData));
            }
        }
        
        // Update resource utilization display
        function updateResources(resourceData) {
            // Update GPU memory
            if (resourceData.gpus && resourceData.gpus.length > 0) {
                const gpu = resourceData.gpus[0];
                document.getElementById('gpu-memory').textContent = 
                    `${gpu.memory_percent.toFixed(1)}% (${gpu.memory_used}MB / ${gpu.memory_total}MB)`;
            }
            
            // Update CPU usage
            document.getElementById('cpu-usage').textContent = 
                `${resourceData.cpu.percent.toFixed(1)}%`;
            
            // Update memory usage
            document.getElementById('memory-usage').textContent = 
                `${resourceData.memory.percent.toFixed(1)}% (${(resourceData.memory.used / 1024 / 1024 / 1024).toFixed(1)}GB / ${(resourceData.memory.total / 1024 / 1024 / 1024).toFixed(1)}GB)`;
            
            // Update alerts based on resource usage
            updateAlerts(resourceData);
        }
        
        // Update alerts based on resource usage and anomalies
        function updateAlerts(resourceData) {
            const alertsContainer = document.getElementById('alerts-container');
            let alerts = [];
            
            // GPU memory alert
            if (resourceData.gpus && resourceData.gpus.length > 0) {
                const gpu = resourceData.gpus[0];
                if (gpu.memory_percent > 90) {
                    alerts.push({
                        type: 'warning',
                        message: `GPU memory usage high: ${gpu.memory_percent.toFixed(1)}%`
                    });
                }
            }
            
            // CPU usage alert
            if (resourceData.cpu.percent > 90) {
                alerts.push({
                    type: 'warning',
                    message: `CPU usage high: ${resourceData.cpu.percent.toFixed(1)}%`
                });
            }
            
            // Memory usage alert
            if (resourceData.memory.percent > 90) {
                alerts.push({
                    type: 'warning',
                    message: `Memory usage high: ${resourceData.memory.percent.toFixed(1)}%`
                });
            }
            
            // Update alerts display
            if (alerts.length === 0) {
                alertsContainer.innerHTML = '<p class="text-muted">No alerts</p>';
            } else {
                let html = '';
                alerts.forEach(alert => {
                    const alertClass = alert.type === 'warning' ? 'anomaly-alert' : 'early-stop-alert';
                    html += `<div class="${alertClass}">${alert.message}</div>`;
                });
                alertsContainer.innerHTML = html;
            }
        }
        
        // Refresh all data
        function refreshData() {
            loadRuns();
            if (currentRunId) {
                loadVisualizations(currentRunId);
            }
        }
        
        // Stop a training run
        async function stopRun(runId) {
            if (!confirm('Are you sure you want to stop this training run?')) {
                return;
            }
            
            try {
                const response = await fetch(`/api/runs/${runId}/stop`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('Training run stopped');
                    refreshData();
                } else {
                    alert('Failed to stop training run');
                }
            } catch (error) {
                console.error('Error stopping run:', error);
                alert('Error stopping training run');
            }
        }
        
        // Event listeners
        document.getElementById('run-selector').addEventListener('change', function(e) {
            if (e.target.value) {
                selectRun(e.target.value);
            }
        });
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            
            // Set up periodic refresh
            setInterval(refreshData, 30000); // Refresh every 30 seconds
        });
    </script>
</body>
</html>
        """
    
    def run(self):
        """Run the dashboard server."""
        logger.info(f"Starting forge Dashboard on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


# Integration with existing forge codebase
class DashboardIntegration:
    """Integration layer for connecting dashboard with forge training."""
    
    @staticmethod
    def create_run_from_config(
        run_id: str,
        name: str,
        model_name: str,
        config: Dict[str, Any]
    ) -> TrainingRun:
        """Create a training run from forge config."""
        return TrainingRun(
            run_id=run_id,
            name=name,
            model_name=model_name,
            start_time=time.time(),
            config=config
        )
    
    @staticmethod
    def metrics_from_trainer(
        trainer_state: Dict[str, Any],
        resource_usage: Optional[Dict[str, float]] = None
    ) -> TrainingMetrics:
        """Convert trainer state to dashboard metrics."""
        metrics = {}
        
        # Extract standard metrics from trainer state
        if "log_history" in trainer_state and trainer_state["log_history"]:
            latest_log = trainer_state["log_history"][-1]
            
            # Common training metrics
            for key in ["loss", "learning_rate", "epoch", "step"]:
                if key in latest_log:
                    metrics[key] = latest_log[key]
            
            # Evaluation metrics
            for key in ["eval_loss", "eval_accuracy", "eval_f1"]:
                if key in latest_log:
                    metrics[key] = latest_log[key]
        
        # Add resource usage if provided
        if resource_usage is None:
            resource_usage = {}
            
            # Try to get current resource usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    resource_usage["gpu_memory"] = gpu.memoryUtil
                    resource_usage["gpu_util"] = gpu.load
            except:
                pass
            
            resource_usage["cpu_util"] = psutil.cpu_percent() / 100
            resource_usage["memory_util"] = psutil.virtual_memory().percent / 100
        
        return TrainingMetrics(
            timestamp=time.time(),
            step=trainer_state.get("global_step", 0),
            epoch=trainer_state.get("epoch", 0),
            metrics=metrics,
            resource_usage=resource_usage
        )


# Example usage and factory function
def create_dashboard(
    host: str = "0.0.0.0",
    port: int = 8050,
    sampling_rate: int = DEFAULT_SAMPLING_RATE
) -> DashboardVisualizer:
    """Factory function to create and configure the dashboard."""
    dashboard = DashboardVisualizer(host=host, port=port)
    dashboard.metric_collector.sampling_rate = sampling_rate
    return dashboard


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="forge Training Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    parser.add_argument("--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE,
                       help="Resource sampling rate in seconds")
    
    args = parser.parse_args()
    
    dashboard = create_dashboard(
        host=args.host,
        port=args.port,
        sampling_rate=args.sampling_rate
    )
    
    # Example: Add a sample run for demonstration
    sample_run = TrainingRun(
        run_id="demo_run_1",
        name="Demo Training Run",
        model_name="llama-2-7b",
        start_time=time.time(),
        config={
            "learning_rate": 2e-5,
            "batch_size": 4,
            "epochs": 3,
            "warmup_steps": 100
        }
    )
    
    dashboard.metric_collector.add_run(sample_run)
    
    # Start the dashboard
    dashboard.run()