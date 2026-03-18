"""
Real-time Training Dashboard for forge

Web-based dashboard for monitoring multiple training runs with real-time metrics,
resource utilization, and early stopping recommendations.
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict, deque
import threading
import psutil
import GPUtil
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel, Field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLING_RATE = 1.0  # seconds
MAX_HISTORY_POINTS = 1000
ANOMALY_THRESHOLD = 3.0  # standard deviations for anomaly detection
EARLY_STOP_PATIENCE = 10  # epochs without improvement
MIN_IMPROVEMENT = 0.001  # minimum improvement threshold

class TrainingStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class MetricType(str, Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"

@dataclass
class TrainingRun:
    """Represents a single training run"""
    run_id: str
    name: str
    model_name: str
    dataset: str
    start_time: datetime
    status: TrainingStatus = TrainingStatus.RUNNING
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS)))
    resource_metrics: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS)))
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    early_stop_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['last_update'] = self.last_update.isoformat()
        # Convert deques to lists for JSON serialization
        for metric_type in data['metrics']:
            data['metrics'][metric_type] = list(data['metrics'][metric_type])
        for resource_type in data['resource_metrics']:
            data['resource_metrics'][resource_type] = list(data['resource_metrics'][resource_type])
        return data

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    step: int
    epoch: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collects and processes training metrics"""
    
    def __init__(self, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.runs: Dict[str, TrainingRun] = {}
        self.active_connections: Set[WebSocket] = set()
        self.lock = threading.RLock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start metrics collection in background thread"""
        if not self._running:
            self._running = True
            self._collection_thread = threading.Thread(target=self._collect_resources, daemon=True)
            self._collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
            logger.info("Metrics collection stopped")
    
    def create_run(self, name: str, model_name: str, dataset: str, config: Dict[str, Any]) -> str:
        """Create a new training run"""
        run_id = str(uuid.uuid4())
        with self.lock:
            self.runs[run_id] = TrainingRun(
                run_id=run_id,
                name=name,
                model_name=model_name,
                dataset=dataset,
                start_time=datetime.now(),
                config=config
            )
        logger.info(f"Created training run: {run_id} - {name}")
        return run_id
    
    def update_run_status(self, run_id: str, status: TrainingStatus):
        """Update training run status"""
        with self.lock:
            if run_id in self.runs:
                self.runs[run_id].status = status
                self.runs[run_id].last_update = datetime.now()
                self._broadcast_update(run_id)
    
    def record_metric(self, run_id: str, metric_type: str, value: float, 
                     step: int, epoch: Optional[int] = None, metadata: Optional[Dict] = None):
        """Record a metric for a training run"""
        with self.lock:
            if run_id not in self.runs:
                logger.warning(f"Run {run_id} not found")
                return
            
            run = self.runs[run_id]
            timestamp = datetime.now()
            
            # Store metric
            metric_point = {
                'timestamp': timestamp.isoformat(),
                'value': value,
                'step': step,
                'epoch': epoch,
                'metadata': metadata or {}
            }
            
            run.metrics[metric_type].append(metric_point)
            run.step = step
            if epoch is not None:
                run.epoch = epoch
            run.last_update = timestamp
            
            # Check for anomalies
            self._detect_anomalies(run_id, metric_type, value, step)
            
            # Check for early stopping
            self._check_early_stopping(run_id, metric_type, value, step)
            
            # Broadcast update
            self._broadcast_update(run_id)
    
    def record_resource_metrics(self, run_id: str):
        """Record system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            gpu_data = []
            for gpu in gpus:
                gpu_data.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            
            with self.lock:
                if run_id in self.runs:
                    run = self.runs[run_id]
                    timestamp = datetime.now()
                    
                    run.resource_metrics['cpu_usage'].append({
                        'timestamp': timestamp.isoformat(),
                        'value': cpu_percent
                    })
                    
                    run.resource_metrics['memory_usage'].append({
                        'timestamp': timestamp.isoformat(),
                        'value': memory_percent,
                        'used_gb': memory_used_gb
                    })
                    
                    for i, gpu in enumerate(gpu_data):
                        run.resource_metrics[f'gpu_{i}_utilization'].append({
                            'timestamp': timestamp.isoformat(),
                            'value': gpu['load']
                        })
                        run.resource_metrics[f'gpu_{i}_memory'].append({
                            'timestamp': timestamp.isoformat(),
                            'value': (gpu['memory_used'] / gpu['memory_total']) * 100 if gpu['memory_total'] > 0 else 0,
                            'used_mb': gpu['memory_used'],
                            'total_mb': gpu['memory_total']
                        })
                        run.resource_metrics[f'gpu_{i}_temperature'].append({
                            'timestamp': timestamp.isoformat(),
                            'value': gpu['temperature']
                        })
                    
                    self._broadcast_update(run_id)
                    
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
    
    def _detect_anomalies(self, run_id: str, metric_type: str, value: float, step: int):
        """Detect anomalies in metric values"""
        with self.lock:
            run = self.runs[run_id]
            metric_history = [point['value'] for point in run.metrics[metric_type]]
            
            if len(metric_history) < 10:  # Need enough history
                return
            
            # Calculate statistics
            mean = np.mean(metric_history)
            std = np.std(metric_history)
            
            if std == 0:
                return
            
            # Check for anomaly (value outside threshold * std from mean)
            z_score = abs(value - mean) / std
            
            if z_score > ANOMALY_THRESHOLD:
                anomaly = {
                    'timestamp': datetime.now().isoformat(),
                    'metric_type': metric_type,
                    'value': value,
                    'step': step,
                    'z_score': z_score,
                    'mean': mean,
                    'std': std,
                    'severity': 'high' if z_score > ANOMALY_THRESHOLD * 2 else 'medium'
                }
                run.anomalies.append(anomaly)
                logger.warning(f"Anomaly detected in run {run_id}: {metric_type} = {value} (z-score: {z_score:.2f})")
    
    def _check_early_stopping(self, run_id: str, metric_type: str, value: float, step: int):
        """Check if early stopping should be suggested"""
        if metric_type not in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            return
        
        with self.lock:
            run = self.runs[run_id]
            
            # Get metric history
            metric_history = [point['value'] for point in run.metrics[metric_type]]
            
            if len(metric_history) < EARLY_STOP_PATIENCE * 2:
                return
            
            # Check for improvement
            recent_values = metric_history[-EARLY_STOP_PATIENCE:]
            older_values = metric_history[-EARLY_STOP_PATIENCE*2:-EARLY_STOP_PATIENCE]
            
            if not older_values or not recent_values:
                return
            
            recent_avg = np.mean(recent_values)
            older_avg = np.mean(older_values)
            
            # For loss metrics, we want decrease; for accuracy, we want increase
            if 'loss' in metric_type:
                improvement = older_avg - recent_avg
                threshold = MIN_IMPROVEMENT
            else:  # accuracy
                improvement = recent_avg - older_avg
                threshold = MIN_IMPROVEMENT
            
            if improvement < threshold:
                suggestion = {
                    'timestamp': datetime.now().isoformat(),
                    'metric_type': metric_type,
                    'current_value': value,
                    'step': step,
                    'epoch': run.epoch,
                    'improvement': improvement,
                    'threshold': threshold,
                    'patience': EARLY_STOP_PATIENCE,
                    'recommendation': f"Consider early stopping: no significant improvement in {metric_type} for {EARLY_STOP_PATIENCE} steps"
                }
                run.early_stop_suggestions.append(suggestion)
                logger.info(f"Early stopping suggested for run {run_id}: {metric_type}")
    
    def _collect_resources(self):
        """Background thread for collecting resource metrics"""
        while self._running:
            try:
                with self.lock:
                    run_ids = list(self.runs.keys())
                
                for run_id in run_ids:
                    self.record_resource_metrics(run_id)
                
                time.sleep(self.sampling_rate)
            except Exception as e:
                logger.error(f"Error in resource collection thread: {e}")
                time.sleep(5)  # Wait before retrying
    
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        
        # Send current state
        await self._send_initial_state(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def _send_initial_state(self, websocket: WebSocket):
        """Send initial state to newly connected client"""
        with self.lock:
            runs_data = {run_id: run.to_dict() for run_id, run in self.runs.items()}
        
        await websocket.send_json({
            'type': 'initial_state',
            'data': runs_data
        })
    
    def _broadcast_update(self, run_id: str):
        """Broadcast update to all connected clients"""
        if not self.active_connections:
            return
        
        with self.lock:
            if run_id not in self.runs:
                return
            
            run_data = self.runs[run_id].to_dict()
        
        update_message = {
            'type': 'run_update',
            'run_id': run_id,
            'data': run_data
        }
        
        # Broadcast to all connections
        for connection in self.active_connections.copy():
            try:
                asyncio.create_task(connection.send_json(update_message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                self.active_connections.discard(connection)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a training run"""
        with self.lock:
            if run_id not in self.runs:
                raise ValueError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
            
            summary = {
                'run_id': run_id,
                'name': run.name,
                'model_name': run.model_name,
                'status': run.status,
                'duration': (datetime.now() - run.start_time).total_seconds(),
                'epoch': run.epoch,
                'step': run.step,
                'total_steps': run.total_steps,
                'metrics_summary': {},
                'resource_summary': {},
                'anomaly_count': len(run.anomalies),
                'early_stop_suggestions': len(run.early_stop_suggestions)
            }
            
            # Calculate metric summaries
            for metric_type, points in run.metrics.items():
                if points:
                    values = [p['value'] for p in points]
                    summary['metrics_summary'][metric_type] = {
                        'current': values[-1] if values else None,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'trend': self._calculate_trend(values)
                    }
            
            # Calculate resource summaries
            for resource_type, points in run.resource_metrics.items():
                if points:
                    values = [p['value'] for p in points]
                    summary['resource_summary'][resource_type] = {
                        'current': values[-1] if values else None,
                        'mean': np.mean(values),
                        'max': np.max(values)
                    }
            
            return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(x) == 0:
            return "stable"
        
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def generate_visualization(self, run_id: str, metric_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate Plotly visualization data for a training run"""
        with self.lock:
            if run_id not in self.runs:
                raise ValueError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
        
        if metric_types is None:
            metric_types = list(run.metrics.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Metrics', 'Resource Utilization', 
                          'Metric Trends', 'Anomaly Detection'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot training metrics
        for metric_type in metric_types:
            if metric_type in run.metrics:
                points = list(run.metrics[metric_type])
                if points:
                    timestamps = [p['timestamp'] for p in points]
                    values = [p['value'] for p in points]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            name=metric_type,
                            mode='lines+markers',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
        
        # Plot resource utilization
        resource_types = ['cpu_usage', 'memory_usage']
        for i in range(4):  # Support up to 4 GPUs
            resource_types.extend([f'gpu_{i}_utilization', f'gpu_{i}_memory'])
        
        for resource_type in resource_types:
            if resource_type in run.resource_metrics:
                points = list(run.resource_metrics[resource_type])
                if points:
                    timestamps = [p['timestamp'] for p in points]
                    values = [p['value'] for p in points]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            name=resource_type,
                            mode='lines',
                            line=dict(width=1, dash='dot')
                        ),
                        row=1, col=2
                    )
        
        # Plot metric trends (moving average)
        for metric_type in metric_types[:3]:  # Limit to first 3 for clarity
            if metric_type in run.metrics:
                points = list(run.metrics[metric_type])
                if len(points) > 10:
                    values = [p['value'] for p in points]
                    # Calculate moving average
                    window_size = min(10, len(values) // 5)
                    if window_size > 0:
                        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                        timestamps = [points[i]['timestamp'] for i in range(window_size-1, len(points))]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=moving_avg,
                                name=f'{metric_type} (MA)',
                                mode='lines',
                                line=dict(width=3)
                            ),
                            row=2, col=1
                        )
        
        # Plot anomalies
        if run.anomalies:
            anomaly_timestamps = [a['timestamp'] for a in run.anomalies]
            anomaly_values = [a['value'] for a in run.anomalies]
            anomaly_types = [a['metric_type'] for a in run.anomalies]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_timestamps,
                    y=anomaly_values,
                    text=anomaly_types,
                    name='Anomalies',
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='DarkSlateGrey')
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Training Dashboard - {run.name}",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Percentage", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Value (Moving Avg)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        return json.loads(fig.to_json())

# Create FastAPI application
app = FastAPI(
    title="forge Training Dashboard",
    description="Real-time monitoring dashboard for forge training runs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Pydantic models for API
class RunCreate(BaseModel):
    name: str = Field(..., description="Name of the training run")
    model_name: str = Field(..., description="Name of the model being trained")
    dataset: str = Field(..., description="Dataset being used")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

class MetricRecord(BaseModel):
    run_id: str = Field(..., description="ID of the training run")
    metric_type: str = Field(..., description="Type of metric (loss, accuracy, etc.)")
    value: float = Field(..., description="Metric value")
    step: int = Field(..., description="Training step")
    epoch: Optional[int] = Field(None, description="Training epoch")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class RunStatusUpdate(BaseModel):
    run_id: str = Field(..., description="ID of the training run")
    status: TrainingStatus = Field(..., description="New status")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Start metrics collection on application startup"""
    metrics_collector.start()
    logger.info("Dashboard started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop metrics collection on application shutdown"""
    metrics_collector.stop()
    logger.info("Dashboard stopped")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>forge Training Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .run-selector {
                margin-bottom: 20px;
            }
            select {
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #ddd;
                width: 100%;
                max-width: 400px;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-running { background-color: #28a745; }
            .status-paused { background-color: #ffc107; }
            .status-completed { background-color: #007bff; }
            .status-failed { background-color: #dc3545; }
            .anomaly-alert {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
            .early-stop-suggestion {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🦙 forge Training Dashboard</h1>
                <p>Real-time monitoring of training runs with anomaly detection</p>
            </div>
            
            <div class="run-selector">
                <h3>Select Training Run</h3>
                <select id="runSelect" onchange="selectRun(this.value)">
                    <option value="">-- Select a run --</option>
                </select>
            </div>
            
            <div id="runDetails"></div>
            <div id="charts"></div>
            <div id="alerts"></div>
        </div>
        
        <script>
            let currentRunId = null;
            let socket = null;
            let runsData = {};
            
            // Connect to WebSocket
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                socket = new WebSocket(wsUrl);
                
                socket.onopen = function(event) {
                    console.log('WebSocket connected');
                };
                
                socket.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleMessage(message);
                };
                
                socket.onclose = function(event) {
                    console.log('WebSocket disconnected. Reconnecting in 5 seconds...');
                    setTimeout(connectWebSocket, 5000);
                };
                
                socket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            // Handle incoming WebSocket messages
            function handleMessage(message) {
                switch(message.type) {
                    case 'initial_state':
                        runsData = message.data;
                        updateRunSelector();
                        if (currentRunId && runsData[currentRunId]) {
                            updateRunDetails(currentRunId);
                        }
                        break;
                        
                    case 'run_update':
                        runsData[message.run_id] = message.data;
                        if (message.run_id === currentRunId) {
                            updateRunDetails(currentRunId);
                        }
                        updateRunSelector();
                        break;
                }
            }
            
            // Update run selector dropdown
            function updateRunSelector() {
                const selector = document.getElementById('runSelect');
                const currentValue = selector.value;
                
                selector.innerHTML = '<option value="">-- Select a run --</option>';
                
                for (const runId in runsData) {
                    const run = runsData[runId];
                    const option = document.createElement('option');
                    option.value = runId;
                    option.textContent = `${run.name} (${run.model_name}) - ${run.status}`;
                    selector.appendChild(option);
                }
                
                if (currentValue && runsData[currentValue]) {
                    selector.value = currentValue;
                }
            }
            
            // Select a training run
            function selectRun(runId) {
                currentRunId = runId;
                if (runId && runsData[runId]) {
                    updateRunDetails(runId);
                    loadVisualization(runId);
                } else {
                    document.getElementById('runDetails').innerHTML = '';
                    document.getElementById('charts').innerHTML = '';
                    document.getElementById('alerts').innerHTML = '';
                }
            }
            
            // Update run details display
            function updateRunDetails(runId) {
                const run = runsData[runId];
                if (!run) return;
                
                const statusClass = `status-${run.status}`;
                const duration = Math.round((new Date() - new Date(run.start_time)) / 1000);
                
                let html = `
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>Run Information</h3>
                            <p><strong>Name:</strong> ${run.name}</p>
                            <p><strong>Model:</strong> ${run.model_name}</p>
                            <p><strong>Dataset:</strong> ${run.dataset}</p>
                            <p><strong>Status:</strong> <span class="status-indicator ${statusClass}"></span>${run.status}</p>
                            <p><strong>Duration:</strong> ${formatDuration(duration)}</p>
                            <p><strong>Epoch:</strong> ${run.epoch}</p>
                            <p><strong>Step:</strong> ${run.step}</p>
                        </div>
                `;
                
                // Add metric summaries
                if (run.metrics_summary) {
                    for (const metricType in run.metrics_summary) {
                        const summary = run.metrics_summary[metricType];
                        html += `
                            <div class="metric-card">
                                <h3>${metricType}</h3>
                                <p><strong>Current:</strong> ${summary.current?.toFixed(4) || 'N/A'}</p>
                                <p><strong>Mean:</strong> ${summary.mean?.toFixed(4) || 'N/A'}</p>
                                <p><strong>Trend:</strong> ${summary.trend}</p>
                            </div>
                        `;
                    }
                }
                
                html += '</div>';
                
                // Add alerts
                let alertsHtml = '';
                
                if (run.anomalies && run.anomalies.length > 0) {
                    alertsHtml += '<div class="anomaly-alert"><h4>⚠️ Anomalies Detected</h4>';
                    run.anomalies.slice(-3).forEach(anomaly => {
                        alertsHtml += `<p>${anomaly.metric_type}: ${anomaly.value.toFixed(4)} (z-score: ${anomaly.z_score.toFixed(2)})</p>`;
                    });
                    alertsHtml += '</div>';
                }
                
                if (run.early_stop_suggestions && run.early_stop_suggestions.length > 0) {
                    alertsHtml += '<div class="early-stop-suggestion"><h4>🛑 Early Stopping Suggestion</h4>';
                    const latestSuggestion = run.early_stop_suggestions[run.early_stop_suggestions.length - 1];
                    alertsHtml += `<p>${latestSuggestion.recommendation}</p>`;
                    alertsHtml += '</div>';
                }
                
                document.getElementById('runDetails').innerHTML = html;
                document.getElementById('alerts').innerHTML = alertsHtml;
            }
            
            // Load visualization for a run
            async function loadVisualization(runId) {
                try {
                    const response = await fetch(`/api/runs/${runId}/visualization`);
                    const vizData = await response.json();
                    
                    const chartDiv = document.getElementById('charts');
                    chartDiv.innerHTML = '<div id="plotlyChart" style="width:100%;height:800px;"></div>';
                    
                    Plotly.newPlot('plotlyChart', vizData.data, vizData.layout);
                    
                } catch (error) {
                    console.error('Error loading visualization:', error);
                }
            }
            
            // Format duration in human-readable format
            function formatDuration(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = seconds % 60;
                
                if (hours > 0) {
                    return `${hours}h ${minutes}m ${secs}s`;
                } else if (minutes > 0) {
                    return `${minutes}m ${secs}s`;
                } else {
                    return `${secs}s`;
                }
            }
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', function() {
                connectWebSocket();
                
                // Auto-refresh every 30 seconds
                setInterval(() => {
                    if (currentRunId) {
                        loadVisualization(currentRunId);
                    }
                }, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await metrics_collector.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        metrics_collector.disconnect(websocket)

@app.post("/api/runs", response_model=Dict[str, str])
async def create_run(run_data: RunCreate):
    """Create a new training run"""
    run_id = metrics_collector.create_run(
        name=run_data.name,
        model_name=run_data.model_name,
        dataset=run_data.dataset,
        config=run_data.config
    )
    return {"run_id": run_id, "status": "created"}

@app.post("/api/metrics")
async def record_metric(metric: MetricRecord):
    """Record a metric for a training run"""
    metrics_collector.record_metric(
        run_id=metric.run_id,
        metric_type=metric.metric_type,
        value=metric.value,
        step=metric.step,
        epoch=metric.epoch,
        metadata=metric.metadata
    )
    return {"status": "recorded"}

@app.put("/api/runs/status")
async def update_run_status(status_update: RunStatusUpdate):
    """Update training run status"""
    metrics_collector.update_run_status(
        run_id=status_update.run_id,
        status=status_update.status
    )
    return {"status": "updated"}

@app.get("/api/runs")
async def list_runs():
    """List all training runs"""
    with metrics_collector.lock:
        runs = {run_id: run.to_dict() for run_id, run in metrics_collector.runs.items()}
    return runs

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get details of a specific training run"""
    with metrics_collector.lock:
        if run_id not in metrics_collector.runs:
            raise HTTPException(status_code=404, detail="Run not found")
        return metrics_collector.runs[run_id].to_dict()

@app.get("/api/runs/{run_id}/summary")
async def get_run_summary(run_id: str):
    """Get summary statistics for a training run"""
    try:
        return metrics_collector.get_run_summary(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/runs/{run_id}/visualization")
async def get_run_visualization(run_id: str, metrics: Optional[str] = None):
    """Get Plotly visualization data for a training run"""
    try:
        metric_types = metrics.split(",") if metrics else None
        return metrics_collector.generate_visualization(run_id, metric_types)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/runs/{run_id}/anomalies")
async def get_run_anomalies(run_id: str):
    """Get anomalies detected for a training run"""
    with metrics_collector.lock:
        if run_id not in metrics_collector.runs:
            raise HTTPException(status_code=404, detail="Run not found")
        return metrics_collector.runs[run_id].anomalies

@app.get("/api/runs/{run_id}/early-stop-suggestions")
async def get_early_stop_suggestions(run_id: str):
    """Get early stopping suggestions for a training run"""
    with metrics_collector.lock:
        if run_id not in metrics_collector.runs:
            raise HTTPException(status_code=404, detail="Run not found")
        return metrics_collector.runs[run_id].early_stop_suggestions

@app.get("/api/system/resources")
async def get_system_resources():
    """Get current system resource utilization"""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        gpus = GPUtil.getGPUs()
        
        gpu_data = []
        for gpu in gpus:
            gpu_data.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024 ** 3),
            'memory_total_gb': memory.total / (1024 ** 3),
            'gpus': gpu_data,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system resources: {str(e)}")

# Integration with existing forge training code
class forgeMetricsCallback:
    """Callback for integrating metrics collection with forge training"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8000", run_name: str = "default"):
        self.dashboard_url = dashboard_url
        self.run_name = run_name
        self.run_id = None
        self.step = 0
        self.epoch = 0
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training"""
        import requests
        
        # Create a new run
        response = requests.post(
            f"{self.dashboard_url}/api/runs",
            json={
                "name": self.run_name,
                "model_name": logs.get("model_name", "unknown") if logs else "unknown",
                "dataset": logs.get("dataset", "unknown") if logs else "unknown",
                "config": logs or {}
            }
        )
        
        if response.status_code == 200:
            self.run_id = response.json()["run_id"]
            logger.info(f"Created dashboard run: {self.run_id}")
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training"""
        if self.run_id:
            import requests
            requests.put(
                f"{self.dashboard_url}/api/runs/status",
                json={
                    "run_id": self.run_id,
                    "status": "completed"
                }
            )
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch"""
        self.epoch = epoch
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch"""
        if logs and self.run_id:
            # Record epoch-level metrics
            for metric_name, value in logs.items():
                if isinstance(value, (int, float)):
                    self.record_metric(metric_name, value, self.step, epoch)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch"""
        self.step += 1
        
        if logs and self.run_id and self.step % 10 == 0:  # Record every 10 steps
            # Record batch-level metrics
            for metric_name, value in logs.items():
                if isinstance(value, (int, float)):
                    self.record_metric(metric_name, value, self.step, self.epoch)
    
    def record_metric(self, metric_type: str, value: float, step: int, epoch: Optional[int] = None):
        """Record a metric to the dashboard"""
        if not self.run_id:
            return
        
        import requests
        try:
            requests.post(
                f"{self.dashboard_url}/api/metrics",
                json={
                    "run_id": self.run_id,
                    "metric_type": metric_type,
                    "value": float(value),
                    "step": step,
                    "epoch": epoch
                },
                timeout=1.0  # Short timeout to avoid blocking training
            )
        except Exception as e:
            logger.warning(f"Failed to record metric: {e}")

# Example usage with forge training
def create_dashboard_callback(config: Dict[str, Any]) -> forgeMetricsCallback:
    """Factory function to create dashboard callback from config"""
    dashboard_config = config.get("dashboard", {})
    return forgeMetricsCallback(
        dashboard_url=dashboard_config.get("url", "http://localhost:8000"),
        run_name=dashboard_config.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )

# Run the dashboard server
def run_dashboard(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Run the dashboard server"""
    uvicorn.run(app, host=host, port=port, **kwargs)

if __name__ == "__main__":
    # Run the dashboard
    run_dashboard()