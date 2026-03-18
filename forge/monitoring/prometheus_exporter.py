"""
Prometheus metrics exporter for forge training monitoring.

This module provides real-time training metrics export for Prometheus, enabling
monitoring dashboards and alerting for training anomalies.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetector:
    """Simple statistical anomaly detector for training metrics."""
    
    window_size: int = 100
    threshold_std: float = 3.0
    min_samples: int = 10
    
    def __post_init__(self):
        self._values = deque(maxlen=self.window_size)
        self._timestamps = deque(maxlen=self.window_size)
    
    def add_value(self, value: float, timestamp: Optional[float] = None) -> bool:
        """Add a value and check for anomalies."""
        if timestamp is None:
            timestamp = time.time()
        
        self._values.append(value)
        self._timestamps.append(timestamp)
        
        if len(self._values) < self.min_samples:
            return False
        
        mean = statistics.mean(self._values)
        std = statistics.stdev(self._values) if len(self._values) > 1 else 0.0
        
        if std == 0:
            return False
        
        z_score = abs(value - mean) / std
        return z_score > self.threshold_std
    
    def reset(self):
        """Reset the detector state."""
        self._values.clear()
        self._timestamps.clear()


@dataclass
class AlertConfig:
    """Configuration for training alerts."""
    
    loss_spike_threshold: float = 2.0  # Multiplicative increase
    memory_leak_threshold_mb: float = 100.0  # MB increase per hour
    gradient_explosion_threshold: float = 10.0
    learning_rate_min: float = 1e-8
    training_stall_minutes: int = 30


class PrometheusExporter:
    """
    Prometheus metrics exporter for forge training.
    
    Exposes training metrics via HTTP endpoint for Prometheus scraping.
    Includes anomaly detection and alerting capabilities.
    """
    
    def __init__(
        self,
        port: int = 9090,
        addr: str = "0.0.0.0",
        prefix: str = "forge_",
        alert_config: Optional[AlertConfig] = None,
        anomaly_window: int = 100,
        anomaly_threshold: float = 3.0
    ):
        self.port = port
        self.addr = addr
        self.prefix = prefix
        self.alert_config = alert_config or AlertConfig()
        self._server_started = False
        self._lock = threading.RLock()
        
        # Training state
        self._training_start_time = None
        self._last_step_time = None
        self._current_step = 0
        self._current_epoch = 0
        
        # Anomaly detectors
        self._loss_detector = AnomalyDetector(
            window_size=anomaly_window,
            threshold_std=anomaly_threshold
        )
        self._memory_detector = AnomalyDetector(
            window_size=anomaly_window,
            threshold_std=anomaly_threshold
        )
        self._gradient_detector = AnomalyDetector(
            window_size=anomaly_window,
            threshold_std=anomaly_threshold
        )
        
        # Metrics storage for custom collection
        self._custom_metrics = {}
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # Training progress metrics
        self.training_loss = Gauge(
            f"{self.prefix}training_loss",
            "Current training loss",
            ["run_id", "model_name"]
        )
        
        self.validation_loss = Gauge(
            f"{self.prefix}validation_loss",
            "Current validation loss",
            ["run_id", "model_name"]
        )
        
        self.learning_rate = Gauge(
            f"{self.prefix}learning_rate",
            "Current learning rate",
            ["run_id", "optimizer"]
        )
        
        self.epoch = Gauge(
            f"{self.prefix}epoch",
            "Current training epoch",
            ["run_id"]
        )
        
        self.step = Counter(
            f"{self.prefix}steps_total",
            "Total training steps",
            ["run_id"]
        )
        
        self.global_step = Gauge(
            f"{self.prefix}global_step",
            "Current global step",
            ["run_id"]
        )
        
        # Performance metrics
        self.samples_per_second = Gauge(
            f"{self.prefix}samples_per_second",
            "Training samples processed per second",
            ["run_id"]
        )
        
        self.tokens_per_second = Gauge(
            f"{self.prefix}tokens_per_second",
            "Tokens processed per second",
            ["run_id"]
        )
        
        self.step_duration = Histogram(
            f"{self.prefix}step_duration_seconds",
            "Duration of training steps",
            ["run_id"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Resource metrics
        self.gpu_memory_used = Gauge(
            f"{self.prefix}gpu_memory_used_bytes",
            "GPU memory used",
            ["run_id", "device"]
        )
        
        self.gpu_memory_total = Gauge(
            f"{self.prefix}gpu_memory_total_bytes",
            "Total GPU memory",
            ["run_id", "device"]
        )
        
        self.gpu_utilization = Gauge(
            f"{self.prefix}gpu_utilization_percent",
            "GPU utilization percentage",
            ["run_id", "device"]
        )
        
        self.cpu_memory_used = Gauge(
            f"{self.prefix}cpu_memory_used_bytes",
            "CPU memory used",
            ["run_id"]
        )
        
        # Gradient metrics
        self.gradient_norm = Gauge(
            f"{self.prefix}gradient_norm",
            "Gradient norm",
            ["run_id"]
        )
        
        self.gradient_norm_std = Gauge(
            f"{self.prefix}gradient_norm_std",
            "Standard deviation of gradient norms",
            ["run_id"]
        )
        
        # Alert metrics
        self.anomaly_detected = Gauge(
            f"{self.prefix}anomaly_detected",
            "Whether an anomaly was detected (1=yes, 0=no)",
            ["run_id", "metric_type"]
        )
        
        self.alert_active = Gauge(
            f"{self.prefix}alert_active",
            "Whether an alert is active (1=yes, 0=no)",
            ["run_id", "alert_type"]
        )
        
        # Custom metrics collector
        self.custom_metrics_collector = CustomMetricsCollector(self)
        REGISTRY.register(self.custom_metrics_collector)
    
    def start(self):
        """Start the Prometheus HTTP server."""
        if self._server_started:
            logger.warning("Prometheus exporter already started")
            return
        
        try:
            start_http_server(self.port, addr=self.addr)
            self._server_started = True
            logger.info(f"Prometheus exporter started on {self.addr}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus exporter: {e}")
            raise
    
    def stop(self):
        """Stop the Prometheus exporter."""
        # Note: prometheus_client doesn't provide a clean way to stop the server
        # This is a limitation of the library
        logger.info("Prometheus exporter stop requested (server continues running)")
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Trigger an alert and notify callbacks."""
        self.alert_active.labels(
            run_id=self._get_run_id(),
            alert_type=alert_type
        ).set(1)
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "run_id": self._get_run_id(),
            "step": self._current_step
        }
        
        for callback in self._alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
    
    def _clear_alert(self, alert_type: str):
        """Clear an alert."""
        self.alert_active.labels(
            run_id=self._get_run_id(),
            alert_type=alert_type
        ).set(0)
    
    def _get_run_id(self) -> str:
        """Get the current run ID (placeholder for actual implementation)."""
        # In a real implementation, this would come from the training config
        return "default"
    
    def update_training_metrics(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        run_id: Optional[str] = None,
        model_name: str = "unknown",
        optimizer: str = "adamw",
        samples_processed: Optional[int] = None,
        tokens_processed: Optional[int] = None,
        step_duration: Optional[float] = None
    ):
        """Update training progress metrics."""
        with self._lock:
            self._current_step = step
            self._current_epoch = epoch
            
            if run_id is None:
                run_id = self._get_run_id()
            
            # Update basic metrics
            self.training_loss.labels(run_id=run_id, model_name=model_name).set(loss)
            self.learning_rate.labels(run_id=run_id, optimizer=optimizer).set(learning_rate)
            self.epoch.labels(run_id=run_id).set(epoch)
            self.global_step.labels(run_id=run_id).set(step)
            
            # Update step counter
            self.step.labels(run_id=run_id).inc()
            
            # Update performance metrics
            if samples_processed is not None and step_duration is not None and step_duration > 0:
                sps = samples_processed / step_duration
                self.samples_per_second.labels(run_id=run_id).set(sps)
            
            if tokens_processed is not None and step_duration is not None and step_duration > 0:
                tps = tokens_processed / step_duration
                self.tokens_per_second.labels(run_id=run_id).set(tps)
            
            if step_duration is not None:
                self.step_duration.labels(run_id=run_id).observe(step_duration)
            
            # Check for anomalies
            self._check_loss_anomaly(loss, run_id)
            
            # Check for training stall
            self._check_training_stall()
            
            # Update last step time
            self._last_step_time = time.time()
    
    def update_validation_metrics(
        self,
        loss: float,
        run_id: Optional[str] = None,
        model_name: str = "unknown"
    ):
        """Update validation metrics."""
        with self._lock:
            if run_id is None:
                run_id = self._get_run_id()
            
            self.validation_loss.labels(run_id=run_id, model_name=model_name).set(loss)
    
    def update_resource_metrics(
        self,
        gpu_memory_used: Optional[Dict[str, float]] = None,
        gpu_memory_total: Optional[Dict[str, float]] = None,
        gpu_utilization: Optional[Dict[str, float]] = None,
        cpu_memory_used: Optional[float] = None,
        run_id: Optional[str] = None
    ):
        """Update resource utilization metrics."""
        with self._lock:
            if run_id is None:
                run_id = self._get_run_id()
            
            if gpu_memory_used:
                for device, memory in gpu_memory_used.items():
                    self.gpu_memory_used.labels(run_id=run_id, device=device).set(memory)
            
            if gpu_memory_total:
                for device, memory in gpu_memory_total.items():
                    self.gpu_memory_total.labels(run_id=run_id, device=device).set(memory)
            
            if gpu_utilization:
                for device, util in gpu_utilization.items():
                    self.gpu_utilization.labels(run_id=run_id, device=device).set(util)
            
            if cpu_memory_used is not None:
                self.cpu_memory_used.labels(run_id=run_id).set(cpu_memory_used)
            
            # Check for memory anomalies
            if gpu_memory_used:
                total_used = sum(gpu_memory_used.values())
                self._check_memory_anomaly(total_used, run_id)
    
    def update_gradient_metrics(
        self,
        gradient_norm: float,
        gradient_norm_std: Optional[float] = None,
        run_id: Optional[str] = None
    ):
        """Update gradient-related metrics."""
        with self._lock:
            if run_id is None:
                run_id = self._get_run_id()
            
            self.gradient_norm.labels(run_id=run_id).set(gradient_norm)
            
            if gradient_norm_std is not None:
                self.gradient_norm_std.labels(run_id=run_id).set(gradient_norm_std)
            
            # Check for gradient explosion
            self._check_gradient_anomaly(gradient_norm, run_id)
    
    def add_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
        documentation: str = ""
    ):
        """Add a custom metric to be exported."""
        with self._lock:
            if labels is None:
                labels = {}
            
            metric_key = (name, tuple(sorted(labels.items())))
            
            if metric_key not in self._custom_metrics:
                self._custom_metrics[metric_key] = {
                    "name": name,
                    "type": metric_type,
                    "labels": labels,
                    "documentation": documentation,
                    "value": value,
                    "timestamp": time.time()
                }
            else:
                self._custom_metrics[metric_key]["value"] = value
                self._custom_metrics[metric_key]["timestamp"] = time.time()
    
    def _check_loss_anomaly(self, loss: float, run_id: str):
        """Check for loss spikes or drops."""
        is_anomaly = self._loss_detector.add_value(loss)
        
        self.anomaly_detected.labels(
            run_id=run_id,
            metric_type="loss"
        ).set(1 if is_anomaly else 0)
        
        if is_anomaly:
            self._trigger_alert(
                "loss_spike",
                f"Loss anomaly detected: {loss:.4f}",
                severity="warning"
            )
    
    def _check_memory_anomaly(self, memory_used: float, run_id: str):
        """Check for memory leaks."""
        # Convert to MB for easier threshold comparison
        memory_mb = memory_used / (1024 * 1024)
        
        is_anomaly = self._memory_detector.add_value(memory_mb)
        
        self.anomaly_detected.labels(
            run_id=run_id,
            metric_type="memory"
        ).set(1 if is_anomaly else 0)
        
        if is_anomaly:
            self._trigger_alert(
                "memory_leak",
                f"Memory anomaly detected: {memory_mb:.2f} MB",
                severity="critical"
            )
    
    def _check_gradient_anomaly(self, gradient_norm: float, run_id: str):
        """Check for gradient explosion."""
        is_anomaly = self._gradient_detector.add_value(gradient_norm)
        
        self.anomaly_detected.labels(
            run_id=run_id,
            metric_type="gradient"
        ).set(1 if is_anomaly else 0)
        
        if is_anomaly:
            self._trigger_alert(
                "gradient_explosion",
                f"Gradient explosion detected: norm={gradient_norm:.4f}",
                severity="critical"
            )
    
    def _check_training_stall(self):
        """Check if training has stalled."""
        if self._last_step_time is None:
            return
        
        stall_seconds = self.alert_config.training_stall_minutes * 60
        time_since_last_step = time.time() - self._last_step_time
        
        if time_since_last_step > stall_seconds:
            self._trigger_alert(
                "training_stall",
                f"Training stalled for {time_since_last_step/60:.1f} minutes",
                severity="critical"
            )
    
    def reset_anomaly_detectors(self):
        """Reset all anomaly detectors."""
        with self._lock:
            self._loss_detector.reset()
            self._memory_detector.reset()
            self._gradient_detector.reset()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        with self._lock:
            return {
                "step": self._current_step,
                "epoch": self._current_epoch,
                "training_start_time": self._training_start_time,
                "last_step_time": self._last_step_time,
                "custom_metrics_count": len(self._custom_metrics),
                "server_started": self._server_started,
                "port": self.port
            }


class CustomMetricsCollector:
    """Custom Prometheus collector for dynamic metrics."""
    
    def __init__(self, exporter: PrometheusExporter):
        self.exporter = exporter
    
    def collect(self):
        """Collect custom metrics for Prometheus."""
        with self.exporter._lock:
            for metric_key, metric_data in self.exporter._custom_metrics.items():
                name = metric_data["name"]
                metric_type = metric_data["type"]
                labels = metric_data["labels"]
                value = metric_data["value"]
                documentation = metric_data["documentation"]
                
                # Create label names and values
                label_names = list(labels.keys())
                label_values = list(labels.values())
                
                if metric_type == "gauge":
                    metric = GaugeMetricFamily(
                        name,
                        documentation,
                        labels=label_names
                    )
                    metric.add_metric(label_values, value)
                    yield metric
                
                elif metric_type == "counter":
                    metric = CounterMetricFamily(
                        name,
                        documentation,
                        labels=label_names
                    )
                    metric.add_metric(label_values, value)
                    yield metric
                
                elif metric_type == "histogram":
                    # For histograms, we'd need buckets - simplified here
                    metric = GaugeMetricFamily(
                        name,
                        documentation,
                        labels=label_names
                    )
                    metric.add_metric(label_values, value)
                    yield metric


def create_grafana_dashboard(
    run_id: str = "default",
    model_name: str = "llama",
    refresh_interval: str = "10s"
) -> Dict[str, Any]:
    """
    Create a Grafana dashboard configuration for forge monitoring.
    
    Returns a dictionary that can be saved as JSON and imported into Grafana.
    """
    dashboard = {
        "dashboard": {
            "title": f"forge Training - {model_name}",
            "tags": ["forge", "training", "monitoring"],
            "timezone": "browser",
            "panels": [
                {
                    "title": "Training Loss",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": f'forge_training_loss{{run_id="{run_id}"}}',
                            "legendFormat": "Training Loss"
                        }
                    ],
                    "yaxes": [
                        {"label": "Loss", "min": 0},
                        {"show": False}
                    ]
                },
                {
                    "title": "Validation Loss",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": f'forge_validation_loss{{run_id="{run_id}"}}',
                            "legendFormat": "Validation Loss"
                        }
                    ]
                },
                {
                    "title": "Learning Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": f'forge_learning_rate{{run_id="{run_id}"}}',
                            "legendFormat": "Learning Rate"
                        }
                    ]
                },
                {
                    "title": "GPU Memory Usage",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": f'forge_gpu_memory_used_bytes{{run_id="{run_id}"}} / 1024 / 1024',
                            "legendFormat": "GPU {{device}}"
                        }
                    ],
                    "yaxes": [
                        {"label": "MB", "min": 0},
                        {"show": False}
                    ]
                },
                {
                    "title": "Gradient Norm",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": f'forge_gradient_norm{{run_id="{run_id}"}}',
                            "legendFormat": "Gradient Norm"
                        }
                    ]
                },
                {
                    "title": "Training Speed",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": f'forge_samples_per_second{{run_id="{run_id}"}}',
                            "legendFormat": "Samples/sec"
                        },
                        {
                            "expr": f'forge_tokens_per_second{{run_id="{run_id}"}}',
                            "legendFormat": "Tokens/sec"
                        }
                    ]
                },
                {
                    "title": "Alerts",
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 24, "x": 0, "y": 24},
                    "targets": [
                        {
                            "expr": f'forge_alert_active{{run_id="{run_id}"}}',
                            "legendFormat": "{{alert_type}}"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "red", "value": 1}
                                ]
                            },
                            "mappings": [
                                {
                                    "type": "value",
                                    "options": {
                                        "0": {"text": "OK", "color": "green"},
                                        "1": {"text": "ALERT", "color": "red"}
                                    }
                                }
                            ]
                        }
                    }
                }
            ],
            "refresh": refresh_interval,
            "schemaVersion": 30,
            "version": 1
        },
        "overwrite": True
    }
    
    return dashboard


# Global exporter instance for easy access
_global_exporter: Optional[PrometheusExporter] = None


def get_exporter() -> Optional[PrometheusExporter]:
    """Get the global Prometheus exporter instance."""
    return _global_exporter


def init_exporter(
    port: int = 9090,
    addr: str = "0.0.0.0",
    prefix: str = "forge_",
    **kwargs
) -> PrometheusExporter:
    """
    Initialize and start the global Prometheus exporter.
    
    Args:
        port: Port to expose metrics on
        addr: Address to bind to
        prefix: Metric name prefix
        **kwargs: Additional arguments for PrometheusExporter
    
    Returns:
        Initialized PrometheusExporter instance
    """
    global _global_exporter
    
    if _global_exporter is not None:
        logger.warning("Exporter already initialized, returning existing instance")
        return _global_exporter
    
    _global_exporter = PrometheusExporter(
        port=port,
        addr=addr,
        prefix=prefix,
        **kwargs
    )
    
    _global_exporter.start()
    return _global_exporter


def shutdown_exporter():
    """Shutdown the global Prometheus exporter."""
    global _global_exporter
    
    if _global_exporter is not None:
        _global_exporter.stop()
        _global_exporter = None