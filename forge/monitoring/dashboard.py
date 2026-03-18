"""Real-Time Training Dashboard & Alerting for forge

Provides real-time monitoring with Prometheus metrics, Grafana dashboard templates,
and anomaly detection for training issues like loss spikes and memory leaks.
Integrates with popular observability platforms (Prometheus, Grafana, Slack, etc.).
"""

import time
import threading
import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
import statistics

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Configuration for anomaly detection thresholds."""
    loss_spike_factor: float = 2.0  # Alert if loss increases by this factor
    loss_spike_window: int = 10  # Window size for loss spike detection
    memory_increase_mb: float = 500.0  # Alert if memory increases by this amount
    memory_leak_window: int = 50  # Window size for memory leak detection
    gradient_norm_threshold: float = 10.0  # Alert if gradient norm exceeds this
    learning_rate_min: float = 1e-8  # Alert if LR drops below this
    custom_metric_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for alerting system."""
    enabled: bool = True
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_sender: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    cooldown_seconds: int = 300  # Minimum time between alerts for same issue
    thresholds: AlertThreshold = field(default_factory=AlertThreshold)


class TrainingMetrics:
    """Container for training metrics with history tracking."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.loss_history = deque(maxlen=max_history)
        self.memory_allocated_history = deque(maxlen=max_history)
        self.memory_reserved_history = deque(maxlen=max_history)
        self.gradient_norm_history = deque(maxlen=max_history)
        self.learning_rate_history = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = {}
        self.timestamps = deque(maxlen=max_history)
        self.steps = deque(maxlen=max_history)
        
    def update(self, step: int, **metrics):
        """Update metrics with new values."""
        self.steps.append(step)
        self.timestamps.append(time.time())
        
        for key, value in metrics.items():
            if key == "loss":
                self.loss_history.append(value)
            elif key == "memory_allocated":
                self.memory_allocated_history.append(value)
            elif key == "memory_reserved":
                self.memory_reserved_history.append(value)
            elif key == "gradient_norm":
                self.gradient_norm_history.append(value)
            elif key == "learning_rate":
                self.learning_rate_history.append(value)
            else:
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = deque(maxlen=self.max_history)
                self.custom_metrics[key].append(value)
    
    def get_recent(self, metric_name: str, n: int = 10) -> List[float]:
        """Get recent values for a metric."""
        if metric_name == "loss":
            return list(self.loss_history)[-n:]
        elif metric_name == "memory_allocated":
            return list(self.memory_allocated_history)[-n:]
        elif metric_name == "memory_reserved":
            return list(self.memory_reserved_history)[-n:]
        elif metric_name == "gradient_norm":
            return list(self.gradient_norm_history)[-n:]
        elif metric_name == "learning_rate":
            return list(self.learning_rate_history)[-n:]
        elif metric_name in self.custom_metrics:
            return list(self.custom_metrics[metric_name])[-n:]
        return []


class AnomalyDetector:
    """Detects anomalies in training metrics using statistical methods."""
    
    def __init__(self, config: AlertThreshold):
        self.config = config
        self.alert_cooldowns: Dict[str, float] = {}
        
    def check_loss_spike(self, loss_history: List[float]) -> Optional[Dict[str, Any]]:
        """Detect sudden spikes in loss."""
        if len(loss_history) < self.config.loss_spike_window:
            return None
            
        recent_losses = loss_history[-self.config.loss_spike_window:]
        if len(recent_losses) < 2:
            return None
            
        current_loss = recent_losses[-1]
        avg_loss = statistics.mean(recent_losses[:-1])
        
        if avg_loss > 0 and current_loss > avg_loss * self.config.loss_spike_factor:
            return {
                "type": "loss_spike",
                "severity": "high",
                "message": f"Loss spike detected: {current_loss:.4f} vs avg {avg_loss:.4f}",
                "current_value": current_loss,
                "average_value": avg_loss,
                "factor": current_loss / avg_loss
            }
        return None
    
    def check_memory_leak(self, memory_history: List[float]) -> Optional[Dict[str, Any]]:
        """Detect memory leaks based on increasing memory usage."""
        if len(memory_history) < self.config.memory_leak_window:
            return None
            
        recent_memory = memory_history[-self.config.memory_leak_window:]
        if len(recent_memory) < 2:
            return None
            
        # Check for consistent increase
        increases = 0
        for i in range(1, len(recent_memory)):
            if recent_memory[i] > recent_memory[i-1]:
                increases += 1
        
        increase_ratio = increases / (len(recent_memory) - 1)
        memory_increase = recent_memory[-1] - recent_memory[0]
        
        if (increase_ratio > 0.7 and  # 70% of steps show increase
            memory_increase > self.config.memory_increase_mb):
            return {
                "type": "memory_leak",
                "severity": "medium",
                "message": f"Potential memory leak: increased by {memory_increase:.1f}MB",
                "increase_mb": memory_increase,
                "increase_ratio": increase_ratio
            }
        return None
    
    def check_gradient_explosion(self, gradient_norms: List[float]) -> Optional[Dict[str, Any]]:
        """Detect gradient explosion."""
        if not gradient_norms:
            return None
            
        current_norm = gradient_norms[-1]
        if current_norm > self.config.gradient_norm_threshold:
            return {
                "type": "gradient_explosion",
                "severity": "high",
                "message": f"Gradient explosion detected: norm = {current_norm:.2f}",
                "gradient_norm": current_norm,
                "threshold": self.config.gradient_norm_threshold
            }
        return None
    
    def check_learning_rate_collapse(self, learning_rates: List[float]) -> Optional[Dict[str, Any]]:
        """Detect learning rate collapse."""
        if not learning_rates:
            return None
            
        current_lr = learning_rates[-1]
        if current_lr < self.config.learning_rate_min:
            return {
                "type": "learning_rate_collapse",
                "severity": "medium",
                "message": f"Learning rate collapsed: {current_lr:.2e}",
                "learning_rate": current_lr,
                "threshold": self.config.learning_rate_min
            }
        return None
    
    def check_custom_metric(self, metric_name: str, values: List[float], 
                           threshold: float) -> Optional[Dict[str, Any]]:
        """Check custom metric against threshold."""
        if not values:
            return None
            
        current_value = values[-1]
        if current_value > threshold:
            return {
                "type": "custom_metric_threshold",
                "severity": "medium",
                "message": f"Custom metric {metric_name} exceeded threshold: {current_value:.2f} > {threshold:.2f}",
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold
            }
        return None
    
    def detect_anomalies(self, metrics: TrainingMetrics) -> List[Dict[str, Any]]:
        """Run all anomaly detection checks."""
        anomalies = []
        
        # Check loss spike
        anomaly = self.check_loss_spike(list(metrics.loss_history))
        if anomaly:
            anomalies.append(anomaly)
        
        # Check memory leak
        anomaly = self.check_memory_leak(list(metrics.memory_allocated_history))
        if anomaly:
            anomalies.append(anomaly)
        
        # Check gradient explosion
        anomaly = self.check_gradient_explosion(list(metrics.gradient_norm_history))
        if anomaly:
            anomalies.append(anomaly)
        
        # Check learning rate collapse
        anomaly = self.check_learning_rate_collapse(list(metrics.learning_rate_history))
        if anomaly:
            anomalies.append(anomaly)
        
        # Check custom metrics
        for metric_name, threshold in self.config.custom_metric_thresholds.items():
            if metric_name in metrics.custom_metrics:
                values = list(metrics.custom_metrics[metric_name])
                anomaly = self.check_custom_metric(metric_name, values, threshold)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies


class AlertManager:
    """Manages alert notifications across different platforms."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.last_alert_times: Dict[str, float] = {}
        
    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if not self.config.enabled:
            return False
            
        current_time = time.time()
        last_time = self.last_alert_times.get(alert_type, 0)
        
        if current_time - last_time < self.config.cooldown_seconds:
            return False
            
        self.last_alert_times[alert_type] = current_time
        return True
    
    def _send_slack_alert(self, anomaly: Dict[str, Any]) -> bool:
        """Send alert to Slack."""
        if not self.config.slack_webhook_url or not REQUESTS_AVAILABLE:
            return False
            
        try:
            payload = {
                "text": f"🚨 Training Alert: {anomaly['type'].upper()}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🚨 Training Alert: {anomaly['type'].upper()}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {anomaly['severity']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Type:* {anomaly['type']}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Message:* {anomaly['message']}"
                        }
                    }
                ]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_discord_alert(self, anomaly: Dict[str, Any]) -> bool:
        """Send alert to Discord."""
        if not self.config.discord_webhook_url or not REQUESTS_AVAILABLE:
            return False
            
        try:
            payload = {
                "embeds": [{
                    "title": f"🚨 Training Alert: {anomaly['type'].upper()}",
                    "description": anomaly['message'],
                    "color": 15158332,  # Red color
                    "fields": [
                        {
                            "name": "Severity",
                            "value": anomaly['severity'],
                            "inline": True
                        },
                        {
                            "name": "Type",
                            "value": anomaly['type'],
                            "inline": True
                        }
                    ],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }]
            }
            
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=5
            )
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def send_alert(self, anomaly: Dict[str, Any]) -> bool:
        """Send alert through configured channels."""
        if not self._can_send_alert(anomaly['type']):
            return False
            
        success = False
        
        # Log the alert
        logger.warning(f"Training alert: {anomaly['message']}")
        
        # Send to Slack
        if self.config.slack_webhook_url:
            if self._send_slack_alert(anomaly):
                success = True
        
        # Send to Discord
        if self.config.discord_webhook_url:
            if self._send_discord_alert(anomaly):
                success = True
        
        # TODO: Implement email alerts
        
        return success


class PrometheusExporter:
    """Exports training metrics to Prometheus."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.metrics = {}
        self._server_started = False
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available. Prometheus export disabled.")
            return
            
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Training metrics
        self.metrics['training_loss'] = Gauge(
            'forge_training_loss', 
            'Current training loss'
        )
        self.metrics['training_loss_avg'] = Gauge(
            'forge_training_loss_avg',
            'Average training loss over last 100 steps'
        )
        self.metrics['learning_rate'] = Gauge(
            'forge_learning_rate',
            'Current learning rate'
        )
        self.metrics['gradient_norm'] = Gauge(
            'forge_gradient_norm',
            'Current gradient norm'
        )
        
        # Memory metrics
        self.metrics['memory_allocated_gb'] = Gauge(
            'forge_memory_allocated_gb',
            'GPU memory allocated in GB'
        )
        self.metrics['memory_reserved_gb'] = Gauge(
            'forge_memory_reserved_gb',
            'GPU memory reserved in GB'
        )
        self.metrics['memory_cached_gb'] = Gauge(
            'forge_memory_cached_gb',
            'GPU memory cached in GB'
        )
        
        # Training progress
        self.metrics['epoch'] = Gauge(
            'forge_epoch',
            'Current training epoch'
        )
        self.metrics['global_step'] = Gauge(
            'forge_global_step',
            'Global training step'
        )
        self.metrics['samples_per_second'] = Gauge(
            'forge_samples_per_second',
            'Training samples processed per second'
        )
        
        # Alert metrics
        self.metrics['alerts_total'] = Counter(
            'forge_alerts_total',
            'Total number of training alerts',
            ['alert_type', 'severity']
        )
        
    def start_server(self):
        """Start Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE or self._server_started:
            return
            
        try:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def update_metrics(self, metrics_dict: Dict[str, float]):
        """Update Prometheus metrics with new values."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        for metric_name, value in metrics_dict.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].set(value)
    
    def increment_alert_counter(self, alert_type: str, severity: str):
        """Increment alert counter metric."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.metrics['alerts_total'].labels(
            alert_type=alert_type,
            severity=severity
        ).inc()


class GrafanaDashboard:
    """Generates Grafana dashboard configurations."""
    
    @staticmethod
    def generate_dashboard_json(
        title: str = "forge Training Dashboard",
        datasource: str = "Prometheus"
    ) -> str:
        """Generate Grafana dashboard JSON configuration."""
        dashboard = {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [
                {
                    "collapsed": False,
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
                    "id": 100,
                    "panels": [],
                    "title": "Training Metrics",
                    "type": "row"
                },
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": datasource,
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 1},
                    "id": 1,
                    "legend": {
                        "avg": False,
                        "current": False,
                        "max": False,
                        "min": False,
                        "show": True,
                        "total": False,
                        "values": False
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {"dataLinks": []},
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "forge_training_loss",
                            "legendFormat": "Loss",
                            "refId": "A"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "Training Loss",
                    "tooltip": {"shared": True, "sort": 0, "value_type": "individual"},
                    "type": "graph",
                    "xaxis": {"buckets": None, "mode": "time", "name": None, "show": True},
                    "yaxes": [
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True},
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True}
                    ],
                    "yaxis": {"align": False, "alignLevel": None}
                },
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": datasource,
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 1},
                    "id": 2,
                    "legend": {
                        "avg": False,
                        "current": False,
                        "max": False,
                        "min": False,
                        "show": True,
                        "total": False,
                        "values": False
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {"dataLinks": []},
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "forge_learning_rate",
                            "legendFormat": "Learning Rate",
                            "refId": "A"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "Learning Rate",
                    "tooltip": {"shared": True, "sort": 0, "value_type": "individual"},
                    "type": "graph",
                    "xaxis": {"buckets": None, "mode": "time", "name": None, "show": True},
                    "yaxes": [
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True},
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True}
                    ],
                    "yaxis": {"align": False, "alignLevel": None}
                },
                {
                    "collapsed": False,
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 9},
                    "id": 101,
                    "panels": [],
                    "title": "Memory & Performance",
                    "type": "row"
                },
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": datasource,
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 10},
                    "id": 3,
                    "legend": {
                        "avg": False,
                        "current": False,
                        "max": False,
                        "min": False,
                        "show": True,
                        "total": False,
                        "values": False
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {"dataLinks": []},
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "forge_memory_allocated_gb",
                            "legendFormat": "Allocated",
                            "refId": "A"
                        },
                        {
                            "expr": "forge_memory_reserved_gb",
                            "legendFormat": "Reserved",
                            "refId": "B"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "GPU Memory Usage",
                    "tooltip": {"shared": True, "sort": 0, "value_type": "individual"},
                    "type": "graph",
                    "xaxis": {"buckets": None, "mode": "time", "name": None, "show": True},
                    "yaxes": [
                        {"format": "decgbytes", "label": None, "logBase": 1, "max": None, "min": None, "show": True},
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True}
                    ],
                    "yaxis": {"align": False, "alignLevel": None}
                },
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": datasource,
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 10},
                    "id": 4,
                    "legend": {
                        "avg": False,
                        "current": False,
                        "max": False,
                        "min": False,
                        "show": True,
                        "total": False,
                        "values": False
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {"dataLinks": []},
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "forge_samples_per_second",
                            "legendFormat": "Samples/sec",
                            "refId": "A"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "Training Throughput",
                    "tooltip": {"shared": True, "sort": 0, "value_type": "individual"},
                    "type": "graph",
                    "xaxis": {"buckets": None, "mode": "time", "name": None, "show": True},
                    "yaxes": [
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True},
                        {"format": "short", "label": None, "logBase": 1, "max": None, "min": None, "show": True}
                    ],
                    "yaxis": {"align": False, "alignLevel": None}
                },
                {
                    "collapsed": False,
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 18},
                    "id": 102,
                    "panels": [],
                    "title": "Alerts & Anomalies",
                    "type": "row"
                },
                {
                    "datasource": datasource,
                    "fieldConfig": {
                        "defaults": {
                            "custom": {}
                        },
                        "overrides": []
                    },
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 19},
                    "id": 5,
                    "options": {
                        "showHeader": True,
                        "sortBy": [{"desc": True, "displayName": "Time"}]
                    },
                    "pluginVersion": "7.0.0",
                    "targets": [
                        {
                            "expr": "forge_alerts_total",
                            "format": "table",
                            "instant": True,
                            "legendFormat": "",
                            "refId": "A"
                        }
                    ],
                    "timeFrom": None,
                    "timeShift": None,
                    "title": "Recent Alerts",
                    "type": "table"
                }
            ],
            "refresh": "5s",
            "schemaVersion": 22,
            "style": "dark",
            "tags": ["forge", "training", "monitoring"],
            "templating": {"list": []},
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "timezone": "",
            "title": title,
            "uid": "forge-training",
            "version": 1
        }
        
        return json.dumps(dashboard, indent=2)
    
    @staticmethod
    def save_dashboard(dashboard_json: str, output_path: str = "grafana_dashboard.json"):
        """Save dashboard JSON to file."""
        with open(output_path, 'w') as f:
            f.write(dashboard_json)
        logger.info(f"Grafana dashboard saved to {output_path}")


class TrainingDashboard:
    """Main dashboard class that integrates all monitoring components."""
    
    def __init__(
        self,
        alert_config: Optional[AlertConfig] = None,
        prometheus_port: int = 8000,
        enable_prometheus: bool = True,
        metrics_history_size: int = 1000
    ):
        self.metrics = TrainingMetrics(max_history=metrics_history_size)
        self.alert_config = alert_config or AlertConfig()
        self.anomaly_detector = AnomalyDetector(self.alert_config.thresholds)
        self.alert_manager = AlertManager(self.alert_config)
        
        self.prometheus_exporter = None
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self.prometheus_exporter = PrometheusExporter(port=prometheus_port)
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._update_interval = 10  # seconds
        
    def start(self, prometheus_server: bool = True):
        """Start the monitoring dashboard."""
        if self.prometheus_exporter and prometheus_server:
            self.prometheus_exporter.start_server()
        
        # Start background monitoring thread
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Training dashboard started")
    
    def stop(self):
        """Stop the monitoring dashboard."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Training dashboard stopped")
    
    def _monitoring_loop(self):
        """Background loop for continuous monitoring."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_anomalies()
                self._update_prometheus()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self._update_interval)
    
    def _check_anomalies(self):
        """Check for anomalies and send alerts."""
        anomalies = self.anomaly_detector.detect_anomalies(self.metrics)
        
        for anomaly in anomalies:
            # Send alert
            self.alert_manager.send_alert(anomaly)
            
            # Update Prometheus metrics
            if self.prometheus_exporter:
                self.prometheus_exporter.increment_alert_counter(
                    anomaly['type'],
                    anomaly['severity']
                )
    
    def _update_prometheus(self):
        """Update Prometheus metrics with latest values."""
        if not self.prometheus_exporter:
            return
            
        metrics_dict = {}
        
        # Update loss metrics
        if self.metrics.loss_history:
            current_loss = self.metrics.loss_history[-1]
            metrics_dict['training_loss'] = current_loss
            
            if len(self.metrics.loss_history) >= 100:
                avg_loss = statistics.mean(list(self.metrics.loss_history)[-100:])
                metrics_dict['training_loss_avg'] = avg_loss
        
        # Update other metrics
        if self.metrics.learning_rate_history:
            metrics_dict['learning_rate'] = self.metrics.learning_rate_history[-1]
        
        if self.metrics.gradient_norm_history:
            metrics_dict['gradient_norm'] = self.metrics.gradient_norm_history[-1]
        
        if self.metrics.memory_allocated_history:
            # Convert MB to GB
            metrics_dict['memory_allocated_gb'] = self.metrics.memory_allocated_history[-1] / 1024
        
        if self.metrics.memory_reserved_history:
            metrics_dict['memory_reserved_gb'] = self.metrics.memory_reserved_history[-1] / 1024
        
        if self.metrics.steps:
            metrics_dict['global_step'] = self.metrics.steps[-1]
        
        self.prometheus_exporter.update_metrics(metrics_dict)
    
    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        memory_allocated_mb: Optional[float] = None,
        memory_reserved_mb: Optional[float] = None,
        epoch: Optional[int] = None,
        samples_per_second: Optional[float] = None,
        **custom_metrics
    ):
        """Log a training step with all metrics."""
        # Update internal metrics
        update_dict = {"loss": loss}
        
        if learning_rate is not None:
            update_dict["learning_rate"] = learning_rate
        
        if gradient_norm is not None:
            update_dict["gradient_norm"] = gradient_norm
        
        if memory_allocated_mb is not None:
            update_dict["memory_allocated"] = memory_allocated_mb
        
        if memory_reserved_mb is not None:
            update_dict["memory_reserved"] = memory_reserved_mb
        
        # Add custom metrics
        update_dict.update(custom_metrics)
        
        self.metrics.update(step, **update_dict)
        
        # Update Prometheus immediately for critical metrics
        if self.prometheus_exporter:
            prom_metrics = {
                'training_loss': loss,
                'global_step': step
            }
            
            if learning_rate is not None:
                prom_metrics['learning_rate'] = learning_rate
            
            if gradient_norm is not None:
                prom_metrics['gradient_norm'] = gradient_norm
            
            if memory_allocated_mb is not None:
                prom_metrics['memory_allocated_gb'] = memory_allocated_mb / 1024
            
            if memory_reserved_mb is not None:
                prom_metrics['memory_reserved_gb'] = memory_reserved_mb / 1024
            
            if epoch is not None:
                prom_metrics['epoch'] = epoch
            
            if samples_per_second is not None:
                prom_metrics['samples_per_second'] = samples_per_second
            
            self.prometheus_exporter.update_metrics(prom_metrics)
    
    def log_custom_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a custom metric."""
        if step is None and self.metrics.steps:
            step = self.metrics.steps[-1]
        
        if step is not None:
            self.metrics.update(step, **{name: value})
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        summary = {
            "current_step": self.metrics.steps[-1] if self.metrics.steps else 0,
            "total_steps": len(self.metrics.steps),
            "metrics_available": []
        }
        
        if self.metrics.loss_history:
            summary["current_loss"] = self.metrics.loss_history[-1]
            summary["metrics_available"].append("loss")
        
        if self.metrics.learning_rate_history:
            summary["current_learning_rate"] = self.metrics.learning_rate_history[-1]
            summary["metrics_available"].append("learning_rate")
        
        if self.metrics.gradient_norm_history:
            summary["current_gradient_norm"] = self.metrics.gradient_norm_history[-1]
            summary["metrics_available"].append("gradient_norm")
        
        if self.metrics.memory_allocated_history:
            summary["current_memory_allocated_mb"] = self.metrics.memory_allocated_history[-1]
            summary["metrics_available"].append("memory_allocated")
        
        summary["custom_metrics"] = list(self.metrics.custom_metrics.keys())
        
        return summary
    
    def export_grafana_dashboard(self, output_path: str = "grafana_dashboard.json"):
        """Export Grafana dashboard configuration."""
        dashboard_json = GrafanaDashboard.generate_dashboard_json()
        GrafanaDashboard.save_dashboard(dashboard_json, output_path)
        return output_path


# Convenience functions for easy integration
def create_dashboard(
    alert_config: Optional[Dict[str, Any]] = None,
    prometheus_port: int = 8000,
    **kwargs
) -> TrainingDashboard:
    """Create a training dashboard with default configuration."""
    config = AlertConfig()
    
    if alert_config:
        # Update config with provided values
        for key, value in alert_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return TrainingDashboard(
        alert_config=config,
        prometheus_port=prometheus_port,
        **kwargs
    )


def log_training_metrics(
    dashboard: TrainingDashboard,
    step: int,
    loss: float,
    **kwargs
):
    """Convenience function to log training metrics."""
    dashboard.log_training_step(step=step, loss=loss, **kwargs)


# Example usage in training loop
if __name__ == "__main__":
    # Example configuration
    alert_config = AlertConfig(
        enabled=True,
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        thresholds=AlertThreshold(
            loss_spike_factor=1.5,
            memory_increase_mb=1000.0
        )
    )
    
    # Create dashboard
    dashboard = TrainingDashboard(alert_config=alert_config)
    dashboard.start()
    
    # Export Grafana dashboard
    dashboard.export_grafana_dashboard()
    
    # Simulate training loop
    try:
        for step in range(1000):
            # Simulate training metrics
            loss = 1.0 / (step + 1) + (0.1 if step % 100 == 50 else 0)  # Simulate spike
            lr = 0.001 * (0.99 ** step)
            grad_norm = 1.0 + 0.1 * step
            memory_mb = 1000 + step * 0.5  # Simulate memory increase
            
            dashboard.log_training_step(
                step=step,
                loss=loss,
                learning_rate=lr,
                gradient_norm=grad_norm,
                memory_allocated_mb=memory_mb,
                memory_reserved_mb=memory_mb * 1.2
            )
            
            # Log custom metric
            dashboard.log_custom_metric("accuracy", 0.5 + 0.4 * (step / 1000))
            
            time.sleep(0.1)  # Simulate training time
            
            if step % 100 == 0:
                summary = dashboard.get_metrics_summary()
                print(f"Step {step}: {summary}")
    
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        dashboard.stop()