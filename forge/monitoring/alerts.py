import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import json
import os

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not installed. Prometheus metrics disabled.")

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    metric: str
    condition: str  # e.g., "> 0.9", "< 0.01", "spike > 2.0"
    threshold: float
    severity: AlertSeverity
    window_size: int = 10  # Number of data points to consider
    cooldown: int = 60  # Seconds between alerts for same rule
    enabled: bool = True


@dataclass
class Alert:
    """Represents a triggered alert."""
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    timestamp: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Simple statistical anomaly detection for training metrics."""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history: Dict[str, deque] = {}
        
    def add_datapoint(self, metric: str, value: float):
        """Add a new datapoint to history."""
        if metric not in self.history:
            self.history[metric] = deque(maxlen=self.window_size)
        self.history[metric].append(value)
    
    def detect_spike(self, metric: str, current_value: float) -> bool:
        """Detect if current value is a statistical spike."""
        if metric not in self.history or len(self.history[metric]) < 10:
            return False
        
        history = list(self.history[metric])
        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0
        
        if stdev == 0:
            return False
        
        z_score = abs(current_value - mean) / stdev
        return z_score > self.z_threshold
    
    def detect_trend(self, metric: str, window: int = 20) -> Optional[str]:
        """Detect upward or downward trend."""
        if metric not in self.history or len(self.history[metric]) < window:
            return None
        
        recent = list(self.history[metric])[-window:]
        if len(recent) < 2:
            return None
        
        # Simple linear regression slope
        x = list(range(len(recent)))
        y = recent
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"


class TrainingMonitor:
    """
    Real-time training monitor with metrics collection, anomaly detection, and alerting.
    
    Integrates with Prometheus for metrics export and provides alerting capabilities
    for common training issues like loss spikes, memory leaks, and gradient problems.
    """
    
    def __init__(
        self,
        experiment_name: str = "forge_training",
        prometheus_port: int = 9090,
        enable_prometheus: bool = True,
        alert_rules: Optional[List[AlertRule]] = None,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        self.experiment_name = experiment_name
        self.prometheus_port = prometheus_port
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.alert_callback = alert_callback or self._default_alert_callback
        
        # Metrics storage
        self.metrics_history: Dict[str, List[float]] = {}
        self.current_step = 0
        self.start_time = time.time()
        
        # Anomaly detection
        self.anomaly_detector = AnomalyDetector()
        
        # Alert management
        self.alert_rules = alert_rules or self._default_alert_rules()
        self.alerts: List[Alert] = []
        self.last_alert_time: Dict[str, float] = {}
        
        # Prometheus metrics
        self.prometheus_metrics: Dict[str, Any] = {}
        self._setup_prometheus_metrics()
        
        # Threading for async operations
        self._lock = threading.RLock()
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        
        logger.info(f"TrainingMonitor initialized for experiment: {experiment_name}")
    
    def _default_alert_rules(self) -> List[AlertRule]:
        """Create default alert rules for common training issues."""
        return [
            AlertRule(
                name="loss_spike",
                metric="train/loss",
                condition="spike > 3.0",
                threshold=3.0,
                severity=AlertSeverity.WARNING,
                window_size=20,
                cooldown=300
            ),
            AlertRule(
                name="loss_not_decreasing",
                metric="train/loss",
                condition="trend increasing",
                threshold=0,
                severity=AlertSeverity.INFO,
                window_size=100,
                cooldown=600
            ),
            AlertRule(
                name="high_memory_usage",
                metric="system/memory_used_percent",
                condition="> 90",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                window_size=5,
                cooldown=60
            ),
            AlertRule(
                name="gradient_explosion",
                metric="train/grad_norm",
                condition="> 10.0",
                threshold=10.0,
                severity=AlertSeverity.ERROR,
                window_size=10,
                cooldown=120
            ),
            AlertRule(
                name="learning_rate_too_high",
                metric="train/lr",
                condition="> 0.01",
                threshold=0.01,
                severity=AlertSeverity.INFO,
                window_size=5,
                cooldown=300
            ),
            AlertRule(
                name="eval_metric_degradation",
                metric="eval/accuracy",
                condition="spike < -0.1",
                threshold=-0.1,
                severity=AlertSeverity.ERROR,
                window_size=10,
                cooldown=600
            )
        ]
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics if enabled."""
        if not self.enable_prometheus:
            return
        
        # Training metrics
        self.prometheus_metrics['train_loss'] = Gauge(
            f'{self.experiment_name}_train_loss', 
            'Training loss'
        )
        self.prometheus_metrics['eval_loss'] = Gauge(
            f'{self.experiment_name}_eval_loss', 
            'Evaluation loss'
        )
        self.prometheus_metrics['learning_rate'] = Gauge(
            f'{self.experiment_name}_learning_rate', 
            'Current learning rate'
        )
        self.prometheus_metrics['grad_norm'] = Gauge(
            f'{self.experiment_name}_grad_norm', 
            'Gradient norm'
        )
        self.prometheus_metrics['epoch'] = Gauge(
            f'{self.experiment_name}_epoch', 
            'Current epoch'
        )
        self.prometheus_metrics['step'] = Gauge(
            f'{self.experiment_name}_step', 
            'Current training step'
        )
        
        # System metrics
        self.prometheus_metrics['memory_used'] = Gauge(
            f'{self.experiment_name}_memory_used_bytes', 
            'Memory used in bytes'
        )
        self.prometheus_metrics['memory_used_percent'] = Gauge(
            f'{self.experiment_name}_memory_used_percent', 
            'Memory used percentage'
        )
        self.prometheus_metrics['gpu_memory_used'] = Gauge(
            f'{self.experiment_name}_gpu_memory_used_bytes', 
            'GPU memory used in bytes'
        )
        self.prometheus_metrics['gpu_utilization'] = Gauge(
            f'{self.experiment_name}_gpu_utilization_percent', 
            'GPU utilization percentage'
        )
        
        # Custom metrics (can be extended)
        self.prometheus_metrics['custom_metrics'] = Gauge(
            f'{self.experiment_name}_custom_metric', 
            'Custom training metric',
            ['metric_name']
        )
        
        # Counters for events
        self.prometheus_metrics['alerts_total'] = Counter(
            f'{self.experiment_name}_alerts_total', 
            'Total alerts triggered',
            ['severity']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def start(self):
        """Start the monitoring server and background tasks."""
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
                self.enable_prometheus = False
        
        # Start background thread for periodic tasks
        self._server_thread = threading.Thread(
            target=self._background_tasks,
            daemon=True,
            name="TrainingMonitor-Background"
        )
        self._server_thread.start()
        
        logger.info("TrainingMonitor started")
    
    def stop(self):
        """Stop the monitoring server."""
        self._running = False
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        
        logger.info("TrainingMonitor stopped")
    
    def _background_tasks(self):
        """Background thread for periodic tasks."""
        while self._running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for anomalies and alerts
                self._check_alerts()
                
                # Clean old data
                self._cleanup_old_data()
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in background tasks: {e}")
                time.sleep(10)
    
    def _update_system_metrics(self):
        """Update system metrics (memory, GPU, etc.)."""
        try:
            import psutil
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system/memory_used_bytes", memory.used)
            self.record_metric("system/memory_used_percent", memory.percent)
            
            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.record_metric("system/gpu_memory_used_bytes", gpu.memoryUsed * 1024 * 1024)
                    self.record_metric("system/gpu_utilization_percent", gpu.load * 100)
            except ImportError:
                pass  # GPUtil not available
                
        except ImportError:
            pass  # psutil not available
    
    def _check_alerts(self):
        """Check all alert rules and trigger alerts if conditions are met."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_alert = self.last_alert_time.get(rule.name, 0)
            if current_time - last_alert < rule.cooldown:
                continue
            
            # Get metric value
            metric_value = self._get_current_metric_value(rule.metric)
            if metric_value is None:
                continue
            
            # Check condition
            alert_triggered = False
            message = ""
            
            if "spike" in rule.condition:
                # Anomaly detection for spikes
                if self.anomaly_detector.detect_spike(rule.metric, metric_value):
                    alert_triggered = True
                    message = f"Spike detected in {rule.metric}: {metric_value:.4f}"
            
            elif "trend" in rule.condition:
                # Trend detection
                trend = self.anomaly_detector.detect_trend(rule.metric)
                if trend and "increasing" in rule.condition and trend == "increasing":
                    alert_triggered = True
                    message = f"Upward trend detected in {rule.metric}"
                elif trend and "decreasing" in rule.condition and trend == "decreasing":
                    alert_triggered = True
                    message = f"Downward trend detected in {rule.metric}"
            
            else:
                # Simple threshold comparison
                if ">" in rule.condition and metric_value > rule.threshold:
                    alert_triggered = True
                    message = f"{rule.metric} ({metric_value:.4f}) > {rule.threshold}"
                elif "<" in rule.condition and metric_value < rule.threshold:
                    alert_triggered = True
                    message = f"{rule.metric} ({metric_value:.4f}) < {rule.threshold}"
                elif "==" in rule.condition and abs(metric_value - rule.threshold) < 1e-6:
                    alert_triggered = True
                    message = f"{rule.metric} ({metric_value:.4f}) == {rule.threshold}"
            
            if alert_triggered:
                alert = Alert(
                    rule_name=rule.name,
                    metric=rule.metric,
                    value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    timestamp=current_time,
                    message=message,
                    metadata={"step": self.current_step}
                )
                
                self._trigger_alert(alert)
                self.last_alert_time[rule.name] = current_time
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert and notify via callback."""
        with self._lock:
            self.alerts.append(alert)
            
            # Update Prometheus counter
            if self.enable_prometheus:
                self.prometheus_metrics['alerts_total'].labels(
                    severity=alert.severity.value
                ).inc()
            
            # Call alert callback
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"Alert triggered: [{alert.severity.value}] {alert.message}")
    
    def _default_alert_callback(self, alert: Alert):
        """Default alert callback that logs alerts."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT: {alert.message} (rule: {alert.rule_name})")
    
    def _get_current_metric_value(self, metric: str) -> Optional[float]:
        """Get the current value of a metric."""
        with self._lock:
            if metric in self.metrics_history and self.metrics_history[metric]:
                return self.metrics_history[metric][-1]
        return None
    
    def _cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old metrics data to prevent memory leaks."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self._lock:
            # In a real implementation, we'd need to track timestamps per metric
            # For simplicity, we'll just limit the history size
            for metric in self.metrics_history:
                if len(self.metrics_history[metric]) > 10000:  # Keep last 10k points
                    self.metrics_history[metric] = self.metrics_history[metric][-5000:]
    
    def record_metric(self, metric: str, value: float, step: Optional[int] = None):
        """
        Record a metric value.
        
        Args:
            metric: Metric name (e.g., "train/loss", "eval/accuracy")
            value: Metric value
            step: Optional step number (if not provided, uses internal counter)
        """
        if step is not None:
            self.current_step = step
        
        with self._lock:
            # Store in history
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(value)
            
            # Update anomaly detector
            self.anomaly_detector.add_datapoint(metric, value)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self._update_prometheus_metric(metric, value)
    
    def _update_prometheus_metric(self, metric: str, value: float):
        """Update the corresponding Prometheus metric."""
        metric_mapping = {
            "train/loss": "train_loss",
            "eval/loss": "eval_loss",
            "train/lr": "learning_rate",
            "train/grad_norm": "grad_norm",
            "train/epoch": "epoch",
            "train/step": "step",
            "system/memory_used_bytes": "memory_used",
            "system/memory_used_percent": "memory_used_percent",
            "system/gpu_memory_used_bytes": "gpu_memory_used",
            "system/gpu_utilization_percent": "gpu_utilization",
        }
        
        if metric in metric_mapping:
            prometheus_name = metric_mapping[metric]
            if prometheus_name in self.prometheus_metrics:
                self.prometheus_metrics[prometheus_name].set(value)
        elif metric.startswith("custom/"):
            # Handle custom metrics
            metric_name = metric.replace("custom/", "")
            self.prometheus_metrics['custom_metrics'].labels(
                metric_name=metric_name
            ).set(value)
    
    def record_training_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        **kwargs
    ):
        """
        Record metrics for a training step.
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Training loss
            lr: Current learning rate
            grad_norm: Gradient norm (optional)
            **kwargs: Additional metrics to record
        """
        self.current_step = step
        
        self.record_metric("train/step", step)
        self.record_metric("train/epoch", epoch)
        self.record_metric("train/loss", loss)
        self.record_metric("train/lr", lr)
        
        if grad_norm is not None:
            self.record_metric("train/grad_norm", grad_norm)
        
        # Record any additional metrics
        for key, value in kwargs.items():
            self.record_metric(f"custom/{key}", value)
    
    def record_eval_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Record evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            step: Optional step number
        """
        for metric_name, value in metrics.items():
            self.record_metric(f"eval/{metric_name}", value, step)
    
    def get_metrics_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """
        Get a summary of recent metrics.
        
        Args:
            last_n: Number of recent data points to include
            
        Returns:
            Dictionary with metrics summary
        """
        summary = {}
        
        with self._lock:
            for metric, values in self.metrics_history.items():
                if values:
                    recent_values = values[-last_n:] if len(values) > last_n else values
                    summary[metric] = {
                        "current": recent_values[-1],
                        "mean": statistics.mean(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                        "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
                        "count": len(recent_values)
                    }
        
        return summary
    
    def get_recent_alerts(self, last_n: int = 10) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return self.alerts[-last_n:] if len(self.alerts) > last_n else self.alerts
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        with self._lock:
            self.alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule by name."""
        with self._lock:
            self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
    
    def enable_alert_rule(self, rule_name: str, enabled: bool = True):
        """Enable or disable an alert rule."""
        with self._lock:
            for rule in self.alert_rules:
                if rule.name == rule_name:
                    rule.enabled = enabled
                    break
    
    def export_dashboard_config(self, output_path: str = "grafana_dashboard.json"):
        """
        Export a Grafana dashboard configuration.
        
        Args:
            output_path: Path to save the dashboard JSON
        """
        dashboard = {
            "dashboard": {
                "title": f"LLaMA Factory Training - {self.experiment_name}",
                "panels": self._generate_grafana_panels(),
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard config exported to {output_path}")
    
    def _generate_grafana_panels(self) -> List[Dict]:
        """Generate Grafana panel configurations."""
        panels = []
        panel_id = 1
        
        # Training metrics panels
        training_metrics = [
            ("train_loss", "Training Loss", "Loss"),
            ("eval_loss", "Evaluation Loss", "Loss"),
            ("learning_rate", "Learning Rate", "LR"),
            ("grad_norm", "Gradient Norm", "Norm"),
        ]
        
        for metric, title, y_axis in training_metrics:
            panels.append({
                "id": panel_id,
                "title": title,
                "type": "graph",
                "targets": [{
                    "expr": f'{self.experiment_name}_{metric}',
                    "legendFormat": title
                }],
                "yaxes": [{"label": y_axis, "format": "short"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            })
            panel_id += 1
        
        # System metrics panels
        system_metrics = [
            ("memory_used_percent", "Memory Usage", "%"),
            ("gpu_utilization_percent", "GPU Utilization", "%"),
        ]
        
        for metric, title, unit in system_metrics:
            panels.append({
                "id": panel_id,
                "title": title,
                "type": "graph",
                "targets": [{
                    "expr": f'{self.experiment_name}_{metric}',
                    "legendFormat": title
                }],
                "yaxes": [{"label": unit, "format": "percent"}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            })
            panel_id += 1
        
        # Alerts panel
        panels.append({
            "id": panel_id,
            "title": "Alerts",
            "type": "stat",
            "targets": [{
                "expr": f'sum({self.experiment_name}_alerts_total) by (severity)',
                "legendFormat": "{{severity}}"
            }],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
        })
        
        return panels


# Singleton instance for global access
_monitor_instance: Optional[TrainingMonitor] = None


def get_monitor(**kwargs) -> TrainingMonitor:
    """Get or create the global TrainingMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TrainingMonitor(**kwargs)
    return _monitor_instance


def init_monitoring(
    experiment_name: str = "forge_training",
    prometheus_port: int = 9090,
    enable_prometheus: bool = True,
    alert_rules: Optional[List[AlertRule]] = None,
    alert_callback: Optional[Callable[[Alert], None]] = None,
    auto_start: bool = True
) -> TrainingMonitor:
    """
    Initialize the global training monitor.
    
    Args:
        experiment_name: Name of the experiment
        prometheus_port: Port for Prometheus metrics server
        enable_prometheus: Whether to enable Prometheus metrics
        alert_rules: Custom alert rules
        alert_callback: Custom alert callback function
        auto_start: Whether to automatically start the monitor
        
    Returns:
        TrainingMonitor instance
    """
    global _monitor_instance
    
    _monitor_instance = TrainingMonitor(
        experiment_name=experiment_name,
        prometheus_port=prometheus_port,
        enable_prometheus=enable_prometheus,
        alert_rules=alert_rules,
        alert_callback=alert_callback
    )
    
    if auto_start:
        _monitor_instance.start()
    
    return _monitor_instance


def record_metric(metric: str, value: float, step: Optional[int] = None):
    """Convenience function to record a metric to the global monitor."""
    monitor = get_monitor()
    monitor.record_metric(metric, value, step)


def record_training_step(step: int, epoch: float, loss: float, lr: float, **kwargs):
    """Convenience function to record a training step to the global monitor."""
    monitor = get_monitor()
    monitor.record_training_step(step, epoch, loss, lr, **kwargs)


def record_eval_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Convenience function to record evaluation metrics to the global monitor."""
    monitor = get_monitor()
    monitor.record_eval_metrics(metrics, step)


# Example usage and integration with training loops
if __name__ == "__main__":
    # Example: Initialize monitoring
    monitor = init_monitoring(
        experiment_name="llama2_7b_finetune",
        prometheus_port=9090,
        enable_prometheus=True
    )
    
    # Example: Simulate training loop
    for step in range(100):
        # Simulate training metrics
        loss = 2.0 / (step + 1) + 0.1 * (step % 10 == 0)  # Simulate occasional spikes
        lr = 0.001 * (0.99 ** step)
        grad_norm = 1.0 + 0.5 * (step % 5 == 0)  # Simulate occasional high gradients
        
        # Record training step
        monitor.record_training_step(
            step=step,
            epoch=step / 100,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            custom_metric=step * 0.01
        )
        
        # Simulate evaluation every 10 steps
        if step % 10 == 0:
            eval_metrics = {
                "loss": loss * 0.9,
                "accuracy": 0.5 + step * 0.005,
                "perplexity": 10.0 / (step + 1)
            }
            monitor.record_eval_metrics(eval_metrics, step)
        
        time.sleep(0.1)  # Simulate training time
    
    # Export dashboard configuration
    monitor.export_dashboard_config("training_dashboard.json")
    
    # Print summary
    summary = monitor.get_metrics_summary()
    print("Metrics Summary:")
    for metric, stats in summary.items():
        print(f"  {metric}: current={stats['current']:.4f}, mean={stats['mean']:.4f}")
    
    # Print recent alerts
    alerts = monitor.get_recent_alerts()
    if alerts:
        print(f"\nRecent Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"  [{alert.severity.value}] {alert.message}")
    
    # Stop monitoring
    monitor.stop()