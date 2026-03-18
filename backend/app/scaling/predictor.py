"""backend/app/scaling/predictor.py"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from enum import Enum

# Optional ML imports - graceful degradation if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("TensorFlow/scikit-learn not available. Predictive scaling will use fallback heuristics.")

from ..channels.message_bus import MessageBus, MessageEvent
from ..channels.manager import ChannelManager
from ..gateway.config import get_config

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class WorkloadMetrics:
    """Time-series metrics for workload prediction"""
    timestamp: datetime
    messages_per_minute: float
    active_channels: int
    avg_agent_duration_ms: float
    queue_depth: int
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    concurrent_agents: int = 0
    
    def to_feature_vector(self) -> List[float]:
        """Convert metrics to feature vector for ML model"""
        return [
            self.messages_per_minute,
            float(self.active_channels),
            self.avg_agent_duration_ms / 1000.0,  # Convert to seconds
            float(self.queue_depth),
            self.cpu_utilization,
            self.memory_utilization,
            float(self.concurrent_agents),
            self.timestamp.hour / 24.0,  # Hour of day (normalized)
            self.timestamp.weekday() / 7.0,  # Day of week (normalized)
        ]


@dataclass
class ScalingPrediction:
    """Prediction result for auto-scaling"""
    predicted_load: float  # Predicted messages per minute
    confidence: float  # 0-1 confidence score
    recommended_replicas: int
    time_horizon_minutes: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_load": self.predicted_load,
            "confidence": self.confidence,
            "recommended_replicas": self.recommended_replicas,
            "time_horizon_minutes": self.time_horizon_minutes,
            "timestamp": self.timestamp.isoformat(),
        }


class TimeSeriesBuffer:
    """Circular buffer for time-series data with windowing"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def add(self, metrics: WorkloadMetrics):
        self.buffer.append(metrics)
        self.timestamps.append(metrics.timestamp)
    
    def get_window(self, window_minutes: int) -> List[WorkloadMetrics]:
        """Get metrics from the last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        result = []
        
        # Iterate in reverse to get most recent first
        for i in range(len(self.buffer) - 1, -1, -1):
            if self.timestamps[i] >= cutoff:
                result.append(self.buffer[i])
            else:
                break
        
        return list(reversed(result))
    
    def to_dataframe(self, window_minutes: int = 60) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame for analysis"""
        window = self.get_window(window_minutes)
        if not window:
            return pd.DataFrame()
        
        data = []
        for metrics in window:
            row = {
                "timestamp": metrics.timestamp,
                "messages_per_minute": metrics.messages_per_minute,
                "active_channels": metrics.active_channels,
                "avg_agent_duration_ms": metrics.avg_agent_duration_ms,
                "queue_depth": metrics.queue_depth,
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "concurrent_agents": metrics.concurrent_agents,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


class LSTMPredictor:
    """Lightweight LSTM model for workload prediction"""
    
    def __init__(self, sequence_length: int = 30, feature_count: int = 9):
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.model = None
        self.scaler = MinMaxScaler() if ML_AVAILABLE else None
        self.is_trained = False
        self.training_data = []
        self.last_training_time = None
        
        if ML_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        if not ML_AVAILABLE:
            return
        
        self.model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, 
                            input_shape=(self.sequence_length, self.feature_count)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)  # Predict messages_per_minute
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # Predict messages_per_minute
        
        return np.array(X), np.array(y)
    
    def train(self, metrics_buffer: TimeSeriesBuffer, epochs: int = 50):
        """Train the LSTM model on historical data"""
        if not ML_AVAILABLE or not self.model:
            logger.warning("ML libraries not available, skipping training")
            return
        
        df = metrics_buffer.to_dataframe(window_minutes=120)  # Use 2 hours of data
        if len(df) < self.sequence_length + 10:  # Need minimum data
            logger.info(f"Insufficient data for training: {len(df)} samples")
            return
        
        try:
            # Prepare features
            features = []
            for _, row in df.iterrows():
                metrics = WorkloadMetrics(
                    timestamp=row.name,
                    messages_per_minute=row["messages_per_minute"],
                    active_channels=row["active_channels"],
                    avg_agent_duration_ms=row["avg_agent_duration_ms"],
                    queue_depth=row["queue_depth"],
                    cpu_utilization=row.get("cpu_utilization", 0.0),
                    memory_utilization=row.get("memory_utilization", 0.0),
                    concurrent_agents=row.get("concurrent_agents", 0),
                )
                features.append(metrics.to_feature_vector())
            
            features_array = np.array(features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_array)
            
            # Prepare sequences
            X, y = self.prepare_sequences(scaled_features)
            
            if len(X) == 0:
                logger.warning("No sequences generated for training")
                return
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            self.is_trained = True
            self.last_training_time = datetime.utcnow()
            
            logger.info(f"LSTM model trained on {len(X)} sequences. "
                       f"Final loss: {history.history['loss'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
    
    def predict(self, recent_metrics: List[WorkloadMetrics]) -> Optional[ScalingPrediction]:
        """Make prediction using trained model"""
        if not ML_AVAILABLE or not self.model or not self.is_trained:
            return None
        
        if len(recent_metrics) < self.sequence_length:
            logger.debug(f"Insufficient recent data for prediction: {len(recent_metrics)}")
            return None
        
        try:
            # Prepare input sequence
            features = [m.to_feature_vector() for m in recent_metrics[-self.sequence_length:]]
            features_array = np.array(features)
            
            # Scale features
            scaled_features = self.scaler.transform(features_array)
            
            # Reshape for LSTM [samples, time steps, features]
            X = scaled_features.reshape(1, self.sequence_length, self.feature_count)
            
            # Make prediction
            predicted_scaled = self.model.predict(X, verbose=0)[0][0]
            
            # Inverse scale for messages_per_minute (first feature)
            dummy = np.zeros((1, self.feature_count))
            dummy[0, 0] = predicted_scaled
            predicted_actual = self.scaler.inverse_transform(dummy)[0, 0]
            
            # Calculate confidence based on prediction variance
            confidence = min(0.95, max(0.5, 1.0 - (abs(predicted_actual) * 0.01)))
            
            return ScalingPrediction(
                predicted_load=max(0, predicted_actual),
                confidence=confidence,
                recommended_replicas=self._calculate_replicas(predicted_actual),
                time_horizon_minutes=10,  # Predict 10 minutes ahead
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _calculate_replicas(self, predicted_load: float) -> int:
        """Calculate required replicas based on predicted load"""
        # Simple heuristic: 1 replica per 10 messages/minute with minimum of 2
        base_replicas = max(2, int(np.ceil(predicted_load / 10.0)))
        
        # Add buffer for confidence
        config = get_config()
        buffer_percent = getattr(config, 'SCALING_BUFFER_PERCENT', 20)
        buffered_replicas = int(base_replicas * (1 + buffer_percent / 100.0))
        
        # Apply min/max constraints
        min_replicas = getattr(config, 'MIN_REPLICAS', 2)
        max_replicas = getattr(config, 'MAX_REPLICAS', 50)
        
        return max(min_replicas, min(max_replicas, buffered_replicas))


class HeuristicPredictor:
    """Fallback predictor using simple heuristics when ML is unavailable"""
    
    def __init__(self):
        self.window_size = 15  # minutes
    
    def predict(self, recent_metrics: List[WorkloadMetrics]) -> ScalingPrediction:
        """Make prediction using moving averages and trends"""
        if not recent_metrics:
            return ScalingPrediction(
                predicted_load=0.0,
                confidence=0.5,
                recommended_replicas=2,
                time_horizon_minutes=10,
            )
        
        # Calculate moving average
        loads = [m.messages_per_minute for m in recent_metrics[-self.window_size:]]
        avg_load = np.mean(loads) if loads else 0
        
        # Calculate trend (simple linear regression)
        if len(loads) >= 5:
            x = np.arange(len(loads))
            slope, _ = np.polyfit(x, loads, 1)
            trend_factor = 1.0 + (slope * 0.5)  # Adjust for trend
        else:
            trend_factor = 1.0
        
        predicted_load = avg_load * trend_factor
        
        # Simple replica calculation
        base_replicas = max(2, int(np.ceil(predicted_load / 12.0)))
        
        return ScalingPrediction(
            predicted_load=max(0, predicted_load),
            confidence=0.7,  # Lower confidence for heuristic
            recommended_replicas=base_replicas,
            time_horizon_minutes=10,
        )


class PredictiveAutoScaler:
    """Main predictive auto-scaling system"""
    
    def __init__(self, message_bus: MessageBus, channel_manager: ChannelManager):
        self.message_bus = message_bus
        self.channel_manager = channel_manager
        self.config = get_config()
        
        # Initialize components
        self.metrics_buffer = TimeSeriesBuffer(max_size=50000)
        self.mode = ScalingMode(getattr(self.config, 'SCALING_MODE', 'hybrid'))
        
        # Initialize predictors
        self.lstm_predictor = LSTMPredictor() if ML_AVAILABLE else None
        self.heuristic_predictor = HeuristicPredictor()
        
        # State tracking
        self.current_replicas = getattr(self.config, 'INITIAL_REPLICAS', 2)
        self.target_replicas = self.current_replicas
        self.last_prediction = None
        self.last_scaling_time = None
        self.message_counts = defaultdict(int)
        self.agent_durations = []
        
        # Configuration
        self.prediction_interval = getattr(self.config, 'PREDICTION_INTERVAL_SECONDS', 60)
        self.training_interval = getattr(self.config, 'TRAINING_INTERVAL_MINUTES', 30)
        self.cooldown_period = getattr(self.config, 'SCALING_COOLDOWN_MINUTES', 5)
        
        # Event subscriptions
        self._setup_event_handlers()
        
        # Background tasks
        self._prediction_task = None
        self._training_task = None
        self._metrics_collection_task = None
        
        logger.info(f"PredictiveAutoScaler initialized in {self.mode.value} mode")
    
    def _setup_event_handlers(self):
        """Subscribe to message bus events"""
        self.message_bus.subscribe("message_received", self._on_message_received)
        self.message_bus.subscribe("message_processed", self._on_message_processed)
        self.message_bus.subscribe("agent_started", self._on_agent_started)
        self.message_bus.subscribe("agent_completed", self._on_agent_completed)
    
    async def _on_message_received(self, event: MessageEvent):
        """Handle incoming message events"""
        channel_id = event.data.get("channel_id", "unknown")
        self.message_counts[channel_id] += 1
        
        # Update metrics every minute
        current_minute = datetime.utcnow().replace(second=0, microsecond=0)
        if not hasattr(self, '_last_metrics_update') or self._last_metrics_update != current_minute:
            await self._update_metrics()
            self._last_metrics_update = current_minute
    
    async def _on_message_processed(self, event: MessageEvent):
        """Handle processed message events"""
        duration_ms = event.data.get("duration_ms", 0)
        if duration_ms > 0:
            self.agent_durations.append(duration_ms)
            # Keep only last 1000 durations
            if len(self.agent_durations) > 1000:
                self.agent_durations = self.agent_durations[-1000:]
    
    async def _on_agent_started(self, event: MessageEvent):
        """Handle agent start events"""
        pass  # Could track concurrent agents here
    
    async def _on_agent_completed(self, event: MessageEvent):
        """Handle agent completion events"""
        pass  # Could track agent completion rate here
    
    async def _update_metrics(self):
        """Update workload metrics"""
        try:
            # Calculate messages per minute
            total_messages = sum(self.message_counts.values())
            messages_per_minute = total_messages / max(1, len(self.message_counts))
            
            # Get active channels
            active_channels = len([c for c in self.channel_manager.get_channels() 
                                 if c.is_active()])
            
            # Calculate average agent duration
            avg_duration = np.mean(self.agent_durations) if self.agent_durations else 0
            
            # Get system metrics (simplified)
            import psutil
            cpu_utilization = psutil.cpu_percent() / 100.0
            memory_utilization = psutil.virtual_memory().percent / 100.0
            
            # Create metrics object
            metrics = WorkloadMetrics(
                timestamp=datetime.utcnow(),
                messages_per_minute=messages_per_minute,
                active_channels=active_channels,
                avg_agent_duration_ms=avg_duration,
                queue_depth=0,  # Would need to integrate with queue system
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                concurrent_agents=0,  # Would need agent pool integration
            )
            
            # Add to buffer
            self.metrics_buffer.add(metrics)
            
            # Reset counters
            self.message_counts.clear()
            
            logger.debug(f"Updated metrics: {messages_per_minute:.1f} msg/min, "
                        f"{active_channels} channels, {avg_duration:.0f}ms avg duration")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def start(self):
        """Start the predictive auto-scaler"""
        logger.info("Starting PredictiveAutoScaler")
        
        # Start background tasks
        self._metrics_collection_task = asyncio.create_task(self._collect_metrics_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        
        if ML_AVAILABLE and self.lstm_predictor:
            self._training_task = asyncio.create_task(self._training_loop())
    
    async def stop(self):
        """Stop the predictive auto-scaler"""
        logger.info("Stopping PredictiveAutoScaler")
        
        # Cancel background tasks
        for task in [self._metrics_collection_task, self._prediction_task, self._training_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _collect_metrics_loop(self):
        """Background task to collect metrics periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                await self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _prediction_loop(self):
        """Background task to make predictions and trigger scaling"""
        while True:
            try:
                await asyncio.sleep(self.prediction_interval)
                
                # Get recent metrics
                recent_metrics = self.metrics_buffer.get_window(window_minutes=30)
                
                if len(recent_metrics) < 10:  # Need minimum data
                    logger.debug("Insufficient data for prediction")
                    continue
                
                # Make prediction based on mode
                prediction = None
                
                if self.mode == ScalingMode.PREDICTIVE and self.lstm_predictor and self.lstm_predictor.is_trained:
                    prediction = self.lstm_predictor.predict(recent_metrics)
                elif self.mode == ScalingMode.HYBRID:
                    # Try LSTM first, fallback to heuristic
                    if self.lstm_predictor and self.lstm_predictor.is_trained:
                        prediction = self.lstm_predictor.predict(recent_metrics)
                    if not prediction:
                        prediction = self.heuristic_predictor.predict(recent_metrics)
                else:  # REACTIVE or fallback
                    prediction = self.heuristic_predictor.predict(recent_metrics)
                
                if prediction:
                    self.last_prediction = prediction
                    logger.info(f"Scaling prediction: {prediction.predicted_load:.1f} msg/min, "
                              f"recommend {prediction.recommended_replicas} replicas "
                              f"(confidence: {prediction.confidence:.2f})")
                    
                    # Apply scaling if needed
                    await self._apply_scaling(prediction)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(5)
    
    async def _training_loop(self):
        """Background task to train the LSTM model"""
        while True:
            try:
                await asyncio.sleep(self.training_interval * 60)  # Convert to seconds
                
                if self.lstm_predictor and ML_AVAILABLE:
                    logger.info("Starting LSTM model training")
                    self.lstm_predictor.train(self.metrics_buffer)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(60)
    
    async def _apply_scaling(self, prediction: ScalingPrediction):
        """Apply scaling decision based on prediction"""
        # Check cooldown period
        now = datetime.utcnow()
        if (self.last_scaling_time and 
            (now - self.last_scaling_time).total_seconds() < self.cooldown_period * 60):
            logger.debug("Scaling cooldown active, skipping")
            return
        
        # Check if scaling is needed
        if prediction.recommended_replicas == self.current_replicas:
            logger.debug("No scaling needed, current replicas match recommendation")
            return
        
        # Check confidence threshold
        min_confidence = getattr(self.config, 'MIN_SCALING_CONFIDENCE', 0.7)
        if prediction.confidence < min_confidence:
            logger.warning(f"Prediction confidence too low: {prediction.confidence:.2f} "
                          f"(minimum: {min_confidence})")
            return
        
        # Apply scaling
        old_replicas = self.current_replicas
        self.target_replicas = prediction.recommended_replicas
        
        try:
            # Integrate with Kubernetes HPA or cloud provider
            success = await self._scale_resources(self.target_replicas)
            
            if success:
                self.current_replicas = self.target_replicas
                self.last_scaling_time = now
                
                logger.info(f"Scaling applied: {old_replicas} -> {self.current_replicas} replicas "
                          f"(predicted load: {prediction.predicted_load:.1f} msg/min)")
                
                # Emit scaling event
                await self.message_bus.emit("scaling_applied", {
                    "old_replicas": old_replicas,
                    "new_replicas": self.current_replicas,
                    "prediction": prediction.to_dict(),
                    "timestamp": now.isoformat(),
                })
            else:
                logger.error(f"Failed to scale to {self.target_replicas} replicas")
                
        except Exception as e:
            logger.error(f"Error applying scaling: {e}")
    
    async def _scale_resources(self, target_replicas: int) -> bool:
        """Scale resources using Kubernetes HPA or cloud provider"""
        # This is a placeholder for actual scaling integration
        # In production, this would call Kubernetes API or cloud provider SDK
        
        scaling_method = getattr(self.config, 'SCALING_METHOD', 'kubernetes')
        
        if scaling_method == 'kubernetes':
            # Example: Update Kubernetes HPA
            # In reality, you'd use kubernetes_asyncio client
            logger.info(f"Would scale Kubernetes deployment to {target_replicas} replicas")
            return True
            
        elif scaling_method == 'aws':
            # Example: Update AWS Auto Scaling Group
            logger.info(f"Would scale AWS ASG to {target_replicas} instances")
            return True
            
        elif scaling_method == 'gcp':
            # Example: Update GCP Managed Instance Group
            logger.info(f"Would scale GCP MIG to {target_replicas} instances")
            return True
            
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the auto-scaler"""
        return {
            "mode": self.mode.value,
            "current_replicas": self.current_replicas,
            "target_replicas": self.target_replicas,
            "last_prediction": self.last_prediction.to_dict() if self.last_prediction else None,
            "last_training_time": self.lstm_predictor.last_training_time.isoformat() 
                if self.lstm_predictor and self.lstm_predictor.last_training_time else None,
            "model_trained": self.lstm_predictor.is_trained if self.lstm_predictor else False,
            "metrics_buffer_size": len(self.metrics_buffer.buffer),
            "ml_available": ML_AVAILABLE,
        }
    
    async def force_prediction(self) -> Optional[ScalingPrediction]:
        """Force an immediate prediction (for testing/manual override)"""
        recent_metrics = self.metrics_buffer.get_window(window_minutes=30)
        
        if len(recent_metrics) < 5:
            return None
        
        if self.mode == ScalingMode.PREDICTIVE and self.lstm_predictor and self.lstm_predictor.is_trained:
            return self.lstm_predictor.predict(recent_metrics)
        else:
            return self.heuristic_predictor.predict(recent_metrics)
    
    async def set_mode(self, mode: ScalingMode):
        """Change scaling mode"""
        self.mode = mode
        logger.info(f"Scaling mode changed to {mode.value}")
        
        # Emit mode change event
        await self.message_bus.emit("scaling_mode_changed", {
            "mode": mode.value,
            "timestamp": datetime.utcnow().isoformat(),
        })


# Factory function for easy integration
def create_predictive_auto_scaler(message_bus: MessageBus, 
                                channel_manager: ChannelManager) -> PredictiveAutoScaler:
    """Create and return a PredictiveAutoScaler instance"""
    return PredictiveAutoScaler(message_bus, channel_manager)