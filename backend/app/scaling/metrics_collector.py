import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Attempt to import ML libraries with fallbacks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("TensorFlow not available. Using statistical fallback for predictions.")

try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. Using manual normalization.")

from backend.app.channels.message_bus import MessageBus
from backend.app.channels.manager import ChannelManager
from backend.app.channels.base import BaseChannel

logger = logging.getLogger(__name__)

class MetricType(Enum):
    MESSAGE_ARRIVAL = "message_arrival"
    AGENT_DURATION = "agent_duration"
    CHANNEL_ACTIVITY = "channel_activity"
    QUEUE_DEPTH = "queue_depth"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"

@dataclass
class TimeSeriesPoint:
    timestamp: datetime
    value: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingPrediction:
    timestamp: datetime
    predicted_load: float
    confidence: float
    recommended_replicas: int
    horizon_minutes: int
    model_version: str

class MetricsStorage:
    """Time-series storage for metrics with efficient querying"""
    
    def __init__(self, retention_hours: int = 168):  # 7 days default
        self.retention_hours = retention_hours
        self._storage: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=retention_hours * 60)  # Store per-minute data
            for metric_type in MetricType
        }
        self._lock = asyncio.Lock()
    
    async def add_point(self, point: TimeSeriesPoint):
        async with self._lock:
            self._storage[point.metric_type].append(point)
    
    async def get_points(self, metric_type: MetricType, 
                        start_time: datetime, 
                        end_time: datetime) -> List[TimeSeriesPoint]:
        async with self._lock:
            points = list(self._storage[metric_type])
            return [p for p in points if start_time <= p.timestamp <= end_time]
    
    async def get_aggregated(self, metric_type: MetricType,
                           window_minutes: int = 5,
                           start_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Get aggregated metrics for a time window"""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=1)
        
        end_time = datetime.utcnow()
        points = await self.get_points(metric_type, start_time, end_time)
        
        if not points:
            return []
        
        # Group by time windows
        aggregated = {}
        for point in points:
            window_key = point.timestamp.replace(
                second=0, microsecond=0,
                minute=point.timestamp.minute // window_minutes * window_minutes
            )
            if window_key not in aggregated:
                aggregated[window_key] = []
            aggregated[window_key].append(point.value)
        
        # Calculate averages
        result = []
        for window, values in sorted(aggregated.items()):
            avg_value = sum(values) / len(values)
            result.append((window, avg_value))
        
        return result

class LSTMPredictor:
    """LSTM-based load predictor with fallback to statistical methods"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 15):
        self.sequence_length = sequence_length  # Look back 60 minutes
        self.prediction_horizon = prediction_horizon  # Predict 15 minutes ahead
        self.model = None
        self.scaler = None
        self.model_version = "v1.0"
        self.is_trained = False
        
        if HAS_TF:
            self._build_model()
        if HAS_SKLEARN:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def _build_model(self):
        """Build LSTM model architecture"""
        if not HAS_TF:
            return
        
        self.model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, 
                       input_shape=(self.sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length + self.prediction_horizon - 1])
        return np.array(X), np.array(y)
    
    def train(self, historical_data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        if not HAS_TF or len(historical_data) < self.sequence_length + self.prediction_horizon + 10:
            logger.warning("Insufficient data or TensorFlow unavailable. Using statistical fallback.")
            self.is_trained = False
            return
        
        try:
            # Normalize data
            if HAS_SKLEARN and self.scaler:
                data_scaled = self.scaler.fit_transform(historical_data.reshape(-1, 1))
            else:
                # Manual normalization
                data_min, data_max = historical_data.min(), historical_data.max()
                data_scaled = (historical_data - data_min) / (data_max - data_min + 1e-10)
                self.scaler = (data_min, data_max)
            
            # Prepare sequences
            X, y = self._prepare_data(data_scaled.flatten())
            
            if len(X) == 0:
                logger.warning("No valid sequences for training")
                return
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Train model
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                          validation_split=0.2, verbose=0)
            
            self.is_trained = True
            self.model_version = f"lstm_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
            logger.info(f"LSTM model trained successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            self.is_trained = False
    
    def predict(self, recent_data: np.ndarray) -> Tuple[float, float]:
        """Make prediction with confidence score"""
        if not self.is_trained or not HAS_TF:
            # Fallback to statistical prediction
            return self._statistical_predict(recent_data)
        
        try:
            # Prepare input
            if len(recent_data) < self.sequence_length:
                # Pad with zeros if not enough data
                padded = np.zeros(self.sequence_length)
                padded[-len(recent_data):] = recent_data
                input_data = padded
            else:
                input_data = recent_data[-self.sequence_length:]
            
            # Scale input
            if HAS_SKLEARN and hasattr(self.scaler, 'transform'):
                input_scaled = self.scaler.transform(input_data.reshape(-1, 1))
            else:
                # Manual scaling
                data_min, data_max = self.scaler
                input_scaled = (input_data - data_min) / (data_max - data_min + 1e-10)
            
            # Reshape for prediction
            input_reshaped = input_scaled.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            prediction_scaled = self.model.predict(input_reshaped, verbose=0)[0][0]
            
            # Inverse transform
            if HAS_SKLEARN and hasattr(self.scaler, 'inverse_transform'):
                prediction = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
            else:
                data_min, data_max = self.scaler
                prediction = prediction_scaled * (data_max - data_min) + data_min
            
            # Calculate confidence based on prediction variance
            confidence = min(0.95, max(0.5, 1.0 - np.std(input_data) / (np.mean(input_data) + 1e-10)))
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return self._statistical_predict(recent_data)
    
    def _statistical_predict(self, data: np.ndarray) -> Tuple[float, float]:
        """Fallback statistical prediction using exponential smoothing"""
        if len(data) == 0:
            return 0.0, 0.5
        
        # Simple exponential smoothing
        alpha = 0.3
        smoothed = data[0]
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Add trend component if enough data
        if len(data) >= 10:
            recent_trend = np.polyfit(range(len(data[-10:])), data[-10:], 1)[0]
            prediction = smoothed + recent_trend * self.prediction_horizon
        else:
            prediction = smoothed
        
        confidence = 0.7  # Lower confidence for statistical method
        return float(prediction), confidence

class MetricsCollector:
    """
    Collects and analyzes metrics for predictive auto-scaling.
    Integrates with existing channel infrastructure.
    """
    
    def __init__(self, 
                 message_bus: MessageBus,
                 channel_manager: ChannelManager,
                 collection_interval: int = 60):  # seconds
        self.message_bus = message_bus
        self.channel_manager = channel_manager
        self.collection_interval = collection_interval
        self.storage = MetricsStorage()
        self.predictor = LSTMPredictor()
        self._running = False
        self._collection_task = None
        self._prediction_task = None
        
        # Metrics tracking
        self._message_counts: Dict[str, int] = {}  # channel_id -> count
        self._agent_durations: List[float] = []
        self._last_collection_time = datetime.utcnow()
        
        # Subscribe to message bus events
        self._setup_event_handlers()
        
        logger.info("MetricsCollector initialized")
    
    def _setup_event_handlers(self):
        """Setup handlers for message bus events"""
        # In a real implementation, we'd subscribe to message events
        # For now, we'll collect metrics during the collection cycle
        pass
    
    async def start(self):
        """Start metrics collection and prediction"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        
        logger.info("MetricsCollector started")
    
    async def stop(self):
        """Stop metrics collection"""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
        if self._prediction_task:
            self._prediction_task.cancel()
        
        try:
            await asyncio.gather(self._collection_task, self._prediction_task, 
                               return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        logger.info("MetricsCollector stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _prediction_loop(self):
        """Periodic prediction generation"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Predict every 5 minutes
                await self._generate_predictions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self):
        """Collect current metrics from all sources"""
        current_time = datetime.utcnow()
        
        # 1. Message arrival metrics
        await self._collect_message_metrics(current_time)
        
        # 2. Agent duration metrics (simulated - would come from agent manager)
        await self._collect_agent_metrics(current_time)
        
        # 3. Channel activity metrics
        await self._collect_channel_metrics(current_time)
        
        # 4. System metrics (simulated)
        await self._collect_system_metrics(current_time)
        
        self._last_collection_time = current_time
        logger.debug(f"Metrics collected at {current_time}")
    
    async def _collect_message_metrics(self, timestamp: datetime):
        """Collect message arrival patterns"""
        # In production, this would query the message bus for recent messages
        # For now, we'll simulate based on channel activity
        
        total_messages = 0
        channels = self.channel_manager.get_all_channels()
        
        for channel_id, channel in channels.items():
            # Simulate message count (in reality, get from message bus)
            # This is a placeholder - actual implementation would track real messages
            simulated_count = np.random.poisson(5)  # Average 5 messages per minute per channel
            total_messages += simulated_count
            
            # Store per-channel metric
            point = TimeSeriesPoint(
                timestamp=timestamp,
                value=simulated_count,
                metric_type=MetricType.MESSAGE_ARRIVAL,
                metadata={"channel_id": channel_id}
            )
            await self.storage.add_point(point)
        
        # Store total message rate
        total_point = TimeSeriesPoint(
            timestamp=timestamp,
            value=total_messages,
            metric_type=MetricType.MESSAGE_ARRIVAL,
            metadata={"scope": "total"}
        )
        await self.storage.add_point(total_point)
    
    async def _collect_agent_metrics(self, timestamp: datetime):
        """Collect agent processing duration metrics"""
        # Simulated agent durations - in production, integrate with agent manager
        if np.random.random() < 0.3:  # 30% chance of having agent activity
            duration = np.random.lognormal(mean=2.0, sigma=0.5)  # seconds
            self._agent_durations.append(duration)
            
            # Keep only recent durations (last hour)
            if len(self._agent_durations) > 3600:
                self._agent_durations = self._agent_durations[-3600:]
            
            point = TimeSeriesPoint(
                timestamp=timestamp,
                value=duration,
                metric_type=MetricType.AGENT_DURATION
            )
            await self.storage.add_point(point)
    
    async def _collect_channel_metrics(self, timestamp: datetime):
        """Collect channel activity metrics"""
        channels = self.channel_manager.get_all_channels()
        active_channels = 0
        
        for channel_id, channel in channels.items():
            # Check if channel has been active recently
            # In production, check last message timestamp
            is_active = np.random.random() < 0.7  # 70% chance channel is active
            if is_active:
                active_channels += 1
        
        point = TimeSeriesPoint(
            timestamp=timestamp,
            value=active_channels,
            metric_type=MetricType.CHANNEL_ACTIVITY
        )
        await self.storage.add_point(point)
    
    async def _collect_system_metrics(self, timestamp: datetime):
        """Collect system resource metrics"""
        # Simulated system metrics - in production, use psutil or similar
        cpu_util = np.random.uniform(20, 80)  # 20-80% CPU
        memory_util = np.random.uniform(30, 70)  # 30-70% memory
        
        cpu_point = TimeSeriesPoint(
            timestamp=timestamp,
            value=cpu_util,
            metric_type=MetricType.CPU_UTILIZATION
        )
        memory_point = TimeSeriesPoint(
            timestamp=timestamp,
            value=memory_util,
            metric_type=MetricType.MEMORY_UTILIZATION
        )
        
        await self.storage.add_point(cpu_point)
        await self.storage.add_point(memory_point)
    
    async def _generate_predictions(self) -> Optional[ScalingPrediction]:
        """Generate load predictions using the LSTM model"""
        try:
            # Get recent message arrival data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=2)  # Last 2 hours of data
            
            message_points = await self.storage.get_points(
                MetricType.MESSAGE_ARRIVAL, start_time, end_time
            )
            
            if len(message_points) < 60:  # Need at least 60 minutes of data
                logger.warning("Insufficient data for prediction")
                return None
            
            # Extract values and timestamps
            values = np.array([p.value for p in message_points])
            
            # Train model if not trained or periodically retrain
            if not self.predictor.is_trained or np.random.random() < 0.01:  # 1% chance to retrain
                logger.info("Training LSTM model...")
                self.predictor.train(values, epochs=20)
            
            # Make prediction
            predicted_load, confidence = self.predictor.predict(values)
            
            # Calculate recommended replicas based on predicted load
            # Assuming each replica can handle 10 messages per minute
            messages_per_replica = 10
            recommended_replicas = max(1, int(np.ceil(predicted_load / messages_per_replica)))
            
            # Create prediction object
            prediction = ScalingPrediction(
                timestamp=datetime.utcnow(),
                predicted_load=predicted_load,
                confidence=confidence,
                recommended_replicas=recommended_replicas,
                horizon_minutes=self.predictor.prediction_horizon,
                model_version=self.predictor.model_version
            )
            
            logger.info(f"Prediction: {predicted_load:.1f} msg/min, "
                       f"confidence: {confidence:.2f}, "
                       f"recommended replicas: {recommended_replicas}")
            
            # Store prediction
            await self._store_prediction(prediction)
            
            # Trigger scaling if confidence is high enough
            if confidence > 0.7:
                await self._trigger_scaling(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None
    
    async def _store_prediction(self, prediction: ScalingPrediction):
        """Store prediction for analysis and monitoring"""
        # In production, store in database or time-series store
        prediction_data = {
            "timestamp": prediction.timestamp.isoformat(),
            "predicted_load": prediction.predicted_load,
            "confidence": prediction.confidence,
            "recommended_replicas": prediction.recommended_replicas,
            "horizon_minutes": prediction.horizon_minutes,
            "model_version": prediction.model_version
        }
        
        # For now, just log it
        logger.info(f"Stored prediction: {json.dumps(prediction_data)}")
    
    async def _trigger_scaling(self, prediction: ScalingPrediction):
        """Trigger auto-scaling based on prediction"""
        # This would integrate with Kubernetes HPA or cloud auto-scaling
        logger.info(f"Triggering scaling to {prediction.recommended_replicas} replicas")
        
        # Example integration points:
        # 1. Kubernetes: Update HPA min replicas
        # 2. AWS: Update Auto Scaling Group desired capacity
        # 3. Custom: Send webhook to scaling service
        
        # For demonstration, we'll just log the action
        scaling_action = {
            "action": "scale",
            "target_replicas": prediction.recommended_replicas,
            "reason": f"Predicted load: {prediction.predicted_load:.1f} msg/min",
            "confidence": prediction.confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Scaling action: {json.dumps(scaling_action)}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary for monitoring"""
        current_time = datetime.utcnow()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # Get recent message rate
        message_points = await self.storage.get_points(
            MetricType.MESSAGE_ARRIVAL, one_hour_ago, current_time
        )
        
        if message_points:
            recent_values = [p.value for p in message_points[-10:]]  # Last 10 points
            avg_message_rate = np.mean(recent_values) if recent_values else 0
        else:
            avg_message_rate = 0
        
        # Get latest prediction
        predictions = []
        # In production, retrieve from storage
        
        return {
            "timestamp": current_time.isoformat(),
            "avg_message_rate_per_min": float(avg_message_rate),
            "active_channels": len(self.channel_manager.get_all_channels()),
            "model_trained": self.predictor.is_trained,
            "model_version": self.predictor.model_version,
            "collection_interval_seconds": self.collection_interval,
            "predictions": predictions[-5:] if predictions else []  # Last 5 predictions
        }
    
    async def get_historical_metrics(self, 
                                   metric_type: MetricType,
                                   hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics for analysis"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        points = await self.storage.get_points(metric_type, start_time, end_time)
        
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "value": p.value,
                "metadata": p.metadata
            }
            for p in points
        ]
    
    async def force_retrain(self):
        """Force retraining of the prediction model"""
        logger.info("Forcing model retraining...")
        
        # Get all available data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)  # Use up to 7 days of data
        
        message_points = await self.storage.get_points(
            MetricType.MESSAGE_ARRIVAL, start_time, end_time
        )
        
        if len(message_points) >= 100:  # Minimum data requirement
            values = np.array([p.value for p in message_points])
            self.predictor.train(values, epochs=50)
            logger.info("Model retrained successfully")
        else:
            logger.warning(f"Insufficient data for retraining: {len(message_points)} points")

# Factory function for easy integration
def create_metrics_collector(message_bus: MessageBus, 
                           channel_manager: ChannelManager) -> MetricsCollector:
    """Create and configure a MetricsCollector instance"""
    return MetricsCollector(
        message_bus=message_bus,
        channel_manager=channel_manager,
        collection_interval=60  # Collect every minute
    )

# Example usage and integration with existing code
async def example_integration():
    """Example of how to integrate with existing forge components"""
    # This would be called from your main application startup
    
    # Import existing components
    from backend.app.channels.message_bus import get_message_bus
    from backend.app.channels.manager import get_channel_manager
    
    # Get instances
    message_bus = get_message_bus()
    channel_manager = get_channel_manager()
    
    # Create metrics collector
    collector = create_metrics_collector(message_bus, channel_manager)
    
    # Start collection
    await collector.start()
    
    # Example: Get current metrics
    metrics = await collector.get_current_metrics()
    print(f"Current metrics: {metrics}")
    
    # Example: Get historical data
    historical = await collector.get_historical_metrics(
        MetricType.MESSAGE_ARRIVAL, hours=24
    )
    print(f"Historical data points: {len(historical)}")
    
    return collector

# Health check endpoint data
async def get_scaling_health() -> Dict[str, Any]:
    """Get health status for monitoring"""
    return {
        "service": "metrics_collector",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "lstm_available": HAS_TF,
            "sklearn_available": HAS_SKLEARN,
            "prediction_horizon_minutes": 15,
            "collection_interval_seconds": 60
        }
    }