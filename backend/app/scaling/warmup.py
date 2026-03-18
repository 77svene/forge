import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from ..gateway.config import settings
from ..channels.manager import ChannelManager
from ..channels.message_bus import MessageBus

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class WorkloadMetrics:
    timestamp: datetime
    message_count: int
    avg_processing_time: float
    active_channels: int
    queue_depth: int
    agent_utilization: float
    channel_type: Optional[str] = None


@dataclass
class ScalingPrediction:
    timestamp: datetime
    predicted_load: float
    confidence: float
    recommended_replicas: int
    horizon_minutes: int


class LSTMWorkloadPredictor:
    """Lightweight LSTM model for predicting agent workload."""
    
    def __init__(self, model_path: str = "models/workload_predictor.h5", 
                 scaler_path: str = "models/workload_scaler.pkl"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # 60 data points (1 hour at 1-minute intervals)
        self.feature_columns = [
            'message_count', 'avg_processing_time', 'active_channels',
            'queue_depth', 'agent_utilization'
        ]
        
        # Create model directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize model
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or create a new one."""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = load_model(str(self.model_path))
                self.scaler = joblib.load(str(self.scaler_path))
                logger.info("Loaded existing LSTM model and scaler")
            else:
                self._build_model()
                logger.info("Initialized new LSTM model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._build_model()
    
    def _build_model(self):
        """Build a lightweight LSTM model."""
        self.model = Sequential([
            LSTM(32, return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def prepare_sequences(self, data: pd.DataFrame) -> tuple:
        """Prepare sequences for LSTM training."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            # Predict total load (message_count * avg_processing_time)
            load = (data['message_count'].iloc[i + self.sequence_length] * 
                   data['avg_processing_time'].iloc[i + self.sequence_length])
            y.append(load)
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data: pd.DataFrame, epochs: int = 50, 
              batch_size: int = 32, validation_split: float = 0.2):
        """Train the LSTM model on historical data."""
        if len(historical_data) < self.sequence_length + 10:
            logger.warning("Insufficient data for training")
            return
        
        try:
            X, y = self.prepare_sequences(historical_data)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Save model and scaler
            self.model.save(str(self.model_path))
            joblib.dump(self.scaler, str(self.scaler_path))
            
            logger.info(f"Model trained. Final loss: {history.history['loss'][-1]:.4f}")
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def predict(self, recent_data: pd.DataFrame, horizon_minutes: int = 10) -> float:
        """Predict workload for the next horizon_minutes."""
        if len(recent_data) < self.sequence_length:
            logger.warning("Insufficient recent data for prediction")
            return 0.0
        
        try:
            # Prepare the last sequence
            last_sequence = recent_data[self.feature_columns].tail(self.sequence_length)
            scaled_sequence = self.scaler.transform(last_sequence)
            scaled_sequence = scaled_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Make prediction
            prediction = self.model.predict(scaled_sequence, verbose=0)[0][0]
            
            # Inverse transform to get actual load value
            # Create dummy array for inverse transform
            dummy = np.zeros((1, len(self.feature_columns)))
            dummy[0, 0] = prediction  # Assume prediction correlates with message_count
            actual_prediction = self.scaler.inverse_transform(dummy)[0, 0]
            
            return max(0, actual_prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0


class PredictiveAutoScaler:
    """Predictive auto-scaling for agent pools."""
    
    def __init__(self, channel_manager: ChannelManager, 
                 message_bus: MessageBus,
                 mode: ScalingMode = ScalingMode.HYBRID):
        self.channel_manager = channel_manager
        self.message_bus = message_bus
        self.mode = mode
        
        # Configuration
        self.collection_interval = 60  # seconds
        self.prediction_interval = 300  # 5 minutes
        self.training_interval = 3600  # 1 hour
        self.min_replicas = 2
        self.max_replicas = 50
        self.target_utilization = 0.7
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        
        # Data storage
        self.metrics_history: List[WorkloadMetrics] = []
        self.predictions: List[ScalingPrediction] = []
        self.max_history_hours = 168  # 1 week
        
        # LSTM predictor
        self.predictor = LSTMWorkloadPredictor()
        
        # Current state
        self.current_replicas = self.min_replicas
        self.last_training_time = datetime.utcnow()
        self.last_prediction_time = datetime.utcnow()
        self.is_running = False
        
        # Metrics
        self.scaling_events = []
        
        logger.info(f"PredictiveAutoScaler initialized in {mode.value} mode")
    
    async def start(self):
        """Start the auto-scaling background tasks."""
        self.is_running = True
        logger.info("Starting PredictiveAutoScaler")
        
        # Start background tasks
        asyncio.create_task(self._collect_metrics_loop())
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._training_loop())
    
    async def stop(self):
        """Stop the auto-scaling background tasks."""
        self.is_running = False
        logger.info("Stopping PredictiveAutoScaler")
    
    async def _collect_metrics_loop(self):
        """Periodically collect workload metrics."""
        while self.is_running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)
    
    async def _prediction_loop(self):
        """Periodically make predictions and adjust scaling."""
        while self.is_running:
            try:
                if self.mode in [ScalingMode.PREDICTIVE, ScalingMode.HYBRID]:
                    await self._make_predictions()
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(30)
    
    async def _training_loop(self):
        """Periodically retrain the LSTM model."""
        while self.is_running:
            try:
                if (datetime.utcnow() - self.last_training_time).seconds >= self.training_interval:
                    await self._train_model()
                    self.last_training_time = datetime.utcnow()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_current_metrics(self):
        """Collect current workload metrics from the system."""
        try:
            # Get metrics from channel manager
            channel_stats = await self.channel_manager.get_channel_stats()
            
            # Get message bus statistics
            bus_stats = await self.message_bus.get_stats()
            
            # Calculate metrics
            total_messages = sum(stat.get('message_count', 0) for stat in channel_stats.values())
            avg_processing_time = np.mean([
                stat.get('avg_processing_time', 0) 
                for stat in channel_stats.values() 
                if stat.get('avg_processing_time', 0) > 0
            ]) if channel_stats else 0
            
            active_channels = len([
                ch for ch, stat in channel_stats.items() 
                if stat.get('is_active', False)
            ])
            
            queue_depth = bus_stats.get('queue_size', 0)
            
            # Calculate agent utilization (simplified)
            agent_utilization = min(1.0, total_messages / (self.current_replicas * 100))
            
            metrics = WorkloadMetrics(
                timestamp=datetime.utcnow(),
                message_count=total_messages,
                avg_processing_time=avg_processing_time,
                active_channels=active_channels,
                queue_depth=queue_depth,
                agent_utilization=agent_utilization
            )
            
            self.metrics_history.append(metrics)
            
            # Trim history
            cutoff_time = datetime.utcnow() - timedelta(hours=self.max_history_hours)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            # Log metrics periodically
            if len(self.metrics_history) % 10 == 0:
                logger.debug(f"Collected metrics: {metrics}")
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _make_predictions(self):
        """Make load predictions and adjust scaling."""
        if len(self.metrics_history) < self.predictor.sequence_length:
            logger.debug("Insufficient data for prediction")
            return
        
        try:
            # Convert metrics to DataFrame
            metrics_df = self._metrics_to_dataframe()
            
            # Make prediction for next 10 minutes
            predicted_load = self.predictor.predict(metrics_df, horizon_minutes=10)
            
            # Calculate confidence based on recent prediction accuracy
            confidence = self._calculate_confidence()
            
            # Determine required replicas
            required_replicas = self._calculate_required_replicas(
                predicted_load, confidence
            )
            
            # Create prediction record
            prediction = ScalingPrediction(
                timestamp=datetime.utcnow(),
                predicted_load=predicted_load,
                confidence=confidence,
                recommended_replicas=required_replicas,
                horizon_minutes=10
            )
            
            self.predictions.append(prediction)
            
            # Apply scaling if needed
            if self.mode == ScalingMode.PREDICTIVE:
                await self._apply_scaling(required_replicas)
            elif self.mode == ScalingMode.HYBRID:
                # Use both predictive and reactive scaling
                reactive_replicas = self._calculate_reactive_scaling()
                final_replicas = max(required_replicas, reactive_replicas)
                await self._apply_scaling(final_replicas)
            
            logger.info(
                f"Prediction: load={predicted_load:.2f}, "
                f"confidence={confidence:.2f}, "
                f"replicas={required_replicas}"
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    
    def _metrics_to_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame."""
        data = []
        for metrics in self.metrics_history:
            data.append({
                'timestamp': metrics.timestamp,
                'message_count': metrics.message_count,
                'avg_processing_time': metrics.avg_processing_time,
                'active_channels': metrics.active_channels,
                'queue_depth': metrics.queue_depth,
                'agent_utilization': metrics.agent_utilization
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_confidence(self) -> float:
        """Calculate prediction confidence based on recent accuracy."""
        if len(self.predictions) < 5:
            return 0.5  # Default confidence
        
        # Calculate MAPE of last 5 predictions
        recent_predictions = self.predictions[-5:]
        errors = []
        
        for pred in recent_predictions:
            # Find actual load around prediction time
            actual_load = self._get_actual_load_at_time(pred.timestamp)
            if actual_load > 0:
                error = abs(pred.predicted_load - actual_load) / actual_load
                errors.append(error)
        
        if not errors:
            return 0.5
        
        mape = np.mean(errors)
        confidence = max(0.1, 1.0 - mape)
        return confidence
    
    def _get_actual_load_at_time(self, timestamp: datetime) -> float:
        """Get actual load at a specific timestamp."""
        # Find metrics closest to the timestamp
        closest_metrics = min(
            self.metrics_history,
            key=lambda m: abs((m.timestamp - timestamp).total_seconds())
        )
        
        return (closest_metrics.message_count * 
                closest_metrics.avg_processing_time)
    
    def _calculate_required_replicas(self, predicted_load: float, 
                                   confidence: float) -> int:
        """Calculate required number of replicas based on predicted load."""
        # Apply confidence weighting
        weighted_load = predicted_load * confidence
        
        # Calculate required capacity
        # Assuming each replica can handle 100 units of load per minute
        capacity_per_replica = 100 * self.target_utilization
        required_replicas = int(np.ceil(weighted_load / capacity_per_replica))
        
        # Apply min/max constraints
        required_replicas = max(self.min_replicas, 
                              min(self.max_replicas, required_replicas))
        
        return required_replicas
    
    def _calculate_reactive_scaling(self) -> int:
        """Calculate scaling based on current reactive metrics."""
        if not self.metrics_history:
            return self.current_replicas
        
        latest_metrics = self.metrics_history[-1]
        
        # Scale up if utilization is high
        if latest_metrics.agent_utilization > self.scale_up_threshold:
            new_replicas = min(
                self.max_replicas,
                int(self.current_replicas * 1.5)  # Scale up by 50%
            )
            return new_replicas
        
        # Scale down if utilization is low
        elif (latest_metrics.agent_utilization < self.scale_down_threshold and 
              self.current_replicas > self.min_replicas):
            new_replicas = max(
                self.min_replicas,
                int(self.current_replicas * 0.7)  # Scale down by 30%
            )
            return new_replicas
        
        return self.current_replicas
    
    async def _apply_scaling(self, target_replicas: int):
        """Apply scaling changes to the system."""
        if target_replicas == self.current_replicas:
            return
        
        try:
            # In production, this would call Kubernetes HPA or cloud scaling API
            # For now, we'll simulate the scaling
            
            logger.info(
                f"Scaling from {self.current_replicas} to {target_replicas} replicas"
            )
            
            # Record scaling event
            self.scaling_events.append({
                'timestamp': datetime.utcnow(),
                'from_replicas': self.current_replicas,
                'to_replicas': target_replicas,
                'reason': 'predictive_scaling'
            })
            
            # Update current replicas
            self.current_replicas = target_replicas
            
            # In production, you would call:
            # await self._update_kubernetes_hpa(target_replicas)
            # or
            # await self._update_cloud_autoscaling(target_replicas)
            
        except Exception as e:
            logger.error(f"Failed to apply scaling: {e}")
    
    async def _train_model(self):
        """Train the LSTM model on collected data."""
        if len(self.metrics_history) < 1000:  # Need sufficient data
            logger.info("Insufficient data for model training")
            return
        
        try:
            logger.info("Starting LSTM model training")
            
            # Convert to DataFrame
            df = self._metrics_to_dataframe()
            
            # Train model
            history = self.predictor.train(df, epochs=30, batch_size=32)
            
            if history:
                logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get current scaling statistics."""
        return {
            'current_replicas': self.current_replicas,
            'mode': self.mode.value,
            'metrics_history_size': len(self.metrics_history),
            'predictions_count': len(self.predictions),
            'scaling_events': len(self.scaling_events),
            'last_training': self.last_training_time.isoformat(),
            'last_prediction': self.last_prediction_time.isoformat(),
            'model_trained': self.predictor.model is not None,
            'recent_predictions': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'predicted_load': p.predicted_load,
                    'confidence': p.confidence,
                    'recommended_replicas': p.recommended_replicas
                }
                for p in self.predictions[-5:]
            ]
        }
    
    def update_config(self, **kwargs):
        """Update scaler configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated config: {key} = {value}")


# Singleton instance for easy import
_auto_scaler_instance: Optional[PredictiveAutoScaler] = None


async def initialize_auto_scaler(channel_manager: ChannelManager,
                               message_bus: MessageBus,
                               mode: ScalingMode = ScalingMode.HYBRID) -> PredictiveAutoScaler:
    """Initialize and start the predictive auto-scaler."""
    global _auto_scaler_instance
    
    if _auto_scaler_instance is None:
        _auto_scaler_instance = PredictiveAutoScaler(
            channel_manager=channel_manager,
            message_bus=message_bus,
            mode=mode
        )
        await _auto_scaler_instance.start()
    
    return _auto_scaler_instance


async def get_auto_scaler() -> Optional[PredictiveAutoScaler]:
    """Get the singleton auto-scaler instance."""
    return _auto_scaler_instance


async def shutdown_auto_scaler():
    """Shutdown the auto-scaler."""
    global _auto_scaler_instance
    
    if _auto_scaler_instance:
        await _auto_scaler_instance.stop()
        _auto_scaler_instance = None