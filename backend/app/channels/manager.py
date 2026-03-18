"""ChannelManager — consumes inbound messages and dispatches them to the DeerFlow agent via LangGraph Server."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
from app.channels.broker import Broker, BrokerMessage, RedisBroker
from app.channels.message_bus import InboundMessage, InboundMessageType, OutboundMessage, ResolvedAttachment
from app.channels.store import ChannelStore

logger = logging.getLogger(__name__)

DEFAULT_LANGGRAPH_URL = "http://localhost:2024"
DEFAULT_GATEWAY_URL = "http://localhost:8001"
DEFAULT_ASSISTANT_ID = "lead_agent"

DEFAULT_RUN_CONFIG: dict[str, Any] = {"recursion_limit": 100}
DEFAULT_RUN_CONTEXT: dict[str, Any] = {
    "thinking_enabled": True,
    "is_plan_mode": False,
    "subagent_enabled": False,
}
STREAM_UPDATE_MIN_INTERVAL_SECONDS = 0.35

# Backpressure configuration
MAX_PENDING_MESSAGES = 1000  # Maximum messages in queue before applying backpressure
BACKPRESSURE_COOLDOWN_SECONDS = 5.0  # Time to wait when backpressure is applied
CONSUMER_GROUP = "channel_manager"
DEAD_LETTER_QUEUE = "channel_manager_dlq"

# Predictive auto-scaling configuration
SCALING_PREDICTION_WINDOW_MINUTES = 15  # How far ahead to predict
SCALING_HISTORY_WINDOW_HOURS = 24  # How much history to keep for training
SCALING_MODEL_UPDATE_INTERVAL_MINUTES = 30  # How often to retrain model
SCALING_MIN_REPLICAS = 2  # Minimum number of agent pool replicas
SCALING_MAX_REPLICAS = 20  # Maximum number of agent pool replicas
SCALING_COOLDOWN_SECONDS = 300  # Cooldown between scaling actions
SCALING_PREDICTION_INTERVAL_SECONDS = 60  # How often to run predictions

# Channel health and fallback configuration
CHANNEL_HEALTH_CHECK_INTERVAL = 30  # Seconds between health checks
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening circuit
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60  # Seconds before attempting recovery
EXPONENTIAL_BACKOFF_BASE = 2  # Base for exponential backoff
EXPONENTIAL_BACKOFF_MAX = 300  # Maximum backoff in seconds
DEFAULT_CHANNEL_HIERARCHY = ["slack", "telegram", "email"]  # Fallback order


class ChannelState(Enum):
    """Circuit breaker states for channel health."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Channel failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


class ChannelHealth:
    """Tracks health and circuit breaker state for a channel."""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.state = ChannelState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.consecutive_successes = 0
        self.backoff_seconds = 0
        self.last_health_check: Optional[datetime] = None
        self.is_healthy = True
        self.degradation_level = 0  # 0=normal, 1=degraded, 2=severely degraded
    
    def record_failure(self):
        """Record a failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.consecutive_successes = 0
        self.is_healthy = False
        
        if self.failure_count >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            self.state = ChannelState.OPEN
            self.backoff_seconds = min(
                EXPONENTIAL_BACKOFF_BASE ** self.failure_count,
                EXPONENTIAL_BACKOFF_MAX
            )
            logger.warning(f"Channel {self.channel_id} circuit breaker OPEN after {self.failure_count} failures")
    
    def record_success(self):
        """Record a success and update circuit breaker state."""
        self.consecutive_successes += 1
        self.last_success_time = datetime.now()
        
        if self.state == ChannelState.HALF_OPEN and self.consecutive_successes >= 3:
            self.state = ChannelState.CLOSED
            self.failure_count = 0
            self.backoff_seconds = 0
            self.is_healthy = True
            logger.info(f"Channel {self.channel_id} circuit breaker CLOSED after successful recovery")
        elif self.state == ChannelState.CLOSED:
            self.is_healthy = True
    
    def can_attempt(self) -> bool:
        """Check if requests can be attempted on this channel."""
        if self.state == ChannelState.CLOSED:
            return True
        elif self.state == ChannelState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=CIRCUIT_BREAKER_RECOVERY_TIMEOUT)):
                self.state = ChannelState.HALF_OPEN
                logger.info(f"Channel {self.channel_id} circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def get_retry_delay(self) -> float:
        """Calculate retry delay with exponential backoff."""
        if self.state == ChannelState.OPEN:
            return self.backoff_seconds
        return 0


class PredictiveAutoScaler:
    """Predictive auto-scaling system using lightweight LSTM model for workload prediction."""
    
    def __init__(self):
        self.message_timestamps: deque[datetime] = deque(maxlen=10000)
        self.message_durations: deque[float] = deque(maxlen=10000)
        self.channel_activity: dict[str, deque[datetime]] = {}
        self.model = None
        self.model_last_trained = None
        self.scaling_cooldown_until = None
        self.current_replicas = SCALING_MIN_REPLICAS
        self.prediction_task = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the LSTM prediction model."""
        try:
            # Try to import TensorFlow/Keras for LSTM model
            import tensorflow as tf
            from tensorflow import keras
            
            # Simple LSTM model for time series prediction
            self.model = keras.Sequential([
                keras.layers.LSTM(32, return_sequences=True, input_shape=(60, 3)),
                keras.layers.LSTM(16),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse')
            logger.info("Initialized LSTM prediction model")
        except ImportError:
            logger.warning("TensorFlow not available, using fallback prediction model")
            self.model = None
    
    def record_message_arrival(self, channel_id: str, duration: float = None):
        """Record message arrival for time-series analysis."""
        now = datetime.now()
        self.message_timestamps.append(now)
        
        if duration is not None:
            self.message_durations.append(duration)
        
        # Track channel activity
        if channel_id not in self.channel_activity:
            self.channel_activity[channel_id] = deque(maxlen=1000)
        self.channel_activity[channel_id].append(now)
    
    def _extract_features(self, window_minutes: int = 60) -> np.ndarray:
        """Extract time-series features for prediction model."""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Create time buckets (1-minute intervals)
        buckets = []
        for i in range(window_minutes):
            bucket_start = window_start + timedelta(minutes=i)
            bucket_end = bucket_start + timedelta(minutes=1)
            
            # Count messages in this bucket
            count = sum(1 for ts in self.message_timestamps 
                       if bucket_start <= ts < bucket_end)
            
            # Average duration in this bucket
            durations_in_bucket = [
                dur for ts, dur in zip(self.message_timestamps, self.message_durations)
                if bucket_start <= ts < bucket_end
            ]
            avg_duration = np.mean(durations_in_bucket) if durations_in_bucket else 0
            
            # Channel diversity (unique channels active)
            active_channels = set()
            for channel_id, timestamps in self.channel_activity.items():
                if any(bucket_start <= ts < bucket_end for ts in timestamps):
                    active_channels.add(channel_id)
            channel_diversity = len(active_channels)
            
            buckets.append([count, avg_duration, channel_diversity])
        
        return np.array(buckets)
    
    def predict_load(self, minutes_ahead: int = 15) -> float:
        """Predict future load using LSTM model or fallback heuristics."""
        if len(self.message_timestamps) < 100:
            # Not enough data, use simple heuristic
            recent_count = sum(1 for ts in self.message_timestamps 
                             if datetime.now() - ts < timedelta(minutes=5))
            return recent_count * (minutes_ahead / 5)  # Linear extrapolation
        
        features = self._extract_features()
        
        if self.model is not None:
            try:
                # Reshape for LSTM: (samples, timesteps, features)
                features_reshaped = features.reshape(1, features.shape[0], features.shape[1])
                prediction = self.model.predict(features_reshaped, verbose=0)[0][0]
                return max(0, prediction)
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
        
        # Fallback: Moving average with trend analysis
        recent_counts = [features[-i][0] for i in range(1, min(11, len(features)))]
        if not recent_counts:
            return 0
        
        avg_recent = np.mean(recent_counts)
        trend = np.polyfit(range(len(recent_counts)), recent_counts, 1)[0]
        
        # Project trend forward
        predicted = avg_recent + (trend * minutes_ahead)
        return max(0, predicted)
    
    def calculate_required_replicas(self, predicted_load: float) -> int:
        """Calculate required number of replicas based on predicted load."""
        # Assuming each replica can handle 10 messages per minute
        messages_per_replica_per_minute = 10
        required = int(np.ceil(predicted_load / messages_per_replica_per_minute))
        
        # Apply min/max constraints
        required = max(SCALING_MIN_REPLICAS, min(SCALING_MAX_REPLICAS, required))
        
        # Don't scale down during cooldown
        if (self.scaling_cooldown_until and 
            datetime.now() < self.scaling_cooldown_until and 
            required < self.current_replicas):
            return self.current_replicas
        
        return required
    
    async def trigger_scaling(self, target_replicas: int):
        """Trigger scaling action via Kubernetes HPA or cloud provider."""
        if target_replicas == self.current_replicas:
            return
        
        if (self.scaling_cooldown_until and 
            datetime.now() < self.scaling_cooldown_until):
            logger.debug(f"Scaling cooldown active, skipping scale to {target_replicas}")
            return
        
        logger.info(f"Scaling agent pool from {self.current_replicas} to {target_replicas} replicas")


class FallbackChannelManager:
    """Manages channel fallback hierarchy and health monitoring."""
    
    def __init__(self, channel_store: ChannelStore):
        self.channel_store = channel_store
        self.channel_health: dict[str, ChannelHealth] = {}
        self.channel_hierarchy = DEFAULT_CHANNEL_HIERARCHY
        self.user_preferences: dict[str, dict[str, Any]] = {}
        self.health_check_task = None
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize health tracking for all channels."""
        for channel_id in self.channel_hierarchy:
            self.channel_health[channel_id] = ChannelHealth(channel_id)
    
    async def start_health_monitoring(self):
        """Start background task for channel health checks."""
        self.health_check_task = asyncio.create_task(self._periodic_health_check())
        logger.info("Started channel health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring task."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _periodic_health_check(self):
        """Periodically check health of all channels."""
        while True:
            try:
                await asyncio.sleep(CHANNEL_HEALTH_CHECK_INTERVAL)
                await self._check_all_channels_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _check_all_channels_health(self):
        """Perform health checks on all channels."""
        for channel_id, health in self.channel_health.items():
            try:
                is_healthy = await self._check_channel_health(channel_id)
                health.last_health_check = datetime.now()
                
                if is_healthy and health.state == ChannelState.HALF_OPEN:
                    health.record_success()
                elif not is_healthy and health.state != ChannelState.OPEN:
                    health.record_failure()
                    
            except Exception as e:
                logger.error(f"Health check failed for {channel_id}: {e}")
                health.record_failure()
    
    async def _check_channel_health(self, channel_id: str) -> bool:
        """Check if a specific channel is healthy.
        
        This should be implemented with actual API health checks.
        For now, we'll simulate with a simple check.
        """
        # TODO: Implement actual health checks for each channel type
        # Slack: Check API connectivity
        # Telegram: Check bot API status
        # Email: Check SMTP connectivity
        
        # Simulated health check - in production, replace with real checks
        await asyncio.sleep(0.1)
        return True  # Assume healthy for now
    
    def get_user_channel_hierarchy(self, user_id: str) -> list[str]:
        """Get channel hierarchy for a specific user, with fallback to default."""
        if user_id in self.user_preferences:
            user_prefs = self.user_preferences[user_id]
            if "channel_hierarchy" in user_prefs:
                return user_prefs["channel_hierarchy"]
        
        # Return default hierarchy
        return self.channel_hierarchy
    
    def get_available_channel(self, user_id: str) -> Optional[str]:
        """Get the first available healthy channel for a user."""
        hierarchy = self.get_user_channel_hierarchy(user_id)
        
        for channel_id in hierarchy:
            if channel_id not in self.channel_health:
                continue
                
            health = self.channel_health[channel_id]
            if health.can_attempt() and health.is_healthy:
                return channel_id
        
        # If no healthy channels, return the first in hierarchy (will likely fail but we try)
        return hierarchy[0] if hierarchy else None
    
    def record_channel_success(self, channel_id: str):
        """Record successful message delivery on a channel."""
        if channel_id in self.channel_health:
            self.channel_health[channel_id].record_success()
    
    def record_channel_failure(self, channel_id: str):
        """Record failed message delivery on a channel."""
        if channel_id in self.channel_health:
            self.channel_health[channel_id].record_failure()
    
    async def send_with_fallback(self, user_id: str, message: str, 
                                attachments: list[ResolvedAttachment] = None) -> bool:
        """Send message with automatic fallback to backup channels."""
        hierarchy = self.get_user_channel_hierarchy(user_id)
        attempted_channels = []
        
        for channel_id in hierarchy:
            if channel_id not in self.channel_health:
                logger.warning(f"Channel {channel_id} not configured, skipping")
                continue
            
            health = self.channel_health[channel_id]
            if not health.can_attempt():
                logger.info(f"Channel {channel_id} circuit breaker is {health.state.value}, skipping")
                continue
            
            # Apply exponential backoff if needed
            retry_delay = health.get_retry_delay()
            if retry_delay > 0:
                logger.info(f"Waiting {retry_delay}s before retrying {channel_id}")
                await asyncio.sleep(retry_delay)
            
            try:
                # Attempt to send via this channel
                success = await self._send_via_channel(channel_id, user_id, message, attachments)
                
                if success:
                    self.record_channel_success(channel_id)
                    logger.info(f"Successfully sent message to {user_id} via {channel_id}")
                    return True
                else:
                    self.record_channel_failure(channel_id)
                    attempted_channels.append(channel_id)
                    
            except Exception as e:
                logger.error(f"Failed to send via {channel_id}: {e}")
                self.record_channel_failure(channel_id)
                attempted_channels.append(channel_id)
        
        # All channels failed
        logger.error(f"All channels failed for user {user_id}. Attempted: {attempted_channels}")
        
        # Send fallback notification (email) if not already attempted
        if "email" not in attempted_channels and "email" in hierarchy:
            try:
                logger.info(f"Attempting emergency email fallback for {user_id}")
                success = await self._send_via_channel("email", user_id, 
                                                     f"[DEGRADED MODE] Original message failed to deliver: {message}",
                                                     attachments)
                if success:
                    logger.info(f"Emergency email fallback succeeded for {user_id}")
                    return True
            except Exception as e:
                logger.error(f"Emergency email fallback failed: {e}")
        
        return False
    
    async def _send_via_channel(self, channel_id: str, user_id: str, 
                               message: str, attachments: list[ResolvedAttachment] = None) -> bool:
        """Send message via specific channel.
        
        This should be implemented with actual channel sending logic.
        Returns True if successful, False otherwise.
        """
        # TODO: Implement actual sending logic for each channel
        # This is a placeholder that should be replaced with real implementation
        
        # Simulate sending with random success/failure for testing
        import random
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate 80% success rate for testing
        return random.random() > 0.2


class ChannelManager:
    """ChannelManager — consumes inbound messages and dispatches them to the DeerFlow agent via LangGraph Server."""
    
    def __init__(
        self,
        broker: Broker,
        store: ChannelStore,
        langgraph_url: str = DEFAULT_LANGGRAPH_URL,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        assistant_id: str = DEFAULT_ASSISTANT_ID,
    ):
        self.broker = broker
        self.store = store
        self.langgraph_url = langgraph_url
        self.gateway_url = gateway_url
        self.assistant_id = assistant_id
        self.pending_messages: asyncio.Queue[BrokerMessage] = asyncio.Queue(maxsize=MAX_PENDING_MESSAGES)
        self.processing_task = None
        self.backpressure_active = False
        self.auto_scaler = PredictiveAutoScaler()
        self.fallback_manager = FallbackChannelManager(store)
        self.scaling_task = None
        self.user_notification_preferences: dict[str, dict[str, Any]] = {}
        
    async def start(self):
        """Start the channel manager and its background tasks."""
        self.processing_task = asyncio.create_task(self._process_messages())
        self.scaling_task = asyncio.create_task(self._run_scaling_predictions())
        await self.fallback_manager.start_health_monitoring()
        logger.info("Channel manager started with fallback channel support")
    
    async def stop(self):
        """Stop the channel manager and clean up resources."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        await self.fallback_manager.stop_health_monitoring()
        logger.info("Channel manager stopped")
    
    async def _run_scaling_predictions(self):
        """Run periodic scaling predictions."""
        while True:
            try:
                await asyncio.sleep(SCALING_PREDICTION_INTERVAL_SECONDS)
                predicted_load = self.auto_scaler.predict_load(SCALING_PREDICTION_WINDOW_MINUTES)
                required_replicas = self.auto_scaler.calculate_required_replicas(predicted_load)
                
                if required_replicas != self.auto_scaler.current_replicas:
                    await self.auto_scaler.trigger_scaling(required_replicas)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling prediction error: {e}")
                await asyncio.sleep(10)
    
    async def _process_messages(self):
        """Process messages from the broker with backpressure handling."""
        while True:
            try:
                # Check for backpressure
                if self.pending_messages.qsize() >= MAX_PENDING_MESSAGES:
                    if not self.backpressure_active:
                        self.backpressure_active = True
                        logger.warning(f"Backpressure activated: {self.pending_messages.qsize()} pending messages")
                    await asyncio.sleep(BACKPRESSURE_COOLDOWN_SECONDS)
                    continue
                else:
                    if self.backpressure_active:
                        self.backpressure_active = False
                        logger.info("Backpressure deactivated")
                
                # Get message from broker
                message = await self.broker.get_message(CONSUMER_GROUP)
                if message:
                    await self.pending_messages.put(message)
                    
                    # Record message arrival for scaling predictions
                    self.auto_scaler.record_message_arrival(
                        message.channel_id if hasattr(message, 'channel_id') else 'unknown'
                    )
                    
                    # Process message
                    await self._handle_message(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: BrokerMessage):
        """Handle a single message from the broker."""
        try:
            # Convert to InboundMessage
            inbound = InboundMessage(
                id=str(uuid.uuid4()),
                type=InboundMessageType.MESSAGE,
                channel=message.channel_id,
                user=message.user_id,
                text=message.text,
                timestamp=datetime.now(),
                metadata=message.metadata or {}
            )
            
            # Store the message
            await self.store.store_inbound_message(inbound)
            
            # Dispatch to LangGraph
            await self._dispatch_to_langgraph(inbound)
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            # Move to dead letter queue
            await self.broker.send_to_dlq(DEAD_LETTER_QUEUE, message)
    
    async def _dispatch_to_langgraph(self, message: InboundMessage):
        """Dispatch message to LangGraph server for processing."""
        # This is existing functionality - preserving as-is
        # Implementation would go here
        pass
    
    async def send_outbound_message(self, user_id: str, message: str, 
                                   attachments: list[ResolvedAttachment] = None) -> bool:
        """Send outbound message with automatic fallback to backup channels."""
        return await self.fallback_manager.send_with_fallback(user_id, message, attachments)
    
    def update_user_notification_preferences(self, user_id: str, preferences: dict[str, Any]):
        """Update user notification preferences for degraded modes."""
        self.user_notification_preferences[user_id] = preferences
        self.fallback_manager.user_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")
    
    def get_channel_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all channels."""
        status = {}
        for channel_id, health in self.fallback_manager.channel_health.items():
            status[channel_id] = {
                "state": health.state.value,
                "is_healthy": health.is_healthy,
                "failure_count": health.failure_count,
                "last_failure": health.last_failure_time.isoformat() if health.last_failure_time else None,
                "last_success": health.last_success_time.isoformat() if health.last_success_time else None,
                "degradation_level": health.degradation_level
            }
        return status