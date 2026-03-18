# backend/app/resilience/channel_health.py

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import wraps

from ..channels.base import BaseChannel
from ..channels.manager import ChannelManager
from ..channels.message_bus import MessageBus
from ..channels.store import ChannelStore
from ..gateway.config import GatewayConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class ChannelHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: int = 60  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close circuit in half-open
    exponential_backoff_base: float = 2.0
    max_recovery_timeout: int = 3600  # 1 hour max


@dataclass
class ChannelHealthMetrics:
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    average_response_time: float = 0.0
    total_requests: int = 0


@dataclass
class FallbackChannel:
    channel_name: str
    priority: int  # Lower number = higher priority
    enabled: bool = True
    user_preference: Optional[str] = None  # User's preference for this channel


class CircuitBreaker:
    """Circuit breaker pattern implementation for channel resilience"""
    
    def __init__(self, channel_name: str, config: CircuitBreakerConfig):
        self.channel_name = channel_name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.recovery_timeout = config.recovery_timeout
        self._lock = asyncio.Lock()
        
    async def before_request(self) -> bool:
        """Check if request can proceed based on circuit state"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit for {self.channel_name} moved to HALF_OPEN")
                    return True
                return False
            return True
    
    async def record_success(self):
        """Record a successful request"""
        async with self._lock:
            self.failure_count = 0
            self.success_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    self.recovery_timeout = self.config.recovery_timeout
                    logger.info(f"Circuit for {self.channel_name} moved to CLOSED")
    
    async def record_failure(self):
        """Record a failed request"""
        async with self._lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit for {self.channel_name} moved to OPEN after {self.failure_count} failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                # If we fail in half-open, go back to open with increased timeout
                self.state = CircuitState.OPEN
                self.recovery_timeout = min(
                    self.recovery_timeout * self.config.exponential_backoff_base,
                    self.config.max_recovery_timeout
                )
                logger.warning(f"Circuit for {self.channel_name} moved back to OPEN from HALF_OPEN")
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if not self.last_failure_time:
            return True
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.recovery_timeout


class ChannelHealthMonitor:
    """Monitors health of all channels and manages fallback routing"""
    
    def __init__(self, channel_manager: ChannelManager, config: GatewayConfig):
        self.channel_manager = channel_manager
        self.config = config
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_metrics: Dict[str, ChannelHealthMetrics] = {}
        self.fallback_hierarchy: List[FallbackChannel] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30  # seconds
        self._initialized = False
        
    async def initialize(self):
        """Initialize the health monitor with all available channels"""
        if self._initialized:
            return
            
        # Initialize circuit breakers for all channels
        channels = self.channel_manager.get_all_channels()
        circuit_config = CircuitBreakerConfig(
            failure_threshold=getattr(self.config, 'circuit_failure_threshold', 5),
            recovery_timeout=getattr(self.config, 'circuit_recovery_timeout', 60)
        )
        
        for channel_name, channel in channels.items():
            self.circuit_breakers[channel_name] = CircuitBreaker(channel_name, circuit_config)
            self.health_metrics[channel_name] = ChannelHealthMetrics()
        
        # Set up default fallback hierarchy
        self._setup_default_fallback_hierarchy()
        
        # Start health monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_health())
        self._initialized = True
        logger.info(f"Channel health monitor initialized with {len(channels)} channels")
    
    def _setup_default_fallback_hierarchy(self):
        """Setup default fallback hierarchy: Slack -> Telegram -> Email"""
        self.fallback_hierarchy = [
            FallbackChannel(channel_name="slack", priority=1),
            FallbackChannel(channel_name="telegram", priority=2),
            FallbackChannel(channel_name="email", priority=3),
        ]
        
        # Add any other available channels with lower priority
        available_channels = set(self.channel_manager.get_all_channels().keys())
        default_channels = {"slack", "telegram", "email"}
        other_channels = available_channels - default_channels
        
        for i, channel_name in enumerate(sorted(other_channels), start=4):
            self.fallback_hierarchy.append(
                FallbackChannel(channel_name=channel_name, priority=i)
            )
    
    async def get_available_channel(self, user_id: str, preferred_channel: Optional[str] = None) -> Optional[BaseChannel]:
        """Get the best available channel for a user based on health and preferences"""
        await self.initialize()
        
        # Get user preferences if available
        user_prefs = self.user_preferences.get(user_id, {})
        user_fallback_order = user_prefs.get("fallback_order", [])
        
        # Build channel priority list
        channel_priority = []
        
        # 1. Try preferred channel first if specified
        if preferred_channel and self._is_channel_available(preferred_channel):
            channel_priority.append(preferred_channel)
        
        # 2. Add user's custom fallback order
        for channel_name in user_fallback_order:
            if channel_name not in channel_priority and self._is_channel_available(channel_name):
                channel_priority.append(channel_name)
        
        # 3. Add default fallback hierarchy
        for fallback in sorted(self.fallback_hierarchy, key=lambda x: x.priority):
            if fallback.channel_name not in channel_priority and fallback.enabled:
                if self._is_channel_available(fallback.channel_name):
                    channel_priority.append(fallback.channel_name)
        
        # Try channels in priority order
        for channel_name in channel_priority:
            circuit_breaker = self.circuit_breakers.get(channel_name)
            if circuit_breaker and await circuit_breaker.before_request():
                channel = self.channel_manager.get_channel(channel_name)
                if channel:
                    logger.debug(f"Selected channel {channel_name} for user {user_id}")
                    return channel
        
        logger.warning(f"No available channels for user {user_id}")
        return None
    
    def _is_channel_available(self, channel_name: str) -> bool:
        """Check if a channel exists and is enabled"""
        channel = self.channel_manager.get_channel(channel_name)
        if not channel:
            return False
        
        # Check if channel is enabled in configuration
        channel_config = getattr(self.config, f"{channel_name}_config", None)
        if channel_config and hasattr(channel_config, "enabled"):
            return channel_config.enabled
        
        return True
    
    async def send_with_fallback(self, user_id: str, message: str, context: Optional[Dict] = None) -> bool:
        """Send message with automatic fallback to backup channels"""
        await self.initialize()
        
        start_time = datetime.utcnow()
        preferred_channel = context.get("preferred_channel") if context else None
        
        # Try to get available channel
        channel = await self.get_available_channel(user_id, preferred_channel)
        if not channel:
            logger.error(f"Failed to send message to {user_id}: no available channels")
            return False
        
        channel_name = channel.__class__.__name__.lower().replace("channel", "")
        
        try:
            # Attempt to send message
            success = await channel.send(user_id, message, context)
            
            # Update metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_metrics(channel_name, success, response_time)
            
            if success:
                await self.circuit_breakers[channel_name].record_success()
                logger.info(f"Message sent successfully to {user_id} via {channel_name}")
                return True
            else:
                await self.circuit_breakers[channel_name].record_failure()
                logger.warning(f"Failed to send message to {user_id} via {channel_name}")
                
                # Try fallback channels
                return await self._try_fallback_channels(user_id, message, context, exclude_channel=channel_name)
                
        except Exception as e:
            logger.error(f"Error sending message via {channel_name}: {str(e)}")
            await self.circuit_breakers[channel_name].record_failure()
            await self._update_metrics(channel_name, False, 0)
            
            # Try fallback channels
            return await self._try_fallback_channels(user_id, message, context, exclude_channel=channel_name)
    
    async def _try_fallback_channels(self, user_id: str, message: str, context: Optional[Dict], exclude_channel: str) -> bool:
        """Try sending through fallback channels"""
        available_channels = [
            fb.channel_name for fb in sorted(self.fallback_hierarchy, key=lambda x: x.priority)
            if fb.channel_name != exclude_channel and fb.enabled
        ]
        
        for channel_name in available_channels:
            circuit_breaker = self.circuit_breakers.get(channel_name)
            if not circuit_breaker or not await circuit_breaker.before_request():
                continue
            
            channel = self.channel_manager.get_channel(channel_name)
            if not channel:
                continue
            
            try:
                start_time = datetime.utcnow()
                success = await channel.send(user_id, message, context)
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                await self._update_metrics(channel_name, success, response_time)
                
                if success:
                    await circuit_breaker.record_success()
                    logger.info(f"Fallback successful: sent to {user_id} via {channel_name}")
                    return True
                else:
                    await circuit_breaker.record_failure()
                    
            except Exception as e:
                logger.error(f"Fallback error via {channel_name}: {str(e)}")
                await circuit_breaker.record_failure()
        
        logger.error(f"All fallback channels failed for user {user_id}")
        return False
    
    async def _update_metrics(self, channel_name: str, success: bool, response_time: float):
        """Update health metrics for a channel"""
        if channel_name not in self.health_metrics:
            self.health_metrics[channel_name] = ChannelHealthMetrics()
        
        metrics = self.health_metrics[channel_name]
        metrics.total_requests += 1
        
        if success:
            metrics.success_count += 1
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
            metrics.last_success = datetime.utcnow()
        else:
            metrics.failure_count += 1
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            metrics.last_failure = datetime.utcnow()
        
        # Update average response time (exponential moving average)
        if metrics.total_requests == 1:
            metrics.average_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            metrics.average_response_time = (
                alpha * response_time + (1 - alpha) * metrics.average_response_time
            )
    
    async def _monitor_health(self):
        """Background task to monitor channel health"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all channels"""
        for channel_name, channel in self.channel_manager.get_all_channels().items():
            try:
                # Simple health check - try to get channel status
                if hasattr(channel, 'health_check'):
                    healthy = await channel.health_check()
                else:
                    # Fallback: try a simple operation
                    healthy = await self._basic_health_check(channel)
                
                circuit_breaker = self.circuit_breakers.get(channel_name)
                if circuit_breaker:
                    if healthy:
                        await circuit_breaker.record_success()
                    else:
                        await circuit_breaker.record_failure()
                        
            except Exception as e:
                logger.warning(f"Health check failed for {channel_name}: {str(e)}")
                circuit_breaker = self.circuit_breakers.get(channel_name)
                if circuit_breaker:
                    await circuit_breaker.record_failure()
    
    async def _basic_health_check(self, channel: BaseChannel) -> bool:
        """Basic health check for channels without dedicated health_check method"""
        try:
            # Try to get channel info or perform a lightweight operation
            if hasattr(channel, 'get_channel_info'):
                await channel.get_channel_info()
                return True
            return True
        except Exception:
            return False
    
    def get_channel_health_status(self, channel_name: str) -> ChannelHealthStatus:
        """Get health status for a specific channel"""
        circuit_breaker = self.circuit_breakers.get(channel_name)
        metrics = self.health_metrics.get(channel_name)
        
        if not circuit_breaker or not metrics:
            return ChannelHealthStatus.UNKNOWN
        
        if circuit_breaker.state == CircuitState.OPEN:
            return ChannelHealthStatus.UNHEALTHY
        elif circuit_breaker.state == CircuitState.HALF_OPEN:
            return ChannelHealthStatus.DEGRADED
        elif metrics.consecutive_failures > 0:
            return ChannelHealthStatus.DEGRADED
        else:
            return ChannelHealthStatus.HEALTHY
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all channels"""
        status = {}
        for channel_name in self.circuit_breakers.keys():
            circuit_breaker = self.circuit_breakers[channel_name]
            metrics = self.health_metrics.get(channel_name, ChannelHealthMetrics())
            
            status[channel_name] = {
                "status": self.get_channel_health_status(channel_name).value,
                "circuit_state": circuit_breaker.state.value,
                "success_rate": (
                    metrics.success_count / metrics.total_requests 
                    if metrics.total_requests > 0 else 0
                ),
                "average_response_time": metrics.average_response_time,
                "consecutive_failures": metrics.consecutive_failures,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
            }
        
        return status
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Set user notification preferences for degraded modes"""
        self.user_preferences[user_id] = preferences
        logger.debug(f"Updated preferences for user {user_id}")
    
    def set_fallback_hierarchy(self, hierarchy: List[FallbackChannel]):
        """Set custom fallback hierarchy"""
        self.fallback_hierarchy = sorted(hierarchy, key=lambda x: x.priority)
        logger.info(f"Updated fallback hierarchy with {len(hierarchy)} channels")
    
    async def shutdown(self):
        """Shutdown the health monitor"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self._initialized = False


# Singleton instance
_channel_health_monitor: Optional[ChannelHealthMonitor] = None


def get_channel_health_monitor(channel_manager: ChannelManager, config: GatewayConfig) -> ChannelHealthMonitor:
    """Get or create the singleton ChannelHealthMonitor instance"""
    global _channel_health_monitor
    if _channel_health_monitor is None:
        _channel_health_monitor = ChannelHealthMonitor(channel_manager, config)
    return _channel_health_monitor


def with_fallback(func: Callable):
    """Decorator to automatically use fallback channels for message sending"""
    @wraps(func)
    async def wrapper(self, user_id: str, message: str, *args, **kwargs):
        # Extract context from kwargs or create new one
        context = kwargs.get('context', {})
        if 'preferred_channel' not in context and hasattr(self, 'channel_name'):
            context['preferred_channel'] = self.channel_name
        
        # Get health monitor from the channel manager
        if hasattr(self, 'channel_manager'):
            health_monitor = get_channel_health_monitor(
                self.channel_manager, 
                getattr(self, 'config', GatewayConfig())
            )
            return await health_monitor.send_with_fallback(user_id, message, context)
        else:
            # Fallback to original method if no health monitor available
            return await func(self, user_id, message, *args, **kwargs)
    
    return wrapper


# Integration with existing channel classes
def patch_channel_classes():
    """Monkey patch channel classes to use fallback mechanism"""
    from ..channels.slack import SlackChannel
    from ..channels.telegram import TelegramChannel
    from ..channels.feishu import FeishuChannel
    
    # Patch send methods to use fallback
    for channel_class in [SlackChannel, TelegramChannel, FeishuChannel]:
        if hasattr(channel_class, 'send'):
            original_send = channel_class.send
            channel_class.send = with_fallback(original_send)


# Auto-patch when module is imported
patch_channel_classes()