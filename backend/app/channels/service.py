"""ChannelService — manages the lifecycle of all IM channels."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

from app.channels.manager import ChannelManager
from app.channels.broker import Broker, RedisStreamsBroker
from app.channels.store import ChannelStore

logger = logging.getLogger(__name__)

# Channel name → import path for lazy loading
_CHANNEL_REGISTRY: dict[str, str] = {
    "feishu": "app.channels.feishu:FeishuChannel",
    "slack": "app.channels.slack:SlackChannel",
    "telegram": "app.channels.telegram:TelegramChannel",
    "email": "app.channels.email:EmailChannel",  # Added email channel
}

class ChannelHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class CircuitBreaker:
    """Circuit breaker with exponential backoff for channel health monitoring."""
    failure_threshold: int = 5
    reset_timeout: int = 60  # seconds
    exponential_base: int = 2
    max_timeout: int = 300  # 5 minutes
    
    failures: int = 0
    last_failure_time: Optional[float] = None
    state: ChannelHealth = ChannelHealth.HEALTHY
    next_attempt_time: Optional[float] = None
    
    def record_failure(self) -> None:
        """Record a failure and update circuit breaker state."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = ChannelHealth.UNHEALTHY
            # Calculate exponential backoff
            backoff = min(
                self.reset_timeout * (self.exponential_base ** (self.failures - self.failure_threshold)),
                self.max_timeout
            )
            self.next_attempt_time = time.time() + backoff
            logger.warning(f"Circuit breaker opened. Next attempt in {backoff:.1f}s")
    
    def record_success(self) -> None:
        """Record a success and reset circuit breaker."""
        self.failures = 0
        self.last_failure_time = None
        self.state = ChannelHealth.HEALTHY
        self.next_attempt_time = None
    
    def can_attempt(self) -> bool:
        """Check if we can attempt to use this channel."""
        if self.state == ChannelHealth.HEALTHY:
            return True
        
        if self.state == ChannelHealth.UNHEALTHY:
            if self.next_attempt_time and time.time() >= self.next_attempt_time:
                self.state = ChannelHealth.DEGRADED
                return True
            return False
        
        # DEGRADED state - allow attempts but monitor closely
        return True

@dataclass
class ChannelHealthStatus:
    """Health status for a channel."""
    name: str
    health: ChannelHealth = ChannelHealth.HEALTHY
    last_check: Optional[float] = None
    last_success: Optional[float] = None
    failure_count: int = 0
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    fallback_channels: List[str] = field(default_factory=list)

class ChannelService:
    """Manages the lifecycle of all configured IM channels with graceful degradation.

    Reads configuration from ``config.yaml`` under the ``channels`` key,
    instantiates enabled channels, and starts the ChannelManager dispatcher.
    Implements health checks, circuit breakers, and fallback channels.
    """

    def __init__(self, channels_config: dict[str, Any] | None = None) -> None:
        config = dict(channels_config or {})
        
        # Extract broker configuration
        broker_config = config.pop("broker", {})
        broker_type = broker_config.pop("type", "redis_streams")
        
        # Initialize the appropriate broker
        if broker_type == "redis_streams":
            self.broker = RedisStreamsBroker(config=broker_config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        
        self.store = ChannelStore()
        
        # Extract service configuration
        langgraph_url = config.pop("langgraph_url", None) or "http://localhost:2024"
        gateway_url = config.pop("gateway_url", None) or "http://localhost:8001"
        default_session = config.pop("session", None)
        channel_sessions = {
            name: channel_config.get("session") 
            for name, channel_config in config.items() 
            if isinstance(channel_config, dict)
        }
        
        # Fallback hierarchy configuration
        self.fallback_hierarchy = config.pop("fallback_hierarchy", ["slack", "telegram", "email"])
        self.health_check_interval = config.pop("health_check_interval", 30)  # seconds
        self.user_notification_preferences = config.pop("user_notification_preferences", {})
        
        self.manager = ChannelManager(
            broker=self.broker,
            store=self.store,
            langgraph_url=langgraph_url,
            gateway_url=gateway_url,
            default_session=default_session if isinstance(default_session, dict) else None,
            channel_sessions=channel_sessions,
        )
        
        self._channels: dict[str, Any] = {}  # name -> Channel instance
        self._channel_health: dict[str, ChannelHealthStatus] = {}
        self._config = config
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._fallback_enabled = True

    @classmethod
    def from_app_config(cls) -> ChannelService:
        """Create a ChannelService from the application config."""
        from deerflow.config.app_config import get_app_config

        config = get_app_config()
        channels_config = {}
        # extra fields are allowed by AppConfig (extra="allow")
        extra = config.model_extra or {}
        if "channels" in extra:
            channels_config = extra["channels"]
        return cls(channels_config=channels_config)

    async def start(self) -> None:
        """Start the manager and all enabled channels."""
        if self._running:
            return

        await self.broker.start()
        await self.manager.start()

        # Initialize health status for all registered channels
        for name in _CHANNEL_REGISTRY:
            self._channel_health[name] = ChannelHealthStatus(
                name=name,
                fallback_channels=self._get_fallback_channels(name)
            )

        for name, channel_config in self._config.items():
            if not isinstance(channel_config, dict):
                continue
            if not channel_config.get("enabled", False):
                logger.info("Channel %s is disabled, skipping", name)
                continue

            await self._start_channel(name, channel_config)

        # Start health check background task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._running = True
        logger.info("ChannelService started with channels: %s", list(self._channels.keys()))

    async def stop(self) -> None:
        """Stop all channels and the manager."""
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        for name, channel in list(self._channels.items()):
            try:
                await channel.stop()
                logger.info("Channel %s stopped", name)
            except Exception:
                logger.exception("Error stopping channel %s", name)
        self._channels.clear()

        await self.manager.stop()
        await self.broker.close()
        self._running = False
        logger.info("ChannelService stopped")

    async def restart_channel(self, name: str) -> bool:
        """Restart a specific channel. Returns True if successful."""
        if name in self._channels:
            try:
                await self._channels[name].stop()
            except Exception:
                logger.exception("Error stopping channel %s for restart", name)
            del self._channels[name]

        config = self._config.get(name)
        if not config or not isinstance(config, dict):
            logger.warning("No config for channel %s", name)
            return False

        return await self._start_channel(name, config)

    async def _start_channel(self, name: str, config: dict[str, Any]) -> bool:
        """Instantiate and start a single channel."""
        import_path = _CHANNEL_REGISTRY.get(name)
        if not import_path:
            logger.warning("Unknown channel type: %s", name)
            return False

        try:
            from deerflow.reflection import resolve_class

            channel_cls = resolve_class(import_path, base_class=None)
        except Exception:
            logger.exception("Failed to import channel class for %s", name)
            return False

        try:
            channel = channel_cls(broker=self.broker, config=config)
            await channel.start()
            self._channels[name] = channel
            
            # Update health status on successful start
            if name in self._channel_health:
                self._channel_health[name].health = ChannelHealth.HEALTHY
                self._channel_health[name].last_success = time.time()
                self._channel_health[name].circuit_breaker.record_success()
            
            logger.info("Channel %s started", name)
            return True
        except Exception as e:
            logger.exception("Failed to start channel %s", name)
            
            # Update health status on failure
            if name in self._channel_health:
                self._channel_health[name].health = ChannelHealth.UNHEALTHY
                self._channel_health[name].failure_count += 1
                self._channel_health[name].circuit_breaker.record_failure()
            
            return False

    async def _health_check_loop(self) -> None:
        """Background task to periodically check channel health."""
        while self._running:
            try:
                await self._check_all_channels_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in health check loop")
                await asyncio.sleep(5)  # Short delay before retrying

    async def _check_all_channels_health(self) -> None:
        """Check health of all channels and update their status."""
        for name, health_status in self._channel_health.items():
            if name not in self._channels:
                continue
                
            try:
                # Perform health check (this would call channel's health_check method if available)
                channel = self._channels[name]
                is_healthy = await self._check_channel_health(channel)
                
                health_status.last_check = time.time()
                
                if is_healthy:
                    health_status.health = ChannelHealth.HEALTHY
                    health_status.failure_count = 0
                    health_status.circuit_breaker.record_success()
                    health_status.last_success = time.time()
                else:
                    health_status.failure_count += 1
                    health_status.circuit_breaker.record_failure()
                    
                    if health_status.failure_count >= health_status.circuit_breaker.failure_threshold:
                        health_status.health = ChannelHealth.UNHEALTHY
                        logger.warning(f"Channel {name} marked as unhealthy after {health_status.failure_count} failures")
                    else:
                        health_status.health = ChannelHealth.DEGRADED
                        
            except Exception as e:
                logger.error(f"Health check failed for channel {name}: {e}")
                health_status.failure_count += 1
                health_status.circuit_breaker.record_failure()

    async def _check_channel_health(self, channel: Any) -> bool:
        """Check if a channel is healthy. Override this for specific health checks."""
        try:
            # Default health check: check if channel is running
            if hasattr(channel, 'is_running'):
                return channel.is_running
            
            # If channel has a health_check method, use it
            if hasattr(channel, 'health_check'):
                return await channel.health_check()
            
            # Default: assume healthy if no specific check
            return True
        except Exception:
            return False

    def _get_fallback_channels(self, channel_name: str) -> List[str]:
        """Get fallback channels for a given channel based on hierarchy."""
        if channel_name not in self.fallback_hierarchy:
            return []
        
        current_index = self.fallback_hierarchy.index(channel_name)
        return self.fallback_hierarchy[current_index + 1:]

    async def send_with_fallback(self, message: Any, user_id: Optional[str] = None, 
                                preferred_channel: Optional[str] = None) -> bool:
        """
        Send a message using fallback channels if primary fails.
        
        Args:
            message: The message to send
            user_id: Optional user ID for personalized fallback preferences
            preferred_channel: Preferred channel to try first
            
        Returns:
            True if message was sent successfully via any channel
        """
        if not self._fallback_enabled:
            # Fallback disabled, try only preferred or first available
            channel_to_try = preferred_channel or next(iter(self._channels.keys()), None)
            if channel_to_try and channel_to_try in self._channels:
                return await self._try_send_message(channel_to_try, message)
            return False
        
        # Determine channel order based on preferences and health
        channels_to_try = self._get_channel_send_order(preferred_channel, user_id)
        
        for channel_name in channels_to_try:
            if channel_name not in self._channels:
                continue
                
            health_status = self._channel_health.get(channel_name)
            if health_status and not health_status.circuit_breaker.can_attempt():
                logger.debug(f"Skipping channel {channel_name} (circuit breaker open)")
                continue
            
            success = await self._try_send_message(channel_name, message)
            if success:
                logger.info(f"Message sent successfully via {channel_name}")
                return True
            else:
                logger.warning(f"Failed to send via {channel_name}, trying next fallback")
        
        logger.error("All channels failed to send message")
        return False

    async def _try_send_message(self, channel_name: str, message: Any) -> bool:
        """Attempt to send a message via a specific channel."""
        try:
            channel = self._channels.get(channel_name)
            if not channel:
                return False
            
            # Check if channel is running
            if hasattr(channel, 'is_running') and not channel.is_running:
                return False
            
            # Send message (implementation depends on channel interface)
            # This is a placeholder - actual implementation would call channel's send method
            if hasattr(channel, 'send'):
                await channel.send(message)
                return True
            elif hasattr(channel, 'send_message'):
                await channel.send_message(message)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error sending via {channel_name}: {e}")
            
            # Update health status
            if channel_name in self._channel_health:
                self._channel_health[channel_name].circuit_breaker.record_failure()
            
            return False

    def _get_channel_send_order(self, preferred_channel: Optional[str] = None, 
                               user_id: Optional[str] = None) -> List[str]:
        """Determine the order of channels to try for sending."""
        order = []
        
        # Add preferred channel first if specified
        if preferred_channel and preferred_channel in self._channels:
            order.append(preferred_channel)
        
        # Check user-specific preferences
        if user_id and user_id in self.user_notification_preferences:
            user_prefs = self.user_notification_preferences[user_id]
            for channel in user_prefs.get("channel_order", []):
                if channel in self._channels and channel not in order:
                    order.append(channel)
        
        # Add remaining channels based on fallback hierarchy
        for channel in self.fallback_hierarchy:
            if channel in self._channels and channel not in order:
                order.append(channel)
        
        return order

    def enable_fallback(self, enabled: bool = True) -> None:
        """Enable or disable fallback channels."""
        self._fallback_enabled = enabled
        logger.info(f"Fallback channels {'enabled' if enabled else 'disabled'}")

    def get_status(self) -> dict[str, Any]:
        """Return status information for all channels with health metrics."""
        channels_status = {}
        for name in _CHANNEL_REGISTRY:
            config = self._config.get(name, {})
            enabled = isinstance(config, dict) and config.get("enabled", False)
            running = name in self._channels and self._channels[name].is_running
            
            health_status = self._channel_health.get(name)
            health_info = {}
            if health_status:
                health_info = {
                    "health": health_status.health.value,
                    "last_check": health_status.last_check,
                    "last_success": health_status.last_success,
                    "failure_count": health_status.failure_count,
                    "circuit_breaker_state": health_status.circuit_breaker.state.value,
                    "fallback_channels": health_status.fallback_channels,
                }
            
            channels_status[name] = {
                "enabled": enabled,
                "running": running,
                **health_info,
            }
        
        # Add broker metrics if available
        broker_metrics = {}
        if hasattr(self.broker, 'get_metrics'):
            broker_metrics = self.broker.get_metrics()
        
        return {
            "service_running": self._running,
            "fallback_enabled": self._fallback_enabled,
            "fallback_hierarchy": self.fallback_hierarchy,
            "channels": channels_status,
            "broker": broker_metrics,
        }

    def get_channel_health(self, channel_name: str) -> Optional[ChannelHealthStatus]:
        """Get health status for a specific channel."""
        return self._channel_health.get(channel_name)

    def set_user_notification_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Set notification preferences for a specific user."""
        self.user_notification_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")


# -- singleton access -------------------------------------------------------

_channel_service: ChannelService | None = None


def get_channel_service() -> ChannelService | None:
    """Get the singleton ChannelService instance (if started)."""
    return _channel_service


async def start_channel_service() -> ChannelService:
    """Create and start the global ChannelService from app config."""
    global _channel_service
    if _channel_service is not None:
        return _channel_service
    _channel_service = ChannelService.from_app_config()
    await _channel_service.start()
    return _channel_service


async def stop_channel_service() -> None:
    """Stop the global ChannelService."""
    global _channel_service
    if _channel_service is not None:
        await _channel_service.stop()
        _channel_service = None