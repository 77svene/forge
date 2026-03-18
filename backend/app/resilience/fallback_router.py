import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from backend.app.channels.base import BaseChannel
from backend.app.channels.manager import ChannelManager
from backend.app.channels.message_bus import MessageBus
from backend.app.channels.service import ChannelService

logger = logging.getLogger(__name__)


class ChannelStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


class FallbackPriority(Enum):
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    EMERGENCY = 4


@dataclass
class HealthCheck:
    channel_name: str
    last_check: datetime
    status: ChannelStatus
    consecutive_failures: int = 0
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    next_retry: Optional[datetime] = None


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_attempts: int = 3
    exponential_backoff_base: float = 1.5
    max_backoff_seconds: int = 300


@dataclass
class UserNotificationPreferences:
    user_id: str
    preferred_channels: List[str] = field(default_factory=lambda: ["slack", "telegram", "email"])
    fallback_enabled: bool = True
    degraded_mode_channels: List[str] = field(default_factory=lambda: ["email"])
    notify_on_fallback: bool = True
    notify_on_recovery: bool = False


class CircuitBreaker:
    """Circuit breaker pattern implementation for channel resilience."""
    
    def __init__(self, channel_name: str, config: CircuitBreakerConfig):
        self.channel_name = channel_name
        self.config = config
        self.state = ChannelStatus.HEALTHY
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_attempts = 0
        self.backoff_seconds = 1.0
        
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.half_open_attempts = 0
        self.backoff_seconds = 1.0
        self.state = ChannelStatus.HEALTHY
        
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == ChannelStatus.CIRCUIT_OPEN:
            self.backoff_seconds = min(
                self.backoff_seconds * self.config.exponential_backoff_base,
                self.config.max_backoff_seconds
            )
        elif self.failure_count >= self.config.failure_threshold:
            self.state = ChannelStatus.CIRCUIT_OPEN
            self.backoff_seconds = self.config.recovery_timeout
            
    def can_attempt(self) -> bool:
        """Check if an operation can be attempted."""
        if self.state == ChannelStatus.HEALTHY:
            return True
            
        if self.state == ChannelStatus.CIRCUIT_OPEN:
            if self.last_failure_time is None:
                return True
                
            time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
            if time_since_failure >= self.backoff_seconds:
                self.state = ChannelStatus.DEGRADED
                self.half_open_attempts = 0
                return True
            return False
            
        # DEGRADED state
        return self.half_open_attempts < self.config.half_open_max_attempts
        
    def record_half_open_attempt(self):
        """Record an attempt in half-open state."""
        self.half_open_attempts += 1


class FallbackRouter:
    """Manages graceful degradation with fallback channels."""
    
    # Default fallback hierarchy
    DEFAULT_FALLBACK_HIERARCHY = {
        "slack": ["telegram", "email"],
        "telegram": ["slack", "email"],
        "feishu": ["slack", "telegram", "email"],
        "email": []  # Email is the final fallback
    }
    
    def __init__(
        self,
        channel_manager: ChannelManager,
        channel_service: ChannelService,
        message_bus: MessageBus,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        self.channel_manager = channel_manager
        self.channel_service = channel_service
        self.message_bus = message_bus
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        # Circuit breakers for each channel
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health status tracking
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # User notification preferences
        self.user_preferences: Dict[str, UserNotificationPreferences] = {}
        
        # Fallback hierarchy configuration
        self.fallback_hierarchy: Dict[str, List[str]] = self.DEFAULT_FALLBACK_HIERARCHY.copy()
        
        # Active fallback sessions
        self.active_fallbacks: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event callbacks
        self._on_fallback_callbacks: List[Callable] = []
        self._on_recovery_callbacks: List[Callable] = []
        
    async def start(self):
        """Start the fallback router and background tasks."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("FallbackRouter started")
        
    async def stop(self):
        """Stop the fallback router and cleanup."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("FallbackRouter stopped")
        
    def get_circuit_breaker(self, channel_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a channel."""
        if channel_name not in self.circuit_breakers:
            self.circuit_breakers[channel_name] = CircuitBreaker(
                channel_name, self.circuit_config
            )
        return self.circuit_breakers[channel_name]
        
    def set_user_preferences(self, user_id: str, preferences: UserNotificationPreferences):
        """Set notification preferences for a user."""
        self.user_preferences[user_id] = preferences
        
    def get_user_preferences(self, user_id: str) -> UserNotificationPreferences:
        """Get notification preferences for a user, with defaults."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserNotificationPreferences(user_id=user_id)
        return self.user_preferences[user_id]
        
    def configure_fallback_hierarchy(self, hierarchy: Dict[str, List[str]]):
        """Configure custom fallback hierarchy."""
        self.fallback_hierarchy.update(hierarchy)
        
    def register_fallback_callback(self, callback: Callable, on_recovery: bool = False):
        """Register callback for fallback or recovery events."""
        if on_recovery:
            self._on_recovery_callbacks.append(callback)
        else:
            self._on_fallback_callbacks.append(callback)
            
    async def _notify_fallback_event(self, user_id: str, from_channel: str, to_channel: str, reason: str):
        """Notify about fallback events."""
        for callback in self._on_fallback_callbacks:
            try:
                await callback(user_id, from_channel, to_channel, reason)
            except Exception as e:
                logger.error(f"Error in fallback callback: {e}")
                
    async def _notify_recovery_event(self, user_id: str, channel: str):
        """Notify about channel recovery events."""
        for callback in self._on_recovery_callbacks:
            try:
                await callback(user_id, channel)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")
                
    async def send_message(
        self,
        user_id: str,
        message: Any,
        primary_channel: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message with automatic fallback handling.
        
        Returns:
            Dict with send results including which channel was used and fallback info.
        """
        preferences = self.get_user_preferences(user_id)
        metadata = metadata or {}
        
        # Build channel attempt order based on preferences and fallback hierarchy
        attempt_order = self._build_attempt_order(user_id, primary_channel, preferences)
        
        last_error = None
        attempted_channels = []
        
        for channel_name in attempt_order:
            circuit_breaker = self.get_circuit_breaker(channel_name)
            
            # Check if channel can be attempted
            if not circuit_breaker.can_attempt():
                logger.debug(f"Channel {channel_name} circuit breaker open, skipping")
                continue
                
            # Check channel health
            health = self.health_checks.get(channel_name)
            if health and health.status == ChannelStatus.FAILED:
                logger.debug(f"Channel {channel_name} health check failed, skipping")
                continue
                
            try:
                # Record half-open attempt if needed
                if circuit_breaker.state == ChannelStatus.DEGRADED:
                    circuit_breaker.record_half_open_attempt()
                    
                # Attempt to send message
                channel = self.channel_manager.get_channel(channel_name)
                if not channel:
                    raise ValueError(f"Channel {channel_name} not found")
                    
                start_time = datetime.utcnow()
                result = await channel.send(user_id, message, metadata)
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Record success
                circuit_breaker.record_success()
                self._update_health_check(channel_name, True, response_time)
                
                # Track fallback if we didn't use primary channel
                if channel_name != primary_channel:
                    self.active_fallbacks[user_id][primary_channel] = {
                        "fallback_to": channel_name,
                        "timestamp": datetime.utcnow(),
                        "reason": last_error or "Primary channel unavailable"
                    }
                    
                    # Notify user about fallback if configured
                    if preferences.notify_on_fallback:
                        await self._notify_fallback_event(
                            user_id, primary_channel, channel_name,
                            last_error or "Primary channel unavailable"
                        )
                        
                return {
                    "success": True,
                    "channel_used": channel_name,
                    "is_fallback": channel_name != primary_channel,
                    "response_time_ms": response_time,
                    "attempted_channels": attempted_channels,
                    "fallback_info": self.active_fallbacks.get(user_id, {}).get(primary_channel)
                }
                
            except Exception as e:
                last_error = str(e)
                attempted_channels.append(channel_name)
                
                # Record failure
                circuit_breaker.record_failure()
                self._update_health_check(channel_name, False, error_message=last_error)
                
                logger.warning(
                    f"Failed to send via {channel_name} for user {user_id}: {last_error}"
                )
                
                # Continue to next channel in fallback hierarchy
                continue
                
        # All channels failed
        return {
            "success": False,
            "error": "All channels failed",
            "attempted_channels": attempted_channels,
            "last_error": last_error
        }
        
    def _build_attempt_order(
        self,
        user_id: str,
        primary_channel: str,
        preferences: UserNotificationPreferences
    ) -> List[str]:
        """Build the order of channels to attempt based on preferences and hierarchy."""
        if not preferences.fallback_enabled:
            return [primary_channel]
            
        # Start with user's preferred channels, but ensure primary is first
        preferred = preferences.preferred_channels.copy()
        if primary_channel in preferred:
            preferred.remove(primary_channel)
        attempt_order = [primary_channel] + preferred
        
        # Add channels from fallback hierarchy if not already included
        fallback_channels = self.fallback_hierarchy.get(primary_channel, [])
        for channel in fallback_channels:
            if channel not in attempt_order:
                attempt_order.append(channel)
                
        # Filter out channels that are in failed state based on circuit breakers
        filtered_order = []
        for channel in attempt_order:
            circuit_breaker = self.get_circuit_breaker(channel)
            if circuit_breaker.state != ChannelStatus.CIRCUIT_OPEN:
                filtered_order.append(channel)
                
        return filtered_order if filtered_order else [primary_channel]  # Always try primary at minimum
        
    def _update_health_check(
        self,
        channel_name: str,
        success: bool,
        response_time_ms: float = 0.0,
        error_message: Optional[str] = None
    ):
        """Update health check status for a channel."""
        now = datetime.utcnow()
        
        if channel_name not in self.health_checks:
            self.health_checks[channel_name] = HealthCheck(
                channel_name=channel_name,
                last_check=now,
                status=ChannelStatus.HEALTHY
            )
            
        health = self.health_checks[channel_name]
        health.last_check = now
        health.response_time_ms = response_time_ms
        
        if success:
            health.consecutive_failures = 0
            health.status = ChannelStatus.HEALTHY
            health.error_message = None
            health.next_retry = None
            
            # Check if we should notify about recovery
            if user_id := self._get_user_for_channel_recovery(channel_name):
                asyncio.create_task(self._notify_recovery_event(user_id, channel_name))
        else:
            health.consecutive_failures += 1
            health.error_message = error_message
            
            if health.consecutive_failures >= 3:
                health.status = ChannelStatus.FAILED
                # Schedule retry with exponential backoff
                backoff_seconds = min(2 ** health.consecutive_failures, 300)
                health.next_retry = now + timedelta(seconds=backoff_seconds)
            else:
                health.status = ChannelStatus.DEGRADED
                
    def _get_user_for_channel_recovery(self, channel_name: str) -> Optional[str]:
        """Get user ID that should be notified about channel recovery."""
        for user_id, fallbacks in self.active_fallbacks.items():
            for primary_channel, fallback_info in fallbacks.items():
                if fallback_info.get("fallback_to") != channel_name:
                    continue
                    
                preferences = self.get_user_preferences(user_id)
                if preferences.notify_on_recovery:
                    return user_id
        return None
        
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
                
    async def _perform_health_checks(self):
        """Perform health checks on all channels."""
        channels = self.channel_manager.get_all_channels()
        
        for channel_name, channel in channels.items():
            try:
                # Skip if circuit breaker is open
                circuit_breaker = self.get_circuit_breaker(channel_name)
                if circuit_breaker.state == ChannelStatus.CIRCUIT_OPEN:
                    continue
                    
                # Check if it's time to retry a failed channel
                health = self.health_checks.get(channel_name)
                if health and health.next_retry and datetime.utcnow() < health.next_retry:
                    continue
                    
                # Perform health check
                start_time = datetime.utcnow()
                is_healthy = await self._check_channel_health(channel)
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self._update_health_check(
                    channel_name,
                    is_healthy,
                    response_time_ms=response_time,
                    error_message=None if is_healthy else "Health check failed"
                )
                
                if is_healthy:
                    logger.debug(f"Channel {channel_name} health check passed")
                else:
                    logger.warning(f"Channel {channel_name} health check failed")
                    
            except Exception as e:
                logger.error(f"Error checking health of {channel_name}: {e}")
                self._update_health_check(
                    channel_name,
                    False,
                    error_message=str(e)
                )
                
    async def _check_channel_health(self, channel: BaseChannel) -> bool:
        """Check if a channel is healthy."""
        try:
            # Try to get channel status or perform a lightweight operation
            if hasattr(channel, 'health_check'):
                return await channel.health_check()
            elif hasattr(channel, 'get_status'):
                status = await channel.get_status()
                return status.get('healthy', False)
            else:
                # Default: assume healthy if channel exists
                return True
        except Exception:
            return False
            
    async def get_channel_status(self, channel_name: str) -> Dict[str, Any]:
        """Get comprehensive status for a channel."""
        circuit_breaker = self.get_circuit_breaker(channel_name)
        health = self.health_checks.get(channel_name)
        
        return {
            "channel": channel_name,
            "circuit_breaker_state": circuit_breaker.state.value,
            "failure_count": circuit_breaker.failure_count,
            "last_failure": circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None,
            "health_status": health.status.value if health else "unknown",
            "last_health_check": health.last_check.isoformat() if health else None,
            "consecutive_failures": health.consecutive_failures if health else 0,
            "response_time_ms": health.response_time_ms if health else 0.0,
            "next_retry": health.next_retry.isoformat() if health and health.next_retry else None,
            "error_message": health.error_message if health else None
        }
        
    async def get_all_channel_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all channels."""
        statuses = {}
        channels = self.channel_manager.get_all_channels()
        
        for channel_name in channels.keys():
            statuses[channel_name] = await self.get_channel_status(channel_name)
            
        return statuses
        
    async def force_channel_recovery(self, channel_name: str):
        """Force a channel to recover (reset circuit breaker)."""
        circuit_breaker = self.get_circuit_breaker(channel_name)
        circuit_breaker.record_success()
        
        if channel_name in self.health_checks:
            self.health_checks[channel_name].status = ChannelStatus.HEALTHY
            self.health_checks[channel_name].consecutive_failures = 0
            self.health_checks[channel_name].next_retry = None
            
        logger.info(f"Forced recovery of channel {channel_name}")
        
    def get_fallback_sessions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get active fallback sessions."""
        if user_id:
            return self.active_fallbacks.get(user_id, {})
        return dict(self.active_fallbacks)
        
    async def send_degraded_mode_notification(
        self,
        user_id: str,
        original_channel: str,
        reason: str
    ) -> bool:
        """Send notification about degraded mode to user's emergency channels."""
        preferences = self.get_user_preferences(user_id)
        
        for channel_name in preferences.degraded_mode_channels:
            try:
                channel = self.channel_manager.get_channel(channel_name)
                if not channel:
                    continue
                    
                message = (
                    f"⚠️ Service Alert: Your primary notification channel "
                    f"({original_channel}) is currently unavailable. "
                    f"Reason: {reason}. "
                    f"Notifications are being sent via {channel_name} as fallback."
                )
                
                await channel.send(user_id, message, {"type": "degraded_alert"})
                return True
                
            except Exception as e:
                logger.error(f"Failed to send degraded mode notification via {channel_name}: {e}")
                continue
                
        return False


# Singleton instance for easy import
fallback_router: Optional[FallbackRouter] = None


def initialize_fallback_router(
    channel_manager: ChannelManager,
    channel_service: ChannelService,
    message_bus: MessageBus,
    circuit_config: Optional[CircuitBreakerConfig] = None
) -> FallbackRouter:
    """Initialize the global fallback router instance."""
    global fallback_router
    fallback_router = FallbackRouter(
        channel_manager, channel_service, message_bus, circuit_config
    )
    return fallback_router


async def get_fallback_router() -> FallbackRouter:
    """Get the global fallback router instance."""
    if fallback_router is None:
        raise RuntimeError("FallbackRouter not initialized. Call initialize_fallback_router first.")
    return fallback_router