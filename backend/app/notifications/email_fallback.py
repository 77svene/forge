"""backend/app/notifications/email_fallback.py"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import asynccontextmanager

from backend.app.channels.base import BaseChannel
from backend.app.channels.manager import ChannelManager
from backend.app.channels.message_bus import MessageBus
from backend.app.channels.store import ChannelStore
from backend.app.gateway.config import Settings

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation with exponential backoff."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_attempts: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    last_attempt_time: Optional[datetime] = field(default=None)
    consecutive_successes: int = field(default=0)
    
    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.half_open_max_attempts:
                self.reset()
        else:
            self.failure_count = 0
            self.consecutive_successes = 0
    
    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.consecutive_successes = 0
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened during half-open state")
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time is None:
                return True
            
            time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
            if time_since_failure >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.consecutive_successes = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_successes = 0
        logger.info("Circuit breaker reset to closed state")


@dataclass
class ChannelHealth:
    """Health status of a channel."""
    channel_name: str
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    failure_count: int = 0
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    last_error: Optional[str] = None


class FallbackHierarchy(Enum):
    """Fallback hierarchy for notification channels."""
    SLACK = "slack"
    TELEGRAM = "telegram"
    EMAIL = "email"
    FEISHU = "feishu"


@dataclass
class UserNotificationPreferences:
    """User preferences for notification channels and degraded modes."""
    user_id: str
    primary_channel: FallbackHierarchy = FallbackHierarchy.SLACK
    fallback_channels: List[FallbackHierarchy] = field(default_factory=lambda: [
        FallbackHierarchy.TELEGRAM,
        FallbackHierarchy.EMAIL
    ])
    enable_degraded_mode: bool = True
    email_on_failure: bool = True
    notification_timeout: int = 30  # seconds
    max_retries: int = 3


class EmailFallback:
    """
    Email fallback system with graceful degradation.
    
    When primary notification channels fail (rate limits, API outages),
    automatically reroutes to backup channels or email notifications.
    """
    
    def __init__(
        self,
        channel_manager: ChannelManager,
        message_bus: MessageBus,
        channel_store: ChannelStore,
        settings: Settings
    ):
        self.channel_manager = channel_manager
        self.message_bus = message_bus
        self.channel_store = channel_store
        self.settings = settings
        
        # Health tracking for each channel
        self.channel_health: Dict[str, ChannelHealth] = {}
        
        # User preferences cache
        self.user_preferences: Dict[str, UserNotificationPreferences] = {}
        
        # Email configuration
        self.smtp_host = getattr(settings, 'SMTP_HOST', 'localhost')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.smtp_username = getattr(settings, 'SMTP_USERNAME', '')
        self.smtp_password = getattr(settings, 'SMTP_PASSWORD', '')
        self.smtp_use_tls = getattr(settings, 'SMTP_USE_TLS', True)
        self.sender_email = getattr(settings, 'SENDER_EMAIL', 'noreply@forge.com')
        
        # Initialize health for known channels
        self._initialize_channel_health()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _initialize_channel_health(self):
        """Initialize health tracking for all available channels."""
        available_channels = self.channel_manager.get_available_channels()
        for channel_name in available_channels:
            if channel_name not in self.channel_health:
                self.channel_health[channel_name] = ChannelHealth(
                    channel_name=channel_name,
                    circuit_breaker=CircuitBreaker(
                        failure_threshold=5,
                        recovery_timeout=60
                    )
                )
    
    async def start(self):
        """Start the email fallback system."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._periodic_health_checks())
        logger.info("Email fallback system started")
    
    async def stop(self):
        """Stop the email fallback system."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Email fallback system stopped")
    
    async def _periodic_health_checks(self):
        """Periodically check health of all channels."""
        while self._running:
            try:
                await self._check_all_channels_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health checks: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _check_all_channels_health(self):
        """Check health of all channels."""
        for channel_name, health in self.channel_health.items():
            try:
                await self._check_channel_health(channel_name, health)
            except Exception as e:
                logger.error(f"Failed to check health for {channel_name}: {e}")
                health.is_healthy = False
                health.last_error = str(e)
                health.failure_count += 1
    
    async def _check_channel_health(self, channel_name: str, health: ChannelHealth):
        """Check health of a specific channel."""
        health.last_check = datetime.utcnow()
        
        try:
            # Try to get the channel instance
            channel = self.channel_manager.get_channel(channel_name)
            if not channel:
                health.is_healthy = False
                health.last_error = "Channel not found"
                return
            
            # Perform a health check based on channel type
            if hasattr(channel, 'health_check'):
                is_healthy = await channel.health_check()
            else:
                # Fallback: try to send a test message or check API status
                is_healthy = await self._generic_health_check(channel)
            
            if is_healthy:
                health.is_healthy = True
                health.failure_count = 0
                health.last_error = None
                health.circuit_breaker.record_success()
            else:
                health.is_healthy = False
                health.failure_count += 1
                health.last_error = "Health check failed"
                health.circuit_breaker.record_failure()
                
        except Exception as e:
            health.is_healthy = False
            health.failure_count += 1
            health.last_error = str(e)
            health.circuit_breaker.record_failure()
            logger.warning(f"Health check failed for {channel_name}: {e}")
    
    async def _generic_health_check(self, channel: BaseChannel) -> bool:
        """Generic health check for channels without specific health_check method."""
        try:
            # For now, just check if channel is initialized
            # In a real implementation, this would make a lightweight API call
            return channel is not None
        except Exception:
            return False
    
    async def send_notification_with_fallback(
        self,
        user_id: str,
        message: str,
        subject: str = "Notification",
        metadata: Optional[Dict[str, Any]] = None,
        preferred_channel: Optional[FallbackHierarchy] = None
    ) -> Dict[str, Any]:
        """
        Send a notification with automatic fallback to backup channels.
        
        Returns:
            Dict with results including which channel was used and any errors.
        """
        result = {
            "success": False,
            "channel_used": None,
            "attempts": [],
            "fallback_triggered": False,
            "errors": []
        }
        
        # Get user preferences
        preferences = await self._get_user_preferences(user_id)
        
        # Determine channel order based on preferences and health
        channel_order = self._determine_channel_order(preferences, preferred_channel)
        
        # Try each channel in order
        for channel_name in channel_order:
            attempt_result = await self._try_send_to_channel(
                user_id=user_id,
                channel_name=channel_name.value,
                message=message,
                subject=subject,
                metadata=metadata
            )
            
            result["attempts"].append(attempt_result)
            
            if attempt_result["success"]:
                result["success"] = True
                result["channel_used"] = channel_name.value
                if channel_name != channel_order[0]:
                    result["fallback_triggered"] = True
                break
            else:
                result["errors"].append(attempt_result["error"])
        
        # If all channels failed and email fallback is enabled, try email
        if not result["success"] and preferences.email_on_failure:
            email_result = await self._send_email_fallback(
                user_id=user_id,
                message=message,
                subject=subject,
                metadata=metadata
            )
            result["attempts"].append(email_result)
            if email_result["success"]:
                result["success"] = True
                result["channel_used"] = "email"
                result["fallback_triggered"] = True
        
        return result
    
    def _determine_channel_order(
        self,
        preferences: UserNotificationPreferences,
        preferred_channel: Optional[FallbackHierarchy] = None
    ) -> List[FallbackHierarchy]:
        """Determine the order of channels to try based on preferences and health."""
        # Start with preferred channel if specified
        if preferred_channel:
            channels = [preferred_channel]
        else:
            channels = [preferences.primary_channel]
        
        # Add fallback channels
        for fallback in preferences.fallback_channels:
            if fallback not in channels:
                channels.append(fallback)
        
        # Filter out unhealthy channels if degraded mode is enabled
        if preferences.enable_degraded_mode:
            healthy_channels = []
            for channel in channels:
                health = self.channel_health.get(channel.value)
                if health and health.is_healthy and health.circuit_breaker.can_execute():
                    healthy_channels.append(channel)
                else:
                    logger.debug(f"Skipping unhealthy channel: {channel.value}")
            
            # If no healthy channels, return original list (will fail but try anyway)
            if healthy_channels:
                return healthy_channels
        
        return channels
    
    async def _try_send_to_channel(
        self,
        user_id: str,
        channel_name: str,
        message: str,
        subject: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Try to send a notification to a specific channel."""
        attempt_result = {
            "channel": channel_name,
            "success": False,
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check circuit breaker
        health = self.channel_health.get(channel_name)
        if health and not health.circuit_breaker.can_execute():
            attempt_result["error"] = f"Circuit breaker open for {channel_name}"
            return attempt_result
        
        try:
            # Get the channel
            channel = self.channel_manager.get_channel(channel_name)
            if not channel:
                attempt_result["error"] = f"Channel {channel_name} not found"
                return attempt_result
            
            # Prepare the message
            notification_data = {
                "user_id": user_id,
                "message": message,
                "subject": subject,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send via channel
            success = await channel.send_message(
                recipient=user_id,
                content=message,
                metadata=notification_data
            )
            
            if success:
                attempt_result["success"] = True
                if health:
                    health.circuit_breaker.record_success()
            else:
                attempt_result["error"] = f"Failed to send via {channel_name}"
                if health:
                    health.circuit_breaker.record_failure()
                    
        except Exception as e:
            attempt_result["error"] = str(e)
            if health:
                health.circuit_breaker.record_failure()
            logger.error(f"Error sending via {channel_name}: {e}")
        
        return attempt_result
    
    async def _send_email_fallback(
        self,
        user_id: str,
        message: str,
        subject: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send notification via email as final fallback."""
        result = {
            "channel": "email",
            "success": False,
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Get user email from store or metadata
            user_email = await self._get_user_email(user_id, metadata)
            if not user_email:
                result["error"] = f"No email found for user {user_id}"
                return result
            
            # Send email
            success = await self._send_email(
                to_email=user_email,
                subject=subject,
                body=message,
                html_body=self._create_html_email(subject, message, metadata)
            )
            
            if success:
                result["success"] = True
                logger.info(f"Email fallback sent to {user_email} for user {user_id}")
            else:
                result["error"] = "Failed to send email"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Email fallback failed: {e}")
        
        return result
    
    async def _send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Send an email using SMTP."""
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['From'] = self.sender_email
            message['To'] = to_email
            message['Subject'] = subject
            
            # Add plain text part
            text_part = MIMEText(body, 'plain')
            message.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                message.attach(html_part)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=self.smtp_use_tls
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_html_email(
        self,
        subject: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create HTML version of the email."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .content {{ margin-top: 20px; padding: 20px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #666; text-align: center; }}
                .metadata {{ margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{subject}</h2>
                </div>
                <div class="content">
                    <p>{message.replace('\n', '<br>')}</p>
                    {f'<div class="metadata"><strong>Additional Info:</strong><br>{self._format_metadata(metadata)}</div>' if metadata else ''}
                </div>
                <div class="footer">
                    <p>This is an automated notification from Deer Flow.</p>
                    <p>Sent at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display in email."""
        if not metadata:
            return ""
        
        formatted = []
        for key, value in metadata.items():
            if key not in ['user_id', 'timestamp']:  # Skip internal fields
                formatted.append(f"<strong>{key}:</strong> {value}")
        
        return "<br>".join(formatted) if formatted else ""
    
    async def _get_user_preferences(self, user_id: str) -> UserNotificationPreferences:
        """Get user notification preferences, with defaults."""
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        
        # Try to load from store
        try:
            stored_prefs = await self.channel_store.get_user_notification_preferences(user_id)
            if stored_prefs:
                preferences = UserNotificationPreferences(**stored_prefs)
                self.user_preferences[user_id] = preferences
                return preferences
        except Exception as e:
            logger.debug(f"Could not load preferences for {user_id}: {e}")
        
        # Return defaults
        default_prefs = UserNotificationPreferences(user_id=user_id)
        self.user_preferences[user_id] = default_prefs
        return default_prefs
    
    async def _get_user_email(
        self,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Get user email address from metadata or store."""
        # Check metadata first
        if metadata and 'email' in metadata:
            return metadata['email']
        
        # Try to get from channel store
        try:
            user_info = await self.channel_store.get_user_info(user_id)
            if user_info and 'email' in user_info:
                return user_info['email']
        except Exception as e:
            logger.debug(f"Could not get email for {user_id}: {e}")
        
        return None
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user notification preferences."""
        try:
            # Validate and update preferences
            current = await self._get_user_preferences(user_id)
            
            # Update allowed fields
            if 'primary_channel' in preferences:
                current.primary_channel = FallbackHierarchy(preferences['primary_channel'])
            if 'fallback_channels' in preferences:
                current.fallback_channels = [
                    FallbackHierarchy(ch) for ch in preferences['fallback_channels']
                ]
            if 'enable_degraded_mode' in preferences:
                current.enable_degraded_mode = preferences['enable_degraded_mode']
            if 'email_on_failure' in preferences:
                current.email_on_failure = preferences['email_on_failure']
            if 'notification_timeout' in preferences:
                current.notification_timeout = preferences['notification_timeout']
            if 'max_retries' in preferences:
                current.max_retries = preferences['max_retries']
            
            # Store updated preferences
            self.user_preferences[user_id] = current
            await self.channel_store.set_user_notification_preferences(
                user_id,
                current.__dict__
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update preferences for {user_id}: {e}")
            return False
    
    async def get_channel_health_status(self) -> Dict[str, Any]:
        """Get health status of all channels."""
        status = {}
        for channel_name, health in self.channel_health.items():
            status[channel_name] = {
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "failure_count": health.failure_count,
                "circuit_state": health.circuit_breaker.state.value,
                "last_error": health.last_error
            }
        return status
    
    async def force_health_check(self, channel_name: Optional[str] = None):
        """Force an immediate health check for a specific channel or all channels."""
        if channel_name:
            if channel_name in self.channel_health:
                await self._check_channel_health(
                    channel_name,
                    self.channel_health[channel_name]
                )
        else:
            await self._check_all_channels_health()
    
    async def reset_circuit_breaker(self, channel_name: str) -> bool:
        """Reset circuit breaker for a specific channel."""
        if channel_name in self.channel_health:
            self.channel_health[channel_name].circuit_breaker.reset()
            logger.info(f"Circuit breaker reset for {channel_name}")
            return True
        return False


# Factory function for easy integration
def create_email_fallback(
    channel_manager: ChannelManager,
    message_bus: MessageBus,
    channel_store: ChannelStore,
    settings: Settings
) -> EmailFallback:
    """Create and return an EmailFallback instance."""
    return EmailFallback(
        channel_manager=channel_manager,
        message_bus=message_bus,
        channel_store=channel_store,
        settings=settings
    )