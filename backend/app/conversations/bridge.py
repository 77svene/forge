"""
Cross-Channel Conversation Bridge for Deer Flow
Enables seamless conversation continuity across Slack, Telegram, and other channels
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session

from backend.app.channels.base import BaseChannel
from backend.app.channels.manager import ChannelManager
from backend.app.channels.message_bus import MessageBus
from backend.app.channels.store import ChannelStore
from backend.app.channels.slack import SlackChannel
from backend.app.channels.telegram import TelegramChannel
from backend.app.gateway.config import get_settings

logger = logging.getLogger(__name__)
Base = declarative_base()


class IdentityProvider(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM = "custom"
    SLACK_ID = "slack_id"
    TELEGRAM_ID = "telegram_id"


class BridgeStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class BridgeContext:
    """Context preserved across channels during bridging"""
    conversation_id: str
    user_id: str
    original_channel: str
    target_channels: List[str]
    context_data: Dict[str, Any] = field(default_factory=dict)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class UserIdentityMapping(Base):
    """Maps user identities across different platforms"""
    __tablename__ = "user_identity_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    provider = Column(String(50), nullable=False)
    provider_id = Column(String(255), nullable=False)
    verified = Column(Boolean, default=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_provider_provider_id', 'provider', 'provider_id', unique=True),
        Index('idx_user_provider', 'user_id', 'provider'),
    )


class ConversationBridge(Base):
    """Tracks active conversation bridges between channels"""
    __tablename__ = "conversation_bridges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bridge_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    conversation_id = Column(String(255), nullable=False)
    source_channel = Column(String(50), nullable=False)
    target_channels = Column(JSON, default=list)
    status = Column(String(20), default=BridgeStatus.ACTIVE.value)
    context_data = Column(JSON, default=dict)
    message_count = Column(Integer, default=0)
    last_activity = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    audit_logs = relationship("BridgeAuditLog", back_populates="bridge", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_conversation', 'user_id', 'conversation_id'),
        Index('idx_status_expires', 'status', 'expires_at'),
    )


class BridgeAuditLog(Base):
    """Audit logging for bridge operations"""
    __tablename__ = "bridge_audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bridge_id = Column(UUID(as_uuid=True), ForeignKey('conversation_bridges.id'), nullable=False)
    action = Column(String(50), nullable=False)
    channel = Column(String(50))
    user_id = Column(String(255))
    details = Column(JSON, default=dict)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bridge = relationship("ConversationBridge", back_populates="audit_logs")


class PrivacySettings(Base):
    """User privacy controls for cross-channel bridging"""
    __tablename__ = "bridge_privacy_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    bridge_enabled = Column(Boolean, default=False)
    share_context = Column(Boolean, default=True)
    share_history = Column(Boolean, default=False)
    allowed_channels = Column(JSON, default=list)
    blocked_channels = Column(JSON, default=list)
    data_retention_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConversationBridgeService:
    """Main service for managing cross-channel conversation bridges"""
    
    def __init__(
        self,
        channel_manager: ChannelManager,
        message_bus: MessageBus,
        channel_store: ChannelStore,
        db_session: Session
    ):
        self.channel_manager = channel_manager
        self.message_bus = message_bus
        self.channel_store = channel_store
        self.db = db_session
        self.settings = get_settings()
        self._active_bridges: Dict[str, BridgeContext] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Register message handlers
        self._register_message_handlers()
    
    def _register_message_handlers(self):
        """Register handlers for cross-channel message events"""
        self.message_bus.subscribe("message_received", self._handle_incoming_message)
        self.message_bus.subscribe("message_sent", self._handle_outgoing_message)
        self.message_bus.subscribe("channel_connected", self._handle_channel_connected)
        self.message_bus.subscribe("channel_disconnected", self._handle_channel_disconnected)
    
    async def start(self):
        """Start the bridge service"""
        logger.info("Starting Conversation Bridge Service")
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_bridges())
        
        # Load active bridges from database
        await self._load_active_bridges()
    
    async def stop(self):
        """Stop the bridge service"""
        logger.info("Stopping Conversation Bridge Service")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save active bridges to database
        await self._save_active_bridges()
    
    async def _load_active_bridges(self):
        """Load active bridges from database on startup"""
        try:
            active_bridges = self.db.query(ConversationBridge).filter(
                ConversationBridge.status == BridgeStatus.ACTIVE.value,
                ConversationBridge.expires_at > datetime.utcnow()
            ).all()
            
            for bridge in active_bridges:
                context = BridgeContext(
                    conversation_id=bridge.conversation_id,
                    user_id=bridge.user_id,
                    original_channel=bridge.source_channel,
                    target_channels=bridge.target_channels,
                    context_data=bridge.context_data,
                    created_at=bridge.created_at,
                    expires_at=bridge.expires_at
                )
                self._active_bridges[bridge.bridge_id] = context
            
            logger.info(f"Loaded {len(active_bridges)} active bridges")
        except Exception as e:
            logger.error(f"Failed to load active bridges: {e}")
    
    async def _save_active_bridges(self):
        """Save active bridges to database on shutdown"""
        try:
            for bridge_id, context in self._active_bridges.items():
                bridge = self.db.query(ConversationBridge).filter_by(bridge_id=bridge_id).first()
                if bridge:
                    bridge.context_data = context.context_data
                    bridge.message_count = len(context.message_history)
                    bridge.last_activity = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Saved {len(self._active_bridges)} active bridges")
        except Exception as e:
            logger.error(f"Failed to save active bridges: {e}")
            self.db.rollback()
    
    async def _cleanup_expired_bridges(self):
        """Periodically clean up expired bridges"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                expired_bridges = self.db.query(ConversationBridge).filter(
                    ConversationBridge.expires_at <= datetime.utcnow(),
                    ConversationBridge.status == BridgeStatus.ACTIVE.value
                ).all()
                
                for bridge in expired_bridges:
                    bridge.status = BridgeStatus.ARCHIVED.value
                    if bridge.bridge_id in self._active_bridges:
                        del self._active_bridges[bridge.bridge_id]
                
                if expired_bridges:
                    self.db.commit()
                    logger.info(f"Archived {len(expired_bridges)} expired bridges")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bridge cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def create_bridge(
        self,
        user_id: str,
        conversation_id: str,
        source_channel: str,
        target_channels: List[str],
        context_data: Optional[Dict[str, Any]] = None,
        ttl_hours: int = 24
    ) -> Optional[str]:
        """
        Create a new conversation bridge between channels
        
        Returns:
            Bridge ID if successful, None otherwise
        """
        try:
            # Check privacy settings
            privacy = self.db.query(PrivacySettings).filter_by(user_id=user_id).first()
            if privacy and not privacy.bridge_enabled:
                logger.warning(f"Bridge creation denied for user {user_id}: bridging disabled")
                return None
            
            # Filter allowed channels
            if privacy:
                allowed = set(privacy.allowed_channels) if privacy.allowed_channels else set()
                blocked = set(privacy.blocked_channels) if privacy.blocked_channels else set()
                target_channels = [
                    ch for ch in target_channels 
                    if ch not in blocked and (not allowed or ch in allowed)
                ]
            
            if not target_channels:
                logger.warning(f"No valid target channels for bridge")
                return None
            
            # Verify user has access to all channels
            for channel in [source_channel] + target_channels:
                if not await self._verify_channel_access(user_id, channel):
                    logger.warning(f"User {user_id} lacks access to channel {channel}")
                    return None
            
            # Create bridge context
            bridge_id = f"bridge_{uuid.uuid4().hex[:12]}"
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            context = BridgeContext(
                conversation_id=conversation_id,
                user_id=user_id,
                original_channel=source_channel,
                target_channels=target_channels,
                context_data=context_data or {},
                expires_at=expires_at
            )
            
            # Store in database
            db_bridge = ConversationBridge(
                bridge_id=bridge_id,
                user_id=user_id,
                conversation_id=conversation_id,
                source_channel=source_channel,
                target_channels=target_channels,
                context_data=context_data or {},
                expires_at=expires_at
            )
            
            self.db.add(db_bridge)
            self.db.commit()
            
            # Store in memory
            self._active_bridges[bridge_id] = context
            
            # Log audit event
            await self._log_audit_event(
                bridge_id=db_bridge.id,
                action="bridge_created",
                user_id=user_id,
                details={
                    "source_channel": source_channel,
                    "target_channels": target_channels,
                    "ttl_hours": ttl_hours
                }
            )
            
            # Notify target channels
            await self._notify_bridge_created(bridge_id, context)
            
            logger.info(f"Created bridge {bridge_id} for user {user_id}")
            return bridge_id
            
        except Exception as e:
            logger.error(f"Failed to create bridge: {e}")
            self.db.rollback()
            return None
    
    async def _verify_channel_access(self, user_id: str, channel_type: str) -> bool:
        """Verify user has access to a specific channel"""
        try:
            channel = self.channel_manager.get_channel(channel_type)
            if not channel:
                return False
            
            # Check if user is registered in channel
            user_mapping = self.db.query(UserIdentityMapping).filter(
                UserIdentityMapping.user_id == user_id,
                UserIdentityMapping.provider.in_([
                    f"{channel_type}_id",
                    IdentityProvider.EMAIL.value,
                    IdentityProvider.PHONE.value
                ])
            ).first()
            
            return user_mapping is not None
            
        except Exception as e:
            logger.error(f"Channel access verification failed: {e}")
            return False
    
    async def _notify_bridge_created(self, bridge_id: str, context: BridgeContext):
        """Notify target channels about new bridge"""
        try:
            notification = {
                "type": "bridge_created",
                "bridge_id": bridge_id,
                "conversation_id": context.conversation_id,
                "source_channel": context.original_channel,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for channel_type in context.target_channels:
                channel = self.channel_manager.get_channel(channel_type)
                if channel:
                    # Send notification to user in target channel
                    user_id_in_channel = await self._get_user_id_in_channel(
                        context.user_id, channel_type
                    )
                    if user_id_in_channel:
                        await channel.send_message(
                            recipient=user_id_in_channel,
                            text=f"🔗 Conversation bridge activated from {context.original_channel}",
                            metadata={"bridge_id": bridge_id, "type": "bridge_notification"}
                        )
                        
        except Exception as e:
            logger.error(f"Failed to notify bridge creation: {e}")
    
    async def _get_user_id_in_channel(self, user_id: str, channel_type: str) -> Optional[str]:
        """Get user's ID in a specific channel"""
        mapping = self.db.query(UserIdentityMapping).filter(
            UserIdentityMapping.user_id == user_id,
            UserIdentityMapping.provider == f"{channel_type}_id"
        ).first()
        
        return mapping.provider_id if mapping else None
    
    async def bridge_message(
        self,
        bridge_id: str,
        message: Dict[str, Any],
        source_channel: str,
        user_id: str
    ) -> bool:
        """
        Bridge a message to all target channels
        
        Returns:
            True if message was bridged successfully
        """
        try:
            context = self._active_bridges.get(bridge_id)
            if not context:
                logger.warning(f"Bridge {bridge_id} not found")
                return False
            
            # Verify user owns the bridge
            if context.user_id != user_id:
                logger.warning(f"User {user_id} not authorized for bridge {bridge_id}")
                return False
            
            # Add message to history
            message_entry = {
                "id": str(uuid.uuid4()),
                "content": message,
                "source_channel": source_channel,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            context.message_history.append(message_entry)
            
            # Update context with latest message
            context.context_data["last_message"] = message
            context.context_data["last_activity"] = datetime.utcnow().isoformat()
            
            # Bridge to target channels
            bridged_count = 0
            for target_channel in context.target_channels:
                if target_channel == source_channel:
                    continue
                
                try:
                    success = await self._send_to_channel(
                        target_channel=target_channel,
                        user_id=user_id,
                        message=message,
                        bridge_id=bridge_id,
                        context=context
                    )
                    if success:
                        bridged_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to bridge to {target_channel}: {e}")
            
            # Update bridge in database
            db_bridge = self.db.query(ConversationBridge).filter_by(bridge_id=bridge_id).first()
            if db_bridge:
                db_bridge.context_data = context.context_data
                db_bridge.message_count = len(context.message_history)
                db_bridge.last_activity = datetime.utcnow()
                self.db.commit()
            
            # Log audit event
            await self._log_audit_event(
                bridge_id=db_bridge.id if db_bridge else None,
                action="message_bridged",
                user_id=user_id,
                channel=source_channel,
                details={
                    "target_channels": context.target_channels,
                    "bridged_count": bridged_count,
                    "message_preview": str(message.get("text", ""))[:100]
                }
            )
            
            logger.info(f"Bridged message from {source_channel} to {bridged_count} channels")
            return bridged_count > 0
            
        except Exception as e:
            logger.error(f"Message bridging failed: {e}")
            self.db.rollback()
            return False
    
    async def _send_to_channel(
        self,
        target_channel: str,
        user_id: str,
        message: Dict[str, Any],
        bridge_id: str,
        context: BridgeContext
    ) -> bool:
        """Send message to a specific channel"""
        try:
            channel = self.channel_manager.get_channel(target_channel)
            if not channel:
                return False
            
            # Get user ID in target channel
            target_user_id = await self._get_user_id_in_channel(user_id, target_channel)
            if not target_user_id:
                logger.warning(f"User {user_id} not found in channel {target_channel}")
                return False
            
            # Format message for target channel
            formatted_message = await self._format_message_for_channel(
                message, target_channel, bridge_id, context
            )
            
            # Send message
            await channel.send_message(
                recipient=target_user_id,
                text=formatted_message.get("text", ""),
                metadata={
                    "bridge_id": bridge_id,
                    "original_channel": context.original_channel,
                    "conversation_id": context.conversation_id,
                    **formatted_message.get("metadata", {})
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send to {target_channel}: {e}")
            return False
    
    async def _format_message_for_channel(
        self,
        message: Dict[str, Any],
        target_channel: str,
        bridge_id: str,
        context: BridgeContext
    ) -> Dict[str, Any]:
        """Format message appropriately for target channel"""
        formatted = message.copy()
        
        # Add bridge context header
        header = f"🔗 [From {context.original_channel}]"
        
        if target_channel == "slack":
            formatted["text"] = f"{header}\n{message.get('text', '')}"
            formatted["metadata"] = {
                "bridge_id": bridge_id,
                "thread_ts": message.get("thread_ts"),
                "blocks": message.get("blocks")
            }
            
        elif target_channel == "telegram":
            formatted["text"] = f"{header}\n{message.get('text', '')}"
            formatted["metadata"] = {
                "bridge_id": bridge_id,
                "parse_mode": "HTML",
                "reply_markup": message.get("reply_markup")
            }
        
        return formatted
    
    async def get_bridge_context(self, bridge_id: str) -> Optional[BridgeContext]:
        """Get context for a specific bridge"""
        return self._active_bridges.get(bridge_id)
    
    async def get_user_bridges(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active bridges for a user"""
        bridges = []
        for bridge_id, context in self._active_bridges.items():
            if context.user_id == user_id:
                bridges.append({
                    "bridge_id": bridge_id,
                    "conversation_id": context.conversation_id,
                    "source_channel": context.original_channel,
                    "target_channels": context.target_channels,
                    "created_at": context.created_at.isoformat(),
                    "expires_at": context.expires_at.isoformat() if context.expires_at else None,
                    "message_count": len(context.message_history)
                })
        return bridges
    
    async def pause_bridge(self, bridge_id: str, user_id: str) -> bool:
        """Pause an active bridge"""
        try:
            bridge = self.db.query(ConversationBridge).filter_by(
                bridge_id=bridge_id,
                user_id=user_id
            ).first()
            
            if not bridge:
                return False
            
            bridge.status = BridgeStatus.PAUSED.value
            self.db.commit()
            
            # Remove from active bridges
            if bridge_id in self._active_bridges:
                del self._active_bridges[bridge_id]
            
            # Log audit event
            await self._log_audit_event(
                bridge_id=bridge.id,
                action="bridge_paused",
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause bridge: {e}")
            self.db.rollback()
            return False
    
    async def resume_bridge(self, bridge_id: str, user_id: str) -> bool:
        """Resume a paused bridge"""
        try:
            bridge = self.db.query(ConversationBridge).filter_by(
                bridge_id=bridge_id,
                user_id=user_id
            ).first()
            
            if not bridge or bridge.status != BridgeStatus.PAUSED.value:
                return False
            
            bridge.status = BridgeStatus.ACTIVE.value
            self.db.commit()
            
            # Recreate context
            context = BridgeContext(
                conversation_id=bridge.conversation_id,
                user_id=bridge.user_id,
                original_channel=bridge.source_channel,
                target_channels=bridge.target_channels,
                context_data=bridge.context_data,
                created_at=bridge.created_at,
                expires_at=bridge.expires_at
            )
            
            self._active_bridges[bridge_id] = context
            
            # Log audit event
            await self._log_audit_event(
                bridge_id=bridge.id,
                action="bridge_resumed",
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume bridge: {e}")
            self.db.rollback()
            return False
    
    async def delete_bridge(self, bridge_id: str, user_id: str) -> bool:
        """Delete a bridge permanently"""
        try:
            bridge = self.db.query(ConversationBridge).filter_by(
                bridge_id=bridge_id,
                user_id=user_id
            ).first()
            
            if not bridge:
                return False
            
            # Log audit event before deletion
            await self._log_audit_event(
                bridge_id=bridge.id,
                action="bridge_deleted",
                user_id=user_id
            )
            
            # Delete from database
            self.db.delete(bridge)
            self.db.commit()
            
            # Remove from active bridges
            if bridge_id in self._active_bridges:
                del self._active_bridges[bridge_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete bridge: {e}")
            self.db.rollback()
            return False
    
    async def update_privacy_settings(
        self,
        user_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """Update privacy settings for a user"""
        try:
            privacy = self.db.query(PrivacySettings).filter_by(user_id=user_id).first()
            
            if not privacy:
                privacy = PrivacySettings(user_id=user_id)
                self.db.add(privacy)
            
            # Update settings
            for key, value in settings.items():
                if hasattr(privacy, key):
                    setattr(privacy, key, value)
            
            self.db.commit()
            
            # If bridging was disabled, pause all user's bridges
            if settings.get("bridge_enabled") is False:
                user_bridges = self.db.query(ConversationBridge).filter_by(
                    user_id=user_id,
                    status=BridgeStatus.ACTIVE.value
                ).all()
                
                for bridge in user_bridges:
                    bridge.status = BridgeStatus.PAUSED.value
                    if bridge.bridge_id in self._active_bridges:
                        del self._active_bridges[bridge.bridge_id]
                
                self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update privacy settings: {e}")
            self.db.rollback()
            return False
    
    async def get_privacy_settings(self, user_id: str) -> Dict[str, Any]:
        """Get privacy settings for a user"""
        privacy = self.db.query(PrivacySettings).filter_by(user_id=user_id).first()
        
        if not privacy:
            return {
                "bridge_enabled": False,
                "share_context": True,
                "share_history": False,
                "allowed_channels": [],
                "blocked_channels": [],
                "data_retention_days": 30
            }
        
        return {
            "bridge_enabled": privacy.bridge_enabled,
            "share_context": privacy.share_context,
            "share_history": privacy.share_history,
            "allowed_channels": privacy.allowed_channels or [],
            "blocked_channels": privacy.blocked_channels or [],
            "data_retention_days": privacy.data_retention_days
        }
    
    async def _log_audit_event(
        self,
        action: str,
        user_id: str,
        bridge_id: Optional[UUID] = None,
        channel: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log an audit event"""
        try:
            audit_log = BridgeAuditLog(
                bridge_id=bridge_id,
                action=action,
                channel=channel,
                user_id=user_id,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            self.db.rollback()
    
    async def _handle_incoming_message(self, event: Dict[str, Any]):
        """Handle incoming message events"""
        try:
            channel_type = event.get("channel_type")
            user_id = event.get("user_id")
            conversation_id = event.get("conversation_id")
            message = event.get("message")
            
            if not all([channel_type, user_id, conversation_id, message]):
                return
            
            # Check if this conversation is part of an active bridge
            for bridge_id, context in self._active_bridges.items():
                if (context.user_id == user_id and 
                    context.conversation_id == conversation_id and
                    context.original_channel == channel_type):
                    
                    # Bridge this message to other channels
                    await self.bridge_message(
                        bridge_id=bridge_id,
                        message=message,
                        source_channel=channel_type,
                        user_id=user_id
                    )
                    break
                    
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
    
    async def _handle_outgoing_message(self, event: Dict[str, Any]):
        """Handle outgoing message events"""
        # Similar to incoming but for messages sent by the bot
        pass
    
    async def _handle_channel_connected(self, event: Dict[str, Any]):
        """Handle channel connection events"""
        channel_type = event.get("channel_type")
        logger.info(f"Channel {channel_type} connected")
    
    async def _handle_channel_disconnected(self, event: Dict[str, Any]):
        """Handle channel disconnection events"""
        channel_type = event.get("channel_type")
        logger.warning(f"Channel {channel_type} disconnected")
        
        # Pause bridges that use this channel
        for bridge_id, context in list(self._active_bridges.items()):
            if (context.original_channel == channel_type or 
                channel_type in context.target_channels):
                
                bridge = self.db.query(ConversationBridge).filter_by(
                    bridge_id=bridge_id
                ).first()
                
                if bridge and bridge.status == BridgeStatus.ACTIVE.value:
                    bridge.status = BridgeStatus.PAUSED.value
                    del self._active_bridges[bridge_id]
        
        self.db.commit()


# Factory function for easy integration
def create_conversation_bridge_service(
    channel_manager: ChannelManager,
    message_bus: MessageBus,
    channel_store: ChannelStore,
    db_session: Session
) -> ConversationBridgeService:
    """Create and return a ConversationBridgeService instance"""
    return ConversationBridgeService(
        channel_manager=channel_manager,
        message_bus=message_bus,
        channel_store=channel_store,
        db_session=db_session
    )


# Example usage in application startup
async def initialize_bridge_service(app):
    """Initialize bridge service on application startup"""
    from backend.app.channels.manager import get_channel_manager
    from backend.app.channels.message_bus import get_message_bus
    from backend.app.channels.store import get_channel_store
    from backend.app.database import get_db_session
    
    channel_manager = get_channel_manager()
    message_bus = get_message_bus()
    channel_store = get_channel_store()
    db_session = get_db_session()
    
    bridge_service = create_conversation_bridge_service(
        channel_manager=channel_manager,
        message_bus=message_bus,
        channel_store=channel_store,
        db_session=db_session
    )
    
    await bridge_service.start()
    app.state.bridge_service = bridge_service
    
    return bridge_service