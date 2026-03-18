import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets

from sqlalchemy import Column, String, Boolean, DateTime, JSON, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

from backend.app.channels.base import BaseChannel, ChannelMessage
from backend.app.channels.manager import ChannelManager
from backend.app.channels.store import ChannelStore
from backend.app.gateway.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class IdentityProvider(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM_ID = "custom_id"
    OAUTH_GOOGLE = "oauth_google"
    OAUTH_GITHUB = "oauth_github"


class PrivacyLevel(str, Enum):
    FULL = "full"  # Share complete conversation history
    CONTEXT_ONLY = "context_only"  # Share only current task context
    ANONYMOUS = "anonymous"  # Share only that user exists across channels
    NONE = "none"  # No cross-channel sharing


@dataclass
class IdentityMapping:
    """Represents a mapping between channel user and global identity"""
    global_user_id: str
    channel: str
    channel_user_id: str
    provider: IdentityProvider
    provider_id: str
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationBridge:
    """Represents a bridged conversation across channels"""
    bridge_id: str
    global_user_id: str
    source_channel: str
    source_conversation_id: str
    target_channel: str
    target_conversation_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_synced: Optional[datetime] = None


class UserIdentity(Base):
    """Database model for user identity mappings"""
    __tablename__ = "user_identities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    global_user_id = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255))
    privacy_level = Column(String(50), default=PrivacyLevel.CONTEXT_ONLY.value)
    opt_in = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    channel_mappings = relationship("ChannelIdentityMapping", back_populates="user_identity")
    audit_logs = relationship("IdentityAuditLog", back_populates="user_identity")
    
    __table_args__ = (
        Index('idx_global_user_id', 'global_user_id'),
    )


class ChannelIdentityMapping(Base):
    """Maps channel-specific user IDs to global identities"""
    __tablename__ = "channel_identity_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    global_user_id = Column(String(255), ForeignKey("user_identities.global_user_id"), nullable=False)
    channel = Column(String(100), nullable=False)
    channel_user_id = Column(String(255), nullable=False)
    provider = Column(String(50), nullable=False)
    provider_id = Column(String(255), nullable=False)
    verified = Column(Boolean, default=False)
    verification_token = Column(String(255))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user_identity = relationship("UserIdentity", back_populates="channel_mappings")
    
    __table_args__ = (
        Index('idx_channel_user', 'channel', 'channel_user_id', unique=True),
        Index('idx_provider_id', 'provider', 'provider_id', unique=True),
    )


class IdentityAuditLog(Base):
    """Audit log for identity operations"""
    __tablename__ = "identity_audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    global_user_id = Column(String(255), ForeignKey("user_identities.global_user_id"), nullable=False)
    action = Column(String(100), nullable=False)
    channel = Column(String(100))
    channel_user_id = Column(String(255))
    details = Column(JSON, default=dict)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_identity = relationship("UserIdentity", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_user', 'global_user_id'),
        Index('idx_audit_created', 'created_at'),
    )


class ConversationBridgeRecord(Base):
    """Database record for conversation bridges"""
    __tablename__ = "conversation_bridges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bridge_id = Column(String(255), unique=True, nullable=False)
    global_user_id = Column(String(255), ForeignKey("user_identities.global_user_id"), nullable=False)
    source_channel = Column(String(100), nullable=False)
    source_conversation_id = Column(String(255), nullable=False)
    target_channel = Column(String(100), nullable=False)
    target_conversation_id = Column(String(255), nullable=False)
    context = Column(JSON, default=dict)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_synced = Column(DateTime)
    
    __table_args__ = (
        Index('idx_bridge_user', 'global_user_id'),
        Index('idx_bridge_source', 'source_channel', 'source_conversation_id'),
        Index('idx_bridge_target', 'target_channel', 'target_conversation_id'),
    )


class IdentityResolver:
    """
    Resolves user identities across different channels and enables
    cross-channel conversation continuity.
    """
    
    def __init__(self, db_session: Session, channel_manager: ChannelManager):
        self.db = db_session
        self.channel_manager = channel_manager
        self.settings = get_settings()
        self._bridge_cache: Dict[str, ConversationBridge] = {}
        
    def generate_global_user_id(self, provider: IdentityProvider, provider_id: str) -> str:
        """Generate a deterministic global user ID from provider info"""
        seed = f"{provider.value}:{provider_id}:{self.settings.SECRET_KEY}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]
    
    def create_or_get_identity(
        self,
        channel: str,
        channel_user_id: str,
        provider: IdentityProvider,
        provider_id: str,
        display_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserIdentity:
        """
        Create or retrieve a user identity mapping.
        Returns the UserIdentity object.
        """
        # Check if mapping already exists
        existing_mapping = self.db.query(ChannelIdentityMapping).filter_by(
            channel=channel,
            channel_user_id=channel_user_id
        ).first()
        
        if existing_mapping:
            self._log_audit(
                existing_mapping.global_user_id,
                "identity_accessed",
                channel,
                channel_user_id,
                {"action": "existing_mapping_found"},
                ip_address,
                user_agent
            )
            return existing_mapping.user_identity
        
        # Check if provider ID already exists
        existing_provider = self.db.query(ChannelIdentityMapping).filter_by(
            provider=provider.value,
            provider_id=provider_id
        ).first()
        
        if existing_provider:
            # Map this channel user to existing global identity
            new_mapping = ChannelIdentityMapping(
                global_user_id=existing_provider.global_user_id,
                channel=channel,
                channel_user_id=channel_user_id,
                provider=provider.value,
                provider_id=provider_id,
                metadata_=metadata or {}
            )
            self.db.add(new_mapping)
            
            self._log_audit(
                existing_provider.global_user_id,
                "channel_mapping_added",
                channel,
                channel_user_id,
                {"provider": provider.value, "provider_id": provider_id},
                ip_address,
                user_agent
            )
            
            self.db.commit()
            return existing_provider.user_identity
        
        # Create new identity
        global_user_id = self.generate_global_user_id(provider, provider_id)
        
        # Check if global user ID already exists (collision handling)
        existing_user = self.db.query(UserIdentity).filter_by(
            global_user_id=global_user_id
        ).first()
        
        if existing_user:
            # Use existing user but add new mapping
            new_mapping = ChannelIdentityMapping(
                global_user_id=global_user_id,
                channel=channel,
                channel_user_id=channel_user_id,
                provider=provider.value,
                provider_id=provider_id,
                metadata_=metadata or {}
            )
            self.db.add(new_mapping)
            
            self._log_audit(
                global_user_id,
                "channel_mapping_added",
                channel,
                channel_user_id,
                {"provider": provider.value, "provider_id": provider_id},
                ip_address,
                user_agent
            )
            
            self.db.commit()
            return existing_user
        
        # Create completely new identity
        user_identity = UserIdentity(
            global_user_id=global_user_id,
            display_name=display_name,
            privacy_level=PrivacyLevel.CONTEXT_ONLY.value,
            opt_in=False  # Default to opt-out for privacy
        )
        
        channel_mapping = ChannelIdentityMapping(
            global_user_id=global_user_id,
            channel=channel,
            channel_user_id=channel_user_id,
            provider=provider.value,
            provider_id=provider_id,
            metadata_=metadata or {}
        )
        
        self.db.add(user_identity)
        self.db.add(channel_mapping)
        
        self._log_audit(
            global_user_id,
            "identity_created",
            channel,
            channel_user_id,
            {"provider": provider.value, "provider_id": provider_id},
            ip_address,
            user_agent
        )
        
        self.db.commit()
        return user_identity
    
    def resolve_global_user_id(self, channel: str, channel_user_id: str) -> Optional[str]:
        """Resolve channel user to global user ID"""
        mapping = self.db.query(ChannelIdentityMapping).filter_by(
            channel=channel,
            channel_user_id=channel_user_id
        ).first()
        
        return mapping.global_user_id if mapping else None
    
    def resolve_channel_user_id(self, global_user_id: str, target_channel: str) -> Optional[str]:
        """Resolve global user to channel-specific user ID"""
        mapping = self.db.query(ChannelIdentityMapping).filter_by(
            global_user_id=global_user_id,
            channel=target_channel
        ).first()
        
        return mapping.channel_user_id if mapping else None
    
    def get_user_channels(self, global_user_id: str) -> List[Dict[str, str]]:
        """Get all channels where user has an identity"""
        mappings = self.db.query(ChannelIdentityMapping).filter_by(
            global_user_id=global_user_id
        ).all()
        
        return [
            {
                "channel": m.channel,
                "channel_user_id": m.channel_user_id,
                "provider": m.provider,
                "verified": m.verified
            }
            for m in mappings
        ]
    
    def update_privacy_settings(
        self,
        global_user_id: str,
        privacy_level: PrivacyLevel,
        opt_in: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Update user's privacy settings for cross-channel sharing"""
        user = self.db.query(UserIdentity).filter_by(
            global_user_id=global_user_id
        ).first()
        
        if not user:
            return False
        
        user.privacy_level = privacy_level.value
        user.opt_in = opt_in
        
        self._log_audit(
            global_user_id,
            "privacy_settings_updated",
            details={
                "privacy_level": privacy_level.value,
                "opt_in": opt_in
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.commit()
        return True
    
    def create_conversation_bridge(
        self,
        global_user_id: str,
        source_channel: str,
        source_conversation_id: str,
        target_channel: str,
        target_conversation_id: Optional[str] = None,
        context: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[ConversationBridge]:
        """
        Create a bridge between conversations in different channels.
        If target_conversation_id is None, a new conversation will be created.
        """
        # Check if user has opted in
        user = self.db.query(UserIdentity).filter_by(
            global_user_id=global_user_id
        ).first()
        
        if not user or not user.opt_in:
            logger.warning(f"User {global_user_id} has not opted in for cross-channel bridging")
            return None
        
        # Check if bridge already exists
        existing_bridge = self.db.query(ConversationBridgeRecord).filter_by(
            global_user_id=global_user_id,
            source_channel=source_channel,
            source_conversation_id=source_conversation_id,
            target_channel=target_channel,
            active=True
        ).first()
        
        if existing_bridge:
            return self._bridge_record_to_object(existing_bridge)
        
        # Generate bridge ID
        bridge_id = f"bridge_{secrets.token_hex(16)}"
        
        # If no target conversation ID, create one
        if not target_conversation_id:
            target_conversation_id = f"bridged_{secrets.token_hex(8)}"
        
        # Create bridge record
        bridge_record = ConversationBridgeRecord(
            bridge_id=bridge_id,
            global_user_id=global_user_id,
            source_channel=source_channel,
            source_conversation_id=source_conversation_id,
            target_channel=target_channel,
            target_conversation_id=target_conversation_id,
            context=context or {},
            active=True
        )
        
        self.db.add(bridge_record)
        
        # Log the bridge creation
        self._log_audit(
            global_user_id,
            "conversation_bridge_created",
            source_channel,
            details={
                "bridge_id": bridge_id,
                "source_conversation_id": source_conversation_id,
                "target_channel": target_channel,
                "target_conversation_id": target_conversation_id
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.commit()
        
        # Cache the bridge
        bridge_obj = self._bridge_record_to_object(bridge_record)
        self._bridge_cache[bridge_id] = bridge_obj
        
        return bridge_obj
    
    def sync_conversation_state(
        self,
        bridge_id: str,
        state_update: Dict[str, Any],
        source_channel: Optional[str] = None
    ) -> bool:
        """
        Sync conversation state across bridged channels.
        Updates the bridge context and notifies target channels.
        """
        bridge_record = self.db.query(ConversationBridgeRecord).filter_by(
            bridge_id=bridge_id,
            active=True
        ).first()
        
        if not bridge_record:
            return False
        
        # Update bridge context
        current_context = bridge_record.context or {}
        current_context.update(state_update)
        current_context["last_updated"] = datetime.utcnow().isoformat()
        
        if source_channel:
            current_context["last_source"] = source_channel
        
        bridge_record.context = current_context
        bridge_record.last_synced = datetime.utcnow()
        
        # Notify target channel if different from source
        if source_channel != bridge_record.target_channel:
            self._notify_channel_of_update(
                bridge_record.target_channel,
                bridge_record.target_conversation_id,
                state_update,
                bridge_record.global_user_id
            )
        
        # Also notify source channel if it's the target that's updating
        if source_channel and source_channel != bridge_record.source_channel:
            self._notify_channel_of_update(
                bridge_record.source_channel,
                bridge_record.source_conversation_id,
                state_update,
                bridge_record.global_user_id
            )
        
        self.db.commit()
        
        # Update cache
        if bridge_id in self._bridge_cache:
            self._bridge_cache[bridge_id].context = current_context
            self._bridge_cache[bridge_id].last_synced = datetime.utcnow()
        
        return True
    
    def get_active_bridges_for_conversation(
        self,
        channel: str,
        conversation_id: str
    ) -> List[ConversationBridge]:
        """Get all active bridges for a specific conversation"""
        bridges = self.db.query(ConversationBridgeRecord).filter(
            ConversationBridgeRecord.active == True,
            (
                (ConversationBridgeRecord.source_channel == channel) &
                (ConversationBridgeRecord.source_conversation_id == conversation_id)
            ) | (
                (ConversationBridgeRecord.target_channel == channel) &
                (ConversationBridgeRecord.target_conversation_id == conversation_id)
            )
        ).all()
        
        return [self._bridge_record_to_object(b) for b in bridges]
    
    def get_bridges_for_user(self, global_user_id: str) -> List[ConversationBridge]:
        """Get all active bridges for a user"""
        bridges = self.db.query(ConversationBridgeRecord).filter_by(
            global_user_id=global_user_id,
            active=True
        ).all()
        
        return [self._bridge_record_to_object(b) for b in bridges]
    
    def deactivate_bridge(
        self,
        bridge_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Deactivate a conversation bridge"""
        bridge_record = self.db.query(ConversationBridgeRecord).filter_by(
            bridge_id=bridge_id
        ).first()
        
        if not bridge_record:
            return False
        
        bridge_record.active = False
        
        self._log_audit(
            bridge_record.global_user_id,
            "conversation_bridge_deactivated",
            bridge_record.source_channel,
            details={"bridge_id": bridge_id},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.commit()
        
        # Remove from cache
        if bridge_id in self._bridge_cache:
            del self._bridge_cache[bridge_id]
        
        return True
    
    def _notify_channel_of_update(
        self,
        channel: str,
        conversation_id: str,
        state_update: Dict[str, Any],
        global_user_id: str
    ):
        """Notify a channel about conversation state updates"""
        try:
            # Get channel instance from manager
            channel_instance = self.channel_manager.get_channel(channel)
            if not channel_instance:
                logger.warning(f"Channel {channel} not found in manager")
                return
            
            # Create a system message about the update
            update_message = ChannelMessage(
                conversation_id=conversation_id,
                content=f"[System] Conversation context updated from another channel",
                sender="system",
                metadata={
                    "type": "bridge_sync",
                    "global_user_id": global_user_id,
                    "state_update": state_update
                }
            )
            
            # Send through channel's message bus
            channel_instance.send_message(update_message)
            
        except Exception as e:
            logger.error(f"Failed to notify channel {channel}: {str(e)}")
    
    def _bridge_record_to_object(self, record: ConversationBridgeRecord) -> ConversationBridge:
        """Convert database record to ConversationBridge object"""
        return ConversationBridge(
            bridge_id=record.bridge_id,
            global_user_id=record.global_user_id,
            source_channel=record.source_channel,
            source_conversation_id=record.source_conversation_id,
            target_channel=record.target_channel,
            target_conversation_id=record.target_conversation_id,
            context=record.context or {},
            active=record.active,
            created_at=record.created_at,
            last_synced=record.last_synced
        )
    
    def _log_audit(
        self,
        global_user_id: str,
        action: str,
        channel: Optional[str] = None,
        channel_user_id: Optional[str] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log an audit event"""
        try:
            audit_log = IdentityAuditLog(
                global_user_id=global_user_id,
                action=action,
                channel=channel,
                channel_user_id=channel_user_id,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            self.db.add(audit_log)
            # Note: We don't commit here, let the caller handle transaction
        except Exception as e:
            logger.error(f"Failed to create audit log: {str(e)}")
    
    def cleanup_expired_bridges(self, max_age_hours: int = 24):
        """Clean up old inactive bridges"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        expired_bridges = self.db.query(ConversationBridgeRecord).filter(
            ConversationBridgeRecord.active == False,
            ConversationBridgeRecord.last_synced < cutoff_time
        ).all()
        
        for bridge in expired_bridges:
            self.db.delete(bridge)
        
        self.db.commit()
        
        # Clean cache
        expired_ids = [b.bridge_id for b in expired_bridges]
        for bridge_id in expired_ids:
            if bridge_id in self._bridge_cache:
                del self._bridge_cache[bridge_id]
        
        return len(expired_bridges)


# Singleton instance for application-wide use
_identity_resolver_instance: Optional[IdentityResolver] = None


def get_identity_resolver(db_session: Session, channel_manager: ChannelManager) -> IdentityResolver:
    """Get or create the singleton IdentityResolver instance"""
    global _identity_resolver_instance
    if _identity_resolver_instance is None:
        _identity_resolver_instance = IdentityResolver(db_session, channel_manager)
    return _identity_resolver_instance


def init_identity_tables(engine):
    """Initialize identity-related database tables"""
    Base.metadata.create_all(engine)