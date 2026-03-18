"""
Channel-Aware Agent Routing for Deer Flow

Routes messages to specialized agent configurations based on channel capabilities,
platform-specific optimizations, and A/B testing for prompts.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta

from backend.app.channels.base import BaseChannel
from backend.app.channels.slack import SlackChannel
from backend.app.channels.telegram import TelegramChannel
from backend.app.channels.feishu import FeishuChannel
from backend.app.channels.manager import ChannelManager

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Supported channel types."""
    SLACK = "slack"
    TELEGRAM = "telegram"
    FEISHU = "feishu"
    UNKNOWN = "unknown"


class ThreadingModel(Enum):
    """Threading models supported by different channels."""
    NONE = "none"  # No threading support
    FLAT = "flat"  # Linear replies (Telegram)
    THREADED = "threaded"  # Nested threads (Slack)
    TOPIC = "topic"  # Topic-based threading (Feishu)


@dataclass
class MediaCapability:
    """Media type capabilities for a channel."""
    images: bool = True
    videos: bool = False
    files: bool = True
    audio: bool = False
    documents: bool = True
    max_file_size_mb: int = 50
    supported_image_formats: Set[str] = field(default_factory=lambda: {"jpg", "jpeg", "png", "gif"})
    supported_video_formats: Set[str] = field(default_factory=lambda: {"mp4", "mov"})


@dataclass
class ChannelCapabilityProfile:
    """Complete capability profile for a channel type."""
    channel_type: ChannelType
    max_message_length: int
    threading_model: ThreadingModel
    supports_markdown: bool
    supports_html: bool
    supports_buttons: bool
    supports_inline_keyboards: bool
    supports_cards: bool
    supports_rich_formatting: bool
    supports_reactions: bool
    supports_edits: bool
    supports_deletes: bool
    supports_threads: bool
    supports_topics: bool
    media: MediaCapability
    rate_limit_per_second: int
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    default_language: str = "en"
    rtl_support: bool = False
    custom_emojis: bool = False
    user_mentions: bool = True
    channel_mentions: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["channel_type"] = self.channel_type.value
        data["threading_model"] = self.threading_model.value
        return data


@dataclass
class AgentVariant:
    """Agent configuration variant for A/B testing."""
    variant_id: str
    name: str
    description: str
    config: Dict[str, Any]
    weight: float = 1.0  # For weighted A/B testing
    is_control: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "weight": self.weight,
            "is_control": self.is_control,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ChannelAgentConfig:
    """Agent configuration optimized for a specific channel."""
    channel_type: ChannelType
    base_config: Dict[str, Any]
    variants: List[AgentVariant] = field(default_factory=list)
    active_variant_id: Optional[str] = None
    routing_rules: Dict[str, Any] = field(default_factory=dict)
    fallback_config: Optional[Dict[str, Any]] = None
    
    def get_variant(self, variant_id: str) -> Optional[AgentVariant]:
        """Get a specific variant by ID."""
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        return None
    
    def select_variant(self, user_id: str, context: Dict[str, Any]) -> AgentVariant:
        """Select a variant for A/B testing based on user ID and context."""
        if not self.variants:
            # Return base config as a variant
            return AgentVariant(
                variant_id="base",
                name="Base Configuration",
                description="Default configuration",
                config=self.base_config,
                is_control=True
            )
        
        # Use consistent hashing for deterministic variant assignment
        hash_input = f"{user_id}:{self.channel_type.value}:{context.get('session_id', '')}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Weighted random selection
        total_weight = sum(v.weight for v in self.variants)
        if total_weight <= 0:
            return self.variants[0]
        
        normalized_hash = (hash_value % 10000) / 10000.0 * total_weight
        cumulative_weight = 0.0
        
        for variant in self.variants:
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                return variant
        
        return self.variants[-1]


class ChannelRouter:
    """
    Routes messages to specialized agent configurations based on channel capabilities.
    
    Features:
    - Channel capability detection and profiling
    - Platform-specific agent configuration selection
    - A/B testing framework for channel-specific prompts
    - Performance metrics tracking per channel/variant
    - Fallback handling for unknown channels
    """
    
    def __init__(self, channel_manager: ChannelManager):
        self.channel_manager = channel_manager
        self._capability_profiles: Dict[ChannelType, ChannelCapabilityProfile] = {}
        self._agent_configs: Dict[ChannelType, ChannelAgentConfig] = {}
        self._metrics: Dict[str, Any] = {
            "total_routings": 0,
            "by_channel": {},
            "by_variant": {},
            "errors": 0
        }
        self._initialize_default_profiles()
        self._initialize_default_configs()
    
    def _initialize_default_profiles(self):
        """Initialize default capability profiles for known channels."""
        
        # Slack capabilities
        self._capability_profiles[ChannelType.SLACK] = ChannelCapabilityProfile(
            channel_type=ChannelType.SLACK,
            max_message_length=4000,
            threading_model=ThreadingModel.THREADED,
            supports_markdown=True,
            supports_html=False,
            supports_buttons=True,
            supports_inline_keyboards=False,
            supports_cards=True,
            supports_rich_formatting=True,
            supports_reactions=True,
            supports_edits=True,
            supports_deletes=True,
            supports_threads=True,
            supports_topics=False,
            media=MediaCapability(
                images=True,
                videos=True,
                files=True,
                audio=True,
                documents=True,
                max_file_size_mb=1024,
                supported_image_formats={"jpg", "jpeg", "png", "gif", "bmp", "webp"},
                supported_video_formats={"mp4", "mov", "avi", "mkv", "webm"}
            ),
            rate_limit_per_second=1,
            rate_limit_per_minute=50,
            rate_limit_per_hour=1000,
            custom_emojis=True,
            channel_mentions=True
        )
        
        # Telegram capabilities
        self._capability_profiles[ChannelType.TELEGRAM] = ChannelCapabilityProfile(
            channel_type=ChannelType.TELEGRAM,
            max_message_length=4096,
            threading_model=ThreadingModel.FLAT,
            supports_markdown=True,
            supports_html=True,
            supports_buttons=True,
            supports_inline_keyboards=True,
            supports_cards=False,
            supports_rich_formatting=True,
            supports_reactions=True,
            supports_edits=True,
            supports_deletes=True,
            supports_threads=False,
            supports_topics=False,
            media=MediaCapability(
                images=True,
                videos=True,
                files=True,
                audio=True,
                documents=True,
                max_file_size_mb=2048,
                supported_image_formats={"jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"},
                supported_video_formats={"mp4", "mov", "avi", "mkv", "webm"}
            ),
            rate_limit_per_second=1,
            rate_limit_per_minute=30,
            rate_limit_per_hour=1000,
            rtl_support=True
        )
        
        # Feishu capabilities
        self._capability_profiles[ChannelType.FEISHU] = ChannelCapabilityProfile(
            channel_type=ChannelType.FEISHU,
            max_message_length=30000,
            threading_model=ThreadingModel.TOPIC,
            supports_markdown=True,
            supports_html=False,
            supports_buttons=True,
            supports_inline_keyboards=False,
            supports_cards=True,
            supports_rich_formatting=True,
            supports_reactions=True,
            supports_edits=True,
            supports_deletes=True,
            supports_threads=True,
            supports_topics=True,
            media=MediaCapability(
                images=True,
                videos=True,
                files=True,
                audio=True,
                documents=True,
                max_file_size_mb=1024,
                supported_image_formats={"jpg", "jpeg", "png", "gif", "bmp"},
                supported_video_formats={"mp4", "mov", "avi"}
            ),
            rate_limit_per_second=2,
            rate_limit_per_minute=60,
            rate_limit_per_hour=2000,
            default_language="zh"
        )
    
    def _initialize_default_configs(self):
        """Initialize default agent configurations for each channel."""
        
        # Slack-optimized configuration
        self._agent_configs[ChannelType.SLACK] = ChannelAgentConfig(
            channel_type=ChannelType.SLACK,
            base_config={
                "max_response_length": 3500,
                "use_threads": True,
                "include_channel_context": True,
                "formatting": "slack_mrkdwn",
                "response_style": "professional_concise",
                "tools": ["web_search", "code_execution", "file_analysis"],
                "memory_window": 10,
                "enable_reactions": True,
                "enable_cards": True
            },
            variants=[
                AgentVariant(
                    variant_id="slack_concise",
                    name="Concise Responses",
                    description="Shorter, more direct responses for Slack",
                    config={
                        "max_response_length": 2000,
                        "response_style": "very_concise",
                        "use_bullet_points": True,
                        "include_examples": False
                    },
                    weight=0.3
                ),
                AgentVariant(
                    variant_id="slack_detailed",
                    name="Detailed Responses",
                    description="More comprehensive responses with examples",
                    config={
                        "max_response_length": 3500,
                        "response_style": "detailed",
                        "use_bullet_points": False,
                        "include_examples": True,
                        "include_code_snippets": True
                    },
                    weight=0.3
                ),
                AgentVariant(
                    variant_id="slack_control",
                    name="Control Group",
                    description="Default configuration",
                    config={},  # Uses base_config
                    weight=0.4,
                    is_control=True
                )
            ],
            routing_rules={
                "thread_replies": "use_thread_context",
                "direct_messages": "personalized",
                "channel_messages": "channel_aware"
            }
        )
        
        # Telegram-optimized configuration
        self._agent_configs[ChannelType.TELEGRAM] = ChannelAgentConfig(
            channel_type=ChannelType.TELEGRAM,
            base_config={
                "max_response_length": 4000,
                "use_inline_keyboards": True,
                "formatting": "telegram_html",
                "response_style": "friendly_engaging",
                "tools": ["web_search", "image_generation", "translation"],
                "memory_window": 5,
                "enable_buttons": True,
                "enable_inline_mode": True
            },
            variants=[
                AgentVariant(
                    variant_id="telegram_quick",
                    name="Quick Replies",
                    description="Fast, button-driven interactions",
                    config={
                        "max_response_length": 1000,
                        "response_style": "quick_answers",
                        "heavy_button_usage": True,
                        "enable_quick_replies": True
                    },
                    weight=0.4
                ),
                AgentVariant(
                    variant_id="telegram_conversational",
                    name="Conversational",
                    description="More natural, conversational responses",
                    config={
                        "max_response_length": 3000,
                        "response_style": "conversational",
                        "use_emojis": True,
                        "ask_followup_questions": True
                    },
                    weight=0.4
                ),
                AgentVariant(
                    variant_id="telegram_control",
                    name="Control Group",
                    description="Default configuration",
                    config={},
                    weight=0.2,
                    is_control=True
                )
            ],
            routing_rules={
                "private_chats": "personalized",
                "group_chats": "group_aware",
                "inline_queries": "inline_optimized"
            }
        )
        
        # Feishu-optimized configuration
        self._agent_configs[ChannelType.FEISHU] = ChannelAgentConfig(
            channel_type=ChannelType.FEISHU,
            base_config={
                "max_response_length": 25000,
                "use_cards": True,
                "formatting": "feishu_markdown",
                "response_style": "professional_structured",
                "tools": ["document_analysis", "calendar_integration", "task_management"],
                "memory_window": 15,
                "enable_topics": True,
                "enable_rich_cards": True,
                "localization": "zh-CN"
            },
            variants=[
                AgentVariant(
                    variant_id="feishu_cards",
                    name="Card-Heavy",
                    description="Use rich cards for complex information",
                    config={
                        "prefer_cards": True,
                        "card_template": "detailed",
                        "include_interactive_elements": True
                    },
                    weight=0.5
                ),
                AgentVariant(
                    variant_id="feishu_text",
                    name="Text-Optimized",
                    description="Focus on well-formatted text responses",
                    config={
                        "prefer_cards": False,
                        "max_response_length": 20000,
                        "use_structured_text": True
                    },
                    weight=0.3
                ),
                AgentVariant(
                    variant_id="feishu_control",
                    name="Control Group",
                    description="Default configuration",
                    config={},
                    weight=0.2,
                    is_control=True
                )
            ],
            routing_rules={
                "group_chats": "topic_aware",
                "direct_messages": "personalized",
                "document_collaboration": "collaborative"
            }
        )
    
    def register_capability_profile(self, profile: ChannelCapabilityProfile):
        """Register or update a capability profile for a channel type."""
        self._capability_profiles[profile.channel_type] = profile
        logger.info(f"Registered capability profile for {profile.channel_type.value}")
    
    def register_agent_config(self, config: ChannelAgentConfig):
        """Register or update an agent configuration for a channel type."""
        self._agent_configs[config.channel_type] = config
        logger.info(f"Registered agent config for {config.channel_type.value}")
    
    def add_variant(self, channel_type: ChannelType, variant: AgentVariant):
        """Add a new A/B testing variant to a channel configuration."""
        if channel_type not in self._agent_configs:
            raise ValueError(f"No configuration found for channel type: {channel_type}")
        
        config = self._agent_configs[channel_type]
        existing_variant = config.get_variant(variant.variant_id)
        
        if existing_variant:
            # Update existing variant
            config.variants = [v for v in config.variants if v.variant_id != variant.variant_id]
        
        config.variants.append(variant)
        logger.info(f"Added variant {variant.variant_id} to {channel_type.value}")
    
    def get_channel_type(self, channel: BaseChannel) -> ChannelType:
        """Determine the channel type from a channel instance."""
        if isinstance(channel, SlackChannel):
            return ChannelType.SLACK
        elif isinstance(channel, TelegramChannel):
            return ChannelType.TELEGRAM
        elif isinstance(channel, FeishuChannel):
            return ChannelType.FEISHU
        else:
            # Try to infer from channel name or attributes
            channel_name = channel.__class__.__name__.lower()
            for channel_type in ChannelType:
                if channel_type.value in channel_name:
                    return channel_type
            return ChannelType.UNKNOWN
    
    def get_capability_profile(self, channel_type: ChannelType) -> ChannelCapabilityProfile:
        """Get capability profile for a channel type."""
        if channel_type in self._capability_profiles:
            return self._capability_profiles[channel_type]
        
        # Return a generic profile for unknown channels
        return ChannelCapabilityProfile(
            channel_type=ChannelType.UNKNOWN,
            max_message_length=2000,
            threading_model=ThreadingModel.NONE,
            supports_markdown=False,
            supports_html=False,
            supports_buttons=False,
            supports_inline_keyboards=False,
            supports_cards=False,
            supports_rich_formatting=False,
            supports_reactions=False,
            supports_edits=False,
            supports_deletes=False,
            supports_threads=False,
            supports_topics=False,
            media=MediaCapability(
                images=True,
                videos=False,
                files=True,
                audio=False,
                documents=False,
                max_file_size_mb=10
            ),
            rate_limit_per_second=1,
            rate_limit_per_minute=20,
            rate_limit_per_hour=500
        )
    
    def get_agent_config(self, channel_type: ChannelType) -> Optional[ChannelAgentConfig]:
        """Get agent configuration for a channel type."""
        return self._agent_configs.get(channel_type)
    
    def route_message(
        self,
        channel: BaseChannel,
        user_id: str,
        message: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], AgentVariant]:
        """
        Route a message to the appropriate agent configuration.
        
        Args:
            channel: The channel instance
            user_id: User identifier for A/B testing
            message: The incoming message
            context: Additional context (session_id, thread_id, etc.)
            
        Returns:
            Tuple of (agent_config, selected_variant)
        """
        context = context or {}
        channel_type = self.get_channel_type(channel)
        
        # Update metrics
        self._metrics["total_routings"] += 1
        self._metrics["by_channel"][channel_type.value] = \
            self._metrics["by_channel"].get(channel_type.value, 0) + 1
        
        # Get capability profile
        capability_profile = self.get_capability_profile(channel_type)
        
        # Get agent configuration
        agent_config = self.get_agent_config(channel_type)
        
        if not agent_config:
            logger.warning(f"No agent config for {channel_type.value}, using fallback")
            agent_config = self._create_fallback_config(channel_type, capability_profile)
        
        # Select variant for A/B testing
        variant = agent_config.select_variant(user_id, context)
        
        # Update variant metrics
        variant_key = f"{channel_type.value}:{variant.variant_id}"
        self._metrics["by_variant"][variant_key] = \
            self._metrics["by_variant"].get(variant_key, 0) + 1
        
        # Merge base config with variant config
        final_config = {**agent_config.base_config, **variant.config}
        
        # Apply channel-specific optimizations
        optimized_config = self._apply_channel_optimizations(
            final_config,
            capability_profile,
            message,
            context
        )
        
        logger.debug(
            f"Routed message for {user_id} in {channel_type.value} "
            f"to variant {variant.variant_id}"
        )
        
        return optimized_config, variant
    
    def _create_fallback_config(
        self,
        channel_type: ChannelType,
        capability_profile: ChannelCapabilityProfile
    ) -> ChannelAgentConfig:
        """Create a fallback configuration for unknown channels."""
        return ChannelAgentConfig(
            channel_type=channel_type,
            base_config={
                "max_response_length": min(capability_profile.max_message_length, 2000),
                "formatting": "plain_text",
                "response_style": "neutral",
                "tools": ["web_search"],
                "memory_window": 5
            },
            fallback_config={
                "use_basic_responses": True,
                "avoid_complex_formatting": True
            }
        )
    
    def _apply_channel_optimizations(
        self,
        config: Dict[str, Any],
        capability_profile: ChannelCapabilityProfile,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply channel-specific optimizations to the configuration."""
        optimized = config.copy()
        
        # Adjust response length based on channel limits
        if "max_response_length" in optimized:
            optimized["max_response_length"] = min(
                optimized["max_response_length"],
                capability_profile.max_message_length
            )
        
        # Disable unsupported features
        if not capability_profile.supports_markdown:
            optimized["formatting"] = "plain_text"
        
        if not capability_profile.supports_buttons and "enable_buttons" in optimized:
            optimized["enable_buttons"] = False
        
        if not capability_profile.supports_cards and "enable_cards" in optimized:
            optimized["enable_cards"] = False
        
        # Thread-aware optimizations
        if capability_profile.supports_threads and context.get("thread_id"):
            optimized["use_thread_context"] = True
            optimized["thread_id"] = context["thread_id"]
        
        # Media handling based on capabilities
        if message.get("has_media"):
            media_type = message.get("media_type")
            if media_type == "image" and not capability_profile.media.images:
                optimized["handle_media"] = "unsupported"
            elif media_type == "video" and not capability_profile.media.videos:
                optimized["handle_media"] = "unsupported"
        
        # Language/localization
        if "localization" not in optimized:
            optimized["localization"] = capability_profile.default_language
        
        return optimized
    
    def record_variant_metric(
        self,
        channel_type: ChannelType,
        variant_id: str,
        metric_name: str,
        value: Any
    ):
        """Record a metric for a specific variant."""
        config = self.get_agent_config(channel_type)
        if not config:
            return
        
        variant = config.get_variant(variant_id)
        if not variant:
            return
        
        if metric_name not in variant.metrics:
            variant.metrics[metric_name] = []
        
        variant.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 1000 metrics per type
        if len(variant.metrics[metric_name]) > 1000:
            variant.metrics[metric_name] = variant.metrics[metric_name][-1000:]
    
    def get_variant_performance(
        self,
        channel_type: ChannelType,
        variant_id: str,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a variant."""
        config = self.get_agent_config(channel_type)
        if not config:
            return {}
        
        variant = config.get_variant(variant_id)
        if not variant or metric_name not in variant.metrics:
            return {}
        
        metrics = variant.metrics[metric_name]
        
        if time_window:
            cutoff = datetime.utcnow() - time_window
            metrics = [
                m for m in metrics
                if datetime.fromisoformat(m["timestamp"]) > cutoff
            ]
        
        if not metrics:
            return {}
        
        values = [m["value"] for m in metrics]
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "time_window": time_window.total_seconds() if time_window else None
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and metrics."""
        return {
            **self._metrics,
            "active_variants": {
                channel_type.value: len(config.variants)
                for channel_type, config in self._agent_configs.items()
            },
            "registered_channels": list(self._capability_profiles.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_configs(self) -> Dict[str, Any]:
        """Export all configurations for backup or analysis."""
        return {
            "capability_profiles": {
                channel_type.value: profile.to_dict()
                for channel_type, profile in self._capability_profiles.items()
            },
            "agent_configs": {
                channel_type.value: {
                    "base_config": config.base_config,
                    "variants": [v.to_dict() for v in config.variants],
                    "routing_rules": config.routing_rules
                }
                for channel_type, config in self._agent_configs.items()
            }
        }
    
    def import_configs(self, data: Dict[str, Any]):
        """Import configurations from exported data."""
        # Import capability profiles
        for channel_type_str, profile_data in data.get("capability_profiles", {}).items():
            try:
                channel_type = ChannelType(channel_type_str)
                profile_data["channel_type"] = channel_type
                profile_data["threading_model"] = ThreadingModel(profile_data["threading_model"])
                profile_data["media"] = MediaCapability(**profile_data["media"])
                profile = ChannelCapabilityProfile(**profile_data)
                self.register_capability_profile(profile)
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to import profile for {channel_type_str}: {e}")
        
        # Import agent configs
        for channel_type_str, config_data in data.get("agent_configs", {}).items():
            try:
                channel_type = ChannelType(channel_type_str)
                
                variants = []
                for variant_data in config_data.get("variants", []):
                    variant_data["created_at"] = datetime.fromisoformat(variant_data["created_at"])
                    variants.append(AgentVariant(**variant_data))
                
                config = ChannelAgentConfig(
                    channel_type=channel_type,
                    base_config=config_data["base_config"],
                    variants=variants,
                    routing_rules=config_data.get("routing_rules", {})
                )
                self.register_agent_config(config)
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to import config for {channel_type_str}: {e}")


# Singleton instance for application-wide use
_channel_router: Optional[ChannelRouter] = None


def get_channel_router(channel_manager: Optional[ChannelManager] = None) -> ChannelRouter:
    """Get or create the singleton ChannelRouter instance."""
    global _channel_router
    
    if _channel_router is None:
        if channel_manager is None:
            from backend.app.channels.manager import get_channel_manager
            channel_manager = get_channel_manager()
        
        _channel_router = ChannelRouter(channel_manager)
    
    return _channel_router


def reset_channel_router():
    """Reset the singleton instance (useful for testing)."""
    global _channel_router
    _channel_router = None