"""Abstract base class for IM channels."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.channels.message_bus import InboundMessage, InboundMessageType, MessageBus, OutboundMessage, ResolvedAttachment

logger = logging.getLogger(__name__)


@dataclass
class ChannelCapabilities:
    """Capability profile for a messaging channel."""
    
    max_message_length: int = 4096
    supports_threads: bool = False
    supports_inline_buttons: bool = False
    supports_cards: bool = False
    supports_file_upload: bool = False
    supported_media_types: list[str] = field(default_factory=list)
    threading_model: str = "none"  # "none", "flat", "nested"
    default_agent_config: str = "general"
    
    def get_optimized_config(self, agent_configs: dict[str, dict]) -> dict:
        """Select the best agent configuration for this channel's capabilities."""
        # Try channel-specific config first
        channel_config_name = f"{self.default_agent_config}"
        if channel_config_name in agent_configs:
            return agent_configs[channel_config_name]
        
        # Fall back to general config
        return agent_configs.get("general", {})


class Channel(ABC):
    """Base class for all IM channel implementations.

    Each channel connects to an external messaging platform and:
    1. Receives messages, wraps them as InboundMessage, publishes to the bus.
    2. Subscribes to outbound messages and sends replies back to the platform.

    Subclasses must implement ``start``, ``stop``, and ``send``.
    """

    def __init__(self, name: str, bus: MessageBus, config: dict[str, Any]) -> None:
        self.name = name
        self.bus = bus
        self.config = config
        self._running = False
        self._capabilities = self._load_capabilities()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def capabilities(self) -> ChannelCapabilities:
        """Return the capability profile for this channel."""
        return self._capabilities

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages from the external platform."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the channel."""

    # -- outbound ----------------------------------------------------------

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message back to the external platform.

        The implementation should use ``msg.chat_id`` and ``msg.thread_ts``
        to route the reply to the correct conversation/thread.
        """

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        """Upload a single file attachment to the platform.

        Returns True if the upload succeeded, False otherwise.
        Default implementation returns False (no file upload support).
        """
        return False

    # -- capabilities ------------------------------------------------------

    def _load_capabilities(self) -> ChannelCapabilities:
        """Load channel capabilities from configuration or defaults."""
        caps_config = self.config.get("capabilities", {})
        return ChannelCapabilities(
            max_message_length=caps_config.get("max_message_length", 4096),
            supports_threads=caps_config.get("supports_threads", False),
            supports_inline_buttons=caps_config.get("supports_inline_buttons", False),
            supports_cards=caps_config.get("supports_cards", False),
            supports_file_upload=caps_config.get("supports_file_upload", False),
            supported_media_types=caps_config.get("supported_media_types", []),
            threading_model=caps_config.get("threading_model", "none"),
            default_agent_config=caps_config.get("default_agent_config", "general"),
        )

    def get_optimized_agent_config(self, agent_configs: dict[str, dict]) -> dict:
        """Get the optimized agent configuration for this channel."""
        return self.capabilities.get_optimized_config(agent_configs)

    # -- helpers -----------------------------------------------------------

    def _make_inbound(
        self,
        chat_id: str,
        user_id: str,
        text: str,
        *,
        msg_type: InboundMessageType = InboundMessageType.CHAT,
        thread_ts: str | None = None,
        files: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InboundMessage:
        """Convenience factory for creating InboundMessage instances."""
        return InboundMessage(
            channel_name=self.name,
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            msg_type=msg_type,
            thread_ts=thread_ts,
            files=files or [],
            metadata=metadata or {},
        )

    async def _on_outbound(self, msg: OutboundMessage) -> None:
        """Outbound callback registered with the bus.

        Only forwards messages targeted at this channel.
        Sends the text message first, then uploads any file attachments.
        File uploads are skipped entirely when the text send fails to avoid
        partial deliveries (files without accompanying text).
        """
        if msg.channel_name == self.name:
            try:
                await self.send(msg)
            except Exception:
                logger.exception("Failed to send outbound message on channel %s", self.name)
                return  # Do not attempt file uploads when the text message failed

            for attachment in msg.attachments:
                try:
                    success = await self.send_file(msg, attachment)
                    if not success:
                        logger.warning("[%s] file upload skipped for %s", self.name, attachment.filename)
                except Exception:
                    logger.exception("[%s] failed to upload file %s", self.name, attachment.filename)