"""MessageBus — async pub/sub hub that decouples channels from the agent dispatcher."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class InboundMessageType(StrEnum):
    """Types of messages arriving from IM channels."""

    CHAT = "chat"
    COMMAND = "command"


@dataclass
class InboundMessage:
    """A message arriving from an IM channel toward the agent dispatcher.

    Attributes:
        channel_name: Name of the source channel (e.g. "feishu", "slack").
        chat_id: Platform-specific chat/conversation identifier.
        user_id: Platform-specific user identifier.
        text: The message text.
        msg_type: Whether this is a regular chat message or a command.
        thread_ts: Optional platform thread identifier (for threaded replies).
        topic_id: Conversation topic identifier used to map to a DeerFlow thread.
            Messages sharing the same ``topic_id`` within a ``chat_id`` will
            reuse the same DeerFlow thread.  When ``None``, each message
            creates a new thread (one-shot Q&A).
        files: Optional list of file attachments (platform-specific dicts).
        metadata: Arbitrary extra data from the channel.
        created_at: Unix timestamp when the message was created.
        cross_channel_id: Optional cross-channel conversation identifier for
            maintaining continuity across different platforms.
        user_global_id: Optional globally unique user identifier for cross-
            channel identity mapping.
        cross_channel_consent: Whether the user has consented to cross-channel
            conversation continuity.
    """

    channel_name: str
    chat_id: str
    user_id: str
    text: str
    msg_type: InboundMessageType = InboundMessageType.CHAT
    thread_ts: str | None = None
    topic_id: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cross_channel_id: str | None = None
    user_global_id: str | None = None
    cross_channel_consent: bool = False


@dataclass
class ResolvedAttachment:
    """A file attachment resolved to a host filesystem path, ready for upload.

    Attributes:
        virtual_path: Original virtual path (e.g. /mnt/user-data/outputs/report.pdf).
        actual_path: Resolved host filesystem path.
        filename: Basename of the file.
        mime_type: MIME type (e.g. "application/pdf").
        size: File size in bytes.
        is_image: True for image/* MIME types (platforms may handle images differently).
    """

    virtual_path: str
    actual_path: Path
    filename: str
    mime_type: str
    size: int
    is_image: bool


@dataclass
class OutboundMessage:
    """A message from the agent dispatcher back to a channel.

    Attributes:
        channel_name: Target channel name (used for routing).
        chat_id: Target chat/conversation identifier.
        thread_id: DeerFlow thread ID that produced this response.
        text: The response text.
        artifacts: List of artifact paths produced by the agent.
        is_final: Whether this is the final message in the response stream.
        thread_ts: Optional platform thread identifier for threaded replies.
        metadata: Arbitrary extra data.
        created_at: Unix timestamp.
        cross_channel_id: Optional cross-channel conversation identifier for
            maintaining continuity across different platforms.
        user_global_id: Optional globally unique user identifier for cross-
            channel identity mapping.
        cross_channel_consent: Whether the user has consented to cross-channel
            conversation continuity.
    """

    channel_name: str
    chat_id: str
    thread_id: str
    text: str
    artifacts: list[str] = field(default_factory=list)
    attachments: list[ResolvedAttachment] = field(default_factory=list)
    is_final: bool = True
    thread_ts: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cross_channel_id: str | None = None
    user_global_id: str | None = None
    cross_channel_consent: bool = False


# ---------------------------------------------------------------------------
# Cross-Channel Identity & Continuity Manager
# ---------------------------------------------------------------------------

@dataclass
class CrossChannelState:
    """State for maintaining cross-channel conversation continuity.

    Attributes:
        cross_channel_id: Unique identifier for the cross-channel conversation.
        user_global_id: Globally unique user identifier.
        thread_id: Associated DeerFlow thread ID.
        last_channel: Last channel where conversation was active.
        last_chat_id: Last chat_id where conversation was active.
        last_thread_ts: Last platform thread identifier.
        created_at: When this cross-channel state was created.
        updated_at: When this cross-channel state was last updated.
        consent_granted: Whether user has granted cross-channel consent.
        audit_log: List of audit events for this conversation.
    """

    cross_channel_id: str
    user_global_id: str
    thread_id: str
    last_channel: str
    last_chat_id: str
    last_thread_ts: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    consent_granted: bool = False
    audit_log: list[dict[str, Any]] = field(default_factory=list)


class CrossChannelManager:
    """Manages cross-channel conversation continuity and user identity mapping."""

    def __init__(self) -> None:
        # Maps user_global_id -> {channel_name: user_id}
        self._user_identities: dict[str, dict[str, str]] = {}
        # Maps cross_channel_id -> CrossChannelState
        self._cross_channel_states: dict[str, CrossChannelState] = {}
        # Maps (channel_name, user_id) -> user_global_id
        self._channel_user_to_global: dict[tuple[str, str], str] = {}
        # Maps thread_id -> cross_channel_id
        self._thread_to_cross_channel: dict[str, str] = {}

    def register_user_identity(
        self, user_global_id: str, channel_name: str, user_id: str
    ) -> None:
        """Register a user's identity across channels."""
        if user_global_id not in self._user_identities:
            self._user_identities[user_global_id] = {}
        self._user_identities[user_global_id][channel_name] = user_id
        self._channel_user_to_global[(channel_name, user_id)] = user_global_id

    def get_global_user_id(self, channel_name: str, user_id: str) -> str | None:
        """Get the global user ID for a channel-specific user ID."""
        return self._channel_user_to_global.get((channel_name, user_id))

    def create_cross_channel_conversation(
        self,
        user_global_id: str,
        thread_id: str,
        channel_name: str,
        chat_id: str,
        thread_ts: str | None = None,
        consent_granted: bool = False,
    ) -> str:
        """Create a new cross-channel conversation and return its ID."""
        cross_channel_id = str(uuid.uuid4())
        state = CrossChannelState(
            cross_channel_id=cross_channel_id,
            user_global_id=user_global_id,
            thread_id=thread_id,
            last_channel=channel_name,
            last_chat_id=chat_id,
            last_thread_ts=thread_ts,
            consent_granted=consent_granted,
        )
        state.audit_log.append({
            "event": "conversation_created",
            "timestamp": time.time(),
            "channel": channel_name,
            "chat_id": chat_id,
        })
        self._cross_channel_states[cross_channel_id] = state
        self._thread_to_cross_channel[thread_id] = cross_channel_id
        return cross_channel_id

    def get_cross_channel_state(self, cross_channel_id: str) -> CrossChannelState | None:
        """Get the state for a cross-channel conversation."""
        return self._cross_channel_states.get(cross_channel_id)

    def get_cross_channel_id_by_thread(self, thread_id: str) -> str | None:
        """Get the cross-channel ID associated with a thread."""
        return self._thread_to_cross_channel.get(thread_id)

    def update_cross_channel_state(
        self,
        cross_channel_id: str,
        channel_name: str,
        chat_id: str,
        thread_ts: str | None = None,
    ) -> None:
        """Update the cross-channel state with new channel information."""
        if cross_channel_id in self._cross_channel_states:
            state = self._cross_channel_states[cross_channel_id]
            state.last_channel = channel_name
            state.last_chat_id = chat_id
            state.last_thread_ts = thread_ts
            state.updated_at = time.time()
            state.audit_log.append({
                "event": "conversation_continued",
                "timestamp": time.time(),
                "channel": channel_name,
                "chat_id": chat_id,
            })

    def log_audit_event(
        self, cross_channel_id: str, event: str, details: dict[str, Any]
    ) -> None:
        """Log an audit event for a cross-channel conversation."""
        if cross_channel_id in self._cross_channel_states:
            self._cross_channel_states[cross_channel_id].audit_log.append({
                "event": event,
                "timestamp": time.time(),
                **details,
            })


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

OutboundCallback = Callable[[OutboundMessage], Coroutine[Any, Any, None]]


class MessageBus:
    """Async pub/sub hub connecting channels and the agent dispatcher.

    Channels publish inbound messages; the dispatcher consumes them.
    The dispatcher publishes outbound messages; channels receive them
    via registered callbacks.
    """

    def __init__(self) -> None:
        self._inbound_queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._outbound_listeners: list[OutboundCallback] = []
        self._cross_channel_manager = CrossChannelManager()

    # -- inbound -----------------------------------------------------------

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Enqueue an inbound message from a channel."""
        # Handle cross-channel continuity if enabled
        if msg.cross_channel_consent and msg.user_global_id:
            # Check if this continues an existing cross-channel conversation
            if msg.cross_channel_id:
                # Update existing cross-channel state
                self._cross_channel_manager.update_cross_channel_state(
                    msg.cross_channel_id,
                    msg.channel_name,
                    msg.chat_id,
                    msg.thread_ts,
                )
                self._cross_channel_manager.log_audit_event(
                    msg.cross_channel_id,
                    "message_received",
                    {
                        "channel": msg.channel_name,
                        "chat_id": msg.chat_id,
                        "user_id": msg.user_id,
                    },
                )
            else:
                # Check if user has an existing conversation in another channel
                existing_state = None
                for state in self._cross_channel_manager._cross_channel_states.values():
                    if (
                        state.user_global_id == msg.user_global_id
                        and state.consent_granted
                    ):
                        existing_state = state
                        break

                if existing_state:
                    # Continue existing cross-channel conversation
                    msg.cross_channel_id = existing_state.cross_channel_id
                    self._cross_channel_manager.update_cross_channel_state(
                        existing_state.cross_channel_id,
                        msg.channel_name,
                        msg.chat_id,
                        msg.thread_ts,
                    )
                    self._cross_channel_manager.log_audit_event(
                        existing_state.cross_channel_id,
                        "conversation_switched",
                        {
                            "from_channel": existing_state.last_channel,
                            "to_channel": msg.channel_name,
                            "user_id": msg.user_id,
                        },
                    )

        await self._inbound_queue.put(msg)
        logger.info(
            "[Bus] inbound enqueued: channel=%s, chat_id=%s, type=%s, "
            "cross_channel=%s, user_global=%s, queue_size=%d",
            msg.channel_name,
            msg.chat_id,
            msg.msg_type.value,
            msg.cross_channel_id,
            msg.user_global_id,
            self._inbound_queue.qsize(),
        )

    async def get_inbound(self) -> InboundMessage:
        """Block until the next inbound message is available."""
        return await self._inbound_queue.get()

    @property
    def inbound_queue(self) -> asyncio.Queue[InboundMessage]:
        return self._inbound_queue

    # -- outbound ----------------------------------------------------------

    def subscribe_outbound(self, callback: OutboundCallback) -> None:
        """Register an async callback for outbound messages."""
        self._outbound_listeners.append(callback)

    def unsubscribe_outbound(self, callback: OutboundCallback) -> None:
        """Remove a previously registered outbound callback."""
        self._outbound_listeners = [cb for cb in self._outbound_listeners if cb is not callback]

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Dispatch an outbound message to all registered listeners."""
        # Handle cross-channel continuity for outbound messages
        if msg.cross_channel_consent and msg.user_global_id:
            if msg.cross_channel_id:
                self._cross_channel_manager.log_audit_event(
                    msg.cross_channel_id,
                    "response_sent",
                    {
                        "channel": msg.channel_name,
                        "chat_id": msg.chat_id,
                        "thread_id": msg.thread_id,
                    },
                )

        logger.info(
            "[Bus] outbound dispatching: channel=%s, chat_id=%s, "
            "cross_channel=%s, user_global=%s, listeners=%d, text_len=%d",
            msg.channel_name,
            msg.chat_id,
            msg.cross_channel_id,
            msg.user_global_id,
            len(self._outbound_listeners),
            len(msg.text),
        )
        for callback in self._outbound_listeners:
            try:
                await callback(msg)
            except Exception:
                logger.exception("Error in outbound callback for channel=%s", msg.channel_name)

    # -- cross-channel management -----------------------------------------

    def get_cross_channel_manager(self) -> CrossChannelManager:
        """Get the cross-channel manager instance."""
        return self._cross_channel_manager

    def register_user_identity(
        self, user_global_id: str, channel_name: str, user_id: str
    ) -> None:
        """Register a user's identity across channels."""
        self._cross_channel_manager.register_user_identity(
            user_global_id, channel_name, user_id
        )

    def get_global_user_id(self, channel_name: str, user_id: str) -> str | None:
        """Get the global user ID for a channel-specific user ID."""
        return self._cross_channel_manager.get_global_user_id(channel_name, user_id)

    def create_cross_channel_conversation(
        self,
        user_global_id: str,
        thread_id: str,
        channel_name: str,
        chat_id: str,
        thread_ts: str | None = None,
        consent_granted: bool = False,
    ) -> str:
        """Create a new cross-channel conversation and return its ID."""
        return self._cross_channel_manager.create_cross_channel_conversation(
            user_global_id,
            thread_id,
            channel_name,
            chat_id,
            thread_ts,
            consent_granted,
        )

    def get_cross_channel_state(self, cross_channel_id: str) -> CrossChannelState | None:
        """Get the state for a cross-channel conversation."""
        return self._cross_channel_manager.get_cross_channel_state(cross_channel_id)

    def get_cross_channel_id_by_thread(self, thread_id: str) -> str | None:
        """Get the cross-channel ID associated with a thread."""
        return self._cross_channel_manager.get_cross_channel_id_by_thread(thread_id)

    def log_audit_event(
        self, cross_channel_id: str, event: str, details: dict[str, Any]
    ) -> None:
        """Log an audit event for a cross-channel conversation."""
        self._cross_channel_manager.log_audit_event(cross_channel_id, event, details)