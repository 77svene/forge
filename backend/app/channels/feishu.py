"""Feishu/Lark channel — connects to Feishu via WebSocket (no public IP needed)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from app.channels.base import Channel
from app.channels.message_bus import InboundMessageType, MessageBus, OutboundMessage, ResolvedAttachment

logger = logging.getLogger(__name__)


class FeishuCardType(Enum):
    """Types of Feishu interactive cards."""
    TEXT = "text"
    APPROVAL = "approval"
    PROGRESS = "progress"
    TABLE = "table"
    IMAGE = "image"
    ACTION = "action"


@dataclass
class FeishuCardElement:
    """Base class for Feishu card elements."""
    tag: str
    content: Optional[str] = None
    text: Optional[Dict] = None
    actions: Optional[List[Dict]] = None
    fields: Optional[List[Dict]] = None
    elements: Optional[List[Dict]] = None
    # Accessibility
    alt_text: Optional[str] = None
    aria_label: Optional[str] = None


class FeishuMediaProcessor:
    """Processes agent outputs into Feishu interactive cards with templates and accessibility."""
    
    def __init__(self):
        self.template_registry = {
            FeishuCardType.APPROVAL: self._build_approval_card,
            FeishuCardType.PROGRESS: self._build_progress_card,
            FeishuCardType.TABLE: self._build_table_card,
            FeishuCardType.IMAGE: self._build_image_card,
            FeishuCardType.ACTION: self._build_action_card,
        }
    
    def process(self, content: Union[str, Dict], metadata: Optional[Dict] = None) -> Dict:
        """Convert agent output to Feishu card format.
        
        Args:
            content: Agent output (text or structured data)
            metadata: Additional context (user info, thread info, etc.)
            
        Returns:
            Feishu card JSON structure
        """
        if isinstance(content, str):
            return self._build_text_card(content, metadata)
        
        # Handle structured content
        card_type = content.get("type", FeishuCardType.TEXT.value)
        data = content.get("data", {})
        
        if card_type in self.template_registry:
            return self.template_registry[card_type](data, metadata)
        
        # Fallback to text card
        return self._build_text_card(str(content), metadata)
    
    def _build_text_card(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Build a simple text card with accessibility features."""
        # Truncate long text for Feishu limits (max 2000 chars for markdown)
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "Response"
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                    # Accessibility: provide alt text for screen readers
                    "alt_text": text[:100] + "..." if len(text) > 100 else text
                }
            ]
        }
        
        # Add metadata if available
        if metadata and metadata.get("user_name"):
            card["elements"].append({
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"Requested by {metadata['user_name']}"
                    }
                ]
            })
        
        return card
    
    def _build_approval_card(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Build an approval card with action buttons."""
        title = data.get("title", "Approval Required")
        description = data.get("description", "Please review and approve/reject.")
        options = data.get("options", [
            {"label": "Approve", "value": "approve", "style": "primary"},
            {"label": "Reject", "value": "reject", "style": "danger"}
        ])
        
        # Build action buttons
        actions = []
        for option in options:
            actions.append({
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": option["label"]
                },
                "type": option.get("style", "default"),
                "value": option["value"],
                # Accessibility
                "aria_label": f"{option['label']} button"
            })
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "orange"
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": description,
                    "alt_text": description
                },
                {
                    "tag": "action",
                    "actions": actions
                }
            ]
        }
        
        # Add expiration if specified
        if data.get("expires_in"):
            card["elements"].append({
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"Expires in {data['expires_in']}"
                    }
                ]
            })
        
        return card
    
    def _build_progress_card(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Build a progress bar card."""
        title = data.get("title", "Progress")
        progress = data.get("progress", 0)
        status = data.get("status", "in_progress")
        details = data.get("details", "")
        
        # Determine color based on progress
        if progress >= 100:
            template = "green"
            status_text = "Completed"
        elif progress >= 50:
            template = "blue"
            status_text = "In Progress"
        else:
            template = "yellow"
            status_text = "Starting"
        
        # Build progress visualization using markdown
        progress_bar = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))
        progress_text = f"**{progress}%** {progress_bar}"
        
        elements = [
            {
                "tag": "markdown",
                "content": progress_text,
                "alt_text": f"Progress: {progress}%"
            }
        ]
        
        if details:
            elements.append({
                "tag": "markdown",
                "content": details,
                "alt_text": details
            })
        
        # Add status note
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"Status: {status_text}"
                }
            ]
        })
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": template
            },
            "elements": elements
        }
        
        return card
    
    def _build_table_card(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Build a data table card."""
        title = data.get("title", "Data Table")
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        caption = data.get("caption", "")
        
        # Build markdown table
        table_lines = []
        
        # Header row
        if headers:
            table_lines.append("| " + " | ".join(headers) + " |")
            table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows
        for row in rows[:10]:  # Limit to 10 rows for readability
            if isinstance(row, dict):
                # Convert dict to list based on headers
                row_data = [str(row.get(h, "")) for h in headers]
            else:
                row_data = [str(cell) for cell in row]
            table_lines.append("| " + " | ".join(row_data) + " |")
        
        if len(rows) > 10:
            table_lines.append(f"\n*Showing 10 of {len(rows)} rows*")
        
        table_content = "\n".join(table_lines)
        
        elements = [
            {
                "tag": "markdown",
                "content": table_content,
                "alt_text": f"Table with {len(rows)} rows and {len(headers)} columns"
            }
        ]
        
        if caption:
            elements.append({
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": caption
                    }
                ]
            })
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "purple"
            },
            "elements": elements
        }
        
        return card
    
    def _build_image_card(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Build an image card with accessibility."""
        image_key = data.get("image_key", "")
        alt_text = data.get("alt_text", "Image")
        title = data.get("title", "Image")
        caption = data.get("caption", "")
        
        if not image_key:
            # Fallback to text if no image key
            return self._build_text_card(f"Image: {alt_text}", metadata)
        
        elements = [
            {
                "tag": "img",
                "img_key": image_key,
                "alt": alt_text,  # Accessibility: alt text for screen readers
                "mode": "fit_horizontal"
            }
        ]
        
        if caption:
            elements.append({
                "tag": "markdown",
                "content": caption,
                "alt_text": caption
            })
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "turquoise"
            },
            "elements": elements
        }
        
        return card
    
    def _build_action_card(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Build a card with multiple action buttons."""
        title = data.get("title", "Actions")
        description = data.get("description", "Choose an action:")
        actions = data.get("actions", [])
        
        action_buttons = []
        for action in actions:
            action_buttons.append({
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": action["label"]
                },
                "type": action.get("style", "default"),
                "value": action["value"],
                "aria_label": action.get("aria_label", action["label"])
            })
        
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "indigo"
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": description,
                    "alt_text": description
                },
                {
                    "tag": "action",
                    "actions": action_buttons
                }
            ]
        }
        
        return card


class FeishuChannel(Channel):
    """Feishu/Lark IM channel using the ``lark-oapi`` WebSocket client.

    Configuration keys (in ``config.yaml`` under ``channels.feishu``):
        - ``app_id``: Feishu app ID.
        - ``app_secret``: Feishu app secret.
        - ``verification_token``: (optional) Event verification token.

    The channel uses WebSocket long-connection mode so no public IP is required.

    Message flow:
        1. User sends a message → bot adds "OK" emoji reaction
        2. Bot replies in thread: "Working on it......"
        3. Agent processes the message and returns a result
        4. Bot replies in thread with the result
        5. Bot adds "DONE" emoji reaction to the original message
    """

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="feishu", bus=bus, config=config)
        self._thread: threading.Thread | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._api_client = None
        self._CreateMessageReactionRequest = None
        self._CreateMessageReactionRequestBody = None
        self._Emoji = None
        self._PatchMessageRequest = None
        self._PatchMessageRequestBody = None
        self._background_tasks: set[asyncio.Task] = set()
        self._running_card_ids: dict[str, str] = {}
        self._running_card_tasks: dict[str, asyncio.Task] = {}
        self._CreateFileRequest = None
        self._CreateFileRequestBody = None
        self._CreateImageRequest = None
        self._CreateImageRequestBody = None
        # Initialize media processor for rich content
        self.media_processor = FeishuMediaProcessor()

    async def start(self) -> None:
        if self._running:
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (
                CreateFileRequest,
                CreateFileRequestBody,
                CreateImageRequest,
                CreateImageRequestBody,
                CreateMessageReactionRequest,
                CreateMessageReactionRequestBody,
                CreateMessageRequest,
                CreateMessageRequestBody,
                Emoji,
                PatchMessageRequest,
                PatchMessageRequestBody,
                ReplyMessageRequest,
                ReplyMessageRequestBody,
            )
        except ImportError:
            logger.error("lark-oapi is not installed. Install it with: uv add lark-oapi")
            return

        self._lark = lark
        self._CreateMessageRequest = CreateMessageRequest
        self._CreateMessageRequestBody = CreateMessageRequestBody
        self._ReplyMessageRequest = ReplyMessageRequest
        self._ReplyMessageRequestBody = ReplyMessageRequestBody
        self._CreateMessageReactionRequest = CreateMessageReactionRequest
        self._CreateMessageReactionRequestBody = CreateMessageReactionRequestBody
        self._Emoji = Emoji
        self._PatchMessageRequest = PatchMessageRequest
        self._PatchMessageRequestBody = PatchMessageRequestBody
        self._CreateFileRequest = CreateFileRequest
        self._CreateFileRequestBody = CreateFileRequestBody
        self._CreateImageRequest = CreateImageRequest
        self._CreateImageRequestBody = CreateImageRequestBody

        app_id = self.config.get("app_id", "")
        app_secret = self.config.get("app_secret", "")

        if not app_id or not app_secret:
            logger.error("Feishu channel requires app_id and app_secret")
            return

        self._api_client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        self._main_loop = asyncio.get_event_loop()

        self._running = True
        self.bus.subscribe_outbound(self._on_outbound)

        # Both ws.Client construction and start() must happen in a dedicated
        # thread with its own event loop.  lark-oapi caches the running loop
        # at construction time and later calls loop.run_until_complete(),
        # which conflicts with an already-running uvloop.
        self._thread = threading.Thread(
            target=self._run_ws,
            args=(app_id, app_secret),
            daemon=True,
        )
        self._thread.start()
        logger.info("Feishu channel started")

    def _run_ws(self, app_id: str, app_secret: str) -> None:
        """Construct and run the lark WS client in a thread with a fresh event loop.

        The lark-oapi SDK captures a module-level event loop at import time
        (``lark_oapi.ws.client.loop``).  When uvicorn uses uvloop, that
        captured loop is the *main* thread's uvloop — which is already
        running, so ``loop.run_until_complete()`` inside ``Client.start()``
        raises ``RuntimeError``.

        We work around this by creating a plain asyncio event loop for this
        thread and patching the SDK's module-level reference before calling
        ``start()``.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import lark_oapi as lark
            import lark_oapi.ws.client as _ws_client_mod

            # Replace the SDK's module-level loop so Client.start() uses
            # this thread's (non-running) event loop instead of the main
            # thread's uvloop.
            _ws_client_mod.loop = loop

            event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(self._on_message).build()
            ws_client = lark.ws.Client(
                app_id=app_id,
                app_secret=app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )
            ws_client.start()
        except Exception:
            if self._running:
                logger.exception("Feishu WebSocket error")

    async def stop(self) -> None:
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        for task in list(self._running_card_tasks.values()):
            task.cancel()
        self._running_card_tasks.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Feishu channel stopped")

    async def send(self, msg: OutboundMessage, *, _max_retries: int = 3) -> None:
        if not self._api_client:
            logger.warning("[Feishu] send called but no api_client available")
            return

        logger.info(
            "[Feishu] sending reply: chat_id=%s, thread_ts=%s, text_len=%d",
            msg.chat_id,
            msg.thread_ts,
            len(msg.text),
        )

        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                await self._send_card_message(msg)
                return  # success
            except Exception as exc:
                last_exc = exc
                if attempt < _max_retries - 1:
                    delay = 2**attempt  # 1s, 2s
                    logger.warning(
                        "[Feishu] send failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        _max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        logger.error("[Feishu] send failed after %d attempts: %s", _max_retries, last_exc)
        raise last_exc  # type: ignore[misc]

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        if not self._api_client:
            return False
        
        try:
            # Upload file to Feishu
            file_key = await self._upload_file(attachment)
            if not file_key:
                return False
            
            # Create message with file
            request = self._CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    self._CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("file")
                    .content(json.dumps({"file_key": file_key}))
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.message.create(request)
            if response.success():
                logger.info("[Feishu] file sent successfully: %s", file_key)
                return True
            else:
                logger.error("[Feishu] failed to send file: %s", response.msg)
                return False
                
        except Exception as e:
            logger.error("[Feishu] error sending file: %s", str(e))
            return False

    async def _upload_file(self, attachment: ResolvedAttachment) -> Optional[str]:
        """Upload file to Feishu and return file_key."""
        try:
            # Prepare file data
            file_data = {
                "file_name": attachment.filename,
                "file": attachment.data,
                "file_type": attachment.mime_type or "stream",
                "duration": 0,  # Required for audio/video
            }
            
            # Create upload request
            request = self._CreateFileRequest.builder() \
                .request_body(
                    self._CreateFileRequestBody.builder()
                    .file_name(attachment.filename)
                    .file_type("stream")
                    .file(attachment.data)
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.file.create(request)
            if response.success():
                return response.data.file_key
            else:
                logger.error("[Feishu] file upload failed: %s", response.msg)
                return None
                
        except Exception as e:
            logger.error("[Feishu] file upload error: %s", str(e))
            return None

    async def _send_card_message(self, msg: OutboundMessage) -> None:
        """Send a card message with rich media processing."""
        try:
            # Try to parse as structured content
            content = msg.text
            metadata = {
                "user_name": msg.metadata.get("user_name") if msg.metadata else None,
                "thread_ts": msg.thread_ts,
                "chat_id": msg.chat_id,
            }
            
            # Process content through media processor
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    structured_content = json.loads(content)
                    card_content = self.media_processor.process(structured_content, metadata)
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain text
                    card_content = self.media_processor.process(content, metadata)
            else:
                card_content = self.media_processor.process(content, metadata)
            
            # Convert to JSON string
            card_json = json.dumps(card_content)
            
            # Send the card
            request = self._CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    self._CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("interactive")
                    .content(card_json)
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.message.create(request)
            
            if response.success():
                logger.info("[Feishu] card message sent successfully")
                
                # Update running card if this is an update
                if msg.metadata and msg.metadata.get("card_id"):
                    card_id = msg.metadata["card_id"]
                    self._running_card_ids[card_id] = response.data.message_id
                    
            else:
                logger.error("[Feishu] failed to send card: %s", response.msg)
                raise Exception(f"Feishu API error: {response.msg}")
                
        except Exception as e:
            logger.error("[Feishu] error in _send_card_message: %s", str(e))
            raise

    async def _on_outbound(self, msg: OutboundMessage) -> None:
        """Handle outbound messages from the message bus."""
        if not self._running:
            return
        
        try:
            # Check if this is a file message
            if msg.metadata and msg.metadata.get("attachment"):
                attachment = msg.metadata["attachment"]
                await self.send_file(msg, attachment)
            else:
                await self.send(msg)
                
        except Exception as e:
            logger.error("[Feishu] error processing outbound message: %s", str(e))

    def _on_message(self, event):
        """Handle incoming Feishu messages."""
        try:
            # Extract message details
            message = event.event.message
            chat_id = message.chat_id
            message_id = message.message_id
            content = json.loads(message.content)
            text = content.get("text", "")
            user_id = event.event.sender.sender_id.union_id
            
            # Add OK reaction
            asyncio.run_coroutine_threadsafe(
                self._add_reaction(message_id, "OK"),
                self._main_loop
            )
            
            # Send "Working on it..." reply
            asyncio.run_coroutine_threadsafe(
                self._send_working_reply(chat_id, message_id),
                self._main_loop
            )
            
            # Create inbound message
            inbound_msg = InboundMessageType(
                channel="feishu",
                chat_id=chat_id,
                thread_ts=message_id,
                user_id=user_id,
                text=text,
                metadata={
                    "message_id": message_id,
                    "user_name": event.event.sender.sender_id.name,
                }
            )
            
            # Publish to message bus
            self.bus.publish_inbound(inbound_msg)
            
        except Exception as e:
            logger.error("[Feishu] error handling incoming message: %s", str(e))

    async def _add_reaction(self, message_id: str, emoji_type: str) -> None:
        """Add an emoji reaction to a message."""
        try:
            request = self._CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    self._CreateMessageReactionRequestBody.builder()
                    .reaction_type(
                        self._Emoji.builder()
                        .type(emoji_type)
                        .build()
                    )
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.message_reaction.create(request)
            if not response.success():
                logger.warning("[Feishu] failed to add reaction: %s", response.msg)
                
        except Exception as e:
            logger.error("[Feishu] error adding reaction: %s", str(e))

    async def _send_working_reply(self, chat_id: str, message_id: str) -> None:
        """Send a 'Working on it...' reply in thread."""
        try:
            request = self._ReplyMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    self._ReplyMessageRequestBody.builder()
                    .content(json.dumps({"text": "Working on it..."}))
                    .msg_type("text")
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.message.reply(request)
            if not response.success():
                logger.warning("[Feishu] failed to send working reply: %s", response.msg)
                
        except Exception as e:
            logger.error("[Feishu] error sending working reply: %s", str(e))

    async def update_card(self, card_id: str, new_content: Dict) -> bool:
        """Update an existing interactive card."""
        if not self._api_client or card_id not in self._running_card_ids:
            return False
        
        try:
            message_id = self._running_card_ids[card_id]
            
            # Process new content through media processor
            card_content = self.media_processor.process(new_content)
            card_json = json.dumps(card_content)
            
            # Update the message
            request = self._PatchMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    self._PatchMessageRequestBody.builder()
                    .content(card_json)
                    .build()
                ) \
                .build()
            
            response = self._api_client.im.v1.message.patch(request)
            
            if response.success():
                logger.info("[Feishu] card updated successfully: %s", card_id)
                return True
            else:
                logger.error("[Feishu] failed to update card: %s", response.msg)
                return False
                
        except Exception as e:
            logger.error("[Feishu] error updating card: %s", str(e))
            return False