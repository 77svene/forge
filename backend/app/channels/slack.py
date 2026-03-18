"""Slack channel — connects via Socket Mode (no public IP needed)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from markdown_to_mrkdwn import SlackMarkdownConverter

from app.channels.base import Channel
from app.channels.message_bus import InboundMessageType, MessageBus, OutboundMessage, ResolvedAttachment

logger = logging.getLogger(__name__)

_slack_md_converter = SlackMarkdownConverter()


class SlackMediaProcessor:
    """Processes agent outputs into Slack Block Kit rich content."""
    
    def __init__(self):
        self.template_registry = {
            "approval": self._build_approval_template,
            "progress": self._build_progress_template,
            "data_table": self._build_data_table_template,
            "interactive_card": self._build_interactive_card_template,
        }
    
    def process(self, content: Union[str, Dict, List], context: Optional[Dict] = None) -> Dict:
        """
        Convert agent output to Slack Block Kit format.
        
        Args:
            content: Can be:
                - Plain text string
                - Dict with template specification
                - List of pre-formatted blocks
            context: Additional context for template rendering
            
        Returns:
            Dict with 'blocks' and 'text' (fallback) keys
        """
        context = context or {}
        
        # If content is already a list of blocks, use directly
        if isinstance(content, list):
            return {
                "blocks": content,
                "text": self._extract_text_from_blocks(content)
            }
        
        # If content is a dict with template specification
        if isinstance(content, dict):
            template_type = content.get("template")
            if template_type and template_type in self.template_registry:
                try:
                    blocks = self.template_registry[template_type](content, context)
                    return {
                        "blocks": blocks,
                        "text": content.get("text", self._extract_text_from_blocks(blocks))
                    }
                except Exception as e:
                    logger.warning(f"Failed to render template {template_type}: {e}")
                    # Fall back to text rendering
        
        # Default: treat as plain text
        text = str(content) if not isinstance(content, str) else content
        return {
            "blocks": self._text_to_blocks(text),
            "text": text
        }
    
    def _text_to_blocks(self, text: str) -> List[Dict]:
        """Convert plain text to basic Slack blocks."""
        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": _slack_md_converter.convert(text)
                }
            }
        ]
    
    def _extract_text_from_blocks(self, blocks: List[Dict]) -> str:
        """Extract plain text from blocks for accessibility/fallback."""
        text_parts = []
        for block in blocks:
            if block.get("type") == "section" and "text" in block:
                text_obj = block["text"]
                if isinstance(text_obj, dict) and "text" in text_obj:
                    text_parts.append(text_obj["text"])
            elif block.get("type") == "header" and "text" in block:
                text_obj = block["text"]
                if isinstance(text_obj, dict) and "text" in text_obj:
                    text_parts.append(text_obj["text"])
        return " ".join(text_parts) if text_parts else "Interactive content"
    
    def _build_approval_template(self, data: Dict, context: Dict) -> List[Dict]:
        """Build approval buttons template."""
        title = data.get("title", "Approval Required")
        description = data.get("description", "")
        buttons = data.get("buttons", ["Approve", "Reject"])
        callback_id = data.get("callback_id", f"approval_{context.get('message_id', 'default')}")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            }
        ]
        
        if description:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": description
                }
            })
        
        # Build action buttons
        actions = []
        for i, button_text in enumerate(buttons):
            style = "primary" if i == 0 else "danger" if i == len(buttons) - 1 else None
            action = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": button_text,
                    "emoji": True
                },
                "value": f"approval_{i}",
                "action_id": f"{callback_id}_btn_{i}"
            }
            if style:
                action["style"] = style
            actions.append(action)
        
        blocks.append({
            "type": "actions",
            "elements": actions
        })
        
        # Add accessibility context
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"♿ *Accessibility:* This is an approval request with {len(buttons)} options. Use screen reader to navigate buttons."
                }
            ]
        })
        
        return blocks
    
    def _build_progress_template(self, data: Dict, context: Dict) -> List[Dict]:
        """Build progress bar template."""
        title = data.get("title", "Progress Update")
        progress = data.get("progress", 0.0)
        status = data.get("status", "In Progress")
        details = data.get("details", "")
        
        # Ensure progress is between 0 and 1
        progress = max(0.0, min(1.0, float(progress)))
        percentage = int(progress * 100)
        
        # Create visual progress bar using Unicode characters
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Status:* {status}\n*Progress:* {bar} {percentage}%"
                }
            }
        ]
        
        if details:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": details
                }
            })
        
        # Add divider and accessibility info
        blocks.extend([
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"♿ *Accessibility:* Progress: {percentage}% complete. {status}."
                    }
                ]
            }
        ])
        
        return blocks
    
    def _build_data_table_template(self, data: Dict, context: Dict) -> List[Dict]:
        """Build data table template."""
        title = data.get("title", "Data Table")
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        max_rows = data.get("max_rows", 10)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            }
        ]
        
        if not headers or not rows:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_No data available_"
                }
            })
            return blocks
        
        # Build table as markdown (Slack doesn't support native tables)
        table_lines = []
        
        # Header row
        header_row = " | ".join(str(h) for h in headers)
        table_lines.append(f"*{header_row}*")
        
        # Separator
        separator = " | ".join(["---"] * len(headers))
        table_lines.append(separator)
        
        # Data rows (limit to max_rows)
        for i, row in enumerate(rows[:max_rows]):
            if len(row) != len(headers):
                continue  # Skip malformed rows
            row_text = " | ".join(str(cell) for cell in row)
            table_lines.append(row_text)
        
        if len(rows) > max_rows:
            table_lines.append(f"_...and {len(rows) - max_rows} more rows_")
        
        table_text = "\n".join(table_lines)
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```\n{table_text}\n```"
            }
        })
        
        # Add summary and accessibility
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"♿ *Accessibility:* Table with {len(headers)} columns and {len(rows)} rows. Showing {min(len(rows), max_rows)} rows."
                }
            ]
        })
        
        return blocks
    
    def _build_interactive_card_template(self, data: Dict, context: Dict) -> List[Dict]:
        """Build interactive card template with multiple elements."""
        title = data.get("title", "Interactive Card")
        sections = data.get("sections", [])
        actions = data.get("actions", [])
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            }
        ]
        
        # Add sections
        for section in sections:
            section_type = section.get("type", "text")
            
            if section_type == "text":
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": section.get("content", "")
                    }
                })
            elif section_type == "fields":
                fields = section.get("fields", [])
                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*{field.get('title', '')}:*\n{field.get('value', '')}"
                        }
                        for field in fields[:10]  # Slack limit
                    ]
                })
            elif section_type == "image":
                blocks.append({
                    "type": "image",
                    "image_url": section.get("url", ""),
                    "alt_text": section.get("alt_text", "Image")
                })
        
        # Add actions if present
        if actions:
            action_elements = []
            for action in actions[:5]:  # Slack limit of 5 actions
                if action.get("type") == "button":
                    action_elements.append({
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": action.get("text", "Button"),
                            "emoji": True
                        },
                        "value": action.get("value", ""),
                        "action_id": action.get("action_id", f"action_{len(action_elements)}")
                    })
                elif action.get("type") == "select":
                    action_elements.append({
                        "type": "static_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": action.get("placeholder", "Select an option"),
                            "emoji": True
                        },
                        "options": [
                            {
                                "text": {
                                    "type": "plain_text",
                                    "text": opt.get("text", ""),
                                    "emoji": True
                                },
                                "value": opt.get("value", "")
                            }
                            for opt in action.get("options", [])[:100]  # Slack limit
                        ],
                        "action_id": action.get("action_id", f"select_{len(action_elements)}")
                    })
            
            if action_elements:
                blocks.append({
                    "type": "actions",
                    "elements": action_elements
                })
        
        # Add accessibility context
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"♿ *Accessibility:* Interactive card with {len(sections)} sections and {len(actions)} actions."
                }
            ]
        })
        
        return blocks


class SlackChannel(Channel):
    """Slack IM channel using Socket Mode (WebSocket, no public IP).

    Configuration keys (in ``config.yaml`` under ``channels.slack``):
        - ``bot_token``: Slack Bot User OAuth Token (xoxb-...).
        - ``app_token``: Slack App-Level Token (xapp-...) for Socket Mode.
        - ``allowed_users``: (optional) List of allowed Slack user IDs. Empty = allow all.
        - ``rich_media``: (optional) Enable rich media processing (default: True).
    """

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="slack", bus=bus, config=config)
        self._socket_client = None
        self._web_client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._allowed_users: set[str] = set(config.get("allowed_users", []))
        self._rich_media_enabled = config.get("rich_media", True)
        self._media_processor = SlackMediaProcessor() if self._rich_media_enabled else None

    async def start(self) -> None:
        if self._running:
            return

        try:
            from slack_sdk import WebClient
            from slack_sdk.socket_mode import SocketModeClient
            from slack_sdk.socket_mode.response import SocketModeResponse
        except ImportError:
            logger.error("slack-sdk is not installed. Install it with: uv add slack-sdk")
            return

        self._SocketModeResponse = SocketModeResponse

        bot_token = self.config.get("bot_token", "")
        app_token = self.config.get("app_token", "")

        if not bot_token or not app_token:
            logger.error("Slack channel requires bot_token and app_token")
            return

        self._web_client = WebClient(token=bot_token)
        self._socket_client = SocketModeClient(
            app_token=app_token,
            web_client=self._web_client,
        )
        self._loop = asyncio.get_event_loop()

        self._socket_client.socket_mode_request_listeners.append(self._on_socket_event)

        self._running = True
        self.bus.subscribe_outbound(self._on_outbound)

        # Start socket mode in background thread
        asyncio.get_event_loop().run_in_executor(None, self._socket_client.connect)
        logger.info("Slack channel started")

    async def stop(self) -> None:
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        if self._socket_client:
            self._socket_client.close()
            self._socket_client = None
        logger.info("Slack channel stopped")

    async def send(self, msg: OutboundMessage, *, _max_retries: int = 3) -> None:
        if not self._web_client:
            return

        # Process content for rich media if enabled
        content = msg.text
        blocks = None
        text = _slack_md_converter.convert(msg.text)
        
        if self._rich_media_enabled and self._media_processor:
            try:
                # Try to parse as JSON for rich content
                if isinstance(msg.text, str) and msg.text.strip().startswith('{'):
                    try:
                        parsed_content = json.loads(msg.text)
                        processed = self._media_processor.process(parsed_content, {
                            "chat_id": msg.chat_id,
                            "thread_ts": msg.thread_ts,
                            "message_id": getattr(msg, 'id', None)
                        })
                        blocks = processed.get("blocks")
                        text = processed.get("text", text)
                    except json.JSONDecodeError:
                        # Not JSON, use as plain text
                        pass
            except Exception as e:
                logger.warning(f"Rich media processing failed, falling back to text: {e}")

        kwargs: dict[str, Any] = {
            "channel": msg.chat_id,
            "text": text,
        }
        
        if blocks:
            kwargs["blocks"] = blocks
        
        if msg.thread_ts:
            kwargs["thread_ts"] = msg.thread_ts

        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                await asyncio.to_thread(self._web_client.chat_postMessage, **kwargs)
                # Add a completion reaction to the thread root
                if msg.thread_ts:
                    await asyncio.to_thread(
                        self._add_reaction,
                        msg.chat_id,
                        msg.thread_ts,
                        "white_check_mark",
                    )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _max_retries - 1:
                    delay = 2**attempt  # 1s, 2s
                    logger.warning(
                        "[Slack] send failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        _max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        logger.error("[Slack] send failed after %d attempts: %s", _max_retries, last_exc)
        # Add failure reaction on error
        if msg.thread_ts:
            try:
                await asyncio.to_thread(
                    self._add_reaction,
                    msg.chat_id,
                    msg.thread_ts,
                    "x",
                )
            except Exception:
                pass
        raise last_exc  # type: ignore[misc]

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        if not self._web_client:
            return False

        try:
            kwargs: dict[str, Any] = {
                "channel": msg.chat_id,
                "file": str(attachment.actual_path),
                "filename": attachment.filename,
                "title": attachment.filename,
            }
            if msg.thread_ts:
                kwargs["thread_ts"] = msg.thread_ts

            await asyncio.to_thread(self._web_client.files_upload_v2, **kwargs)
            logger.info("[Slack] file uploaded: %s to channel=%s", attachment.filename, msg.chat_id)
            return True
        except Exception:
            logger.exception("[Slack] failed to upload file: %s", attachment.filename)
            return False

    # -- internal ----------------------------------------------------------

    def _add_reaction(self, channel_id: str, timestamp: str, emoji: str) -> None:
        """Add an emoji reaction to a message (best-effort, non-blocking)."""
        if not self._web_client:
            return
        try:
            self._web_client.reactions_add(
                channel=channel_id,
                timestamp=timestamp,
                name=emoji,
            )
        except Exception as exc:
            if "already_reacted" not in str(exc):
                logger.warning("[Slack] failed to add reaction %s: %s", emoji, exc)

    def _send_running_reply(self, channel_id: str, thread_ts: str) -> None:
        """Send a 'Working on it......' reply in the thread (called from SDK thread)."""
        if not self._web_client:
            return
        try:
            self._web_client.chat_postMessage(
                channel=channel_id,
                text=":hourglass_flowing_sand: Working on it...",
                thread_ts=thread_ts,
            )
            logger.info("[Slack] 'Working on it...' reply sent in channel=%s, thread_ts=%s", channel_id, thread_ts)
        except Exception:
            logger.exception("[Slack] failed to send running reply in channel=%s", channel_id)

    def _on_socket_event(self, client, req) -> None:
        """Called by slack-sdk for each Socket Mode event."""
        try:
            # Acknowledge the event
            response = self._SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)

            event_type = req.type
            if event_type != "events_api":
                return

            event = req.payload.get("event", {})
            etype = event.get("type", "")

            # Handle message events (DM or @mention)
            if etype in ("message", "app_mention"):
                self._handle_message_event(event)
            
            # Handle interactive actions (button clicks, menu selections)
            elif etype == "interactive" and self._rich_media_enabled:
                self._handle_interactive_event(req.payload)

        except Exception:
            logger.exception("Error processing Slack event")

    def _handle_message_event(self, event: dict) -> None:
        # Ignore bot messages
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user", "")

        # Check allowed users
        if self._allowed_users and user_id not in self._allowed_users:
            logger.debug("Ignoring message from non-allowed user: %s", user_id)
            return

        text = event.get("text", "").strip()
        if not text:
            return

        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        # Send to message bus for processing
        asyncio.run_coroutine_threadsafe(
            self.bus.publish(InboundMessageType.MESSAGE, {
                "channel": "slack",
                "channel_id": channel_id,
                "user_id": user_id,
                "text": text,
                "thread_ts": thread_ts,
                "raw_event": event,
            }),
            self._loop
        )

    def _handle_interactive_event(self, payload: dict) -> None:
        """Handle interactive events like button clicks."""
        try:
            user = payload.get("user", {})
            user_id = user.get("id", "")
            
            # Check allowed users
            if self._allowed_users and user_id not in self._allowed_users:
                logger.debug("Ignoring interactive event from non-allowed user: %s", user_id)
                return
            
            actions = payload.get("actions", [])
            if not actions:
                return
            
            action = actions[0]
            action_id = action.get("action_id", "")
            action_value = action.get("value", "")
            
            # Extract message context
            message = payload.get("message", {})
            channel_id = payload.get("channel", {}).get("id", "")
            thread_ts = message.get("thread_ts") or message.get("ts")
            
            # Send interactive action to message bus
            asyncio.run_coroutine_threadsafe(
                self.bus.publish(InboundMessageType.INTERACTIVE, {
                    "channel": "slack",
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "action_id": action_id,
                    "action_value": action_value,
                    "thread_ts": thread_ts,
                    "raw_payload": payload,
                }),
                self._loop
            )
            
        except Exception:
            logger.exception("Error handling interactive event")