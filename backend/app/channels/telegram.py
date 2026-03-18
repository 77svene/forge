"""Telegram channel — connects via long-polling (no public IP needed)."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from app.channels.base import Channel
from app.channels.message_bus import InboundMessageType, MessageBus, OutboundMessage, ResolvedAttachment

logger = logging.getLogger(__name__)


class TelegramChannel(Channel):
    """Telegram bot channel using long-polling.

    Configuration keys (in ``config.yaml`` under ``channels.telegram``):
        - ``bot_token``: Telegram Bot API token (from @BotFather).
        - ``allowed_users``: (optional) List of allowed Telegram user IDs. Empty = allow all.
    """

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="telegram", bus=bus, config=config)
        self._application = None
        self._thread: threading.Thread | None = None
        self._tg_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._allowed_users: set[int] = set()
        for uid in config.get("allowed_users", []):
            try:
                self._allowed_users.add(int(uid))
            except (ValueError, TypeError):
                pass
        # chat_id -> last sent message_id for threaded replies
        self._last_bot_message: dict[str, int] = {}

    async def start(self) -> None:
        if self._running:
            return

        try:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
        except ImportError:
            logger.error("python-telegram-bot is not installed. Install it with: uv add python-telegram-bot")
            return

        bot_token = self.config.get("bot_token", "")
        if not bot_token:
            logger.error("Telegram channel requires bot_token")
            return

        self._main_loop = asyncio.get_event_loop()
        self._running = True
        self.bus.subscribe_outbound(self._on_outbound)

        # Build the application
        app = ApplicationBuilder().token(bot_token).build()

        # Command handlers
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("new", self._cmd_generic))
        app.add_handler(CommandHandler("status", self._cmd_generic))
        app.add_handler(CommandHandler("models", self._cmd_generic))
        app.add_handler(CommandHandler("memory", self._cmd_generic))
        app.add_handler(CommandHandler("help", self._cmd_generic))

        # General message handler
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

        self._application = app

        # Run polling in a dedicated thread with its own event loop
        self._thread = threading.Thread(target=self._run_polling, daemon=True)
        self._thread.start()
        logger.info("Telegram channel started")

    async def stop(self) -> None:
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        if self._tg_loop and self._tg_loop.is_running():
            self._tg_loop.call_soon_threadsafe(self._tg_loop.stop)
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        self._application = None
        logger.info("Telegram channel stopped")

    async def send(self, msg: OutboundMessage, *, _max_retries: int = 3) -> None:
        if not self._application:
            return

        try:
            chat_id = int(msg.chat_id)
        except (ValueError, TypeError):
            logger.error("Invalid Telegram chat_id: %s", msg.chat_id)
            return

        kwargs: dict[str, Any] = {"chat_id": chat_id}
        
        # Check for rich media content
        rich_content = self._process_rich_media(msg)
        if rich_content:
            # Use rich media if available
            kwargs.update(rich_content)
        else:
            # Fallback to plain text
            kwargs["text"] = msg.text

        # Reply to the last bot message in this chat for threading
        reply_to = self._last_bot_message.get(msg.chat_id)
        if reply_to:
            kwargs["reply_to_message_id"] = reply_to

        bot = self._application.bot
        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                sent = await bot.send_message(**kwargs)
                self._last_bot_message[msg.chat_id] = sent.message_id
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _max_retries - 1:
                    delay = 2**attempt  # 1s, 2s
                    logger.warning(
                        "[Telegram] send failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        _max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        logger.error("[Telegram] send failed after %d attempts: %s", _max_retries, last_exc)
        raise last_exc  # type: ignore[misc]

    def _process_rich_media(self, msg: OutboundMessage) -> Optional[dict[str, Any]]:
        """Process rich media content from outbound message.
        
        Supports:
        - Inline keyboards (callback buttons)
        - Reply keyboards
        - Formatted text (HTML/Markdown)
        - Accessibility features
        """
        if not hasattr(msg, 'rich_media') or not msg.rich_media:
            return None

        rich_media = msg.rich_media
        result: dict[str, Any] = {}
        
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
            from telegram.constants import ParseMode
        except ImportError:
            logger.warning("Telegram library not available for rich media processing")
            return None

        # Process inline keyboards
        if 'inline_keyboard' in rich_media:
            keyboard_data = rich_media['inline_keyboard']
            keyboard = []
            for row in keyboard_data:
                button_row = []
                for btn in row:
                    if 'callback_data' in btn:
                        button_row.append(InlineKeyboardButton(
                            text=btn['text'],
                            callback_data=btn['callback_data']
                        ))
                    elif 'url' in btn:
                        button_row.append(InlineKeyboardButton(
                            text=btn['text'],
                            url=btn['url']
                        ))
                    elif 'switch_inline_query' in btn:
                        button_row.append(InlineKeyboardButton(
                            text=btn['text'],
                            switch_inline_query=btn['switch_inline_query']
                        ))
                keyboard.append(button_row)
            result['reply_markup'] = InlineKeyboardMarkup(keyboard)
        
        # Process reply keyboards
        elif 'reply_keyboard' in rich_media:
            keyboard_data = rich_media['reply_keyboard']
            keyboard = []
            for row in keyboard_data:
                button_row = []
                for btn in row:
                    button_row.append(KeyboardButton(text=btn['text']))
                keyboard.append(button_row)
            result['reply_markup'] = ReplyKeyboardMarkup(
                keyboard,
                one_time_keyboard=rich_media.get('one_time_keyboard', True),
                resize_keyboard=rich_media.get('resize_keyboard', True)
            )
        
        # Process text formatting
        if 'parse_mode' in rich_media:
            result['parse_mode'] = rich_media['parse_mode']
            result['text'] = rich_media.get('text', msg.text)
        else:
            result['text'] = msg.text
        
        # Process accessibility features
        if 'accessibility' in rich_media:
            accessibility = rich_media['accessibility']
            if 'disable_notification' in accessibility:
                result['disable_notification'] = accessibility['disable_notification']
            if 'protect_content' in accessibility:
                result['protect_content'] = accessibility['protect_content']
        
        return result

    def create_approval_keyboard(self, approval_id: str, approve_text: str = "✅ Approve", 
                                 reject_text: str = "❌ Reject") -> dict[str, Any]:
        """Create an inline keyboard for approval workflows."""
        return {
            'inline_keyboard': [
                [
                    {'text': approve_text, 'callback_data': f'approve:{approval_id}'},
                    {'text': reject_text, 'callback_data': f'reject:{approval_id}'}
                ]
            ]
        }

    def create_progress_keyboard(self, progress_id: str, current: int, total: int) -> dict[str, Any]:
        """Create a progress bar keyboard with cancel option."""
        progress_bar = "▓" * current + "░" * (total - current)
        percentage = int((current / total) * 100) if total > 0 else 0
        
        return {
            'inline_keyboard': [
                [{'text': f'{progress_bar} {percentage}%', 'callback_data': 'noop'}],
                [{'text': '⏹ Cancel', 'callback_data': f'cancel:{progress_id}'}]
            ]
        }

    def create_data_table_keyboard(self, data: list[dict], columns: list[str], 
                                   page: int = 0, page_size: int = 5) -> dict[str, Any]:
        """Create a paginated data table keyboard."""
        start_idx = page * page_size
        end_idx = start_idx + page_size
        page_data = data[start_idx:end_idx]
        
        keyboard = []
        # Header row
        header = [{'text': col, 'callback_data': 'noop'} for col in columns]
        keyboard.append(header)
        
        # Data rows
        for row in page_data:
            button_row = []
            for col in columns:
                text = str(row.get(col, ''))[:20]  # Truncate long text
                button_row.append({'text': text, 'callback_data': 'noop'})
            keyboard.append(button_row)
        
        # Pagination controls
        total_pages = (len(data) + page_size - 1) // page_size
        pagination = []
        if page > 0:
            pagination.append({'text': '⬅️ Previous', 'callback_data': f'page:{page-1}'})
        if page < total_pages - 1:
            pagination.append({'text': 'Next ➡️', 'callback_data': f'page:{page+1}'})
        if pagination:
            keyboard.append(pagination)
        
        return {'inline_keyboard': keyboard}

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        if not self._application:
            return False

        try:
            chat_id = int(msg.chat_id)
        except (ValueError, TypeError):
            logger.error("[Telegram] Invalid chat_id: %s", msg.chat_id)
            return False

        # Telegram limits: 10MB for photos, 50MB for documents
        if attachment.size > 50 * 1024 * 1024:
            logger.warning("[Telegram] file too large (%d bytes), skipping: %s", attachment.size, attachment.filename)
            return False

        bot = self._application.bot
        reply_to = self._last_bot_message.get(msg.chat_id)

        # Process rich media for file captions
        caption = msg.text
        caption_entities = None
        reply_markup = None
        
        if hasattr(msg, 'rich_media') and msg.rich_media:
            rich_media = msg.rich_media
            if 'caption' in rich_media:
                caption = rich_media['caption']
            if 'caption_entities' in rich_media:
                caption_entities = rich_media['caption_entities']
            if 'inline_keyboard' in rich_media:
                rich_content = self._process_rich_media(msg)
                if rich_content and 'reply_markup' in rich_content:
                    reply_markup = rich_content['reply_markup']

        try:
            if attachment.is_image and attachment.size <= 10 * 1024 * 1024:
                with open(attachment.actual_path, "rb") as f:
                    kwargs: dict[str, Any] = {
                        "chat_id": chat_id,
                        "photo": f,
                        "caption": caption
                    }
                    if reply_to:
                        kwargs["reply_to_message_id"] = reply_to
                    if caption_entities:
                        kwargs["caption_entities"] = caption_entities
                    if reply_markup:
                        kwargs["reply_markup"] = reply_markup
                    sent = await bot.send_photo(**kwargs)
            else:
                from telegram import InputFile

                with open(attachment.actual_path, "rb") as f:
                    input_file = InputFile(f, filename=attachment.filename)
                    kwargs = {
                        "chat_id": chat_id,
                        "document": input_file,
                        "caption": caption
                    }
                    if reply_to:
                        kwargs["reply_to_message_id"] = reply_to
                    if caption_entities:
                        kwargs["caption_entities"] = caption_entities
                    if reply_markup:
                        kwargs["reply_markup"] = reply_markup
                    sent = await bot.send_document(**kwargs)

            self._last_bot_message[msg.chat_id] = sent.message_id
            logger.info("[Telegram] file sent: %s to chat=%s", attachment.filename, msg.chat_id)
            return True
        except Exception:
            logger.exception("[Telegram] failed to send file: %s", attachment.filename)
            return False

    # -- helpers -----------------------------------------------------------

    async def _send_running_reply(self, chat_id: str, reply_to_message_id: int) -> None:
        """Send a 'Working on it...' reply to the user's message."""
        if not self._application:
            return
        try:
            bot = self._application.bot
            await bot.send_message(
                chat_id=int(chat_id),
                text="Working on it...",
                reply_to_message_id=reply_to_message_id,
            )
            logger.info("[Telegram] 'Working on it...' reply sent in chat=%s", chat_id)
        except Exception:
            logger.exception("[Telegram] failed to send running reply in chat=%s", chat_id)

    # -- internal ----------------------------------------------------------

    def _run_polling(self) -> None:
        """Run telegram polling in a dedicated thread."""
        self._tg_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._tg_loop)
        try:
            # Cannot use run_polling() because it calls add_signal_handler(),
            # which only works in the main thread.  Instead, manually
            # initialize the application and start the updater.
            self._tg_loop.run_until_complete(self._application.initialize())
            self._tg_loop.run_until_complete(self._application.start())
            updater = self._application.updater
            if updater:
                self._tg_loop.run_until_complete(updater.start_polling())
            self._tg_loop.run_forever()
        except Exception:
            logger.exception("[Telegram] polling error")
        finally:
            if self._application:
                self._tg_loop.run_until_complete(self._application.stop())
            self._tg_loop.close()