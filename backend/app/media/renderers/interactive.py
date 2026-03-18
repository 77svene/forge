"""
Interactive Media Renderer for Channel-Specific Rich Content Processing
Converts agent outputs to platform-native interactive content with template system
and accessibility features.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Supported channel types for interactive content."""
    FEISHU = "feishu"
    SLACK = "slack"
    TELEGRAM = "telegram"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Types of interactive content."""
    CARD = "card"
    BUTTONS = "buttons"
    TABLE = "table"
    PROGRESS = "progress"
    FORM = "form"
    LIST = "list"
    MEDIA = "media"
    CUSTOM = "custom"


@dataclass
class AccessibilityOptions:
    """Accessibility configuration for interactive content."""
    alt_text: Optional[str] = None
    aria_label: Optional[str] = None
    aria_describedby: Optional[str] = None
    screen_reader_text: Optional[str] = None
    high_contrast: bool = False
    reduced_motion: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alt_text": self.alt_text,
            "aria_label": self.aria_label,
            "aria_describedby": self.aria_describedby,
            "screen_reader_text": self.screen_reader_text,
            "high_contrast": self.high_contrast,
            "reduced_motion": self.reduced_motion
        }


@dataclass
class InteractiveElement:
    """Base class for interactive elements."""
    element_id: str
    element_type: ContentType
    content: Dict[str, Any]
    accessibility: AccessibilityOptions = field(default_factory=AccessibilityOptions)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate element structure."""
        if not self.element_id or not isinstance(self.element_id, str):
            return False
        if not isinstance(self.element_type, ContentType):
            return False
        return True


@dataclass
class InteractiveContent:
    """Container for interactive content to be rendered."""
    channel_type: ChannelType
    elements: List[InteractiveElement]
    fallback_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "channel_type": self.channel_type.value,
            "elements": [
                {
                    "element_id": elem.element_id,
                    "element_type": elem.element_type.value,
                    "content": elem.content,
                    "accessibility": elem.accessibility.to_dict(),
                    "metadata": elem.metadata
                }
                for elem in self.elements
            ],
            "fallback_text": self.fallback_text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }, ensure_ascii=False)


class TemplateRegistry:
    """Registry for interactive content templates."""
    
    def __init__(self):
        self._templates: Dict[str, Callable] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default templates for common UI patterns."""
        self.register("approval_buttons", self._create_approval_buttons)
        self.register("progress_bar", self._create_progress_bar)
        self.register("data_table", self._create_data_table)
        self.register("selection_menu", self._create_selection_menu)
        self.register("confirmation_card", self._create_confirmation_card)
    
    def register(self, name: str, template_func: Callable):
        """Register a template function."""
        self._templates[name] = template_func
    
    def get_template(self, name: str) -> Optional[Callable]:
        """Get template function by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    @staticmethod
    def _create_approval_buttons(
        title: str,
        approve_text: str = "Approve",
        reject_text: str = "Reject",
        element_id: str = "approval_buttons"
    ) -> InteractiveElement:
        """Create approval buttons template."""
        return InteractiveElement(
            element_id=element_id,
            element_type=ContentType.BUTTONS,
            content={
                "title": title,
                "buttons": [
                    {
                        "id": f"{element_id}_approve",
                        "text": approve_text,
                        "style": "primary",
                        "action": {"type": "approve", "value": True}
                    },
                    {
                        "id": f"{element_id}_reject",
                        "text": reject_text,
                        "style": "danger",
                        "action": {"type": "reject", "value": False}
                    }
                ]
            },
            accessibility=AccessibilityOptions(
                aria_label=title,
                screen_reader_text=f"{title}. Choose {approve_text} or {reject_text}."
            )
        )
    
    @staticmethod
    def _create_progress_bar(
        title: str,
        progress: float,
        show_percentage: bool = True,
        element_id: str = "progress_bar"
    ) -> InteractiveElement:
        """Create progress bar template."""
        progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        return InteractiveElement(
            element_id=element_id,
            element_type=ContentType.PROGRESS,
            content={
                "title": title,
                "progress": progress,
                "show_percentage": show_percentage,
                "percentage": f"{progress * 100:.1f}%"
            },
            accessibility=AccessibilityOptions(
                aria_label=f"{title}: {progress * 100:.1f}% complete",
                screen_reader_text=f"{title}. Progress: {progress * 100:.1f}% complete."
            )
        )
    
    @staticmethod
    def _create_data_table(
        title: str,
        headers: List[str],
        rows: List[List[Any]],
        sortable: bool = False,
        element_id: str = "data_table"
    ) -> InteractiveElement:
        """Create data table template."""
        return InteractiveElement(
            element_id=element_id,
            element_type=ContentType.TABLE,
            content={
                "title": title,
                "headers": headers,
                "rows": rows,
                "sortable": sortable,
                "row_count": len(rows)
            },
            accessibility=AccessibilityOptions(
                aria_label=f"{title} with {len(rows)} rows and {len(headers)} columns",
                screen_reader_text=f"{title}. Table with {len(rows)} rows and {len(headers)} columns."
            )
        )
    
    @staticmethod
    def _create_selection_menu(
        title: str,
        options: List[Dict[str, Any]],
        multi_select: bool = False,
        element_id: str = "selection_menu"
    ) -> InteractiveElement:
        """Create selection menu template."""
        return InteractiveElement(
            element_id=element_id,
            element_type=ContentType.BUTTONS,
            content={
                "title": title,
                "options": options,
                "multi_select": multi_select,
                "selection_type": "multi" if multi_select else "single"
            },
            accessibility=AccessibilityOptions(
                aria_label=title,
                screen_reader_text=f"{title}. {'Select multiple options.' if multi_select else 'Select one option.'}"
            )
        )
    
    @staticmethod
    def _create_confirmation_card(
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
        element_id: str = "confirmation_card"
    ) -> InteractiveElement:
        """Create confirmation card template."""
        return InteractiveElement(
            element_id=element_id,
            element_type=ContentType.CARD,
            content={
                "title": title,
                "message": message,
                "buttons": [
                    {
                        "id": f"{element_id}_confirm",
                        "text": confirm_text,
                        "style": "primary",
                        "action": {"type": "confirm", "value": True}
                    },
                    {
                        "id": f"{element_id}_cancel",
                        "text": cancel_text,
                        "style": "secondary",
                        "action": {"type": "cancel", "value": False}
                    }
                ]
            },
            accessibility=AccessibilityOptions(
                aria_label=title,
                screen_reader_text=f"{title}. {message}. Choose {confirm_text} or {cancel_text}."
            )
        )


class BaseInteractiveRenderer(ABC):
    """Abstract base class for interactive content renderers."""
    
    def __init__(self, channel_type: ChannelType):
        self.channel_type = channel_type
        self.template_registry = TemplateRegistry()
    
    @abstractmethod
    def render(self, content: InteractiveContent) -> Dict[str, Any]:
        """Render interactive content to channel-specific format."""
        pass
    
    @abstractmethod
    def validate_content(self, content: InteractiveContent) -> bool:
        """Validate content is compatible with this renderer."""
        pass
    
    def apply_template(
        self,
        template_name: str,
        **kwargs
    ) -> Optional[InteractiveElement]:
        """Apply a template to create an interactive element."""
        template_func = self.template_registry.get_template(template_name)
        if template_func:
            try:
                return template_func(**kwargs)
            except Exception as e:
                logger.error(f"Error applying template '{template_name}': {e}")
                return None
        logger.warning(f"Template '{template_name}' not found")
        return None
    
    def _ensure_accessibility(self, content: InteractiveContent) -> InteractiveContent:
        """Ensure all elements have accessibility features."""
        for element in content.elements:
            if not element.accessibility.aria_label and element.content.get("title"):
                element.accessibility.aria_label = element.content["title"]
            if not element.accessibility.screen_reader_text:
                element.accessibility.screen_reader_text = self._generate_screen_reader_text(element)
        return content
    
    def _generate_screen_reader_text(self, element: InteractiveElement) -> str:
        """Generate screen reader text for an element."""
        base_text = element.content.get("title", "")
        if element.element_type == ContentType.BUTTONS:
            buttons = element.content.get("buttons", [])
            if buttons:
                button_texts = [btn.get("text", "") for btn in buttons]
                base_text += f". Options: {', '.join(button_texts)}."
        elif element.element_type == ContentType.TABLE:
            row_count = element.content.get("row_count", 0)
            headers = element.content.get("headers", [])
            base_text += f". Table with {row_count} rows and {len(headers)} columns."
        return base_text


class FeishuInteractiveRenderer(BaseInteractiveRenderer):
    """Renderer for Feishu interactive cards."""
    
    def __init__(self):
        super().__init__(ChannelType.FEISHU)
    
    def render(self, content: InteractiveContent) -> Dict[str, Any]:
        """Render to Feishu interactive card format."""
        content = self._ensure_accessibility(content)
        
        card_elements = []
        for element in content.elements:
            if element.element_type == ContentType.CARD:
                card_elements.extend(self._render_card_element(element))
            elif element.element_type == ContentType.BUTTONS:
                card_elements.extend(self._render_button_element(element))
            elif element.element_type == ContentType.TABLE:
                card_elements.extend(self._render_table_element(element))
            elif element.element_type == ContentType.PROGRESS:
                card_elements.extend(self._render_progress_element(element))
        
        feishu_card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": content.metadata.get("title", "Interactive Content")
                    },
                    "template": content.metadata.get("template", "blue")
                },
                "elements": card_elements
            }
        }
        
        # Add fallback text
        if content.fallback_text:
            feishu_card["fallback_text"] = content.fallback_text
        
        return feishu_card
    
    def _render_card_element(self, element: InteractiveElement) -> List[Dict]:
        """Render card element to Feishu format."""
        elements = []
        if "title" in element.content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": element.content["title"],
                    "text_size": "normal"
                }
            })
        if "message" in element.content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": element.content["message"]
                }
            })
        if "buttons" in element.content:
            elements.append(self._create_button_group(element.content["buttons"]))
        return elements
    
    def _render_button_element(self, element: InteractiveElement) -> List[Dict]:
        """Render button element to Feishu format."""
        elements = []
        if "title" in element.content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": element.content["title"]
                }
            })
        if "buttons" in element.content:
            elements.append(self._create_button_group(element.content["buttons"]))
        return elements
    
    def _render_table_element(self, element: InteractiveElement) -> List[Dict]:
        """Render table element to Feishu format."""
        elements = []
        if "title" in element.content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": element.content["title"],
                    "text_size": "normal"
                }
            })
        
        headers = element.content.get("headers", [])
        rows = element.content.get("rows", [])
        
        if headers and rows:
            # Create markdown table
            table_md = "| " + " | ".join(headers) + " |\n"
            table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": table_md
                }
            })
        
        return elements
    
    def _render_progress_element(self, element: InteractiveElement) -> List[Dict]:
        """Render progress element to Feishu format."""
        elements = []
        if "title" in element.content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": element.content["title"]
                }
            })
        
        progress = element.content.get("progress", 0)
        percentage = element.content.get("percentage", f"{progress * 100:.1f}%")
        
        # Create progress bar using markdown
        progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"`{progress_bar}` {percentage}"
            }
        })
        
        return elements
    
    def _create_button_group(self, buttons: List[Dict]) -> Dict:
        """Create Feishu button group."""
        actions = []
        for btn in buttons:
            actions.append({
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": btn.get("text", "")
                },
                "type": btn.get("style", "default"),
                "value": btn.get("action", {})
            })
        
        return {
            "tag": "action",
            "actions": actions
        }
    
    def validate_content(self, content: InteractiveContent) -> bool:
        """Validate content for Feishu."""
        if content.channel_type != ChannelType.FEISHU:
            return False
        for element in content.elements:
            if not element.validate():
                return False
        return True


class SlackInteractiveRenderer(BaseInteractiveRenderer):
    """Renderer for Slack Block Kit."""
    
    def __init__(self):
        super().__init__(ChannelType.SLACK)
    
    def render(self, content: InteractiveContent) -> Dict[str, Any]:
        """Render to Slack Block Kit format."""
        content = self._ensure_accessibility(content)
        
        blocks = []
        for element in content.elements:
            if element.element_type == ContentType.CARD:
                blocks.extend(self._render_card_element(element))
            elif element.element_type == ContentType.BUTTONS:
                blocks.extend(self._render_button_element(element))
            elif element.element_type == ContentType.TABLE:
                blocks.extend(self._render_table_element(element))
            elif element.element_type == ContentType.PROGRESS:
                blocks.extend(self._render_progress_element(element))
        
        slack_message = {
            "blocks": blocks,
            "text": content.fallback_text or "Interactive content"
        }
        
        return slack_message
    
    def _render_card_element(self, element: InteractiveElement) -> List[Dict]:
        """Render card element to Slack format."""
        blocks = []
        if "title" in element.content:
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": element.content["title"],
                    "emoji": True
                }
            })
        if "message" in element.content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": element.content["message"]
                }
            })
        if "buttons" in element.content:
            blocks.extend(self._create_action_block(element.content["buttons"]))
        return blocks
    
    def _render_button_element(self, element: InteractiveElement) -> List[Dict]:
        """Render button element to Slack format."""
        blocks = []
        if "title" in element.content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{element.content['title']}*"
                }
            })
        if "buttons" in element.content:
            blocks.extend(self._create_action_block(element.content["buttons"]))
        return blocks
    
    def _render_table_element(self, element: InteractiveElement) -> List[Dict]:
        """Render table element to Slack format."""
        blocks = []
        if "title" in element.content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{element.content['title']}*"
                }
            })
        
        headers = element.content.get("headers", [])
        rows = element.content.get("rows", [])
        
        if headers and rows:
            # Create markdown table
            table_text = "```"
            table_text += " | ".join(headers) + "\n"
            table_text += "-" * (len(" | ".join(headers))) + "\n"
            for row in rows:
                table_text += " | ".join(str(cell) for cell in row) + "\n"
            table_text += "```"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": table_text
                }
            })
        
        return blocks
    
    def _render_progress_element(self, element: InteractiveElement) -> List[Dict]:
        """Render progress element to Slack format."""
        blocks = []
        if "title" in element.content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{element.content['title']}*"
                }
            })
        
        progress = element.content.get("progress", 0)
        percentage = element.content.get("percentage", f"{progress * 100:.1f}%")
        
        # Create progress bar
        progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"`{progress_bar}` {percentage}"
            }
        })
        
        return blocks
    
    def _create_action_block(self, buttons: List[Dict]) -> List[Dict]:
        """Create Slack action block with buttons."""
        elements = []
        for btn in buttons:
            elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": btn.get("text", ""),
                    "emoji": True
                },
                "style": btn.get("style", "default"),
                "value": json.dumps(btn.get("action", {})),
                "action_id": btn.get("id", f"button_{len(elements)}")
            })
        
        return [{
            "type": "actions",
            "elements": elements
        }]
    
    def validate_content(self, content: InteractiveContent) -> bool:
        """Validate content for Slack."""
        if content.channel_type != ChannelType.SLACK:
            return False
        for element in content.elements:
            if not element.validate():
                return False
        return True


class TelegramInteractiveRenderer(BaseInteractiveRenderer):
    """Renderer for Telegram inline keyboards."""
    
    def __init__(self):
        super().__init__(ChannelType.TELEGRAM)
    
    def render(self, content: InteractiveContent) -> Dict[str, Any]:
        """Render to Telegram inline keyboard format."""
        content = self._ensure_accessibility(content)
        
        text_parts = []
        keyboard = []
        
        for element in content.elements:
            if element.element_type == ContentType.CARD:
                text_parts.extend(self._render_card_text(element))
                keyboard.extend(self._render_card_keyboard(element))
            elif element.element_type == ContentType.BUTTONS:
                text_parts.extend(self._render_button_text(element))
                keyboard.extend(self._render_button_keyboard(element))
            elif element.element_type == ContentType.TABLE:
                text_parts.extend(self._render_table_text(element))
            elif element.element_type == ContentType.PROGRESS:
                text_parts.extend(self._render_progress_text(element))
        
        telegram_message = {
            "text": "\n".join(text_parts) if text_parts else content.fallback_text,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": keyboard
            } if keyboard else None
        }
        
        return telegram_message
    
    def _render_card_text(self, element: InteractiveElement) -> List[str]:
        """Render card element to Telegram text."""
        text_parts = []
        if "title" in element.content:
            text_parts.append(f"*{element.content['title']}*")
        if "message" in element.content:
            text_parts.append(element.content["message"])
        return text_parts
    
    def _render_card_keyboard(self, element: InteractiveElement) -> List[List[Dict]]:
        """Render card element to Telegram keyboard."""
        if "buttons" in element.content:
            return self._create_keyboard_row(element.content["buttons"])
        return []
    
    def _render_button_text(self, element: InteractiveElement) -> List[str]:
        """Render button element to Telegram text."""
        text_parts = []
        if "title" in element.content:
            text_parts.append(f"*{element.content['title']}*")
        return text_parts
    
    def _render_button_keyboard(self, element: InteractiveElement) -> List[List[Dict]]:
        """Render button element to Telegram keyboard."""
        if "buttons" in element.content:
            return self._create_keyboard_row(element.content["buttons"])
        return []
    
    def _render_table_text(self, element: InteractiveElement) -> List[str]:
        """Render table element to Telegram text."""
        text_parts = []
        if "title" in element.content:
            text_parts.append(f"*{element.content['title']}*")
        
        headers = element.content.get("headers", [])
        rows = element.content.get("rows", [])
        
        if headers and rows:
            # Create markdown table
            table_text = "```\n"
            table_text += " | ".join(headers) + "\n"
            table_text += "-" * (len(" | ".join(headers))) + "\n"
            for row in rows:
                table_text += " | ".join(str(cell) for cell in row) + "\n"
            table_text += "```"
            text_parts.append(table_text)
        
        return text_parts
    
    def _render_progress_text(self, element: InteractiveElement) -> List[str]:
        """Render progress element to Telegram text."""
        text_parts = []
        if "title" in element.content:
            text_parts.append(f"*{element.content['title']}*")
        
        progress = element.content.get("progress", 0)
        percentage = element.content.get("percentage", f"{progress * 100:.1f}%")
        
        # Create progress bar
        progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
        text_parts.append(f"`{progress_bar}` {percentage}")
        
        return text_parts
    
    def _create_keyboard_row(self, buttons: List[Dict]) -> List[List[Dict]]:
        """Create Telegram keyboard row from buttons."""
        row = []
        for btn in buttons:
            row.append({
                "text": btn.get("text", ""),
                "callback_data": json.dumps(btn.get("action", {}))
            })
        return [row] if row else []
    
    def validate_content(self, content: InteractiveContent) -> bool:
        """Validate content for Telegram."""
        if content.channel_type != ChannelType.TELEGRAM:
            return False
        for element in content.elements:
            if not element.validate():
                return False
        return True


class InteractiveMediaProcessor:
    """Main processor for interactive media content."""
    
    def __init__(self):
        self.renderers: Dict[ChannelType, BaseInteractiveRenderer] = {
            ChannelType.FEISHU: FeishuInteractiveRenderer(),
            ChannelType.SLACK: SlackInteractiveRenderer(),
            ChannelType.TELEGRAM: TelegramInteractiveRenderer()
        }
        self.template_registry = TemplateRegistry()
    
    def get_renderer(self, channel_type: Union[ChannelType, str]) -> Optional[BaseInteractiveRenderer]:
        """Get renderer for specified channel type."""
        if isinstance(channel_type, str):
            try:
                channel_type = ChannelType(channel_type)
            except ValueError:
                logger.error(f"Unknown channel type: {channel_type}")
                return None
        return self.renderers.get(channel_type)
    
    def process(
        self,
        content: InteractiveContent,
        channel_type: Optional[ChannelType] = None
    ) -> Optional[Dict[str, Any]]:
        """Process interactive content for specified channel."""
        target_channel = channel_type or content.channel_type
        renderer = self.get_renderer(target_channel)
        
        if not renderer:
            logger.error(f"No renderer available for channel: {target_channel}")
            return None
        
        if not renderer.validate_content(content):
            logger.error(f"Invalid content for channel: {target_channel}")
            return None
        
        try:
            return renderer.render(content)
        except Exception as e:
            logger.error(f"Error rendering content for {target_channel}: {e}")
            return None
    
    def create_from_template(
        self,
        template_name: str,
        channel_type: ChannelType,
        **kwargs
    ) -> Optional[InteractiveContent]:
        """Create interactive content from a template."""
        renderer = self.get_renderer(channel_type)
        if not renderer:
            return None
        
        element = renderer.apply_template(template_name, **kwargs)
        if not element:
            return None
        
        return InteractiveContent(
            channel_type=channel_type,
            elements=[element],
            fallback_text=kwargs.get("fallback_text", "Interactive content")
        )
    
    def register_template(self, name: str, template_func: Callable):
        """Register a custom template."""
        self.template_registry.register(name, template_func)
        # Also register with all renderers
        for renderer in self.renderers.values():
            renderer.template_registry.register(name, template_func)
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return self.template_registry.list_templates()
    
    def convert_to_all_channels(
        self,
        content: InteractiveContent
    ) -> Dict[ChannelType, Dict[str, Any]]:
        """Convert content to all supported channel formats."""
        results = {}
        for channel_type in self.renderers.keys():
            converted = self.process(content, channel_type)
            if converted:
                results[channel_type] = converted
        return results


# Factory function for easy access
def create_interactive_processor() -> InteractiveMediaProcessor:
    """Create and return an InteractiveMediaProcessor instance."""
    return InteractiveMediaProcessor()


# Example usage and integration helper
def render_agent_output_to_channel(
    agent_output: Dict[str, Any],
    channel_type: Union[ChannelType, str],
    fallback_text: str = "Interactive content from agent"
) -> Optional[Dict[str, Any]]:
    """
    Convert agent output to channel-specific interactive content.
    
    This is a convenience function for integrating with existing agent systems.
    """
    processor = create_interactive_processor()
    
    # Convert agent output to InteractiveContent
    # This is a simplified example - adapt based on your agent output format
    elements = []
    
    if "buttons" in agent_output:
        elements.append(InteractiveElement(
            element_id="agent_buttons",
            element_type=ContentType.BUTTONS,
            content=agent_output["buttons"]
        ))
    
    if "card" in agent_output:
        elements.append(InteractiveElement(
            element_id="agent_card",
            element_type=ContentType.CARD,
            content=agent_output["card"]
        ))
    
    if "table" in agent_output:
        elements.append(InteractiveElement(
            element_id="agent_table",
            element_type=ContentType.TABLE,
            content=agent_output["table"]
        ))
    
    if "progress" in agent_output:
        elements.append(InteractiveElement(
            element_id="agent_progress",
            element_type=ContentType.PROGRESS,
            content=agent_output["progress"]
        ))
    
    if not elements:
        logger.warning("No interactive elements found in agent output")
        return None
    
    content = InteractiveContent(
        channel_type=ChannelType(channel_type) if isinstance(channel_type, str) else channel_type,
        elements=elements,
        fallback_text=fallback_text,
        metadata=agent_output.get("metadata", {})
    )
    
    return processor.process(content)


# Integration with existing channel modules
class ChannelInteractiveMixin:
    """
    Mixin class for adding interactive content support to channel classes.
    
    Can be used with existing channel classes like FeishuChannel, SlackChannel, etc.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_processor = create_interactive_processor()
    
    def send_interactive_content(
        self,
        content: InteractiveContent,
        recipient: str,
        **kwargs
    ) -> bool:
        """Send interactive content through the channel."""
        rendered = self._interactive_processor.process(content, self.channel_type)
        if not rendered:
            logger.error(f"Failed to render interactive content for {self.channel_type}")
            return False
        
        # Call the channel's send method with rendered content
        # This assumes the channel has a method like `send_message` or similar
        if hasattr(self, 'send_message'):
            return self.send_message(recipient, rendered, **kwargs)
        elif hasattr(self, 'send'):
            return self.send(recipient, rendered, **kwargs)
        else:
            logger.error(f"Channel {self.channel_type} does not have a send method")
            return False
    
    def send_template(
        self,
        template_name: str,
        recipient: str,
        **template_kwargs
    ) -> bool:
        """Send content created from a template."""
        content = self._interactive_processor.create_from_template(
            template_name,
            self.channel_type,
            **template_kwargs
        )
        if not content:
            return False
        return self.send_interactive_content(content, recipient)


# Export main classes and functions
__all__ = [
    "ChannelType",
    "ContentType",
    "AccessibilityOptions",
    "InteractiveElement",
    "InteractiveContent",
    "TemplateRegistry",
    "BaseInteractiveRenderer",
    "FeishuInteractiveRenderer",
    "SlackInteractiveRenderer",
    "TelegramInteractiveRenderer",
    "InteractiveMediaProcessor",
    "create_interactive_processor",
    "render_agent_output_to_channel",
    "ChannelInteractiveMixin"
]