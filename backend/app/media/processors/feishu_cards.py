"""
Feishu Interactive Cards Processor
===================================

Channel-specific rich media processor for Feishu interactive cards.
Converts agent outputs to Feishu card messages with templates for common UI patterns.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from backend.app.channels.base import BaseChannel
from backend.app.channels.feishu import FeishuChannel
from backend.app.media.processors.base import BaseMediaProcessor

logger = logging.getLogger(__name__)


class FeishuCardType(Enum):
    """Feishu card types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    DATA = "data"
    APPROVAL = "approval"
    PROGRESS = "progress"
    TABLE = "table"
    CUSTOM = "custom"


@dataclass
class FeishuCardElement:
    """Base class for Feishu card elements."""
    tag: str
    content: Optional[str] = None
    text: Optional[Dict[str, Any]] = None
    elements: Optional[List[Dict[str, Any]]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert element to Feishu card format."""
        result = {"tag": self.tag}
        
        if self.content:
            result["content"] = self.content
        if self.text:
            result["text"] = self.text
        if self.elements:
            result["elements"] = self.elements
        if self.actions:
            result["actions"] = self.actions
        
        result.update(self.extra)
        return result


@dataclass
class FeishuCard:
    """Feishu interactive card structure."""
    header: Dict[str, Any]
    elements: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=lambda: {"wide_screen_mode": True})
    card_link: Optional[Dict[str, Any]] = None
    i18n_elements: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert card to Feishu API format."""
        card = {
            "config": self.config,
            "header": self.header,
            "elements": self.elements
        }
        
        if self.card_link:
            card["card_link"] = self.card_link
        if self.i18n_elements:
            card["i18n_elements"] = self.i18n_elements
        
        return card


class FeishuCardTemplates:
    """Template system for common Feishu card patterns."""
    
    @staticmethod
    def create_header(
        title: str,
        template: FeishuCardType = FeishuCardType.INFO,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create card header with appropriate styling."""
        colors = {
            FeishuCardType.INFO: "blue",
            FeishuCardType.SUCCESS: "green",
            FeishuCardType.WARNING: "orange",
            FeishuCardType.ERROR: "red",
            FeishuCardType.DATA: "purple",
            FeishuCardType.APPROVAL: "yellow",
            FeishuCardType.PROGRESS: "blue",
            FeishuCardType.TABLE: "turquoise",
            FeishuCardType.CUSTOM: "grey"
        }
        
        icons = {
            FeishuCardType.INFO: "info",
            FeishuCardType.SUCCESS: "success",
            FeishuCardType.WARNING: "warning",
            FeishuCardType.ERROR: "error",
            FeishuCardType.DATA: "data",
            FeishuCardType.APPROVAL: "approval",
            FeishuCardType.PROGRESS: "progress",
            FeishuCardType.TABLE: "table",
            FeishuCardType.CUSTOM: "custom"
        }
        
        header = {
            "title": {
                "tag": "plain_text",
                "content": title
            },
            "template": colors.get(template, "blue")
        }
        
        if subtitle:
            header["subtitle"] = {
                "tag": "plain_text",
                "content": subtitle
            }
        
        if icon:
            header["icon"] = {
                "tag": "standard_icon",
                "token": icon
            }
        elif template in icons:
            header["icon"] = {
                "tag": "standard_icon",
                "token": icons[template]
            }
        
        return header
    
    @staticmethod
    def create_text_element(
        content: str,
        text_type: str = "plain_text",
        is_markdown: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create text element with accessibility support."""
        element = {
            "tag": "div",
            "text": {
                "tag": text_type,
                "content": content
            }
        }
        
        if is_markdown:
            element["text"]["tag"] = "markdown"
        
        # Add accessibility attributes
        if "aria_label" in kwargs:
            element["text"]["aria_label"] = kwargs["aria_label"]
        
        element.update(kwargs)
        return element
    
    @staticmethod
    def create_approval_buttons(
        approve_text: str = "Approve",
        reject_text: str = "Reject",
        callback_data: Optional[Dict[str, Any]] = None,
        confirm_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Create approval button group with confirmation."""
        actions = []
        
        approve_button = {
            "tag": "button",
            "text": {
                "tag": "plain_text",
                "content": approve_text
            },
            "type": "primary",
            "value": {"action": "approve"}
        }
        
        reject_button = {
            "tag": "button",
            "text": {
                "tag": "plain_text",
                "content": reject_text
            },
            "type": "danger",
            "value": {"action": "reject"}
        }
        
        if callback_data:
            approve_button["value"].update(callback_data)
            reject_button["value"].update(callback_data)
        
        if confirm_text:
            approve_button["confirm"] = {
                "title": {
                    "tag": "plain_text",
                    "content": "Confirmation"
                },
                "text": {
                    "tag": "plain_text",
                    "content": confirm_text
                }
            }
        
        return [
            {
                "tag": "action",
                "actions": [approve_button, reject_button]
            }
        ]
    
    @staticmethod
    def create_progress_bar(
        progress: float,
        text: str = "Progress",
        show_percentage: bool = True
    ) -> Dict[str, Any]:
        """Create progress bar element."""
        progress_text = f"{progress:.0%}" if show_percentage else ""
        
        return {
            "tag": "div",
            "text": {
                "tag": "plain_text",
                "content": f"{text}: {progress_text}" if text else progress_text
            },
            "extra": {
                "tag": "progress",
                "value": progress
            }
        }
    
    @staticmethod
    def create_data_table(
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Create data table with accessibility support."""
        elements = []
        
        if title:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": title
                }
            })
        
        # Create table
        table = {
            "tag": "table",
            "rows": []
        }
        
        # Add header row
        if headers:
            header_row = {
                "tag": "tr",
                "cells": [
                    {
                        "tag": "td",
                        "text": {
                            "tag": "plain_text",
                            "content": header
                        }
                    } for header in headers
                ]
            }
            table["rows"].append(header_row)
        
        # Add data rows
        for row in rows:
            data_row = {
                "tag": "tr",
                "cells": [
                    {
                        "tag": "td",
                        "text": {
                            "tag": "plain_text",
                            "content": cell
                        }
                    } for cell in row
                ]
            }
            table["rows"].append(data_row)
        
        elements.append(table)
        return elements
    
    @staticmethod
    def create_note_element(
        content: str,
        is_alert: bool = False,
        alert_type: str = "info"
    ) -> Dict[str, Any]:
        """Create note or alert element."""
        element = {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": content
                }
            ]
        }
        
        if is_alert:
            element["elements"].append({
                "tag": "standard_icon",
                "token": alert_type
            })
        
        return element
    
    @staticmethod
    def create_image_element(
        image_key: str,
        alt_text: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create image element with accessibility."""
        element = {
            "tag": "img",
            "img_key": image_key,
            "alt": {
                "tag": "plain_text",
                "content": alt_text
            }
        }
        
        if width:
            element["width"] = width
        if height:
            element["height"] = height
        
        return element


class FeishuCardProcessor(BaseMediaProcessor):
    """
    Processor for Feishu interactive cards.
    
    Converts agent outputs to Feishu card messages with templates
    for common UI patterns and accessibility features.
    """
    
    def __init__(self):
        """Initialize Feishu card processor."""
        self.templates = FeishuCardTemplates()
        self._supported_channels = {"feishu"}
    
    @property
    def supported_channels(self) -> set:
        """Return set of supported channel types."""
        return self._supported_channels
    
    def process(
        self,
        content: Union[str, Dict[str, Any]],
        channel: BaseChannel,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process content into Feishu card format.
        
        Args:
            content: Agent output content (text or structured data)
            channel: Target channel instance
            context: Optional context for processing
            
        Returns:
            Feishu card message in API format
        """
        if not isinstance(channel, FeishuChannel):
            raise ValueError("FeishuCardProcessor only supports Feishu channels")
        
        # Parse content if it's a string (JSON)
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"text": content, "type": "info"}
        
        # Determine card type
        card_type = self._determine_card_type(content, context)
        
        # Build card based on content type
        if isinstance(content, dict):
            card = self._build_card_from_dict(content, card_type, context)
        else:
            card = self._build_card_from_text(str(content), card_type, context)
        
        # Convert to Feishu API format
        return self._to_feishu_message(card, channel, context)
    
    def _determine_card_type(
        self,
        content: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> FeishuCardType:
        """Determine appropriate card type from content."""
        if isinstance(content, dict):
            content_type = content.get("type", "").lower()
            
            type_mapping = {
                "info": FeishuCardType.INFO,
                "success": FeishuCardType.SUCCESS,
                "warning": FeishuCardType.WARNING,
                "error": FeishuCardType.ERROR,
                "data": FeishuCardType.DATA,
                "approval": FeishuCardType.APPROVAL,
                "progress": FeishuCardType.PROGRESS,
                "table": FeishuCardType.TABLE
            }
            
            if content_type in type_mapping:
                return type_mapping[content_type]
        
        # Check context for hints
        if context:
            if context.get("requires_approval"):
                return FeishuCardType.APPROVAL
            elif context.get("has_progress"):
                return FeishuCardType.PROGRESS
            elif context.get("has_data"):
                return FeishuCardType.DATA
        
        return FeishuCardType.INFO
    
    def _build_card_from_dict(
        self,
        content: Dict[str, Any],
        card_type: FeishuCardType,
        context: Optional[Dict[str, Any]]
    ) -> FeishuCard:
        """Build card from structured dictionary content."""
        elements = []
        
        # Header
        title = content.get("title", "Agent Response")
        subtitle = content.get("subtitle")
        header = self.templates.create_header(title, card_type, subtitle)
        
        # Main content
        if "text" in content:
            elements.append(self.templates.create_text_element(
                content["text"],
                is_markdown=content.get("is_markdown", False),
                aria_label=content.get("aria_label")
            ))
        
        # Handle specific card types
        if card_type == FeishuCardType.APPROVAL:
            elements.extend(self._build_approval_elements(content, context))
        elif card_type == FeishuCardType.PROGRESS:
            elements.extend(self._build_progress_elements(content, context))
        elif card_type == FeishuCardType.TABLE:
            elements.extend(self._build_table_elements(content, context))
        elif card_type == FeishuCardType.DATA:
            elements.extend(self._build_data_elements(content, context))
        
        # Add custom elements if provided
        if "elements" in content:
            elements.extend(content["elements"])
        
        # Add actions if provided
        if "actions" in content:
            elements.append({
                "tag": "action",
                "actions": content["actions"]
            })
        
        # Add notes/alerts
        if "note" in content:
            elements.append(self.templates.create_note_element(
                content["note"],
                content.get("is_alert", False),
                content.get("alert_type", "info")
            ))
        
        # Add images
        if "images" in content:
            for image in content["images"]:
                if isinstance(image, str):
                    elements.append(self.templates.create_image_element(image))
                elif isinstance(image, dict):
                    elements.append(self.templates.create_image_element(
                        image.get("key", ""),
                        image.get("alt", ""),
                        image.get("width"),
                        image.get("height")
                    ))
        
        # Card link
        card_link = content.get("card_link")
        
        # Internationalization support
        i18n_elements = content.get("i18n_elements")
        
        return FeishuCard(
            header=header,
            elements=elements,
            card_link=card_link,
            i18n_elements=i18n_elements
        )
    
    def _build_card_from_text(
        self,
        text: str,
        card_type: FeishuCardType,
        context: Optional[Dict[str, Any]]
    ) -> FeishuCard:
        """Build card from plain text."""
        header = self.templates.create_header("Agent Response", card_type)
        
        elements = [
            self.templates.create_text_element(text)
        ]
        
        return FeishuCard(header=header, elements=elements)
    
    def _build_approval_elements(
        self,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build approval-specific elements."""
        elements = []
        
        # Add approval context if provided
        if "approval_context" in content:
            elements.append(self.templates.create_text_element(
                content["approval_context"],
                aria_label="Approval context"
            ))
        
        # Create approval buttons
        approve_text = content.get("approve_text", "Approve")
        reject_text = content.get("reject_text", "Reject")
        callback_data = content.get("callback_data")
        confirm_text = content.get("confirm_text")
        
        elements.extend(self.templates.create_approval_buttons(
            approve_text=approve_text,
            reject_text=reject_text,
            callback_data=callback_data,
            confirm_text=confirm_text
        ))
        
        return elements
    
    def _build_progress_elements(
        self,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build progress-specific elements."""
        elements = []
        
        progress = content.get("progress", 0.0)
        progress_text = content.get("progress_text", "Progress")
        show_percentage = content.get("show_percentage", True)
        
        elements.append(self.templates.create_progress_bar(
            progress=progress,
            text=progress_text,
            show_percentage=show_percentage
        ))
        
        # Add progress details
        if "progress_details" in content:
            elements.append(self.templates.create_text_element(
                content["progress_details"],
                aria_label="Progress details"
            ))
        
        return elements
    
    def _build_table_elements(
        self,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build table-specific elements."""
        headers = content.get("headers", [])
        rows = content.get("rows", [])
        title = content.get("table_title")
        
        return self.templates.create_data_table(
            headers=headers,
            rows=rows,
            title=title
        )
    
    def _build_data_elements(
        self,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build data visualization elements."""
        elements = []
        
        # Add data summary
        if "summary" in content:
            elements.append(self.templates.create_text_element(
                content["summary"],
                aria_label="Data summary"
            ))
        
        # Add metrics
        if "metrics" in content:
            for metric in content["metrics"]:
                if isinstance(metric, dict):
                    metric_text = f"{metric.get('label', '')}: {metric.get('value', '')}"
                    elements.append(self.templates.create_text_element(
                        metric_text,
                        aria_label=metric.get("aria_label", metric.get("label", ""))
                    ))
        
        # Add charts/visualizations as images
        if "chart" in content:
            chart = content["chart"]
            if isinstance(chart, str):
                elements.append(self.templates.create_image_element(
                    chart,
                    alt_text=content.get("chart_alt", "Data visualization")
                ))
            elif isinstance(chart, dict):
                elements.append(self.templates.create_image_element(
                    chart.get("key", ""),
                    alt_text=chart.get("alt", "Data visualization"),
                    width=chart.get("width"),
                    height=chart.get("height")
                ))
        
        return elements
    
    def _to_feishu_message(
        self,
        card: FeishuCard,
        channel: FeishuChannel,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert card to Feishu message format."""
        message = {
            "msg_type": "interactive",
            "card": card.to_dict()
        }
        
        # Add message metadata
        if context:
            if "message_id" in context:
                message["message_id"] = context["message_id"]
            if "root_id" in context:
                message["root_id"] = context["root_id"]
            if "parent_id" in context:
                message["parent_id"] = context["parent_id"]
        
        return message
    
    def create_approval_card(
        self,
        title: str,
        description: str,
        approve_callback: str,
        reject_callback: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an approval card with standard structure.
        
        Args:
            title: Approval request title
            description: Detailed description
            approve_callback: Callback ID for approval
            reject_callback: Callback ID for rejection
            metadata: Additional metadata
            
        Returns:
            Feishu approval card
        """
        content = {
            "type": "approval",
            "title": title,
            "text": description,
            "approve_text": "Approve",
            "reject_text": "Reject",
            "callback_data": {
                "approve_callback": approve_callback,
                "reject_callback": reject_callback,
                "metadata": metadata or {}
            },
            "confirm_text": "Are you sure you want to approve this request?"
        }
        
        # Create a dummy channel for processing
        # In real usage, this would be called from the actual channel
        from backend.app.channels.manager import ChannelManager
        channel_manager = ChannelManager()
        feishu_channel = channel_manager.get_channel("feishu")
        
        return self.process(content, feishu_channel)
    
    def create_progress_card(
        self,
        title: str,
        progress: float,
        details: Optional[str] = None,
        status: str = "in_progress"
    ) -> Dict[str, Any]:
        """
        Create a progress tracking card.
        
        Args:
            title: Progress title
            progress: Progress value (0.0 to 1.0)
            details: Optional progress details
            status: Progress status
            
        Returns:
            Feishu progress card
        """
        content = {
            "type": "progress",
            "title": title,
            "progress": progress,
            "progress_text": "Progress",
            "show_percentage": True
        }
        
        if details:
            content["progress_details"] = details
        
        # Create a dummy channel for processing
        from backend.app.channels.manager import ChannelManager
        channel_manager = ChannelManager()
        feishu_channel = channel_manager.get_channel("feishu")
        
        return self.process(content, feishu_channel)
    
    def create_data_card(
        self,
        title: str,
        headers: List[str],
        rows: List[List[str]],
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a data table card.
        
        Args:
            title: Table title
            headers: Table headers
            rows: Table rows
            summary: Optional data summary
            
        Returns:
            Feishu data table card
        """
        content = {
            "type": "table",
            "title": title,
            "headers": headers,
            "rows": rows,
            "table_title": title
        }
        
        if summary:
            content["summary"] = summary
        
        # Create a dummy channel for processing
        from backend.app.channels.manager import ChannelManager
        channel_manager = ChannelManager()
        feishu_channel = channel_manager.get_channel("feishu")
        
        return self.process(content, feishu_channel)
    
    def validate_card(self, card: Dict[str, Any]) -> bool:
        """
        Validate Feishu card structure.
        
        Args:
            card: Card to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if "card" not in card:
                return False
            
            card_data = card["card"]
            if "header" not in card_data or "elements" not in card_data:
                return False
            
            # Validate header structure
            header = card_data["header"]
            if "title" not in header:
                return False
            
            # Validate elements
            elements = card_data["elements"]
            if not isinstance(elements, list):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Card validation failed: {e}")
            return False
    
    def add_accessibility_features(
        self,
        card: Dict[str, Any],
        screen_reader_text: Optional[str] = None,
        high_contrast: bool = False
    ) -> Dict[str, Any]:
        """
        Add accessibility features to a card.
        
        Args:
            card: Card to enhance
            screen_reader_text: Additional text for screen readers
            high_contrast: Whether to use high contrast mode
            
        Returns:
            Enhanced card with accessibility features
        """
        if "card" not in card:
            return card
        
        card_data = card["card"]
        
        # Add screen reader text
        if screen_reader_text:
            if "config" not in card_data:
                card_data["config"] = {}
            card_data["config"]["screen_reader"] = screen_reader_text
        
        # Apply high contrast if requested
        if high_contrast:
            if "config" not in card_data:
                card_data["config"] = {}
            card_data["config"]["high_contrast"] = True
        
        # Add ARIA labels to interactive elements
        for element in card_data.get("elements", []):
            if element.get("tag") == "action":
                for action in element.get("actions", []):
                    if "aria_label" not in action and "text" in action:
                        action["aria_label"] = action["text"].get("content", "")
        
        return card


# Factory function for easy instantiation
def create_feishu_card_processor() -> FeishuCardProcessor:
    """Create and return a FeishuCardProcessor instance."""
    return FeishuCardProcessor()


# Integration helper for existing channels
def integrate_with_feishu_channel(channel: FeishuChannel) -> None:
    """
    Integrate card processor with Feishu channel.
    
    This function should be called during channel initialization
    to enable rich card processing.
    """
    processor = create_feishu_card_processor()
    
    # Monkey-patch the send_message method to support cards
    original_send = channel.send_message
    
    async def enhanced_send_message(
        self,
        content: Union[str, Dict[str, Any]],
        target: str,
        **kwargs
    ) -> bool:
        """Enhanced send_message with card support."""
        # Check if content should be sent as a card
        if isinstance(content, dict) and content.get("msg_type") == "interactive":
            # Already a card, send directly
            return await original_send(content, target, **kwargs)
        
        # Try to process as card if it looks like structured data
        if isinstance(content, dict):
            try:
                card_message = processor.process(content, self, kwargs.get("context"))
                return await original_send(card_message, target, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to process as card, falling back to text: {e}")
        
        # Fall back to original text sending
        return await original_send(content, target, **kwargs)
    
    # Replace the method
    channel.send_message = enhanced_send_message.__get__(channel, FeishuChannel)
    
    # Add processor reference to channel
    channel.card_processor = processor
    
    logger.info("Feishu card processor integrated with channel")