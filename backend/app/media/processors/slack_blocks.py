"""
Slack Blocks Media Processor
Converts agent outputs to native Slack Block Kit format with rich interactive elements.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from backend.app.channels.slack import SlackChannel
from backend.app.media.processors.base import BaseMediaProcessor

logger = logging.getLogger(__name__)


class SlackBlockProcessor(BaseMediaProcessor):
    """
    Processes agent outputs into Slack Block Kit format.
    Supports interactive elements, data visualization, and accessibility features.
    """
    
    PLATFORM = "slack"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.template_registry = self._register_templates()
        self.accessibility_enabled = config.get("accessibility_enabled", True) if config else True
        
    def _register_templates(self) -> Dict[str, callable]:
        """Register common UI pattern templates."""
        return {
            "approval": self._create_approval_template,
            "progress": self._create_progress_template,
            "data_table": self._create_data_table_template,
            "status_card": self._create_status_card_template,
            "interactive_form": self._create_interactive_form_template,
            "notification": self._create_notification_template,
        }
    
    def process(self, content: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content into Slack Block Kit format.
        
        Args:
            content: Agent output to process (can be dict, string, or structured data)
            context: Additional context including channel, user, message metadata
            
        Returns:
            Slack Block Kit formatted message
        """
        context = context or {}
        
        try:
            # Determine content type and process accordingly
            if isinstance(content, str):
                blocks = self._process_text_content(content, context)
            elif isinstance(content, dict):
                blocks = self._process_structured_content(content, context)
            elif isinstance(content, list):
                blocks = self._process_list_content(content, context)
            else:
                blocks = self._process_generic_content(content, context)
            
            # Apply accessibility features
            if self.accessibility_enabled:
                blocks = self._apply_accessibility_features(blocks, context)
            
            # Format final message
            message = self._format_slack_message(blocks, context)
            
            logger.debug(f"Processed content into Slack blocks: {len(blocks)} blocks")
            return message
            
        except Exception as e:
            logger.error(f"Error processing content to Slack blocks: {e}")
            return self._create_fallback_message(str(content), context)
    
    def _process_text_content(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process plain text content."""
        blocks = []
        
        # Split long text into sections
        sections = self._split_text_into_sections(text)
        
        for section in sections:
            if section.strip():
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": self._format_markdown(section)
                    }
                })
        
        return blocks
    
    def _process_structured_content(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process structured content (dicts with type hints)."""
        content_type = data.get("type", "text")
        
        # Check if we have a template for this type
        if content_type in self.template_registry:
            return self.template_registry[content_type](data, context)
        
        # Default processing for unknown structured content
        blocks = []
        
        # Add header if present
        if "title" in data:
            blocks.append(self._create_header_block(data["title"]))
        
        # Add main content
        if "content" in data:
            blocks.extend(self._process_text_content(str(data["content"]), context))
        
        # Add metadata
        if "metadata" in data:
            blocks.extend(self._create_metadata_section(data["metadata"]))
        
        # Add actions if present
        if "actions" in data:
            blocks.append(self._create_actions_block(data["actions"]))
        
        return blocks
    
    def _process_list_content(self, items: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process list content."""
        blocks = []
        
        for item in items:
            if isinstance(item, dict):
                item_blocks = self._process_structured_content(item, context)
            else:
                item_blocks = self._process_text_content(str(item), context)
            
            blocks.extend(item_blocks)
            
            # Add divider between items
            if len(items) > 1 and item != items[-1]:
                blocks.append({"type": "divider"})
        
        return blocks
    
    def _process_generic_content(self, content: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process generic content types."""
        return self._process_text_content(str(content), context)
    
    def _create_approval_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create approval request template with interactive buttons."""
        blocks = []
        
        # Header
        title = data.get("title", "Approval Required")
        blocks.append(self._create_header_block(title))
        
        # Description
        description = data.get("description", "")
        if description:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": description
                }
            })
        
        # Details section
        if "details" in data:
            blocks.extend(self._create_details_section(data["details"]))
        
        # Action buttons
        actions = []
        approve_text = data.get("approve_text", "✅ Approve")
        reject_text = data.get("reject_text", "❌ Reject")
        
        actions.append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": approve_text,
                "emoji": True
            },
            "style": "primary",
            "value": "approve",
            "action_id": f"approval_approve_{context.get('message_id', 'unknown')}"
        })
        
        actions.append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": reject_text,
                "emoji": True
            },
            "style": "danger",
            "value": "reject",
            "action_id": f"approval_reject_{context.get('message_id', 'unknown')}"
        })
        
        # Add optional additional actions
        if "additional_actions" in data:
            for action in data["additional_actions"]:
                actions.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get("text", "Action"),
                        "emoji": True
                    },
                    "value": action.get("value", "action"),
                    "action_id": f"approval_action_{action.get('id', 'custom')}_{context.get('message_id', 'unknown')}"
                })
        
        blocks.append({
            "type": "actions",
            "elements": actions
        })
        
        # Footer with metadata
        if "footer" in data:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": data["footer"]
                    }
                ]
            })
        
        return blocks
    
    def _create_progress_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create progress indicator template."""
        blocks = []
        
        # Header
        title = data.get("title", "Progress Update")
        blocks.append(self._create_header_block(title))
        
        # Progress bar
        progress = data.get("progress", 0)
        status = data.get("status", "in_progress")
        
        # Create visual progress bar using emoji
        bar_length = 10
        filled_length = int(bar_length * progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        status_emoji = {
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
            "paused": "⏸️"
        }.get(status, "🔄")
        
        progress_text = (
            f"{status_emoji} *Progress:* {bar} {progress}%\n"
            f"*Status:* {status.replace('_', ' ').title()}"
        )
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": progress_text
            }
        })
        
        # Details
        if "details" in data:
            details = data["details"]
            detail_text = "\n".join([f"• *{k}:* {v}" for k, v in details.items()])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": detail_text
                }
            })
        
        # Estimated completion
        if "eta" in data:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏱️ *ETA:* {data['eta']}"
                    }
                ]
            })
        
        return blocks
    
    def _create_data_table_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create data table template."""
        blocks = []
        
        # Header
        title = data.get("title", "Data Table")
        blocks.append(self._create_header_block(title))
        
        # Table data
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        
        if headers and rows:
            # Format as markdown table
            table_lines = []
            
            # Header row
            header_line = " | ".join([f"*{h}*" for h in headers])
            table_lines.append(header_line)
            
            # Separator
            separator = " | ".join(["---"] * len(headers))
            table_lines.append(separator)
            
            # Data rows
            for row in rows:
                if len(row) == len(headers):
                    row_line = " | ".join([str(cell) for cell in row])
                    table_lines.append(row_line)
            
            table_text = "\n".join(table_lines)
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```\n{table_text}\n```"
                }
            })
        
        # Summary
        if "summary" in data:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": data["summary"]
                    }
                ]
            })
        
        return blocks
    
    def _create_status_card_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create status card template."""
        blocks = []
        
        # Status indicator
        status = data.get("status", "unknown")
        status_config = {
            "success": {"emoji": "✅", "color": "#36a64f"},
            "warning": {"emoji": "⚠️", "color": "#ff9900"},
            "error": {"emoji": "❌", "color": "#ff0000"},
            "info": {"emoji": "ℹ️", "color": "#17a2b8"},
            "unknown": {"emoji": "❓", "color": "#6c757d"}
        }.get(status, {"emoji": "❓", "color": "#6c757d"})
        
        # Header with status
        title = data.get("title", "Status Update")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_config['emoji']} {title}",
                "emoji": True
            }
        })
        
        # Main content
        if "message" in data:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": data["message"]
                }
            })
        
        # Metrics
        if "metrics" in data:
            metrics = data["metrics"]
            metric_blocks = []
            
            for metric in metrics[:4]:  # Limit to 4 metrics for readability
                metric_blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{metric.get('label', 'Metric')}:*\n{metric.get('value', 'N/A')}"
                    }
                })
            
            # Add metrics in columns if multiple
            if len(metric_blocks) > 1:
                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*{m.get('label', 'Metric')}:*\n{m.get('value', 'N/A')}"
                        }
                        for m in metrics[:10]  # Slack allows up to 10 fields
                    ]
                })
            else:
                blocks.extend(metric_blocks)
        
        # Timestamp
        if "timestamp" in data:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🕐 {data['timestamp']}"
                    }
                ]
            })
        
        return blocks
    
    def _create_interactive_form_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create interactive form template."""
        blocks = []
        
        # Header
        title = data.get("title", "Form")
        blocks.append(self._create_header_block(title))
        
        # Description
        if "description" in data:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": data["description"]
                }
            })
        
        # Form fields
        fields = data.get("fields", [])
        for field in fields:
            field_type = field.get("type", "text")
            
            if field_type == "text":
                blocks.append({
                    "type": "input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": field.get("id", "text_input"),
                        "placeholder": {
                            "type": "plain_text",
                            "text": field.get("placeholder", "Enter text...")
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": field.get("label", "Field"),
                        "emoji": True
                    },
                    "optional": field.get("optional", False)
                })
            
            elif field_type == "select":
                options = [
                    {
                        "text": {
                            "type": "plain_text",
                            "text": opt.get("text", opt.get("value", "Option")),
                            "emoji": True
                        },
                        "value": opt.get("value", opt.get("text", "option"))
                    }
                    for opt in field.get("options", [])
                ]
                
                blocks.append({
                    "type": "input",
                    "element": {
                        "type": "static_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": field.get("placeholder", "Select an option"),
                            "emoji": True
                        },
                        "options": options,
                        "action_id": field.get("id", "select_input")
                    },
                    "label": {
                        "type": "plain_text",
                        "text": field.get("label", "Select"),
                        "emoji": True
                    },
                    "optional": field.get("optional", False)
                })
        
        # Submit button
        submit_text = data.get("submit_text", "Submit")
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": submit_text,
                        "emoji": True
                    },
                    "style": "primary",
                    "value": "submit",
                    "action_id": f"form_submit_{context.get('message_id', 'unknown')}"
                }
            ]
        })
        
        return blocks
    
    def _create_notification_template(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create notification template."""
        blocks = []
        
        # Notification type
        notification_type = data.get("notification_type", "info")
        
        # Header with appropriate emoji
        emoji_map = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "mention": "👋",
            "reminder": "⏰"
        }
        
        emoji = emoji_map.get(notification_type, "ℹ️")
        title = data.get("title", "Notification")
        
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True
            }
        })
        
        # Message
        if "message" in data:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": data["message"]
                }
            })
        
        # Source/context
        if "source" in data:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📌 *Source:* {data['source']}"
                    }
                ]
            })
        
        # Actions
        if "actions" in data:
            action_elements = []
            for action in data["actions"]:
                action_elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get("text", "Action"),
                        "emoji": True
                    },
                    "url": action.get("url"),
                    "value": action.get("value"),
                    "action_id": f"notification_{action.get('id', 'action')}_{context.get('message_id', 'unknown')}"
                })
            
            blocks.append({
                "type": "actions",
                "elements": action_elements
            })
        
        return blocks
    
    def _create_header_block(self, title: str) -> Dict[str, Any]:
        """Create a header block."""
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
                "emoji": True
            }
        }
    
    def _create_details_section(self, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a details section from a dictionary."""
        blocks = []
        
        if details:
            detail_text = "\n".join([f"• *{k}:* {v}" for k, v in details.items()])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": detail_text
                }
            })
        
        return blocks
    
    def _create_metadata_section(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a metadata section."""
        blocks = []
        
        if metadata:
            metadata_text = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": metadata_text
                    }
                ]
            })
        
        return blocks
    
    def _create_actions_block(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an actions block with buttons."""
        action_elements = []
        
        for action in actions:
            action_elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": action.get("text", "Action"),
                    "emoji": True
                },
                "style": action.get("style"),
                "value": action.get("value", "action"),
                "action_id": action.get("action_id", f"action_{len(action_elements)}")
            })
        
        return {
            "type": "actions",
            "elements": action_elements
        }
    
    def _apply_accessibility_features(self, blocks: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply accessibility features to blocks."""
        # Add alt text to images
        for block in blocks:
            if block.get("type") == "image":
                if "alt_text" not in block:
                    block["alt_text"] = block.get("title", "Image")
            
            # Ensure all interactive elements have proper labels
            if block.get("type") == "actions":
                for element in block.get("elements", []):
                    if element.get("type") == "button" and "accessibility_label" not in element:
                        element["accessibility_label"] = element.get("text", {}).get("text", "Button")
        
        return blocks
    
    def _format_markdown(self, text: str) -> str:
        """Format text with Slack markdown."""
        # Basic markdown formatting
        # Convert **bold** to *bold*
        text = text.replace("**", "*")
        
        # Convert __italic__ to _italic_
        text = text.replace("__", "_")
        
        # Convert `code` to `code` (already correct)
        # Convert ```code blocks``` to ```code blocks```
        
        return text
    
    def _split_text_into_sections(self, text: str, max_length: int = 3000) -> List[str]:
        """Split long text into sections for Slack's character limits."""
        if len(text) <= max_length:
            return [text]
        
        sections = []
        paragraphs = text.split("\n\n")
        current_section = ""
        
        for paragraph in paragraphs:
            if len(current_section) + len(paragraph) + 2 <= max_length:
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph
            else:
                if current_section:
                    sections.append(current_section)
                current_section = paragraph
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _format_slack_message(self, blocks: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final Slack message."""
        message = {
            "blocks": blocks,
            "metadata": {
                "processor": "slack_blocks",
                "timestamp": datetime.utcnow().isoformat(),
                "context": {
                    "channel": context.get("channel"),
                    "user": context.get("user"),
                    "message_id": context.get("message_id")
                }
            }
        }
        
        # Add fallback text for notifications
        if blocks:
            first_text_block = next(
                (b for b in blocks if b.get("type") in ["section", "header"]),
                None
            )
            if first_text_block:
                if first_text_block.get("type") == "section":
                    message["text"] = first_text_block.get("text", {}).get("text", "Message")[:150]
                elif first_text_block.get("type") == "header":
                    message["text"] = first_text_block.get("text", {}).get("text", "Message")[:150]
        
        return message
    
    def _create_fallback_message(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback message when processing fails."""
        return {
            "text": str(content)[:150],
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": str(content)
                    }
                }
            ],
            "metadata": {
                "processor": "slack_blocks",
                "fallback": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def get_supported_templates(self) -> List[str]:
        """Get list of supported template types."""
        return list(self.template_registry.keys())
    
    def register_template(self, name: str, template_func: callable):
        """Register a custom template."""
        self.template_registry[name] = template_func
        logger.info(f"Registered custom template: {name}")