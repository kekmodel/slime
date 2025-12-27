"""
Prompt formatter for ChatML-style models.
"""

import re
from typing import Any

from jinja2 import Template


class ChatMLFormatter:
    """
    ChatML format formatter.

    Compatible with: Qwen, Yi, and other ChatML-based models.
    """

    TEMPLATE = """<|im_start|>system
{%- if system_prompt %}
{{ system_prompt }}
{%- else %}
You are a helpful customer service agent.
{%- endif %}
{%- if tools %}

# Available Tools

You can use the following tools to help customers:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

To use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

After receiving tool results, continue assisting the customer.
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message.role == 'user' %}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == 'assistant' %}
<|im_start|>assistant
{{ message.content }}<|im_end|}
{%- elif message.role == 'tool' %}
<|im_start|>tool
{{ message.content }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

    def __init__(self):
        self._template = Template(self.TEMPLATE)

    def format(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> str:
        """Format messages and tools into a prompt string."""
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", system_prompt)
            messages = messages[1:]

        return self._template.render(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )

    def postprocess_response(self, text: str) -> str:
        """Ensure response ends at complete tool_call tag."""
        if "<tool_call>" in text:
            pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                return text[: matches[-1].end()]
        return text

    def format_tool_result(self, name: str, output: str) -> str:
        """Format tool execution result."""
        return f'\n<tool_result name="{name}">\n{output}\n</tool_result>\n'


def get_formatter(model_type: str = "chatml") -> ChatMLFormatter:
    """Get formatter instance. Currently only ChatML is supported."""
    return ChatMLFormatter()
