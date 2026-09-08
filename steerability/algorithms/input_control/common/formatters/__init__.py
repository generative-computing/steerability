"""Formatters render Memory content into adapted prompts (token-level or message-level)."""
from steerability.algorithms.input_control.common.formatters.base import BaseFormatter
from steerability.algorithms.input_control.common.formatters.chat_template_slot import ChatTemplateSlotFormatter
from steerability.algorithms.input_control.common.formatters.few_shot_block import FewShotBlockFormatter
from steerability.algorithms.input_control.common.formatters.prepend_text import PrependTextFormatter
from steerability.algorithms.input_control.common.formatters.system_prompt import SystemPromptFormatter

__all__ = [
    "BaseFormatter",
    "ChatTemplateSlotFormatter",
    "FewShotBlockFormatter",
    "PrependTextFormatter",
    "SystemPromptFormatter",
]
