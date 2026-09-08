from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Requirements
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.criteria import BudgetTokens, StopOnSubstring, StopOnTokens
from steerability.algorithms.output_control.stopping_rules.args import StoppingRulesArgs


class StoppingRules(OutputControl):
    """Stop rules as configuration: substring, token, and budget stops.

    `StoppingRules` is the smallest member of the generic family. It is sampling-expressible:
    the pipeline merges its configuration into the call's normalized generation parameters
    (`export_generation_params`), and the backend session composes the resulting stop rules,
    so the control runs on every backend:

        - `stop_texts=["\\n\\nQ:"]` halts a row once its continuation contains the substring.
        - `stop_token_ids=[13]` halts a row once its last generated token is one of the ids.
        - `budget=64` tightens `max_new_tokens` to at most `budget`.

    Token ids are returned as generated (the stop text plus any token-boundary overrun stays in
    the ids); the pipeline truncates decoded text at the first stop-string occurrence and rows
    halted by these rules report `finish_reason="stop"` (budget stops report `"length"`).
    `get_stopping_criteria` remains available for direct composition outside the pipeline and
    returns fresh criteria anchored at the current prompt length.
    """

    Args = StoppingRulesArgs

    supports_batching: bool = True

    tokenizer: PreTrainedTokenizerBase | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Attach the tokenizer (required whenever `stop_texts` is configured)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.stop_texts and self.tokenizer is None:
            raise RuntimeError("StoppingRules requires a tokenizer when 'stop_texts' is configured.")
        return model

    def requirements(self) -> Requirements:
        """Stop rules are session contract on every backend, so no phase requires anything."""
        return Requirements()

    def export_generation_params(self, runtime_kwargs: dict | None = None) -> Mapping[str, Any]:
        """The configured stops as normalized generation parameters."""
        contribution: dict[str, Any] = {}
        if self.stop_texts:
            contribution["stop_strings"] = tuple(self.stop_texts)
        if self.stop_token_ids:
            contribution["stop_token_ids"] = tuple(self.stop_token_ids)
        if self.budget is not None:
            contribution["max_new_tokens"] = self.budget
        return contribution

    def get_stopping_criteria(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return fresh criteria anchored at the current prompt length."""
        prompt_len = input_ids.size(1)
        criteria = []
        for text in self.stop_texts:
            criteria.append(StopOnSubstring(self.tokenizer, text, prompt_len))
        if self.stop_token_ids:
            criteria.append(StopOnTokens(self.stop_token_ids))
        if self.budget is not None:
            criteria.append(BudgetTokens(self.budget, prompt_len))
        return criteria
