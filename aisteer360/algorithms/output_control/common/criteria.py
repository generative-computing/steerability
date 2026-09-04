"""Composable stopping criteria.

Plain `StoppingCriteria` usable both as pipeline-composed criteria (via
`OutputControl.get_stopping_criteria`) and inside phased-driver phases. Each returns a per-row
boolean tensor: True where that row should stop.
"""
from __future__ import annotations

import torch
from transformers import StoppingCriteria


class StopOnSubstring(StoppingCriteria):
    """Stop a row once the newly generated text contains `text`.

    Decodes only the tokens generated after `prompt_len` per check (so the substring test runs on
    the continuation, not the prompt). Decoding per step is the cost of a text-level stop condition;
    for token-level stops prefer `StopOnTokens`.

    Args:
        tokenizer: Tokenizer used to decode the continuation.
        text: The substring that triggers stopping.
        prompt_len: Number of prompt tokens to skip when decoding the continuation.
    """

    def __init__(self, tokenizer, text: str, prompt_len: int):
        self._tokenizer = tokenizer
        self._text = text
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        fired = []
        for row in input_ids:
            continuation = self._tokenizer.decode(
                row[self._prompt_len:], skip_special_tokens=False
            )
            fired.append(self._text in continuation)
        return torch.tensor(fired, dtype=torch.bool, device=input_ids.device)


class StopOnTokens(StoppingCriteria):
    """Stop a row once its last token is one of `ids`.

    Args:
        ids: Token ids that trigger stopping.
    """

    def __init__(self, ids):
        self._ids = set(int(i) for i in ids)

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        last = input_ids[:, -1]
        fired = torch.tensor(
            [int(t.item()) in self._ids for t in last],
            dtype=torch.bool,
            device=input_ids.device,
        )
        return fired


class BudgetTokens(StoppingCriteria):
    """Stop a row once it has generated `n` tokens past `prompt_len`.

    Args:
        n: Token budget (new tokens) before stopping.
        prompt_len: Number of prompt tokens (the budget counts tokens beyond this length).
    """

    def __init__(self, n: int, prompt_len: int):
        self._n = n
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        fired = input_ids.size(1) - self._prompt_len >= self._n
        return torch.full((input_ids.size(0),), fired, dtype=torch.bool, device=input_ids.device)
