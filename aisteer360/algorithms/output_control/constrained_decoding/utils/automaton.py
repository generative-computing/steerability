"""Client-side automaton compilation for declarative constraints, over xgrammar.

Json-schema constraints compile compact (`any_whitespace=False`) so the grammar matches the
whitespace policy applied on every venue.
"""
from __future__ import annotations

import json
import re

import torch

from aisteer360.algorithms.core.execution.payloads import ConstraintSource
from aisteer360.utils.optional import require


class XGrammarAutomaton:
    """A `ConstraintAutomaton` over an xgrammar `GrammarMatcher`.

    `reset` starts a fresh matcher at the prompt boundary (prompt tokens are not part of the
    constrained output); `allowed` feeds any newly generated tokens since the last call and
    returns the token ids the grammar permits next. Once the grammar terminates, only the
    tokenizer's stop tokens are permitted.

    Args:
        compiled: The compiled xgrammar grammar.
        vocab_size: The tokenizer's full vocabulary size.
        stop_token_ids: Token ids permitted after grammar termination.
    """

    def __init__(self, compiled, vocab_size: int, stop_token_ids: list[int]):
        xgrammar = require("xgrammar")
        self._xgrammar = xgrammar
        self._compiled = compiled
        self._vocab_size = vocab_size
        self._stop_token_ids = list(stop_token_ids)
        self._bitmask = xgrammar.allocate_token_bitmask(1, vocab_size)
        self._matcher = None
        self._consumed = 0

    def reset(self, prefix_ids: torch.Tensor) -> None:
        """Start a fresh matcher; `prefix_ids` is the prompt the constraint begins after."""
        self._matcher = self._xgrammar.GrammarMatcher(self._compiled)
        self._consumed = prefix_ids.size(-1)

    def allowed(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Token ids the grammar permits at the current step."""
        row = prefix_ids[0] if prefix_ids.ndim == 2 else prefix_ids
        for token_id in row[self._consumed:].tolist():
            self._matcher.accept_token(int(token_id))
        self._consumed = row.size(-1)
        if self._matcher.is_terminated():
            return torch.tensor(self._stop_token_ids, dtype=torch.long)
        self._xgrammar.reset_token_bitmask(self._bitmask)
        self._matcher.fill_next_token_bitmask(self._bitmask)
        mask_row = self._bitmask[0]
        bits = ((mask_row.unsqueeze(1) >> torch.arange(32)) & 1).to(torch.bool)
        return torch.nonzero(bits.reshape(-1)[: self._vocab_size], as_tuple=True)[0]


def compile_constraint_automaton(source: ConstraintSource, tokenizer) -> XGrammarAutomaton:
    """Compile a declarative constraint into a client-side automaton.

    Args:
        source: The declarative constraint.
        tokenizer: The tokenizer the automaton masks against.

    Returns:
        The compiled automaton.

    Raises:
        ModuleNotFoundError: If `xgrammar` is not installed. The message names the
            `aisteer360[guided]` extra.
    """
    xgrammar = require("xgrammar")
    vocab_size = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0) or 0)
    tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    if source.kind == "json_schema":
        schema = source.value if isinstance(source.value, str) else json.dumps(dict(source.value))
        # compile compact on every venue rather than inheriting each backend's whitespace default
        compiled = compiler.compile_json_schema(schema, any_whitespace=False)
    elif source.kind == "regex":
        compiled = compiler.compile_regex(source.value)
    elif source.kind == "grammar":
        compiled = compiler.compile_grammar(source.value)
    else:
        pattern = "(" + "|".join(re.escape(candidate) for candidate in source.value) + ")"
        compiled = compiler.compile_regex(pattern)
    stop_token_ids = list(tokenizer_info.stop_token_ids)
    if not stop_token_ids and tokenizer.eos_token_id is not None:
        stop_token_ids = [tokenizer.eos_token_id]
    return XGrammarAutomaton(compiled, vocab_size, stop_token_ids)
