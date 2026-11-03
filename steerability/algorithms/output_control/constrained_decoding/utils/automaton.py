"""Client-side automaton compilation for declarative constraints, over xgrammar.

Json-schema constraints compile compact (`any_whitespace=False` with `","` and `":"` separators) so the
grammar matches the whitespace policy applied on every venue.
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from typing import Any

import torch
import xgrammar
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.core.execution.payloads import ConstraintSource


@dataclass
class _RowGrammar:
    """The grammar state of one row.

    Attributes:
        matcher: The xgrammar `GrammarMatcher` of the row.
        prompt: The index of the row's prompt among the automaton's known prompts.
        consumed: The row's token ids that the matcher has been advanced over, of shape `[T]`, with the prompt
            included and leading pad tokens removed. Tokens after a rejected token or after the grammar terminates
            are included but are not fed to the matcher.
        rejected: Whether the matcher rejected one of the row's tokens.
    """

    matcher: Any
    prompt: int
    consumed: torch.Tensor
    rejected: bool = False

    @property
    def stopped(self) -> bool:
        """Whether only the stop tokens remain permitted for the row.

        This is the case after the matcher rejects a token or once the grammar has terminated.
        """
        return self.rejected or self.matcher.is_terminated()


def _extends(prefix: torch.Tensor, row: torch.Tensor) -> bool:
    """Return whether `row` begins with `prefix`.

    Args:
        prefix: A 1-D tensor of token ids.
        row: A 1-D tensor of token ids.

    Returns:
        True when `row` is at least as long as `prefix` and its first tokens equal `prefix`.
    """
    length = prefix.numel()
    return length <= row.numel() and torch.equal(row[:length], prefix)


class XGrammarAutomaton:
    """A `ConstraintAutomaton` that tracks a separate xgrammar grammar state for each row of a batch.

    `compile_constraint_automaton()` builds an automaton from a `ConstraintSource`. `fresh()` returns a new automaton
    over the same compiled grammar with no row state, optionally with the prompts of a generation. `allowed()`
    returns the tokens that the grammar permits next in each row.

    The constrained output of a row is the part of the row after its prompt. The prompts start as the rows passed to
    `fresh()`, or as the rows of the first call when none are passed. Leading pad tokens are removed before rows are
    compared, since a decoding driver may pad a row differently between calls. A prompt counts as decoded once a
    row's output has begun at its end in an earlier call. The grammar state of each row in a call is determined as
    follows:

    1. **Continued rows**: a row that begins with a row of the previous call continues that row's output. This covers
       ordinary decode steps, reordered beams, and several candidates that continue one row.
    2. **Replayed rows**: a row that begins with no row of the previous call replays its output from the longest
       prompt it begins with. This applies when that prompt has been decoded and the grammar accepts the row's next
       token after it (e.g., a rewind to a shorter prefix, or a candidate that decodes after another candidate of its
       prompt).
    3. **New prompts**: any other row is treated as a prompt, and its constrained output begins after it (e.g., a
       prompt followed by text inserted before the first generated token).

    A row that begins with a row of the previous call is also treated as a prompt when the longest prompt it begins
    with is longer than that row and has not been decoded (e.g., a later prompt that begins with an earlier prompt
    and its output). In a batched call, every prompt is decoded on the first call. A row whose own output reaches
    another prompt in a batched call therefore continues its output.

    After the grammar rejects one of a row's tokens (e.g., text inserted by a decoding driver after generated tokens),
    or once the grammar has terminated, only the stop tokens are permitted for that row.

    Rows are identified by their token ids alone, which leads to the following behavior when prompts overlap:

    - When the prompts decode one at a time (e.g., under `seed_scope="item"`), a row whose own output reaches a
      longer prompt that has not been decoded yet restarts its grammar at the end of that prompt.
    - Without prompts passed to `fresh()`, a later prompt that begins with an earlier prompt and its output continues
      that output.

    An automaton tracks one generation at a time. `ConstrainedDecoding` calls `fresh()` with the prompts of each
    generation to create the automaton for that generation.

    Args:
        compiled: The compiled xgrammar grammar.
        vocab_size: The size of the tokenizer's full vocabulary.
        stop_token_ids: The token ids permitted after the grammar terminates or rejects a token.
        pad_token_id: The tokenizer's pad token id, which is removed from the start of each row. With None, rows are
            compared as given.
    """

    def __init__(self, compiled: Any, vocab_size: int, stop_token_ids: list[int], pad_token_id: int | None = None):
        self._compiled = compiled
        self._vocab_size = vocab_size
        self._stop_token_ids = list(stop_token_ids)
        self._pad_token_id = pad_token_id
        self._bitmask = xgrammar.allocate_token_bitmask(1, vocab_size)
        self._bit_shifts = torch.arange(32, dtype=torch.int32)
        self._prompts: list[torch.Tensor] = []
        self._started: list[bool] = []  # per prompt, whether a matcher has started at its boundary
        self._prompt_lengths: list[int] = []  # the distinct prompt lengths, ascending
        self._prompts_by_length: dict[int, list[int]] = {}
        self._rows: list[_RowGrammar] = []

    def fresh(self, prompt_ids: torch.Tensor | None = None) -> XGrammarAutomaton:
        """Return an automaton over the same compiled grammar with no row state.

        Args:
            prompt_ids: The prompts of the generation, of shape `[B, T]` or `[T]` for a single prompt. Their rows,
                without leading pad tokens, become the prompts of the new automaton. With None, the rows of the first
                call become the prompts. A later prompt that begins with one of them then starts its own output only
                when the grammar rejects its next token after that prompt.

        Returns:
            The new automaton.
        """
        automaton = XGrammarAutomaton(self._compiled, self._vocab_size, self._stop_token_ids, self._pad_token_id)
        if prompt_ids is not None:
            for row in automaton._unpadded_rows(prompt_ids):
                automaton._register(row)
        return automaton

    def reset(self, prefix_ids: torch.Tensor) -> None:
        """Update the grammar state of each row to the tokens of `prefix_ids`.

        Each row continues, replays, or starts its constrained output as described in the class docstring.

        Args:
            prefix_ids: The prefix token ids of shape `[B, T]`, or `[T]` for a single row.
        """
        self._sync(prefix_ids)

    def allowed(self, prefix_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return the token ids that the grammar permits next in each row of `prefix_ids`.

        The grammar state of each row is first updated to the tokens of `prefix_ids`. A row whose grammar has
        terminated or has rejected one of its tokens is permitted only the stop tokens.

        Args:
            prefix_ids: The prefix token ids of shape `[B, T]`, or `[T]` for a single row.

        Returns:
            A list with one 1-D `LongTensor` of permitted token ids per row, on the CPU.
        """
        self._sync(prefix_ids)
        batch_size = len(self._rows)
        if self._bitmask.size(0) != batch_size:
            self._bitmask = xgrammar.allocate_token_bitmask(batch_size, self._vocab_size)
        xgrammar.reset_token_bitmask(self._bitmask)
        stopped = [state.stopped for state in self._rows]
        for index, state in enumerate(self._rows):
            if not stopped[index]:
                state.matcher.fill_next_token_bitmask(self._bitmask, index)
        bits = ((self._bitmask.unsqueeze(-1) >> self._bit_shifts) & 1).to(torch.bool)
        bits = bits.reshape(batch_size, -1)[:, : self._vocab_size]
        stop_ids = torch.tensor(self._stop_token_ids, dtype=torch.long)
        return [
            stop_ids if stopped[index] else torch.nonzero(bits[index], as_tuple=True)[0]
            for index in range(batch_size)
        ]

    def _sync(self, prefix_ids: torch.Tensor) -> None:
        """Assign each row of `prefix_ids` a matcher that has consumed exactly that row.

        Before any prompt is known, the rows of the call are registered as prompts. Each row is then matched to the
        matchers of the previous call whose consumed ids it extends. The row prefers its own matcher (the one at the
        same index) when that matcher is unclaimed, and otherwise takes the unclaimed matcher with the longest
        consumed ids. A row whose matching matchers are all claimed gets a new matcher at the prompt of the longest
        one. When the longest known prompt that the row extends is longer than the consumed ids of the selected
        matcher and no earlier call has started a matcher at it, the row is registered as a prompt and gets a new
        matcher at its end instead. A row that extends no matcher gets a new matcher at the longest known prompt it
        extends, when that prompt has started. Otherwise, or when the grammar rejects the row's next token after
        that prompt, the row is registered as a prompt and gets a new matcher at its end. The prompts at which this
        call starts a matcher are marked as started after every row is assigned.

        Args:
            prefix_ids: The prefix token ids of shape `[B, T]`, or `[T]` for a single row.
        """
        rows = self._unpadded_rows(prefix_ids)
        if not self._prompts:
            for row in rows:
                self._register(row)
        previous = self._rows
        previous_ids = [state.consumed for state in previous]
        lengths = [ids.numel() for ids in previous_ids]
        claimed: set[int] = set()
        started: set[int] = set()  # the prompts a matcher starts at in this call
        states: list[_RowGrammar] = []
        for index, row in enumerate(rows):
            if index < len(previous) and index not in claimed and _extends(previous_ids[index], row):
                extended = [index]
            else:
                # the matchers whose ids the row extends, longest first
                extended = sorted(
                    (other for other, ids in enumerate(previous_ids) if _extends(ids, row)),
                    key=lambda other: -lengths[other],
                )
            source = next((other for other in extended if other not in claimed), None)
            reference = source if source is not None else next(iter(extended), None)
            state = None
            if reference is not None:
                prompt = self._longest_prompt(row, longer_than=lengths[reference])
                if prompt is not None and not self._started[prompt]:
                    # a longer prompt that no earlier call has started a matcher at begins the row's output
                    state = self._start(self._register(row), row)
                elif source is not None:
                    claimed.add(source)
                    state = previous[source]
                else:
                    # a row that extends a claimed matcher's ids continues that matcher's prompt
                    state = self._start(previous[reference].prompt, row)
            else:
                prompt = self._longest_prompt(row)
                if prompt is not None and self._started[prompt]:
                    state = self._start(prompt, row)
                if state is None or state.rejected:
                    # a row whose next token after a started prompt the grammar rejects is a new prompt
                    state = self._start(self._register(row), row)
            if source is None or state is not previous[source]:
                started.add(state.prompt)
            self._advance(state, row)
            states.append(state)
        for prompt in started:
            self._started[prompt] = True
        self._rows = states

    def _unpadded_rows(self, prefix_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return the rows of `prefix_ids` on the CPU, each without its leading pad tokens.

        Args:
            prefix_ids: Token ids of shape `[B, T]`, or `[T]` for a single row.

        Returns:
            A list of 1-D tensors, one per row, copied from `prefix_ids`.
        """
        rows = prefix_ids.detach().to("cpu", copy=True)
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        return [self._strip_leading_pads(row) for row in rows]

    def _strip_leading_pads(self, row: torch.Tensor) -> torch.Tensor:
        """Return `row` without its leading pad tokens.

        Args:
            row: A 1-D tensor of token ids.

        Returns:
            The part of `row` from its first non-pad token on, or `row` unchanged when it is entirely pad or no pad
            token id is set.
        """
        if self._pad_token_id is None:
            return row
        real = torch.nonzero(row != self._pad_token_id, as_tuple=True)[0]
        return row[int(real[0]):] if real.numel() else row

    def _register(self, row: torch.Tensor) -> int:
        """Return the index of the known prompt equal to `row`, registering `row` when there is none.

        A newly registered prompt is stored as a copy and is not yet started.

        Args:
            row: A 1-D tensor of token ids without leading pad tokens.

        Returns:
            The index of the prompt among the known prompts.
        """
        length = row.numel()
        same_length = self._prompts_by_length.get(length)
        if same_length is None:
            bisect.insort(self._prompt_lengths, length)
            same_length = self._prompts_by_length[length] = []
        for index in same_length:
            if torch.equal(self._prompts[index], row):
                return index
        self._prompts.append(row.clone())
        self._started.append(False)
        same_length.append(len(self._prompts) - 1)
        return len(self._prompts) - 1

    def _longest_prompt(self, row: torch.Tensor, longer_than: int = -1) -> int | None:
        """Return the index of the longest known prompt that `row` begins with and that is longer than `longer_than`.

        Args:
            row: A 1-D tensor of token ids without leading pad tokens.
            longer_than: The length that the prompt must exceed. The default of -1 admits every prompt.

        Returns:
            The index of the prompt, or None when no known prompt qualifies.
        """
        low = bisect.bisect_right(self._prompt_lengths, longer_than)
        high = bisect.bisect_right(self._prompt_lengths, row.numel())
        for length in reversed(self._prompt_lengths[low:high]):
            for index in self._prompts_by_length[length]:
                if torch.equal(row[:length], self._prompts[index]):
                    return index
        return None

    def _start(self, prompt: int, row: torch.Tensor) -> _RowGrammar:
        """Return a new row state whose matcher begins at the end of `prompt`.

        The matcher is fed the first token of `row` after the prompt, when there is one.

        Args:
            prompt: The index of the known prompt.
            row: A 1-D tensor of token ids that begins with the prompt.

        Returns:
            The new `_RowGrammar`, with `rejected` set when the grammar rejects that token.
        """
        boundary = self._prompts[prompt].numel()
        state = _RowGrammar(matcher=xgrammar.GrammarMatcher(self._compiled), prompt=prompt, consumed=row[:boundary])
        self._advance(state, row[: boundary + 1])
        return state

    def _advance(self, state: _RowGrammar, row: torch.Tensor) -> None:
        """Feed the matcher the tokens of `row` after the consumed ids of `state`, and record `row` as consumed.

        Feeding stops once `state` is stopped, i.e., after a rejected token or once the grammar has terminated. A
        rejected token sets `rejected`.

        Args:
            state: The row state to advance.
            row: A 1-D tensor of token ids that begins with the consumed ids of `state`.
        """
        for token_id in row[state.consumed.numel():].tolist():
            if state.stopped:
                break
            if not state.matcher.accept_token(token_id):
                state.rejected = True
        state.consumed = row


def compile_constraint_automaton(source: ConstraintSource, tokenizer: PreTrainedTokenizerBase) -> XGrammarAutomaton:
    """Compile a declarative constraint into a client-side automaton.

    Args:
        source: The declarative constraint.
        tokenizer: The tokenizer the automaton masks against.

    Returns:
        The compiled automaton.
    """
    vocab_size = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0) or 0)
    tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    if source.kind == "json_schema":
        schema = source.value if isinstance(source.value, str) else json.dumps(dict(source.value))
        # compile compact on every venue rather than inheriting each backend's whitespace default
        compiled = compiler.compile_json_schema(schema, any_whitespace=False, separators=(",", ":"))
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
    return XGrammarAutomaton(compiled, vocab_size, stop_token_ids, pad_token_id=tokenizer.pad_token_id)
