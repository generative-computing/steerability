"""Helpers for model access through a session at steer and generate time.

`session_generate` runs one generation call through a `SteeringSession` with the `model.generate`
calling convention, and `session_score` runs one teacher-forced scoring call. Components written
against these helpers run on any backend. `session_generate_items` runs one generation call over
unpadded rows and returns each row's continuation and finish reason. `ScopedSession` enforces a
control's declared `ModelAccess` during its steer step. `SessionLM` wraps a session in an object with
`generate` and `device` for helpers that expect a model.
"""
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import UnsupportedOperationError
from steerability.algorithms.core.execution.params import GenerationParams
from steerability.algorithms.core.execution.payloads import GenerationItem, PreparedPrompt, ScoringItem

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession


def session_generate(
    session: "SteeringSession",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **gen_kwargs,
) -> torch.Tensor:
    """Run one generate call through a `SteeringSession`, returning full sequences.

    Drop-in replacement for `model.generate(input_ids=..., attention_mask=..., **gen_kwargs)`
    inside driver rollouts and steer-time helpers. Each row of `input_ids` becomes one
    `GenerationItem`; the keyword arguments normalize through
    `GenerationParams.from_gen_kwargs`, so live `logits_processor` and `stopping_criteria`
    stacks travel in `extra` (consumable in process only). The returned tensor holds each
    caller row followed by its continuation per candidate, right-padded to a common length
    with the session tokenizer's pad token, so slicing at the input length recovers the
    continuations on every backend.

    Args:
        session: The `SteeringSession` to generate on.
        input_ids: Prompt token ids of shape `[batch, seq_len]`.
        attention_mask: Attention mask matching `input_ids`, or None.

    Returns:
        Full sequences of shape `[batch * n, seq_len + gen_len]`.
    """
    params = GenerationParams.from_gen_kwargs(**gen_kwargs)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    items = []
    for row in range(input_ids.size(0)):
        mask_row = attention_mask[row:row + 1] if attention_mask is not None else None
        items.append(GenerationItem(
            prompt=PreparedPrompt.from_token_ids(input_ids[row:row + 1], mask_row),
        ))
    results = session.generate(items, params)

    tokenizer = getattr(session, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None) or 0

    full_rows: list[torch.Tensor] = []
    for row, result in enumerate(results):
        prompt_ids = input_ids[row:row + 1]
        out_ids = result.output.output_ids.to(prompt_ids.device)
        repeated = prompt_ids.expand(out_ids.size(0), -1)
        full_rows.append(torch.cat([repeated, out_ids], dim=1))
    max_len = max(row.size(1) for row in full_rows)
    padded = [
        torch.nn.functional.pad(row, (0, max_len - row.size(1)), value=pad_token_id)
        for row in full_rows
    ]
    return torch.cat(padded, dim=0)


def session_generate_items(
    session: "SteeringSession",
    rows: list[torch.Tensor],
    eos_token_ids: tuple[int, ...] = (),
    **gen_kwargs,
) -> list[tuple[torch.Tensor, str | None]]:
    """Run one generation call through a `SteeringSession` and return each row's continuation.

    Each 1-D row becomes one `GenerationItem` whose attention mask covers every token. The call
    requests one candidate per item and ignores `num_return_sequences`. The keyword arguments are
    converted with `GenerationParams.from_gen_kwargs`, which passes in-memory `logits_processor` and
    `stopping_criteria` objects through in `extra`. Only the in-process backend accepts them.

    Trailing pad tokens are removed from each continuation, since a session that batches items
    right-pads their continuations to a common length. When the pad token is also an eos token (the
    tokenizer's eos or one of `eos_token_ids`), one trailing pad token is kept as the eos the model
    emitted. The pad token is kept only when the item finished on eos or reported no reason, and when
    the last remaining token is not already a terminal token (an eos id or one of the call's stop
    token ids).

    Args:
        session: The `SteeringSession` to generate on.
        rows: Unpadded prompt token ids, one 1-D tensor per item.
        eos_token_ids: Additional eos ids to treat as terminal when removing trailing pad tokens (e.g.,
            the ids on the model's generation config). They do not change when generation stops.
        **gen_kwargs: Generation keyword arguments for the call, in `model.generate` vocabulary.

    Returns:
        One `(continuation_ids, finish_reason)` pair per row, in row order. `continuation_ids` is a 1-D
        tensor, and `finish_reason` is the reason the session reports for the item's candidate, or None
        when it reports none.
    """
    gen_kwargs.pop("num_return_sequences", None)
    params = replace(GenerationParams.from_gen_kwargs(**gen_kwargs), n=1)
    items = [
        GenerationItem(prompt=PreparedPrompt.from_token_ids(
            row.reshape(1, -1), torch.ones(1, row.numel(), dtype=torch.long, device=row.device),
        ))
        for row in rows
    ]
    results = session.generate(items, params)

    tokenizer = getattr(session, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else {eos_token_id}
    eos_ids |= {int(token_id) for token_id in eos_token_ids}
    pad_is_eos = pad_token_id is not None and pad_token_id in eos_ids
    terminal_ids = eos_ids | {int(token_id) for token_id in params.stop_token_ids}

    continuations: list[tuple[torch.Tensor, str | None]] = []
    for result in results:
        output = result.output
        ids = output.output_ids[0]
        reason = output.finish_reasons[0]
        end = ids.numel()
        if pad_token_id is not None:
            real_positions = (ids != pad_token_id).nonzero()
            end = int(real_positions[-1]) + 1 if real_positions.numel() else 0
            # the first stripped pad is the emitted eos when pad and eos share an id; a session that
            # knows only the tokenizer's eos reports no reason for a stop on another eos id
            ends_on_terminal = end > 0 and int(ids[end - 1]) in terminal_ids
            if end < ids.numel() and reason in ("eos", None) and pad_is_eos and not ends_on_terminal:
                end += 1
        continuations.append((ids[:end], reason))
    return continuations


def session_score(
    session: "SteeringSession",
    input_ids: torch.Tensor,
    ref_output_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **forward_kwargs,
) -> torch.Tensor:
    """Score reference tokens through a `SteeringSession`, teacher-forced.

    Each row of `input_ids` becomes one `ScoringItem`; a single reference row broadcasts
    across the batch. Keyword arguments travel as forward keyword arguments.

    Args:
        session: The `SteeringSession` to score on.
        input_ids: Prompt token ids of shape `[batch, seq_len]` (a 1-D tensor is one row).
        ref_output_ids: Reference tokens of shape `[ref_len]`, `[1, ref_len]`, or
            `[batch, ref_len]`.
        attention_mask: Attention mask matching `input_ids`, or None.

    Returns:
        Log probabilities of shape `[batch, ref_len]`.
    """
    params = GenerationParams(extra=forward_kwargs)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if ref_output_ids.dim() == 1:
        ref_output_ids = ref_output_ids.unsqueeze(0)
    if ref_output_ids.size(0) == 1 and input_ids.size(0) > 1:
        ref_output_ids = ref_output_ids.expand(input_ids.size(0), -1)
    items = []
    for row in range(input_ids.size(0)):
        mask_row = attention_mask[row:row + 1] if attention_mask is not None else None
        items.append(ScoringItem(
            prompt=PreparedPrompt.from_token_ids(input_ids[row:row + 1], mask_row),
            ref_output_ids=ref_output_ids[row:row + 1],
        ))
    return session.score(items, params)


class ScopedSession:
    """A `SteeringSession` view scoped to one control's declared steer access.

    The pipeline hands each control's `steer()` a scoped session over the venue session.
    `layout` and `tokenizer` are always available; `generate` and `score` delegate at
    `ModelAccess.ROLLOUTS` and above; `capture` delegates at `ModelAccess.CAPTURE` and above.
    Calls below the declared rung raise, so undeclared steer-time model contact fails
    immediately and attributably on every backend. The wrapper exposes no `model` attribute at
    any rung; the live module travels only through the `model=` argument of `steer()`.

    Attributes:
        inner: The wrapped venue session.
    """

    def __init__(self, inner, control_name: str, access: ModelAccess) -> None:
        self.inner = inner
        self._control_name = control_name
        self._access = access

    @property
    def layout(self):
        """Structural facts about the venue session's model."""
        return self.inner.layout

    @property
    def tokenizer(self):
        """The venue session's tokenizer, or None."""
        return getattr(self.inner, "tokenizer", None)

    @property
    def in_process(self) -> bool:
        """True when the venue session serves a live in-process model, so `layout` facts such
        as `model_fingerprint` are weights-grade rather than config-grade. Venue-matched
        identity checks dispatch on this."""
        return hasattr(type(self.inner), "model")

    def _require_rollouts(self) -> None:
        if self._access < ModelAccess.ROLLOUTS:
            raise UnsupportedOperationError(
                f"{self._control_name} declared steer access '{self._access.name.lower()}', "
                "which does not include session generation; declare ModelAccess.ROLLOUTS or "
                "higher."
            )

    def generate(self, items, params):
        """Generate through the venue session; requires `ModelAccess.ROLLOUTS` or higher."""
        self._require_rollouts()
        return self.inner.generate(items, params)

    def score(self, items, params):
        """Score through the venue session; requires `ModelAccess.ROLLOUTS` or higher."""
        self._require_rollouts()
        return self.inner.score(items, params)

    def capture(self, prompts, layers, mode, location="layer_output"):
        """Capture through the venue session; requires `ModelAccess.CAPTURE` or higher."""
        if self._access < ModelAccess.CAPTURE:
            raise UnsupportedOperationError(
                f"{self._control_name} declared steer access '{self._access.name.lower()}', "
                "which does not include hidden-state capture; declare ModelAccess.CAPTURE or "
                "higher."
            )
        return self.inner.capture(prompts, layers, mode, location=location)


class SessionLM:
    """A model-shaped adapter whose generation executes through a `SteeringSession`.

    Gives steer-time helpers written against the `model.generate` calling convention
    (proposers, rollout scorers) an object with `generate` and `device`, so the helper runs on
    any backend. `pad_token_id` keyword arguments are dropped before submission, since
    sessions derive padding from their tokenizer.

    Attributes:
        session: The wrapped session.
    """

    def __init__(self, session) -> None:
        self.session = session

    @property
    def device(self) -> torch.device:
        """CPU; sessions place prompt tensors themselves."""
        return torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, **gen_kwargs) -> torch.Tensor:
        """Generate full sequences through the session (`model.generate` convention)."""
        gen_kwargs.pop("pad_token_id", None)
        return session_generate(self.session, input_ids, attention_mask, **gen_kwargs)
