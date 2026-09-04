"""Session-side helpers for steer- and generate-time model access.

`session_generate` and `session_score` run one generation or scoring call through a
`SteeringSession` with the `model.generate` calling convention, so components written against
that convention execute on any backend. `ScopedSession` enforces a control's declared
`ModelAccess` during its steer step, and `SessionLM` adapts a session into a model-shaped
object for helpers that expect one.
"""
import torch

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.execution.contracts import UnsupportedOperationError
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.payloads import GenerationItem, PreparedPrompt, ScoringItem


def session_generate(session, input_ids, attention_mask=None, **gen_kwargs) -> torch.Tensor:
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
        **gen_kwargs: Generation keyword arguments in `model.generate` vocabulary.

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


def session_score(session, input_ids, ref_output_ids, attention_mask=None, **forward_kwargs) -> torch.Tensor:
    """Score reference tokens through a `SteeringSession`, teacher-forced.

    Each row of `input_ids` becomes one `ScoringItem`; a single reference row broadcasts
    across the batch. Keyword arguments travel as forward keyword arguments.

    Args:
        session: The `SteeringSession` to score on.
        input_ids: Prompt token ids of shape `[batch, seq_len]` (a 1-D tensor is one row).
        ref_output_ids: Reference tokens of shape `[ref_len]`, `[1, ref_len]`, or
            `[batch, ref_len]`.
        attention_mask: Attention mask matching `input_ids`, or None.
        **forward_kwargs: Forward keyword arguments.

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
