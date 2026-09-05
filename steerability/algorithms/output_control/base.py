"""Output control base classes.

Output controls participate in decoding through two mechanisms:

- Logits-processor composition: each control's `get_logits_processors()` results are gathered in
  pipeline `controls` list order and composed by `LogitsProcessorList`.
- Stopping-criteria composition: each control's `get_stopping_criteria()` results are gathered the
  same way; generation stops when any criterion fires.
- The decode loop is exclusive: it is implemented by exactly one `DecodingDriver`, which receives
  the composed stacks as explicit parameters and must apply them at every scoring step of every
  forward pass it issues.

Examples of output controls:

- Reward-augmented decoding (a step-level control)
- Self-disciplined autoregressive sampling (a step-level control)
- Decoding-time alignment / lookahead search (a decoding driver)
- Phase splicing / thinking intervention (a decoding driver)

See Also:

- `steerability.algorithms.output_control`: Implementations of output control methods
- `steerability.algorithms.output_control.common`: Shared component library
- `steerability.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Type

import torch
from transformers import LogitsProcessorList, PreTrainedModel, StoppingCriteriaList

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl
from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.core.execution.session_utils import session_generate

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.payloads import ConstraintSource, ProcessorSpec


def stack_generate_kwargs(logits_processors, stopping_criteria) -> dict:
    """Build the `model.generate` kwargs for the composed stacks, each included only when non-empty.

    Shared by every driver that delegates to `model.generate` (the default driver, and the segment
    and phase drivers per rollout or per phase) so the "pass the stack only when non-empty" rule
    lives in one place.
    """
    extra: dict = {}
    if logits_processors is not None and len(logits_processors):
        extra["logits_processor"] = logits_processors
    if stopping_criteria is not None and len(stopping_criteria):
        extra["stopping_criteria"] = stopping_criteria
    return extra


def resolve_generate_callable(
    model: PreTrainedModel | None, runtime_kwargs: dict | None, session: Any = None
) -> Callable[..., torch.Tensor]:
    """Resolve the generate callable a driver rolls out with.

    Drivers generate through the pipeline's session (a `SteeredSession` carrying this
    generation's control entries), so a driver runs on any backend whose session serves its
    rollout parameters.

    Args:
        model: The pipeline model, or None on backends without a live model; unused.
        runtime_kwargs: Per-call parameters; unused.
        session: The `SteeringSession` for this generation.

    Returns:
        A callable with the `model.generate` calling convention returning full sequences.

    Raises:
        ValueError: If no session was provided.
    """
    if session is None:
        raise ValueError("No generate callable available: the driver received no session.")

    def _generate(input_ids, attention_mask=None, **gen_kwargs):
        return session_generate(session, input_ids, attention_mask, **gen_kwargs)

    return _generate


class OutputControl(BaseControl):
    """Base class for output-control steering methods.

    An `OutputControl` participates in decoding through the composable mechanisms above.
    Controls that implement a decoding procedure subclass `DecodingDriver` instead.

    Class attributes:
        include_in_scoring: Whether this control's logits processors also apply during
            `SteeringPipeline.compute_logprobs()` (per-position, teacher-forced). Defaults
            to True. Set False when the processors are too expensive to evaluate per reference
            position (see `BaseCandidateValue.scoring_cost`).
        same_model_forwards: Whether this component issues additional forward passes through the
            pipeline's own model during decoding. Such passes must be wrapped in
            `auxiliary_pass()` (see `steerability.algorithms.core.utils.auxiliary_pass`), which
            keeps them out of state-control condition scoring, gate updates, and fallback
            position counting. Defaults to False; the flag is declarative metadata and is not
            read by the pipeline.
    """

    Args: Type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False
    include_in_scoring: bool = True
    same_model_forwards: bool = False

    def get_logits_processors(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> list:
        """The control's logits processors for the current generation.

        Called once per `generate()` / `compute_logprobs()` call, after input and state
        controls have prepared the prompt (mirrors `StateControl.get_hooks`). `**kwargs`
        carries `attention_mask` and the caller's generation kwargs. Returned objects
        follow the HF `LogitsProcessor` convention; in-list order is preserved by the composition.

        A processor must behave as a function of `(prefix_ids, scores)`. Internal state is
        permitted only as memoization keyed on the prefix and must re-derive on a prefix mismatch,
        since drivers may restart, rewind, or reorder sequences, and scoring replays prefixes
        teacher-forced (subclass `common.processors.base.PrefixKeyedProcessor` to satisfy this
        mechanically). Return fresh processor instances from this hook; it is invoked once per call
        precisely so that per-generation state is isolated.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            A list of HF `LogitsProcessor`-style objects.
        """
        return []

    def get_stopping_criteria(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> list:
        """The control's stopping criteria.

        Not applied during scoring (there is no loop to stop). Same call convention as
        `get_logits_processors`.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            A list of HF `StoppingCriteria`-style objects.
        """
        return []

    def export_generation_params(self, runtime_kwargs: dict | None = None) -> Mapping[str, Any] | None:
        """The control's sampling-expressible contribution, or None.

        A control whose behavior is expressible as normalized generation parameters returns a
        mapping over a subset of `stop_strings`, `stop_token_ids`, `max_new_tokens`, and
        `min_new_tokens`; the pipeline merges it into the call's `GenerationParams` (stop rules
        union with the caller's; token bounds only tighten) and does not additionally collect
        the control's live processors and criteria for that call, so the control executes on
        every backend through the session's composed stop rules. The default returns None, which
        keeps the control on the live processor/criteria mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The parameter contribution, or None.
        """
        return None

    def export_processor_spec(self, runtime_kwargs: dict | None = None) -> ProcessorSpec | None:
        """The control's engine-hosted processor form, or None.

        A control whose per-step logit math is expressible in an engine's served processor
        vocabulary returns a `ProcessorSpec`; on a backend advertising
        `Capability.PER_STEP_LOGIT_SPECS` with the spec's kind, the pipeline submits it as a
        `ProcessorSpecEntry` in place of the control's live processor. The default returns
        None, which keeps the control on the live processor mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The processor spec, or None.
        """
        return None

    def export_constraint(self, runtime_kwargs: dict | None = None) -> ConstraintSource | None:
        """The control's declarative constrained-decoding source, or None.

        A control whose per-step masking compiles from a declarative source returns a
        `ConstraintSource`; on a backend advertising `Capability.GUIDED_DECODING` the pipeline
        renders it onto the engine's native structured-output parameters in place of the
        control's live processor. The default returns None, which keeps the control on the live
        processor mechanism.

        Args:
            runtime_kwargs: Per-call parameters supplied to `generate()`.

        Returns:
            The constraint source, or None.
        """
        return None

    def steer(self, model: PreTrainedModel, tokenizer=None, session=None, **kwargs) -> None:
        """Optional one-time preparation (e.g., load a reward model, fit a probe).

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The default requires `Capability.IN_PROCESS_TORCH` at generate and, when
        `include_in_scoring` is True, at score as well, since remote prompt-logprob computation
        applies neither live processors nor engine-registered sampling processors to prefill
        logits. Setting `include_in_scoring=False` removes the score-phase requirement.

        Returns:
            The control's phase-keyed requirements.
        """
        score = needs(Capability.IN_PROCESS_TORCH) if self.include_in_scoring else ()
        return Requirements(generate=needs(Capability.IN_PROCESS_TORCH), score=score)


class DecodingDriver(OutputControl):
    """An output control that implements the decoding procedure.

    Exactly one enabled driver may exist per pipeline (the decode loop does not compose).
    Driver contract: `logits_processors` and `stopping_criteria` are the composed,
    authoritative stacks for this generation; the driver applies them at every scoring
    step of every forward pass it issues. Delegating to `model.generate(...,
    logits_processor=..., stopping_criteria=...)` satisfies the contract; hand-rolled
    loops apply them explicitly.

    A driver is also an `OutputControl`: it may additionally contribute processors or
    criteria of its own via the `get_*` hooks, which the pipeline composes like any other
    control's.

    The pipeline passes `session=`, the `SteeringSession` for this generation. Drivers issue
    their rollouts through it (`resolve_generate_callable` returns the right callable), so a
    driver runs on any backend whose session serves its rollout parameters; `model` is None on
    backends without a live model.
    """

    def max_rollouts_per_query(self) -> int | None:
        """An upper bound on the number of continuations this driver generates per input row.

        Counts every sequence the driver requests through the session for one row of one
        `decode()` call, including proposals it discards. A session call over a frontier of `F`
        rows with `num_return_sequences=n` counts `F * n`. Returns None when the configuration
        admits no static bound.

        Callers use the bound to budget or refuse a configuration before executing it. The
        default returns None.

        Returns:
            The per-row rollout bound, or None when no static bound applies.
        """
        return None

    @abstractmethod
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: PreTrainedModel | None,
        logits_processors: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        runtime_kwargs: dict | None,
        session=None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the decoding procedure; return full sequence ids (prompt + continuation)."""
