"""Per-call payload assembly for `SteeringPipeline`.

Builds the entry payloads a session executes: state-control entries (per-generation hooks in
process, lowered intervention specs on spec-consuming backends), output-control entries
(processor and criteria stacks, lowered generation parameters, declarative constraints,
engine-hosted processor specs), and the scoring-time processor application. Functions receive
the pipeline's control lists and per-call tensors explicitly and hold no pipeline state.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Sequence

import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from steerability.algorithms.core.execution.contracts import BackendCapabilities, Capability, UnsupportedOperationError
from steerability.algorithms.core.execution.payloads import (
    ConstraintSource,
    HookEntry,
    InterventionEntry,
    ProcessorSpecEntry,
    StackEntry,
    StateControlEntry,
    remap_prompt_relative_scopes,
)
from steerability.algorithms.output_control.base import DecodingDriver, OutputControl
from steerability.algorithms.state_control.base import StateControl

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from steerability.algorithms.core.execution.backend import Backend
    from steerability.algorithms.core.execution.contracts import InterventionKinds

logger = logging.getLogger(__name__)


def collect_state_entries(
        state_controls: Sequence[StateControl],
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        *,
        attention_mask: torch.Tensor | None = None,
        hooks_in_process: bool,
        lowered_state: dict[int, InterventionEntry],
        backend: "Backend | None" = None,
        intervention_kinds: "InterventionKinds | None" = None,
        model: "PreTrainedModel | None" = None,
        **gen_kwargs,
) -> tuple[StateControlEntry, ...]:
    """Collect every enabled state control's entries for the current logical generation.

    With `hooks_in_process` True, hooks are per-generation artifacts built here: they close
    over the prompt anchor, sized gate state, and a fresh position clock, and travel only as
    `HookEntry` contributions (the session that executes forwards owns registration).
    Otherwise entries come from the steer-time lowering cache, filled lazily for a control
    enabled after `steer()`.

    Args:
        state_controls: The pipeline's state controls, in list order.
        input_ids: Input token IDs after input control transformation
        runtime_kwargs: Per-call parameters for state controls
        attention_mask: The prompt attention mask matching `input_ids`. Forwarded to
            hook construction so condition scorers see the real (non-pad) prompt tokens
            rather than re-deriving a pad mask by token identity.
        hooks_in_process: Whether the backend executes in-process torch hooks.
        lowered_state: Steer-time lowering cache keyed by `id(control)`, updated in place on
            a lazy fill.
        backend: Inference backend consulted on a lazy fill.
        intervention_kinds: Advertised kinds verified on a lazy fill.
        model: Live model forwarded to hook construction.

    Returns:
        One entry per enabled state control, in controls-list order.
    """
    if not hooks_in_process:
        entries = []
        for state_control in state_controls:
            if not state_control.enabled:
                continue
            entry = lowered_state.get(id(state_control))
            if entry is None:
                served = ((getattr(backend, "_discovery", None) or {}).get("model") or {})
                payloads: dict = {}
                entry = _lower_control(
                    state_control, intervention_kinds, served, payloads,
                )
                backend.stage_artifacts(payloads)
                lowered_state[id(state_control)] = entry
            entries.append(entry)
        return tuple(entries)

    entries = []
    for state_control in state_controls:
        if not state_control.enabled:
            continue
        hooks = state_control.get_hooks(
            input_ids, runtime_kwargs, attention_mask=attention_mask, model=model, **gen_kwargs
        )
        entries.append(HookEntry(hooks=hooks))
    return tuple(entries)


def per_item_state_entries(
        state_controls: Sequence[StateControl],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict | None,
        *,
        model: "PreTrainedModel | None" = None,
        **gen_kwargs,
) -> list[tuple[HookEntry, ...]]:
    """Per-row state entries computed by per-call control clones.

    Distinct per-item derived seeds force the in-process session onto its serial path, where
    each row runs its own forward. Hooks computed once on the batch hold batch-sized position
    and gate state, so each row instead gets hooks computed by a fresh clone on that row's
    prompt tensors.

    Args:
        state_controls: The pipeline's state controls, in list order.
        input_ids: Adapted prompt ids of shape `[batch, seq_len]`.
        attention_mask: Attention mask matching `input_ids`.
        runtime_kwargs: Per-call parameters for state controls.
        model: Live model forwarded to `get_hooks()`.

    Returns:
        One tuple of `HookEntry` per row, each in controls-list order.
    """
    rows: list[tuple[HookEntry, ...]] = []
    for index in range(input_ids.size(0)):
        entries: list[HookEntry] = []
        for state_control in state_controls:
            if not state_control.enabled:
                continue
            clone = state_control.clone_for_call()
            hooks = clone.get_hooks(
                input_ids[index:index + 1],
                runtime_kwargs,
                attention_mask=attention_mask[index:index + 1],
                model=model,
                **gen_kwargs,
            )
            entries.append(HookEntry(hooks=hooks))
        rows.append(tuple(entries))
    return rows


def lower_state_controls(
        state_controls: Sequence[StateControl],
        backend,
        capabilities: BackendCapabilities,
) -> dict[int, InterventionEntry]:
    """Lower every enabled state control's interventions for a spec-consuming backend and
    stage their artifacts, returning the entries keyed by `id(control)`.

    Runs at the end of `steer()` when the backend executes interventions as
    specs rather than in-process hooks. Specs are per-steer artifacts: the worker anchors
    positions per request server-side and the spec is prompt-independent by construction,
    so one lowering serves every subsequent generation. Each spec is verified against the
    backend's negotiated kinds (the intersection of the static tables and discovery), and
    a control's steering-artifact provenance is cross-checked against the served model's
    when the backend carries a discovery payload. Returns an empty mapping when the backend
    executes hooks in process, hosts no intervention specs, or no control is enabled.

    Raises:
        UnsupportedOperationError: If an enabled control's configuration has no wire form
            (the failure names the control, the intervention, and the reason), or its spec
            requires a kind the backend does not advertise.
    """
    if Capability.IN_PROCESS_TORCH in capabilities.atoms:
        return {}
    if Capability.INTERVENTION_SPECS not in capabilities.atoms:
        return {}
    enabled = [c for c in state_controls if c.enabled]
    if not enabled:
        return {}

    advertised = capabilities.intervention_kinds
    served_model = ((getattr(backend, "_discovery", None) or {}).get("model") or {})
    payloads: dict = {}
    lowered: dict[int, InterventionEntry] = {}
    for state_control in enabled:
        lowered[id(state_control)] = _lower_control(
            state_control, advertised, served_model, payloads,
        )
    backend.stage_artifacts(payloads)
    return lowered


def _lower_control(state_control, advertised, served_model, payloads) -> InterventionEntry:
    """Lower one control to an `InterventionEntry`, verifying kinds and provenance."""
    if served_model:
        _warn_on_provenance_mismatch(state_control, served_model)
    exporter = getattr(state_control, "export_intervention_spec", None)
    spec = exporter() if callable(exporter) else None
    if spec is None:
        reason = _lowering_failure_reason(state_control)
        raise UnsupportedOperationError(
            f"{type(state_control).__name__} has no intervention-spec form for this "
            f"configuration ({reason}); run this pipeline on the huggingface backend."
        )
    required = spec.required_kinds()
    if advertised is None or not advertised.contains(required):
        missing = sorted(
            (required.transforms - (advertised.transforms if advertised else frozenset()))
            | (required.modifiers - (advertised.modifiers if advertised else frozenset()))
            | (required.scopes - (advertised.scopes if advertised else frozenset()))
            | (required.readouts - (advertised.readouts if advertised else frozenset()))
            | (required.rules - (advertised.rules if advertised else frozenset()))
        )
        raise UnsupportedOperationError(
            f"{type(state_control).__name__} requires intervention kind(s) "
            f"{', '.join(missing)} that the serving backend does not advertise; update the "
            "server's vllm_hook_plugins or run this pipeline on the huggingface backend."
        )
    payloads.update(spec.artifacts)
    return InterventionEntry(spec=spec)


def rollout_entries(state_entries, steered_input_ids, steered_attention_mask) -> tuple:
    """Rollout variants of the lowered entries for a driver on a spec-consuming backend.

    Prompt-relative scopes are rewritten to absolute positions at the generation's
    original prompt boundary. The rewrite needs one exact anchor per generation, and a
    rollout item cannot be traced back to a batch row, so uneven batches (rows whose true
    prompt lengths differ under padding) are refused. Conditional gates are refused too:
    a worker gate re-anchors its evidence at each rollout request's own prompt end, which
    would decide from generated text instead of the original prompt.

    Raises:
        UnsupportedOperationError: If the batch is uneven, a scope has no absolute rollout
            form, or an entry carries a conditional gate.
    """
    if steered_attention_mask is not None and not bool(steered_attention_mask.bool().all()):
        raise UnsupportedOperationError(
            "Driver rollouts on a spec-consuming backend need one exact prompt anchor per "
            "generation, and padded batch rows have per-row anchors; submit prompts of "
            "equal length, one prompt per call, or run this pipeline on the huggingface "
            "backend."
        )
    anchor = steered_input_ids.size(1)
    rollout_entries = []
    for entry in state_entries:
        if not isinstance(entry, InterventionEntry):
            rollout_entries.append(entry)
            continue
        if any(op.get("gate") is not None for op in entry.spec.to_wire()["ops"]):
            raise UnsupportedOperationError(
                "Conditional gating has no rollout form on a spec-consuming backend: the "
                "worker anchors gate evidence at each rollout request's own prompt end; "
                "run gated controls under a decoding driver on the huggingface backend."
            )
        try:
            rewritten = remap_prompt_relative_scopes(entry.spec, anchor)
        except ValueError as error:
            raise UnsupportedOperationError(str(error)) from error
        rollout_entries.append(InterventionEntry(spec=rewritten))
    return tuple(rollout_entries)


def _lowering_failure_reason(state_control) -> str:
    """Name the intervention (and hint) behind a lowering failure, for the raised error."""
    from steerability.algorithms.state_control.common.lowering import lower_interventions

    interventions = getattr(state_control, "interventions", ())
    num_layers = getattr(state_control, "_num_layers", None)
    if interventions and num_layers:
        for index, intervention in enumerate(interventions):
            if lower_interventions([intervention], num_layers=num_layers) is None:
                core = type(intervention.transform).__name__
                hint = getattr(state_control, "hook_only_hint", None)
                detail = f"intervention {index} ({core}) has no wire form"
                return f"{detail}; {hint}" if hint else detail
    hint = getattr(state_control, "hook_only_hint", None)
    return hint or "the configuration has no wire form"


def _warn_on_provenance_mismatch(state_control, served_model: Mapping) -> None:
    """Warn when a control's steering-artifact fingerprints differ from the served model's.

    A served `chat_template_fingerprint` equal to the absent-template digest means the
    engine exposes no chat template; that key is skipped since a mismatch against it
    reflects exposure rather than divergence.
    """
    from steerability.algorithms.core.internals.fingerprint import is_absent_chat_template_fingerprint

    artifact = getattr(state_control, "_steering_vector", None)
    meta = getattr(artifact, "meta", None) or {}
    for key in ("config_fingerprint", "chat_template_fingerprint"):
        local = meta.get(key)
        remote = served_model.get(key)
        if not local or not remote or local == remote:
            continue
        if key == "chat_template_fingerprint" and is_absent_chat_template_fingerprint(remote):
            continue
        warnings.warn(
            f"{type(state_control).__name__}'s steering artifact records a {key} of "
            f"{local}, but the serving engine reports {remote}; the artifact was fitted "
            "on a different model or tokenizer configuration than the one serving it.",
            UserWarning,
        )


def processor_spec_contributions(
    output_controls: Sequence[OutputControl],
    runtime_kwargs: dict | None,
    capabilities: BackendCapabilities,
) -> dict[int, ProcessorSpecEntry]:
    """Engine-hosted processor contributions from enabled output controls, keyed by `id()`.

    A control that returns a `ProcessorSpec` from `export_processor_spec` whose kind the
    backend serves is lowered for that call: the spec travels as a `ProcessorSpecEntry`
    and the control's live processor is not collected. The lowering choice is a ladder,
    highest supported rung first: normalized parameters, then engine-hosted specs, then
    live processors.
    """
    served = capabilities.processor_kinds
    if served is None:
        return {}
    contributions: dict[int, ProcessorSpecEntry] = {}
    for control in output_controls:
        if not control.enabled:
            continue
        exporter = getattr(control, "export_processor_spec", None)
        spec = exporter(runtime_kwargs) if callable(exporter) else None
        if spec is not None and spec.kind in served.processors:
            contributions[id(control)] = ProcessorSpecEntry(spec=spec)
    return contributions


def constraint_contributions(
    output_controls: Sequence[OutputControl],
    runtime_kwargs: dict | None,
) -> dict[int, ConstraintSource]:
    """Declarative constraint sources from enabled output controls, keyed by `id()`.

    A control that returns a source from `export_constraint` is lowered for that call on
    backends hosting structured outputs natively: the source renders onto the engine's
    request parameters and the control's live processor is not collected.
    """
    contributions: dict[int, ConstraintSource] = {}
    for control in output_controls:
        if not control.enabled:
            continue
        exporter = getattr(control, "export_constraint", None)
        source = exporter(runtime_kwargs) if callable(exporter) else None
        if source is not None:
            contributions[id(control)] = source
    return contributions


def resolve_decoding_driver(output_controls: Sequence[OutputControl]) -> DecodingDriver | None:
    """The sole enabled DecodingDriver, or None for the pipeline's default decode loop.

    merge_controls guarantees at most one enabled driver at construction; `enabled` is
    re-checked here so a driver disabled afterward falls back cleanly. The default loop
    (per-prompt items executed by the inference session) is pipeline infrastructure, not a
    phantom control.
    """
    for control in output_controls:
        if isinstance(control, DecodingDriver) and control.enabled:
            return control
    return None


def lowered_contributions(
    output_controls: Sequence[OutputControl],
    runtime_kwargs: dict | None,
) -> dict[int, Mapping]:
    """Sampling-expressible contributions from enabled output controls, keyed by `id()`.

    A control that returns a mapping from `export_generation_params` is lowered for this
    call: its contribution merges into the call's `GenerationParams` and its live processor
    and criteria hooks are not collected.
    """
    contributions: dict[int, Mapping] = {}
    for control in output_controls:
        if not control.enabled:
            continue
        exporter = getattr(control, "export_generation_params", None)
        contribution = exporter(runtime_kwargs) if callable(exporter) else None
        if contribution is not None:
            contributions[id(control)] = contribution
    return contributions


def collect_output_entries(
    output_controls: Sequence[OutputControl],
    input_ids,
    runtime_kwargs,
    *,
    attention_mask=None,
    for_scoring=False,
    skip_ids=frozenset(),
    **gen_kwargs,
) -> tuple[StackEntry, ...]:
    """One `StackEntry` per contributing output control, in controls-list order.

    With `for_scoring=True`, only `include_in_scoring` controls contribute processors and
    criteria are skipped (there is no loop to stop). Controls whose `id()` is in `skip_ids`
    (lowered to generation parameters for this call) contribute nothing. Controls
    contributing neither processors nor criteria yield no entry.
    """
    entries: list[StackEntry] = []
    for control in output_controls:
        if not control.enabled or id(control) in skip_ids:
            continue
        if for_scoring and not getattr(control, "include_in_scoring", True):
            logger.info(
                "compute_logprobs: skipping %s (include_in_scoring=False); scored logprobs will "
                "not reflect this control's logits processors.",
                type(control).__name__,
            )
            continue
        processors = control.get_logits_processors(
            input_ids, runtime_kwargs, attention_mask=attention_mask, **gen_kwargs) or []
        criteria = [] if for_scoring else (control.get_stopping_criteria(
            input_ids, runtime_kwargs, attention_mask=attention_mask, **gen_kwargs) or [])
        if processors or criteria:
            entries.append(StackEntry(
                logits_processors=tuple(processors), stopping_criteria=tuple(criteria),
            ))
    return tuple(entries)


def compose_stacks(
    output_controls: Sequence[OutputControl],
    input_ids,
    runtime_kwargs,
    attention_mask,
    gen_kwargs,
    *,
    skip_ids=frozenset(),
) -> tuple[LogitsProcessorList, StoppingCriteriaList]:
    """Compose the controls' processors and criteria, then append caller extras popped from
    `gen_kwargs` (mutates `gen_kwargs`).

    Caller-supplied `logits_processor` / `stopping_criteria` entries append after the
    pipeline's own processors and criteria (per-call extras apply on top of the pipeline's
    standing configuration) and the keys are removed from `gen_kwargs`, so exactly one
    authoritative stack of each kind exists, travelling as an explicit parameter. gen_kwargs
    reaching the driver never contains processor or criteria objects, so drivers that copy or
    serialize their kwargs are safe by construction, and a driver that ignores the stacks
    visibly ignores named parameters.
    """
    entries = collect_output_entries(
        output_controls, input_ids, runtime_kwargs, attention_mask=attention_mask,
        skip_ids=skip_ids, **gen_kwargs
    )
    processors = [p for entry in entries for p in entry.logits_processors]
    criteria = [c for entry in entries for c in entry.stopping_criteria]
    user_processors = gen_kwargs.pop("logits_processor", None) or []
    user_criteria = gen_kwargs.pop("stopping_criteria", None) or []
    return (
        LogitsProcessorList([*processors, *user_processors]),
        StoppingCriteriaList([*criteria, *user_criteria]),
    )


def apply_scoring_processors(
    output_controls: Sequence[OutputControl],
    logits,
    steered_input_ids,
    ref_output_ids,
    runtime_kwargs,
    attention_mask,
    is_encoder_decoder,
    **forward_kwargs,
) -> torch.Tensor:
    """Apply scoring-time logits processors position-by-position (teacher forcing).

    Processors receive the same `(prefix_ids, scores)` view as during generation. For causal
    models the prefix is `input ++ ref[:t]` when scoring `ref[t]`; for encoder-decoder models
    the prefix is the decoder ids `ref[:t+1]` when scoring `ref[t+1]` (matching the existing
    target alignment in both paths).
    """
    entries = collect_output_entries(
        output_controls, steered_input_ids, runtime_kwargs, attention_mask=attention_mask,
        for_scoring=True, **forward_kwargs,
    )
    processors = [p for entry in entries for p in entry.logits_processors]
    if not processors:
        return logits
    stack = LogitsProcessorList(processors)
    with torch.no_grad():
        for t in range(logits.size(1)):
            prefix = (ref_output_ids[:, : t + 1] if is_encoder_decoder
                      else torch.cat([steered_input_ids, ref_output_ids[:, :t]], dim=1))
            logits[:, t, :] = stack(prefix, logits[:, t, :])
    return logits
