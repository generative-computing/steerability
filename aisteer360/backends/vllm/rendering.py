"""Request and response rendering for the vLLM sessions.

Holds the strict sampling-argument table, the finish-reason and logprob mapping helpers, the
per-item entry split and intervention-spec merge/remap, and the server-side spec-rejection
parser. These are plain functions; this module imports cleanly without vLLM installed, and the
`vllm` imports it needs stay function-local.
"""
import re
from collections.abc import Sequence
from typing import Any

import torch

from aisteer360.algorithms.core.execution.contracts import UnsupportedOperationError
from aisteer360.algorithms.core.execution.params import GenerationParams
from aisteer360.algorithms.core.execution.payloads import (
    ConstraintEntry,
    ConstraintSource,
    GenerationItem,
    HookEntry,
    InterventionEntry,
    InterventionSpec,
    ProcessorSpecEntry,
    ScoringItem,
    StackEntry,
)


def render_vllm_sampling_args(params: GenerationParams) -> dict[str, Any]:
    """Render normalized generation parameters onto vLLM sampling-parameter names.

    The table is exhaustive on this arm. Every normalized field maps to its vLLM name
    (`max_new_tokens` to `max_tokens`, `min_new_tokens` to `min_tokens`, `greedy=True` to
    `temperature=0.0`, `n` to `n`, stop strings to `stop` with
    `include_stop_str_in_output=True`, extra stop ids to `stop_token_ids`), and any key left in
    `extra` raises rather than being dropped. `seed` is not rendered here; sessions derive and
    attach per-item seeds.

    Args:
        params: The normalized parameters.

    Returns:
        Keyword arguments for `vllm.SamplingParams` (also valid as vLLM completions-request
        fields).

    Raises:
        ValueError: If `params.extra` is non-empty; the message names the unmapped keys.
        ValueError: If `params.greedy` is True while a non-zero `temperature` is also set.
    """
    if params.extra:
        raise ValueError(
            f"Generation parameter(s) {sorted(params.extra)} have no vLLM rendering; the vLLM "
            "table is exhaustive and unmapped parameters are rejected rather than dropped."
        )
    args: dict[str, Any] = {}
    if params.max_new_tokens is not None:
        args["max_tokens"] = params.max_new_tokens
    if params.min_new_tokens is not None:
        args["min_tokens"] = params.min_new_tokens
    if params.temperature is not None:
        args["temperature"] = params.temperature
    if params.top_p is not None:
        args["top_p"] = params.top_p
    if params.top_k is not None:
        args["top_k"] = params.top_k
    if params.repetition_penalty is not None:
        args["repetition_penalty"] = params.repetition_penalty
    if params.n is not None:
        args["n"] = params.n
    if params.greedy is True:
        if params.temperature not in (None, 0.0):
            raise ValueError(
                "greedy decoding conflicts with a non-zero temperature; drop one of the two."
            )
        args["temperature"] = 0.0
    if params.stop_strings:
        args["stop"] = list(params.stop_strings)
        args["include_stop_str_in_output"] = True
    if params.stop_token_ids:
        args["stop_token_ids"] = list(params.stop_token_ids)
    return args


def map_vllm_finish_reason(finish_reason: str | None, stop_reason: Any) -> str | None:
    """Map a vLLM candidate's finish reason onto the toolkit vocabulary.

    vLLM reports `"stop"` for EOS, stop strings, and stop token ids alike, with `stop_reason`
    None for EOS and the matched string or token id otherwise; `"length"` maps through
    unchanged, and anything else (e.g. `"abort"`) maps to None.

    Args:
        finish_reason: The vLLM candidate's finish reason.
        stop_reason: The vLLM candidate's stop reason.

    Returns:
        One of `"stop"`, `"eos"`, `"length"`, or None.
    """
    if finish_reason == "stop":
        return "eos" if stop_reason is None else "stop"
    if finish_reason == "length":
        return "length"
    return None


def extract_ref_logprobs(prompt_logprobs: Sequence | None, ref_ids: Sequence[int]) -> list[float]:
    """Pull the reference tokens' log-probabilities from a prompt-logprobs structure.

    Accepts both the offline shape (per-position mappings from token id to an object with a
    `logprob` attribute) and the serve JSON shape (string token-id keys mapping to dicts with a
    `"logprob"` entry). The reference occupies the last `len(ref_ids)` prompt positions.

    Args:
        prompt_logprobs: The per-prompt-position logprob entries, aligned with the submitted
            prompt tokens (position 0 is None).
        ref_ids: The reference token ids.

    Returns:
        One log-probability per reference token.

    Raises:
        ValueError: If the structure is missing or a reference position lacks its token's entry.
    """
    if prompt_logprobs is None:
        raise ValueError(
            "The response carries no prompt_logprobs; scoring requires prompt_logprobs=0 support."
        )
    if len(prompt_logprobs) < len(ref_ids):
        raise ValueError(
            f"prompt_logprobs has {len(prompt_logprobs)} positions for {len(ref_ids)} reference tokens."
        )
    values: list[float] = []
    offset = len(prompt_logprobs) - len(ref_ids)
    for position, token_id in enumerate(ref_ids):
        entry = prompt_logprobs[offset + position]
        if entry is None:
            raise ValueError(f"No logprob entry at reference position {position}.")
        record = entry.get(token_id, entry.get(str(token_id))) if hasattr(entry, "get") else None
        if record is None:
            raise ValueError(f"Token {token_id} missing from the logprob entry at position {position}.")
        if hasattr(record, "logprob"):
            values.append(float(record.logprob))
        elif isinstance(record, dict):
            values.append(float(record["logprob"]))
        else:
            values.append(float(record))
    return values


def _split_item_entries(
    items: Sequence[GenerationItem | ScoringItem],
    backend_name: str,
    *,
    plugin_active: bool,
    allow_constraints: bool = True,
) -> tuple[list[InterventionSpec | None], list[ConstraintSource | None]]:
    """Per-item intervention spec and constraint source after refusing unservable entries.

    `InterventionEntry` contributions are merged per item (ops concatenated in entry order,
    tensor payloads unioned); an item without spec entries yields None. A `ConstraintEntry`
    renders onto the engine's native structured-output parameters, one per item. Hook and
    live-processor entries name the in-process gap; intervention entries on a plugin-free
    backend name the `hook_plugin` fix.
    """
    specs: list[InterventionSpec | None] = []
    constraints: list[ConstraintSource | None] = []
    for item in items:
        item_specs: list[InterventionSpec] = []
        item_constraint: ConstraintSource | None = None
        for entry in (*item.state_entries, *item.output_entries):
            if isinstance(entry, HookEntry):
                raise UnsupportedOperationError(
                    f"HookEntry requires in-process torch hooks; the {backend_name} session "
                    "executes no client-side hooks. Run this pipeline on the huggingface backend."
                )
            if isinstance(entry, StackEntry):
                if entry.logits_processors or entry.stopping_criteria:
                    raise UnsupportedOperationError(
                        f"StackEntry carries live processor or criteria objects, which the "
                        f"{backend_name} session cannot execute; run this pipeline on the "
                        "huggingface backend."
                    )
            elif isinstance(entry, InterventionEntry):
                if not plugin_active:
                    raise UnsupportedOperationError(
                        f"InterventionEntry requires the vLLM-Hook plugin; declare "
                        f"hook_plugin=True on the {backend_name} backend spec, or run this "
                        "pipeline on the huggingface backend."
                    )
                item_specs.append(entry.spec)
            elif isinstance(entry, ConstraintEntry):
                if not allow_constraints:
                    raise UnsupportedOperationError(
                        "Structured outputs do not apply to prompt logprobs; scoring with an "
                        "enabled constraint control requires the huggingface backend or "
                        "include_in_scoring=False."
                    )
                if item_constraint is not None:
                    raise UnsupportedOperationError(
                        "The engine hosts one structured-output constraint per request; compose "
                        "constraints into one source or run this pipeline on the huggingface "
                        "backend."
                    )
                item_constraint = entry.source
            elif isinstance(entry, ProcessorSpecEntry):
                raise UnsupportedOperationError(
                    f"ProcessorSpecEntry requires engine-hosted processor kinds, which the "
                    f"{backend_name} backend does not serve; run this pipeline on the "
                    "huggingface backend."
                )
        specs.append(merge_intervention_specs(item_specs) if item_specs else None)
        constraints.append(item_constraint)
    return specs, constraints


def render_guided_decoding_field(source: ConstraintSource) -> tuple[str, Any]:
    """The vLLM structured-output parameter name and payload for a constraint source."""
    if source.kind == "json_schema":
        value = source.value if isinstance(source.value, str) else dict(source.value)
        return "json", value
    if source.kind == "regex":
        return "regex", source.value
    if source.kind == "grammar":
        return "grammar", source.value
    return "choice", list(source.value)


def render_constraint_sampling_args(field: str, value: Any) -> dict:
    """Constraint kwargs for `SamplingParams`, tolerant of the structured-outputs rename.

    Newer vLLM removes `GuidedDecodingParams` in favor of `StructuredOutputsParams` passed as
    `structured_outputs=`; older versions serve `guided_decoding=`. The declarative field names
    (`json`, `regex`, `grammar`, `choice`) are shared by both surfaces.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        # legacy api: compact whitespace is only enforced on the structured-outputs surface
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(**{field: value})}
    return {"structured_outputs": StructuredOutputsParams(**{field: value})}


def merge_intervention_specs(specs: Sequence[InterventionSpec]) -> InterventionSpec:
    """One spec carrying every op of `specs`, in order, with tensor payloads unioned."""
    if len(specs) == 1:
        return specs[0]
    ops: list = []
    artifacts: dict = {}
    for spec in specs:
        ops.extend(spec.ops)
        artifacts.update(spec.artifacts)
    return InterventionSpec(ops=tuple(ops), artifacts=artifacts)


def _load_safetensors_bytes(data: bytes) -> dict[str, torch.Tensor]:
    import safetensors.torch

    return safetensors.torch.load(data)


def remap_spec_for_scoring(spec: InterventionSpec, prompt_len: int) -> InterventionSpec:
    """A scoring copy of `spec` with `after_prompt` scopes rewritten to `from_position`.

    The teacher-forced reference is part of the server-side prompt, so the worker's "after the
    prompt" would select nothing; the rewrite anchors the scope at the original prompt length,
    the position of the first reference token in the submitted ids.
    """
    ops = []
    changed = False
    for op in spec.to_wire()["ops"]:
        if op.get("scope", {}).get("kind") == "after_prompt":
            op = {**op, "scope": {"kind": "from_position", "position": int(prompt_len)}}
            changed = True
        ops.append(op)
    if not changed:
        return spec
    return InterventionSpec(ops=tuple(ops), artifacts=spec.artifacts)


# spec-rejection codes that are support facts (a capability or constraint the backend lacks)
# rather than malformed payloads
_SUPPORT_FACT_CODES = ("E_UNKNOWN_KIND", "E_CONSTRAINT")
_SPEC_ERROR_RE = re.compile(r"\bE_[A-Z_]+ at \S+:")


def raise_for_spec_rejection(message: str) -> None:
    """Raise the toolkit error for a server-side spec rejection message carrying an `E_*` code.

    Kind and constraint gaps (`E_UNKNOWN_KIND`, `E_CONSTRAINT`) are support facts a stale
    client missed and raise `UnsupportedOperationError`; every other `E_*` rejection is a
    malformed spec and raises `ValueError`. The code and JSON path are preserved verbatim.
    A message without an `E_*` code returns without raising.
    """
    if not _SPEC_ERROR_RE.search(message):
        return
    if any(code in message for code in _SUPPORT_FACT_CODES):
        raise UnsupportedOperationError(message)
    raise ValueError(message)
