"""The vLLM capability tables, discovery cache and negotiation, and capability refusals.

The kind tables and baseline capabilities are static data used by `check()`; the discovery
cache and negotiation narrow them to what a live engine or server confirms. This module imports
cleanly without vLLM installed.
"""
import logging
from collections.abc import Sequence

from aisteer360.algorithms.core.execution.contracts import (
    BackendCapabilities,
    Capability,
    CaptureKinds,
    ConstraintKinds,
    InterventionKinds,
    ProcessorKinds,
    UnsupportedOperationError,
)
from aisteer360.algorithms.core.execution.payloads import InterventionSpec
from aisteer360.algorithms.core.execution.spec import BackendSpec

logger = logging.getLogger(__name__)

_PLUGIN_INTERVENTION_KINDS = InterventionKinds(
    transforms=frozenset({"additive", "projection", "rotation", "head_additive"}),
    modifiers=frozenset({"norm_preserving", "alignment_adaptive"}),
    scopes=frozenset({"all", "after_prompt", "last_k", "from_position"}),
    readouts=frozenset({"affine", "cosine", "projected_cosine"}),
    rules=frozenset({"per_key_threshold", "sum_threshold"}),
    constraints={"head_additive": "tensor_parallel_size==1"},
)


_PLUGIN_CAPTURE_KINDS = CaptureKinds(
    kinds=frozenset({"residual"}),
    locations=frozenset({"layer_output", "layer_input"}),
    modes=frozenset({"all_tokens", "last_token"}),
)

_VLLM_CONSTRAINT_KINDS = ConstraintKinds(
    constraints=frozenset({"json_schema", "regex", "grammar", "choice"}),
)
VLLM_BASELINE_CAPABILITIES = BackendCapabilities(
    atoms=frozenset({
        Capability.SERVE_CHECKPOINT,
        Capability.SERVE_LORA,
        Capability.GUIDED_DECODING,
    }),
    constraint_kinds=_VLLM_CONSTRAINT_KINDS,
)

_DISCOVERY_CACHE: dict[str, dict] = {}


def _vllm_capabilities(spec: BackendSpec, *, offline: bool) -> BackendCapabilities:
    """Capabilities implied by a vLLM spec: the plugin-free baseline, extended when the spec
    declares the vLLM-Hook plugin active. Hidden capture is advertised on the offline engine
    only, since serve-mode capture needs a bulk-tensor return path.

    Once a backend for the spec has fetched discovery, the advertised kind sets are the
    intersection of the static tables and the discovery payload, so a server missing a kind
    stops advertising it."""
    if not spec.get_option("hook_plugin"):
        return VLLM_BASELINE_CAPABILITIES
    atoms = VLLM_BASELINE_CAPABILITIES.atoms | {
        Capability.INTERVENTION_SPECS,
    }
    capture_kinds = None
    if offline:
        atoms = atoms | {Capability.HIDDEN_CAPTURE}
        capture_kinds = _PLUGIN_CAPTURE_KINDS
    capabilities = BackendCapabilities(
        atoms=frozenset(atoms),
        intervention_kinds=_PLUGIN_INTERVENTION_KINDS,
        capture_kinds=capture_kinds,
        constraint_kinds=_VLLM_CONSTRAINT_KINDS,
    )
    payload = _DISCOVERY_CACHE.get(spec.spec_hash)
    if payload is not None:
        capabilities = _intersect_with_discovery(capabilities, payload)
    return capabilities


def _intersect_with_discovery(capabilities: BackendCapabilities, payload: dict) -> BackendCapabilities:
    """The static capability tables narrowed to what the discovery payload confirms."""
    remote_interventions = payload.get("intervention_kinds") or {}
    intervention_kinds = capabilities.intervention_kinds
    if intervention_kinds is not None:
        intervention_kinds = InterventionKinds(
            transforms=intervention_kinds.transforms & frozenset(remote_interventions.get("transforms", ())),
            modifiers=intervention_kinds.modifiers & frozenset(remote_interventions.get("modifiers", ())),
            scopes=intervention_kinds.scopes & frozenset(remote_interventions.get("scopes", ())),
            readouts=intervention_kinds.readouts & frozenset(remote_interventions.get("readouts", ())),
            rules=intervention_kinds.rules & frozenset(remote_interventions.get("rules", ())),
            constraints=dict(remote_interventions.get("constraints", {}) or intervention_kinds.constraints),
        )
    remote_processors = payload.get("processor_kinds") or {}
    processor_kinds = capabilities.processor_kinds
    if processor_kinds is not None:
        processor_kinds = ProcessorKinds(
            processors=processor_kinds.processors & frozenset(remote_processors.get("processors", ())),
        )
    remote_capture = payload.get("capture_kinds") or {}
    capture_kinds = capabilities.capture_kinds
    if capture_kinds is not None:
        capture_kinds = CaptureKinds(
            kinds=capture_kinds.kinds & frozenset(remote_capture.get("kinds", ())),
            locations=capture_kinds.locations & frozenset(remote_capture.get("locations", ())),
            modes=capture_kinds.modes & frozenset(remote_capture.get("modes", ())),
        )
    return BackendCapabilities(
        atoms=capabilities.atoms,
        intervention_kinds=intervention_kinds,
        processor_kinds=processor_kinds,
        capture_kinds=capture_kinds,
        constraint_kinds=capabilities.constraint_kinds,
    )


def _refuse_by_engine_facts(discovery: dict | None, operation: str) -> None:
    """Refuse intervention or capture submission when discovery reports incompatible engine facts."""
    engine = (discovery or {}).get("engine", {})
    if engine.get("speculative_decoding"):
        raise UnsupportedOperationError(
            f"The serving engine runs speculative decoding, so {operation} requests are refused: "
            "draft-model forwards are unhooked and verification passes break the worker's "
            "position accounting. Disable speculative decoding on the engine."
        )
    if engine.get("enforce_eager") is False:
        raise UnsupportedOperationError(
            f"The serving engine compiles CUDA graphs, so {operation} requests are refused: "
            "worker hooks do not run under CUDA-graph replay. Start the engine with "
            "enforce_eager=True / --enforce-eager."
        )


def _refuse_by_constraints(
    specs: Sequence[InterventionSpec | None],
    discovery: dict | None,
    advertised: InterventionKinds | None,
) -> None:
    """Refuse specs whose kinds violate an advertised engine constraint, naming the fix.

    The only shipped constraint is `head_additive: tensor_parallel_size==1`; the check reads
    the constraint table from the negotiated kinds and the live value from discovery's engine
    facts, so the refusal matches what server-side staging would reject with `E_CONSTRAINT`.
    """
    constraints = dict(advertised.constraints) if advertised is not None else {}
    if not constraints or discovery is None:
        return
    tensor_parallel_size = (discovery.get("engine") or {}).get("tensor_parallel_size", 1)
    if tensor_parallel_size == 1:
        return
    for spec in specs:
        if spec is None:
            continue
        constrained = spec.required_kinds().transforms & set(constraints)
        if constrained:
            kind = sorted(constrained)[0]
            raise UnsupportedOperationError(
                f"Intervention kind {kind!r} requires {constraints[kind]}, but the serving engine "
                f"reports tensor_parallel_size={tensor_parallel_size}; serve the model with "
                "tensor_parallel_size=1 or run this pipeline on the huggingface backend."
            )


def _reconcile_discovery(spec: BackendSpec, static: BackendCapabilities, payload: dict) -> None:
    """Warn when the discovery payload disagrees with the static advertisement.

    The static tables are the spec-implied advertisement; the discovery payload is the runtime
    authority. Kind-set gating consumes the intersection when spec lowering lands; at this
    phase a mismatch is surfaced as a warning.
    """
    discovered = payload.get("intervention_kinds", {})
    static_kinds = static.intervention_kinds
    if static_kinds is not None:
        for field_name, advertised in (
            ("transforms", static_kinds.transforms),
            ("modifiers", static_kinds.modifiers),
            ("scopes", static_kinds.scopes),
            ("readouts", static_kinds.readouts),
            ("rules", static_kinds.rules),
        ):
            remote = set(discovered.get(field_name, []))
            missing = advertised - remote
            if missing:
                logger.warning(
                    "vLLM-Hook discovery for spec %s lacks advertised %s %s; the intersection "
                    "governs spec execution.",
                    spec.spec_hash, field_name, sorted(missing),
                )
