"""Backend contracts: capability advertisement, the requirement language, and support verdicts.

A capability atom marks a mechanism that some control requirement can fail on; facts true of
every backend belong to the session protocol contract instead. Kind sets state which activation
edits, per-step logit processors, capture forms, and native constraints a capable backend
executes. Controls state what a backend must provide as phase-keyed `Requirements`, and
`evaluate_support` renders binary per-control, per-phase verdicts against the pipeline's
backend. The steer phase produces no verdicts; steer-time model access is declared through
`ModelAccess` and satisfied by the pipeline's steer plan.
"""
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from steerability.algorithms.core.execution.access import SteerPlan
from steerability.algorithms.core.execution.spec import BackendSpec


class Capability(Enum):
    """Distinguishing capability atoms advertised by backends.

    Attributes:
        IN_PROCESS_TORCH: The backend exposes the model as a live `torch.nn.Module` in the client
            process, so torch hooks, live logits processors, and direct weight access are
            available. The name refers to this mechanism rather than to process locality.
        INTERVENTION_SPECS: The backend executes activation interventions submitted as
            `InterventionSpec` payloads. The Hugging Face backend does not advertise this atom,
            since torch hooks cover every intervention a spec expresses; requirements state the
            relationship as alternatives.
        PER_STEP_LOGIT_SPECS: The backend hosts per-step logit math submitted as `ProcessorSpec`
            payloads.
        HIDDEN_CAPTURE: The backend serves hidden-state capture through `SteeringSession.capture`.
        BEAM_PROPOSALS: The backend implements beam-search proposal semantics (`num_beams` with
            multiple returned sequences).
        SERVE_CHECKPOINT: The backend can serve a checkpoint directory produced elsewhere.
        SERVE_LORA: The backend can serve a LoRA adapter produced elsewhere.
        GUIDED_DECODING: The backend hosts declarative constrained decoding natively, rendered
            from a `ConstraintSource` onto its structured-output request parameters. The
            Hugging Face backend does not advertise this atom, since the in-process arm serves
            the constraint class through a client-compiled automaton; requirements state the
            relationship as alternatives.
    """

    IN_PROCESS_TORCH = "in_process_torch"
    INTERVENTION_SPECS = "intervention_specs"
    PER_STEP_LOGIT_SPECS = "per_step_logit_specs"
    HIDDEN_CAPTURE = "hidden_capture"
    BEAM_PROPOSALS = "beam_proposals"
    SERVE_CHECKPOINT = "serve_checkpoint"
    SERVE_LORA = "serve_lora"
    GUIDED_DECODING = "guided_decoding"


@dataclass(frozen=True, slots=True)
class InterventionKinds:
    """Activation-intervention kinds a backend executes, by permanent wire name.

    Wire names mirror toolkit class names (`AdditiveTransform` serializes as `"additive"`,
    `SumThreshold` as `"sum_threshold"`), so the mapping is definitional rather than
    maintained. Kind names are permanent and their meanings never change; new behavior is a
    new kind. Compatibility is set containment on kind names.

    Attributes:
        transforms: Transform kinds, e.g. `{"additive", "projection", "rotation",
            "head_additive"}`.
        modifiers: Wrapper-transform kinds, e.g. `{"norm_preserving", "alignment_adaptive"}`.
        scopes: Token-scope kinds, e.g. `{"all", "after_prompt", "last_k", "from_position"}`.
        readouts: Gate readout kinds, e.g. `{"affine", "cosine", "projected_cosine"}`; an
            ungated op needs none.
        rules: Gate rule kinds, e.g. `{"sum_threshold", "per_key_threshold"}`.
        constraints: Per-kind execution constraints, e.g.
            `{"head_additive": "tensor_parallel_size==1"}`. Informational; containment checks
            ignore this field.
    """

    transforms: frozenset[str] = frozenset()
    modifiers: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    readouts: frozenset[str] = frozenset()
    rules: frozenset[str] = frozenset()
    constraints: Mapping[str, str] = field(default_factory=dict)

    def contains(self, required: "InterventionKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return (
            required.transforms <= self.transforms
            and required.modifiers <= self.modifiers
            and required.scopes <= self.scopes
            and required.readouts <= self.readouts
            and required.rules <= self.rules
        )


@dataclass(frozen=True, slots=True)
class ProcessorKinds:
    """Engine-hosted logit-processor kinds a backend executes, by permanent wire name.

    Attributes:
        processors: Processor kinds, e.g. `{"constraint"}`.
    """

    processors: frozenset[str] = frozenset()

    def contains(self, required: "ProcessorKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return required.processors <= self.processors


@dataclass(frozen=True, slots=True)
class CaptureKinds:
    """Hidden-state capture forms a backend serves, by permanent wire name.

    Attributes:
        kinds: Capture kinds, e.g. `{"residual"}`.
        locations: Capture locations, e.g. `{"layer_output", "layer_input"}`.
        modes: Capture modes, e.g. `{"all_tokens", "last_token"}`.
    """

    kinds: frozenset[str] = frozenset()
    locations: frozenset[str] = frozenset()
    modes: frozenset[str] = frozenset()

    def contains(self, required: "CaptureKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return (
            required.kinds <= self.kinds
            and required.locations <= self.locations
            and required.modes <= self.modes
        )


@dataclass(frozen=True, slots=True)
class ConstraintKinds:
    """Constrained-decoding kinds a backend hosts natively, by declarative kind name.

    The kind set is static per backend version (the engine's structured-output surface needs no
    discovery), e.g. `{"json_schema", "regex", "grammar", "choice"}`.

    Attributes:
        constraints: Constraint kind names.
    """

    constraints: frozenset[str] = frozenset()

    def contains(self, required: "ConstraintKinds") -> bool:
        """Return True when every required kind name is advertised."""
        return required.constraints <= self.constraints


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """A backend's full capability advertisement: atoms plus negotiated kind sets.

    Attributes:
        atoms: The advertised `Capability` atoms.
        intervention_kinds: Advertised intervention kinds, present when
            `Capability.INTERVENTION_SPECS` is among the atoms.
        processor_kinds: Advertised processor kinds, present when
            `Capability.PER_STEP_LOGIT_SPECS` is among the atoms.
        capture_kinds: Advertised capture kinds, present when `Capability.HIDDEN_CAPTURE` is
            among the atoms.
        constraint_kinds: Advertised constrained-decoding kinds, present when
            `Capability.GUIDED_DECODING` is among the atoms.
    """

    atoms: frozenset[Capability] = frozenset()
    intervention_kinds: InterventionKinds | None = None
    processor_kinds: ProcessorKinds | None = None
    capture_kinds: CaptureKinds | None = None
    constraint_kinds: ConstraintKinds | None = None


KindSet = InterventionKinds | ProcessorKinds | CaptureKinds | ConstraintKinds

PHASES: tuple[str, ...] = ("generate", "score")


@dataclass(frozen=True, slots=True)
class Alternative:
    """One way to satisfy a phase requirement, i.e., a conjunction of capability atoms with
    optional kind predicates over the backend's advertised kind sets.

    Attributes:
        atoms: Capability atoms that must all be advertised.
        kinds: Kind sets whose names must all be contained in the backend's advertisement of the
            corresponding kind-set type.
        hint: Optional fix text used in unsupported-verdict messages in place of the default.
    """

    atoms: frozenset[Capability] = frozenset()
    kinds: tuple[KindSet, ...] = ()
    hint: str | None = None

    def satisfied_by(self, capabilities: BackendCapabilities) -> bool:
        """Return True when every atom is advertised and every kind set is contained."""
        if not self.atoms <= capabilities.atoms:
            return False
        for kind_set in self.kinds:
            advertised = _advertised_for(kind_set, capabilities)
            if advertised is None or not advertised.contains(kind_set):
                return False
        return True

    def missing(self, capabilities: BackendCapabilities) -> list[str]:
        """Names of the atoms and kind sets this alternative needs but `capabilities` lacks."""
        gaps = [atom.name for atom in sorted(self.atoms - capabilities.atoms, key=lambda a: a.name)]
        for kind_set in self.kinds:
            advertised = _advertised_for(kind_set, capabilities)
            if advertised is None or not advertised.contains(kind_set):
                gaps.append(f"{type(kind_set).__name__}({_kind_names(kind_set)})")
        return gaps


def _advertised_for(kind_set: KindSet, capabilities: BackendCapabilities) -> KindSet | None:
    """The backend's advertised kind set of the same type as `kind_set`, or None."""
    if isinstance(kind_set, InterventionKinds):
        return capabilities.intervention_kinds
    if isinstance(kind_set, ProcessorKinds):
        return capabilities.processor_kinds
    if isinstance(kind_set, ConstraintKinds):
        return capabilities.constraint_kinds
    return capabilities.capture_kinds


def _kind_names(kind_set: KindSet) -> str:
    """Comma-joined sorted kind names across the set's name-bearing fields."""
    if isinstance(kind_set, InterventionKinds):
        names = kind_set.transforms | kind_set.modifiers | kind_set.scopes | kind_set.readouts | kind_set.rules
    elif isinstance(kind_set, ProcessorKinds):
        names = kind_set.processors
    elif isinstance(kind_set, ConstraintKinds):
        names = kind_set.constraints
    else:
        names = kind_set.kinds | kind_set.locations | kind_set.modes
    return ", ".join(sorted(names))


def needs(
    *atoms: Capability,
    kinds: KindSet | tuple[KindSet, ...] | None = None,
    hint: str | None = None,
) -> tuple[Alternative, ...]:
    """Build a single-alternative phase requirement.

    Args:
        *atoms: Capability atoms that must all be advertised.
        kinds: One kind set, or a tuple of kind sets, whose names must be contained in the
            backend's advertisement.
        hint: Optional fix text for unsupported-verdict messages.

    Returns:
        A one-element tuple of `Alternative`, directly assignable to a `Requirements` phase.
    """
    if kinds is None:
        kind_sets: tuple[KindSet, ...] = ()
    elif isinstance(kinds, tuple):
        kind_sets = kinds
    else:
        kind_sets = (kinds,)
    return (Alternative(atoms=frozenset(atoms), kinds=kind_sets, hint=hint),)


def any_of(*alternatives: tuple[Alternative, ...] | Alternative) -> tuple[Alternative, ...]:
    """Combine alternatives into a disjunction, satisfied by its first satisfied alternative.

    Args:
        *alternatives: `Alternative` instances or tuples of them (as returned by `needs`).

    Returns:
        The flattened tuple of alternatives.
    """
    flattened: list[Alternative] = []
    for alternative in alternatives:
        if isinstance(alternative, Alternative):
            flattened.append(alternative)
        else:
            flattened.extend(alternative)
    return tuple(flattened)


@dataclass(frozen=True, slots=True)
class SpecConstraint:
    """A predicate over a resolved `BackendSpec`, for backend-configuration facts.

    Attributes:
        description: The unsupported-verdict message shown when the predicate fails. It should
            name the conflict and a fix.
        predicate: Callable evaluated against the phase's `BackendSpec`; True means satisfied.
        phases: Phases whose backend spec the predicate is evaluated against.
    """

    description: str
    predicate: Callable[[BackendSpec], bool]
    phases: tuple[str, ...] = ("generate",)

    def __post_init__(self) -> None:
        unknown = [phase for phase in self.phases if phase not in PHASES]
        if unknown:
            raise ValueError(f"Unknown phases {unknown}; phases are {', '.join(PHASES)}.")


@dataclass(frozen=True, slots=True)
class Requirements:
    """Phase-keyed backend requirements computed by a control instance.

    Each phase holds a tuple of `Alternative`s (a disjunction); an empty tuple requires nothing
    beyond the session contract, which includes the model layout. The steer phase carries no
    requirements; a control declares its steer-time model access through `steer_access()`.

    Attributes:
        generate: Alternatives for the generate phase.
        score: Alternatives for the score phase.
        spec_constraints: Backend-configuration predicates, each evaluated against the spec of
            every phase it names.
    """

    generate: tuple[Alternative, ...] = ()
    score: tuple[Alternative, ...] = ()
    spec_constraints: tuple[SpecConstraint, ...] = ()

    def for_phase(self, phase: str) -> tuple[Alternative, ...]:
        """The alternatives for `phase` (one of `"generate"`, `"score"`).

        Raises:
            ValueError: If `phase` is not a known phase name.
        """
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; phases are {', '.join(PHASES)}.")
        return getattr(self, phase)


_DEFAULT_HINT = "run this pipeline on the huggingface backend"


class UnsupportedPipelineError(RuntimeError):
    """Raised when an operation targets a backend that does not support the pipeline.

    Attributes:
        report: The `SupportReport` whose failures triggered the error.
    """

    def __init__(self, report: "SupportReport", phases: tuple[str, ...]) -> None:
        self.report = report
        failures = report.failures_for(*phases)
        lines = "\n".join(f"- {failure.message}" for failure in failures)
        super().__init__(
            f"Pipeline is unsupported on the configured backends ({len(failures)} unsupported "
            f"requirement(s)):\n{lines}"
        )


class UnsupportedOperationError(RuntimeError):
    """Raised when a session receives work it cannot execute on its backend."""


@dataclass(frozen=True, slots=True)
class SupportFailure:
    """One unsupported verdict.

    Attributes:
        control: Class name of the failing control.
        phase: The phase the verdict applies to (`"generate"` or `"score"`).
        message: Stable, tested message naming the gap and a fix.
    """

    control: str
    phase: str
    message: str


@dataclass(frozen=True, slots=True)
class SupportReport:
    """The result of evaluating every enabled control against a backend.

    Attributes:
        spec: The backend spec the phases were evaluated against.
        plan: The deterministic steer plan for this configuration.
        failures: All unsupported verdicts, in controls-list order then phase order.
    """

    spec: BackendSpec
    plan: SteerPlan = field(default_factory=SteerPlan)
    failures: tuple[SupportFailure, ...] = ()

    @property
    def ok(self) -> bool:
        """True when no phase of any enabled control is unsupported."""
        return not self.failures

    def failures_for(self, *phases: str) -> tuple[SupportFailure, ...]:
        """The failures whose phase is among `phases`."""
        return tuple(failure for failure in self.failures if failure.phase in phases)

    def supported(self, *phases: str) -> bool:
        """True when no failure falls in any of `phases`."""
        return not self.failures_for(*phases)

    def raise_for(self, *phases: str) -> None:
        """Raise `UnsupportedPipelineError` listing every failing control in `phases`, if any."""
        if not self.supported(*phases):
            raise UnsupportedPipelineError(self, phases)


def _phase_failure_message(
    control_name: str,
    phase: str,
    spec: BackendSpec,
    requirements: Requirements,
    capabilities: BackendCapabilities,
) -> str:
    """Build the unsupported message for one control phase, naming the gaps and a fix."""
    alternatives = requirements.for_phase(phase)
    gap_parts = []
    hint = None
    for alternative in alternatives:
        gaps = alternative.missing(capabilities)
        gap_parts.append(" + ".join(gaps) if gaps else "unsatisfied alternative")
        if hint is None and alternative.hint is not None:
            hint = alternative.hint
    if hint is None:
        missing_atom_names = {gap for part in gap_parts for gap in part.split(" + ")}
        if Capability.IN_PROCESS_TORCH.name in missing_atom_names:
            hint = _DEFAULT_HINT
    message = (
        f"{control_name} is unsupported at {phase} on backend kind '{spec.kind}': "
        f"missing {' or '.join(gap_parts)}"
    )
    return f"{message}; {hint}." if hint else f"{message}."


def evaluate_support(
    controls: Iterable[Any],
    spec: BackendSpec,
    capabilities: BackendCapabilities,
) -> SupportReport:
    """Evaluate every enabled control's requirements against a backend.

    For each enabled control, `control.requirements()` is read once and each declared phase is
    checked against the backend's capabilities. Spec constraints are checked for every phase
    they name. Controls whose `enabled` attribute is False are skipped.

    Args:
        controls: Control instances, in pipeline order.
        spec: The backend spec.
        capabilities: Capability advertisement of the backend.

    Returns:
        A `SupportReport` whose `failures` hold one entry per unsupported (control, phase) pair
        and per violated spec constraint.
    """
    failures: list[SupportFailure] = []
    for control in controls:
        if not getattr(control, "enabled", True):
            continue
        control_name = type(control).__name__
        requirements: Requirements = control.requirements()

        for phase in PHASES:
            alternatives = requirements.for_phase(phase)
            if not alternatives:
                continue
            if any(alternative.satisfied_by(capabilities) for alternative in alternatives):
                continue
            failures.append(SupportFailure(
                control=control_name,
                phase=phase,
                message=_phase_failure_message(control_name, phase, spec, requirements, capabilities),
            ))

        for constraint in requirements.spec_constraints:
            for phase in constraint.phases:
                if constraint.predicate(spec):
                    continue
                failures.append(SupportFailure(
                    control=control_name,
                    phase=phase,
                    message=(
                        f"{control_name} is unsupported at {phase} on backend kind "
                        f"'{spec.kind}': {constraint.description}"
                    ),
                ))

    return SupportReport(spec=spec, failures=tuple(failures))
