"""The intervention IR for state control components.

The intervention IR (`TokenScope`, `Intervention`) is the single declarative statement of a
residual-stream state control's behavior. Both compilers read it: `runtime.build_hooks` turns
a bound intervention tuple into torch hooks for one generation, and
`lowering.lower_interventions` turns it into an `InterventionSpec` for intervention-capable
backends. Components describe their own wire form (`WireForm` via each component's `export`),
so no layer of the system re-derives another layer's configuration by introspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, get_args

import torch

from steerability.algorithms.core.execution.contracts import InterventionKinds

from .gating import Gate, GateSource

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from steerability.algorithms.core.execution.backend import SteeringSession
    from steerability.algorithms.core.execution.payloads import ModelFacts

    from .selectors.base import BaseSelector
    from .transforms.base import BaseTransform

Boundary = Literal["layer_output", "layer_input"]
Site = Literal["decoder_layer", "o_proj", "norm_input"]
ScopeKind = Literal["all", "after_prompt", "last_k", "from_position"]


@dataclass(frozen=True, slots=True)
class WireForm:
    """One component's form on the wire: the kind name, scalar params, and named tensors.

    `params` follow the plugin's `KIND_PARAMS` table for the kind; `tensors` follow its
    `ARTIFACT_TENSORS` table.

    Attributes:
        kind: The permanent wire kind name.
        params: Scalar parameters, inlined next to `kind` on the wire.
        tensors: Named tensor payloads, materialized as one content-addressed artifact.
    """

    kind: str
    params: Mapping[str, float | int | bool | str | tuple[int, ...] | list[int]] = field(default_factory=dict)
    tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenScope:
    """A token-position selector with its parameters.

    Attributes:
        kind: One of `"all"`, `"after_prompt"`, `"last_k"`, `"from_position"`.
        last_k: Number of trailing positions, required when `kind == "last_k"`.
        from_position: Absolute start position (inclusive), required when
            `kind == "from_position"`.
    """

    kind: ScopeKind
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        if self.kind not in get_args(ScopeKind):
            raise ValueError(f"Unknown token scope kind {self.kind!r}.")
        if self.kind == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when kind is 'last_k'.")
        if self.kind == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when kind is 'from_position'.")

    def export(self) -> WireForm:
        """The scope's wire form. Total, since every scope kind is a wire kind."""
        if self.kind == "last_k":
            return WireForm(kind="last_k", params={"k": int(self.last_k)})
        if self.kind == "from_position":
            return WireForm(kind="from_position", params={"position": int(self.from_position)})
        return WireForm(kind=self.kind)


class CoveredLayers:
    """Layer selector resolving to the bound transform's covered layers.

    Used when the behavior layers are a fact of the artifact rather than of the model, e.g. a
    steering plane supplied for a subset of layers. `Intervention.bind` binds the transform
    first and takes its `covered_layer_ids` (intersected with the model's layer range, and
    with `within` when given) as the behavior layers, raising when none remain.

    Args:
        within: Optional base selection the covered layers are intersected with, as explicit
            layer ids or a selector resolved against the model's layer count.
    """

    def __init__(self, within: "Sequence[int] | BaseSelector | None" = None):
        self.within = tuple(int(lid) for lid in within) if isinstance(within, (list, tuple)) else within

    def resolve(self, covered, num_layers: int) -> tuple[int, ...]:
        """The covered layers intersected with the model range and the base selection.

        Raises:
            ValueError: If no layer survives the intersection.
        """
        layer_ids = {int(lid) for lid in covered if 0 <= int(lid) < num_layers}
        if self.within is not None:
            if isinstance(self.within, tuple):
                requested = set(self.within)
            else:
                selected = self.within.select(num_layers=num_layers)
                requested = (
                    {int(lid) for lid in selected}
                    if isinstance(selected, (list, tuple, set, frozenset))
                    else {int(selected)}
                )
            layer_ids &= requested
            if not layer_ids:
                raise ValueError(
                    f"No target layer has a direction in the steering artifact "
                    f"(requested {sorted(requested)}, available {sorted(int(lid) for lid in covered)})."
                )
        if not layer_ids:
            raise ValueError("No active layers for this intervention after filtering.")
        return tuple(sorted(layer_ids))


def _default_scope() -> TokenScope:
    return TokenScope("after_prompt")


@dataclass(frozen=True, slots=True)
class Intervention:
    """One activation edit: apply `transform` at `layers`, at `scope` positions, on the
    `boundary` side of the layer, whenever `gate` is open.

    Declared unbound at control construction: `layers` may be a layer selector, the transform
    may carry an `ArtifactSource` (or be a factory over a `TransformContext`), and the gate may
    be given as a `GateSource`. `bind(model, tokenizer, layout=...)` returns the resolved form
    with layer coverage validated. Kind identity (`wire_kinds`) is readable on the unbound
    form, which is what lets `check()` run before `steer()`.

    Interventions are generation-invariant: prompt lengths, pad masks, and position offsets
    are runtime facts consumed by `build_hooks` in process and resolved per request by the
    worker on the wire. Nothing prompt-dependent appears here.

    IR dataclasses never use instance defaults for object-valued fields; object-valued
    defaults use `default_factory` only.

    Attributes:
        layers: Behavior layers (0-based decoder-layer indices), a selector resolved at bind
            time, or `CoveredLayers` to take the bound transform's covered layers.
        transform: The transform applied at masked positions of open rows.
        scope: Token positions to steer.
        gate: The gate consulted at apply time, a source resolving to one, or None for
            unconditional application.
        gate_driven_externally: Follower mode for shared-gate composition. When True, another
            intervention's condition hooks feed the shared `Gate` instance, so binding skips
            the readout compatibility checks and hook compilation builds no condition hooks
            for this intervention; its behavior hooks only read the shared decision.
        boundary: Which side of the hooked module the edit applies at. `"layer_output"`
            builds forward hooks; `"layer_input"` builds forward pre-hooks.
        site: The hooked module family. None derives it from the transform kind
            (`head_additive` targets the attention output projection, everything else the
            decoder layer); `"norm_input"` targets each layer's normalization sub-modules
            and has no wire form.
        require_coverage: When True (default), `bind` raises if the resolved transform lacks
            a direction for any behavior layer; when False, uncovered layers are hooked and
            pass through unchanged.
    """

    layers: tuple[int, ...] | "BaseSelector" | CoveredLayers
    transform: "BaseTransform"
    scope: TokenScope = field(default_factory=_default_scope)
    gate: "Gate | GateSource | None" = None
    gate_driven_externally: bool = False
    boundary: Boundary = "layer_output"
    site: Site | None = None
    require_coverage: bool = True

    def __post_init__(self):
        if self.boundary not in ("layer_output", "layer_input"):
            raise ValueError(f"boundary must be 'layer_output' or 'layer_input'; got {self.boundary!r}.")
        if self.site not in (None, "decoder_layer", "o_proj", "norm_input"):
            raise ValueError(f"Unknown site {self.site!r}.")
        if isinstance(self.layers, (list, tuple)):
            object.__setattr__(self, "layers", tuple(int(lid) for lid in self.layers))

    @property
    def is_unbound(self) -> bool:
        """True when binding must run model-side work: a layer selector to resolve, a
        transform source or factory to fit, or a gate source to search."""
        from .transforms.base import BaseTransform

        if not isinstance(self.layers, tuple):
            return True
        if not isinstance(self.transform, BaseTransform) or not self.transform.is_bound:
            return True
        if self.gate is not None and not isinstance(self.gate, Gate):
            return True
        return False

    def resolved_site(self) -> Site:
        """The module family this intervention hooks, deriving None from the transform kind."""
        if self.site is not None:
            return self.site
        from .transforms.base import BaseTransform, unwrap_modifiers

        if isinstance(self.transform, BaseTransform):
            core, _ = unwrap_modifiers(self.transform)
            if type(core).wire_kind == "head_additive":
                return "o_proj"
        return "decoder_layer"

    def bind(
        self,
        model: "PreTrainedModel | None",
        tokenizer: "PreTrainedTokenizerBase",
        *,
        layout: "ModelFacts | None" = None,
        session: "SteeringSession | None" = None,
    ) -> "Intervention":
        """Resolve every declared element against `model` (or a session `layout`).

        Resolves the layer selector, binds the transform (fitting artifact sources and
        invoking factories), resolves gate sources, validates layer coverage and readout
        compatibility, and returns the bound intervention. Never mutates `self`.

        Args:
            model: The live model, or None for concrete-artifact configurations bound
                against a session layout.
            tokenizer: Tokenizer used when fitting sources.
            layout: Structural facts (`ModelFacts`) used when `model` is None.
            session: Optional `SteeringSession` forwarded to sources for capture-backed
                fitting and searching.

        Returns:
            The bound intervention.

        Raises:
            ValueError: If a layer is out of range, the transform lacks coverage for a
                behavior layer, the transform's artifact records an extraction location that
                differs from this intervention's boundary, or the gate's readout is
                incompatible with the boundary or model.
        """
        from .layout_facts import resolve_layout
        from .transforms.context import resolve_transform_slot

        layout = layout if layout is not None else resolve_layout(model, session)
        num_layers = layout.num_layers

        transform = self.transform
        if isinstance(self.layers, CoveredLayers):
            transform = resolve_transform_slot(
                transform, model, tokenizer, [], layout=layout,
                require_coverage=False, session=session,
            )
            covered = transform.covered_layer_ids
            if not covered:
                raise ValueError("No active layers for this intervention after filtering.")
            layer_ids = self.layers.resolve(covered, num_layers)
        elif isinstance(self.layers, tuple):
            layer_ids = self.layers
        else:
            selected = self.layers.select(num_layers=num_layers)
            if isinstance(selected, (list, tuple, set, frozenset)):
                layer_ids = tuple(sorted(int(lid) for lid in selected))
            else:
                layer_ids = (int(selected),)
        for lid in layer_ids:
            if not 0 <= lid < num_layers:
                raise ValueError(f"layer_id {lid} out of range [0, {num_layers}).")

        gate = self.gate
        if gate is not None and not isinstance(gate, Gate):
            gate = gate.resolve_gate(model, tokenizer, layout=layout, session=session)
        if gate is not None:
            if not isinstance(gate, Gate):
                raise ValueError(
                    f"gate must resolve to a Gate or None; got {type(gate).__name__}."
                )
            for lid in gate.evidence.layer_ids:
                if not 0 <= lid < num_layers:
                    raise ValueError(f"condition_layer_id {lid} out of range [0, {num_layers}).")
            if not self.gate_driven_externally:
                self._validate_readout(gate.evidence.readout, layout)

        if not isinstance(self.layers, CoveredLayers):
            transform = resolve_transform_slot(
                transform, model, tokenizer, list(layer_ids), layout=layout,
                require_coverage=self.require_coverage, session=session,
            )
        self._validate_transform_artifact(transform)

        bound = replace(self, layers=layer_ids, transform=transform, gate=gate)
        unbound_kinds = self.wire_kinds()
        bound_kinds = bound.wire_kinds()
        # binding may replace parameter values and tensors and may shrink the kind set (a gate
        # source resolving to unconditional drops its readout and rule kinds), never add kinds;
        # narrowing to None is the artifact-dependent case caught by eager steer-time lowering
        assert unbound_kinds is None or bound_kinds is None or unbound_kinds.contains(bound_kinds), (
            f"binding changed wire kinds from {unbound_kinds} to {bound_kinds}"
        )
        return bound

    def _validate_transform_artifact(self, transform) -> None:
        """Check the bound transform's recorded artifact extraction boundary against this
        intervention.

        The check validates recorded provenance only: an artifact whose metadata carries no
        `"location"` key has unknown extraction provenance and passes, with the boundary
        requirement documented on the consuming control's args instead.

        Raises:
            ValueError: If the artifact records a `"location"` that differs from this
                intervention's `boundary`.
        """
        meta = getattr(transform, "artifact_meta", None) or {}
        location = meta.get("location")
        if location is not None and location != self.boundary:
            raise ValueError(
                f"Transform artifact was extracted at '{location}' but this intervention "
                f"hooks '{self.boundary}'. Declare the intervention with "
                f"boundary='{location}', or refit the artifact with "
                f"location='{self.boundary}'."
            )

    def _validate_readout(self, readout, layout) -> None:
        """Check a readout's declared boundary and model identity against this intervention."""
        readout_location = getattr(readout, "location", None)
        if readout_location is not None and readout_location != self.boundary:
            raise ValueError(
                f"Gate readout expects features at '{readout_location}' but this "
                f"intervention hooks '{self.boundary}'. Declare the intervention with "
                f"boundary='{readout_location}', or refit the artifact with "
                f"location='{self.boundary}'."
            )
        readout_fingerprint = getattr(readout, "model_fingerprint", None)
        if readout_fingerprint is not None and layout is not None:
            live_fingerprint = getattr(layout, "model_fingerprint", None)
            if live_fingerprint is not None and readout_fingerprint != live_fingerprint:
                raise ValueError(
                    f"Gate readout was fitted on a different model (fingerprint "
                    f"{readout_fingerprint!r} vs {live_fingerprint!r}). Refit the probe on "
                    "this model, or disarm the check with allow_model_mismatch=True on "
                    "gate_from_probe() or Probe.as_gate()."
                )

    def wire_kinds(self) -> InterventionKinds | None:
        """The wire kind names this configuration lowers to, or None when hook-only.

        Readable on the unbound form: sources and components declare kind identity at
        construction, so `check()` consults this before `steer()`. Artifact-dependent
        inexpressibility (e.g. a resolved artifact missing a direction for an uncovered
        behavior layer) is undetectable here and is caught by eager steer-time lowering.
        """
        from .transforms.base import BaseTransform, unwrap_modifiers

        if self.resolved_site() == "norm_input":
            return None
        modifiers: set[str] = set()
        if isinstance(self.transform, BaseTransform):
            core, wrappers = unwrap_modifiers(self.transform)
            kind = core.wire_plan()
            if kind is None:
                return None
            for wrapper in wrappers:
                modifier_kind = wrapper.modifier_wire_kind(kind)
                if modifier_kind is None:
                    return None
                modifiers.add(modifier_kind)
        else:
            # a factory slot may declare its plan (wire_plan / wire_modifiers); an undeclared
            # factory is unknown before binding
            plan = getattr(self.transform, "wire_plan", None)
            if not callable(plan):
                return None
            kind = plan()
            if kind is None:
                return None
            declared = getattr(self.transform, "wire_modifiers", ())
            modifiers = set(declared() if callable(declared) else declared)
        if (
            self.boundary == "layer_input"
            and self.resolved_site() == "decoder_layer"
            and isinstance(self.layers, tuple)
            and 0 in self.layers
        ):
            # layer 0 input edits precede the first wire boundary; the o_proj site keeps its
            # layer index on the wire, so layer 0 stays expressible there
            return None
        gate = self.gate
        if gate is None:
            readouts: frozenset[str] = frozenset()
            rules: frozenset[str] = frozenset()
        elif isinstance(gate, Gate):
            pair = gate.wire_kinds()
            if pair is None:
                return None
            readouts, rules = pair
        else:
            readouts = getattr(type(gate), "wire_readouts", None)
            rules = getattr(type(gate), "wire_rules", None)
            if readouts is None or rules is None:
                return None
        return InterventionKinds(
            transforms=frozenset({kind}),
            modifiers=frozenset(modifiers),
            scopes=frozenset({self.scope.kind}),
            readouts=readouts,
            rules=rules,
        )


def combine_kinds(kind_sets) -> InterventionKinds | None:
    """Union `InterventionKinds` across an iterable, propagating None (hook-only)."""
    transforms: set[str] = set()
    modifiers: set[str] = set()
    scopes: set[str] = set()
    readouts: set[str] = set()
    rules: set[str] = set()
    empty = True
    for kinds in kind_sets:
        if kinds is None:
            return None
        empty = False
        transforms |= kinds.transforms
        modifiers |= kinds.modifiers
        scopes |= kinds.scopes
        readouts |= kinds.readouts
        rules |= kinds.rules
    if empty:
        return InterventionKinds()
    return InterventionKinds(
        transforms=frozenset(transforms),
        modifiers=frozenset(modifiers),
        scopes=frozenset(scopes),
        readouts=frozenset(readouts),
        rules=frozenset(rules),
    )
