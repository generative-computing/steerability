"""ActivationAdapter: assemble an activation-steering recipe from `common` components."""
from __future__ import annotations

import logging

from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.gating import Gate
from steerability.algorithms.state_control.common.selectors import ConditionPointSelector
from steerability.algorithms.state_control.common.specs import Intervention, TokenScope

from .args import ActivationAdapterArgs

logger = logging.getLogger(__name__)


class ActivationAdapter(InterventionControl):
    """Composable activation-steering control (single-behavior atom).

    `ActivationAdapter` wires together the `state_control/common` component families (a transform
    that carries its own steering artifact, a selector, a gate, and a token scope) so an
    activation-steering recipe can be assembled directly without writing a new control class. It
    edits the residual stream at one or more layers during generation, applying the transform at
    masked positions whenever its gate is open.

    The transform is the sole artifact carrier. It holds a concrete `SteeringVector` / directions
    mapping (bound at construction), or an `ArtifactSource` such as `ContrastiveFit(data=...)` that
    is resolved once at `steer()` time. The adapter has no artifact slots and never sees a
    `SteeringVector` directly.

    Gating is likewise self-describing. A `Gate` carries its evidence (condition layers, pooling,
    and a readout mapping pooled hidden states to per-row values) and a rule deciding over the
    values; `gate_from_probe` and `Probe.as_gate` assemble one from a fitted probe, and a
    `GateSource` (e.g. `ConditionPointSearch`) resolves one at `steer()` time. The gate freezes
    its decision once every evidence layer has reported on the prompt and holds it for the
    generation.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    (transform, layers, scope, and gate), and the base class binds it at `steer()`, verifying the
    transform covers every behavior layer.

    Steering multiple behaviors is done by placing multiple adapters in a pipeline's `controls`
    list (each adapter owns exactly one transform chain / gate / token scope). Joint conditioning
    is achieved by sharing one `Gate` instance across adapters. One driver carries the gate and
    feeds it through its condition hooks; N followers pass the same gate instance with
    `gate_driven_externally=True` and read its decision. Gate reads are side-effect-free and gate
    reset is idempotent, so the shared instance is reset harmlessly once per adapter when hooks
    are built.

    Within a forward pass, a follower's behavior hook at layer L reads `is_open()` when L forwards,
    so it observes driver evidence only from condition layers `< L`. Evidence from layers `>= L`
    takes effect on the next pass. When a driver and follower hook the same layer index, place the
    driver before the follower in the pipeline's `controls` list (registration order = execution
    order).

    Batching is native (`supports_batching = True`); gates are row-vectorized, so a gated adapter
    scores and gates each prompt of a batch independently. The gate rejects scalar values for
    multi-row batches, so a mis-specified readout fails loudly rather than silently applying one
    decision batch-wide.
    """

    Args = ActivationAdapterArgs
    supports_batching = True

    def _configure(self):
        if self.layer_ids is not None:
            layers = tuple(sorted(set(int(lid) for lid in self.layer_ids)))
        else:
            if isinstance(self.layer_selector, ConditionPointSelector):
                raise ValueError(
                    "ConditionPointSelector returns a ConditionPoint for gating, not a behavior layer; "
                    "supply layer_ids or a layer selector that returns layer id(s)."
                )
            layers = self.layer_selector

        self._template = (Intervention(
            layers=layers,
            transform=self.transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            gate=self.gate,
            gate_driven_externally=self.gate_driven_externally,
            boundary=self.hook_point,
        ),)

    @property
    def hook_only_hint(self) -> str:
        gate = self.gate
        if isinstance(gate, Gate) and gate.wire_kinds() is None:
            readout_name = type(gate.evidence.readout).__name__
            rule_name = type(gate.rule).__name__
            offender = readout_name if type(gate.evidence.readout).wire_kind is None else rule_name
            return (
                f"gating through {offender} has no intervention-spec form; run on the "
                "huggingface backend"
            )
        return (
            "this transform configuration has no intervention-spec form; run on the "
            "huggingface backend"
        )

    @property
    def _layer_ids(self) -> list[int]:
        """The resolved behavior layers (empty before `steer()`)."""
        return list(self.interventions[0].layers) if self.interventions else []

    @property
    def _condition_layer_ids(self) -> list[int]:
        """The gate's evidence layers (empty when ungated)."""
        gate = self.interventions[0].gate if self.interventions else self.gate
        if isinstance(gate, Gate):
            return list(gate.evidence.layer_ids)
        return []

    def cleanup(self) -> None:
        """Drop references to the bound interventions."""
        self.interventions = ()
