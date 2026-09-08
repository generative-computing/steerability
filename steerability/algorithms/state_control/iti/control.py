"""Inference-Time Intervention (ITI) state control."""
from __future__ import annotations

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.selectors import TopKHeadSelector
from steerability.algorithms.state_control.common.sources import _Precomputed
from steerability.algorithms.state_control.common.specs import CoveredLayers, Intervention, TokenScope
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import HeadAdditiveTransform, NormPreservingTransform
from steerability.algorithms.state_control.common.transforms.base import BaseTransform, unwrap_modifiers

from .args import ITIArgs
from .utils import ProbeMassShiftEstimator


class _HeadSelectionBuild:
    """Transform factory: resolve the head-shift artifact, select heads, build the transform.

    Head selection is a fact of the artifact (top-K heads by probe accuracy), so the transform
    is constructed at bind time from the resolved `SteeringVector`. The factory declares its
    own steer access: a precomputed vector builds model-free, while fitting captures
    pre-`o_proj` per-head activations, which no backend serves remotely.
    """

    def __init__(self, source, selected_heads, num_heads: int, alpha: float, norm_preserving: bool):
        self._source = source
        self._selected_heads = selected_heads
        self._num_heads = num_heads
        self._alpha = alpha
        self._norm_preserving = norm_preserving

    @property
    def access(self) -> ModelAccess:
        return getattr(self._source, "access", ModelAccess.MODULE)

    @property
    def artifact_class(self) -> str | None:
        return getattr(self._source, "artifact_class", None)

    def fit_ingredients(self) -> dict:
        """The encodable fit-identity form: the source's ingredients plus the head selection."""
        from steerability.algorithms.state_control.base import _fit_ingredients

        return {
            "kind": "head_selection",
            "source": _fit_ingredients(self._source),
            "selected_heads": self._selected_heads,
            "num_heads": self._num_heads,
        }

    def __call__(self, ctx) -> BaseTransform:
        steering_vector = ctx.resolve(self._source)

        if self._selected_heads is not None:
            selected = self._selected_heads
        else:
            if steering_vector.probe_accuracies is None:
                raise ValueError(
                    "steering_vector has no probe_accuracies. "
                    "Either provide selected_heads explicitly or use data to train a new vector."
                )
            selected = TopKHeadSelector(self._num_heads).select(steering_vector=steering_vector)

        active_heads: dict[int, set[int]] = {}
        for layer_id, head_id in selected:
            active_heads.setdefault(layer_id, set()).add(head_id)

        transform: BaseTransform = HeadAdditiveTransform(
            steering_vector, active_heads=active_heads, strength=self._alpha,
        )
        if self._norm_preserving:
            transform = NormPreservingTransform(transform)
        return transform


class _ProbeMassShiftFit:
    """A fit recipe for ITI's per-head mass-shift vector.

    Fitting captures pre-`o_proj` per-head activations, a capture kind no backend advertises,
    so the fit requires a live model.
    """

    access = ModelAccess.MODULE
    artifact_class = "direction"

    def __init__(self, data, train_spec):
        self._data = data
        self._train_spec = train_spec

    def fit_ingredients(self) -> dict:
        """The encodable fit-identity form: the labeled data and the train spec."""
        return {"kind": "probe_mass_shift", "data": self._data, "train_spec": self._train_spec}

    def resolve(self, model, tokenizer, *, session=None) -> SteeringVector:
        if model is None:
            raise ValueError("Fitting ITI from data requires a live model at steer time.")
        return ProbeMassShiftEstimator().fit(model, tokenizer, data=self._data, spec=self._train_spec)


class ITI(InterventionControl):
    """Inference-Time Intervention (ITI).

    Steers model behavior by shifting activations at a sparse set of attention heads
    during inference. The intervention operates at the residual stream level by adding
    direction vectors to head-associated slices of the hidden dimension.

    ITI operates in two phases:

    1. **Offline (during steer())**: For every attention head across all layers,
       extract the head's output activations on labeled true/false statements.
       Train a per-head linear probe on an 80/20 held-out split (over groups when
       the data carries them) and rank heads by validation accuracy. For the
       top-K heads, compute the mass mean shift: direction = mean(activations_true)
       - mean(activations_false).

    2. **Online (during generation)**: At each generated token, for each selected
       (layer, head) pair, add alpha * direction to that head's slice of the
       residual stream. The intervention fires unconditionally on every token
       in the specified token_scope.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    at the attention output projection (the site derived from the `head_additive` transform
    kind), over the layers hosting selected heads.

    Reference:

    - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
    Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
    [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
    """

    Args = ITIArgs
    supports_batching = True
    hook_only_hint = (
        "norm preservation over per-head streams has no intervention-spec form; "
        "run on the huggingface backend"
    )

    def _configure(self):
        if self.steering_vector is not None:
            if isinstance(self.steering_vector, SteeringVector):
                source = _Precomputed(self.steering_vector.clone())
            else:
                source = self.steering_vector
        else:
            source = _ProbeMassShiftFit(self.data, self.train_spec)

        self._template = (Intervention(
            layers=CoveredLayers(),
            transform=_HeadSelectionBuild(
                source,
                selected_heads=self.selected_heads,
                num_heads=self.num_heads,
                alpha=self.alpha,
                norm_preserving=self.use_norm_preservation,
            ),
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            boundary="layer_input",
        ),)

    def wire_kinds(self):
        """`head_additive` kinds for the bound configuration; conservative before `steer()`.

        The factory-built transform is unknown before binding, but its kind is definitional
        for this control, so the plan is stated directly: `head_additive` unless norm
        preservation is on (the wire modifier rescales the residual row, not the per-head
        stream).
        """
        from steerability.algorithms.core.execution.contracts import InterventionKinds
        from steerability.algorithms.state_control.common.specs import combine_kinds

        if self.interventions:
            return combine_kinds(intervention.wire_kinds() for intervention in self.interventions)
        if self.use_norm_preservation:
            return None
        return InterventionKinds(
            transforms=frozenset({"head_additive"}),
            scopes=frozenset({self.token_scope}),
        )

    @property
    def _active_layer_ids(self) -> set[int]:
        """Layers hosting selected heads (empty before `steer()`)."""
        return set(self.interventions[0].layers) if self.interventions else set()

    @property
    def _steering_vector(self) -> SteeringVector | None:
        """The bound head-shift artifact (None before `steer()`)."""
        if not self.interventions:
            return None
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        return getattr(core, "steering_vector", None)

    def export_state(self) -> dict:
        """The bound per-head steering vector under the `"steering_vector"` key (after `steer()`)."""
        vector = self._steering_vector
        return {"steering_vector": vector} if vector is not None else {}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """A same-class frozen form: the bound per-head vector plus the resolved head selection."""
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        selected = sorted(
            (int(layer_id), int(head_id))
            for layer_id, heads in core.active_heads.items()
            for head_id in heads
        )
        return "state_control/iti", {
            "steering_vector": state["steering_vector"],
            "selected_heads": selected,
            "num_heads": self.num_heads,
            "alpha": self.alpha,
            "token_scope": self.token_scope,
            "last_k": self.last_k,
            "from_position": self.from_position,
            "use_norm_preservation": self.use_norm_preservation,
        }
