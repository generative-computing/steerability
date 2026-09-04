"""Angular Steering control: rotational activation steering in a learned 2D subspace."""
from __future__ import annotations

from aisteer360.algorithms.state_control.base import InterventionControl
from aisteer360.algorithms.state_control.common.estimators import SteeringPlaneEstimator
from aisteer360.algorithms.state_control.common.sources import ContrastiveFit, LayerFilteredFit, _Precomputed
from aisteer360.algorithms.state_control.common.specs import CoveredLayers, Intervention, TokenScope
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import (
    AlignmentAdaptiveTransform,
    NormPreservingTransform,
    RotationTransform,
)
from aisteer360.algorithms.state_control.common.transforms.base import unwrap_modifiers

from .args import AngularSteeringArgs


class AngularSteering(InterventionControl):
    """Angular Steering.

    Rotates the hidden state within a per-layer 2D plane spanned by a feature axis (row 0 of the
    steering vector) and a companion axis (row 1), leaving the orthogonal complement (the other
    `d_model - 2` directions) untouched. Because a 2D rotation is orthogonal, the intervention is
    norm-preserving by construction and offers continuous control via a single angle.

    The method operates in two phases:

    1. **Offline plane fitting**: A per-layer feature axis is estimated via difference-in-means
       over contrastive data, and a single global companion axis is taken as the first principal
       component across the stacked per-layer feature directions. Gram-Schmidt yields an
       orthonormal `(b1, b2)` per layer. A precomputed `[2, H]`-per-layer plane may be supplied
       directly instead.

    2. **Online rotation**: A `forward_pre_hook` on each layer's normalization sub-modules
       (`input_layernorm` and `post_attention_layernorm`, or `ln_1`/`ln_2` on GPT-2) rotates the
       residual stream entering the norm to the target angle (`mode="target"`) or by the angle
       (`mode="offset"`). Vector addition and directional ablation are special cases of this
       rotation. The adaptive variant rotates only tokens already positively aligned with the
       feature axis, improving coherence on smaller models.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    over the artifact's covered layers, at the `"norm_input"` site by default or the decoder
    layer output when `intervention_point="layer_output"` (the only placement with an
    intervention-spec form, since the mid-layer boundary exists only inside the in-process
    forward pass).

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
    Hieu M. Vu, Tan M. Nguyen
    [https://arxiv.org/abs/2510.26243](https://arxiv.org/abs/2510.26243)
    """

    Args = AngularSteeringArgs
    supports_batching = True
    hook_only_hint = (
        "norm-input rotation has no intervention-spec form; set "
        "intervention_point='layer_output' or run on the huggingface backend"
    )

    def _configure(self):
        if self.steering_vector is not None:
            inner = _Precomputed(self.steering_vector.clone())
        else:
            inner = ContrastiveFit(
                data=self.data,
                estimator=SteeringPlaneEstimator(),
                estimator_kwargs={"spec": self.train_spec},
            )
        source = LayerFilteredFit(inner, layer_range=self.layer_range)

        transform = RotationTransform(source, angle=self.angle_radians, mode=self.mode)
        if self.adaptive:
            transform = AlignmentAdaptiveTransform(
                transform,
                source,
                threshold=self.adaptive_threshold,
                use_cosine=self.adaptive_use_cosine,
            )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        if self.intervention_point == "layer_output":
            boundary, site = "layer_output", "decoder_layer"
        else:
            boundary, site = "layer_input", "norm_input"

        self._template = (Intervention(
            layers=CoveredLayers(),
            transform=transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            boundary=boundary,
            site=site,
        ),)

    @property
    def _steering_vector(self) -> SteeringVector | None:
        """The bound steering plane as a `SteeringVector` view (None before `steer()`)."""
        if not self.interventions:
            return None
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        if getattr(core, "steering_vector", None) is None:
            return None
        return core.steering_vector
