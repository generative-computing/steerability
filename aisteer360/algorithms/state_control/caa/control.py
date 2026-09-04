from __future__ import annotations

from aisteer360.algorithms.state_control.base import InterventionControl
from aisteer360.algorithms.state_control.common.selectors import FractionalDepthSelector
from aisteer360.algorithms.state_control.common.sources import ContrastiveFit, _Precomputed
from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform
from aisteer360.algorithms.state_control.common.transforms.base import unwrap_modifiers

from .args import CAAArgs


class CAA(InterventionControl):
    """Contrastive Activation Addition (CAA).

    Steers model behavior by adding a learned mean-difference direction
    vector to the residual stream at a single layer during generation.

    CAA operates in two phases:

    1. **Training (offline)**: Given contrastive prompt pairs where each pair
       shares the same question but ends with opposite answer tokens, extract
       residual stream activations at the answer-token position. The steering
       vector is the mean difference between positive and negative activations.

    2. **Inference (online)**: Add `multiplier * v_L` to the residual stream
       at a chosen layer L, at all token positions after the user's prompt.
       A positive multiplier increases the target behavior; negative decreases it.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    (an additive transform at one layer, optionally norm-preserving), and the base class binds
    it at `steer()`.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
    Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
    [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    Args = CAAArgs
    supports_batching = True

    def _configure(self):
        if self.steering_vector is not None:
            artifact = self.steering_vector.clone()
            if self.normalize_vector:
                artifact = artifact.normalized()
            source = _Precomputed(artifact)
        else:
            source = ContrastiveFit(
                data=self.data,
                method=self.train_spec.method,
                accumulate=self.train_spec.accumulate,
                batch_size=self.train_spec.batch_size,
                prompt_format=self.train_spec.prompt_format,
                location=self.train_spec.location,
                normalize=self.normalize_vector,
            )
        transform = AdditiveTransform(source, strength=self.multiplier)
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        self._template = (Intervention(
            # heuristic default: ~40% depth (the paper finds layer 13/32 optimal)
            layers=(self.layer_id,) if self.layer_id is not None
                   else FractionalDepthSelector(fraction=0.4),
            transform=transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
        ),)

    @property
    def _layer_id(self) -> int | None:
        """The resolved behavior layer (None before `steer()`)."""
        return self.interventions[0].layers[0] if self.interventions else None

    @property
    def _steering_vector(self) -> SteeringVector | None:
        """The bound steering artifact as a `SteeringVector` view (None before `steer()`)."""
        if not self.interventions:
            return None
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        if getattr(core, "directions", None) is None:
            return None
        return SteeringVector(
            model_type="unknown",
            directions=core.directions,
            meta=core.artifact_meta or {},
        )
