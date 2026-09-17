"""VJP-delta control."""
from __future__ import annotations

from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.sources import _Precomputed
from steerability.algorithms.state_control.common.specs import CoveredLayers, Intervention, TokenScope
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import AdditiveTransform

from .args import VJPDeltaArgs
from .fit import VJPDeltaFit


class VJPDelta(InterventionControl):
    """VJP-delta activation steering.

    The control fits one normalized additive direction per source layer during `steer()`. Its
    target contrast is the positive-minus-negative final unpadded target state. The fit applies
    that contrast as a cotangent at valid target tokens and averages valid source-token gradients
    per prompt before separately averaging the positive and negative classes. At generation it
    uses the standard additive intervention and token scopes.

    A precomputed `SteeringVector` or `ArtifactSource` avoids gradient extraction. The frozen form
    is `ActivationAdapter`, so a reloaded `.spipe` resolves the stored vectors without a VJP fit.

    Reference:

    - Clark, Michael J. (2026). "vjp-steering: contrastive steering vectors from
      vector-Jacobian products."
      [https://github.com/wassname/vjp-steering](https://github.com/wassname/vjp-steering)
      Adapts the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/).
    """

    Args = VJPDeltaArgs
    supports_batching = True

    def _configure(self) -> None:
        if self.steering_vector is None:
            source = VJPDeltaFit(
                data=self.data,
                target_layer=self.target_layer,
                source_layer_ids=self.source_layer_ids,
                skip_first=self.skip_first,
                max_length=self.max_length,
                batch_size=self.batch_size,
            )
        elif isinstance(self.steering_vector, SteeringVector):
            source = _Precomputed(self.steering_vector.clone())
        else:
            source = self.steering_vector
        self._template = (
            Intervention(
                layers=CoveredLayers(),
                transform=AdditiveTransform(source, strength=self.strength),
                scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
            ),
        )

    def cleanup(self) -> None:
        """Drop fitted intervention tensors and their bound artifacts."""
        self.interventions = ()
