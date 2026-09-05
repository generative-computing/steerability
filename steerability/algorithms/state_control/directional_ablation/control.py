"""Directional Ablation control: projects a learned direction out of the residual stream."""
from __future__ import annotations

from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from steerability.algorithms.state_control.common.selectors import FractionalDepthSelector
from steerability.algorithms.state_control.common.sources import ContrastiveFit, LayerFilteredFit, _Precomputed
from steerability.algorithms.state_control.common.specs import CoveredLayers, Intervention, TokenScope
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import NormPreservingTransform, ProjectionTransform
from steerability.algorithms.state_control.common.transforms.base import unwrap_modifiers

from .args import DirectionalAblationArgs


class DirectionalAblation(InterventionControl):
    """Directional Ablation (feature removal via projection).

    Removes a learned feature direction from the residual stream at one or more layers during
    generation, `h' = h - alpha * (d_hat^T h) d_hat` at masked positions. This is the abliteration
    technique of Arditi et al., which learns a direction as the difference in means over
    contrastive data and projects it out.

    The method operates in two phases:

    1. Training (offline). Extract residual activations for contrastive pairs and take the mean
       difference, or the PCA of paired differences, as the feature direction. A precomputed
       direction (or an orthonormal subspace, `K > 1`) may be supplied directly.

    2. Inference (online). At each target layer's output, project the direction out of the
       residual stream at masked positions. `alpha = 1.0` fully removes the component
       (`h'.d_hat == 0`); `alpha < 1.0` gives graded partial suppression.

    Ablation is a projection (idempotent at `alpha=1`, norm-reducing). It can compose with the
    alignment-adaptive gate (`AlignmentAdaptiveTransform`) to ablate only where the feature is
    present.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    whose behavior layers are the target layers intersected with the artifact's coverage.

    Reference:

    - "Refusal in Language Models Is Mediated by a Single Direction"
    Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
    [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
    """

    Args = DirectionalAblationArgs
    supports_batching = True

    def _configure(self):
        if self.steering_vector is not None:
            inner = _Precomputed(self.steering_vector.clone())
        else:
            estimator = (
                ContrastiveDirectionEstimator()
                if self.train_spec.method == "pca_pairwise"
                else MeanDifferenceEstimator()
            )
            inner = ContrastiveFit(
                data=self.data, estimator=estimator, estimator_kwargs={"spec": self.train_spec},
            )
        source = LayerFilteredFit(inner, layer_range=self.layer_range)

        transform = ProjectionTransform(source, alpha=self.alpha)
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        self._template = (Intervention(
            # heuristic default: single layer at ~40% depth (matches CAA)
            layers=CoveredLayers(
                within=tuple(sorted(set(self.layer_ids))) if self.layer_ids is not None
                       else FractionalDepthSelector(fraction=0.4)
            ),
            transform=transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
        ),)

    @property
    def hook_only_hint(self) -> str:
        if self.alpha != 1.0:
            return "graded ablation (alpha < 1) has no intervention-spec form; run on the huggingface backend"
        return "subspace ablation has no intervention-spec form; run on the huggingface backend"

    @property
    def _layer_ids(self) -> list[int]:
        """The resolved target layers (empty before `steer()`)."""
        return list(self.interventions[0].layers) if self.interventions else []

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
