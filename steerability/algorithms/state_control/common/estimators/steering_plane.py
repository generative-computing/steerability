"""Steering-plane estimator: per-layer 2D bases for angular steering."""
import logging
from typing import Callable

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.state_control.common.estimators.base import BaseEstimator
from steerability.algorithms.state_control.common.estimators.mean_difference import MeanDifferenceEstimator
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)


class SteeringPlaneEstimator(BaseEstimator[SteeringVector]):
    """Learns a 2D steering plane per layer (paper sections 4.4-4.5).

    The plane is built offline in three steps:

    1. Per-layer feature axis computed as the difference in means between positive and negative
        activations.
    2. One global companion axis: the first PCA component across the stacked per-layer feature
        directions.
    3. Per-layer Gram-Schmidt into an orthonormal `(b1, b2)`, with `b1 = normalize(d_feat)` and
        `b2 = normalize(d_PC0 - (d_PC0·b1) b1)`.

    Returns a `SteeringVector` with `directions[layer]` of shape `[2, H]` (row 0 = feature axis,
    row 1 = companion axis). The cross-layer PCA explained-variance fraction is stored in
    `explained_variances` under the sentinel key `-1`.

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
      Hieu M. Vu, Tan M. Nguyen
      [https://arxiv.org/abs/2510.26243](https://arxiv.org/abs/2510.26243)
    """

    def fit(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
        on_progress: Callable[[int, int], None] | None = None,
        session=None,
    ) -> SteeringVector:
        """Fit the per-layer steering planes.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size, prompt_format).
            on_progress: Optional `(completed, total)` callback forwarded to the underlying
                mean-difference extraction.

        Returns:
            SteeringVector with a `[2, H]` orthonormal basis per usable layer.

        Raises:
            ValueError: If fewer than two layers have usable feature directions, or if no plane
                survives orthogonalization.
        """
        # step 1: per-layer feature axis (reuse CAA's estimator)
        feature_sv = MeanDifferenceEstimator().fit(
            model, tokenizer, data=data, spec=spec, on_progress=on_progress, session=session
        )

        layer_ids = sorted(feature_sv.directions.keys())
        feature_dirs = {lid: feature_sv.directions[lid].squeeze(0).float() for lid in layer_ids}

        usable = [lid for lid in layer_ids if feature_dirs[lid].norm() >= 1e-8]
        if len(usable) < 2:
            raise ValueError(
                "SteeringPlaneEstimator needs feature directions from at least two layers "
                f"to run cross-layer PCA; got {len(usable)}."
            )

        # step 2: cross-layer PCA -> single companion axis
        feature_matrix = torch.stack([feature_dirs[lid] for lid in usable], dim=0)  # [L, H]
        pca = PCA(n_components=1)
        pca.fit(feature_matrix.numpy())
        pc0 = torch.tensor(pca.components_[0], dtype=torch.float32)
        pc0_variance = float(pca.explained_variance_ratio_[0])

        # step 3: per-layer Gram-Schmidt -> [b1, b2]
        directions: dict[int, torch.Tensor] = {}
        for lid in usable:
            b1 = feature_dirs[lid]
            b1 = b1 / b1.norm()
            b2 = pc0 - (pc0 @ b1) * b1
            if b2.norm() < 1e-8:
                logger.warning("PC0 collinear with feature direction at layer %d; skipping.", lid)
                continue
            b2 = b2 / b2.norm()
            directions[lid] = torch.stack([b1, b2], dim=0)  # [2, H]

        if not directions:
            raise ValueError("No usable steering planes after orthogonalization.")

        logger.debug("Fitted %d steering planes; PC0 explained variance %.4f", len(directions), pc0_variance)
        return SteeringVector(
            model_type=feature_sv.model_type,
            directions=directions,
            explained_variances={-1: pc0_variance},
            meta=dict(feature_sv.meta),
        )
