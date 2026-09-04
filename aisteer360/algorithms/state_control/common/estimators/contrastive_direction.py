"""Contrastive direction estimator using paired PCA."""
import logging
import math
from typing import Callable, Literal

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.internals.capture import capture_hidden
from aisteer360.algorithms.core.internals.data import ContrastivePairs
from aisteer360.algorithms.core.internals.encoding import tokenize_texts
from aisteer360.algorithms.core.internals.fingerprint import artifact_provenance_meta, session_artifact_identity
from aisteer360.algorithms.core.internals.pooling import pool_over_spans, select_spans
from aisteer360.algorithms.core.internals.render import render_contrastive

from ..fit_specs import VectorTrainSpec
from ..steering_vector import SteeringVector
from .base import BaseEstimator

logger = logging.getLogger(__name__)

PcaMethod = Literal["pca_pairwise", "pca_center"]


def _prepare_pca_samples(
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: PcaMethod,
) -> torch.Tensor:
    """Build the PCA sample matrix from pooled positive/negative activations.

    - `pca_pairwise`: centers each pair `(H^+_i, H^-_i)` at its midpoint, yielding the two samples
        `±(H^+_i - H^-_i)/2`.
    - `pca_center`: stacks positive and negative activations and centers by their grand mean.

    Args:
        positive: Pooled positive activations, shape `[N, H]`.
        negative: Pooled negative activations, shape `[N, H]`.
        method: Which sample construction to use.

    Returns:
        A float32 sample matrix of shape `[2N, H]`.

    Raises:
        ValueError: If the shapes disagree, the method is unsupported, or the samples are non-finite.
    """
    if positive.shape != negative.shape:
        raise ValueError(
            "positive and negative pooled activations must have equal shape; "
            f"got {tuple(positive.shape)} and {tuple(negative.shape)}."
        )

    positive = positive.float()
    negative = negative.float()

    if method == "pca_pairwise":
        delta = positive - negative
        samples = torch.cat((0.5 * delta, -0.5 * delta), dim=0)
    elif method == "pca_center":
        stacked = torch.cat((positive, negative), dim=0)
        samples = stacked - stacked.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown PCA method: {method!r}.")

    if not torch.isfinite(samples).all():
        raise ValueError("PCA samples contain non-finite values.")
    return samples


def _orient_direction(
    direction: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """Orient `direction` so the positive class projects above the negative class.

    Uses a majority vote over pairs, breaking ties by the sign of the mean projection margin.

    Args:
        direction: Direction to orient, shape `[H]`.
        positive: Pooled positive activations, shape `[N, H]`.
        negative: Pooled negative activations, shape `[N, H]`.

    Returns:
        The direction, flipped if positives projected below negatives.
    """
    direction = direction.float()
    positive_projection = positive.float() @ direction
    negative_projection = negative.float() @ direction

    positive_wins = (positive_projection > negative_projection).float().mean()
    if positive_wins < 0.5:
        return -direction
    if positive_wins == 0.5:
        mean_margin = (positive_projection - negative_projection).mean()
        if mean_margin < 0:
            return -direction
    return direction


class ContrastiveDirectionEstimator(BaseEstimator[SteeringVector]):
    """Learns per-layer direction vectors from contrastive text pairs via PCA.

    Two PCA variants are supported, selected by `spec.method`:

    - `pca_pairwise`: centers each pair `(H_l^+, H_l^-)` at its midpoint, giving the samples
        `±(H_l^+ - H_l^-)/2`, and takes the first principal component of that symmetric set.
    - `pca_center`: fits PCA on the union of positive and negative pooled activations centered by
        their grand mean: `vector_l = PCA(H_l^+ - mu_l, H_l^- - mu_l)` with `mu_l` the mean over all
        examples of both classes.

    For both methods the first principal component is oriented so positive examples project above
    negative examples (see `_orient_direction`).

    Examples are rendered via `render_for_model` according to `spec.prompt_format` and tokenized with
    `add_special_tokens=False` for chat-templated text.
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
        """Extract contrastive direction vectors.

        Args:
            model: Model to extract hidden states from, or None to extract through `session`.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size).
            on_progress: Optional `(completed, total)` callback fired as each forward-pass batch
                finishes. `total` covers both positive and negative passes.
            session: A `SteeringSession` serving hidden-state capture when no live model is
                available.

        Returns:
            SteeringVector with one direction per layer.
        """
        device = next(model.parameters()).device if model is not None else torch.device("cpu")
        if model is not None:
            model_type = getattr(model.config, "model_type", "unknown")
            session_meta: dict = {}
        else:
            model_type, session_meta = session_artifact_identity(session)

        # render full texts according to prompt_format (shared with inference)
        rendered = render_contrastive(tokenizer, data, spec.prompt_format)

        logger.debug(
            "Tokenizing %d positive and %d negative examples", len(rendered.pos_texts), len(rendered.neg_texts)
        )

        # tokenize
        enc_pos = tokenize_texts(tokenizer, rendered.pos_texts, device, add_special_tokens=rendered.add_special_tokens)
        enc_neg = tokenize_texts(tokenizer, rendered.neg_texts, device, add_special_tokens=rendered.add_special_tokens)

        # tokenize prompts separately if needed for suffix-only
        prompt_enc = None
        if spec.accumulate == "suffix-only" and rendered.prompt_texts is not None:
            prompt_enc = tokenize_texts(
                tokenizer, rendered.prompt_texts, device, add_special_tokens=rendered.add_special_tokens
            )
            prompt_enc = {k: v.cpu() for k, v in prompt_enc.items()}

        # extract hidden states
        logger.debug("Extracting hidden states with batch_size=%d", spec.batch_size)
        n_pos = enc_pos["input_ids"].size(0)
        n_neg = enc_neg["input_ids"].size(0)
        total_batches = math.ceil(n_pos / spec.batch_size) + math.ceil(n_neg / spec.batch_size)
        completed = {"n": 0}

        def _tick() -> None:
            completed["n"] += 1
            if on_progress is not None:
                on_progress(completed["n"], total_batches)

        if on_progress is not None:
            on_progress(0, total_batches)
        hs_pos, mask_pos = capture_hidden(
            enc_pos, model=model, session=session,
            batch_size=spec.batch_size, on_batch=_tick, location=spec.location,
        )
        hs_neg, mask_neg = capture_hidden(
            enc_neg, model=model, session=session,
            batch_size=spec.batch_size, on_batch=_tick, location=spec.location,
        )

        # select spans against the returned layout (spans are mask-derived, so the returned
        # mask keeps them aligned when a remote capture re-packs rows)
        def _span_enc(enc, mask):
            if mask is None:
                return {k: v.cpu() for k, v in enc.items()}
            return {"input_ids": mask, "attention_mask": mask}

        spans_pos = select_spans(_span_enc(enc_pos, mask_pos), prompt_enc, spec.accumulate)
        spans_neg = select_spans(_span_enc(enc_neg, mask_neg), prompt_enc, spec.accumulate)

        # compute directions via PCA
        directions: dict[int, torch.Tensor] = {}
        explained_variances: dict[int, float] = {}

        num_layers = len(hs_pos)
        logger.debug("Computing directions for %d layers", num_layers)

        for layer_id in range(num_layers):
            # pool over spans
            Hp = pool_over_spans(hs_pos[layer_id], spans_pos)  # [N, H]
            Hn = pool_over_spans(hs_neg[layer_id], spans_neg)  # [N, H]

            samples = _prepare_pca_samples(Hp, Hn, spec.method)  # [2N, H]

            pca = PCA(n_components=1)
            pca.fit(samples.numpy())
            direction = torch.from_numpy(pca.components_[0]).float()  # [H]
            variance = float(pca.explained_variance_ratio_[0])

            direction = _orient_direction(direction, Hp, Hn)
            if not torch.isfinite(direction).all():
                raise ValueError(f"Non-finite direction produced for layer {layer_id}.")

            directions[layer_id] = direction.unsqueeze(0)  # [1, H]
            explained_variances[layer_id] = variance

        logger.debug("Finished fitting contrastive directions")
        meta = artifact_provenance_meta(model, tokenizer) if model is not None else session_meta
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            explained_variances=explained_variances,
            meta=meta,
        )
