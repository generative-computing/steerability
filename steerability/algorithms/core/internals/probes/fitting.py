"""Probe fitting: contrastive feature extraction, direction estimation, and bias calibration."""
import logging
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Sequence

import torch
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.internals.capture import capture_hidden
from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.algorithms.core.internals.encoding import tokenize_texts
from steerability.algorithms.core.internals.fingerprint import (
    artifact_provenance_meta,
    model_fingerprint,
    session_artifact_identity,
)
from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.internals.pooling import aggregate_condition_hidden
from steerability.algorithms.core.internals.probes.probe import POLARITY_MARKER, Probe
from steerability.algorithms.core.internals.render import render_contrastive
from steerability.algorithms.core.internals.stats import ActivationStats
from steerability.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from steerability.utils.rendering import PromptFormat

logger = logging.getLogger(__name__)

CalibrationSpec = Literal["max_f1", "midpoint"] | tuple[Literal["target_fpr"], float]

try:
    _PACKAGE_VERSION = version("steerability")
except PackageNotFoundError:
    _PACKAGE_VERSION = "unknown"


def _validate_calibration(calibration: CalibrationSpec) -> None:
    """Raise on a malformed calibration specification."""
    if calibration in ("max_f1", "midpoint"):
        return
    if (
        isinstance(calibration, tuple)
        and len(calibration) == 2
        and calibration[0] == "target_fpr"
        and isinstance(calibration[1], (int, float))
        and 0.0 < float(calibration[1]) < 1.0
    ):
        return
    raise ValueError(
        f"calibration must be 'max_f1', 'midpoint', or ('target_fpr', x) with 0 < x < 1; "
        f"got {calibration!r}."
    )


@dataclass(frozen=True)
class ProbeFitSpec:
    """Configuration for fitting a `Probe` from contrastive pairs.

    Attributes:
        method: Direction estimation method. `"lda"` (default) computes the difference in class
            means on standardized features (diagonal LDA) and requires `stats`; `"logreg"` is L2
            logistic regression on standardized features and also requires `stats`;
            `"mean_diff"` computes the raw difference in class means and never consults `stats`;
            `"fisher"` is a full-covariance discriminant computed from the class features alone
            (the pooled within-class covariance, pseudo-inverted with singular values truncated
            at `1e-6`, applied to the class-mean difference) and never consults `stats`. The
            truncated pseudo-inverse handles rank deficiency when sample counts are below the
            hidden size. `"fisher"` weights are unit-normalized; the scale is part of the
            method's contract, since downstream consumers score raw margins.
        pooling: Token aggregation for feature extraction (`"mean"` or `"last"`, mask-aware).
        location: Residual-stream boundary features are captured at.
        prompt_format: How pairs are rendered into model-ready text before tokenization.
        candidate_layers: Explicit layer ids to sweep. None sweeps all decoder layers, optionally
            windowed by `layer_range`.
        layer_range: Fractional-depth window `(lo, hi)` with `0 <= lo < hi <= 1`, applied to the
            full layer count when `candidate_layers` is None.
        calibration: Operating-point rule folded into the bias (see `calibrate_bias`).
        C: Inverse regularization strength for `"logreg"`.
        seed: Random seed forwarded to `"logreg"`.
    """

    method: Literal["lda", "mean_diff", "logreg", "fisher"] = "lda"
    pooling: Literal["mean", "last"] = "last"
    location: str = "layer_input"
    prompt_format: PromptFormat = "chat_prompt"
    candidate_layers: Sequence[int] | None = None
    layer_range: tuple[float, float] | None = None
    calibration: CalibrationSpec = "max_f1"
    C: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.method not in ("lda", "mean_diff", "logreg", "fisher"):
            raise ValueError(
                f"method must be 'lda', 'mean_diff', 'logreg', or 'fisher', got {self.method!r}."
            )
        if self.pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {self.pooling!r}.")
        if self.location not in ("layer_input", "layer_output"):
            raise ValueError(
                f"location must be 'layer_input' or 'layer_output', got {self.location!r}."
            )
        if self.prompt_format not in ("raw", "chat_completion", "chat_prompt"):
            raise ValueError(
                f"prompt_format must be one of raw/chat_completion/chat_prompt, "
                f"got {self.prompt_format!r}."
            )
        if self.candidate_layers is not None and len(self.candidate_layers) == 0:
            raise ValueError("candidate_layers must name at least one layer when supplied.")
        if self.layer_range is not None:
            lo, hi = self.layer_range
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(
                    f"layer_range must satisfy 0 <= lo < hi <= 1, got ({lo}, {hi})."
                )
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}.")
        _validate_calibration(self.calibration)


def calibrate_bias(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    calibration: CalibrationSpec = "max_f1",
) -> float:
    """Bias placing the operating point of `raw_score + bias >= 0` at the chosen threshold.

    The returned bias equals the negated threshold. `"midpoint"` places the threshold halfway
    between the class score means; `"max_f1"` picks the F1-maximizing threshold over the midpoints
    of consecutive distinct observed scores, breaking F1 ties toward the largest symmetric margin
    `min(pos.min() - t, t - neg.max())` (predictions `score >= threshold`; a single distinct
    observed score is used as the threshold directly);
    `("target_fpr", x)` places it at the `(1 - x)`-quantile of the negative scores. Calibration
    never repairs polarity; inverted inputs raise instead.

    Args:
        pos_scores: Raw scores of positive examples, shape `[N_pos]`.
        neg_scores: Raw scores of negative examples, shape `[N_neg]`.
        calibration: The operating-point rule.

    Returns:
        The bias `b` such that the decision is `raw_score + b >= 0`.

    Raises:
        ValueError: If either score tensor is empty, `calibration` is malformed, or
            `mean(pos_scores) < mean(neg_scores)` ("scores are inverted; negate your direction").
    """
    _validate_calibration(calibration)
    pos = torch.as_tensor(pos_scores, dtype=torch.float32).reshape(-1)
    neg = torch.as_tensor(neg_scores, dtype=torch.float32).reshape(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        raise ValueError("pos_scores and neg_scores must each contain at least one score.")
    if pos.mean() < neg.mean():
        raise ValueError("scores are inverted; negate your direction.")

    if calibration == "midpoint":
        threshold = float((pos.mean() + neg.mean()) / 2)
    elif calibration == "max_f1":
        values = torch.cat([pos, neg]).unique(sorted=True)
        if values.numel() == 1:
            threshold = float(values[0])
        else:
            pos_min, neg_max = float(pos.min()), float(neg.max())
            best = (-1.0, -float("inf"))
            threshold = float(values[:2].mean())
            for t in ((values[:-1] + values[1:]) / 2).tolist():
                f1 = _f1_at(pos, neg, t)
                margin = min(pos_min - t, t - neg_max)
                if (f1, margin) > best:
                    best, threshold = (f1, margin), t
    else:  # ("target_fpr", x)
        target = float(calibration[1])
        threshold = float(torch.quantile(neg, 1.0 - target))

    return -threshold


def _f1_at(pos: torch.Tensor, neg: torch.Tensor, threshold: float) -> float:
    """F1 of the decision `score >= threshold` over positive/negative score tensors."""
    tp = int((pos >= threshold).sum())
    fp = int((neg >= threshold).sum())
    fn = int(pos.numel()) - tp
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def _pooled_std(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Pooled within-class standard deviation of two score tensors (population form, floored)."""
    centered = torch.cat([pos - pos.mean(), neg - neg.mean()])
    return float(centered.pow(2).mean().sqrt().clamp_min(1e-8))


def _resolve_num_layers(model: PreTrainedModel | None, session) -> int:
    """Decoder layer count from a live model (given or session-exposed) or a session layout."""
    live_model = model
    if live_model is None and session is not None:
        try:
            live_model = session.model
        except (AttributeError, RuntimeError):
            live_model = None
    if live_model is not None:
        return resolve_model_layout(live_model).num_layers
    if session is not None and getattr(session, "layout", None) is not None:
        return int(session.layout.num_layers)
    raise ValueError("Layer resolution requires a live model or a capture-capable session.")


def _pooled_features(
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    data: ContrastivePairs | LabeledExamples,
    spec: ProbeFitSpec,
    layers: Sequence[int],
    batch_size: int = 8,
    max_length: int | None = None,
    session=None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Render, tokenize, capture, and pool the classes into per-layer `[N, H]` features.

    Each class's texts are processed in descending length order, in chunks of `batch_size`;
    every chunk is pooled immediately and only the pooled rows of the layers in `layers` are
    retained, so peak memory holds one chunk of per-token states plus the accumulated `[N, H]`
    features. Feature order within a class follows the length sort (features are pooled per
    class, so order carries no meaning downstream).
    """
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    rendered = render_contrastive(tokenizer, data, spec.prompt_format)

    features: list[dict[int, torch.Tensor]] = []
    for texts in (rendered.pos_texts, rendered.neg_texts):
        ordered = sorted(texts, key=len, reverse=True)
        pooled: dict[int, list[torch.Tensor]] = {lid: [] for lid in layers}
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            enc = tokenize_texts(
                tokenizer, chunk, device,
                add_special_tokens=rendered.add_special_tokens, max_length=max_length,
            )
            with auxiliary_pass(aligned=True):
                hidden, mask = capture_hidden(
                    enc, model=model, session=session, batch_size=batch_size, location=spec.location
                )
            for lid in layers:
                pooled[lid].append(
                    aggregate_condition_hidden(
                        hidden[lid].to(torch.float32), spec.pooling, attention_mask=mask
                    )
                )
        features.append({lid: torch.cat(chunks, dim=0) for lid, chunks in pooled.items()})
    return features[0], features[1]


def _fit_direction(
    pos: torch.Tensor,
    neg: torch.Tensor,
    layer_id: int,
    spec: ProbeFitSpec,
    stats: ActivationStats | None,
) -> torch.Tensor:
    """Fit one layer's weight vector in raw-activation coordinates.

    Any standardization is folded into the returned weights, so downstream scoring is a dot
    product on raw features.
    """
    if spec.method == "mean_diff":
        return (pos.mean(dim=0) - neg.mean(dim=0)).to(torch.float32)

    if spec.method == "fisher":
        mu_pos = pos.mean(dim=0)
        mu_neg = neg.mean(dim=0)
        cov = torch.cov(pos.T) * (pos.size(0) - 1) + torch.cov(neg.T) * (neg.size(0) - 1)
        cov = cov / (pos.size(0) + neg.size(0) - 2)
        # truncated pseudo-inverse of the pooled covariance; handles rank deficiency when
        # sample counts are below the hidden size
        basis, spectrum, _ = torch.linalg.svd(cov)
        keep = spectrum > 1e-6
        basis = basis[:, keep]
        w = basis @ ((basis.T @ (mu_pos - mu_neg)) / spectrum[keep])
        return (w / torch.linalg.norm(w)).to(torch.float32)

    if spec.method == "lda":
        pos_z = stats.standardize(pos, layer_id)
        neg_z = stats.standardize(neg, layer_id)
        delta_z = pos_z.mean(dim=0) - neg_z.mean(dim=0)
        return (delta_z / stats.var[layer_id].sqrt()).to(torch.float32)

    # logreg
    pos_z = stats.standardize(pos, layer_id)
    neg_z = stats.standardize(neg, layer_id)
    X = torch.cat([pos_z, neg_z]).numpy()
    y = [1] * pos_z.size(0) + [0] * neg_z.size(0)
    clf = LogisticRegression(C=spec.C, random_state=spec.seed, max_iter=5000)
    with warnings.catch_warnings():
        # sklearn's lbfgs path passes an iprint option that recent scipy rejects; the fit is
        # unaffected. the convergence filter covers small pools that reach the iteration cap.
        warnings.simplefilter("ignore", category=OptimizeWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf.fit(X, y)
    w_z = torch.as_tensor(clf.coef_[0], dtype=torch.float32)
    return w_z / stats.var[layer_id].sqrt()


def fit_probe(
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    *,
    data: ContrastivePairs | LabeledExamples,
    spec: ProbeFitSpec = ProbeFitSpec(),
    stats: ActivationStats | None = None,
    calibration_data: ContrastivePairs | LabeledExamples | None = None,
    allow_model_mismatch: bool = False,
    batch_size: int = 8,
    max_length: int | None = None,
    session=None,
) -> Probe:
    """Fit a calibrated single-layer `Probe` from contrastive data.

    The data is rendered per `spec.prompt_format`, tokenized, captured at `spec.location`
    (inside `auxiliary_pass(aligned=True)`), and pooled per `spec.pooling`. Extraction runs in
    chunks of `batch_size`, pooling each chunk immediately and retaining features only for the
    candidate layers, so memory holds pooled `[N, H]` features rather than per-token states.
    Candidate layers are swept; for each, a direction is fitted per `spec.method`, oriented on
    fit-set scores (weights are negated when the mean positive score falls below the mean
    negative score, which can occur only for `"logreg"`, since for `"mean_diff"`, `"lda"`, and
    `"fisher"` the fit-set score gap is nonnegative by construction), and the bias is
    calibrated via `calibrate_bias`. The layer with the best F1 at the calibrated point is
    kept, with the calibration score gap in pooled within-class standard deviations as
    tie-break, and the full sweep is recorded in `probe.meta["layer_sweep"]`.

    Calibration scores come from `calibration_data` when supplied, else from `data`. The split
    lets the contrast be fitted on discriminative pairs while the operating point is calibrated
    on a covering set; orientation is never re-evaluated on `calibration_data`, and a calibration
    set whose scores contradict the fit is caught by the inversion raise in `calibrate_bias`.

    Args:
        model: Model whose activations are captured.
        tokenizer: Tokenizer for rendering and encoding the data.
        data: Contrastive pairs or unpaired labeled examples the direction is fitted on.
            Unpaired classes require `spec.prompt_format` of `"raw"` or `"chat_prompt"`.
        spec: Fitting configuration.
        stats: Ambient activation statistics, required by `"lda"` and `"logreg"` (the
            standardization is folded into the stored weights). Ignored for the `"mean_diff"`
            and `"fisher"` directions.
        calibration_data: Optional data the operating point is calibrated on.
        allow_model_mismatch: When True, a `stats` artifact estimated on a different model is
            accepted instead of raising.
        batch_size: Chunk size for feature extraction.
        max_length: Tokenization truncation bound for feature extraction. None truncates to the
            tokenizer's model maximum length.

    Returns:
        A fitted `Probe` with canonical polarity, a single chosen layer, and a provenance
        `meta` record.

    Raises:
        ValueError: If the method requires `stats` and none is supplied, a supplied `stats` was
            estimated on a different model (without `allow_model_mismatch`) or at a different
            `location` than `spec.location`, a requested candidate layer is out of range or has
            no recorded statistics, `spec.prompt_format` is `"chat_completion"` with unpaired
            data, or the calibration scores are inverted.
    """
    if spec.method in ("lda", "logreg") and stats is None:
        raise ValueError(
            f"method={spec.method!r} requires ambient activation statistics; estimate "
            "ActivationStats once per model; see core.internals.stats."
        )

    if model is not None:
        fingerprint = model_fingerprint(model)
        fitted_model_type = getattr(model.config, "model_type", "unknown")
        session_meta: dict = {}
    else:
        fitted_model_type, session_meta = session_artifact_identity(session)
        fingerprint = session_meta.get("model_fingerprint")
    if (
        stats is not None
        and fingerprint is not None
        and stats.model_fingerprint != fingerprint
        and not allow_model_mismatch
    ):
        raise ValueError(
            "stats were estimated on a different model (fingerprint "
            f"{stats.model_fingerprint!r} vs {fingerprint!r}); whitening with another model's "
            "statistics produces a miscalibrated probe. Re-estimate ActivationStats on this "
            "model, or pass allow_model_mismatch=True."
        )
    if stats is not None and stats.location != spec.location:
        raise ValueError(
            f"stats were estimated at location {stats.location!r} but the fit captures at "
            f"{spec.location!r}; standardizing features with moments from a different boundary "
            "produces a miscalibrated probe. Re-estimate ActivationStats at the fit location."
        )

    num_layers = _resolve_num_layers(model, session)
    if spec.candidate_layers is not None:
        candidates = [int(lid) for lid in dict.fromkeys(spec.candidate_layers)]
        for lid in candidates:
            if not 0 <= lid < num_layers:
                raise ValueError(f"candidate layer {lid} out of range [0, {num_layers}).")
    elif spec.layer_range is not None:
        lo, hi = spec.layer_range
        candidates = [lid for lid in range(num_layers) if lo <= lid / max(num_layers - 1, 1) <= hi]
        if not candidates:
            raise ValueError(f"layer_range ({lo}, {hi}) selects no layers out of {num_layers}.")
    else:
        candidates = list(range(num_layers))

    if spec.method in ("lda", "logreg"):
        uncovered = [lid for lid in candidates if lid not in stats.mean]
        if uncovered:
            raise ValueError(
                f"stats record no statistics for candidate layer(s) {uncovered}; estimate "
                f"ActivationStats over these layers, or restrict candidate_layers to "
                f"{sorted(stats.mean)}."
            )

    pos_features, neg_features = _pooled_features(
        model, tokenizer, data, spec, candidates,
        batch_size=batch_size, max_length=max_length, session=session,
    )
    if calibration_data is not None:
        cal_pos_features, cal_neg_features = _pooled_features(
            model, tokenizer, calibration_data, spec, candidates,
            batch_size=batch_size, max_length=max_length, session=session,
        )
    else:
        cal_pos_features, cal_neg_features = pos_features, neg_features

    sweep: list[dict] = []
    best: dict | None = None
    for lid in candidates:
        w = _fit_direction(pos_features[lid], neg_features[lid], lid, spec, stats)

        fit_pos_scores = pos_features[lid] @ w
        fit_neg_scores = neg_features[lid] @ w
        fit_gap = float(fit_pos_scores.mean() - fit_neg_scores.mean())
        flipped = fit_gap < 0
        if flipped:
            assert spec.method == "logreg", (
                "fit-set orientation cannot invert for mean_diff, lda, or fisher; the score gap "
                "is a squared class-mean difference in the fit metric."
            )
            w = -w

        cal_pos_scores = cal_pos_features[lid] @ w
        cal_neg_scores = cal_neg_features[lid] @ w
        if float(cal_pos_scores.mean()) < float(cal_neg_scores.mean()):
            # the fit-set direction does not generalize to the calibration pool at
            # this layer; record the layer as unusable and keep sweeping rather than
            # aborting the whole fit (calibrate_bias would raise on these scores)
            sweep.append({
                "layer_id": lid,
                "f1": None,
                "margin": None,
                "gap": None,
                "fpr": None,
                "orientation_flipped": flipped,
                "calibration_inverted": True,
            })
            continue

        bias = calibrate_bias(cal_pos_scores, cal_neg_scores, spec.calibration)

        f1 = _f1_at(cal_pos_scores, cal_neg_scores, -bias)
        fpr = float((cal_neg_scores >= -bias).float().mean())
        gap = float(cal_pos_scores.min() - cal_neg_scores.max())
        margin = gap / _pooled_std(cal_pos_scores, cal_neg_scores)
        sweep.append({
            "layer_id": lid,
            "f1": f1,
            "margin": margin,
            "gap": gap,
            "fpr": fpr,
            "orientation_flipped": flipped,
            "calibration_inverted": False,
        })
        if best is None or (f1, margin) > (best["f1"], best["margin"]):
            best = {"layer_id": lid, "weights": w, "bias": bias, "f1": f1, "margin": margin,
                    "fpr": fpr, "orientation_flipped": flipped}

    if best is None:
        inverted = [entry["layer_id"] for entry in sweep if entry.get("calibration_inverted")]
        raise ValueError(
            f"Calibration scores are inverted at every candidate layer ({inverted}); the "
            "calibration pairs contradict the fit pairs for this concept. Revisit the "
            "pools, or the spec's pooling / layer_range."
    )

    calibration_record = {
        "kind": list(spec.calibration) if isinstance(spec.calibration, tuple) else spec.calibration,
        "on": "calibration_data" if calibration_data is not None else "data",
        "f1": best["f1"],
        "fpr": best["fpr"],
    }
    meta = {
        "method": spec.method,
        "pooling": spec.pooling,
        "location": spec.location,
        "prompt_format": spec.prompt_format,
        "n_pos": len(data.positives),
        "n_neg": len(data.negatives),
        "calibration": calibration_record,
        "layer_id": best["layer_id"],
        "layer_sweep": sweep,
        "orientation_flipped": best["orientation_flipped"],
        "stats_used": spec.method in ("lda", "logreg"),
        "model_fingerprint": fingerprint,
        "package_version": _PACKAGE_VERSION,
        "polarity": POLARITY_MARKER,
    }
    if model is not None:
        provenance = artifact_provenance_meta(model, tokenizer)
        for key in ("config_fingerprint", "chat_template_fingerprint"):
            if key in provenance:
                meta[key] = provenance[key]
    elif "model_ref" in session_meta:
        meta["model_ref"] = session_meta["model_ref"]
    if meta["stats_used"]:
        meta["stats_fingerprint"] = stats.fingerprint()

    return Probe(
        model_type=fitted_model_type,
        location=spec.location,
        pooling=spec.pooling,
        layer_ids=[best["layer_id"]],
        weights={best["layer_id"]: best["weights"]},
        bias=best["bias"],
        meta=meta,
    )


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    """Held-out scores of a probe on labeled data.

    Attributes:
        positive_scores: Signed decision scores of the positive class, shape `[N_pos]`, ordered by
            descending rendered text length rather than input order.
        negative_scores: Signed decision scores of the negative class, shape `[N_neg]`, in the same
            order.
        accuracy: Fraction correct at the calibrated point (`score >= 0` is positive).
        f1: F1 of the same decision.
    """

    positive_scores: torch.Tensor
    negative_scores: torch.Tensor
    accuracy: float
    f1: float


def evaluate_probe(
    probe: Probe,
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    data: ContrastivePairs | LabeledExamples,
    *,
    prompt_format: PromptFormat = "raw",
    batch_size: int = 8,
    max_length: int | None = None,
    session=None,
) -> ProbeEvaluation:
    """Score labeled data against a fitted probe in the probe's recorded space.

    Renders `data` per `prompt_format`, captures at the probe's `location`, pools per the probe's
    `pooling` over the probe's layers, and applies `probe.decision_function`. The probe is not
    refitted or recalibrated; the decision uses the probe's calibrated bias, so `score >= 0` is
    positive at the calibrated point.

    Args:
        probe: The fitted `Probe` to score against.
        model: Model whose activations are captured, or None when a capture-capable `session` is
            provided.
        tokenizer: Tokenizer for rendering and encoding the data.
        data: Contrastive pairs or unpaired labeled examples to score. The positives are scored as
            the positive class and the negatives as the negative class.
        prompt_format: How the data is rendered before tokenization. Must match the format the
            probe was fitted under for the scores to be comparable.
        batch_size: Chunk size for feature extraction.
        max_length: Tokenization truncation bound. None truncates to the tokenizer's model maximum.
        session: Optional capture-capable session used in place of a live model.

    Returns:
        A `ProbeEvaluation` with the per-class scores, accuracy, and F1.
    """
    spec = ProbeFitSpec(
        method="mean_diff",
        pooling=probe.pooling,
        location=probe.location,
        prompt_format=prompt_format,
        candidate_layers=list(probe.layer_ids),
        calibration="midpoint",
    )
    pos_features, neg_features = _pooled_features(
        model, tokenizer, data, spec, list(probe.layer_ids),
        batch_size=batch_size, max_length=max_length, session=session,
    )
    pos_scores = probe.decision_function(pos_features)
    neg_scores = probe.decision_function(neg_features)

    correct = int((pos_scores >= 0).sum()) + int((neg_scores < 0).sum())
    accuracy = correct / (pos_scores.numel() + neg_scores.numel())
    f1 = _f1_at(pos_scores, neg_scores, 0.0)
    return ProbeEvaluation(
        positive_scores=pos_scores, negative_scores=neg_scores, accuracy=accuracy, f1=f1
    )
