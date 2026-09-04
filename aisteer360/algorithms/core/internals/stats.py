"""Ambient activation statistics: per-layer moments and residual-norm characterization."""
import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.internals.capture import HiddenStateLocation, capture_hidden, layerwise_tokenwise_hidden
from aisteer360.algorithms.core.internals.encoding import tokenize_texts
from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.pooling import get_last_token_positions, masked_mean, select_at_positions
from aisteer360.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from aisteer360.utils.rendering import PromptFormat, has_chat_template, render_for_model

logger = logging.getLogger(__name__)

STATS_FORMAT_VERSION = 1

_TENSOR_FILE = "stats.safetensors"
_META_FILE = "stats.json"


@dataclass
class StatsSpec:
    """Deferred recipe for `ActivationStats`; holds every argument except the model.

    Attributes:
        texts: Texts whose activations define the ambient distribution.
        layer_ids: Layers to estimate (0-based). None selects all decoder layers.
        location: Residual-stream boundary the statistics are measured at.
        pooling: Sample construction. `"tokens"` treats every retained real token position as one
            sample; `"mean"` and `"last"` pool per prompt first.
        exclude_first_n: Number of leading real token positions dropped per sequence under
            `pooling="tokens"`; ignored for `"mean"` and `"last"`.
        batch_size: Batch size for the capture forward passes.
    """

    texts: Sequence[str]
    layer_ids: Sequence[int] | None = None  # None selects all decoder layers
    location: str = "layer_input"
    pooling: Literal["mean", "last", "tokens"] = "tokens"
    exclude_first_n: int = 1
    batch_size: int = 8

    def estimate(
        self, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase, session=None,
    ) -> "ActivationStats":
        """Estimate `ActivationStats` on `model` with this recipe's settings."""
        return ActivationStats.estimate(
            model,
            tokenizer,
            self.texts,
            session=session,
            layer_ids=self.layer_ids,
            location=self.location,
            pooling=self.pooling,
            exclude_first_n=self.exclude_first_n,
            batch_size=self.batch_size,
        )


@dataclass
class ActivationStats:
    """Per-layer mean and variance of a model's ambient activation distribution.

    The statistics support per-coordinate centering and scaling of pooled activations
    (`standardize`), the whitening used by diagonal-LDA probe fitting. Covariance is diagonal:
    the dominant pathologies of residual-stream geometry (a large shared component in every
    activation and rogue dimensions dominating inner products) are axis-aligned, so per-coordinate
    centering and scaling removes most of the damage, and a full covariance is fragile to estimate
    and expensive to invert at a few thousand samples.

    Under `pooling="tokens"`, every retained real token position is one sample of the ambient
    distribution, and `exclude_first_n` (default 1) drops the leading real positions per sequence,
    since the first real position (BOS or an opening delimiter) carries known massive-activation
    outliers that distort the mean disproportionately. `"mean"` and `"last"` pool per prompt first,
    yielding one sample per prompt, and ignore `exclude_first_n`.

    `count` is the number of pooled samples accumulated, i.e., tokens under `pooling="tokens"` and
    prompts otherwise. `estimate` warns when `count` falls below its `min_samples` argument; a few
    hundred varied texts clears the default under token pooling, while `"mean"` and `"last"` need a
    few thousand prompts.

    Attributes:
        model_type: HuggingFace `model_type` of the model the statistics were estimated on.
        model_fingerprint: Identity digest of that model, recorded at estimate time.
        location: Residual-stream boundary the statistics were measured at.
        count: Number of pooled samples accumulated.
        mean: Per-layer mean, `[H]` float32.
        var: Per-layer variance, `[H]` float32, floored at `var_floor`.
        var_floor: Lower bound applied to every variance coordinate.
    """

    model_type: str
    model_fingerprint: str
    location: str
    count: int
    mean: dict[int, torch.Tensor]
    var: dict[int, torch.Tensor]
    var_floor: float = 1e-6

    @classmethod
    def estimate(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        *,
        layer_ids: Sequence[int] | None = None,
        location: str = "layer_input",
        pooling: Literal["mean", "last", "tokens"] = "tokens",
        exclude_first_n: int = 1,
        batch_size: int = 8,
        min_samples: int = 5000,
        session=None,
    ) -> "ActivationStats":
        """Estimate per-layer activation statistics over `texts`.

        Texts are tokenized in chunks of `batch_size`, captured at `location`, and pooled per
        `pooling`. Sequences with no real tokens contribute no samples, and duplicate entries in
        `layer_ids` are collapsed. Accumulation is streaming (per-chunk Welford merge) in float32
        on CPU, so the full feature matrix is never materialized. Capture runs under `torch.no_grad()` with
        `use_cache=False`, wrapped in `auxiliary_pass(aligned=True)`, so co-resident state
        controls' condition scorers, gates, and position counters ignore the passes while
        `"all"`-scoped behavior transforms still apply (the statistics measure the stream as
        deployed).

        Args:
            model: Model to capture activations from.
            tokenizer: Tokenizer for encoding `texts`.
            texts: Texts whose activations define the ambient distribution.
            layer_ids: Layers to estimate (0-based). None selects all decoder layers.
            location: Residual-stream boundary to measure at.
            pooling: `"tokens"` treats every retained real token position as one sample; `"mean"`
                and `"last"` pool per prompt first.
            exclude_first_n: Leading real token positions dropped per sequence under
                `pooling="tokens"`; ignored for `"mean"` and `"last"`.
            batch_size: Batch size for the capture forward passes.
            min_samples: Pooled-sample count below which a low-sample warning is emitted.

        Returns:
            An `ActivationStats` with float32 CPU tensors and the model's fingerprint recorded.

        Raises:
            ValueError: If `texts` is empty, `pooling` is unsupported, `exclude_first_n` is
                negative, `batch_size < 1`, a requested layer id is out of range, or no pooled
                samples remain after exclusion.

        Warns:
            UserWarning: When the accumulated pooled-sample `count` is below `min_samples`.
        """
        if len(texts) == 0:
            raise ValueError("texts must contain at least one text.")
        if pooling not in ("mean", "last", "tokens"):
            raise ValueError(f"pooling must be 'mean', 'last', or 'tokens', got {pooling!r}.")
        if exclude_first_n < 0:
            raise ValueError(f"exclude_first_n must be >= 0, got {exclude_first_n}.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")
        fingerprint = model_fingerprint(model)

        target_layers: list[int] | None = None
        count = 0
        running_mean: dict[int, torch.Tensor] = {}
        running_m2: dict[int, torch.Tensor] = {}

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            enc = tokenize_texts(tokenizer, chunk, device)
            with auxiliary_pass(aligned=True):
                hidden, chunk_mask = capture_hidden(
                    enc, model=model, session=session, batch_size=len(chunk), location=location
                )

            if target_layers is None:
                num_layers = len(hidden)
                if layer_ids is None:
                    target_layers = sorted(hidden.keys())
                else:
                    target_layers = list(dict.fromkeys(int(lid) for lid in layer_ids))
                    for lid in target_layers:
                        if not 0 <= lid < num_layers:
                            raise ValueError(f"layer id {lid} out of range [0, {num_layers}).")

            mask = (
                chunk_mask.to("cpu", torch.bool)
                if chunk_mask is not None
                else torch.ones(hidden[target_layers[0]].shape[:2], dtype=torch.bool)
            )

            if pooling == "tokens":
                # rank real positions per row and drop the first exclude_first_n of them
                rank = mask.long().cumsum(dim=1)
                keep = mask & (rank > exclude_first_n)  # [B, T]
                row_keep = None
            else:
                keep = None
                row_keep = mask.any(dim=1)  # rows with no real tokens contribute no sample

            chunk_count = 0
            for lid in target_layers:
                states = hidden[lid].to(torch.float32)  # [B, T, H]
                if pooling == "tokens":
                    features = states[keep]  # [K, H]
                else:
                    row_states = states[row_keep]
                    row_mask = mask[row_keep]
                    if pooling == "mean":
                        features = masked_mean(row_states, row_mask.to(row_states.dtype))
                    else:  # last
                        positions = get_last_token_positions(row_mask.long(), row_states.size(1), row_states.size(0))
                        features = select_at_positions(row_states, positions)

                chunk_count = features.size(0)
                if chunk_count == 0:
                    break

                batch_mean = features.mean(dim=0)
                batch_m2 = (features - batch_mean).pow(2).sum(dim=0)
                if lid not in running_mean:
                    running_mean[lid] = batch_mean
                    running_m2[lid] = batch_m2
                else:
                    delta = batch_mean - running_mean[lid]
                    total = count + chunk_count
                    running_mean[lid] = running_mean[lid] + delta * (chunk_count / total)
                    running_m2[lid] = running_m2[lid] + batch_m2 + delta.pow(2) * (count * chunk_count / total)

            count += chunk_count

        if count == 0:
            raise ValueError(
                "No pooled samples were accumulated; every sequence was shorter than "
                f"exclude_first_n={exclude_first_n} real tokens."
            )
        if count < min_samples:
            suggestion = (
                "supply more texts"
                if pooling == "tokens"
                else "supply more texts, or use pooling='tokens' to draw one sample per token"
            )
            warnings.warn(
                f"ActivationStats accumulated {count} pooled samples, below min_samples={min_samples}. "
                f"Estimates of per-coordinate variance may be unstable; {suggestion}.",
                UserWarning,
                stacklevel=2,
            )

        var_floor = cls.var_floor
        variances = {
            lid: (running_m2[lid] / (count - 1) if count > 1 else torch.zeros_like(running_m2[lid]))
            .clamp_min(var_floor)
            for lid in running_mean
        }
        return cls(
            model_type=model_type,
            model_fingerprint=fingerprint,
            location=location,
            count=count,
            mean=running_mean,
            var=variances,
            var_floor=var_floor,
        )

    def standardize(self, features: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Center and scale features per coordinate with this artifact's layer statistics.

        Args:
            features: Tensor of shape `[..., H]` in raw activation coordinates.
            layer_id: Layer whose statistics to apply.

        Returns:
            `(features - mean) / sqrt(var)` as a float32 tensor of the same shape.

        Raises:
            KeyError: If `layer_id` has no recorded statistics.
        """
        if layer_id not in self.mean:
            raise KeyError(
                f"ActivationStats has no statistics for layer {layer_id}; "
                f"available layers: {sorted(self.mean)}."
            )
        features = features.to(torch.float32)
        return (features - self.mean[layer_id]) / self.var[layer_id].sqrt()

    def save(self, dir_path: str | Path) -> None:
        """Save the artifact to a directory (safetensors for tensors, JSON sidecar for metadata).

        Args:
            dir_path: Directory to write to; created if missing.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        tensors = {}
        for lid in sorted(self.mean):
            tensors[f"mean.{lid}"] = self.mean[lid].to(torch.float32).contiguous()
            tensors[f"var.{lid}"] = self.var[lid].to(torch.float32).contiguous()
        save_file(tensors, str(dir_path / _TENSOR_FILE))

        metadata = {
            "format_version": STATS_FORMAT_VERSION,
            "model_type": self.model_type,
            "model_fingerprint": self.model_fingerprint,
            "location": self.location,
            "count": self.count,
            "var_floor": self.var_floor,
            "layer_ids": sorted(self.mean),
        }
        with open(dir_path / _META_FILE, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        logger.debug("Saved ActivationStats to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str | Path) -> "ActivationStats":
        """Load an artifact saved by `save`.

        Args:
            dir_path: Directory containing the safetensors file and JSON sidecar.

        Returns:
            The loaded `ActivationStats`.

        Raises:
            ValueError: If the artifact-format version is unsupported.
        """
        dir_path = Path(dir_path)
        with open(dir_path / _META_FILE) as f:
            metadata = json.load(f)

        version = metadata.get("format_version")
        if version != STATS_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported ActivationStats format version {version!r}; expected {STATS_FORMAT_VERSION}."
            )

        tensors = load_file(str(dir_path / _TENSOR_FILE))
        mean: dict[int, torch.Tensor] = {}
        var: dict[int, torch.Tensor] = {}
        for lid in metadata["layer_ids"]:
            mean[int(lid)] = tensors[f"mean.{lid}"]
            var[int(lid)] = tensors[f"var.{lid}"]

        logger.debug("Loaded ActivationStats from %s with layers %s", dir_path, sorted(mean))
        return cls(
            model_type=metadata["model_type"],
            model_fingerprint=metadata["model_fingerprint"],
            location=metadata["location"],
            count=int(metadata["count"]),
            mean=mean,
            var=var,
            var_floor=float(metadata["var_floor"]),
        )

    def fingerprint(self) -> str:
        """Digest of this artifact (not of the model it was estimated on).

        Returns:
            A 16-character lowercase hex digest over the metadata and tensor contents.
        """
        digest = hashlib.sha256()
        header = json.dumps(
            {
                "model_type": self.model_type,
                "model_fingerprint": self.model_fingerprint,
                "location": self.location,
                "count": self.count,
                "var_floor": self.var_floor,
            },
            sort_keys=True,
        )
        digest.update(header.encode("utf-8"))
        for lid in sorted(self.mean):
            digest.update(str(lid).encode("utf-8"))
            digest.update(self.mean[lid].to(torch.float32).contiguous().numpy().tobytes())
            digest.update(self.var[lid].to(torch.float32).contiguous().numpy().tobytes())
        return digest.hexdigest()[:16]


@torch.no_grad()
def measure_residual_norms(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_ids: Sequence[int] | None,
    prompts: Sequence[str],
    *,
    location: HiddenStateLocation = "layer_output",
    stat: Literal["median", "mean"] = "median",
    prompt_format: PromptFormat = "chat_prompt",
    batch_size: int = 8,
) -> dict[int, float]:
    """Typical per-token residual-stream norm at each requested layer boundary.

    The statistic is aggregated over the real (non-pad) tokens of `prompts` at the residual-stream
    boundary named by `location`. Residual norms grow with depth, so the returned per-layer norms
    let a caller rescale a direction to a fixed fraction of the norm at each layer (see
    `SteeringVector.scaled_to_norms`).

    Prompts are rendered via `render_for_model(tokenizer, prompt=p, mode=prompt_format)`. The default
    `prompt_format="chat_prompt"` renders with `add_generation_prompt=True`, matching what inference
    produces for a prompt. The rendered string is tokenized with `add_special_tokens=False` whenever
    a chat template was applied, matching how the estimators consume contrastive text.

    Args:
        model: Model to extract hidden states from.
        tokenizer: Tokenizer for rendering and encoding the prompts.
        layer_ids: Layers to measure (0-based). `None` measures every layer (no extra cost, since
            all layers are extracted regardless).
        prompts: Calibration prompts. Use fit-distribution, inference-style prompts rather than a
            held-out evaluation set.
        location: Residual-stream boundary each layer key maps to. `"layer_output"` (default)
            measures the output of each layer; `"layer_input"` measures the input of each layer, the
            boundary a layer pre-hook observes.
        stat: Aggregation over the pooled real-token norms, `"median"` (default, robust) or `"mean"`.
        prompt_format: How each prompt is rendered into model-ready text (via `render_for_model`).
        batch_size: Batch size for the extraction forward passes.

    Returns:
        Mapping from layer id to the aggregated per-token residual norm (a plain float).

    Raises:
        ValueError: If `prompts` is empty, `stat` is unsupported, or any requested layer id is out of
            range.
    """
    if len(prompts) == 0:
        raise ValueError("prompts must contain at least one prompt.")
    if stat not in ("median", "mean"):
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}.")

    device = next(model.parameters()).device

    # render then tokenize with add_special_tokens=False when a template was applied
    rendered = [render_for_model(tokenizer, prompt=p, mode=prompt_format) for p in prompts]
    template_applied = has_chat_template(tokenizer) and prompt_format != "raw"
    enc = tokenize_texts(tokenizer, rendered, device, add_special_tokens=not template_applied)

    hidden = layerwise_tokenwise_hidden(model, enc, batch_size=batch_size, location=location)
    num_layers = len(hidden)

    if layer_ids is None:
        target_layers = sorted(hidden.keys())
    else:
        target_layers = [int(lid) for lid in layer_ids]
        for lid in target_layers:
            if not 0 <= lid < num_layers:
                raise ValueError(f"layer id {lid} out of range [0, {num_layers}).")

    # real-token mask
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        keep = attention_mask.to("cpu", torch.bool)  # [N, T]
    else:
        keep = None

    norms: dict[int, float] = {}
    for lid in target_layers:
        per_token = hidden[lid].to(torch.float32).norm(dim=-1)  # [N, T]
        values = per_token[keep] if keep is not None else per_token.reshape(-1)
        agg = values.median() if stat == "median" else values.mean()
        norms[lid] = float(agg)

    return norms
