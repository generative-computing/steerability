"""Calibrated affine readout over pooled hidden states, with canonical polarity."""
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden

logger = logging.getLogger(__name__)

PROBE_FORMAT_VERSION = 1
POLARITY_MARKER = "positives_high"

_TENSOR_FILE = "probe.safetensors"
_META_FILE = "probe.json"


@dataclass
class Probe:
    """Calibrated affine readout of a concept from pooled hidden states.

    A probe scores pooled per-layer features as `sum_l features[l] @ weights[l] + bias`, and its
    decision is `score >= 0` (ties open). Polarity is canonical, i.e., positives score at or
    above zero, and the operating threshold is folded into `bias` at calibration time, so no
    comparator or threshold field exists. Weights are stored in raw activation coordinates (any
    standardization used during fitting is folded into the weights), so scoring needs no
    activation statistics.

    The class is model-free feature math. It runs no forwards and registers no hooks; batched
    reads over a live model go through `ProbeSet`.

    Attributes:
        model_type: HuggingFace `model_type` of the model the probe was fitted on.
        location: Residual-stream boundary the weights were fitted at (`"layer_input"` or
            `"layer_output"`).
        pooling: Token aggregation the probe expects (`"mean"` or `"last"`, mask-aware).
        layer_ids: Layers whose features the decision function consumes.
        weights: Per-layer weight vectors, `[H]` float32, raw-activation coordinates.
        bias: Calibrated offset; the decision is `raw_score + bias >= 0`.
        meta: Provenance record (fit method, sample counts, calibration record, layer sweep,
            fingerprints, package version, polarity marker). May be empty for hand-constructed
            probes, which disarms all fingerprint checks.
    """

    model_type: str
    location: str
    pooling: Literal["mean", "last"]
    layer_ids: list[int]
    weights: dict[int, torch.Tensor]
    bias: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {self.pooling!r}.")
        if not self.layer_ids:
            raise ValueError("layer_ids must name at least one layer.")
        self.layer_ids = [int(lid) for lid in self.layer_ids]
        missing = [lid for lid in self.layer_ids if lid not in self.weights]
        if missing:
            raise ValueError(f"weights missing for layer(s) {missing}.")
        self.weights = {
            int(lid): torch.as_tensor(w).detach().reshape(-1).to(torch.float32)
            for lid, w in self.weights.items()
        }
        self.bias = float(self.bias)

    def decision_function(self, features: Mapping[int, torch.Tensor]) -> torch.Tensor:
        """Signed decision scores for pooled features.

        Args:
            features: Mapping from layer id to pooled features of shape `[B, H]` (a `[H]` tensor
                is treated as one row).

        Returns:
            Float32 tensor of shape `[B]`: the sum over `layer_ids` of `features[l] @ weights[l]`,
            plus `bias`.

        Raises:
            KeyError: If a layer in `layer_ids` is absent from `features`; the message names the
                missing layer and the layers supplied.
        """
        scores: torch.Tensor | None = None
        for lid in self.layer_ids:
            if lid not in features:
                raise KeyError(
                    f"Probe requires features for layer {lid}, but only layers "
                    f"{sorted(features)} were supplied; capture or pool every layer in "
                    f"probe.layer_ids ({self.layer_ids})."
                )
            feats = torch.as_tensor(features[lid]).to(torch.float32)
            if feats.ndim == 1:
                feats = feats.unsqueeze(0)
            contribution = feats @ self.weights[lid].to(feats.device)
            scores = contribution if scores is None else scores + contribution
        return (scores + self.bias).cpu()

    def predict(self, features: Mapping[int, torch.Tensor]) -> torch.Tensor:
        """Boolean decisions for pooled features: `decision_function(features) >= 0` (ties open).

        Args:
            features: Mapping from layer id to pooled features of shape `[B, H]`.

        Returns:
            Bool tensor of shape `[B]`.
        """
        return self.decision_function(features) >= 0

    def score_hidden(
        self,
        hidden: Mapping[int, torch.Tensor],
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool per-layer hidden states and apply the decision function.

        Each layer's `[B, T, H]` tensor is aggregated to `[B, H]` per `self.pooling` using the
        mask, so `"last"` selects the last real (non-pad) token per row.

        Args:
            hidden: Mapping from layer id to hidden states of shape `[B, T, H]`.
            prompt_mask: Attention mask of shape `[B, T]` (1 at real tokens), or None to treat
                every position as real.

        Returns:
            Float32 tensor of signed decision scores, shape `[B]`.

        Raises:
            KeyError: If a layer in `layer_ids` is absent from `hidden`.
        """
        features: dict[int, torch.Tensor] = {}
        for lid in self.layer_ids:
            if lid not in hidden:
                raise KeyError(
                    f"Probe requires hidden states for layer {lid}, but only layers "
                    f"{sorted(hidden)} were supplied; capture every layer in probe.layer_ids "
                    f"({self.layer_ids})."
                )
            features[lid] = aggregate_condition_hidden(
                hidden[lid], self.pooling, attention_mask=prompt_mask
            )
        return self.decision_function(features)

    def save(self, dir_path: str | Path) -> None:
        """Save the probe to a directory (safetensors for weights, JSON sidecar for metadata).

        Args:
            dir_path: Directory to write to; created if missing.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        tensors = {
            f"weights.{lid}": self.weights[lid].to(torch.float32).contiguous()
            for lid in sorted(self.weights)
        }
        save_file(tensors, str(dir_path / _TENSOR_FILE))

        metadata = {
            "format_version": PROBE_FORMAT_VERSION,
            "polarity": POLARITY_MARKER,
            "model_type": self.model_type,
            "location": self.location,
            "pooling": self.pooling,
            "layer_ids": list(self.layer_ids),
            "weight_layer_ids": sorted(self.weights),
            "bias": self.bias,
            "meta": self.meta,
        }
        with open(dir_path / _META_FILE, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        logger.debug("Saved Probe to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str | Path) -> "Probe":
        """Load a probe saved by `save`.

        Args:
            dir_path: Directory containing the safetensors file and JSON sidecar.

        Returns:
            The loaded `Probe`.

        Raises:
            ValueError: If the artifact-format version is unsupported or the recorded polarity
                marker is not canonical.
        """
        dir_path = Path(dir_path)
        with open(dir_path / _META_FILE) as f:
            metadata = json.load(f)

        version = metadata.get("format_version")
        if version != PROBE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Probe format version {version!r}; expected {PROBE_FORMAT_VERSION}."
            )
        polarity = metadata.get("polarity")
        if polarity != POLARITY_MARKER:
            raise ValueError(
                f"Probe artifact records polarity {polarity!r}; expected {POLARITY_MARKER!r} "
                "(positives score at or above zero)."
            )

        tensors = load_file(str(dir_path / _TENSOR_FILE))
        weights = {int(lid): tensors[f"weights.{lid}"] for lid in metadata["weight_layer_ids"]}

        logger.debug("Loaded Probe from %s with layers %s", dir_path, metadata["layer_ids"])
        return cls(
            model_type=metadata["model_type"],
            location=metadata["location"],
            pooling=metadata["pooling"],
            layer_ids=[int(lid) for lid in metadata["layer_ids"]],
            weights=weights,
            bias=float(metadata["bias"]),
            meta=metadata["meta"],
        )

    def as_gate(self, *, allow_model_mismatch: bool = False):
        """A steering gate reproducing this probe's decision, for gated interventions.

        The gate reads the probe's layers at the probe's pooling, scores each pooled state
        against the probe's weights, and opens where the summed contributions plus the
        calibrated bias are at or above zero (ties open). The decision is evaluated on the
        prompt and holds for the whole generation.

        Args:
            allow_model_mismatch: When True, the intervention's model-identity check for this
                probe is disarmed.

        Returns:
            A `Gate` for an `Intervention`'s gate slot (e.g. `ActivationAdapter`'s `gate=`).
        """
        # the single sanctioned function-local import from core/internals into a category package
        from aisteer360.algorithms.state_control.common.gating import gate_from_probe

        return gate_from_probe(self, allow_model_mismatch=allow_model_mismatch)
