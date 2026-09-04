"""Steering vector: per-layer direction vectors learned by an estimator."""
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """Per-layer direction tensors for activation steering.

    Directions are stored as [K, D] tensors per layer, where K and D are
    interpreted by the consuming transform:

        CAA, CAST: K=1, D=hidden_size (single broadcast direction)
        ActAdd: K=T, D=hidden_size (positional directions)
        Angular Steering: K=2, D=hidden_size (orthonormal basis pair)
        ITI: K=num_heads, D=head_dim (per-head directions)

    The container is agnostic to what K and D mean (varies depending on the method).
    Semantics come from the consumer (transform).

    Attributes:
        model_type: HuggingFace model_type string (e.g., "llama").
        directions: Mapping from layer_id to direction tensor of shape [K, D].
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each head's output.
        explained_variances: Optional mapping from layer_id to explained
            variance scalar. Only meaningful for estimators that produce a
            real variance (e.g., PCA-based). None when not applicable.
        probe_accuracies: Optional mapping from (layer_id, head_id) to linear
            probe validation accuracy (used for head selection in ITI).
        meta: Provenance record (model, config, tokenizer, and chat-template fingerprints,
            package version). May be empty for hand-constructed vectors, which disarms
            cross-backend fingerprint checks.
    """

    model_type: str
    directions: dict[int, torch.Tensor]
    num_heads: int | None = None
    head_dim: int | None = None
    explained_variances: dict[int, float] | None = None
    probe_accuracies: dict[tuple[int, int], float] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of token positions in the steering vector (K dimension)."""
        if not self.directions:
            return 0
        return next(iter(self.directions.values())).size(0)

    @property
    def is_positional(self) -> bool:
        """True if the vector carries per-token positional structure (K > 1)."""
        return self.num_tokens > 1

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SteeringVector":
        """Move all direction tensors to device/dtype.

        This mutates `self.directions` in place and returns `self` (so callers may chain).
        To leave the original untouched, `clone()` first.
        """
        self.directions = {
            k: v.to(device=device, dtype=dtype) if dtype else v.to(device=device)
            for k, v in self.directions.items()
        }
        return self

    def clone(self) -> "SteeringVector":
        """Return a deep copy with independent direction tensors and metadata dicts.

        Use this before `to()` / normalization when the vector is caller-supplied and may be
        reused (e.g., shared across controls or across a `Benchmark`/`ControlSpec` sweep).
        """
        return SteeringVector(
            model_type=self.model_type,
            directions={k: v.clone() for k, v in self.directions.items()},
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            explained_variances=dict(self.explained_variances) if self.explained_variances is not None else None,
            probe_accuracies=dict(self.probe_accuracies) if self.probe_accuracies is not None else None,
            meta=dict(self.meta),
        )

    def normalized(self) -> "SteeringVector":
        """Return a per-layer L2-normalized deep clone.

        Each direction tensor is normalized by its own L2 norm (over all elements). Rows with a
        zero (or near-zero) norm are left unchanged. This is the bound-mode twin of
        `ContrastiveFit(normalize=True)`: the original vector is untouched, so a shared/caller-held
        vector stays intact.

        Returns:
            A new `SteeringVector` with normalized directions and copied metadata.
        """
        clone = self.clone()
        for layer_id, direction in clone.directions.items():
            norm = direction.norm()
            if norm > 0:
                clone.directions[layer_id] = direction / norm
        return clone

    def scaled_to_norms(
        self,
        target_norms: Mapping[int, float],
        scale: float = 1.0,
    ) -> "SteeringVector":
        """Return a deep clone whose per-layer directions are rescaled to a target L2 norm.

        At each layer present in both `target_norms` and `self.directions`, the direction is rescaled
        to L2 norm `scale * target_norms[layer]` while keeping its orientation, scaling each direction
        to a caller-chosen (typically residual-norm-relative) magnitude. The original vector is
        untouched.

        Semantics are defined for broadcast directions only (`K == 1`), where every steered
        token in the layer receives the same vector.

        Args:
            target_norms: Mapping from layer id to the per-layer base norm to scale to (e.g. the
                output of `measure_residual_norms`). Layers absent from either `target_norms` or
                `self.directions` are dropped from the result.
            scale: Multiplier applied to each target norm (the "dose"). Must be positive.

        Returns:
            A new `SteeringVector` covering the intersection of `target_norms` and `self.directions`,
            each direction stored as `[1, H]`, with metadata copied from the original.

        Raises:
            ValueError: If `scale <= 0`, any covered target norm is non-positive, a covered source
                direction is positional (`K > 1`) or has zero norm, or the layer intersection is
                empty.
        """
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")

        covered = [lid for lid in target_norms if lid in self.directions]
        if not covered:
            raise ValueError(
                "scaled_to_norms: no overlap between target_norms layers "
                f"{sorted(target_norms.keys())} and vector layers {sorted(self.directions.keys())}."
            )

        clone = self.clone()
        scaled: dict[int, torch.Tensor] = {}
        for lid in covered:
            target = float(target_norms[lid])
            if target <= 0:
                raise ValueError(f"target norm for layer {lid} must be positive, got {target}.")

            direction = clone.directions[lid]
            if direction.ndim == 2 and direction.shape[0] == 1:
                direction = direction.squeeze(0)
            elif direction.ndim != 1:
                raise ValueError(
                    f"scaled_to_norms is defined for broadcast directions (K=1); layer {lid} has "
                    f"shape {tuple(clone.directions[lid].shape)}."
                )

            source_norm = direction.norm()
            if source_norm <= 0:
                raise ValueError(
                    f"layer {lid} has a zero-norm source direction; a target norm is unreachable "
                    "from the zero vector."
                )

            scaled[lid] = (direction * (scale * target / source_norm)).unsqueeze(0)  # [1, H]

        clone.directions = scaled
        return clone

    def validate(self) -> None:
        """Validate that required fields are populated.

        Raises:
            ValueError: If model_type or directions are empty.
        """
        if not self.model_type:
            raise ValueError("model_type must be provided.")
        if not self.directions:
            raise ValueError("directions must not be empty.")

    def save(self, file_path: str) -> None:
        """Save the SteeringVector to a JSON file.

        Args:
            file_path: Path to save to. ".svec" extension added if not present.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)
        data: dict = {
            "model_type": self.model_type,
            "directions": {str(k): v.tolist() for k, v in self.directions.items()},
        }

        if self.num_heads is not None:
            data["num_heads"] = self.num_heads
        if self.head_dim is not None:
            data["head_dim"] = self.head_dim
        if self.explained_variances is not None:
            data["explained_variances"] = {str(k): v for k, v in self.explained_variances.items()}
        if self.probe_accuracies is not None:
            data["probe_accuracies"] = {
                f"{layer}:{head}": acc for (layer, head), acc in self.probe_accuracies.items()
            }
        if self.meta:
            data["meta"] = self.meta

        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.debug("Saved SteeringVector to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """Load a SteeringVector from a JSON file.

        Args:
            file_path: Path to load from. ".svec" extension added if not present.

        Returns:
            Loaded SteeringVector instance.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        with open(file_path) as f:
            data = json.load(f)

        directions = {}
        for k, v in data["directions"].items():
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # [D] -> [1, D] backward compatibility
            directions[int(k)] = t

        explained_variances = None
        if "explained_variances" in data:
            explained_variances = {int(k): float(v) for k, v in data["explained_variances"].items()}

        num_heads = data.get("num_heads")
        head_dim = data.get("head_dim")

        probe_accuracies = None
        if "probe_accuracies" in data:
            probe_accuracies = {}
            for k, acc in data["probe_accuracies"].items():
                layer_str, head_str = k.split(":")
                probe_accuracies[(int(layer_str), int(head_str))] = float(acc)

        logger.debug("Loaded SteeringVector from %s with layers %s", file_path, list(directions.keys()))
        return cls(
            model_type=data["model_type"],
            directions=directions,
            num_heads=num_heads,
            head_dim=head_dim,
            explained_variances=explained_variances,
            probe_accuracies=probe_accuracies,
            meta=data.get("meta", {}),
        )
