"""Head-level additive transform for activation steering."""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from ..specs import WireForm
    from .context import TransformContext


class HeadAdditiveTransform(BaseTransform):
    """Adds scaled direction vectors to specific head slices.

    For each selected (layer, head) pair, it adds a direction vector to the slice
    [h * head_dim : (h+1) * head_dim].

    For ITI, this operates in pre-o_proj space: the input to the output projection
    where each head_dim-sized slice corresponds to an individual attention head's
    output. The directions must be computed in the same space.

    Expects a SteeringVector whose directions are shaped [num_heads, head_dim]
    per layer, with num_heads and head_dim metadata set. Only head indices
    present in ``active_heads`` are applied; other heads are left untouched.

    A bare directions mapping cannot carry the required `num_heads`/`head_dim` metadata and is
    rejected; supply a `SteeringVector` (or an `ArtifactSource` resolving to one).

    Args:
        artifact: A `SteeringVector` with per-head directions and `num_heads`/`head_dim` metadata,
            or an `ArtifactSource` (unbound until `bind(ctx)`). Required.
        active_heads: Mapping from layer_id to set of head indices to intervene on.
        strength: Global scaling factor (alpha in ITI).
    """

    wire_kind: ClassVar[str | None] = "head_additive"

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        active_heads: dict[int, set[int]],
        strength: float = 1.0,
    ):
        self.active_heads = active_heads
        self.strength = strength
        self._source: ArtifactSource | None = None
        self.steering_vector: SteeringVector | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.steering_vector = artifact
            self._validate_artifact()
        elif isinstance(artifact, Mapping):
            raise ValueError(
                "HeadAdditiveTransform requires num_heads and head_dim metadata on the SteeringVector; "
                "a bare directions mapping cannot carry it."
            )
        else:
            raise TypeError(
                f"HeadAdditiveTransform artifact must be a SteeringVector or an ArtifactSource; got "
                f"{type(artifact).__name__} (did you mean strength=?)."
            )

    def _validate_artifact(self) -> None:
        """Validate the artifact carries num_heads/head_dim metadata."""
        if self.steering_vector.num_heads is None or self.steering_vector.head_dim is None:
            raise ValueError("HeadAdditiveTransform requires num_heads and head_dim metadata on the SteeringVector.")
        self.num_heads = self.steering_vector.num_heads
        self.head_dim = self.steering_vector.head_dim

    @property
    def is_bound(self) -> bool:
        return self.steering_vector is not None

    @property
    def artifact_meta(self) -> dict | None:
        if self.steering_vector is not None and self.steering_vector.meta:
            return dict(self.steering_vector.meta)
        return None

    def bind(self, ctx: "TransformContext") -> "HeadAdditiveTransform":
        if self.is_bound:
            return self
        return HeadAdditiveTransform(ctx.resolve(self._source), active_heads=self.active_heads, strength=self.strength)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        """Layers the transform can act on: layers with directions and active heads."""
        if self.steering_vector is None:
            return None
        return {
            layer_id
            for layer_id in self.steering_vector.directions
            if self.active_heads.get(layer_id)
        }

    def to_config(self) -> tuple[dict, object | None, "BaseTransform | None"]:
        """The `(params, artifact, inner)` serialized form under the `head_additive` kind.

        `active_heads` serializes with string layer keys and sorted head lists.
        """
        params = {
            "strength": float(self.strength),
            "active_heads": {str(layer): sorted(int(h) for h in heads)
                             for layer, heads in self.active_heads.items()},
        }
        artifact = self.steering_vector if self.steering_vector is not None else self._source
        return params, artifact, None

    @classmethod
    def from_config(cls, params: dict, *, artifact=None, inner=None) -> "HeadAdditiveTransform":
        """Rebuild a `head_additive` transform from its serialized form."""
        active_heads = {int(layer): set(heads) for layer, heads in params.get("active_heads", {}).items()}
        return cls(artifact, active_heads=active_heads, strength=params.get("strength", 1.0))

    def export(self, layer_id: int) -> "WireForm | None":
        """The `head_additive` wire form for `layer_id`.

        The wire vector is `[num_heads, head_dim]` with zeros at heads outside `active_heads`,
        so the broadcast wire addition reproduces the selective per-head addition exactly.
        """
        from ..specs import WireForm

        if self.steering_vector is None:
            return None
        heads = self.active_heads.get(layer_id)
        dirs = self.steering_vector.directions.get(layer_id)
        if not heads or dirs is None:
            return None
        vector = torch.zeros(self.num_heads, self.head_dim, dtype=dirs.dtype)
        for head_id in heads:
            vector[head_id] = dirs[head_id]
        return WireForm(
            kind="head_additive",
            params={"strength": float(self.strength)},
            tensors={"vector": vector},
        )


    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply head-level additive steering.

        Args:
            hidden_states: Shape [B, T, H] where H = num_heads * head_dim.
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        heads = self.active_heads.get(layer_id)
        if not heads:
            return hidden_states

        dirs = self.steering_vector.directions.get(layer_id)
        if dirs is None:
            return hidden_states

        hidden_states = hidden_states.clone()

        for head_id in heads:
            start = head_id * self.head_dim
            end = start + self.head_dim
            direction = dirs[head_id]  # [head_dim]
            v = (self.strength * direction).to(dtype=hidden_states.dtype, device=hidden_states.device)
            # scale by token mask so unmasked positions are untouched
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            hidden_states[:, :, start:end] = hidden_states[:, :, start:end] + delta

        return hidden_states
