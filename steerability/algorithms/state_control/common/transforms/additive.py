"""Additive activation steering transform."""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from ..specs import WireForm
    from .context import TransformContext


def _canonical_directions(directions: Mapping[int, torch.Tensor], positional: bool) -> dict[int, torch.Tensor]:
    """Canonicalize per-layer directions to `[T, H]`, validating shape against the mode.

    A 1-D `[H]` tensor becomes `[1, H]`. Broadcast mode (`positional=False`) requires
    `T == 1` at every layer.

    Raises:
        ValueError: If a direction is not 1-D or 2-D, or has `T > 1` while `positional` is
            False.
    """
    canonical: dict[int, torch.Tensor] = {}
    for layer_id, direction in directions.items():
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)
        if direction.ndim != 2:
            raise ValueError(
                f"AdditiveTransform direction for layer {layer_id} must be [H] or [T, H]; got "
                f"shape {tuple(direction.shape)}."
            )
        if not positional and direction.size(0) != 1:
            raise ValueError(
                f"AdditiveTransform direction for layer {layer_id} has T={direction.size(0)} rows "
                f"but positional=False; broadcast semantics are defined for single-row directions "
                f"only. Construct the transform with positional=True for per-position injection."
            )
        canonical[int(layer_id)] = direction
    return canonical


class AdditiveTransform(BaseTransform):
    """Adds scaled direction vector(s) to hidden states.

    Two modes, selected by the `positional` flag:

    Broadcast (default, e.g., CAA):
        ``h'[pos] = h[pos] + mask[pos] * strength * direction[0]``

        The same `[1, H]` vector is added at every masked position. The ``alignment``
        parameter is unused.

    Positional (e.g., ActAdd):
        ``h'[p] = h[p] + mask[p] * strength * direction[p - alignment]``

        Row `t` of a `[T, H]` direction is added at absolute sequence position
        ``alignment + t``. The window ``[alignment, alignment + T)`` is evaluated in
        absolute positions, so during KV-cached generation a window inside the prompt
        fires only on the prefill pass, and a window extending past the prompt fires at
        exactly the covered generated positions, once each.

    Direction tensors are canonicalized to `[T, H]` at construction or bind (a 1-D `[H]`
    tensor becomes `[1, H]`). A direction with `T > 1` requires `positional=True`, since
    broadcast semantics are undefined for multi-row directions.

    Args:
        artifact: The steering artifact, given as a `SteeringVector`, a per-layer directions
            mapping (`Mapping[int, Tensor]`, each `[T, H]` or `[H]`), or an `ArtifactSource`
            (unbound until `bind(ctx)`). Required.
        strength: Global scaling factor.
        alignment: Absolute token position of the first direction row (default: 0). Used
            only when `positional` is True.
        positional: Selects per-position injection. When False (default), the direction
            broadcasts over masked positions and every layer's direction must have `T == 1`.

    Raises:
        ValueError: If a concrete direction has `T > 1` while `positional` is False (raised
            at construction for concrete artifacts, at `bind` for sources).
        TypeError: If `artifact` is not a supported type.
    """

    wire_kind: ClassVar[str | None] = "additive"

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        strength: float = 1.0,
        alignment: int = 0,
        positional: bool = False,
    ):
        self.strength = strength
        self.alignment = alignment
        self.positional = positional
        self._source: ArtifactSource | None = None
        self.directions: dict[int, torch.Tensor] | None = None

        self._artifact_meta: dict | None = None
        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.directions = _canonical_directions(artifact.directions, positional)
            self._artifact_meta = dict(artifact.meta) if artifact.meta else None
        elif isinstance(artifact, Mapping):
            self.directions = _canonical_directions(artifact, positional)
        else:
            raise TypeError(
                f"AdditiveTransform artifact must be a SteeringVector, a Mapping[int, Tensor], or an "
                f"ArtifactSource; got {type(artifact).__name__} (did you mean strength=?)."
            )

    @property
    def is_bound(self) -> bool:
        return self.directions is not None

    @property
    def artifact_meta(self) -> dict | None:
        return self._artifact_meta

    def bind(self, ctx: "TransformContext") -> "AdditiveTransform":
        if self.is_bound:
            return self
        return AdditiveTransform(
            ctx.resolve(self._source),
            strength=self.strength,
            alignment=self.alignment,
            positional=self.positional,
        )

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.directions.keys()) if self.directions is not None else None

    def to_config(self) -> tuple[dict, object | None, "BaseTransform | None"]:
        """The `(params, artifact, inner)` serialized form under the `additive` kind.

        The artifact slot carries the concrete directions as a `SteeringVector` when bound,
        or the unresolved source when not.
        """
        params = {"strength": float(self.strength), "alignment": int(self.alignment),
                  "positional": bool(self.positional)}
        if self.directions is not None:
            artifact = SteeringVector(
                model_type="unknown", directions=dict(self.directions), meta=dict(self._artifact_meta or {}),
            )
        else:
            artifact = self._source
        return params, artifact, None

    @classmethod
    def from_config(cls, params: dict, *, artifact=None, inner=None) -> "AdditiveTransform":
        """Rebuild an `additive` transform from its serialized form."""
        return cls(
            artifact,
            strength=params.get("strength", 1.0),
            alignment=params.get("alignment", 0),
            positional=params.get("positional", False),
        )

    def wire_plan(self) -> str | None:
        """`"additive"` for broadcast transforms; None when `positional` is True.

        Positional injection has no wire form regardless of `T`.
        """
        return None if self.positional else "additive"

    def export(self, layer_id: int) -> "WireForm | None":
        """The `additive` wire form for `layer_id`, or None for positional transforms.

        Semantics are defined for broadcast directions only, where every steered token
        receives the same vector.
        """
        from ..specs import WireForm

        if self.positional or self.directions is None:
            return None
        direction = self.directions.get(layer_id)
        if direction is None:
            return None
        return WireForm(
            kind="additive",
            params={"strength": float(self.strength)},
            tensors={"vector": direction.squeeze(0)},
        )


    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        position_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Apply additive steering.

        Args:
            hidden_states: Shape [B, T_seq, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T_seq]. True at positions to modify.
            position_offset: Absolute position of the pass's first token. Consumed in
                positional mode, where the local slice covering absolute positions
                `[alignment, alignment + T)` receives the direction rows; ignored in
                broadcast mode.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        direction = self.directions.get(layer_id)
        if direction is None:
            return hidden_states

        if not self.positional:
            # broadcast mode (e.g., CAA); same vector at all masked positions
            v = (self.strength * direction.squeeze(0)).to(
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            return hidden_states + delta

        # positional mode (e.g., ActAdd); the pass covers absolute positions
        # [position_offset, position_offset + seq_len), the window covers [a, a + T)
        seq_len = hidden_states.size(1)
        T_steer = direction.size(0)
        a = self.alignment

        window_start = max(a, position_offset)
        window_end = min(a + T_steer, position_offset + seq_len)
        if window_start >= window_end:
            return hidden_states

        local_start = window_start - position_offset
        local_end = window_end - position_offset
        vec_start = window_start - a
        vec_end = vec_start + (window_end - window_start)

        v = (self.strength * direction[vec_start:vec_end]).to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )  # [inject_len, H]

        mask_slice = token_mask[:, local_start:local_end]  # [B, inject_len]
        gated_v = mask_slice.unsqueeze(-1).to(hidden_states.dtype) * v.unsqueeze(0)

        # add in-place at the injection slice
        out = hidden_states.clone()
        out[:, local_start:local_end] += gated_v
        return out
