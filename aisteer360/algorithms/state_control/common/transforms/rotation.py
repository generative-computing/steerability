"""Angular (rotational) activation steering transform."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from ..specs import WireForm
    from .context import TransformContext

RotationMode = Literal["target", "offset"]


class RotationTransform(BaseTransform):
    """Rotates hidden states within a per-layer 2D steering plane.

    Given an orthonormal basis pair `(b1, b2)` per layer, decompose each hidden state into its
    in-plane coordinates `(c1, c2) = (h·b1, h·b2)` and an out-of-plane remainder
    `h_perp = h - c1 b1 - c2 b2`, rotate the in-plane part, then reconstruct. The out-of-plane
    component is untouched; because a 2D rotation is orthogonal the in-plane magnitude (hence the
    full activation norm) is preserved by construction.

    Modes:

    - `"target"`: rotate the in-plane component TO the absolute angle `angle` measured from `b1`
        (paper Eq. 2). Sweeping over `[0, 2*pi)` traces the behavior circle.
    - `"offset"`: rotate the in-plane component BY `angle` (the same rotation for every token);
        recovers vector-addition / ablation as special cases.

    The stored basis need not be orthonormal; it is re-orthonormalized via Gram-Schmidt and cached
    per `(layer_id, device, dtype)`. Only positions selected by `token_mask` are modified; at every
    other position the input is returned unchanged.

    Args:
        artifact: The steering artifact whose per-layer directions are `[2, H]` (row 0 = feature
            axis, row 1 = companion axis), given as a `SteeringVector`, a per-layer directions
            mapping, or an `ArtifactSource` (unbound until `bind(ctx)`). Required.
        angle: Rotation angle in radians. Interpreted as an absolute target angle in `"target"`
            mode and as a signed offset in `"offset"` mode.
        mode: Either `"target"` or `"offset"`.

    Raises:
        ValueError: If any layer's directions are not shaped `[2, H]`, or if `mode` is invalid.

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
      Hieu M. Vu, Tan M. Nguyen
      [https://arxiv.org/abs/2510.26243](https://arxiv.org/abs/2510.26243)
    """

    wire_kind: ClassVar[str | None] = "rotation"

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        angle: float = 0.0,
        mode: RotationMode = "target",
    ):
        if mode not in ("target", "offset"):
            raise ValueError(f"mode must be 'target' or 'offset'; got {mode!r}.")

        self.angle = float(angle)
        self.mode = mode
        self._source: ArtifactSource | None = None
        self.steering_vector: SteeringVector | None = None
        self._basis_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.steering_vector = artifact
            self._validate_artifact()
        elif isinstance(artifact, Mapping):
            self.steering_vector = SteeringVector(model_type="unknown", directions=dict(artifact))
            self._validate_artifact()
        else:
            raise TypeError(
                f"RotationTransform artifact must be a SteeringVector, a Mapping[int, Tensor], or an "
                f"ArtifactSource; got {type(artifact).__name__} (did you mean angle=?)."
            )

    def _validate_artifact(self) -> None:
        """Validate the per-layer directions are `[2, H]` basis pairs."""
        for layer_id, dirs in self.steering_vector.directions.items():
            if dirs.ndim != 2 or dirs.size(0) != 2:
                raise ValueError(
                    f"RotationTransform expects [2, H] directions (basis pair); "
                    f"got shape {tuple(dirs.shape)} at layer {layer_id}."
                )

    @property
    def is_bound(self) -> bool:
        return self.steering_vector is not None

    @property
    def artifact_meta(self) -> dict | None:
        if self.steering_vector is not None and self.steering_vector.meta:
            return dict(self.steering_vector.meta)
        return None

    def bind(self, ctx: "TransformContext") -> "RotationTransform":
        if self.is_bound:
            return self
        return RotationTransform(ctx.resolve(self._source), angle=self.angle, mode=self.mode)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.steering_vector.directions.keys()) if self.steering_vector is not None else None


    def export(self, layer_id: int) -> "WireForm | None":
        """The `rotation` wire form for `layer_id` (angle, mode, and the `[2, H]` basis)."""
        from ..specs import WireForm

        if self.steering_vector is None:
            return None
        basis = self.steering_vector.directions.get(layer_id)
        if basis is None:
            return None
        return WireForm(
            kind="rotation",
            params={"angle": float(self.angle), "mode": self.mode},
            tensors={"basis": basis},
        )


    def _basis(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached orthonormal `(b1, b2)` for a layer, computing it on first use."""
        key = (layer_id, device, dtype)
        cached = self._basis_cache.get(key)
        if cached is None:
            raw = self.steering_vector.directions[layer_id].to(device=device, dtype=dtype)
            b1 = raw[0] / (raw[0].norm() + 1e-8)
            b2 = raw[1] - (raw[1] @ b1) * b1  # Gram-Schmidt
            b2 = b2 / (b2.norm() + 1e-8)
            cached = (b1, b2)
            self._basis_cache[key] = cached
        return cached

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Rotate the in-plane component of each masked hidden state.

        Args:
            hidden_states: Shape `[B, T, H]`.
            layer_id: Which layer this is being applied at.
            token_mask: Shape `[B, T]`. True at positions to rotate.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        if layer_id not in self.steering_vector.directions:
            return hidden_states

        b1, b2 = self._basis(layer_id, hidden_states.device, hidden_states.dtype)

        c1 = hidden_states @ b1  # [B, T]
        c2 = hidden_states @ b2  # [B, T]

        if self.mode == "target":
            delta = self.angle - torch.atan2(c2, c1)  # per-token, to hit the absolute angle
            cos_d, sin_d = torch.cos(delta), torch.sin(delta)
        else:
            cos_d = hidden_states.new_tensor(math.cos(self.angle))
            sin_d = hidden_states.new_tensor(math.sin(self.angle))

        c1_new = cos_d * c1 - sin_d * c2
        c2_new = sin_d * c1 + cos_d * c2

        # h' = h + (c1_new - c1) b1 + (c2_new - c2) b2
        delta_c1 = (c1_new - c1).unsqueeze(-1)
        delta_c2 = (c2_new - c2).unsqueeze(-1)
        rotated = hidden_states + delta_c1 * b1 + delta_c2 * b2

        return torch.where(token_mask.unsqueeze(-1), rotated, hidden_states)
