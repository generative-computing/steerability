"""Projection transform: projects learned directions out of the residual stream."""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from ..specs import WireForm
    from .context import TransformContext


class ProjectionTransform(BaseTransform):
    """Removes one or more learned directions from hidden states by projection.

    For an orthonormal set of directions `{d_1..d_k}` at a layer (rows of a `[K, H]` tensor):

        h' = h - alpha * sum_i (h . d_i) d_i          (applied at masked positions)

    `K=1` is single-direction ablation (the abliteration / directional-ablation technique of
    Arditi et al.); `K>1` ablates the whole subspace. `alpha` in `[0, 1]` scales the removal:
    `1.0` fully removes the component (`h'.d_i == 0`), values `< 1.0` give graded partial
    suppression.

    This is a projection, not a translation or a rotation:

    - It is idempotent at `alpha=1` (`P^2 = P`).
    - It is norm-reducing, since it drops a component.

    The stored rows need not be orthonormal; they are orthonormalized via Gram-Schmidt and cached
    per `(layer_id, device, dtype)` so `K>1` removal is basis-correct and order-independent.

    Args:
        artifact: The steering artifact, given as a `SteeringVector`, a per-layer directions
            mapping (`Mapping[int, Tensor]`, each `[K, H]` or `[H]` treated as `K=1`), or an
            `ArtifactSource` (unbound until `bind(ctx)`). Required.
        alpha: Ablation strength in `[0, 1]`. `1.0` = full removal (default); `< 1.0` = partial.

    Reference:

    - "Refusal in Language Models Is Mediated by a Single Direction"
      Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee,
      Neel Nanda
      [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
    """

    wire_kind: ClassVar[str | None] = "projection"

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        alpha: float = 1.0,
    ):
        self.alpha = float(alpha)
        self._source: ArtifactSource | None = None
        self.directions: dict[int, torch.Tensor] | None = None
        self._basis_cache: dict[tuple, torch.Tensor] = {}  # (layer_id, device, dtype) -> [K, H] orthonormal

        self._artifact_meta: dict | None = None
        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.directions = artifact.directions
            self._artifact_meta = dict(artifact.meta) if artifact.meta else None
        elif isinstance(artifact, Mapping):
            self.directions = dict(artifact)
        else:
            raise TypeError(
                f"ProjectionTransform artifact must be a SteeringVector, a "
                f"Mapping[int, Tensor], or an ArtifactSource; got {type(artifact).__name__} "
                f"(did you mean alpha={artifact!r}?)."
            )

    @property
    def is_bound(self) -> bool:
        return self.directions is not None

    @property
    def artifact_meta(self) -> dict | None:
        return self._artifact_meta

    def bind(self, ctx: "TransformContext") -> "ProjectionTransform":
        if self.is_bound:
            return self
        return ProjectionTransform(ctx.resolve(self._source), alpha=self.alpha)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.directions.keys()) if self.directions is not None else None


    def wire_plan(self) -> str | None:
        """`"projection"` for single-direction full removal; None otherwise.

        The wire kind removes a single direction's component in full, so only `K == 1`
        directions at `alpha == 1.0` serialize; subspace ablation (`K > 1`) and graded
        removal (`alpha < 1.0`) are hook-only.
        """
        if self.alpha != 1.0:
            return None
        if self.directions is not None and any(
            direction.ndim == 2 and direction.size(0) > 1 for direction in self.directions.values()
        ):
            return None
        return "projection"

    def export(self, layer_id: int) -> "WireForm | None":
        """The `projection` wire form for `layer_id`, or None when the configuration is
        hook-only (`K > 1` or `alpha != 1.0`)."""
        from ..specs import WireForm

        if self.directions is None or self.alpha != 1.0:
            return None
        direction = self.directions.get(layer_id)
        if direction is None:
            return None
        if direction.ndim == 2:
            if direction.size(0) != 1:
                return None
            direction = direction.squeeze(0)
        return WireForm(kind="projection", tensors={"vector": direction})


    def _basis(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the cached orthonormal `[K, H]` basis for a layer, computing it on first use."""
        key = (layer_id, device, dtype)
        cached = self._basis_cache.get(key)
        if cached is None:
            raw = self.directions[layer_id].to(device=device, dtype=dtype)
            if raw.ndim == 1:
                raw = raw.unsqueeze(0)  # [H] -> [1, H]

            # Gram-Schmidt -> orthonormal rows (drop near-zero / dependent rows)
            basis_rows: list[torch.Tensor] = []
            for row in raw:
                v = row.clone()
                for b in basis_rows:
                    v = v - (v @ b) * b
                n = v.norm()
                if n > 1e-8:
                    basis_rows.append(v / n)

            cached = (
                torch.stack(basis_rows, dim=0)
                if basis_rows
                else raw.new_zeros((0, raw.size(-1)))
            )
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
        """Project the learned directions out of each masked hidden state.

        Args:
            hidden_states: Shape `[B, T, H]`.
            layer_id: Which layer this is being applied at.
            token_mask: Shape `[B, T]`. True at positions to ablate.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        if layer_id not in self.directions:
            return hidden_states

        basis = self._basis(layer_id, hidden_states.device, hidden_states.dtype)
        if basis.size(0) == 0:
            return hidden_states

        # coefficients c[..., i] = h . d_i  ->  [B, T, K]
        coeffs = torch.einsum("bth,kh->btk", hidden_states, basis)
        # component to remove: sum_i c_i d_i  ->  [B, T, H]
        component = torch.einsum("btk,kh->bth", coeffs, basis)
        ablated = hidden_states - self.alpha * component

        return torch.where(token_mask.unsqueeze(-1), ablated, hidden_states)
