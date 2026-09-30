"""Coordinate-wise affine activation transport."""
import torch

from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.specs import Intervention, TokenScope
from steerability.algorithms.state_control.common.transforms.base import BaseTransform
from steerability.algorithms.state_control.common.transforms.context import TransformContext

from .args import LinearAcTArgs


def fit_linear_act(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Return `[slope, bias]` from coordinate-wise sorted least squares."""
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[0] < 2:
        raise ValueError("Linear-AcT needs equal [N, hidden_size] shapes with N >= 2")
    source = negative.detach().float().cpu().sort(dim=0).values
    target = positive.detach().float().cpu().sort(dim=0).values
    mean_source = source.mean(0)
    mean_target = target.mean(0)
    source_centered = source - mean_source
    target_centered = target - mean_target
    denominator = source_centered.square().sum(0)
    if (denominator <= 0).any():
        raise ValueError("Linear-AcT source coordinates must have nonzero variance")
    slope = (source_centered * target_centered).sum(0) / denominator
    return torch.stack((slope, mean_target - slope * mean_source))


class _LinearAcTTransform(BaseTransform):
    def __init__(
        self, affine: dict[int, torch.Tensor] | None,
        positive: dict[int, torch.Tensor] | None = None,
        negative: dict[int, torch.Tensor] | None = None, strength: float = 1.0,
    ) -> None:
        self.affine = affine
        self.positive = positive
        self.negative = negative
        self.strength = strength
        self._validated = False
        self._runtime_affine = {}

    @property
    def is_bound(self) -> bool:
        return self._validated

    @property
    def covered_layer_ids(self) -> set[int]:
        return set(self.affine if self.affine is not None else self.positive)

    def bind(self, ctx: TransformContext) -> "_LinearAcTTransform":
        if self.affine is not None:
            affine = {lid: value.detach().cpu().clone() for lid, value in self.affine.items()}
        else:
            affine = {lid: fit_linear_act(pos, self.negative[lid]) for lid, pos in self.positive.items()}
        for lid, value in affine.items():
            if value.shape != (2, ctx.hidden_size) or not torch.isfinite(value).all():
                raise ValueError(f"layer {lid}: affine must be finite [2, hidden_size]")
        bound = _LinearAcTTransform(affine, strength=self.strength)
        bound._runtime_affine = {
            (lid, ctx.device, ctx.dtype): value.to(device=ctx.device, dtype=ctx.dtype)
            for lid, value in affine.items()
        }
        bound._validated = True
        return bound

    def apply(
        self, hidden_states: torch.Tensor, *, layer_id: int, token_mask: torch.BoolTensor, **kwargs,
    ) -> torch.Tensor:
        self._require_bound()
        if self.strength == 0:
            return hidden_states
        key = (layer_id, hidden_states.device, hidden_states.dtype)
        if key not in self._runtime_affine:
            self._runtime_affine[key] = self.affine[layer_id].to(hidden_states)
        slope, bias = self._runtime_affine[key]
        transported = (1 - self.strength) * hidden_states + self.strength * (hidden_states * slope + bias)
        return torch.where(token_mask.unsqueeze(-1), transported, hidden_states)


class LinearAcT(InterventionControl):
    """Transport decoder-output coordinates with sorted affine least squares.

    Positive activations are the target and negative activations are the source.
    Sorting is independent per coordinate, so fitting requires equal sample counts
    but does not use row pairing. Inference needs only slope and bias vectors; the
    map is interpolated with identity by `strength` and applied to every token.
    Model weights stay fixed. Constant source coordinates raise.

    Reference:

        - Rodriguez et al., "Controlling Language and Diffusion Models by Transporting Activations", ICLR 2025.
          https://openreview.net/forum?id=l2zFn6TIQi
        - Implementation: wassname, steering-lite `linear_act.py` at `0a064ba`.
          https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/linear_act.py
    """

    Args = LinearAcTArgs
    hook_only_hint = "Linear-AcT affine transport has no wire form; use the huggingface backend"

    def _configure(self) -> None:
        transform = _LinearAcTTransform(self.affine, self.positive_activations, self.negative_activations, self.strength)
        self._template = (Intervention(layers=tuple(sorted(transform.covered_layer_ids)), transform=transform,
                                       scope=TokenScope("all")),)

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        return () if self.affine is not None else (("linear_act", "calibrated"),)

    def export_state(self) -> dict[str, torch.Tensor]:
        if not self.interventions:
            return {}
        return {str(lid): value for lid, value in self.interventions[0].transform.affine.items()}

    def export_state_classes(self) -> dict[str, str]:
        return {name: "calibrated" for name in self.export_state()}

    def frozen_form(self, state: dict[str, torch.Tensor]) -> tuple[str, dict]:
        return "state_control/linear_act", {"affine": {int(lid): value for lid, value in state.items()},
                                            "strength": self.strength}

    def fit_identity(self) -> tuple | None:
        if self.affine is not None:
            return None
        return self.positive_activations, self.negative_activations
