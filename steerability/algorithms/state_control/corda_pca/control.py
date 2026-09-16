"""CorDA-derived PCA steering."""
import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.state_control.base import HookControl

from .args import CordaPCAArgs


def fit_corda_direction(
    weight: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
    rank: int = -1, damping: float = 0.01, normalize: bool = True,
) -> torch.Tensor:
    """Return an output-space offset from paired `[N, in_features]` inputs."""
    weight = weight.detach().float().cpu()
    positive = positive.detach().float().cpu()
    negative = negative.detach().float().cpu()
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[0] < 2:
        raise ValueError("CorDA PCA needs at least two paired [N, in_features] samples")
    if positive.shape[1] != weight.shape[1]:
        raise ValueError("calibration input width does not match Linear.in_features")
    context = torch.cat((positive, negative))
    lam = context.square().mean() * damping
    if not torch.isfinite(lam) or lam <= 0:
        raise ValueError("CorDA damping lambda must be finite and positive")
    n = context.shape[0]
    weighted = (weight @ context.T) @ context / n + lam * weight
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    r = min(weight.shape) if rank < 0 else min(rank, min(weight.shape))
    u, sqrt_s, v = u[:, :r], singular[:r].sqrt(), vh[:r].T
    b = context / n**0.5
    gram = torch.eye(n) + (b @ b.T) / lam
    v_corda = v / lam - b.T @ torch.linalg.solve(gram, b @ v) / lam.square()
    differences = ((positive - negative) @ v_corda) * sqrt_s
    centered = differences - differences.mean(0)
    _, variation, pcs = torch.linalg.svd(centered, full_matrices=False)
    tolerance = torch.finfo(differences.dtype).eps * max(differences.shape) * differences.norm()
    if variation[0] <= tolerance:
        raise ValueError("CorDA centered paired differences need numerically nonzero variance")
    direction = pcs[0]
    direction = direction * torch.sign(differences.mean(0) @ direction + 1e-8)
    if normalize:
        direction = direction / (direction.norm() + 1e-8)
    return ((direction * sqrt_s) @ u.T).contiguous()


class CordaPCA(HookControl):
    """Add a fixed output offset fitted from paired Linear inputs.

    The weights and pooled positive/negative input second moments define a CorDA
    basis. Centered PCA on paired differences in that basis produces a direction,
    which is mapped back to a fixed output vector. Inference adds the vector to
    every token; it needs no decomposition, and model weights stay fixed.
    `directions` accepts already fitted output vectors instead of calibration inputs.

    Reference:

        - Yang et al., "CorDA: Context-Oriented Decomposition Adaptation of Large Language Models" (2024).
          https://arxiv.org/abs/2406.05223
          This control adapts the decomposition, not the paper's fine-tuning procedure.
        - Implementation: wassname, steering-lite `corda_pca.py` at `0a064ba`.
          https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/corda_pca.py
    """

    Args = CordaPCAArgs
    supports_batching = True

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model: torch.nn.Module, tokenizer=None, **kwargs) -> None:
        names = self.positive_inputs if self.directions is None else self.directions
        self.fitted_directions = {}
        for name in names:
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{name}: expected torch.nn.Linear with [out, in] weight")
            if self.directions is None:
                direction = fit_corda_direction(
                    module.weight, self.positive_inputs[name], self.negative_inputs[name],
                    self.rank, self.damping, self.normalize,
                )
            else:
                direction = self.directions[name].detach().cpu().clone()
            if direction.shape != (module.out_features,) or not torch.isfinite(direction).all():
                raise ValueError(f"{name}: direction must be finite [out_features]")
            self.fitted_directions[name] = direction

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None = None, **kwargs) -> dict:
        hooks = []
        for name, direction in self.fitted_directions.items():
            runtime = {}

            def hook(module, inputs, kwargs, output, direction=direction, runtime=runtime):
                if self.strength == 0:
                    return output
                key = (output.device, output.dtype)
                if key not in runtime:
                    runtime[key] = direction.to(output)
                return output + self.strength * runtime[key]

            hooks.append({"module": name, "hook_func": hook})
        return {"pre": [], "forward": hooks, "backward": []}

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        return () if self.directions is not None else (("corda_pca", "direction"),)

    def export_state(self) -> dict[str, torch.Tensor]:
        return self.fitted_directions

    def export_state_classes(self) -> dict[str, str]:
        return {name: "direction" for name in self.export_state()}

    def frozen_form(self, state: dict[str, torch.Tensor]) -> tuple[str, dict]:
        return "state_control/corda_pca", {"directions": state, "strength": self.strength}

    def fit_identity(self) -> tuple | None:
        if self.directions is not None:
            return None
        return (self.positive_inputs, self.negative_inputs, self.rank, self.damping, self.normalize)
