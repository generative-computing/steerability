"""Weight-SVD S-space steering."""
import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.state_control.base import HookControl

from .args import SSpaceArgs


def fit_sspace(
    weight: torch.Tensor, bias: torch.Tensor | None, positive: torch.Tensor, negative: torch.Tensor,
    rank: int = -1,
) -> dict[str, torch.Tensor]:
    """Fit an S-space artifact from `[N, out_features]` Linear outputs."""
    weight = weight.detach().float().cpu()
    positive = positive.detach().float().cpu()
    negative = negative.detach().float().cpu()
    if (positive.ndim != 2 or negative.ndim != 2 or not len(positive) or not len(negative)
            or positive.shape[1] != weight.shape[0] or negative.shape[1] != weight.shape[0]):
        raise ValueError("calibration outputs must be nonempty [N, out_features]")
    u, s, _ = torch.linalg.svd(weight, full_matrices=False)
    sqrt_s = s.sqrt()
    if (sqrt_s <= 0).any():
        raise ValueError("S-space requires nonzero singular values")
    b = torch.zeros(weight.shape[0]) if bias is None else bias.detach().float().cpu()
    pos_s, neg_s = ((positive - b) @ u) / sqrt_s, ((negative - b) @ u) / sqrt_s
    contrast = pos_s.mean(0) - neg_s.mean(0)
    r = min(weight.shape) if rank < 0 else min(rank, min(weight.shape))
    indices = contrast.abs().topk(r).indices.sort().values
    direction = contrast[indices]
    direction = direction / (direction.norm() + 1e-8)
    return {"u": u[:, indices].contiguous(), "sqrt_s": sqrt_s[indices].contiguous(),
            "bias": b.clone(), "directions": direction.unsqueeze(0).contiguous()}


def apply_sspace(
    output: torch.Tensor, artifact: dict[str, torch.Tensor], strength: float = 1.0, gate: str = "cosine",
) -> torch.Tensor:
    """Apply cosine-scaled direction rows, using float32 for fp16/bf16 inputs."""
    if strength == 0:
        return output
    dtype = torch.float32 if output.dtype in (torch.float16, torch.bfloat16) else output.dtype
    with torch.autocast(output.device.type, enabled=False):
        values = output.to(dtype)
        u = artifact["u"].to(values)
        sqrt_s = artifact["sqrt_s"].to(values)
        directions = artifact["directions"].to(values)
        if gate == "off":
            delta = directions.sum(0)
        elif gate in ("cosine", "signed"):
            projected = ((values - artifact["bias"].to(values)) @ u) / sqrt_s
            projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
            amplitudes = directions.norm(dim=-1)
            unit = directions / (amplitudes.unsqueeze(-1) + 1e-8)
            cosine = projected @ unit.T
            engagement = cosine.abs() if gate == "cosine" else cosine
            delta = (engagement * amplitudes) @ unit
        else:
            raise ValueError("gate must be cosine, signed, or off")
        return (values + strength * (delta * sqrt_s) @ u.T).to(output.dtype)


class SSpace(HookControl):
    """Apply weight-SVD steering to named Linear outputs.

    The weight SVD defines scaled coordinates for positive and negative outputs.
    The fit retains coordinates with the largest absolute mean contrast, then
    normalizes the contrast direction. Inference uses the retained basis to read
    each token and reconstruct its edit. `cosine` uses absolute cosine similarity;
    `signed` keeps its sign and reinforces either pole at positive strength;
    `off` applies a constant edit. Model weights stay fixed.

    Reference:

        - Michael J Clark, "S-Space Steering for Eval-Awareness Control in Reasoning Models", Apart Research project.
          https://apartresearch.com/project/sspace-steering-for-evalawareness-control-in-reasoning-models-7j1i
        - Implementation: wassname, steering-lite `sspace.py` at `0a064ba`.
          https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/sspace.py
    """

    Args = SSpaceArgs
    supports_batching = True

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model: torch.nn.Module, tokenizer=None, **kwargs) -> None:
        names = self.positive_outputs if self.artifacts is None else self.artifacts
        self.fitted = {}
        for name in names:
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{name}: expected torch.nn.Linear with [out, in] weight")
            if self.artifacts is None:
                artifact = fit_sspace(
                    module.weight, module.bias, self.positive_outputs[name], self.negative_outputs[name], self.rank,
                )
            else:
                artifact = {key: value.detach().cpu().clone() for key, value in self.artifacts[name].items()}
            u, sqrt_s, directions, bias = (artifact[k] for k in ("u", "sqrt_s", "directions", "bias"))
            if (u.ndim != 2 or u.shape[0] != module.out_features or sqrt_s.shape != (u.shape[1],)
                    or directions.ndim != 2 or directions.shape[1] != u.shape[1]
                    or bias.shape != (module.out_features,) or (sqrt_s <= 0).any()
                    or not all(torch.isfinite(v).all() for v in artifact.values())):
                raise ValueError(f"{name}: invalid S-space artifact dimensions or values")
            self.fitted[name] = artifact

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None = None, **kwargs) -> dict:
        hooks = []
        for name, artifact in self.fitted.items():
            runtime = {}

            def hook(module, inputs, kwargs, output, artifact=artifact, runtime=runtime):
                if self.strength == 0:
                    return output
                dtype = torch.float32 if output.dtype in (torch.float16, torch.bfloat16) else output.dtype
                key = (output.device, dtype)
                if key not in runtime:
                    runtime[key] = {k: v.to(device=output.device, dtype=dtype) for k, v in artifact.items()}
                return apply_sspace(output, runtime[key], self.strength, self.gate)

            hooks.append({"module": name, "hook_func": hook})
        return {"pre": [], "forward": hooks, "backward": []}

    def steer_fits(self) -> tuple[tuple[str, str], ...]:
        return () if self.artifacts is not None else (("sspace", "calibrated"),)

    def export_state(self) -> dict[str, torch.Tensor]:
        return {f"{name}:{key}": value for name, artifact in self.fitted.items() for key, value in artifact.items()}

    def export_state_classes(self) -> dict[str, str]:
        return {name: "calibrated" for name in self.export_state()}

    def frozen_form(self, state: dict[str, torch.Tensor]) -> tuple[str, dict]:
        artifacts = {name: {key: state[f"{name}:{key}"] for key in artifact} for name, artifact in self.fitted.items()}
        return "state_control/sspace", {"artifacts": artifacts, "strength": self.strength, "gate": self.gate}

    def fit_identity(self) -> tuple | None:
        if self.artifacts is not None:
            return None
        return self.positive_outputs, self.negative_outputs, self.rank
