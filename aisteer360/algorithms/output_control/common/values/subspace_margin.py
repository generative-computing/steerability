"""Candidate value from a linear-probe margin, plus the single-file probe checkpoint loader.

For each candidate token, `SubspaceMarginValue` forwards `prefix + candidate` through the
pipeline's own model via `CandidateForward` and reads the final-layer hidden state `h` at the raw
output boundary; the value is the probe score `h @ w + bias`. This is the value underlying `SASA`.
"""
from __future__ import annotations

import json

import torch

from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.output_control.common.candidate_forward import CandidateForward
from aisteer360.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext


def load_single_file_probe(file_path: str, layer_id: int) -> Probe:
    """Load a single-file probe checkpoint into a `Probe`.

    Accepts two forms: a `.probe` JSON file with `direction` and `midpoint` lists, and a legacy
    `{'wv', 'mu_mu'}` torch tensor checkpoint. Both record a direction and a class-mean midpoint
    and no space metadata; they are assumed fitted at the raw output boundary of the final decoder
    layer over last-token features. The returned probe carries the direction as the weight vector
    of `layer_id`, `bias = -direction . midpoint` (so its score equals the margin
    `direction . (h - midpoint)`), `location="layer_output"`, `pooling="last"`,
    `model_type="unknown"`, and empty `meta` (which disarms fingerprint checks).

    Args:
        file_path: Path to the checkpoint.
        layer_id: Layer id the direction is registered under (the model's final decoder layer).

    Returns:
        The adapted `Probe`.

    Raises:
        ValueError: If the checkpoint is not one of the supported forms.
    """
    if str(file_path).endswith(".probe"):
        with open(file_path) as f:
            payload = json.load(f)
        direction = torch.tensor(payload["direction"], dtype=torch.float32)
        midpoint = torch.tensor(payload["midpoint"], dtype=torch.float32)
    else:
        try:
            loaded = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as err:
            raise ValueError(
                f"Unrecognized probe checkpoint at {file_path!r}; expected a .probe JSON file or "
                "a legacy {'wv', 'mu_mu'} torch checkpoint."
            ) from err
        if not (isinstance(loaded, dict) and "wv" in loaded and "mu_mu" in loaded):
            raise ValueError(
                f"Unrecognized probe checkpoint at {file_path!r}; expected a .probe JSON file or "
                "a legacy {'wv', 'mu_mu'} torch checkpoint."
            )
        direction = loaded["wv"].float()
        midpoint = loaded["mu_mu"].float()

    direction = direction.reshape(-1)
    bias = -float(torch.dot(direction, midpoint.reshape(-1)))
    return Probe(
        model_type="unknown",
        location="layer_output",
        pooling="last",
        layer_ids=[layer_id],
        weights={layer_id: direction},
        bias=bias,
    )


class SubspaceMarginValue(BaseCandidateValue):
    """Per-candidate margin against a fitted single-layer `Probe`.

    The margin of a candidate hidden state `h` is `h @ w + bias`, with `w` the probe's weight
    vector at its single layer and `bias` its calibrated offset (the documented `Probe` score).
    Hidden states are read by `CandidateForward` at the raw output boundary of the final decoder
    layer, so the probe is expected to be fitted at that boundary (`location="layer_output"`,
    `pooling="last"`, final decoder layer).

    This value forwards the pipeline's own model per candidate, so it declares
    `same_model_forwards=True`; the forwards run inside `auxiliary_pass(aligned=True)` (via
    `CandidateForward`), so state-control transforms apply to them at the candidates' true
    positions while condition scoring and gates ignore them.

    Args:
        probe: The fitted single-layer `Probe`.

    Note:
        `scoring_cost="model_forward"` and `supports_batching` is False (the prefix cache tracks a
        single row).
    """

    supports_batching: bool = False
    scoring_cost = "model_forward"
    same_model_forwards: bool = True

    def __init__(self, probe: Probe):
        self.probe = probe
        self._forward: CandidateForward | None = None
        self._aligned: tuple[torch.device, torch.dtype] | None = None
        self._weights: torch.Tensor | None = None

    def _align_probe(self, device: torch.device, dtype: torch.dtype) -> None:
        """Cache the probe weights aligned to (device, dtype); the probe is fixed per generation."""
        if self._aligned != (device, dtype):
            self._weights = self.probe.weights[self.probe.layer_ids[0]].to(device, dtype)
            self._aligned = (device, dtype)

    def score(self, ctx: StepContext) -> torch.Tensor:
        if ctx.model is None:
            raise RuntimeError("SubspaceMarginValue requires the pipeline model in StepContext.")
        if self._forward is None or self._forward.model is not ctx.model:
            self._forward = CandidateForward(ctx.model)
        hidden = self._forward.last_hidden_states(
            ctx.prefix_ids, ctx.candidate_ids, ctx.attention_mask
        )  # [K, H]
        self._align_probe(hidden.device, hidden.dtype)
        margins = hidden @ self._weights + self.probe.bias  # [K]
        return margins.unsqueeze(0)  # [1, K]
