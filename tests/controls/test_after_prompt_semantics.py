"""Cross-control behavioral test for `token_scope="after_prompt"` position tracking.

This is the anti-drift guard for Issue 3 (KV-cache position heuristic). The four scoped
controls (CAA, ITI, AngularSteering, DirectionalAblation) share one position-tracking
implementation (`TransformHookRuntime`); this parametrized test pins the semantics for all of
them, and any new runtime client can be added to the parameter list.

The bug it guards against: inferring the prefill/decode phase by comparing `seq_len` to the
prompt length silently disables steering for length-1 prompts (prefill and decode become
indistinguishable). With explicit first-call tracking, steering must fire on every generated
position regardless of prompt length.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.angular_steering.control import AngularSteering
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.directional_ablation.control import DirectionalAblation
from aisteer360.algorithms.state_control.iti.control import ITI
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4
HEAD_DIM = HIDDEN // HEADS


class _PassTracker:
    """Counts model forward passes and, per pass, whether steering fired.

    A control may register several hooks per pass (one per layer/norm module) and even reuse a
    `layer_id` across distinct modules, so pass boundaries cannot be inferred from the transform
    alone. Instead we bump `pass_index` from a forward-pre-hook on the model itself, and the
    recording transform reports True-mask activity against the current pass.
    """

    def __init__(self):
        self.pass_index = -1
        self._steered_in_pass: dict[int, bool] = {}

    def begin_pass(self, *_):
        self.pass_index += 1

    def mark_steered(self):
        self._steered_in_pass[self.pass_index] = True

    @property
    def total_passes(self) -> int:
        return self.pass_index + 1

    @property
    def steered_passes(self) -> int:
        return sum(1 for fired in self._steered_in_pass.values() if fired)


class _RecordingTransform:
    """Wraps a transform; flags the current model pass as steered when a True mask arrives."""

    def __init__(self, inner, tracker: _PassTracker):
        self._inner = inner
        self._tracker = tracker

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        if bool(token_mask.any()):
            self._tracker.mark_steered()
        return self._inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs)


def _make_caa():
    g = torch.Generator().manual_seed(1)
    sv = SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(1, HIDDEN, generator=g) for lid in range(LAYERS)},
    )
    return CAA(steering_vector=sv, layer_id=1, multiplier=1.0, token_scope="after_prompt")


def _make_iti():
    g = torch.Generator().manual_seed(2)
    sv = SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(HEADS, HEAD_DIM, generator=g) for lid in range(LAYERS)},
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        probe_accuracies={(lid, h): 0.9 for lid in range(LAYERS) for h in range(HEADS)},
    )
    return ITI(
        steering_vector=sv,
        selected_heads=[(1, 0), (1, 1)],
        alpha=1.0,
        token_scope="after_prompt",
    )


def _make_angular():
    # per-layer orthonormal [2, H] basis pair
    g = torch.Generator().manual_seed(3)
    directions = {}
    for lid in range(LAYERS):
        raw = torch.randn(2, HIDDEN, generator=g)
        b1 = raw[0] / raw[0].norm()
        b2 = raw[1] - (raw[1] @ b1) * b1
        b2 = b2 / b2.norm()
        directions[lid] = torch.stack([b1, b2], dim=0)
    sv = SteeringVector(model_type="llama", directions=directions)
    return AngularSteering(steering_vector=sv, target_degree=90.0, token_scope="after_prompt")


def _make_ablation():
    g = torch.Generator().manual_seed(4)
    sv = SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(1, HIDDEN, generator=g) for lid in range(LAYERS)},
    )
    return DirectionalAblation(
        steering_vector=sv, alpha=1.0, layer_ids=[1], token_scope="after_prompt"
    )


CONTROL_FACTORIES = {
    "caa": _make_caa,
    "iti": _make_iti,
    "angular": _make_angular,
    "ablation": _make_ablation,
}


@pytest.mark.parametrize("control_name", list(CONTROL_FACTORIES))
@pytest.mark.parametrize("prompt_len", [1, 4])
def test_after_prompt_steers_every_generated_position(control_name, prompt_len):
    """Steering fires on exactly `max_new_tokens - 1` decode passes for prompt lengths {1, 4}.

    The final generated token is emitted but never re-processed by the model, so no hook fires
    for it. This is structural (verified with `eos_token_id=None`, no early stopping), hence the
    expected count is `max_new_tokens - 1`, not `max_new_tokens`.
    """
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()

    control = CONTROL_FACTORIES[control_name]()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    # count model forward passes via a pre-hook, and steered passes via a recording transform
    tracker = _PassTracker()
    control._transform = _RecordingTransform(control._transform, tracker)
    handle = model.register_forward_pre_hook(tracker.begin_pass)

    input_ids = torch.arange(3, 3 + prompt_len, dtype=torch.long).unsqueeze(0)
    max_new_tokens = 5

    try:
        out = pipeline.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=None,
        )
    finally:
        handle.remove()

    assert out.size(1) >= 1
    # prefill + (max_new_tokens - 1) decode passes see a hook fire; the final generated token is
    # emitted but never re-processed, so one fewer decode pass runs than tokens generated
    assert tracker.total_passes == max_new_tokens, (
        f"{control_name} (prompt_len={prompt_len}): expected {max_new_tokens} total passes, "
        f"got {tracker.total_passes}"
    )
    # after_prompt must steer on every decode pass (all but the prefill pass)
    assert tracker.steered_passes == max_new_tokens - 1, (
        f"{control_name} (prompt_len={prompt_len}): expected {max_new_tokens - 1} steered passes, "
        f"got {tracker.steered_passes}"
    )
