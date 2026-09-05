"""Golden token-id sequences for the `TransformHookRuntime`-backed state controls.

Pins the greedy generations of `ITI`, `AngularSteering`, and `ActAdd` on the hub-free tiny
fixtures, in both position-tracking modes, so any change to position tracking is caught as a
token-level difference rather than a merely approximate one.

The literals are produced by the controls themselves; regenerate with
`STEERABILITY_CAPTURE_GOLDENS=1 pytest tests/controls/test_position_tracking_goldens.py -s` and paste
the printed mapping into `GOLDENS`.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import os

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.act_add.control import ActAdd
from steerability.algorithms.state_control.angular_steering.control import AngularSteering
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.iti.control import ITI
from tests.utils.runtime_helpers import strip_clock
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4
HEAD_DIM = HIDDEN // HEADS
MAX_NEW_TOKENS = 8


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
        selected_heads=[(1, 0), (1, 1), (2, 3)],
        alpha=1.0,
        token_scope="after_prompt",
    )


def _make_angular():
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


def _make_act_add():
    g = torch.Generator().manual_seed(5)
    sv = SteeringVector(
        model_type="llama",
        directions={1: torch.randn(2, HIDDEN, generator=g)},
    )
    return ActAdd(steering_vector=sv, layer_id=1, multiplier=1.0, alignment=1)


CONTROL_FACTORIES = {
    "iti": _make_iti,
    "angular": _make_angular,
    "act_add": _make_act_add,
}

# recorded token-id sequences (control_name, prompt_len) -> list[int]
GOLDENS: dict[tuple[str, int], list[int]] = {
    ("iti", 1): [29, 55, 55, 55, 55, 55, 55, 55],
    ("iti", 4): [29, 55, 55, 55, 55, 55, 55, 55],
    ("angular", 1): [29, 66, 97, 29, 66, 97, 29, 66],
    ("angular", 4): [29, 66, 70, 14, 66, 70, 14, 66],
    ("act_add", 1): [29, 45, 27, 33, 29, 66, 97, 38],
    ("act_add", 4): [29, 66, 70, 91, 10, 82, 10, 95],
}


def _strip_clock_from_hooks(control) -> None:
    """Wrap the control's hook callables so they drop `cache_position` (fallback counting)."""
    original_get_hooks = control.get_hooks

    def stripped_get_hooks(*args, **kwargs):
        hooks = original_get_hooks(*args, **kwargs)
        for phase in ("pre", "forward"):
            for spec in hooks.get(phase, []):
                spec["hook_func"] = strip_clock(spec["hook_func"])
        return hooks

    control.get_hooks = stripped_get_hooks


def _generate(control_name: str, prompt_len: int, strip: bool = False) -> list[int]:
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()

    control = CONTROL_FACTORIES[control_name]()
    if strip:
        _strip_clock_from_hooks(control)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    input_ids = torch.arange(3, 3 + prompt_len, dtype=torch.long).unsqueeze(0)
    out = pipeline.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
    )
    return out[0].tolist()


@pytest.mark.parametrize("strip", [False, True], ids=["clock", "fallback"])
@pytest.mark.parametrize("control_name", list(CONTROL_FACTORIES))
@pytest.mark.parametrize("prompt_len", [1, 4])
def test_position_tracking_goldens(control_name, prompt_len, strip):
    """Greedy generation is bit-identical to the recorded golden sequence in both position modes."""
    produced = _generate(control_name, prompt_len, strip=strip)

    if os.environ.get("STEERABILITY_CAPTURE_GOLDENS"):
        print(f'    ("{control_name}", {prompt_len}): {produced},')
        return

    expected = GOLDENS[(control_name, prompt_len)]
    assert produced == expected, (
        f"{control_name} (prompt_len={prompt_len}): golden drift\n"
        f"  expected: {expected}\n"
        f"  produced: {produced}"
    )
