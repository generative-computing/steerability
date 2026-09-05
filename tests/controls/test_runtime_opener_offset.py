"""Pass-opener and offset invariants of the shared `TransformHookRuntime` under ITI and
AngularSteering.

Structural checks that complement the golden and `after_prompt` semantics tests: exactly one
pass opener per generation, the shared offset advances once per forward pass to
`prompt_len + decode_passes`, and the single-forward (`compute_logprobs`) path steers the
expected scope.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.angular_steering.control import AngularSteering
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.iti.control import ITI
from tests.utils.runtime_helpers import capture_built_runtimes
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4
HEAD_DIM = HIDDEN // HEADS


def _make_iti_multilayer():
    g = torch.Generator().manual_seed(2)
    sv = SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(HEADS, HEAD_DIM, generator=g) for lid in range(LAYERS)},
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        probe_accuracies={(lid, h): 0.9 for lid in range(LAYERS) for h in range(HEADS)},
    )
    # heads on two distinct layers -> two hooked o_proj modules
    return ITI(
        steering_vector=sv,
        selected_heads=[(1, 0), (1, 1), (2, 3)],
        alpha=1.0,
        token_scope="after_prompt",
    )


def _make_angular_multilayer():
    g = torch.Generator().manual_seed(3)
    directions = {}
    for lid in range(LAYERS):
        raw = torch.randn(2, HIDDEN, generator=g)
        b1 = raw[0] / raw[0].norm()
        b2 = raw[1] - (raw[1] @ b1) * b1
        b2 = b2 / b2.norm()
        directions[lid] = torch.stack([b1, b2], dim=0)
    sv = SteeringVector(model_type="llama", directions=directions)
    # two layers -> four hooked norm modules (two norms per layer)
    return AngularSteering(
        steering_vector=sv, target_degree=90.0, token_scope="after_prompt", layer_range=(1, 3)
    )


def _steered_pipeline(control):
    torch.manual_seed(0)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


def test_angular_single_opener_and_offset_advance(monkeypatch):
    """Four hooked norms, one opener; offset ends at prompt_len + decode_passes."""
    control = _make_angular_multilayer()
    pipeline, _, _ = _steered_pipeline(control)
    capture = capture_built_runtimes(monkeypatch)

    # four hooked norm modules (2 active layers x 2 norms), exactly one built as pass opener
    input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)  # prompt_len 4
    hooks = control.get_hooks(input_ids, None, model=pipeline.model)
    assert len(hooks["pre"]) == 4
    assert capture.last._opener_built is True  # exactly one opener (two would have raised)

    prompt_len = 4
    max_new_tokens = 5
    pipeline.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=None
    )
    # prefill sets offset=prompt_len; each of (max_new_tokens - 1) decode passes adds 1
    assert capture.last._offset == prompt_len + (max_new_tokens - 1)


def test_iti_multilayer_single_opener_and_offset_advance(monkeypatch):
    """Two hooked o_proj modules, one opener; offset ends at prompt_len + decode_passes."""
    control = _make_iti_multilayer()
    pipeline, _, _ = _steered_pipeline(control)
    capture = capture_built_runtimes(monkeypatch)

    input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)  # prompt_len 4
    hooks = control.get_hooks(input_ids, None, model=pipeline.model)
    assert len(hooks["pre"]) == 2  # layers 1 and 2
    assert capture.last._opener_built is True

    prompt_len = 4
    max_new_tokens = 5
    pipeline.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=None
    )
    assert capture.last._offset == prompt_len + (max_new_tokens - 1)


@pytest.mark.parametrize("factory", [_make_iti_multilayer, _make_angular_multilayer])
def test_single_forward_compute_logprobs(factory, monkeypatch):
    """The single-forward path (compute_logprobs) completes and steers under the control."""
    control = factory()
    pipeline, _, _ = _steered_pipeline(control)
    capture = capture_built_runtimes(monkeypatch)

    input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
    ref_output_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

    logprobs = pipeline.compute_logprobs(input_ids=input_ids, ref_output_ids=ref_output_ids)

    assert logprobs.shape == (1, 3)
    assert torch.isfinite(logprobs).all()
    # a single forward is one pass (prefill only): offset equals the combined prompt+ref length
    assert capture.last._offset == input_ids.size(1) + ref_output_ids.size(1)
