"""Tests for Angular Steering.

Two tiers:

- Unit tests (no model download) asserting the transform/estimator/args invariants.
- Integration tests (parametrized over CI models) mirroring test_cast.py: steer via a
  precomputed plane and via the estimation path, then generate.
"""
import math

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.angular_steering.args import AngularSteeringArgs
from aisteer360.algorithms.state_control.angular_steering.control import AngularSteering
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import AlignmentAdaptiveTransform, RotationTransform
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = "Give me a short set of instructions to follow when you respond."


# helpers

def _no_hooks_on(model) -> bool:
    """True when no forward or pre hooks remain on any module (nothing leaked)."""
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            return False
    return True


def _basis_vector(hidden_size, num_layers, seed=0):
    gen = torch.Generator().manual_seed(seed)
    directions = {lid: torch.randn(2, hidden_size, generator=gen) for lid in range(num_layers)}
    return SteeringVector(model_type="test", directions=directions)


def _orthonormalize(raw):
    b1 = raw[0] / raw[0].norm()
    b2 = raw[1] - (raw[1] @ b1) * b1
    return b1, b2 / b2.norm()


# RotationTransform unit tests

class TestRotationTransform:
    def test_preserves_norm_and_orthogonal_complement(self):
        """Rotation preserves per-token norm and leaves the out-of-plane component unchanged."""
        sv = _basis_vector(16, 1, seed=1)
        t = RotationTransform(sv, angle=1.1, mode="target")
        hidden = torch.randn(2, 5, 16) * 3.0
        mask = torch.ones(2, 5, dtype=torch.bool)
        out = t.apply(hidden, layer_id=0, token_mask=mask)

        torch.testing.assert_close(hidden.norm(dim=-1), out.norm(dim=-1), atol=1e-4, rtol=1e-4)

        b1, b2 = _orthonormalize(sv.directions[0])
        p_in = (hidden @ b1).unsqueeze(-1) * b1 + (hidden @ b2).unsqueeze(-1) * b2
        p_out = (out @ b1).unsqueeze(-1) * b1 + (out @ b2).unsqueeze(-1) * b2
        torch.testing.assert_close(hidden - p_in, out - p_out, atol=1e-4, rtol=1e-4)

    def test_target_mode_lands_on_angle(self):
        """mode='target' rotates the in-plane component to the absolute angle (mod 2*pi)."""
        sv = _basis_vector(12, 1, seed=2)
        t = RotationTransform(sv, angle=0.7, mode="target")
        hidden = torch.randn(1, 4, 12) * 2.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(1, 4, dtype=torch.bool))
        b1, b2 = _orthonormalize(sv.directions[0])
        got = torch.atan2(out @ b2, out @ b1)
        torch.testing.assert_close(
            torch.remainder(got, 2 * math.pi),
            torch.remainder(torch.full_like(got, 0.7), 2 * math.pi),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_ninety_degrees_is_ablation(self):
        """At theta=90deg (target), the feature-axis component is annihilated (out . b1 == 0)."""
        sv = _basis_vector(16, 1, seed=3)
        t = RotationTransform(sv, angle=math.pi / 2, mode="target")
        hidden = torch.randn(1, 6, 16) * 2.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(1, 6, dtype=torch.bool))
        b1, _ = _orthonormalize(sv.directions[0])
        torch.testing.assert_close(out @ b1, torch.zeros(1, 6), atol=1e-4, rtol=1e-4)

    def test_masked_positions_unchanged_and_masked_in_changes(self):
        """Masked-out positions are byte-identical to input; a masked-in position changes."""
        sv = _basis_vector(16, 1, seed=4)
        t = RotationTransform(sv, angle=1.0, mode="target")
        hidden = torch.randn(1, 4, 16) * 2.0
        mask = torch.tensor([[False, True, False, False]])
        out = t.apply(hidden, layer_id=0, token_mask=mask)

        assert torch.equal(out[0, 0], hidden[0, 0])
        assert torch.equal(out[0, 2], hidden[0, 2])
        assert torch.equal(out[0, 3], hidden[0, 3])
        assert not torch.equal(out[0, 1], hidden[0, 1])

    def test_missing_layer_returns_unchanged(self):
        """A layer_id absent from the plane returns the input untouched."""
        sv = _basis_vector(16, 1, seed=5)
        t = RotationTransform(sv, angle=1.0, mode="target")
        hidden = torch.randn(2, 3, 16)
        out = t.apply(hidden, layer_id=99, token_mask=torch.ones(2, 3, dtype=torch.bool))
        assert torch.equal(out, hidden)

    def test_non_basis_pair_raises(self):
        """A non-[2, H] plane raises in __init__."""
        bad = SteeringVector(model_type="test", directions={0: torch.randn(1, 16)})
        with pytest.raises(ValueError, match=r"\[2, H\]"):
            RotationTransform(bad, angle=1.0)

    def test_invalid_mode_raises(self):
        sv = _basis_vector(8, 1, seed=6)
        with pytest.raises(ValueError, match="mode must be"):
            RotationTransform(sv, angle=1.0, mode="spin")


# AlignmentAdaptiveTransform unit tests

class TestAlignmentAdaptiveTransform:
    def test_threshold_zero_gates_by_sign(self):
        """With threshold=0.0, tokens with h.unit(b1) > 0 get the full rotation; others unchanged."""
        sv = _basis_vector(16, 1, seed=7)
        rotation = RotationTransform(sv, angle=1.3, mode="target")
        adaptive = AlignmentAdaptiveTransform(rotation, sv, threshold=0.0)

        hidden = torch.randn(1, 8, 16) * 2.5
        mask = torch.ones(1, 8, dtype=torch.bool)

        full = rotation.apply(hidden, layer_id=0, token_mask=mask)
        out = adaptive.apply(hidden, layer_id=0, token_mask=mask)

        b1, _ = _orthonormalize(sv.directions[0])
        unit = sv.directions[0][0] / (sv.directions[0][0].norm() + 1e-8)
        alignment = hidden @ unit  # [B, T]

        for pos in range(hidden.size(1)):
            if alignment[0, pos] > 0:
                torch.testing.assert_close(out[0, pos], full[0, pos], atol=1e-5, rtol=1e-5)
            else:
                torch.testing.assert_close(out[0, pos], hidden[0, pos], atol=1e-5, rtol=1e-5)


# AngularSteeringArgs unit tests

class TestAngularSteeringArgs:
    def test_neither_source_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            AngularSteeringArgs()

    def test_both_sources_raises(self):
        sv = _basis_vector(8, 2, seed=8)
        with pytest.raises(ValueError, match="not both"):
            AngularSteeringArgs(steering_vector=sv, data={"positives": ["a"], "negatives": ["b"]})

    def test_target_degree_resolves_to_radians(self):
        sv = _basis_vector(8, 2, seed=9)
        args = AngularSteeringArgs(steering_vector=sv, target_degree=180.0)
        assert abs(args.angle_radians - math.pi) < 1e-9

    def test_angle_radians_passthrough(self):
        sv = _basis_vector(8, 2, seed=10)
        args = AngularSteeringArgs(steering_vector=sv, angle=1.25)
        assert abs(args.angle_radians - 1.25) < 1e-9

    def test_both_angle_specs_raise(self):
        sv = _basis_vector(8, 2, seed=11)
        with pytest.raises(ValueError, match="not both"):
            AngularSteeringArgs(steering_vector=sv, target_degree=90.0, angle=1.0)

    def test_precomputed_non_k2_raises(self):
        bad = SteeringVector(model_type="test", directions={0: torch.randn(1, 8), 1: torch.randn(1, 8)})
        with pytest.raises(ValueError, match=r"\[2, H\]"):
            AngularSteeringArgs(steering_vector=bad)

    def test_dict_data_coerced_to_contrastive_pairs(self):
        from aisteer360.algorithms.core.internals.data import ContrastivePairs

        args = AngularSteeringArgs(data={"positives": ["p1", "p2"], "negatives": ["n1", "n2"]})
        assert isinstance(args.data, ContrastivePairs)

    def test_invalid_layer_range_raises(self):
        sv = _basis_vector(8, 3, seed=12)
        with pytest.raises(ValueError, match="layer_range"):
            AngularSteeringArgs(steering_vector=sv, layer_range=(2, 2))


# integration tests over CI models

def _dims(model):
    model_type = model.config.model_type
    hidden_size = getattr(model.config, "hidden_size") if model_type != "gpt2" else getattr(model.config, "n_embd")
    num_layers = (
        getattr(model.config, "num_hidden_layers") if model_type != "gpt2" else getattr(model.config, "n_layer")
    )
    return hidden_size, num_layers


ANGULAR_GRID = {
    "target_degree": [90.0],
    "adaptive": [False, True],
}


@pytest.mark.parametrize("conf", build_param_grid(ANGULAR_GRID))
def test_angular_precomputed_vector(model_and_tokenizer, device: torch.device, conf: dict):
    """Steer with a precomputed K=2 plane and confirm generation produces tokens."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    hidden_size, num_layers = _dims(model)
    steering_vector = _basis_vector(hidden_size, num_layers, seed=123)

    angular = AngularSteering(
        steering_vector=steering_vector,
        target_degree=conf["target_degree"],
        adaptive=conf["adaptive"],
        layer_range=(0, min(2, num_layers)),
    )
    pipeline = SteeringPipeline(controls=[angular],  device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)

    # generate twice; assert hooks are removed after each call so they do not accumulate
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)
    assert _no_hooks_on(model), "Hooks leaked after first generation"
    out_ids_again = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)
    assert _no_hooks_on(model), "Hooks leaked after second generation"

    for out in (out_ids, out_ids_again):
        assert isinstance(out, torch.Tensor), "Output is not torch.Tensor"
        assert out.ndim == 2, "Expected (batch, seq_len) tensor"
        assert out.size(1) >= 1, "No new tokens generated"


def test_steer_does_not_mutate_caller_vector(model_and_tokenizer, device: torch.device):
    """steer() must not mutate or narrow a caller-supplied plane (safe for ControlSpec sweeps)."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    hidden_size, num_layers = _dims(model)
    if num_layers < 2:
        pytest.skip("Needs at least two layers to exercise layer_range filtering.")

    steering_vector = _basis_vector(hidden_size, num_layers, seed=321)
    original_layers = set(steering_vector.directions.keys())
    original_dtype = steering_vector.directions[0].dtype

    angular = AngularSteering(steering_vector=steering_vector, target_degree=90.0, layer_range=(0, 1))
    pipeline = SteeringPipeline(controls=[angular],  device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    # caller's vector keeps every layer and its original dtype; the control uses a private copy
    assert set(steering_vector.directions.keys()) == original_layers
    assert steering_vector.directions[0].dtype == original_dtype
    assert angular._steering_vector is not steering_vector
    assert set(angular._steering_vector.directions.keys()) == {0}


def test_angular_estimation_path(model_and_tokenizer, device: torch.device):
    """Fit the plane from contrastive data and confirm each direction is [2, H] before generating."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    _, num_layers = _dims(model)

    data = {
        "positives": [
            "Sure, here is how to do that.",
            "Absolutely, let me help you with this.",
            "Of course, here are the steps.",
        ],
        "negatives": [
            "I cannot help with that request.",
            "Sorry, I will not assist with this.",
            "I must refuse to answer that.",
        ],
    }

    angular = AngularSteering(
        data=data,
        target_degree=180.0,
        layer_range=(0, min(2, num_layers)),
    )
    pipeline = SteeringPipeline(controls=[angular],  device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    assert angular._steering_vector is not None
    assert len(angular._steering_vector.directions) >= 1
    for direction in angular._steering_vector.directions.values():
        assert direction.ndim == 2 and direction.size(0) == 2, "Each learned direction must be [2, H]"

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert isinstance(out_ids, torch.Tensor)
    assert out_ids.ndim == 2
    assert out_ids.size(1) >= 1
