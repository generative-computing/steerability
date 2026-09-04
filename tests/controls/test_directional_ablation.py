"""Tests for Directional Ablation.

Two tiers:

- Unit tests (no model download) asserting the projection invariants.
- Integration tests (parametrized over CI models) mirroring test_cast.py: steer via a
  precomputed direction (K=1 and K=3) and via the estimation path, then generate.
"""
import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import ProjectionTransform
from aisteer360.algorithms.state_control.directional_ablation.args import DirectionalAblationArgs
from aisteer360.algorithms.state_control.directional_ablation.control import DirectionalAblation
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = "Give me a short set of instructions to follow when you respond."


# helpers

def _no_hooks_on(model) -> bool:
    """True when no forward or pre hooks remain on any module (nothing leaked)."""
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            return False
    return True


def _sv(hidden_size, num_layers, k=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    dirs = {lid: torch.randn(k, hidden_size, generator=g) for lid in range(num_layers)}
    return SteeringVector(model_type="test", directions=dirs)


# ProjectionTransform unit tests

class TestProjectionTransform:
    def test_single_direction_removed(self):
        """At alpha=1, the feature component is annihilated (out . d_hat == 0)."""
        sv = _sv(16, 1, k=1, seed=1)
        t = ProjectionTransform(sv.directions, alpha=1.0)
        hidden = torch.randn(2, 5, 16) * 3.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(2, 5, dtype=torch.bool))
        dhat = sv.directions[0][0] / sv.directions[0][0].norm()
        torch.testing.assert_close(out @ dhat, torch.zeros(2, 5), atol=1e-4, rtol=1e-4)

    def test_idempotent(self):
        """Applying ablation twice equals applying it once (P^2 = P) at alpha=1."""
        sv = _sv(12, 1, k=1, seed=2)
        t = ProjectionTransform(sv.directions, alpha=1.0)
        hidden = torch.randn(1, 4, 12) * 2.0
        m = torch.ones(1, 4, dtype=torch.bool)
        once = t.apply(hidden, layer_id=0, token_mask=m)
        twice = t.apply(once, layer_id=0, token_mask=m)
        torch.testing.assert_close(once, twice, atol=1e-5, rtol=1e-5)

    def test_partial_alpha_monotone_and_bounds(self):
        """alpha=0 is identity; residual component |out . d_hat| is monotone non-increasing in alpha."""
        sv = _sv(16, 1, k=1, seed=3)
        hidden = torch.randn(1, 4, 16) * 2.0
        m = torch.ones(1, 4, dtype=torch.bool)
        dhat = sv.directions[0][0] / sv.directions[0][0].norm()

        identity = ProjectionTransform(sv.directions, alpha=0.0).apply(hidden, layer_id=0, token_mask=m)
        torch.testing.assert_close(identity, hidden, atol=1e-6, rtol=1e-6)

        prev = None
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = ProjectionTransform(sv.directions, alpha=alpha).apply(hidden, layer_id=0, token_mask=m)
            comp = (out @ dhat).abs().max().item()
            if prev is not None:
                assert comp <= prev + 1e-5, f"|out.d_hat| increased with alpha at alpha={alpha}"
            prev = comp

    def test_subspace_removes_all_and_shrinks_norm(self):
        """K=3 subspace ablation removes every basis row and never increases the norm."""
        sv = _sv(16, 1, k=3, seed=4)
        t = ProjectionTransform(sv.directions, alpha=1.0)
        hidden = torch.randn(2, 6, 16) * 2.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(2, 6, dtype=torch.bool))
        basis = t._basis(0, hidden.device, hidden.dtype)
        for b in basis:
            torch.testing.assert_close(out @ b, torch.zeros(2, 6), atol=1e-4, rtol=1e-4)
        assert (out.norm(dim=-1) <= hidden.norm(dim=-1) + 1e-4).all()

    def test_subspace_order_independent(self):
        """Ablation is order-independent once the basis is orthonormalized (K=3, reversed rows)."""
        raw = torch.randn(3, 16, generator=torch.Generator().manual_seed(5))
        t = ProjectionTransform({0: raw}, alpha=1.0)
        t_rev = ProjectionTransform({0: raw.flip(0)}, alpha=1.0)
        hidden = torch.randn(2, 4, 16) * 2.0
        m = torch.ones(2, 4, dtype=torch.bool)
        out = t.apply(hidden, layer_id=0, token_mask=m)
        out_rev = t_rev.apply(hidden, layer_id=0, token_mask=m)
        torch.testing.assert_close(out, out_rev, atol=1e-4, rtol=1e-4)

    def test_norm_strictly_shrinks_with_component(self):
        """Norm strictly decreases when the hidden state has a non-zero component along d."""
        d = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        t = ProjectionTransform({0: d}, alpha=1.0)
        hidden = torch.tensor([[[3.0, 1.0, 0.0, 0.0]]])  # component along d is 3.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(1, 1, dtype=torch.bool))
        assert out.norm(dim=-1).item() < hidden.norm(dim=-1).item()

    def test_masked_positions_unchanged_and_masked_in_changes(self):
        """Masked-out positions are byte-identical; a masked-in position changes."""
        sv = _sv(16, 1, k=1, seed=6)
        t = ProjectionTransform(sv.directions, alpha=1.0)
        hidden = torch.randn(1, 4, 16) * 2.0
        mask = torch.tensor([[False, True, False, False]])
        out = t.apply(hidden, layer_id=0, token_mask=mask)
        assert torch.equal(out[0, 0], hidden[0, 0])
        assert torch.equal(out[0, 2], hidden[0, 2])
        assert torch.equal(out[0, 3], hidden[0, 3])
        assert not torch.equal(out[0, 1], hidden[0, 1])

    def test_missing_layer_returns_unchanged(self):
        """A layer_id absent from the directions returns the input untouched."""
        sv = _sv(16, 1, k=1, seed=7)
        t = ProjectionTransform(sv.directions, alpha=1.0)
        hidden = torch.randn(2, 3, 16)
        out = t.apply(hidden, layer_id=99, token_mask=torch.ones(2, 3, dtype=torch.bool))
        assert torch.equal(out, hidden)

    def test_1d_direction_treated_as_k1(self):
        """A plain [H] direction (CAA-style) is treated as K=1 and ablated correctly."""
        d = torch.randn(16, generator=torch.Generator().manual_seed(8))
        t = ProjectionTransform({0: d}, alpha=1.0)
        hidden = torch.randn(1, 3, 16) * 2.0
        out = t.apply(hidden, layer_id=0, token_mask=torch.ones(1, 3, dtype=torch.bool))
        dhat = d / d.norm()
        torch.testing.assert_close(out @ dhat, torch.zeros(1, 3), atol=1e-4, rtol=1e-4)


# DirectionalAblationArgs unit tests

class TestDirectionalAblationArgs:
    def test_neither_source_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            DirectionalAblationArgs()

    def test_both_sources_raise(self):
        sv = _sv(8, 2, k=1, seed=9)
        with pytest.raises(ValueError, match="not both"):
            DirectionalAblationArgs(steering_vector=sv, data={"positives": ["a"], "negatives": ["b"]})

    def test_alpha_out_of_range_raises(self):
        sv = _sv(8, 2, k=1, seed=10)
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            DirectionalAblationArgs(steering_vector=sv, alpha=1.5)
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            DirectionalAblationArgs(steering_vector=sv, alpha=-0.1)

    def test_alpha_bounds_accepted(self):
        sv = _sv(8, 2, k=1, seed=11)
        assert DirectionalAblationArgs(steering_vector=sv, alpha=0.0).alpha == 0.0
        assert DirectionalAblationArgs(steering_vector=sv, alpha=1.0).alpha == 1.0

    def test_k_ge_1_precomputed_accepted(self):
        """A precomputed vector with K > 1 is accepted (no K=2 forcing, unlike angular)."""
        for k in (1, 3, 5):
            sv = _sv(8, 2, k=k, seed=12 + k)
            args = DirectionalAblationArgs(steering_vector=sv)
            assert args.steering_vector.directions[0].size(0) == k

    def test_dict_data_coerced_to_contrastive_pairs(self):
        from aisteer360.algorithms.core.internals.data import ContrastivePairs

        args = DirectionalAblationArgs(data={"positives": ["p1", "p2"], "negatives": ["n1", "n2"]})
        assert isinstance(args.data, ContrastivePairs)

    def test_invalid_layer_range_raises(self):
        sv = _sv(8, 3, k=1, seed=20)
        with pytest.raises(ValueError, match="layer_range"):
            DirectionalAblationArgs(steering_vector=sv, layer_range=(2, 2))

    def test_empty_layer_ids_raises(self):
        sv = _sv(8, 3, k=1, seed=21)
        with pytest.raises(ValueError, match="layer_ids must be non-empty"):
            DirectionalAblationArgs(steering_vector=sv, layer_ids=[])


# integration tests over CI models

def _dims(model):
    model_type = model.config.model_type
    hidden_size = getattr(model.config, "hidden_size") if model_type != "gpt2" else getattr(model.config, "n_embd")
    num_layers = (
        getattr(model.config, "num_hidden_layers") if model_type != "gpt2" else getattr(model.config, "n_layer")
    )
    return hidden_size, num_layers


ABLATION_GRID = {
    "k": [1, 3],
    "alpha": [1.0, 0.5],
}


@pytest.mark.parametrize("conf", build_param_grid(ABLATION_GRID))
def test_ablation_precomputed_vector(model_and_tokenizer, device: torch.device, conf: dict):
    """Steer with a precomputed K>=1 direction and confirm generation produces tokens."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    hidden_size, num_layers = _dims(model)
    steering_vector = _sv(hidden_size, num_layers, k=conf["k"], seed=123)

    ablation = DirectionalAblation(
        steering_vector=steering_vector,
        alpha=conf["alpha"],
        layer_ids=list(range(min(2, num_layers))),
    )
    pipeline = SteeringPipeline(controls=[ablation],  device_map=device, model=model, tokenizer=tokenizer)
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
    """steer() must not mutate or narrow a caller-supplied direction (safe for sweeps)."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    hidden_size, num_layers = _dims(model)
    if num_layers < 2:
        pytest.skip("Needs at least two layers to exercise layer_range filtering.")

    steering_vector = _sv(hidden_size, num_layers, k=1, seed=321)
    original_layers = set(steering_vector.directions.keys())
    original_dtype = steering_vector.directions[0].dtype

    ablation = DirectionalAblation(steering_vector=steering_vector, alpha=1.0, layer_range=(0, 1))
    pipeline = SteeringPipeline(controls=[ablation],  device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    assert set(steering_vector.directions.keys()) == original_layers
    assert steering_vector.directions[0].dtype == original_dtype
    assert ablation._steering_vector is not steering_vector
    assert set(ablation._steering_vector.directions.keys()) == {0}


def test_ablation_estimation_path(model_and_tokenizer, device: torch.device):
    """Fit the direction from contrastive data and confirm generation."""
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

    ablation = DirectionalAblation(
        data=data,
        alpha=1.0,
        layer_ids=list(range(min(2, num_layers))),
    )
    pipeline = SteeringPipeline(controls=[ablation],  device_map=device, model=model, tokenizer=tokenizer)
    pipeline.steer()

    assert ablation._steering_vector is not None
    assert len(ablation._layer_ids) >= 1

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert isinstance(out_ids, torch.Tensor)
    assert out_ids.ndim == 2
    assert out_ids.size(1) >= 1
