"""S-space regressions through Linear modules and pipelines."""
import json

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.sspace.control import SSpace, apply_sspace, fit_sspace
from steerability.spipe import SPipe, SpipeStaleError
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
MODULE = "model.layers.0.self_attn.o_proj"


@pytest.mark.parametrize("gate", ["cosine", "signed", "off"])
def test_contrast_rank_bias_and_independent_gate_formula(gate):
    weight = torch.diag(torch.tensor([9., 4., 1.]))
    bias = torch.tensor([10., -7., 3.])
    negative = bias + torch.tensor([[1., 2., 0.], [-1., -2., 0.]])
    positive = negative + torch.tensor([0.03, 0.2, 2.])
    artifact = fit_sspace(weight, bias, positive, negative, rank=1)
    torch.testing.assert_close(artifact["u"].abs(), torch.tensor([[0.], [0.], [1.]]))
    y = bias + torch.tensor([[[2., 1., 4.], [0., 0., -3.], [0., 0., 0.]]])
    before = y.clone()
    out = apply_sspace(y, artifact, 0.7, gate)
    z = (y[..., 2] - bias[2]).unsqueeze(-1)
    d = artifact["directions"]
    unit = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
    cosine = (z / (z.norm(dim=-1, keepdim=True) + 1e-8)) @ unit.T
    g = torch.ones_like(cosine) if gate == "off" else cosine.abs() if gate == "cosine" else cosine
    expected = y.clone()
    expected[..., 2] += (0.7 * g * d.norm() * unit.squeeze()).squeeze(-1)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(y, before, rtol=0, atol=0)
    torch.testing.assert_close(apply_sspace(y, artifact, 0, gate), y, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 7), (7, 4)])
def test_rectangular_weight_output_projection(shape):
    torch.manual_seed(31)
    weight = torch.randn(*shape)
    bias = torch.randn(shape[0])
    pos_x, neg_x = torch.randn(9, shape[1]) + 1, torch.randn(9, shape[1])
    pos, neg = pos_x @ weight.T + bias, neg_x @ weight.T + bias
    artifact = fit_sspace(weight, bias, pos, neg, rank=2)
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    contrast = (pos_x @ vh.T * s.sqrt()).mean(0) - (neg_x @ vh.T * s.sqrt()).mean(0)
    indices = contrast.abs().topk(2).indices.sort().values
    direction = contrast[indices] / (contrast[indices].norm() + 1e-8)
    torch.testing.assert_close(artifact["directions"][0], direction)
    x = torch.randn(2, 3, shape[1])
    y = x @ weight.T + bias
    z = x @ vh[indices].T * s[indices].sqrt()
    unit = direction / (direction.norm() + 1e-8)
    gate = ((z / (z.norm(dim=-1, keepdim=True) + 1e-8)) @ unit).abs()
    expected = y + ((gate.unsqueeze(-1) * direction.norm() * unit) * s[indices].sqrt()) @ u[:, indices].T
    torch.testing.assert_close(apply_sspace(y, artifact), expected, atol=2e-5, rtol=2e-5)


def test_stacked_directions_keep_independent_gates():
    artifact = {"u": torch.eye(2), "sqrt_s": torch.ones(2), "bias": torch.zeros(2),
                "directions": torch.tensor([[2., 0.], [0., 3.]])}
    y = torch.tensor([[[1., 0.], [0., -1.]]])
    expected = torch.tensor([[[3., 0.], [0., 2.]]])
    torch.testing.assert_close(apply_sspace(y, artifact), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast_dtype", [None, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate", ["cosine", "signed", "off"])
def test_dtype_and_autocast_preserve_finite_formula(dtype, autocast_dtype, gate):
    artifact = {"u": torch.eye(2), "sqrt_s": torch.ones(2), "bias": torch.zeros(2),
                "directions": torch.tensor([[2., 0.], [0., 3.], [0., 0.]])}
    output = torch.tensor([[[0., 0.], [2., -1.]]], dtype=dtype)
    before = output.clone()
    if gate == "off":
        expected = output.double() + torch.tensor([2., 3.])
    else:
        cosine = output.double() / (output.double().norm(dim=-1, keepdim=True) + 1e-8)
        multiplier = cosine.abs() if gate == "cosine" else cosine
        expected = output.double() + multiplier * torch.tensor([2., 3.])
    with torch.autocast("cpu", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        actual = apply_sspace(output, artifact, gate=gate)
        assert apply_sspace(output, artifact, strength=0, gate=gate) is output
        zero_artifact = dict(artifact, directions=torch.zeros(1, 2))
        torch.testing.assert_close(apply_sspace(output, zero_artifact, gate=gate), output)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected.to(dtype))
    assert not torch.equal(actual, output)
    torch.testing.assert_close(output, before, rtol=0, atol=0)


def test_fp16_retains_small_basis_scales():
    artifact = {"u": torch.eye(2), "sqrt_s": torch.tensor([1e-8, 1.]), "bias": torch.zeros(2),
                "directions": torch.tensor([[1., 1.]])}
    output = torch.tensor([[[0., 1.]]], dtype=torch.float16)
    expected = torch.tensor([[[0., 1. + 2**-0.5]]], dtype=output.dtype)
    torch.testing.assert_close(apply_sspace(output, artifact), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cached_hook_preserves_small_basis_scales_under_autocast(dtype, device):
    artifact = {"u": torch.eye(2), "sqrt_s": torch.tensor([1e-8, 1.]), "bias": torch.zeros(2),
                "directions": torch.tensor([[1., 1.]])}
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(2, 2)
    control = SSpace(artifacts={"linear": artifact})
    control.steer(model)
    hook = control.get_hooks(torch.ones(1, 1, dtype=torch.long))["forward"][0]["hook_func"]
    output = torch.tensor([[[0., 1.]]], dtype=dtype, device=device)
    expected = torch.tensor([[[0., 1. + 2**-0.5]]], dtype=dtype, device=device)
    with torch.autocast(device.type, dtype=dtype):
        for _ in range(2):
            actual = hook(model.linear, (), {}, output)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, expected)


def test_control_generates_and_frozen_fit_rejects_changed_rank(tmp_path):
    torch.manual_seed(0)
    model = tiny_llama(hidden=HIDDEN, heads=4)
    control = SSpace(
        positive_outputs={MODULE: torch.randn(6, HIDDEN) + 0.3},
        negative_outputs={MODULE: torch.randn(6, HIDDEN)},
        rank=1,
    )
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    output = pipeline.generate(
        input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=1, do_sample=False, eos_token_id=None,
    )
    assert output.ndim == 2

    saved = pipeline.to_spipe(model_ref="tiny-llama").save(tmp_path / "sspace")
    manifest_path = saved / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["controls"][0]["resolved"]["artifacts"]
    assert all(record["artifact_class"] == "calibrated" for record in artifacts.values())
    assert all(record["fit_digest"] for record in artifacts.values())
    manifest["controls"][0]["args"]["rank"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SpipeStaleError, match="fit digest"):
        SPipe.load(saved)


def test_shared_tensor_artifacts_keep_each_fit_digest(tmp_path):
    torch.manual_seed(0)
    controls = [
        SSpace(
            positive_outputs={MODULE: torch.randn(6, HIDDEN) + offset},
            negative_outputs={MODULE: torch.randn(6, HIDDEN)},
            rank=rank,
        )
        for offset, rank in ((0.3, 1), (-0.4, 2))
    ]
    pipeline = SteeringPipeline(
        controls=controls,
        model=tiny_llama(hidden=HIDDEN, heads=4),
        tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    saved = pipeline.to_spipe(model_ref="tiny-llama").save(tmp_path / "two-sspace")
    entries = SPipe.load(saved).manifest["controls"]
    biases = [entry["resolved"]["artifacts"][f"{MODULE}:bias"] for entry in entries]
    assert biases[0]["id"] == biases[1]["id"]
    assert biases[0]["fit_digest"] != biases[1]["fit_digest"]


def test_reject_zero_singular_values_and_wrong_width():
    with pytest.raises(ValueError, match="singular"):
        fit_sspace(torch.zeros(3, 4), None, torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="out_features"):
        fit_sspace(torch.ones(3, 4), None, torch.ones(2, 4), torch.ones(2, 4))
