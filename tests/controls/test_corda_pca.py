"""CorDA formula and public pipeline regressions."""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.corda_pca.control import CordaPCA, fit_corda_direction
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
MODULE = "model.layers.0.self_attn.o_proj"


@pytest.mark.parametrize("shape,rank", [((5, 9), -1), ((9, 5), 3), ((5, 5), 1)])
def test_dense_covariance_formula(shape, rank):
    torch.manual_seed(4)
    w = torch.randn(*shape)
    pos, neg = torch.randn(12, shape[1]) + 0.6, torch.randn(12, shape[1])
    x = torch.cat((pos, neg))
    covariance = x.T @ x / len(x) + 0.07 * x.square().mean() * torch.eye(shape[1])
    u, s, vh = torch.linalg.svd(w @ covariance, full_matrices=False)
    r = min(shape) if rank == -1 else rank
    z = (pos - neg) @ torch.linalg.solve(covariance, vh[:r].T) * s[:r].sqrt()
    _, _, pc = torch.linalg.svd(z - z.mean(0), full_matrices=False)
    v = pc[0] * torch.sign(z.mean(0) @ pc[0] + 1e-8)
    expected = (v / (v.norm() + 1e-8) * s[:r].sqrt()) @ u[:, :r].T
    actual = fit_corda_direction(w, pos, neg, rank, 0.07)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    assert actual.norm() > 0


@pytest.mark.parametrize("negative", [
    torch.tensor([[-2., 0., 1.], [0., 1., -1.], [2., -1., 0.]]),
    torch.eye(3),
    1024 + torch.eye(3),
])
@pytest.mark.parametrize("contrast", [0.0, 4.0])
def test_reject_undefined_centered_pca(contrast, negative):
    positive = negative + torch.tensor([0., contrast, 0.])
    with pytest.raises(ValueError, match="centered.*variance"):
        fit_corda_direction(torch.diag(torch.tensor([3., 2., 1.])), positive, negative)


@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_resolvable_variation_is_not_rejected_by_absolute_threshold(scale):
    torch.manual_seed(42)
    negative = torch.randn(8, 3)
    positive = negative + scale * torch.randn(8, 3)
    direction = fit_corda_direction(torch.eye(3), positive, negative)
    assert torch.isfinite(direction).all()
    assert direction.norm() > 0


def test_hook_preserves_flattened_linear_output_shape():
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    direction = torch.arange(HIDDEN, dtype=torch.float32)
    control = CordaPCA(directions={"linear": direction}, strength=0.5)
    control.steer(model)
    hook = control.get_hooks(torch.ones(1, 1, dtype=torch.long))["forward"][0]["hook_func"]
    output = torch.randn(3, HIDDEN)
    actual = hook(model.linear, (), {}, output)
    assert actual.shape == output.shape
    torch.testing.assert_close(actual, output + 0.5 * direction)


def test_control_fits_and_generates_without_retaining_hooks():
    torch.manual_seed(0)
    model = tiny_llama(hidden=HIDDEN, heads=4)
    control = CordaPCA(
        positive_inputs={MODULE: torch.randn(6, HIDDEN) + 0.3},
        negative_inputs={MODULE: torch.randn(6, HIDDEN)},
    )
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    input_ids = torch.tensor([[1, 2, 3]])
    baseline = model(input_ids).logits.detach().clone()
    output = pipeline.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
    assert output.ndim == 2
    torch.testing.assert_close(model(input_ids).logits, baseline)
