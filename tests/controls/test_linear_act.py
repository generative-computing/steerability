"""Linear-AcT formula, scope and pipeline tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.linear_act.control import LinearAcT, _LinearAcTTransform, fit_linear_act
from steerability.spipe import SPipe, SpipeStaleError
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16


def test_sorted_least_squares_not_gaussian_moment_map():
    negative = torch.tensor([[4., -1.], [0., 3.], [2., 0.], [1., 2.]])
    positive = torch.tensor([[2., 7.], [20., 2.], [1., 1.], [5., 11.]])
    actual = fit_linear_act(positive, negative)
    expected = np.stack([np.linalg.lstsq(
        np.column_stack((np.sort(negative[:, j].numpy()), np.ones(4))),
        np.sort(positive[:, j].numpy()), rcond=None,
    )[0] for j in range(2)], axis=1)
    torch.testing.assert_close(actual, torch.from_numpy(expected).float())
    assert not torch.allclose(actual[0], positive.std(0) / negative.std(0), atol=1e-3)
    torch.testing.assert_close(fit_linear_act(positive.flip(0), negative.roll(1, 0)), actual)


@pytest.mark.parametrize("strength", [0., 0.4, 1., -0.5, 1.5])
def test_affine_strength_and_mask(strength):
    affine = torch.tensor([[2., 0.5], [3., -2.]])
    transform = _LinearAcTTransform({0: affine}, strength=strength).bind(
        SimpleNamespace(hidden_size=2, device=torch.device("cpu"), dtype=torch.float32),
    )
    y = torch.tensor([[[1., 2.], [3., 4.]]])
    mask = torch.tensor([[True, False]])
    out = transform.apply(y, layer_id=0, token_mask=mask)
    expected = y.clone()
    expected[:, 0] = (1 - strength) * y[:, 0] + strength * (y[:, 0] * affine[0] + affine[1])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("affine", [
    torch.ones(2, 1),
    torch.tensor([[float("nan")] * HIDDEN, [0.] * HIDDEN]),
])
def test_supplied_affine_is_validated_before_generation(affine):
    control = LinearAcT(affine={0: affine})
    pipeline = SteeringPipeline(controls=[control], model=tiny_llama(hidden=HIDDEN, heads=4))
    with pytest.raises(ValueError, match="affine must be finite"):
        pipeline.steer()


def test_control_binds_and_generates_with_calibrated_affine():
    torch.manual_seed(0)
    model = tiny_llama(hidden=HIDDEN, heads=4)
    control = LinearAcT(
        positive_activations={0: torch.randn(5, HIDDEN) + 0.3},
        negative_activations={0: torch.randn(5, HIDDEN)},
    )
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    output = pipeline.generate(
        input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=1, do_sample=False, eos_token_id=None,
    )
    assert output.ndim == 2
    assert control.interventions[0].transform.is_bound


def test_supplied_affine_is_cloned_at_steer_time():
    affine = {0: torch.tensor([[2.] * HIDDEN, [3.] * HIDDEN])}
    control = LinearAcT(affine=affine)
    pipeline = SteeringPipeline(
        controls=[control], model=tiny_llama(hidden=HIDDEN, heads=4), tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    affine[0].fill_(float("nan"))
    assert torch.isfinite(control.interventions[0].transform.affine[0]).all()


def test_precomputed_affine_freezes_without_loading_or_steering_model(tmp_path):
    """An unbound precomputed control is already a complete recipe. Authored by PI/Astra."""
    affine = {0: torch.stack((torch.full((HIDDEN,), 1.5), torch.full((HIDDEN,), 0.2)))}
    control = LinearAcT(affine=affine, strength=0.4)
    pipeline = SteeringPipeline(model_name_or_path="tiny-llama", controls=[control])
    saved = pipeline.to_spipe(freeze=True).save(tmp_path / "precomputed.spipe")
    assert pipeline.model is None
    assert not pipeline._is_steered
    assert control.export_state() == {}
    loaded = SPipe.load(saved)
    assert loaded.manifest["controls"][0]["resolved"] is None
    assert loaded.manifest["lock"]["model_fingerprint"] is None
    model, tokenizer = tiny_llama(hidden=HIDDEN, heads=4), wordlevel_tokenizer()
    rebuilt = loaded.pipeline(model=model, tokenizer=tokenizer)
    rebuilt.steer()
    torch.testing.assert_close(rebuilt.state_controls[0].export_state()["0"], affine[0])
    pipeline.model, pipeline.tokenizer = model, tokenizer
    pipeline.steer()
    query, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    torch.testing.assert_close(
        rebuilt.compute_logprobs(query, ref_output_ids=answer),
        pipeline.compute_logprobs(query, ref_output_ids=answer),
    )


def test_frozen_calibrated_affine_records_fit_digest():
    torch.manual_seed(0)
    control = LinearAcT(
        positive_activations={0: torch.randn(5, HIDDEN) + 0.3},
        negative_activations={0: torch.randn(5, HIDDEN)},
    )
    pipeline = SteeringPipeline(
        controls=[control], model=tiny_llama(hidden=HIDDEN, heads=4), tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    entry = pipeline.to_spipe(model_ref="tiny-llama").manifest["controls"][0]["resolved"]
    assert all(record["artifact_class"] == "calibrated" for record in entry["artifacts"].values())
    assert all(record["fit_digest"] for record in entry["artifacts"].values())


def test_frozen_calibrated_affine_rejects_changed_calibration(tmp_path):
    torch.manual_seed(0)
    control = LinearAcT(
        positive_activations={0: torch.randn(5, HIDDEN) + 0.3},
        negative_activations={0: torch.randn(5, HIDDEN)},
    )
    pipeline = SteeringPipeline(
        controls=[control], model=tiny_llama(hidden=HIDDEN, heads=4), tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    saved = pipeline.to_spipe(model_ref="tiny-llama").save(tmp_path / "linear-act")
    manifest_path = saved / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    args = manifest["controls"][0]["args"]
    args["positive_activations"]["$map"][0][1] = args["negative_activations"]["$map"][0][1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SpipeStaleError, match="fit digest"):
        SPipe.load(saved)


def test_degenerate_and_mismatched_samples_raise():
    with pytest.raises(ValueError, match="variance"):
        fit_linear_act(torch.randn(3, 2), torch.ones(3, 2))
    with pytest.raises(ValueError, match="equal"):
        fit_linear_act(torch.randn(3, 2), torch.randn(2, 2))
    with pytest.raises(ValueError, match="N >= 2"):
        fit_linear_act(torch.randn(1, 2), torch.randn(1, 2))
