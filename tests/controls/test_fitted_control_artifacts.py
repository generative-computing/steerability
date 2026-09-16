"""Fit provenance and runtime tensor reuse for the three ports. Authored by PI/Astra."""
import json

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.corda_pca.control import CordaPCA
from steerability.algorithms.state_control.linear_act.control import LinearAcT
from steerability.algorithms.state_control.sspace.control import SSpace
from steerability.spipe import SPipe, SpipeStaleError
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
MODULES = tuple(f"model.layers.{i}.self_attn.o_proj" for i in range(2))


@pytest.fixture(params=[CordaPCA, SSpace, LinearAcT], ids=lambda cls: cls.__name__)
def fitted_pipeline(request):
    torch.manual_seed(17)
    positive = {name: torch.randn(6, HIDDEN) + 0.3 for name in MODULES}
    negative = {name: torch.randn(6, HIDDEN) for name in MODULES}
    cls = request.param
    if cls is CordaPCA:
        control = cls(positive_inputs=positive, negative_inputs=negative, rank=2)
    elif cls is SSpace:
        control = cls(positive_outputs=positive, negative_outputs=negative, rank=2)
    else:
        control = cls(
            positive_activations={i: positive[name] for i, name in enumerate(MODULES)},
            negative_activations={i: negative[name] for i, name in enumerate(MODULES)},
        )
    pipeline = SteeringPipeline(
        controls=[control], model=tiny_llama(hidden=HIDDEN, heads=4), tokenizer=wordlevel_tokenizer(),
    )
    pipeline.steer()
    return pipeline


def test_every_fitted_artifact_has_provenance_and_reload_preserves_scores(fitted_pipeline, tmp_path):
    pipeline = fitted_pipeline
    control = pipeline.state_controls[0]
    query, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    expected = pipeline.compute_logprobs(query, ref_output_ids=answer)
    saved = pipeline.to_spipe(model_ref="tiny-llama").save(tmp_path / "fitted")
    loaded = SPipe.load(saved)
    records = loaded.manifest["controls"][0]["resolved"]["artifacts"]
    assert set(records) == set(control.export_state())
    assert len(records) >= 2
    assert all(record["fit_digest"] for record in records.values())
    assert len({record["fit_digest"] for record in records.values()}) == 1
    assert all(record["source"] == control.steer_fits()[0][0] for record in records.values())
    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = pipeline.model, pipeline.tokenizer
    rebuilt.steer()
    torch.testing.assert_close(rebuilt.compute_logprobs(query, ref_output_ids=answer), expected)

    second_save = rebuilt.to_spipe(model_ref="tiny-llama").save(tmp_path / "precomputed")
    manifest_path = second_save / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["controls"][0]
    assert rebuilt.state_controls[0].steer_fits() == ()
    assert rebuilt.state_controls[0].fit_identity() is None
    assert all(record["fit_digest"] is None and record["source"] is None
               for record in entry["resolved"]["artifacts"].values())
    entry["args"]["strength"] = 0.7
    if isinstance(control, (CordaPCA, SSpace)):
        entry["args"]["rank"] = 3
    if isinstance(control, CordaPCA):
        entry["args"].update(damping=0.5, normalize=False)
    manifest_path.write_text(json.dumps(manifest))
    SPipe.load(second_save)


def test_changed_fit_recipe_is_stale(fitted_pipeline, tmp_path):
    saved = fitted_pipeline.to_spipe(model_ref="tiny-llama").save(tmp_path / "edited")
    manifest_path = saved / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    args = manifest["controls"][0]["args"]
    if isinstance(fitted_pipeline.state_controls[0], LinearAcT):
        args["positive_activations"]["$map"][1][1] = args["negative_activations"]["$map"][1][1]
    else:
        args["rank"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SpipeStaleError, match="fit digest"):
        SPipe.load(saved)


def test_runtime_reuses_converted_artifacts_without_changing_cpu_export(fitted_pipeline, monkeypatch, device):
    control = fitted_pipeline.state_controls[0]
    exported = control.export_state()
    snapshots = {name: tensor.clone() for name, tensor in exported.items()}
    originals = {id(tensor) for tensor in exported.values()}
    conversions = []
    tensor_to = torch.Tensor.to

    def track_to(tensor, *args, **kwargs):
        result = tensor_to(tensor, *args, **kwargs)
        if id(tensor) in originals and result is not tensor:
            conversions.append(id(tensor))
        return result

    monkeypatch.setattr(torch.Tensor, "to", track_to)
    inputs = torch.ones(1, 3, dtype=torch.long, device=device)
    hooks = control.get_hooks(inputs, model=fitted_pipeline.model)["forward"]
    for dtype in (torch.float64, torch.float32, torch.float64):
        output = torch.randn(1, 3, HIDDEN, dtype=dtype, device=device)
        for index in range(2):
            if isinstance(control, LinearAcT):
                transform = control.interventions[0].transform
                apply = lambda: transform.apply(output, layer_id=index, token_mask=inputs.bool())
            else:
                entry = hooks[index]
                module = fitted_pipeline.model.get_submodule(entry["module"])
                apply = lambda: entry["hook_func"](module, (), {}, output)
            first = apply()
            count = len(conversions)
            second = apply()
            assert len(conversions) == count
            assert first.dtype == dtype
            torch.testing.assert_close(first, second)
    assert len(conversions) == len(exported) * (1 if device.type == "cpu" else 2)
    for name, tensor in control.export_state().items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, snapshots[name], rtol=0, atol=0)
