"""Regression tests for VJP-delta extraction and its frozen additive form."""
from __future__ import annotations

import gc
import json
import weakref

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.utils.assembly import collect_state_entries
from steerability.algorithms.state_control.activation_adapter import ActivationAdapter
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.vjp_delta import VJPDelta, VJPDeltaFit
from steerability.spipe import SPipe, SpipeStaleError
from tests.utils.tiny_models import tiny_gpt2, tiny_llama, wordlevel_tokenizer

HIDDEN = 32


def _fit_data():
    return {
        "positives": ["the cat sat", "the dog ran"],
        "negatives": ["dog ran fast"],
    }


def _source():
    return VJPDeltaFit(_fit_data(), target_layer=2, source_layer_ids=[0, 1], skip_first=0, batch_size=2)


def _capture_layer(model, pipeline, layer_id, input_ids):
    entries = collect_state_entries(
        pipeline.state_controls, input_ids, {}, hooks_in_process=True,
        lowered_state=pipeline._lowered_state, model=pipeline.model,
    )
    backend = pipeline._backend_for(pipeline._resolve_backend_spec(None))
    captured = {}
    with backend.open_session() as session, session.entries_applied(entries):
        def capture(_module, _args, _kwargs, output):
            captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        handle = model.model.layers[layer_id].register_forward_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            handle.remove()
    return captured["hidden"]


def _replace_hidden(output, hidden):
    """Replace the residual tensor while preserving a decoder layer's output container."""
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def _explicit_jacobian_direction(model, tokenizer, texts, cotangent, *, source_layer, target_layer, skip_first):
    """Compute the VJP contraction from an explicit Jacobian, without production extraction code."""
    source_module = model.model.layers[source_layer]
    target_module = model.model.layers[target_layer]
    total = None
    for text in texts:
        batch = tokenizer(text, return_tensors="pt")
        source_states = []

        def capture_source(_module, _args, output):
            source_states.append(output[0] if isinstance(output, tuple) else output)
            return None

        source_handle = source_module.register_forward_hook(capture_source)
        try:
            with torch.no_grad():
                model(**batch)
        finally:
            source_handle.remove()
        source_value = source_states.pop().detach()
        valid = torch.ones(source_value.shape[:2], dtype=torch.bool)
        valid[:, :skip_first] = False
        valid[:, -1] = False

        def target_from_source(replacement):
            target_states = []

            def replace_source(_module, _args, output):
                return _replace_hidden(output, replacement)

            def capture_target(_module, _args, output):
                target_states.append(output[0] if isinstance(output, tuple) else output)
                return None

            source_hook = source_module.register_forward_hook(replace_source)
            target_hook = target_module.register_forward_hook(capture_target)
            try:
                model(**batch)
            finally:
                source_hook.remove()
                target_hook.remove()
            return target_states.pop()[valid].reshape(-1)

        jacobian = torch.autograd.functional.jacobian(target_from_source, source_value)
        repeated_cotangent = cotangent.repeat(int(valid.sum())).to(jacobian.dtype)
        gradient = torch.tensordot(repeated_cotangent, jacobian, dims=([0], [0]))
        mean = gradient[valid].reshape(1, -1, gradient.size(-1)).mean(dim=1).squeeze(0)
        total = mean if total is None else total + mean
    return total / len(texts)


def _manual_target_mean(model, tokenizer, texts, target_layer):
    module = model.model.layers[target_layer]
    rows = []
    for text in texts:
        captured = []
        handle = module.register_forward_hook(
            lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output)
        )
        try:
            with torch.no_grad():
                model(**tokenizer(text, return_tensors="pt"))
        finally:
            handle.remove()
        rows.append(captured.pop()[0, -1].float())
    return torch.stack(rows).mean(dim=0)


def test_tiny_jacobian_matches_production_for_unequal_classes_and_cross_token_dependence():
    torch.manual_seed(7)
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    source = _source()
    vector = source.resolve(model, tokenizer)

    positive_mean = _manual_target_mean(model, tokenizer, _fit_data()["positives"], target_layer=2)
    negative_mean = _manual_target_mean(model, tokenizer, _fit_data()["negatives"], target_layer=2)
    cotangent = positive_mean - negative_mean
    positive = _explicit_jacobian_direction(
        model, tokenizer, _fit_data()["positives"], cotangent,
        source_layer=0, target_layer=2, skip_first=0,
    )
    negative = _explicit_jacobian_direction(
        model, tokenizer, _fit_data()["negatives"], cotangent,
        source_layer=0, target_layer=2, skip_first=0,
    )
    expected = (positive - negative) / (positive - negative).norm()
    assert torch.allclose(vector.directions[0].squeeze(0), expected, atol=1e-5)

    batch = tokenizer("the cat sat", return_tensors="pt")
    source_module = model.model.layers[0]
    target_module = model.model.layers[2]
    captured = []
    capture_handle = source_module.register_forward_hook(
        lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output)
    )
    try:
        with torch.no_grad():
            model(**batch)
    finally:
        capture_handle.remove()
    source_value = captured.pop().detach()

    def target_position_from_source(replacement):
        target_states = []
        source_handle = source_module.register_forward_hook(
            lambda _m, _a, output: _replace_hidden(output, replacement)
        )
        target_handle = target_module.register_forward_hook(
            lambda _m, _a, output: target_states.append(output[0] if isinstance(output, tuple) else output)
        )
        try:
            model(**batch)
        finally:
            source_handle.remove()
            target_handle.remove()
        return target_states.pop()[0, 2]

    jacobian = torch.autograd.functional.jacobian(target_position_from_source, source_value)
    assert jacobian[:, 0, 1].abs().sum() > 0


def test_tuple_decoder_outputs_are_not_replaced_by_extraction_hooks():
    model = tiny_gpt2()
    vector = VJPDeltaFit(_fit_data(), target_layer=2, source_layer_ids=[0], skip_first=0).resolve(
        model, wordlevel_tokenizer(),
    )
    assert set(vector.directions) == {0}


def test_fit_restores_mode_flags_and_existing_parameter_grads_after_error():
    model = tiny_llama().train()
    tokenizer = wordlevel_tokenizer()
    parameters = list(model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = torch.full_like(parameter, float(index + 1))
    flags = [parameter.requires_grad for parameter in parameters]
    grads = [parameter.grad.clone() for parameter in parameters]
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(ValueError, match="valid token span"):
        VJPDeltaFit(_fit_data(), target_layer=2, source_layer_ids=[0], skip_first=20).resolve(model, tokenizer)

    assert model.training is True
    for parameter, flag, grad in zip(parameters, flags, grads, strict=True):
        assert parameter.requires_grad is flag
        assert torch.equal(parameter.grad, grad)
    for name, value in weights.items():
        assert torch.equal(model.state_dict()[name], value)


def test_fit_restores_mode_flags_and_existing_parameter_grads_after_success():
    model = tiny_llama().train()
    tokenizer = wordlevel_tokenizer()
    parameters = list(model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = torch.full_like(parameter, float(index + 1))
    flags = [parameter.requires_grad for parameter in parameters]
    grads = [parameter.grad.clone() for parameter in parameters]
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    vector = _source().resolve(model, tokenizer)

    assert set(vector.directions) == {0, 1}
    assert model.training is True
    for parameter, flag, grad in zip(parameters, flags, grads, strict=True):
        assert parameter.requires_grad is flag
        assert torch.equal(parameter.grad, grad)
    for name, value in weights.items():
        assert torch.equal(model.state_dict()[name], value)


def test_invalid_layer_order_and_zero_direction_fail_clearly():
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    with pytest.raises(ValueError, match="must precede"):
        VJPDeltaFit(_fit_data(), target_layer=1, source_layer_ids=[1], skip_first=0).resolve(model, tokenizer)
    repeated = {"positives": ["the cat sat"], "negatives": ["the cat sat"]}
    with pytest.raises(ValueError, match="nonzero norm"):
        VJPDeltaFit(repeated, target_layer=2, source_layer_ids=[0], skip_first=0).resolve(model, tokenizer)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        _source().resolve(model, tokenizer)


def test_autograd_error_keeps_the_original_diagnostic(monkeypatch):
    def out_of_memory(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory while allocating a gradient buffer")

    monkeypatch.setattr(torch.autograd, "grad", out_of_memory)
    with pytest.raises(RuntimeError, match="VJP-delta autograd.grad failed: CUDA out of memory"):
        _source().resolve(tiny_llama(), wordlevel_tokenizer())


def test_additive_zero_strength_preserves_hidden_and_nonzero_adds_vector():
    torch.manual_seed(11)
    tokenizer = wordlevel_tokenizer()
    model = tiny_llama()
    vector = SteeringVector(model_type="llama", directions={0: torch.ones(1, HIDDEN)})
    input_ids = tokenizer("the cat sat", return_tensors="pt")["input_ids"]

    baseline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[])
    baseline.steer()
    zero = SteeringPipeline(
        model=model, tokenizer=tokenizer,
        controls=[VJPDelta(steering_vector=vector, strength=0.0, token_scope="all")],
    )
    zero.steer()
    nonzero = SteeringPipeline(
        model=model, tokenizer=tokenizer,
        controls=[VJPDelta(steering_vector=vector, strength=2.0, token_scope="all")],
    )
    nonzero.steer()

    baseline_hidden = _capture_layer(model, baseline, 0, input_ids)
    zero_hidden = _capture_layer(model, zero, 0, input_ids)
    nonzero_hidden = _capture_layer(model, nonzero, 0, input_ids)
    assert torch.equal(zero_hidden, baseline_hidden)
    assert torch.allclose(nonzero_hidden - baseline_hidden, torch.full_like(baseline_hidden, 2.0), atol=1e-6)


def test_pipeline_freeze_reload_skips_vjp_and_strength_does_not_stale(tmp_path, monkeypatch):
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    control = VJPDelta(
        data=_fit_data(), target_layer=2, source_layer_ids=[0, 1], skip_first=0, strength=0.5,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[control], model_name_or_path="tiny-llama")
    pipeline.steer()
    reference = pipeline.generate(text="the cat", max_new_tokens=3, do_sample=False)
    batch = pipeline.generate(text=["the cat", "the dog"], max_new_tokens=3, do_sample=False)
    assert isinstance(batch, list) and len(batch) == 2
    saved = pipeline.to_spipe().save(tmp_path / "vjp")

    rebuilt = SPipe.load(saved).pipeline()
    assert isinstance(rebuilt.state_controls[0], ActivationAdapter)
    assert rebuilt.state_controls[0].steer_fits() == ()
    monkeypatch.setattr(VJPDeltaFit, "resolve", lambda *_a, **_k: pytest.fail("frozen reload ran a VJP fit"))
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    assert rebuilt.generate(text="the cat", max_new_tokens=3, do_sample=False) == reference

    manifest = json.loads((saved / "spipe.json").read_text())
    manifest["controls"][0]["args"]["strength"] = 2.0
    (saved / "spipe.json").write_text(json.dumps(manifest))
    assert isinstance(SPipe.load(saved).pipeline().state_controls[0], ActivationAdapter)


def test_recipe_edit_is_stale_and_frozen_vector_uses_existing_mismatch_policy(tmp_path):
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(
        model=model, tokenizer=tokenizer,
        controls=[VJPDelta(data=_fit_data(), target_layer=2, source_layer_ids=[0], skip_first=0)],
        model_name_or_path="tiny-llama",
    )
    pipeline.steer()
    saved = pipeline.to_spipe().save(tmp_path / "vjp")
    manifest_path = saved / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["controls"][0]["args"]["data"]["fields"]["positives"] = ["the cat sat", "the cat sat"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SpipeStaleError):
        SPipe.load(saved)

    vector = SteeringVector(model_type="llama", directions={0: torch.ones(1, HIDDEN)})
    precomputed = SteeringPipeline(
        model=model, tokenizer=tokenizer, controls=[VJPDelta(steering_vector=vector)], model_name_or_path="tiny",
    )
    precomputed.steer()
    assert precomputed.state_controls[0].steer_fits() == ()
    saved_precomputed = precomputed.to_spipe().save(tmp_path / "precomputed")
    changed = tiny_llama()
    changed.load_state_dict(model.state_dict())
    with torch.no_grad():
        next(changed.parameters()).add_(0.01)
    same_architecture = SPipe.load(saved_precomputed).pipeline()
    same_architecture.model, same_architecture.tokenizer = changed, tokenizer
    with pytest.warns(UserWarning, match="direction artifact"):
        same_architecture.steer()

    frozen = SPipe.load(saved_precomputed).pipeline()
    frozen.model, frozen.tokenizer = tiny_gpt2(), tokenizer
    with pytest.raises(ValueError, match="model_type"):
        frozen.steer()


def test_fit_holds_only_a_weak_model_reference_and_control_cleanup_drops_bound_tensors():
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    source = _source()
    source.resolve(model, tokenizer)
    ref = weakref.ref(model)
    assert source._model_ref is ref or source._model_ref() is model
    control = VJPDelta(data=_fit_data(), target_layer=2, source_layer_ids=[0], skip_first=0)
    control.steer(model, tokenizer)
    assert control.steer_fits() == (("VJPDeltaFit", "direction"),)
    assert control.interventions
    control.cleanup()
    assert control.interventions == ()
    del model
    gc.collect()
    assert ref() is None
    with pytest.raises(ValueError, match="requires a live model"):
        source.resolve(None, tokenizer)
