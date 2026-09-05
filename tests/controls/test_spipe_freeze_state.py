"""Freezing state controls: same-class CAA/ITI/ActAdd, generic CAST lowering, verify policy."""
import json
import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec
from steerability.spipe import SPipe

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def load(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture(scope="module")
def llama():
    return load(LLAMA)


def steer_quietly(pipeline):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()


def freeze_reload(pipeline, tmp_path, name, **pipeline_kwargs):
    saved = pipeline.to_spipe().save(tmp_path / f"{name}.spipe")
    return SPipe.load(saved).pipeline(**pipeline_kwargs)


def test_caa_end_to_end(tmp_path, llama):
    model, tokenizer = llama
    caa = CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
        multiplier=2.0,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    reference = pipeline.generate(text="Hello there", max_new_tokens=6, do_sample=False)

    rebuilt = freeze_reload(pipeline, tmp_path, "caa")
    frozen = rebuilt.state_controls[0]
    assert type(frozen).__name__ == "CAA"
    assert frozen.steer_fits() == ()

    plan = rebuilt.check().plan
    assert [(step.control, step.access) for step in plan.steps] == [("CAA", ModelAccess.FACTS)]
    assert plan.fits == ()

    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    steer_quietly(rebuilt)
    assert rebuilt.generate(text="Hello there", max_new_tokens=6, do_sample=False) == reference


def test_cast_lowers_to_activation_adapter(tmp_path, llama):
    model, tokenizer = llama
    cast = CAST(
        behavior_data={"positives": ["be kind", "be nice"], "negatives": ["be mean", "be rude"]},
        behavior_layer_ids=[1],
        behavior_vector_strength=1.5,
        condition_data={
            "positives": ["math question one", "algebra query two"],
            "negatives": ["cooking recipe one", "sports news two"],
        },
        search=ConditionSearchSpec(candidate_layers=[1], threshold_range=(0.0, 0.2), threshold_step=0.1),
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[cast], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    reference = pipeline.generate(text="math question one please", max_new_tokens=6, do_sample=False)

    spipe = pipeline.to_spipe()
    entry = spipe.manifest["controls"][0]
    assert entry["method"] == "state_control/cast"
    assert entry["resolved"]["method"] == "state_control/activation_adapter"
    assert entry["resolved"]["origin"]["method"] == "state_control/cast"
    artifacts = entry["resolved"]["artifacts"]
    assert artifacts["intervention_0/transform"]["artifact_class"] == "direction"
    assert artifacts["intervention_0/gate"]["artifact_class"] == "calibrated"
    assert artifacts["intervention_0/gate"]["source"] == "ConditionPointSearch"

    saved = spipe.save(tmp_path / "cast.spipe")
    rebuilt = SPipe.load(saved).pipeline()
    assert type(rebuilt.state_controls[0]).__name__ == "ActivationAdapter"
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    steer_quietly(rebuilt)
    assert rebuilt.generate(text="math question one please", max_new_tokens=6, do_sample=False) == reference


def test_iti_and_act_add_same_class(tmp_path, llama):
    model, tokenizer = llama
    from steerability.algorithms.state_control.act_add.control import ActAdd
    from steerability.algorithms.state_control.iti.control import ITI

    iti = ITI(
        data={
            "positives": [f"true {i}" for i in range(8)],
            "negatives": [f"false {i}" for i in range(8)],
            "positive_groups": [i // 2 for i in range(8)],
            "negative_groups": [i // 2 for i in range(8)],
        },
        num_heads=2, alpha=5.0,
    )
    act_add = ActAdd(positive_prompt="Love", negative_prompt="Hate", layer_id=1, multiplier=3.0)
    for name, control in (("iti", iti), ("act_add", act_add)):
        pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[control],
                                    model_name_or_path=LLAMA)
        steer_quietly(pipeline)
        reference = pipeline.generate(text="Hello there", max_new_tokens=5, do_sample=False)
        rebuilt = freeze_reload(pipeline, tmp_path, name)
        assert type(rebuilt.state_controls[0]).__name__ == type(control).__name__
        rebuilt.model, rebuilt.tokenizer = model, tokenizer
        steer_quietly(rebuilt)
        assert rebuilt.generate(text="Hello there", max_new_tokens=5, do_sample=False) == reference


@pytest.fixture(scope="module")
def frozen_caa_file(tmp_path_factory):
    model, tokenizer = load(LLAMA)
    caa = CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=LLAMA)
    pipeline.steer()
    return pipeline.to_spipe().save(tmp_path_factory.mktemp("verify") / "caa.spipe")


def test_verify_strict_rejects_wrong_architecture(frozen_caa_file):
    model, tokenizer = load(MISTRAL)
    rebuilt = SPipe.load(frozen_caa_file).pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    with pytest.raises(ValueError, match="model_type"):
        rebuilt.steer()


def test_verify_direction_fingerprint_warns_not_raises(frozen_caa_file):
    model, tokenizer = load(LLAMA)
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    rebuilt = SPipe.load(frozen_caa_file).pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebuilt.steer()
    assert any("direction artifact" in str(item.message) for item in caught)


def test_verify_off_is_silent(frozen_caa_file):
    model, tokenizer = load(LLAMA)
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    rebuilt = SPipe.load(frozen_caa_file).pipeline(verify="off")
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebuilt.steer()
    assert not any("Precomputed" in str(item.message) for item in caught)


def test_calibrated_gate_fingerprint_strict_raises(tmp_path):
    model, tokenizer = load(LLAMA)
    cast = CAST(
        behavior_data={"positives": ["be kind", "be nice"], "negatives": ["be mean", "be rude"]},
        behavior_layer_ids=[1],
        condition_data={
            "positives": ["math question one", "algebra query two"],
            "negatives": ["cooking recipe one", "sports news two"],
        },
        search=ConditionSearchSpec(candidate_layers=[1], threshold_range=(0.0, 0.2), threshold_step=0.1),
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[cast], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    saved = pipeline.to_spipe().save(tmp_path / "cast.spipe")

    other_model, other_tokenizer = load(MISTRAL)
    rebuilt = SPipe.load(saved).pipeline()
    rebuilt.model, rebuilt.tokenizer = other_model, other_tokenizer
    with pytest.raises(ValueError):
        rebuilt.steer()

    relaxed = SPipe.load(saved).pipeline(verify="off")
    relaxed.model, relaxed.tokenizer = other_model, other_tokenizer
    steer_quietly(relaxed)


def test_verify_warn_downgrades_gate_mismatch(tmp_path):
    model, tokenizer = load(LLAMA)
    cast = CAST(
        behavior_data={"positives": ["be kind", "be nice"], "negatives": ["be mean", "be rude"]},
        behavior_layer_ids=[1],
        condition_data={
            "positives": ["math question one", "algebra query two"],
            "negatives": ["cooking recipe one", "sports news two"],
        },
        search=ConditionSearchSpec(candidate_layers=[1], threshold_range=(0.0, 0.2), threshold_step=0.1),
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[cast], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    saved = pipeline.to_spipe().save(tmp_path / "cast.spipe")

    other_model, other_tokenizer = load(MISTRAL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        relaxed = SPipe.load(saved).pipeline(verify="warn")
    assert any("disarmed" in str(item.message) for item in caught)
    relaxed.model, relaxed.tokenizer = other_model, other_tokenizer
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        relaxed.steer()
    assert any("Precomputed" in str(item.message) for item in caught)


def test_shared_gate_follower_refuses_to_save(llama):
    from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
    from steerability.algorithms.state_control.common.gating import Evidence, Gate, ProjectedCosineReadout, SumThreshold
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector
    from steerability.algorithms.state_control.common.transforms import AdditiveTransform
    from steerability.spipe import SpipeSaveError

    model, tokenizer = llama
    hidden = model.config.hidden_size
    gate = Gate(Evidence((1,), ProjectedCosineReadout({1: torch.randn(hidden)})), SumThreshold())
    vector = SteeringVector(model_type="llama", directions={1: torch.randn(1, hidden)})
    driver = ActivationAdapter(transform=AdditiveTransform(vector, strength=1.0), layer_ids=[1], gate=gate)
    follower = ActivationAdapter(
        transform=AdditiveTransform(vector.clone(), strength=0.5), layer_ids=[1],
        gate=gate, gate_driven_externally=True,
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[driver, follower],
                                model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    with pytest.raises(SpipeSaveError, match="shared"):
        pipeline.to_spipe()


def test_precomputed_recipe_refreezes(tmp_path, llama):
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    model, tokenizer = llama
    hidden = model.config.hidden_size
    vector = SteeringVector(model_type="llama", directions={1: torch.randn(1, hidden)})
    caa = CAA(steering_vector=vector, layer_id=1, multiplier=2.0)
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    reference = pipeline.generate(text="Hello there", max_new_tokens=5, do_sample=False)

    spipe = pipeline.to_spipe()
    entry = spipe.manifest["controls"][0]
    record = entry["resolved"]["artifacts"]["steering_vector"]
    assert record["artifact_class"] == "direction"
    assert record["fit_digest"] is None  # no fit produced the vector

    saved = spipe.save(tmp_path / "pre.spipe")
    rebuilt = SPipe.load(saved).pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    steer_quietly(rebuilt)
    assert rebuilt.generate(text="Hello there", max_new_tokens=5, do_sample=False) == reference


def test_norm_site_configuration_refuses_to_freeze(llama):
    from steerability.algorithms.state_control.angular_steering.control import AngularSteering
    from steerability.spipe import SpipeSaveError

    model, tokenizer = llama
    angular = AngularSteering(
        data={"positives": ["happy joy", "great fun"], "negatives": ["sad gloom", "bad pain"]},
        target_degree=90.0,
        intervention_point="norms",
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[angular], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    with pytest.raises(SpipeSaveError, match="norm_input"):
        pipeline.to_spipe()


def test_recipe_artifact_of_another_control_does_not_shadow_fit_record(tmp_path, llama):
    from steerability.algorithms.state_control.activation_adapter.control import ActivationAdapter
    from steerability.algorithms.state_control.common.transforms import AdditiveTransform
    from steerability.spipe import SpipeStaleError

    model, tokenizer = llama
    pairs = {"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]}
    train_spec = {"method": "mean_diff", "accumulate": "last_token"}
    caa = CAA(data=pairs, train_spec=train_spec, layer_id=1)
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[caa], model_name_or_path=LLAMA)
    steer_quietly(pipeline)
    vector = caa.export_state()["steering_vector"]

    # a disabled control is not frozen but its recipe args are still encoded, and the vector it
    # carries has the same content id as the one the second CAA's fit exports
    adapter = ActivationAdapter(transform=AdditiveTransform(vector), layer_ids=[1])
    adapter.enabled = False
    refit = CAA(data=pairs, train_spec=train_spec, layer_id=1)
    pipeline = SteeringPipeline(
        model=model, tokenizer=tokenizer, controls=[adapter, refit], model_name_or_path=LLAMA,
    )
    steer_quietly(pipeline)
    spipe = pipeline.to_spipe()

    record = spipe.manifest["controls"][1]["resolved"]["artifacts"]["steering_vector"]
    assert record["artifact_class"] == "direction"
    assert record["source"] == "ContrastiveFit"
    assert record["fit_digest"] is not None

    saved = spipe.save(tmp_path / "shadowed")
    manifest_path = saved / "spipe.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["controls"][1]["args"]["data"]["fields"]["positives"] = ["edited a", "edited b"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SpipeStaleError, match=r"controls\[1\]"):
        SPipe.load(saved)
