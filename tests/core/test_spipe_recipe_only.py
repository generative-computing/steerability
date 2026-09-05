"""Recipe-only spipes: unsteered save/load/steer refits, thaw equivalence, verify report."""
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.spipe import SPipe, SpipeSaveError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def make_caa():
    return CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
        multiplier=2.0,
    )


@pytest.fixture(scope="module")
def model_and_tok():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_unsteered_pipeline_saves_recipe_only(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[make_caa()])
    spipe = pipeline.to_spipe()
    assert not spipe.is_frozen
    assert spipe.manifest["lock"] is None
    assert spipe.manifest["controls"][0]["resolved"] is None

    saved = spipe.save(tmp_path / "recipe.spipe")
    loaded = SPipe.load(saved)
    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    plan = rebuilt._support_report.plan
    assert [(fit.control, fit.artifact) for fit in plan.fits] == [("CAA", "ContrastiveFit")]
    assert rebuilt.generate(text="hello", max_new_tokens=3, do_sample=False)


def test_freeze_requires_steered_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[make_caa()])
    with pytest.raises(SpipeSaveError, match="steer"):
        pipeline.to_spipe(freeze=True)


def test_thaw_equivalence(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    frozen = pipeline.to_spipe()
    recipe_only = pipeline.to_spipe(freeze=False)
    thawed = frozen.thaw()

    assert not thawed.is_frozen
    assert thawed.recipe_id == frozen.recipe_id == recipe_only.recipe_id
    thawed_controls = thawed.manifest["controls"]
    recipe_controls = recipe_only.manifest["controls"]
    assert [entry["args"] for entry in thawed_controls] == [entry["args"] for entry in recipe_controls]
    assert all(entry["resolved"] is None for entry in thawed_controls)


def test_verify_report_contents(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    spipe = pipeline.to_spipe()
    report = spipe.verify()
    assert report.ok
    assert not report.errors
    assert "verify: ok" in report.render()

    thin = spipe.save(tmp_path / "thin_dir", artifacts="thin")
    thin_report = SPipe.load(thin).verify()
    assert thin_report.ok
    assert any("thin" in message for message in thin_report.warnings)


def test_describe_lists_entries(model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    text = pipeline.to_spipe().describe()
    assert "state_control/caa" in text
    assert "steering_vector" in text
