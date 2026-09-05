"""Identity digests: config_id parity, recipe_id sensitivity, fit-digest staleness."""
import json

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.sweeps import expand_configurations
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.state_control.caa.control import CAA
from steerability.spipe import SPipe, SpipeStaleError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CAA_KWARGS = dict(
    data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
    train_spec={"method": "mean_diff", "accumulate": "last_token"},
    layer_id=1,
    multiplier=2.0,
)


def make_controls():
    few_shot = FewShot(
        directive="Answer formally.",
        positive_example_pool=[{"prompt": "hey", "response": "Good day."}],
        k_positive=1,
    )
    return [few_shot, CAA(**CAA_KWARGS)]


def test_config_id_matches_sweep_layer():
    controls = make_controls()
    point = next(iter(expand_configurations({"combo": controls}, base_model_name_or_path=TINY_MODEL)))
    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=controls).to_spipe(freeze=False)
    assert spipe.config_id == point.config_id


def test_recipe_id_sensitive_to_model_ref():
    controls = make_controls()
    a = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=controls).to_spipe(freeze=False)
    b = SteeringPipeline(model_name_or_path="org/other-model", controls=make_controls()).to_spipe(freeze=False)
    assert a.config_id == b.config_id
    assert a.recipe_id != b.recipe_id


@pytest.fixture(scope="module")
def frozen_caa_dir(tmp_path_factory):
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipeline = SteeringPipeline(
        model=model, tokenizer=tokenizer, controls=[CAA(**CAA_KWARGS)], model_name_or_path=TINY_MODEL,
    )
    pipeline.steer()
    return pipeline.to_spipe().save(tmp_path_factory.mktemp("spipe") / "caa_dir")


def _edit_manifest(directory, mutate):
    manifest = json.loads((directory / "spipe.json").read_text())
    mutate(manifest)
    (directory / "spipe.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))


def test_fit_digest_invariant_under_multiplier_edit(frozen_caa_dir):
    _edit_manifest(frozen_caa_dir, lambda m: m["controls"][0]["args"].__setitem__("multiplier", 42.0))
    SPipe.load(frozen_caa_dir)  # not stale


def test_fit_digest_sensitive_to_data_edit(frozen_caa_dir):
    def mutate(manifest):
        manifest["controls"][0]["args"]["data"]["fields"]["positives"] = ["EDITED", "kind b"]

    _edit_manifest(frozen_caa_dir, mutate)
    with pytest.raises(SpipeStaleError, match="fit digest|digests to"):
        SPipe.load(frozen_caa_dir)
    loaded = SPipe.load(frozen_caa_dir, allow_stale=True)
    report = loaded.verify()
    assert not report.ok
    assert any("digest" in message for message in report.errors)
