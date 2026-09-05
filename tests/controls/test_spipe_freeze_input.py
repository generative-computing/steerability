"""Freezing input controls: the precomputed `memory=` slot on prewrite/gepa/cpo."""
import warnings

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.common.memory.text import TextMemory
from steerability.algorithms.input_control.gepa.control import GEPA
from steerability.algorithms.input_control.prewrite.control import PRewrite
from steerability.spipe import SPipe, SpipeCodeRefError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def length_scorer(response, row):
    return float(len(response))


@pytest.fixture(scope="module")
def model_and_tok():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_provided_memory_skips_optimization(model_and_tok):
    model, tokenizer = model_and_tok
    gepa = GEPA(memory=TextMemory(slots={"instruction": "Be terse."}))
    assert gepa.steer_access() is ModelAccess.FACTS
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[gepa],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()  # no rollouts run; the memory installs directly
    assert gepa.memory["instruction"] == "Be terse."
    assert pipeline.generate(messages=[{"role": "user", "content": "hi"}], max_new_tokens=3, do_sample=False)


def test_prewrite_memory_dict_form_roundtrip(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    prewrite = PRewrite(memory={"slots": {"instruction": "Be formal."}})
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[prewrite],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    saved = pipeline.to_spipe().save(tmp_path / "prewrite.spipe")
    rebuilt = SPipe.load(saved).pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    assert rebuilt.input_controls[0].memory["instruction"] == "Be formal."


def test_gepa_freeze_roundtrip_and_code_gating(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    gepa = GEPA(
        seed_instruction="Answer briefly.",
        train_set=[{"input": f"q{i}"} for i in range(4)],
        row_scorer=length_scorer,
        budget=6, minibatch_size=1, pareto_set_size=2, seed=0,
        gen_kwargs={"max_new_tokens": 4, "do_sample": False},
    )
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[gepa],
                                model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    reference = pipeline.generate(messages=[{"role": "user", "content": "hi"}],
                                  max_new_tokens=4, do_sample=False)

    spipe = pipeline.to_spipe()
    assert spipe.code_dependent
    entry = spipe.manifest["controls"][0]
    assert entry["resolved"]["method"] == "input_control/gepa"
    assert entry["resolved"]["artifacts"]["memory"]["type"] == "TextMemory"

    saved = spipe.save(tmp_path / "gepa.spipe")
    with pytest.raises(SpipeCodeRefError, match="allow_code"):
        SPipe.load(saved).pipeline()

    rebuilt = SPipe.load(saved, allow_code=True).pipeline()
    frozen = rebuilt.input_controls[0]
    assert frozen.args.memory is not None
    assert frozen.steer_access() is ModelAccess.FACTS
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()  # installs the memory; no search runs
    assert frozen.memory["instruction"] == gepa.memory["instruction"]
    assert rebuilt.generate(messages=[{"role": "user", "content": "hi"}],
                            max_new_tokens=4, do_sample=False) == reference


def test_gepa_fit_identity_excludes_memory_and_progress():
    gepa = GEPA(
        seed_instruction="Answer briefly.",
        train_set=[{"input": "q"}],
        row_scorer=length_scorer,
        budget=6, minibatch_size=1, pareto_set_size=2,
    )
    identity = gepa.fit_identity()
    assert "memory" not in identity and "progress_callback" not in identity
    assert identity["seed_instruction"] == "Answer briefly."

    frozen_like = GEPA(memory=TextMemory(slots={"instruction": "x"}))
    assert frozen_like.fit_identity() is None
