"""Freezing structural controls: trained products become load_lora/load_checkpoint entries."""
import warnings

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from steerability.spipe import SPipe, SpipeSaveError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def load():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_sft(tmp_path, **overrides):
    kwargs = dict(
        train_dataset=Dataset.from_dict({"text": [f"hello world {i}" for i in range(4)]}),
        output_dir=str(tmp_path / "adapter"),
        use_peft=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=100,
        load_best_model_at_end=False,
        max_length=32,
    )
    kwargs.update(overrides)
    return SFT(**kwargs)


def test_sft_lora_freezes_to_load_lora(tmp_path):
    model, tokenizer = load()
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_sft(tmp_path)],
                                model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    reference = pipeline.generate(text="hello", max_new_tokens=4, do_sample=False)

    spipe = pipeline.to_spipe()
    entry = spipe.manifest["controls"][0]
    assert entry["method"] == "structural_control/sft"
    assert entry["resolved"]["method"] == "structural_control/load_lora"
    assert entry["resolved"]["origin"]["method"] == "structural_control/sft"
    record = entry["resolved"]["artifacts"]["artifact"]
    assert record["type"] == "LoRAArtifact"
    assert record["fit_digest"]

    saved = spipe.save(tmp_path / "sft.spipe")
    rebuilt = SPipe.load(saved).pipeline()
    assert type(rebuilt.structural_controls[0]).__name__ == "LoadLoRA"

    fresh_model, _ = load()
    rebuilt.model, rebuilt.tokenizer = fresh_model, tokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt.steer()  # attaches the adapter; no training runs
    assert rebuilt.generate(text="hello", max_new_tokens=4, do_sample=False) == reference


def test_untrained_wrapper_is_trivial(tmp_path):
    sft = SFT(output_dir=str(tmp_path / "unused"))
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[sft])
    spipe = pipeline.to_spipe(freeze=False)
    assert spipe.manifest["controls"][0]["resolved"] is None
    assert sft.export_state() == {}
    assert sft.fit_identity() is None


def test_trained_without_product_raises(tmp_path):
    model, tokenizer = load()
    sft = make_sft(tmp_path, merge_lora_after_train=True, merged_output_dir=None)
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[sft],
                                model_name_or_path=TINY_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.steer()
    with pytest.raises(SpipeSaveError, match="merged_output_dir|freeze=False"):
        pipeline.to_spipe()
    assert pipeline.to_spipe(freeze=False) is not None
