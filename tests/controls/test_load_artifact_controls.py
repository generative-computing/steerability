"""The `load_checkpoint` and `load_lora` structural controls, standalone."""
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
from steerability.algorithms.core.registry import REGISTRY
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.structural_control.load_checkpoint.control import LoadCheckpoint
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def model_and_tok():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_registry_discovery():
    assert "load_checkpoint" in REGISTRY["structural_control"]
    assert "load_lora" in REGISTRY["structural_control"]


def test_load_checkpoint_steers(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    checkpoint = tmp_path / "ckpt"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    control = LoadCheckpoint(path=str(checkpoint))
    assert control.artifact_capability() is Capability.SERVE_CHECKPOINT
    artifact = control.export_artifact()
    assert isinstance(artifact, CheckpointArtifact) and artifact.path == str(checkpoint)

    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[control],
                                tokenizer=tokenizer, model=model)
    pipeline.steer()
    assert pipeline.model is not model  # checkpoint replaced the incoming model
    assert pipeline.generate(text="hello", max_new_tokens=3, do_sample=False)


def test_load_lora_capability_and_artifact():
    control = LoadLoRA(path="/tmp/adapter", base_model=TINY_MODEL)
    assert control.artifact_capability() is Capability.SERVE_LORA
    artifact = control.export_artifact()
    assert isinstance(artifact, LoRAArtifact)
    assert artifact.base_model == TINY_MODEL


def test_load_lora_base_mismatch_raises(model_and_tok):
    model, tokenizer = model_and_tok
    control = LoadLoRA(path="/tmp/adapter", base_model="some/other-model")
    with pytest.raises(ValueError, match="base model"):
        control.steer(model, tokenizer)


def test_load_lora_requires_model():
    control = LoadLoRA(path="/tmp/adapter", base_model=TINY_MODEL)
    with pytest.raises(ValueError, match="base model|pipeline model"):
        control.steer(None)


def test_args_validation():
    with pytest.raises(ValueError, match="path"):
        LoadCheckpoint()
    with pytest.raises(ValueError, match="base_model"):
        LoadLoRA(path="/tmp/adapter")
