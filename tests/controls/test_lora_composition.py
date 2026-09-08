"""Structural-then-state composition through an unmerged LoRA wrapper.

Pins that `LoadLoRA(merge=False)` followed by a CAA state control steers and generates on a
hub-free tiny model, that the pipeline model stays a `PeftModel`, that the session layout reports
the inner model's facts, and that the CAA hook fires on the adapted decoder layer
(`base_model.model.model.layers.{layer}`).
"""
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.layout_facts import resolve_layout
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA
from tests.utils.tiny_models import tiny_llama, tiny_lora, wordlevel_tokenizer

LAYERS = 4
HIDDEN = 16
HEADS = 2
STEER_LAYER = 1


def test_lora_then_caa_steers_and_hooks_adapted_layer(tmp_path):
    from peft import PeftModel

    tiny_lora(tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)).save_pretrained(tmp_path)

    base = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    steering_vector = SteeringVector(
        model_type="unknown", directions={STEER_LAYER: torch.ones(1, HIDDEN)},
    )

    load_lora = LoadLoRA(path=str(tmp_path), base_model="tiny", allow_base_mismatch=True)
    caa = CAA(steering_vector=steering_vector, layer_id=STEER_LAYER, multiplier=1.0)
    pipeline = SteeringPipeline(controls=[load_lora, caa], model=base, tokenizer=tokenizer)
    pipeline.steer()

    assert isinstance(pipeline.model, PeftModel)

    facts = resolve_layout(model=pipeline.model)
    assert facts.num_layers == LAYERS
    assert facts.hidden_size == HIDDEN

    hooks = caa.get_hooks(torch.tensor([[3, 4, 5]]), {}, model=pipeline.model)
    hooked_modules = {spec["module"] for spec in hooks["forward"]}
    assert f"base_model.model.model.layers.{STEER_LAYER}" in hooked_modules

    out = pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=4)
    assert isinstance(out, torch.Tensor)
    assert out.size(1) >= 1
