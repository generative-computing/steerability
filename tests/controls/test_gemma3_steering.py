"""End-to-end residual-stream steering of a composite multimodal wrapper under text-only prompting.

Pins that a CAA state control steers and generates on a hub-free tiny Gemma 3 conditional wrapper
(decoder at the nested `model.language_model.layers` root), with the hook firing on the text
decoder layer.
"""
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.tiny_models import tiny_gemma3_conditional, wordlevel_tokenizer

LAYERS = 4
HIDDEN = 32
HEADS = 4
STEER_LAYER = 1


def test_caa_steers_and_generates_on_gemma3_conditional():
    model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    tokenizer = wordlevel_tokenizer()
    steering_vector = SteeringVector(
        model_type="unknown", directions={STEER_LAYER: torch.ones(1, HIDDEN)},
    )
    caa = CAA(steering_vector=steering_vector, layer_id=STEER_LAYER, multiplier=1.0)
    pipeline = SteeringPipeline(controls=[caa], model=model, tokenizer=tokenizer)
    pipeline.steer()

    hooks = caa.get_hooks(torch.tensor([[3, 4, 5]]), {}, model=model)
    hooked_modules = {spec["module"] for spec in hooks["forward"]}
    assert f"model.language_model.layers.{STEER_LAYER}" in hooked_modules

    out = pipeline.generate(text="the cat sat on mat", max_new_tokens=4)
    assert isinstance(out, str)
