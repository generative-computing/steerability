"""Queued CUDA smoke test for VJP-delta extraction and generation."""
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.vjp_delta import VJPDelta, VJPDeltaFit
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

assert torch.cuda.is_available(), "CUDA was unavailable when this queued VJP-delta check ran."
model = tiny_llama().to("cuda")
tokenizer = wordlevel_tokenizer()
source = VJPDeltaFit(
    data={"positives": ["the cat sat", "the dog ran"], "negatives": ["dog ran fast"]},
    target_layer=2,
    source_layer_ids=[0, 1],
    skip_first=0,
)
master = source.resolve(model, tokenizer)
assert all(vector.device.type == "cpu" for vector in master.directions.values())
control = VJPDelta(steering_vector=source)
pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[control], model_name_or_path="tiny-cuda")
pipeline.steer()
bound = control.export_state()["intervention_0/transform"]
assert all(vector.device.type == "cuda" for vector in bound.directions.values())
reply = pipeline.generate(text="the cat", max_new_tokens=2, do_sample=False)
import steerability

print({
    "module": steerability.__file__,
    "master_device": master.directions[0].device.type,
    "bound_device": bound.directions[0].device.type,
    "reply": reply,
    "layers": sorted(bound.directions),
})
