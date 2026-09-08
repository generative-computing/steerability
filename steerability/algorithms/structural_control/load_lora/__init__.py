from steerability.algorithms.structural_control.load_lora.args import LoadLoRAArgs
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA

STEERING_METHOD = {
    "category": "structural_control",
    "name": "load_lora",
    "control": LoadLoRA,
    "args": LoadLoRAArgs,
}
