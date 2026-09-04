# Adding a structural control method

**Required override**: `steer`

Structural control methods modify the model's weights or underlying architecture, creating a new model. This tutorial
implements a `NoiseInjection` method that perturbs a model's weights by (scaled) Gaussian noise.

The registry follows the standard pattern as:

```python
from .control import NoiseInjection
from .args import NoiseInjectionArgs

STEERING_METHOD = {
    "category": "structural_control",
    "name": "noise_injection",
    "control": NoiseInjection,
    "args": NoiseInjectionArgs,
}
```


Next, the args dataclass contains three parameters: `noise_scale` controlling the standard deviation of Gaussian noise
to inject, `target_modules` specifying which layer patterns to modify (or None for all linear layers), and `seed`
ensuring reproducible noise generation. The default for `target_modules` is `None` (all linear layers); note that (as
indicated in
[the general instructions for the arguments dataclass](../add_new_steering_method.md#2-arguments-dataclass-argspy)) a
mutable default, such as a non-empty list of patterns, would need `default_factory` instead of `default`.

```python
from dataclasses import dataclass, field
from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class NoiseInjectionArgs(BaseArgs):
    noise_scale: float = field(
        default=0.01,
        metadata={"help": "Standard deviation of Gaussian noise to inject, in [0, 1]."},
    )
    target_modules: list[str] | None = field(
        default=None,
        metadata={"help": "List of module name patterns to target. None means all linear layers."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducible noise generation."},
    )

    # validation
    def __post_init__(self):
        if not (0.0 <= self.noise_scale <= 1.0):
            raise ValueError("`noise_scale` must be in the interval [0, 1].")

        if self.target_modules is not None:
            if not isinstance(self.target_modules, list):
                raise TypeError("`target_modules` must be a list of strings or None.")
            if not all(isinstance(module, str) for module in self.target_modules):
                raise TypeError("All elements in `target_modules` must be strings.")
            if len(self.target_modules) == 0:
                raise ValueError("`target_modules` cannot be an empty list. Use None for all modules.")
```

Lastly, the control is implemented via the `steer` method by iterating over the model's linear modules and adding
scaled Gaussian noise to their parameters in place.

```python
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.noise_injection.args import NoiseInjectionArgs


class NoiseInjection(StructuralControl):
    """Injects controlled Gaussian noise into model weights (e.g., for robustness testing)."""

    Args = NoiseInjectionArgs

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer = None,
            **kwargs
    ) -> PreTrainedModel:
        torch.manual_seed(self.seed)

        with torch.no_grad():
            for name, module in model.named_modules():

                # check if this is a Linear layer and matches target patterns
                if not isinstance(module, torch.nn.Linear):
                    continue

                # if no specific targets, inject into all Linear layers; otherwise, check if module name contains any
                # target pattern
                if self.target_modules is not None:
                    if not any(target in name for target in self.target_modules):
                        continue

                # inject noise into all parameters of this module
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.noise_scale
                        param.data.add_(noise)

        return model

```

The control can then be called via:

```python
from aisteer360.algorithms.structural_control.noise_injection.control import NoiseInjection
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

noise_injection = NoiseInjection(
    noise_scale=0.005,
    target_modules=["q_proj", "v_proj"],
    seed=42
)

noise_injection_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[noise_injection]
)

noise_injection_pipeline.steer()

prompt = "What is a neural network?"
print(noise_injection_pipeline.generate(prompt, max_new_tokens=50))
```
