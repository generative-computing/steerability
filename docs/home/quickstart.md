# Quickstart

This guide will walk you through how to run a simple control in AISteer360.

!!! note
    By default, AISteer360 runs the model inside your process. For efficient inference on more complex steering
    operations, please run the toolkit from a machine that has enough GPU memory for both the base checkpoint and the
    extra overhead your steering method/pipeline adds. Inference through vLLM (offline engine or server) is available
    via the [execution backends](../concepts/steering_pipelines.md#execution-backends).

The first step in steering any model is to define how you want to steer, i.e., the control. For this guide, we will use
an `ActivationAdapter`, a state control that edits the model's internal activations at inference time. The desired
target behavior for this example is "positivity".

An activation adapter is assembled from a few slots: a **transform** that carries the steering artifact and edits the
activation, a **selector** (or explicit layer ids) that chooses which layer(s) to steer, and optionally a gate and a
token scope. Here we use the simplest configuration: an additive transform at a single layer.

The transform's artifact is a steering direction. We obtain it from a contrast between examples of the target behavior
(positive, upbeat text) and its opposite (negative, downbeat text). A `ContrastiveFit` holds these pairs and the
extraction settings, and fits one direction per layer when the adapter steers:

```python
from aisteer360.algorithms.state_control.common.sources import ContrastiveFit
from aisteer360.algorithms.core.internals.data import ContrastivePairs

pairs = ContrastivePairs(
    positives=[
        "What a wonderful day, everything is going so well!",
        "I love this, it makes me incredibly happy.",
        "This is fantastic news, I couldn't be more thrilled.",
        "You did an amazing job, I'm so proud of you.",
    ],
    negatives=[
        "What an awful day, everything is going so badly.",
        "I hate this, it makes me miserable.",
        "This is terrible news, I couldn't be more upset.",
        "You did a dreadful job, I'm so disappointed in you.",
    ],
)

positivity = ContrastiveFit(data=pairs, method="mean_diff", accumulate="last_token", prompt_format="raw")
```

We wrap the fitted direction in an `AdditiveTransform`, which adds a scaled copy of it to the residual stream. A
positive `strength` pushes activations toward the positive examples; a negative `strength` pushes the other way. That
transform, placed at a single layer, defines the control:

```python
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform

activation_adapter = ActivationAdapter(
    transform=AdditiveTransform(positivity, strength=1.0),
    layer_ids=16,
    token_scope="all",
)
```

An additive edit is measured against the residual-stream norm, which varies by model and layer, so
`strength` is the knob to tune first: too small and the effect is invisible, too large and the
output degenerates into repetition. Start near `1.0` and adjust for your model and layer.

We can then define a `SteeringPipeline` on a given base model using the above control:

```python
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
adapter_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[activation_adapter],
    device_map="auto",
)
adapter_pipeline.steer()
```

Calling `steer()` fits the direction (one forward pass over the contrastive pairs) and binds it to the transform.
Inference can now be run on the steered pipeline as follows:

```python
prompt = "Tell me about your day."
print(adapter_pipeline.generate(prompt, max_new_tokens=100))
```

`SteeringPipeline.generate` dispatches on keyword: `text=` for a `str` or `list[str]`, `messages=` for chat
messages, and `input_ids=` for a pre-tokenized tensor. A positional `str`/`list[str]` is a convenience for `text=`.
The return shape matches the source (decoded text for text and chat input, a tensor for token input). Pass
`return_output=True` to get an `Output` object instead.

Swapping the transform for a projection (`ProjectionTransform`), the explicit `layer_ids` for a
`layer_selector`, or adding a gate turns this same adapter into other steering methods without writing a new control
class. And there you
have it, a simple activation-steering control. For a full walkthrough of the adapter's slots, as well as examples on
how controls can be compared on a given task, please see the [example notebooks](../examples/index.md).
