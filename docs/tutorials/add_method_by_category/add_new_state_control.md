# Adding a state control method

**Required override**: an intervention template in `_configure` (declarative methods) or `get_hooks` (custom hooks)

State control methods steer by editing the model's internal states during the forward pass. Most methods are
declarative, i.e., the control states its behavior once, as a tuple of interventions, and the toolkit compiles that
statement for whichever backend runs it (torch hooks in process, intervention specs on engine backends). As part of
this tutorial, we'll implement an `ActivationBias` method that adds a fixed bias vector, scaled by `alpha`, to the
hidden state output at a specified transformer layer.

First, create the registry file:

```python
from .control import ActivationBias
from .args import ActivationBiasArgs

STEERING_METHOD = {
    "category": "state_control",
    "name": "activation_bias",
    "control": ActivationBias,
    "args": ActivationBiasArgs,
}
```

Next, define the arguments class. This is where we define the required arguments; the transformer layer (via
`layer_idx`) and the bias (via `alpha`):

```python
from dataclasses import dataclass, field
from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class ActivationBiasArgs(BaseArgs):
    layer_idx: int = field(
        default=0,
        metadata={"help": "Transformer block to patch."}
    )
    alpha: float = field(
        default=0.02,
        metadata={"help": "Bias magnitude."}
    )

    def __post_init__(self):
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
```

## Declarative controls

A declarative control subclasses `InterventionControl` and maps its validated args onto an intervention template in
`_configure`. An `Intervention` names the behavior layers (explicit ids or a selector resolved at steer time), a
transform (which may carry an `ArtifactSource` fitted at steer time), a `TokenScope`, and optionally a gate and
condition. The base class does the rest: `steer()` binds the template against the model (or a remote session's
structural facts), hooks are built once per generation by the pipeline, and configurations whose components all
have a wire form run on vLLM backends through the vLLM-Hook plugin with no extra code.

`ActivationBias` is an additive edit, so its template is one intervention over an `AdditiveTransform`:

```python
import torch

from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform
from aisteer360.algorithms.state_control.base import InterventionControl
from aisteer360.algorithms.state_control.activation_bias.args import ActivationBiasArgs

HIDDEN_SIZE = 4096  # or resolve from the artifact you steer with


class ActivationBias(InterventionControl):
    """Adds a fixed bias to hidden states at the selected layer."""

    Args = ActivationBiasArgs

    def _configure(self):
        bias = {self.layer_idx: torch.full((1, HIDDEN_SIZE), self.alpha)}
        self._template = (Intervention(
            layers=(self.layer_idx,),
            transform=AdditiveTransform(bias),
            scope=TokenScope("all"),
        ),)
```

There is no hook code, no per-generation state, and no backend knowledge in the control. The shipped residual-stream
methods (`caa`, `act_add`, `directional_ablation`, `angular_steering`, `cast`, `iti`, and the composable
`activation_adapter`) all follow this pattern; read them for templates that fit artifacts from data
(`ContrastiveFit`), select layers at steer time (`FractionalDepthSelector`, `CoveredLayers`), or gate
conditionally (`ConditionPointSearch`, `gate_from_probe`).

## Custom hook controls

A method that hooks a mechanism the intervention vocabulary does not cover (for example attention weights, as in
PASTA) subclasses `HookControl` and implements `get_hooks`. The hooks travel as entries on session items and the
session that executes forwards owns registration, so `get_hooks` must fully re-derive its state on every call:

```python
import torch

from aisteer360.algorithms.state_control.base import HookControl, HookSpec
from aisteer360.algorithms.state_control.activation_bias.args import ActivationBiasArgs


class ActivationBiasHooks(HookControl):
    """Adds alpha to hidden states at the selected layer (raw-hook variant)."""

    Args = ActivationBiasArgs

    def get_hooks(
            self,
            input_ids: torch.Tensor,
            runtime_kwargs,
            **__
    ) -> dict[str, list[HookSpec]]:
        """Returns a forward hook that adds alpha to a specific layer's output.

        Args:
            input_ids (torch.Tensor): Input tensor (unused).
            runtime_kwargs: Optional runtime parameters (unused).

        Returns:
            dict[str, list[HookSpec]]: A dictionary mapping hook phases ("pre", "forward", "backward") to lists of hook
            specifications. Each HookSpec contains:
              - "module": The name of the module to hook
              - "hook_func": The hook function to apply (pre, forward, or backward)
        """

        def fwd_hook(module, args, kwargs, output):

            # handle different output formats
            if isinstance(output, tuple):
                return (output[0] + self.alpha,) + output[1:]
            elif isinstance(output, dict):
                output = output.copy()
                output['hidden_states'] += self.alpha
                return output
            else:  # direct tensor
                return output + self.alpha

        return {
            "pre": [],
            "forward": [{
                "module": f"model.layers.{self.layer_idx}",
                "hook_func": fwd_hook,
            }],
            "backward": [],
        }
```

## Position tracking in hooks

Scoped intervention controls get position tracking for free. `build_hooks` compiles every intervention through the
shared `TransformHookRuntime`, which reads each pass's absolute offset from the `cache_position` kwarg when the
hooked module receives it and falls back to pass counting otherwise, with exactly one designated pass-opener hook
advancing the shared offset per forward pass.

A custom `HookControl` honoring `token_scope="after_prompt"` or `"from_position"` needs the same care. During
prefill the hook sees the whole prompt (`seq_len == prompt_len`); during KV-cached decode it sees only the newly
generated token(s) (`seq_len == 1`). Do **not** infer the phase by comparing `seq_len` to the prompt length, since a
length-1 prompt makes prefill and decode indistinguishable and steering would then silently never fire. Track the phase
in state the hook closures own, created fresh inside `get_hooks` so every generation starts clean:

```python
# inside get_hooks(), before building the hook closures:
state = {"position_offset": 0, "prefill_seen": False}

# inside the hook function:
seq_len = hidden.size(1)
if state["prefill_seen"]:         # decode step (or a later chunk)
    position_offset = state["position_offset"]
    state["position_offset"] += seq_len
else:                             # first pass of this generation == prefill
    position_offset = 0
    state["position_offset"] = seq_len
    state["prefill_seen"] = True

mask = make_token_mask(self.token_scope, seq_len=seq_len, prompt_lens=prompt_lens,
                       position_offset=position_offset)
```

If a control registers several hooks per pass (e.g. one per layer), designate a single hook to advance the
shared counter and gate both the advance and the flag flip on it, so earlier hooks in the same prefill pass
still read `position_offset = 0`.

## Using the control

The session executing the generation registers the hooks for exactly the span of the work, so the control can be
used like any other:

```python
from aisteer360.algorithms.state_control.activation_bias.control import ActivationBias
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

activation_bias_control = ActivationBias(layer_idx=2, alpha=0.03)

activation_bias_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[activation_bias_control],
)
activation_bias_pipeline.steer()

prompt = "What should I do in Prague?"
print(activation_bias_pipeline.generate(prompt, max_new_tokens=50))
```
