# Adding your own steering method

Steering methods span four categories of controls: input, structural, state, and output. The specific category of a
steering method is dictated by what aspects of the model the method influences. Please refer to the conceptual guide on
[steering](../concepts/controls.md) for information on choosing the appropriate category for your method.

## Required files

Once you have determined the steering category, create the following files in `steerability/algorithms`:

```
steerability/
└── algorithms/
        └── <category>/
            └── <custom_control>/
                ├── utils/ (optional)
                ├── __init__.py
                ├── args.py
                └── control.py
```

where `<category>` must be one of the existing directories (`input_control`, `structural_control`, `state_control`, `output_control`) and
`<custom_control>` is the directory name for your method. We encourage you to keep your implementations as
self-contained as possible (within the control class), but any additional files/utils beyond the core implementation
can be placed in a `utils/` directory within `<custom_control>/`. The following outlines how each file (`__init__.py`,
`args.py`, `control.py`) is constructed.



### 1. Registry: `__init__.py`

The `__init__.py` file exposes the method to the toolkit's registry.

```python
from .control import CustomControl
from .args import CustomControlArgs

STEERING_METHOD = {
    "category": "<category>",
    "name": "custom_control",
    "control": CustomControl,
    "args": CustomControlArgs,
}
```

### 2. Arguments dataclass: `args.py`

The args file contains a dataclass that specifies the method's required arguments along with any associated validation
logic.

```python
from dataclasses import dataclass, field
from steerability.algorithms.core.base_args import BaseArgs

@dataclass
class CustomControlArgs(BaseArgs):

    prefix: str = field(
        default="You are an expert assistant.",
        metadata={"help": "Hard-coded text prepended to every user prompt."},
    )
    strip_newlines: bool = field(
        default=True,
        metadata={"help": "Remove trailing newlines from the original prompt before concatenation."},
    )

    # validate
    def __post_init__(self):

        if not self.prefix:
            raise ValueError("`prefix` must be non-empty.")
```

List all parameters that your method takes as input. Each parameter is written as a `field` with two arguments,
`default` (included only if the parameter is optional and omitted if the parameter is required) and `metadata` (a
dictionary containing the description of the argument under the key `help`). Include all validation logic for your
method's parameters in the `__post_init__` method so that validation runs automatically upon initialization.

!!! warning
    Immutable defaults are safe with `default=`, i.e., `int`, `float`, `str`, and `bool` can be given directly (`default=5`, `default=True`, ...), but mutable defaults need `default_factory`. For example, for a `list`, `dict`, `set`, or any custom object you expect to mutate, you must write:
    ```python
    my_list: list[str] = field(default_factory=list, metadata={...})
    ```
    See the [example output control](./add_method_by_category/add_new_output_control.md) implementation for details.


### 3. Control implementation: `control.py`

The control file contains the method's main implementation. The control class does not contain an `__init__` method.
Instead, the method's parameters are handled by the args class via the line `Args = CustomControlArgs`.[^1] The
`__init__` method of the control's base class automatically validates these fields (via `Args.validate`) and converts
them into class attributes.

[^1]: This is intended to minimize boilerplate code (parameter/argument parsing and validation) that would otherwise be needed in each control's `__init__` method.

Any one-time preparation of the steering method is done in the `.steer()` method of the control. This is optional for all
control categories except structural control methods, where the `.steer()` method contains the necessary logic for
modifying the model's weights/architecture. Note that while including a steer method is optional
in every control type other than structural, it is often useful to include one for attaching necessary objects to the
control for later use (e.g., the tokenizer). This is illustrated in the tutorials below.

A control's steer step also declares what it needs from the model via `steer_access()`. The levels are cumulative:
`facts` (layout and tokenizer), `rollouts` (generation and scoring through the session), `capture` (hidden states), and
`module` (the model as a loaded `torch.nn.Module`). Declare the highest level your `steer()` uses. Intervention
templates derive it from their sources, and structural controls are `module` by definition. The pipeline hands your
`steer()` a session scoped to that level, and the model itself only at `module`. On an engine backend, module-level
steps run on a temporary in-process model that is freed before the engine starts, with exported artifacts as the
handoff. Do not keep a reference to the model past `steer()` unless your generate phase requires `IN_PROCESS_TORCH`.

The implementation of a control method depends on its steering category. Specific instructions for adding a method
under each of the four categories, via a small example implementation, are given below:

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called.

    **Required override**: `adapt`

    [:octicons-arrow-right-24: Add your own input control method](./add_method_by_category/add_new_input_control.md)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture.

    **Required override**: `steer`

    [:octicons-arrow-right-24: Add your own structural control method](./add_method_by_category/add_new_structural_control.md)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time.

    **Required override**: `get_hooks`

    [:octicons-arrow-right-24: Add your own state control method](./add_method_by_category/add_new_state_control.md)

-   __Output control__

    ---

    Output control methods influence the model's generations via the decoding process.

    **Required override**: `get_logits_processors` and/or `get_stopping_criteria` (step-level), or `decode` (decoding driver)

    [:octicons-arrow-right-24: Add your own output control method](./add_method_by_category/add_new_output_control.md)

</div>

!!! note
    If your steering method requires two distinct control knobs, e.g., it both rewrites the prompt and constrains
    decoding, split it into two small controls and chain them together in `controls=[...]`.


## Testing your method

To ensure your method is operating as intended, we ask that you write a small unit test in `./tests/controls/`. We
advise that these tests are written using lightweight models (e.g., via
[Hugging Face internal testing](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM)). This allows
the tests to be run locally (on your CPU) before submitting your PR. See the `tests/` directory for examples.


## Document it and write a notebook


Ensure you have written a meaningful docstring for your method in the main control class. Docstrings should contain a
brief description of the method, a reference to the method's paper/documentation, and a list of the method's args
(please use the Google docstring format). An example class docstring (for the `DeAL` method) is given below:

```python
"""
Implementation of DeAL (Decoding-time Alignment) from Huang et al., 2024.

DeAL performs controlled text generation through iterative lookahead search and reward-guided beam selection. Unlike
training-time alignment methods, DeAL operates purely at inference time to steer language model outputs toward
desired behaviors.

The algorithm works in three phases:

1. **Lookahead Generation**: Generate multiple candidate continuations using beam search from the current context.

2. **Reward-based Scoring**: Evaluate each candidate continuation using a provided reward function that measures
alignment with the desired objective (e.g., helpfulness, safety).

3. **Iterative Refinement**: Select the top-k highest-scoring beams and repeat the process until termination
conditions are met (EOS token, max length, or max iterations reached).

DeAL is a decoding driver implemented as a preset of the generic `SearchDriver`, mapping its arguments onto the
search fields (`scorer`, `segment_len`, `num_candidates`, `keep_k`, `max_iterations`, and `propose_mode="beam"`).
The composed logits processors and stopping criteria apply inside every lookahead rollout, which means that a
step-level control such as RAD steers every DeAL rollout. The `reward_params` runtime kwarg is honored per row.

Args:
    reward_func (Callable): Function that scores generated continuations. Should accept
        (prompt: str, continuations: list[str], reward_params: dict) and return list[float].
    lookahead (int): Number of tokens to generate in each lookahead step. Defaults to 10.
    init_beams (int): Number of initial beams to generate at each iteration. Defaults to 5.
    topk (int): Number of top-scoring beams to retain for the next iteration. Defaults to 3.
    max_iterations (int): Maximum number of search iterations before termination. Defaults to 10.

Reference:

- "DeAL: Decoding-time Alignment for Large Language Models"
James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour,
Katrin Kirchhoff, Dan Roth
https://arxiv.org/abs/2402.06147
"""
```


Demonstrate your method by writing a notebook (in `examples/notebooks/algorithms/`). A good notebook
should contain the following:

- A description of what the method does and how it works
- How to initialize the control using the toolkit
- A small example of it working, ideally illustrating how the steered behavior compares with the baseline
(non-steered) behavior

See the [DeAL notebook](../examples/notebooks/algorithms/deal.ipynb) for an example.

A new method also needs its documentation surfaces updated: a reference page
`docs/reference/algorithms/<category>_control/<method>.md` (copy the mkdocstrings block from an existing page), a nav
entry in `docs/.nav.yml`, a mention in the category's list in `docs/concepts/controls.md`, and an entry for the
notebook in `examples/index.md`.
