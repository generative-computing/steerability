# Steering Pipelines

![Steering pipeline](../assets/pipeline_darkmode.png#gh-dark-mode-only)
![Steering pipeline](../assets/pipeline_lightmode.png#gh-light-mode-only)
<p align="center">
  <em>The structure of a steering pipeline.</em>
</p>

Steering pipelines allow for the composition of multiple controls (across the [four control types](controls.md)) into a
single steering operation on a model. This allows for individual controls to be easily *mixed* to form novel steering
interventions.

Steering pipelines are created using the `SteeringPipeline` class. The most common pattern is to specify a Hugging Face
model name via `model_name_or_path` along with instantiated controls, e.g.,
[`few_shot`](../examples/notebooks/algorithms/few_shot.ipynb) and [`dpo`](../examples/notebooks/algorithms/trl.ipynb), as follows:

```python
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

pipeline = SteeringPipeline(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    controls=[few_shot, dpo]
)
```
The above chains the two controls into a single operation on the model.

!!! note
    Some structural controls (e.g., model merging methods) produce a model as output rather than modifying/tuning an
    existing model. In these cases, the steering pipeline is initialized without the `model_name_or_path` argument;
    the structural control supplies the model during the steer step.

!!! note
    A pipeline may contain **any number of controls in every category**, each applied in list order. When multiple
    state controls are supplied, list order is the single, well-defined composition surface: list order = `steer()`
    order = hook registration order = execution order for hooks on the same module. PyTorch forward hooks chain (a
    later hook receives the previous hook's returned output; pre-hooks chain likewise on inputs), so a combination like
    "control A then control B at layer 12" is well-defined, and non-commuting pairs (e.g. ablation ∘ addition vs.
    addition ∘ ablation) are order-sensitive by design. An `ActivationAdapter` is the natural single-behavior atom
    here, i.e., steering with N behaviors is N adapters in the `controls` list.

!!! note "Input controls: two-phase chaining"
    Multiple input controls chain in list order across two phases. On chat input, every control's `adapt_messages`
    runs in list order over the message batch (each non-None return feeds the next control); the result is templated
    and tokenized once, then every control whose `adapt_messages` returned None runs its token-level `adapt` in list
    order over the token stream. On text/tensor input there is no message phase; every control's `adapt` runs in list
    order. Each control is applied exactly once per generation: at message level if its `adapt_messages` returned a
    non-None result for that call, else at token level. List order is authoritative within each phase, but the message
    phase structurally precedes the token phase: with `[TokenOnlyControl, MessageLevelControl]` on chat input, the
    message-level control's effect lands first even though it is listed second (tokens do not exist before
    templating). Recommended ordering: place semantic rewriters (`PRewrite`, `CPO`, `GEPA`) before surface formatting
    (`FewShot`), since a rewriter trained on bare instructions degrades on exemplar-prepended input.

!!! note "Structural controls: model threading"
    Multiple structural controls thread the model through `steer()` in list order: each control receives the previous
    control's returned model (and the possibly mutated tokenizer). Nothing implicit happens between stages, i.e., no
    adapter merging and no embedding-resize reconciliation; stage compatibility (a PEFT-wrapped model into a second
    trainer, resized embeddings, and the like) is the user's responsibility. Note that the TRL wrapper controls load
    their own base model when `base_model_name_or_path` is set in their args, silently discarding the threaded
    upstream model, so downstream structural controls should leave `base_model_name_or_path` unset to receive the
    threaded model.

!!! note "Output controls: step-level controls compose, the decode loop does not"
    Output controls participate through two mechanisms. Most are step-level controls supplying logits processors and/or
    stopping criteria, which the pipeline gathers in `controls`-list order, then appends any per-call
    `logits_processor` / `stopping_criteria` supplied in `generate()`, into one authoritative stack of each kind. The
    decode loop itself is exclusive. It is owned by at most one `DecodingDriver`, and supplying two enabled drivers
    raises at construction (two decoding procedures cannot both control generation). With no driver present, the loop
    defaults to the model's own `generate`, so a pipeline with no output controls decodes exactly as the base model
    does. Because the loop is a single owner while step-level controls compose, a step-level control (e.g. `RAD`)
    applies inside every rollout a driver issues (e.g. `DeAL`'s lookahead), a composition rather than a conflict.
    Step-level controls' logits processors also apply during `compute_logprobs`, so scoring reflects the steered
    next-token distribution; a control sets `include_in_scoring=False` to opt out (e.g. when the per-position cost is
    prohibitive).

## Steering the pipeline

Before a steering pipeline can be used for inference, all of the controls in the pipeline must be prepared and applied
to the model (e.g, training logic in a `DPO` control, or subspace learning in the `SASA` control). This step is referred
to as the *steer* step and is executed via:

```python
pipeline.steer()
```

Calling the `steer()` method on a pipeline instance invokes the steering logic for every control in the pipeline. Methods are
steered independently; the effect of composing steered/trained controls is one of the main functionalities provided by the
toolkit. Note that the `steer()` step can be resource-heavy, e.g., especially if any of the controls in the pipeline require any training.
Steering must be called before using the pipeline for inference; a repeated `steer()` call is a no-op.


## Execution backends

Pipelines execute on a configurable backend. By default, the pipeline loads and runs the model *in process* (via
Hugging Face `transformers`); passing `backend=` selects the offline vLLM engine (`kind="vllm"`) or a
running vLLM server (`kind="vllm-serve"`). Support is binary per control configuration and backend:
`pipeline.check()` returns a report with one verdict per unsupported (control, phase) pair, naming the gap and the
fix, and `steer()` runs the same check and raises before any work happens. The per-control support boundary is
recorded on each control's `Backends` line in [steering controls](controls.md).

```python
from aisteer360.algorithms.core.execution import BackendSpec

pipeline = SteeringPipeline(
    controls=[caa],
    backend=BackendSpec(
        kind="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        options={"hook_plugin": True},
    ),
)
report = pipeline.check()  # optional standalone check; steer() runs it and raises on failures
report.plan               # where each control's steer step and each fit will run
```

The above fits `caa` through the engine's capture surface and generates through the vLLM-Hook plugin.

### Scoring rule

Intervention controls score in-process only, since remote prompt-logprob scoring anchors token scopes at the
request's prompt end (the end of the prompt-plus-reference concatenation), which would silently unanchor
prompt-relative interventions. An enabled output control with `include_in_scoring=True` likewise makes the pipeline
score-unsupported off-torch, and encoder-decoder scoring is in-process-only.

### The model-access ladder

Each control declares its steer step's model access via `steer_access()`, on the cumulative `ModelAccess` ladder.
The pipeline satisfies every declaration deterministically; `check()` returns the resulting steer plan alongside
the generate and score verdicts.

| Rung | Grants | HF venue | vLLM offline (plugin) | vLLM serve |
| --- | --- | --- | --- | --- |
| `facts` | `session.layout` and a tokenizer | live model | engine session | engine session |
| `rollouts` | facts plus generation and scoring through the session | live model | engine session | engine session |
| `capture` | rollouts plus hidden-state capture through the session | live model | engine session (staged when capture is absent or `fit="in_process"`) | staged model |
| `module` | the model as a live `torch.nn.Module` | live model | staged model | staged model |

On engine backends the staged in-process model is loaded, used, and freed before the engine boots; exported
artifacts are the handoff, so the pipeline's in-process weights and its engine-served weights never coexist.
`fit="in_process"` forces every fit onto the stage for engine-independent numerics; a calibrated artifact fitted in
process while its read venue is an engine warns that its thresholds may shift across execution boundaries.

### Lifecycle

Backends are constructed lazily per pipeline and cached by spec. `SteeringPipeline.release_backends()`, or using the
pipeline as a context manager, releases and evicts every backend the pipeline constructed, shutting engine-owning
backends down deterministically rather than waiting for garbage collection. A released pipeline stays usable. The
next operation reconstructs backends against the same specs, so a re-booted engine serves subsequent generations.
`Benchmark` releases each configuration's backends automatically after its trials. The offline engine's release is
process-global with respect to vLLM distributed state, so it assumes no other live vLLM engine in the process.

```python
with SteeringPipeline(controls=[caa], backend="vllm") as pipeline:
    pipeline.steer()  # fits stage or ride the engine session per the steer plan
    response = pipeline.generate(text="...", max_new_tokens=64)
# the engine is shut down on exit
```

### Benchmarking

`Benchmark` forwards its `backend` and `fit` arguments to the pipelines it builds and pre-flights support over every
sweep point (via `SteeringPipeline.check()`) before any model or engine work, so the per-control support recorded on
each control's `Backends` line in [steering controls](controls.md) governs benchmarking too. A sweep point that is
unsupported on the configured backend either fails the whole run (`on_unsupported="raise"`, the default) or is
skipped with a warning (`on_unsupported="skip"`).

### Running a server

The offline vLLM engine (`BackendSpec(kind="vllm")`) boots vLLM inside the current process, so it needs no server and
is the automatic path for single-process runs. The serve backend targets a vLLM server you launch yourself, which is
the answer for a remote GPU box, one server shared across processes or benchmark runs, a client with no local vLLM
install, or process isolation from the steering client.

Start a server with `vllm serve <model> --port 8000` (any extra engine flags as usual), then target it with a spec
carrying `base_url`:

```python
from aisteer360.algorithms.core.execution import BackendSpec

spec = BackendSpec(
    kind="vllm-serve",
    model="meta-llama/Llama-3.1-8B-Instruct",
    options={"base_url": "http://localhost:8000"},
)
```

When serving activation interventions through the vLLM-Hook plugin, the serving environment carries the plugin, the
server starts with `VLLM_HOOK_WORKER=unified` and eager execution, the spec adds `hook_plugin: True`, and
`artifact_dir` names the server's registry directory (its `VLLM_HOOK_REGISTRY_DIR`) on a filesystem shared with the
server; without `artifact_dir` the client PUTs artifacts over the server's artifact route instead.


## Running inference on the pipeline

Once the pipeline has been steered, inference can be run using the `generate()` method. The prompt source is declared
by keyword, with exactly one source per call: `text=` for a `str` or `list[str]`, `messages=` for one conversation
(a sequence of chat-message mappings) or a batch of conversations, and `input_ids=` for a pre-tokenized 1-D/2-D
integer tensor (`attention_mask` is valid only alongside `input_ids=`, and is derived automatically for `text=` and
`messages=`). A positional `str`/`list[str]` is also accepted as a convenience for text prompts. Unlike bare
`model.generate`, the returned token ids exclude the prompt by default; pass `return_full_sequence=True` for
prompt-plus-continuation output. The `text=` and
`messages=` paths tokenize for you, so passing chat directly is the most direct route:

```python
output = pipeline.generate(
    messages=[{"role": "user", "content": PROMPT}],
    max_new_tokens=20,
)
```

For reasoning models that toggle thinking through a chat-template keyword, we pass `chat_template_kwargs` alongside
`messages=`. This mapping is forwarded to `apply_chat_template` and is not interpreted by the toolkit, so the keys
are whatever the model family expects (for example `enable_thinking`). It is valid only with `messages=`, and pairing
it with `text=` or `input_ids=` raises a `TypeError`.

```python
output = pipeline.generate(
    messages=[{"role": "user", "content": PROMPT}],
    chat_template_kwargs={"enable_thinking": False},
    max_new_tokens=20,
)
```

To tokenize explicitly and pass token IDs, encode via the pipeline's tokenizer, applying the chat template if
available:

```python
tokenizer = pipeline.tokenizer
chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(chat, return_tensors="pt")
```

Inference can then be run as usual, for instance:
```python
gen_params = {
    "max_new_tokens": 20,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

steered_output_ids = pipeline.generate(
    input_ids=inputs.input_ids,
    **gen_params,
)
```

On the default in-process backend, steering pipelines accept any of the generation parameters available in
[Hugging Face's `GenerationConfig` class](https://huggingface.co/docs/transformers/en/main_classes/text_generation),
including the generation strategies for [custom decoding](https://huggingface.co/docs/transformers/en/generation_strategies).
Generation parameters are normalized across backends: the sampling-facing subset (e.g., `max_new_tokens`,
`temperature`, `top_p`, `stop_strings`) is portable, while parameters outside it pass through to `model.generate` in
process and raise on the vLLM backends.
