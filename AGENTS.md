# AGENTS.md

Guidance for AI agents working in this repository, covering both using the toolkit as a library (Usage guide) and
extending it (Developer guide). The Invariants section lists rules that apply to every task. When this file and the
code disagree, the code is authoritative; verify against the source before acting on a claim here.

## Overview

AISteer360 is a toolkit for steering large language models (Hugging Face causal LMs). It provides steering methods
("controls") across four model control surfaces, a `SteeringPipeline` that composes controls from any categories into
one operation on a model, and an evaluation stack (use cases, metrics, benchmarks) for comparing steering pipelines.

Pipelines execute on one configurable backend: the in-process Hugging Face backend (default), the offline vLLM
engine (`kind="vllm"`), or a vLLM server (`kind="vllm-serve"`). Support is binary per control configuration and
backend for the generate and score phases; `pipeline.check()` reports unsupported combinations with a verdict naming
the gap and the fix, and unsupported operations raise before any work happens. The steer phase produces no verdicts:
each control declares its steer step's model access on the `ModelAccess` ladder (`facts` < `rollouts` < `capture` <
`module`), and `check()` additionally returns a deterministic steer plan stating where each step and fit will run
(see Execution backends below).

The four control categories, defined by what a method touches:

- **input**: manipulates the prompt only; generations follow `y ~ p_theta(sigma(x))` for a prompt adapter `sigma`.
- **structural**: persistently modifies weights or architecture (fine-tuning, DPO, merging); `y ~ p_theta'(x)`.
- **state**: edits internal activations at runtime via forward hooks, without changing weights.
- **output**: shapes the decoding process (logits processing, stopping, re-ranking, custom decode loops).

Vocabulary used throughout the codebase:

- **control**: one steering method, subclassing the base class of its category.
- **generic**: a dedicated recipe control class (`activation_adapter`, `value_guidance`, `search_decoding`, ...) that
  exposes common component slots through flat, sweepable `Args`, so a method from the literature is a configuration
  rather than a new class; named methods are siblings of generics, not children.
- **common library**: the per-category building blocks in `common/` (transforms, gating, drivers, selectors,
  formatters, ...) from which generics and named methods alike are assembled.
- **probe**: a calibrated linear readout over hidden states used for detection (reads, never edits); gating and
  routing consume its decisions.

## Repository map

```
aisteer360/
├── algorithms/
│   ├── core/                    # SteeringPipeline, registry, ControlSpec, BaseArgs, shared types
│   │   ├── execution/           # backend seam: spec, contracts, payloads, backend/session/registry, params, fanout
│   │   ├── internals/           # activation capture, pooling, stats; probes/ (detection)
│   │   └── utils/               # control merging, generation helpers, auxiliary_pass
│   ├── input_control/           # each category: base.py + one folder per method (triplet layout below)
│   │   └── common/             # building blocks: memory, formatters, proposers, scorers, selectors
│   ├── state_control/
│   │   └── common/             # building blocks: transforms, estimators, gating, selectors, hook runtime
│   ├── output_control/          # methods incl. routed_decoding/ (control, routing.py, actions.py)
│   │   └── common/             # building blocks: drivers, processors, scorers, values, criteria
│   └── structural_control/
│       └── wrappers/            # trl/ (sft, dpo, ppo, grpo, apo) and mergekit/
├── backends/                    # huggingface/ (HFBackend, ExclusiveSession); vllm/ (VLLMBackend, VLLMServeBackend)
├── evaluation/
│   ├── benchmark.py             # Benchmark runner (trials, sweeps, checkpoint/resume)
│   ├── metrics/                 # base.py, base_judge.py; generic/ and custom/<use_case>/
│   ├── use_cases/               # base.py; one folder per use case (use_case.py)
│   └── utils/                   # data_utils, generation_utils, metric_utils, viz_utils
└── utils/                       # tokenization, rendering, optional-dependency guard

docs/                            # MkDocs site: home/, concepts/, tutorials/, reference/, .nav.yml
examples/                        # notebooks/{algorithms,generics,benchmarks,recipes}/ + index.md
tests/                           # controls/, core/, internals/, evaluation/, utils/; conftest.py
```

## Setup and commands

Python 3.11+ with `uv` as the package manager:

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
source .venv/bin/activate
```

On Windows, run the two chained commands separately. Optional extras: `merging` (MergeKit), `cpo` (econml), `plots`
(matplotlib/seaborn), `vllm` (the vLLM backends plus the `vllm_hook_plugins` core, git-pinned until its PyPI
release), `guided` (xgrammar, for in-process constrained decoding), `all` (all features except `vllm` and `guided`), `dev`
(`all` plus the plugin core, pytest, pre-commit, notebook), `docs` (site tooling).

Hugging Face access uses a `.env` file at the repo root containing `HUGGINGFACE_TOKEN=hf_***` (see
`.env.example`). Some models (e.g. `meta-llama/*`) are gated; the account behind the token needs access on the
model's Hub page. Never commit tokens; a detect-secrets pre-commit hook scans against `.secrets.baseline`.

Models run inside the current process on the default Hugging Face backend; the vLLM backends execute on a local
engine or a remote server instead. Real steering runs need GPU memory for the base checkpoint plus the method's
overhead; for smoke tests use the tiny models listed in `tests/utils/ci_models.yaml`
(e.g. `hf-internal-testing/tiny-random-LlamaForCausalLM`).

Common commands:

```bash
pytest tests/controls/                    # all control tests
pytest tests/controls/test_pasta.py       # one control
pytest tests/core/ tests/internals/       # pipeline, registry, probes
pre-commit install                        # once per clone
pre-commit run --all-files                # detect-secrets, whitespace, isort (black profile)
uv pip install -e ".[docs]" && uv run mkdocs serve   # docs at localhost:8000
```

Tests parametrize over the models in `tests/utils/ci_models.yaml` and over devices (`cpu`, `cuda`, `mps`); unavailable
devices are skipped automatically, so the suite runs on CPU-only machines. Commit messages require a DCO
`Signed-off-by:` line (see `CONTRIBUTING.md`).

## Usage guide

### Minimal steering loop

Every use of the toolkit follows the same loop: instantiate controls, wrap them in a `SteeringPipeline`, call
`steer()` once, then call `generate()` for inference.

```python
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.few_shot.control import FewShot

few_shot = FewShot(
    directive="Answer in a formal, professional tone.",
    positive_example_pool=[
        {"prompt": "hey what's up", "response": "Good afternoon. How may I assist you?"},
        {"prompt": "thx", "response": "You are most welcome."},
    ],
    k_positive=1,
)

pipeline = SteeringPipeline(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",  # any HF causal LM id or local path
    controls=[few_shot],
    device_map="auto",
)
pipeline.steer()  # required once before generate(); heavy work (model loading, training, fitting) happens here

response = pipeline.generate(
    messages=[{"role": "user", "content": "Where is the Eiffel Tower?"}],
    max_new_tokens=64,
)
```

Constructor arguments for a control are defined by its `Args` dataclass (in the method's `args.py`) and validated at
construction. For lightweight controls `steer()` only attaches artifacts like the tokenizer; for controls that train
(structural controls, activation-steering fits) the training runs there.

### Choosing a control category

| Goal | Category | Base class |
| --- | --- | --- |
| Change the prompt (few-shot, rewriting, prompt search) | input | `InputControl` |
| Change the weights (fine-tune, DPO, merge) | structural | `StructuralControl` |
| Edit activations or attention at runtime | state | `StateControl` |
| Shape decoding (rerank, guided sampling, custom loops) | output | `OutputControl` / `DecodingDriver` |

### Available controls

Enumerate the live registry rather than trusting any static list:

```python
from aisteer360.algorithms.core.registry import REGISTRY  # import triggers discovery

for category, methods in REGISTRY.items():
    print(category, sorted(methods))
```

The registered names at the time of writing:

- input: `cpo`, `few_shot`, `gepa`, `prewrite`
- state: `act_add`, `activation_adapter`, `angular_steering`, `caa`, `cast`, `directional_ablation`, `iti`, `pasta`
- output: `best_of_n`, `budget_forcing`, `constrained_decoding`, `contrastive_decoding`, `contrastive_guidance`,
  `deal`, `dexperts`, `phased_decoding`, `rad`, `routed_decoding`, `sasa`, `search_decoding`, `stopping_rules`,
  `value_guidance`
- structural: `mergekit`, `sft`, `dpo`, `ppo`, `grpo`, `apo` (MergeKit and TRL wrappers)

### Pipeline semantics

`generate()` dispatches on the declared keyword source (exactly one per call) and returns the matching shape:

| Source | Tokenization | Return |
| --- | --- | --- |
| `text=` (`str`) | plain text | `str` |
| `text=` (`list[str]`) | batched text | `list[str]` |
| `messages=` (one chat) | chat template | `str` |
| `messages=` (batch of chats) | batched chat template | `list[str]` |
| `input_ids=` (tensor / token id lists) | passed through | `torch.Tensor` |

Positional `str`/`list[str]` behaves like `text=`; any other positional shape raises a `TypeError`. The
per-source methods `generate_text`, `generate_messages`, and `generate_tokens` sit alongside `generate()`
with the same behavior and named parameters for the reserved keys. Decoded text returns carry exactly one
candidate per prompt: `num_return_sequences`/`n` greater than 1 with `text=`/`messages=` raises `ValueError`
unless `return_output=True` (one `output_ids` row and one finish reason per candidate); the token return is
`[batch * n, gen_len]` with each prompt's candidates contiguous, as in `model.generate`.

Behaviors that differ from bare Hugging Face usage:

- Returned token ids exclude the prompt by default. Do not slice the result by prompt length; pass
  `return_full_sequence=True` for HF-style prompt-plus-continuation output.
- `chat_template_kwargs` is a reserved key inside `gen_kwargs`, forwarded to `apply_chat_template`
  after the pipeline-owned template kwargs. It is valid only with `messages=` (pairing it with
  `text=`/`input_ids=` raises `TypeError`), may not name a pipeline-owned template kwarg
  (`return_tensors`, `padding`, `add_generation_prompt`, `return_dict`), and is not interpreted by
  the toolkit (keys are model-family specific, e.g. `enable_thinking`). Because it rides inside
  `gen_kwargs`, thinking-on and thinking-off runs get distinct benchmark checkpoint identities.
- Token ids are returned as generated on every backend (stop text and any token-boundary overrun stay in the ids);
  decoded continuation text is truncated at the first stop-string occurrence by one client-side rule.
- `generate(..., return_output=True)` returns an `Output` object (or list of them) with fields `output_ids`,
  `adapted_input_ids` (the prompt after input controls, useful for inspecting the steered prompt), a per-item
  `finish_reason` (`"stop"`, `"eos"`, `"length"`, or `None`, with that precedence), and `finish_reasons` (one reason
  per candidate for `n > 1`). Import it via `from aisteer360.algorithms.core import Output`.
- `generate()` before `steer()` raises `RuntimeError`; a second `steer()` call is a silent no-op.
- `attention_mask` is valid only with `input_ids=`; it is derived automatically for `text=` and `messages=`, and passing it with either (or with positional text) raises a `TypeError`.
- `device` and a non-default `device_map` are mutually exclusive on the `SteeringPipeline` constructor.
- Construction never loads the model. `steer()` acquires it from `model_name_or_path`, reuses preloaded
  `model=`/`tokenizer=` objects passed at construction, or receives it from a structural control that produces the
  final weights itself (e.g. `mergekit`). `lazy_init` is accepted and inert.
- `pipeline.supports_batching` is `True` only when every enabled control declares batch safety; evaluation utilities
  batch when it is `True` and fall back to per-example generation otherwise.
- `pipeline.compute_logprobs(input_ids, ref_output_ids=...)` scores reference tokens teacher-forced with the full
  steering applied; output controls with `include_in_scoring=False` are excluded from scoring.
- Controls with a `tokenizer` attribute left as `None` get the pipeline tokenizer injected automatically.

### Execution backends

`SteeringPipeline` takes `backend=`, a `BackendSpec` or a kind string. The default is the in-process Hugging Face
backend, and pipelines that never name a backend behave exactly as before. `fit=` (`"auto"` or `"in_process"`)
selects the fit venue policy.

```python
from aisteer360.algorithms.core.execution import BackendSpec

pipeline = SteeringPipeline(
    controls=[caa],
    backend=BackendSpec(kind="vllm", model="meta-llama/Llama-3.1-8B-Instruct", options={"hook_plugin": True}),
)
```

- `pipeline.check()` returns a `SupportReport` without doing any work; `steer()` runs it and raises
  `UnsupportedPipelineError` for unsupported control/backend combinations at generate. Verdict messages are
  stable tested strings naming the gap and the fix. The report also carries `plan`, the deterministic steer
  plan (per-control access and venue, per-fit venue, whether a stage runs, and the warnings that will fire).
  The per-control support boundary is recorded on each control's `Backends` line in `docs/concepts/controls.md`.
- The steer phase satisfies each control's declared `steer_access()` by venue: `facts` and `rollouts` run
  through the backend's session on every kind, `capture` runs through session capture where the spec
  advertises it (the offline plugin engine) and on a staged in-process model where not (serve, or
  `fit="in_process"`), and `module` always stages. On engine backends the staged model is loaded, used, and
  freed before the engine boots; exported artifacts are the handoff, so in-process weights and engine-served
  weights never coexist. If engine capture fails a steer-time smoke test, fitting degrades to the stage with a
  warning; support verdicts never depend on the plugin's presence.
- Activation-steering state controls execute on vLLM through the vLLM-Hook plugin (`hook_plugin: True` on the
  spec): the control's steering tuple serializes as an intervention spec, and tensor payloads travel as
  content-addressed artifacts (`artifact_dir` option; on serve this must be a filesystem shared with the server).
  A configuration either serializes exactly or is honestly in-process-only; there is no approximate lowering.
- Structural controls train on the staged model and serve their artifacts (checkpoint or LoRA) on vLLM backends.
- Declarative constrained decoding lowers to vLLM's native structured outputs; hidden-state capture (probe
  fitting and reads, routed decoding) is served in process and on the offline plugin engine, not on serve.
- `compute_logprobs` scores through the backend; an enabled output control with
  `include_in_scoring=True` keeps scoring in-process.
- Discarding a pipeline that booted a vLLM engine should go through `release_backends()` (or a
  `with` block over the pipeline) rather than relying on garbage collection, which is not prompt at
  freeing the engine. `Benchmark` does this per configuration.

### Composition rules

- A pipeline accepts any number of controls per category. `steer()` runs in a fixed bottom-up order (structural, then
  input, then state, then output) and in list order within each category, so higher layers always see the final model.
- At most one enabled `DecodingDriver` may be present; the decode loop does not compose. When none is supplied the
  pipeline uses `model.generate`. Step-level output controls (processors, stopping criteria) compose freely.
- Input controls run in two phases on chat input. Every control's `adapt_messages` runs in list order before chat
  templating; controls whose `adapt_messages` returns `None` then run their token-level `adapt` in list order after
  tokenization. Text and tensor inputs skip the message phase entirely. Place semantic rewriters (`prewrite`, `cpo`,
  `gepa`) before surface formatters (`few_shot`) in the controls list.
- State hooks register in list order and chain on the same module, so non-commuting edits (e.g. ablate then add
  versus add then ablate) are order sensitive.
- Structural controls thread the model: each receives the model returned by the previous one, with no implicit
  reconciliation between stages.

### Runtime kwargs

Some controls need per-call information at inference time. All controls read from the single `runtime_kwargs` dict
passed to `generate()`; each control declares the names it consumes in its `RUNTIME_KWARGS_SCHEMA`, and the pipeline
warns at `steer()` time when two controls declare the same name (they will share one value).

```python
pipeline.generate(
    prompt,
    runtime_kwargs={"substrings": ["The answer must be in JSON."]},  # e.g. pasta's emphasis spans
    max_new_tokens=128,
)
```

### Evaluation

A `Metric` implements `compute(responses, prompts=None, **kwargs) -> dict` (subclass `LLMJudgeMetric` for judge-based
scoring). A `UseCase` bundles `evaluation_data` and `evaluation_metrics` and implements `generate()` and `evaluate()`.
A `Benchmark` compares steering pipelines on one use case:

```python
from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.evaluation.benchmark import Benchmark

benchmark = Benchmark(
    use_case=use_case,
    base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    steering_pipelines={
        "baseline": [],  # empty list denotes the unsteered baseline
        "few_shot": [few_shot],
        "caa_sweep": [ControlSpec(control_cls=CAA, params={"data": caa_train_data}, vars={"multiplier": [1.0, 2.0]})],
    },
    runtime_overrides={"PASTA": {"substrings": "emphasis_column"}},  # routed by control class name
    num_trials=3,
    seed=7,  # derives one seed per (config, trial); recorded on each run dict
    save_dir="runs/exp1",  # versioned checkpoint.json; resume completes only missing trials
)
profiles = benchmark.run()
```

`ControlSpec.vars` accepts a mapping (cartesian grid, traversed fully or sampled via `search_strategy="random"` and
`num_samples`), a sequence of parameter dicts, or a callable yielding dicts given a context. Each trial reuses the
same steered model and re-samples generate-time randomness; setting `seed=` derives one seed per (config, trial),
threads it through `gen_kwargs` into core's seed path and into use-case-side RNG, and records it on the run dict, so
a resumed trial reproduces what an uninterrupted trial would have sampled (same hardware, dtype, and torch/vLLM
versions). On the in-process Hugging Face backend, pipelines with a structural control load a fresh model while
others reuse a shared preloaded base model; `runtime_overrides` is keyed by control class name, so two instances of
one class in a pipeline share a single entry.

`backend=` and `fit=` forward to the pipelines the benchmark builds (a `BackendSpec` or a known kind name);
before any model or engine work, a pre-flight `check()` over every sweep point either raises one aggregate error
(`on_unsupported="raise"`, the default) or skips the unsupported points with a warning (`on_unsupported="skip"`). Only
a checkpoint whose identity metadata matches (`format` first, then model, backend, fit, use case, and digests)
resumes; a well-shaped envelope from a different configuration or an earlier format is refused naming the differing
field, and anything unreadable or wrong-shaped at the checkpoint path is ignored with one warning and overwritten on
the next save.

Every benchmark generation, baseline included, routes through `pipeline.generate(messages=...)` (or `text=` for a
template-less tokenizer), so the pipeline owns chat templating, tokenization, and padding, `adapt_messages` input
controls fire during benchmarking, and `runtime_overrides` columns live on the prompt rows (aligned under retry and
prompt expansion by construction). The shared-preloaded-model reuse and its fingerprint tripwire are Hugging Face
features: after each shared-base configuration, the tripwire checks the shared model for mutation and, on detecting
one, warns naming the configuration and reloads a clean base for the next.

`batch_retry_generate` and `generate_on_pipeline` split each decoded continuation into a thinking segment and an
answer segment (the `think_tags` parameter, default `("<think>", "</think>")`). Metrics and `parse_fn` see the answer
segment only, so reasoning tokens do not blend into scoring; the thinking segment is retained and the built-in use
cases store it under a `"thinking"` generation-dict column (`str | None`). Pass `think_tags=None` to disable the
split and score the full continuation. A generation that opens a thinking segment but never closes it (the budget was
spent thinking) logs one warning naming the count.

## Developer guide

### Adding a steering control

A method is a sub-package of its category directory with a three-file layout:

```
aisteer360/algorithms/<category>_control/<method_name>/
├── __init__.py    # exports STEERING_METHOD for registry discovery
├── args.py        # hyperparameter dataclass (single source of truth)
├── control.py     # the control class; all steering behavior lives here
└── utils/         # optional local helpers; use sparingly
```

`args.py` defines every hyperparameter as a `BaseArgs` dataclass; use `__post_init__` for cross-field validation:

```python
from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class MyMethodArgs(BaseArgs):
    """Arguments for MyMethod."""

    strength: float = field(default=1.0, metadata={"help": "Scale applied to the steering edit."})

    def __post_init__(self):
        if self.strength <= 0:
            raise ValueError("strength must be positive.")
```

`control.py` subclasses the category base and sets `Args = MyMethodArgs`. The base constructor validates the args and
promotes every field to an instance attribute (`self.strength`), so the control class defines no `__init__` of its
own in the common case. Required hooks per category:

- **input**: `adapt(input_ids, runtime_kwargs) -> input_ids` (required); optionally `adapt_messages(messages,
  runtime_kwargs) -> messages | None` for pre-template chat editing. A non-`None` return from `adapt_messages` skips
  that control's token-level `adapt` for the call, so implementing both does not double-apply.
- **structural**: `steer(model, tokenizer, **kwargs) -> PreTrainedModel`; return the new or modified model.
- **state**: residual-stream methods subclass `InterventionControl` and declare an unbound intervention template
  in `_configure()` (a tuple of `Intervention` objects from `state_control/common/specs.py`: layers or a selector,
  a transform possibly carrying an `ArtifactSource`, a `TokenScope`, an optional gate); the base `steer()`
  binds it, `build_hooks` compiles it to torch hooks per generation, and `lower_interventions` compiles it to an
  `InterventionSpec` per steer, so the control contains no hook code, no per-generation state, and no backend
  knowledge. Methods hooking other mechanisms subclass `HookControl` and implement
  `get_hooks(input_ids, runtime_kwargs, **kwargs) -> {"pre": [...], "forward": [...], "backward": [...]}` where each
  spec is `{"module": <dotted submodule path>, "hook_func": <callable>}`, fully re-deriving per-generation state on
  every call. The session that executes forwards owns registration.
- **output**, step-level: `get_logits_processors(...)` and `get_stopping_criteria(...)`, returning fresh instances on
  each call. Loop-owning methods subclass `DecodingDriver` and implement `decode(input_ids, attention_mask, model,
  logits_processors, stopping_criteria, runtime_kwargs, **gen_kwargs)`, returning full prompt-plus-continuation ids
  and applying the received stacks at every scoring step.
- **all categories**: optional `steer()` for one-time preparation and `cleanup()` for releasing resources. The
  pipeline passes `session=` (a `SteeringSession` on the steering backend) into `steer()`; controls that only need
  structural facts read `session.layout` rather than the live model, and fitting call sites accept `session=` for
  capture-backed extraction.

Declare the class attributes the pipeline reads: `supports_batching` (default `False`; set `True` only
when the control is batch-safe), `enabled`, `RUNTIME_KWARGS_SCHEMA` (a list of `{"name": ...}` entries), and for
output controls `include_in_scoring` and `same_model_forwards`.

Backend support is declared through `requirements()`. The default (`IN_PROCESS_TORCH` at generate) is honest for a
new control and keeps it Hugging Face-only; do not widen it speculatively. An `InterventionControl` derives its
requirements from the template: generate offers the intervention-spec alternative exactly when every component has
a wire form (`Intervention.wire_kinds()` reads component and source declarations before `steer()`), and score is
in-process. Components describe their own wire form (`wire_kind` class attribute, `export()` per configuration),
and the equivalence of hooks and specs is pinned by `tests/core/test_spec_hook_equivalence.py`. An output control
whose behavior is sampling-expressible lowers via `export_generation_params()`, a declarative constraint via
`export_constraint()`, and an engine-hosted per-step processor via `export_processor_spec()`.

A control's steer step declares one of four access levels via `steer_access()`: `facts` (layout and tokenizer),
`rollouts` (generate and score through the session), `capture` (hidden states), or `module` (the model as a live
`torch.nn.Module`). Declare the highest rung your steer touches; intervention templates derive it from their
sources, and structural controls are `module` by definition. The pipeline hands your `steer()` a session scoped to
that rung — and the model itself only at `module` — and it arranges residency: on an engine backend, module-level
steps run on a temporary in-process model that is freed before the engine starts, with exported artifacts as the
handoff. Do not hold the model past `steer()` unless your generate phase requires `IN_PROCESS_TORCH`. Generate- and
score-phase requirements are unchanged.

`__init__.py` exports the discovery dict:

```python
from .args import MyMethodArgs
from .control import MyMethod

STEERING_METHOD = {
    "category": "state_control",
    "name": "my_method",
    "control": MyMethod,
    "args": MyMethodArgs,
}
```

The registry crawls the category directories at import time, requires the `name`, `control`, and `args` keys, and
rejects duplicate names. For a method with a heavy optional dependency, import it through
`aisteer360.utils.optional.require("<module>")` at the module boundary and add the extra to `pyproject.toml` (and to
`OPTIONAL_MODULE_EXTRAS` in `aisteer360/utils/optional.py`); discovery then skips the method with an actionable hint
when the dependency is absent instead of failing.

### Generics before new machinery

Before writing new components, check the category's `common/` library and compose from it:

- **state**: transforms (`AdditiveTransform`, `ProjectionTransform`, `RotationTransform`,
  `HeadAdditiveTransform`, `NormPreservingTransform`, `AlignmentAdaptiveTransform`), estimators
  (`MeanDifferenceEstimator`, `ContrastiveDirectionEstimator`, `SinglePairEstimator`, `SteeringPlaneEstimator`),
  gating (`Gate` over an `Evidence` and a rule; readouts `AffineReadout`, `CosineReadout`,
  `ProjectedCosineReadout`, `CallableReadout`; rules `SumThreshold`, `PerKeyThreshold`; `gate_from_probe`),
  selectors (`FixedLayerSelector`, `FractionalDepthSelector`, `TopKHeadSelector`, `ConditionPointSelector`),
  token scopes, `SteeringVector`, and `TransformHookRuntime`.
- **output**: `SearchDriver` (propose, score, keep, iterate) and `PhasedDriver` (`Fixed` / `Generated` phase plans),
  processors (`PrefixKeyedProcessor` base, constraint, contrastive mixture, value-guided), scorers (reward model,
  metric, majority vote), value functions, criteria (`StopOnSubstring`, `BudgetTokens`), and KV-cache utilities.
- **input**: memories (text, pool), formatters (system prompt, few-shot block, prepend, chat-template slot),
  proposers (LLM meta-prompt, retrieval), scorers, selectors (random, top-k, MMR, dense retrieval), and
  budget/Pareto utilities.
- **detection**: probes live in `core/internals/probes` (`fit_probe`, `calibrate_bias`, `ProbeSet`); prefer these
  over ad hoc classifiers, and consume their decisions through `Probe.as_gate()` for gated interventions or
  `routed_decoding`'s `Router` (ordered `Route`s with `P(name)` predicates) for routing.

Published methods are frequently presets over generics (`deal` presets `SearchDriver`; `budget_forcing`
presets `PhasedDriver`; `caa` composes an estimator with `AdditiveTransform`). Driver presets map their `Args` onto
the generic base's fields in `_configure()` rather than overriding `__init__`; follow that pattern for new decoding
methods. Before writing a new state control, check whether an `ActivationAdapter` configuration (transform, layer
selector, gate, token scope) already covers the behavior.

### Adding metrics and use cases

A metric subclasses `Metric` (or `LLMJudgeMetric` from `evaluation/metrics/base_judge.py`) and implements
`compute(responses, prompts=None, **kwargs) -> dict`. Task-agnostic metrics go in `evaluation/metrics/generic/`;
task-specific ones in `evaluation/metrics/custom/<use_case>/`.

A use case is a folder `evaluation/use_cases/<name>/` containing `use_case.py` with a `UseCase` subclass implementing
`generate()` and `evaluate()`. A use case declares each extra constructor parameter as a class-level annotation (a bare
annotation is required; a class-attribute default makes it optional) rather than writing an `__init__`; unknown
keywords and missing required parameters raise `TypeError` at construction, and each retained instance is checked by
`validate_evaluation_data` (which raises `ValueError` prefixed with `evaluation_data[<index>]`). Follow the existing use
cases, where `generate()` builds prompt rows and calls `batch_retry_generate` from
`evaluation/utils/generation_utils.py` (batched decoding with parsing and retry), and `evaluate()` maps metric names to
computed results. Build each prompt row by spreading its source instance (`{**instance, "prompt": ...}`) so the row
carries its own columns and `runtime_overrides` map per row; constructed keys (`"prompt"`, `"reference_answer"`,
`"thinking"`, ...) shadow same-named instance columns, so name override columns distinctly from them.

### Testing

Fixtures in `tests/conftest.py` provide a parametrized `device` fixture (`cpu` / `cuda` / `mps`, skipping unavailable
devices), a session-scoped `model_and_tokenizer` fixture over the tiny models in `tests/utils/ci_models.yaml`, mock
controls for every category, and evaluation fixtures. A new control needs `tests/controls/test_<method>.py` following
the existing pattern: a parameter grid expanded with `build_param_grid()`, then build the control, wrap it in a
`SteeringPipeline`, `steer()`, `generate()`, and assert on the output. Unit-test any new generics directly
(`tests/controls/` for control components, `tests/internals/` for the probes substrate, `tests/core/` for pipeline
behavior).

### Code style

- Python 3.11+ with modern typing (`list`, `dict`, `T | None`); line length 120; snake_case variables, PascalCase
  classes, UPPER_SNAKE_CASE constants; descriptive, not overly abbreviated, names.
- Comments describe current functionality only, in lowercase, with two spaces before inline comments
  (`a = 1  # some comment`) and no decorative formatting. Do not narrate edits or prior designs.
- Use a module logger (`logger = logging.getLogger(__name__)`) instead of `print` in library code.
- Keep imports simple; use the optional-dependency guard rather than broad try/except import fallbacks. Import order
  is enforced by isort (black profile) via pre-commit.

### Docstrings and documentation

Docstrings use the Google format (`Args:`, `Returns:`, `Raises:`, `Attributes:`); mkdocstrings parses them for the
reference site, and lists render correctly only with a blank line before them. Wrap code identifiers in backticks.
State what the code currently does, with the factual guarantees plainly stated (shapes, dtypes, defaults, side
effects such as in-place mutation, raise conditions, lifecycle constraints); keep the register neutral (no
intensifiers, evaluative adjectives, or rhetorical constructions) and do not use em-dashes. Describe behavior in
place rather than by analogy to another method. Cautionary content goes in the description body as plain prose before
`Args:`; `Warns:` is reserved for warnings the function emits at runtime. Control docstrings end with the paper
reference:

```
Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
      Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
      [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
```

A new method also needs its documentation surfaces updated: a reference page
`docs/reference/algorithms/<category>_control/<method>.md` (copy the mkdocstrings block from an existing page), a nav
entry in `docs/.nav.yml`, and a mention in the category's list in `docs/concepts/controls.md`.

### Notebooks

Each method gets a demonstration notebook: `examples/notebooks/algorithms/` for named methods, `generics/` for
config-first controls, `benchmarks/` for use-case studies (run artifacts stay in that subfolder), `recipes/` for
composite workflows. Add an entry to `examples/index.md`. Conventions: imports in a setup cell; explanation lives in
markdown cells rather than code comments; one plot per cell, drawn with `evaluation/utils/viz_utils.py` (call
`apply_plot_style()` first if custom matplotlib is unavoidable); no special characters in axis text; f-strings when
titles reference variable values. Markdown prose is plain technical reporting: narrate with "we", keep the register
neutral, signpost with connectives ("Note that ...", "For instance, ..."), write paragraphs rather than bolded bullet
lists, and link the paper in the introduction when demonstrating a published method.

### Definition of done

Before significant changes, check: does this fit the existing abstractions and patterns; does documentation need
updating; does it affect other parts of the system. A finished method contribution includes:

- [ ] the `args.py` / `control.py` / `__init__.py` triplet, importing cleanly with a valid `STEERING_METHOD` export
- [ ] honest `supports_batching`, `RUNTIME_KWARGS_SCHEMA`, and (output) `include_in_scoring` / `same_model_forwards`
- [ ] tests in `tests/controls/test_<method>.py` passing on CPU with the CI models
- [ ] a Google-style docstring ending with the paper reference
- [ ] reference page, `docs/.nav.yml` entry, and `docs/concepts/controls.md` mention
- [ ] a notebook plus its `examples/index.md` entry
- [ ] a `pyproject.toml` extra and `optional.py` entry for any new heavy dependency
- [ ] `pre-commit run --all-files` clean; commits signed off

## Invariants

Rules that hold regardless of task:

1. `steer()` must run before `generate()` or `compute_logprobs()`; it runs once per pipeline and heavy work belongs
   there, not in control constructors. On engine backends the steer phase runs in residency phases: stage-venued
   steps (module access, and capture where the engine serves none) run first on a temporary in-process model that
   is freed before the engine boots, then session-venued steps run through the engine session. The pipeline
   model's in-process weights and its engine-served weights never coexist.
2. Steering order is fixed (structural, input, state, output); list order within a category is the composition
   order, preserved within each residency phase (phases run module-first, and the only channel between steers is
   the pipeline model, which only stage-phase controls can touch). For state controls, entry order equals spec op
   order equals worker application order, so an in-process composition and its wire form apply edits in the same
   sequence.
3. The steer phase produces no support verdicts; each control declares its steer step's model access via
   `steer_access()`, and scoped sessions enforce the declaration on every backend. A control may retain the
   pipeline model beyond `steer()` only if its generate phase requires `IN_PROCESS_TORCH`; on engine backends the
   free protocol verifies the staged weights are gone and raises naming any retaining control.
4. The decode loop does not compose; at most one enabled `DecodingDriver` exists per pipeline, and a driver must
   apply the received `logits_processors` and `stopping_criteria` at every scoring step of every forward pass it
   issues.
5. Logits processors behave as functions of `(prefix_ids, scores)`; internal state is permitted only as memoization
   keyed on the prefix (subclass `PrefixKeyedProcessor`), and `get_logits_processors` returns fresh instances per
   call.
6. Extra forward passes through the pipeline's own model during decoding are wrapped in `auxiliary_pass()` (from
   `core/utils/auxiliary_pass.py`), and the component declares `same_model_forwards = True`.
7. Hooks exist only inside a session's execution of work (per item, or for the span of a driver decode the
   session hosts); controls never register hooks and hold no model reference. Hooks travel exclusively as
   `HookEntry` contributions built by the pipeline.
8. Never mutate caller-supplied artifacts (steering vectors, probes, configs); clone before moving devices or
   normalizing.
9. One in-flight generation per control instance: gate instances embedded in a control's interventions carry
   per-generation decisions, so do not share control instances across concurrently running pipelines.
10. `generate()` returns continuation-only ids by default; never re-slice its result by prompt length.
11. `runtime_kwargs` is a single shared namespace per call; declare consumed names in `RUNTIME_KWARGS_SCHEMA` and
    expect shared values on name collisions.
12. Declare `supports_batching=True` only when a control is safe under batched prompts; the pipeline and the
    evaluation utilities read it to choose between batched and per-example generation.
13. A control's behavior has exactly one declarative statement (the adapted prompt, a structural artifact, an
    intervention tuple, or exported params/specs); every backend consumes the highest representation it supports;
    hooks are per-generation products of the pipeline and specs are per-steer products of it; and no code path
    reconstructs a control's configuration by inspecting another representation of it.
14. Prompt-relative scope kinds (`after_prompt`, `last_k`) are client-side sugar; their wire form inside a driver
    generation is absolute (`from_position` at the generation's original prompt boundary).

## Pointers

- `docs/concepts/`: conceptual guides on controls, steering pipelines, and probes.
- `docs/tutorials/`: step-by-step guides for adding a steering method, metric, use case, and benchmark.
- `examples/notebooks/`: runnable references for every method, the generic controls, and full benchmarks.
- `tests/index.md`: test-suite layout and the pattern for adding control tests.
- Hosted documentation: <https://ibm.github.io/AISteer360/>.
