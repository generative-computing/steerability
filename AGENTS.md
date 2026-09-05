# AGENTS.md

Guidance for AI agents working in this repository. The Usage guide covers using the toolkit as a library; the
Developer guide covers extending it; the Invariants section lists rules that apply to every task. When this file
and the code disagree, the code is authoritative: verify against the source before acting on a claim made here.

## Overview

Steerability is a toolkit for steering large language models (Hugging Face causal LMs). It provides steering methods
("controls") across four model control surfaces, a `SteeringPipeline` that composes controls from any of the categories
into one operation on a model, and an Inspect AI evaluation stack (a registered model provider, task suites, and a sweep
runner) for comparing steering pipelines.

Pipelines execute on one configurable backend: the in-process Hugging Face backend (default), the offline vLLM
engine (`kind="vllm"`), or a vLLM server (`kind="vllm-serve"`). For the generate and score phases, each control
configuration is either supported or unsupported on a given backend: `pipeline.check()` reports unsupported
combinations with a verdict naming the gap and the fix, and unsupported operations raise before any work happens.
The steer phase produces no verdicts. Instead, each control declares the model access its steer step needs on the
`ModelAccess` ladder (`facts` < `rollouts` < `capture` < `module`), and `check()` also returns a deterministic steer
plan stating where each step and fit will run (see Execution backends below).

The four control categories, defined by what a method touches:

- **input**: manipulates the prompt only; generations follow `y ~ p_theta(sigma(x))` for a prompt adapter `sigma`.
- **structural**: persistently modifies weights or architecture (fine-tuning, DPO, merging); `y ~ p_theta'(x)`.
- **state**: edits internal activations at runtime via forward hooks, without changing weights.
- **output**: shapes the decoding process (logits processing, stopping, re-ranking, custom decode loops).

Vocabulary used throughout the codebase:

- **control**: one steering method, subclassing the base class of its category.
- **generic**: a dedicated recipe control class (`activation_adapter`, `value_guidance`, `search_decoding`, ...) that
  exposes common component slots through flat, sweepable `Args`, so that a method from the literature is a
  configuration rather than a new class. Named methods are siblings of generics, not children.
- **common library**: the per-category building blocks in `common/` (transforms, gating, drivers, selectors,
  formatters, ...) from which generics and named methods alike are assembled.
- **probe**: a calibrated linear readout over hidden states used for detection (reads, never edits); gating and
  routing consume its decisions.

## Repository map

```
steerability/
├── algorithms/
│   ├── core/                    # SteeringPipeline, registry, ControlSpec (specs.py), BaseArgs, BaseControl,
│   │   │                        # Output; identity.py (config identity, trial seeds), sweeps.py
│   │   │                        # (configuration sweeps, PipelineFactory), scoring.py (SampleScorer)
│   │   ├── execution/           # backend seam: spec, contracts, payloads, backend/session/registry, params,
│   │   │                        # fanout; access (ModelAccess, SteerPlan), session_utils (scoped sessions), staging
│   │   ├── internals/           # activation capture, pooling, stats, model_layout (decoder-stack resolution,
│   │   │                        # text_config), fingerprint, data/encoding/render; probes/ (detection)
│   │   └── utils/               # control merging, generation helpers, auxiliary_pass, assembly (per-generation
│   │                            # hook/spec/processor entry assembly)
│   ├── input_control/           # each category: base.py + one folder per method (triplet layout below)
│   │   └── common/             # building blocks: memory, formatters, proposers, scorers, selectors, budget, pareto
│   ├── state_control/
│   │   └── common/             # building blocks: intervention IR (specs.py), transforms, estimators, sources,
│   │                            # gating, selectors, token scopes, steering vectors, hook runtime, lowering
│   ├── output_control/          # methods incl. routed_decoding/ (control, routing.py, actions.py)
│   │   └── common/             # building blocks: drivers, processors, scorers, values, criteria, kv_cache,
│   │                            # candidates
│   └── structural_control/      # load_checkpoint/ and load_lora/ (artifact loaders; frozen forms of trained
│       └── wrappers/            # structural controls); wrappers/: trl/ (sft, dpo, ppo, grpo, apo) and mergekit/
├── spipe/                       # .spipe serialization: SPipe, manifest format, value codec,
│                                # content-addressed artifact store, freeze orchestration
├── backends/                    # huggingface/ (HFBackend, ExclusiveSession); vllm/ (VLLMBackend, VLLMServeBackend)
├── evaluation/                  # Inspect AI stack (optional `eval` extra; __init__ stays empty)
│   ├── provider.py              # ProviderOptions, SteeringPipelineModelAPI, as_inspect_model
│   ├── batching.py              # lock-leader collator (batched dispatch over concurrent requests)
│   ├── solvers.py               # runtime_kwargs_solver (per-sample runtime kwargs)
│   ├── scorers.py               # sample_scorer_from_inspect (Inspect scorers as SampleScorer rewards)
│   ├── suite.py                 # InspectSuite (task sets over eval_set)
│   ├── runner.py                # SteeringEval (configs x trials x suites, results frame, runs_frame)
│   └── plotting.py              # summary-frame plots over runs_frame output (optional `eval` extra)
└── utils/                       # tokenization, rendering, thinking, optional-dependency guard, verbosity
                                 # (opt-in package logging)

docs/                            # MkDocs site: home/, concepts/, tutorials/ (incl. add_method_by_category/),
                                 # reference/, .nav.yml
examples/                        # notebooks/{algorithms (incl. generics/, wrappers/),studies,recipes}/, index.md
tests/                           # controls/, core/, internals/, evaluation/, utils/; conftest.py
```

## Setup and commands

Python 3.12+ with `uv` as the package manager:

```bash
uv sync --extra all
source .venv/bin/activate
```

Optional extras, in three tiers:

- backends: `vllm`, the vLLM backends plus the `vllm_hook_plugins` core; pulls in `trl[vllm]` so the resolved vLLM
  stays inside trl's supported vLLM range
- workflows: `eval`, the Inspect AI evaluation stack plus matplotlib and seaborn for `evaluation/plotting.py`
- method-specific: `merging` (MergeKit)
- `all`: `eval`

Contributor tooling lives in `[dependency-groups]`: `dev` (pytest, pre-commit, the plugin core, and the `notebooks`
group), `notebooks` (notebook, ipywidgets, textstat, nltk), and `docs` (site tooling). `uv sync` installs `dev` by
default; add `--group docs` or `--group notebooks` as needed.

`merging` cannot share an environment with `eval` (MergeKit pins an older pydantic than Inspect requires), so it stays
out of `all` and `vllm`; `pyproject.toml` declares these as `[tool.uv] conflicts`. The optional-module-to-extra mapping
lives in `OPTIONAL_MODULE_EXTRAS` (`steerability/utils/optional.py`).

Hugging Face access uses the standard mechanism: `hf auth login` once, or `HF_TOKEN=hf_***` in the environment. Some
models (e.g. `meta-llama/*`) are gated; the account behind the token needs access on the model's Hub page. Never commit
tokens; a detect-secrets pre-commit hook scans against `.secrets.baseline`.

Models run inside the current process on the default Hugging Face backend; the vLLM backends execute on a local
engine or a remote server instead. Real steering runs need GPU memory for the base checkpoint plus the method's
overhead. For smoke tests, use the tiny models listed in `tests/utils/ci_models.yaml`
(e.g. `hf-internal-testing/tiny-random-LlamaForCausalLM`).

Common commands:

```bash
pytest tests/controls/                    # all control tests
pytest tests/controls/test_pasta.py       # one control
pytest tests/core/ tests/internals/       # pipeline, registry, probes
pre-commit install                        # once per clone
pre-commit run --all-files                # detect-secrets, whitespace/EOF fixers, large files, isort (black profile)
uv sync --extra all --group docs && uv run mkdocs serve   # docs at localhost:8000
```

Tests parametrize over the models in `tests/utils/ci_models.yaml` and over devices (`cpu`, `cuda`, `mps`); unavailable
devices are skipped automatically, so the suite runs on CPU-only machines. Commit messages require a DCO
`Signed-off-by:` line (see `CONTRIBUTING.md`).

## Usage guide

### Minimal steering loop

Every use of the toolkit follows the same loop: instantiate controls, wrap them in a `SteeringPipeline`, call
`steer()` once, then call `generate()` for inference.

```python
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot

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
construction. For lightweight controls, `steer()` only attaches artifacts like the tokenizer; for controls that train
(structural controls, activation-steering fits), the training runs there.

### Choosing a control category

| Goal | Category | Base class |
| --- | --- | --- |
| Change the prompt (few-shot, rewriting, prompt search) | input | `InputControl` |
| Change the weights (fine-tune, DPO, merge, load a checkpoint or adapter) | structural | `StructuralControl` |
| Edit activations or attention at runtime | state | `StateControl` |
| Shape decoding (rerank, guided sampling, custom loops) | output | `OutputControl` / `DecodingDriver` |

### Available controls

Enumerate the live registry rather than trusting any static list:

```python
from steerability.algorithms.core.registry import REGISTRY  # import triggers discovery

for category, methods in REGISTRY.items():
    print(category, sorted(methods))
```

The registered names at the time of writing:

- input: `cpo`, `few_shot`, `gepa`, `prewrite`, `system_prompt`, `user_prefix`
- state: `act_add`, `activation_adapter`, `angular_steering`, `caa`, `cast`, `directional_ablation`, `iti`, `pasta`
- output: `best_of_n`, `budget_forcing`, `constrained_decoding`, `contrastive_decoding`, `contrastive_guidance`,
  `deal`, `dexperts`, `phased_decoding`, `rad`, `routed_decoding`, `sasa`, `search_decoding`, `stopping_rules`,
  `value_guidance`
- structural: `load_checkpoint`, `load_lora` (artifact loaders, also the frozen forms of trained structural controls
  in a `.spipe`), `mergekit`, `sft`, `dpo`, `ppo`, `grpo`, `apo` (MergeKit and TRL wrappers)

### Pipeline semantics

`generate()` dispatches on the declared keyword source (exactly one per call) and returns the matching shape:

| Source | Tokenization | Return |
| --- | --- | --- |
| `text=` (`str`) | plain text | `str` |
| `text=` (`list[str]`) | batched text | `list[str]` |
| `messages=` (one chat) | chat template | `str` |
| `messages=` (batch of chats) | batched chat template | `list[str]` |
| `input_ids=` (tensor / token id lists) | passed through | `torch.Tensor` |

- Positional `str`/`list[str]` behaves like `text=`; any other positional shape raises a `TypeError`.
- The per-source methods `generate_text`, `generate_messages`, and `generate_tokens` sit alongside `generate()`
  with the same behavior, and take named parameters for the reserved keys.
- Decoded text returns carry exactly one candidate per prompt. Requesting `num_return_sequences`/`n` greater
  than 1 with `text=`/`messages=` raises `ValueError` unless `return_output=True` (which yields one `output_ids`
  row and one finish reason per candidate). The token return is `[batch * n, gen_len]` with each prompt's
  candidates contiguous, as in `model.generate`.

Behaviors that differ from bare Hugging Face usage:

- Returned token ids exclude the prompt by default. Do not slice the result by prompt length; pass
  `return_full_sequence=True` for HF-style prompt-plus-continuation output; a padded batch returns those rows
  left-packed (`[pads, prompt, continuation]`).
- Batched prompts are left-packed after the input chain, before state hooks are built and items are dispatched, so
  the prompt mask a conditional state control pools over aligns with the layout the sessions execute, and
  full-sequence returns and decoding-driver inputs carry that same layout. Single prompts and equal-length batches
  carry no padding and are unaffected.
- `chat_template_kwargs` is a reserved key inside `gen_kwargs`, forwarded to `apply_chat_template` after the
  pipeline-owned template kwargs. It is valid only with `messages=` (pairing it with `text=`/`input_ids=` raises
  `TypeError`) and may not name a pipeline-owned template kwarg (`return_tensors`, `padding`,
  `add_generation_prompt`, `return_dict`). The toolkit does not interpret its contents; keys are model-family
  specific (e.g. `enable_thinking`). Because it rides inside `gen_kwargs`, thinking-on and thinking-off runs get
  distinct configuration identities in sweeps.
- Token ids are returned as generated on every backend (stop text and any token-boundary overrun stay in the ids);
  decoded continuation text is truncated at the first stop-string occurrence by one client-side rule.
- `generate(..., return_output=True)` returns an `Output` object (or list of them) with fields `output_ids`,
  `adapted_input_ids` (the prompt after input controls, useful for inspecting the steered prompt), a per-item
  `finish_reason` (`"stop"`, `"eos"`, `"length"`, or `None`, with that precedence), and `finish_reasons` (one reason
  per candidate for `n > 1`). Import it via `from steerability.algorithms.core import Output`.
- A seeded `generate()` call maps its `seed` onto the items of a multi-item dispatch according to `seed_scope`
  (default `"item"`). Under `"item"`, one seed is derived per row, and on the Hugging Face backend the dispatch then
  decodes one row at a time. Under `"dispatch"`, one seed is derived for the whole dispatch, which is batched in one
  pass (reproducible as a whole). The scope is inert on vLLM backends, and an item carrying its own seed is honored
  under either scope.
- `generate()` before `steer()` raises `RuntimeError`; a second `steer()` call is a silent no-op.
- `attention_mask` is valid only with `input_ids=`; it is derived automatically for `text=` and `messages=`, and
  passing it with either (or with positional text) raises a `TypeError`.
- `device` and a non-default `device_map` are mutually exclusive on the `SteeringPipeline` constructor.
- Construction never loads the model. `steer()` acquires it from `model_name_or_path`, reuses preloaded
  `model=`/`tokenizer=` objects passed at construction, or receives it from a structural control that produces the
  final weights itself (e.g. `mergekit`). `lazy_init` is accepted and inert.
- `pipeline.supports_batching` is `True` only when every enabled control declares batch safety; the Inspect model
  provider batches concurrent requests when it is `True` and serializes them otherwise.
- `pipeline.compute_logprobs(input_ids, ref_output_ids=...)` scores reference tokens teacher-forced with the full
  steering applied; output controls with `include_in_scoring=False` are excluded from scoring.
- Controls with a `tokenizer` attribute left as `None` get the pipeline tokenizer injected automatically.

### Execution backends

`SteeringPipeline` takes `backend=`, a `BackendSpec` or a kind string. The default is the in-process Hugging Face
backend, so a pipeline that never names a backend runs entirely in process. `fit=` (`"auto"` or `"in_process"`)
selects the fit venue policy.

```python
from steerability.algorithms.core.execution import BackendSpec

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
- The steer phase satisfies each control's declared `steer_access()` by venue. `facts` and `rollouts` steps run
  through the backend's session on every kind. `capture` steps run through session capture where the spec
  advertises it (the offline plugin engine), and on a staged in-process model where it does not (serve, or
  `fit="in_process"`). `module` steps always stage. On engine backends the staged model is loaded, used, and freed
  before the engine boots; exported artifacts are the handoff, so in-process weights and engine-served weights
  never coexist. If engine capture fails a steer-time smoke test, fitting degrades to the stage with a warning;
  support verdicts never depend on the plugin's presence.
- Activation-steering state controls execute on vLLM through the vLLM-Hook plugin (`hook_plugin: True` on the
  spec). The control's steering tuple serializes as an intervention spec, and tensor payloads travel as
  content-addressed artifacts (`artifact_dir` option; on serve this must be a filesystem shared with the server).
  A configuration either serializes exactly or is honestly in-process-only; there is no approximate lowering.
- The offline backend applies a scoped engine-process environment per boot and restores it
  (`backends/vllm/environment.py`), defaulting `VLLM_USE_FLASHINFER_SAMPLER=0` (an explicit caller value wins) and
  forcing `VLLM_HOOK_WORKER=unified` for `hook_plugin` boots. `serve_environment` returns the same policy as a fresh
  mapping for a `vllm serve` process. The model-runner constraint is owned by the vLLM-Hook plugin (which pins the
  legacy runner or supports V2), not the toolkit.
- Structural controls train on the staged model and serve their artifacts (checkpoint or LoRA) on vLLM backends.
- Declarative constrained decoding lowers to vLLM's native structured outputs. Hidden-state capture (probe fitting
  and reads, routed decoding) is served in process and on the offline plugin engine, not on serve.
- `compute_logprobs` scores through the backend; an enabled output control with
  `include_in_scoring=True` keeps scoring in-process.
- Discarding a pipeline that booted a vLLM engine should go through `release_backends()` (or a
  `with` block over the pipeline) rather than relying on garbage collection, which is not prompt at
  freeing the engine. A failed `steer()` releases the backends it constructed before re-raising, so a
  retried steer re-boots. `PipelineFactory` (and so `SteeringEval`) releases per configuration.
- Spec options the vLLM backends read: `hook_plugin`, `artifact_dir`, `engine_kwargs` (offline engine);
  `base_url`, `api_key`, `max_concurrency`, `request_timeout`, `max_retries`, `retry_backoff` (server);
  and `tokenizer_name_or_path` / `trust_remote_code` for the client-side tokenizer. Options must be plain
  data; `BackendSpec` canonicalizes them and its hash is the backend identity. A spec combining
  `hook_plugin` with speculative decoding, or an offline `hook_plugin` engine with
  `enforce_eager=False`, is rejected at construction.

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

State controls and hidden-state capture resolve the decoder stack at one of three roots: `model.layers` (text-only
decoder models: Llama, Mistral, Qwen, Gemma text), `model.language_model.layers` (composite multimodal wrappers such
as Gemma 3/4 and Qwen3.5 loaded under `AutoModelForCausalLM`), and `transformer.h` (GPT-2). Resolution selects the
per-layer naming convention (`llama_style`, `gemma_style`, `gpt2_style`) whose norm markers exist on the first decoder
layer and whose attention module exists on at least one layer.

A hybrid stack that interleaves attention layers with another token mixer (Qwen3.5 and Qwen3-Next, where three Gated
DeltaNet `linear_attn` layers precede each `self_attn` layer) resolves to its attention layers' family, with
`ModelLayout.attention_layer_ids` recording which layers carry attention. Residual-stream controls (`caa`, `act_add`,
`angular_steering`, `activation_adapter`) and hidden-state capture work unchanged on such a stack; `head_geometry`,
o_proj-site interventions, and `pasta` refuse the other layers with a message naming the attention layers, and `iti`
refuses hybrid stacks.

A multimodal checkpoint is steered on its text decoder under text-only prompting; images and audio stay out. An
unmerged LoRA adapter (`LoadLoRA(merge=False)`, or a TRL LoRA run without `merge_lora_after_train`) is hooked through
the PEFT wrapper, so a state control listed after it steers the adapted model. Register a detector with
`register_layout_detector` (from `steerability.algorithms.core.internals`) for an architecture not on this list.

### Runtime kwargs

Some controls need per-call information at inference time. All controls read from the single `runtime_kwargs` dict
passed to `generate()`. Each control declares the names it consumes in its `RUNTIME_KWARGS_SCHEMA`, together with a
`scope` per entry: `"row"` for a per-prompt value (delivered as a row-aligned sequence in batched calls) or `"call"`
for one value per call; a missing `scope` means `"call"`.

The pipeline validates the declarations at `steer()`. It raises on disagreeing declarations of one name and warns when
two controls declare the same name with agreeing declarations (they will share one value).

Two examples of row-scoped kwargs. For PASTA's `substrings`, a `str` broadcasts to every row, a `list[list[str]]` of
batch length carries one group per row, and a flat `list[str]` is accepted only at batch size 1 (to broadcast one group
over a batch, pass `[[...]] * batch_size`). The `SearchDriver` presets (`DeAL`, `BestOfN`, `SearchDecoding`) declare
`reward_params` row-scoped: one mapping per row, merged into the scorer's params.

```python
pipeline.generate(
    prompt,
    runtime_kwargs={"substrings": ["The answer must be in JSON."]},  # e.g. pasta's emphasis spans
    max_new_tokens=128,
)
```

### Saving and loading pipelines (`.spipe`)

`pipeline.to_spipe()` serializes a pipeline as an `SPipe`: the model reference plus the controls as constructed
(the recipe), and, once the pipeline is steered, the frozen resolution (fitted vectors, probes, adapters, optimized
prompts) in a content-addressed artifact store with a lock section (fingerprints, per-fit digests).

`spipe.save(path)` writes a zip when `path` ends in `.spipe` and a directory otherwise; `artifacts="thin"` writes the
manifest only, with artifact ids resolved at load through `artifact_store=`. `SPipe.load(path)` reads either form.

`spipe.pipeline()` reconstructs a `SteeringPipeline`. Frozen entries instantiate from their resolution, so `steer()` is
cheap and model-free; `prefer="recipe"` forces re-fits instead. Backend, device, dtype, and `hf_model_kwargs` stay the
caller's. `verify()` is the model-free report, `thaw()` drops the resolution, and `allow_code=True` at load gates
callable references, non-toolkit dataclass imports, and pickle-backed memories.

Loading a stale bundle (fit-relevant recipe fields edited after freezing) raises unless `allow_stale=True`. Trained
structural controls freeze as `load_checkpoint` / `load_lora` entries; intervention controls freeze as
`activation_adapter` entries unless they declare a same-class frozen form (as `caa`, `act_add`, and `iti` do).

```python
pipeline.steer()
pipeline.to_spipe().save("formal_tone.spipe")

from steerability.spipe import SPipe
loaded = SPipe.load("formal_tone.spipe").pipeline()
loaded.steer()
```

### Evaluation

Evaluation runs steered pipelines on [Inspect AI](https://inspect.aisi.org.uk/) tasks (optional `eval` extra).
`as_inspect_model(pipeline)` wraps a steered pipeline as a generation-only Inspect model; an `InspectSuite` names a
set of tasks; `SteeringEval` runs configurations (fixed controls, `ControlSpec` sweeps, and the empty-list baseline
arm) x trials x suites, sequentially, one GPU-resident pipeline at a time:

```python
from steerability.evaluation.provider import ProviderOptions
from steerability.evaluation.runner import SteeringEval
from steerability.evaluation.suite import InspectSuite

runner = SteeringEval(
    pipelines={"baseline": [], "few_shot": [few_shot], "caa_sweep": [ControlSpec(control_cls=CAA, ...)]},
    base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    suites=[InspectSuite(name="capability", tasks=("inspect_evals/gsm8k",), limit=200)],
    num_trials=3,
    seed=7,  # derives one seed per (config, trial), attached to sampling dispatches
    generate_defaults={"temperature": 0},  # greedy is the recommended posture
    provider_options=ProviderOptions(max_batch_size=8),
    save_dir="runs/exp1",
    display="plain",  # stream Inspect's per-sample progress inside each cell (progress=True draws a cell bar)
)
results = runner.run()       # {config_name: [{trial_id, seed, config_id, params, suites, provenance}, ...]}
frame = runner.results()     # one row per (config, trial, suite, task, scorer/metric)
runs = runner.runs_frame(metrics={"accuracy": "choice/accuracy"})  # one row per (pipeline, trial)
```

`summarize_runs` (in `evaluation/runner.py`) aggregates the per-trial frame into one row per configuration with
`{metric}_mean` / `{metric}_std` columns, the contract every function in `evaluation/plotting.py` consumes (`eval`
extra).

Every generation flows through `pipeline.generate()`. Prompts enter as `messages=` when the tokenizer has a chat
template (so `adapt_messages` input controls fire exactly as in deployment) and as rendered `text=` otherwise, with
the path recorded as `prompt_path` in provenance. Scoring is generation-based only; logprob parameters, tools, and
multimodal content are refused with actionable messages.

Concurrent Inspect requests collate into batched pipeline calls when every enabled control is batch-safe. A seeded
dispatch carries `seed_scope` from `ProviderOptions` (default `"dispatch"`), so a seeded batch decodes in one pass on
the Hugging Face backend, and bitwise reproducibility of stochastic sampling is not preserved under concurrency (see
the `steerability/evaluation/batching.py` module docstring for the full contract).

There is no results checkpoint: the `.eval` logs under `save_dir/inspect_logs/` are the store, and `eval_set`
resumes each (config, trial, suite) cell from them at sample granularity. Because `eval_set` matches task identity
only, a changed protocol (seed, generate defaults, provider options, suites, fit, backend) needs a new `save_dir`.
Pre-flight `check()` runs over every sweep point before any model or engine work (`on_unsupported="raise"` or
`"skip"`).

Per-sample steering inputs travel on `Sample.metadata` and are delivered by the shipped `runtime_kwargs_solver` (used
in place of a bare `generate()` in the task's solver chain). Static per-arm kwargs go in
`ProviderOptions.runtime_kwargs`: a static value of a `"row"`-scoped kwarg is one row's value in the control's per-row
form, broadcast to every row, and a static name no configuration declares warns at pre-flight. Controls that consume
a per-row reward take a `SampleScorer` (`(response, row) -> float`, from `algorithms/core/scoring.py`);
`sample_scorer_from_inspect` adapts any Inspect scorer into that shape. See
`docs/tutorials/evaluate_steering_pipelines.md` for the full guide, including task authoring and grader-model
guidance.

The core sweep layer (`algorithms/core/sweeps.py`: `expand_configurations`, `preflight`, `PipelineFactory`;
`algorithms/core/identity.py`: canonical config identity and trial seeds) has no Inspect dependency. A planned
`steerability/optimization/` package (not yet in the tree) is to compose the same pieces with a suite as its objective.

## Developer guide

### Adding a steering control

A method is a sub-package of its category directory with a three-file layout:

```
steerability/algorithms/<category>_control/<method_name>/
├── __init__.py    # exports STEERING_METHOD for registry discovery
├── args.py        # hyperparameter dataclass (single source of truth)
├── control.py     # the control class; all steering behavior lives here
└── utils/         # optional local helpers; use sparingly
```

`args.py` defines every hyperparameter as a `BaseArgs` dataclass; use `__post_init__` for cross-field validation:

```python
from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


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
- **structural**: `steer(model, tokenizer, **kwargs) -> PreTrainedModel`; return the new or modified model. The
  TRL wrappers forward `training_args` verbatim to the installed TRL config, so a convenience field loses to a
  `training_args` entry of the same name, and a key the config does not declare raises at construction.
- **state**: residual-stream methods subclass `InterventionControl` and declare an unbound intervention template
  in `_configure()`: a tuple of `Intervention` objects from `state_control/common/specs.py`, each naming layers or
  a selector, a transform possibly carrying an `ArtifactSource`, a `TokenScope`, and an optional gate. The base
  `steer()` binds the template, `build_hooks` compiles it to torch hooks per generation, and `lower_interventions`
  compiles it to an `InterventionSpec` per steer, so the control contains no hook code, no per-generation state,
  and no backend knowledge. Methods hooking other mechanisms subclass `HookControl` and implement
  `get_hooks(input_ids, runtime_kwargs, **kwargs) -> {"pre": [...], "forward": [...], "backward": [...]}`, where
  each spec is `{"module": <dotted submodule path>, "hook_func": <callable>}`, fully re-deriving per-generation
  state on every call. The session that executes forwards owns registration.
- **output**, step-level: `get_logits_processors(...)` and `get_stopping_criteria(...)`, returning fresh instances on
  each call. Loop-owning methods subclass `DecodingDriver` and implement `decode(input_ids, attention_mask, model,
  logits_processors, stopping_criteria, runtime_kwargs, session=None, **gen_kwargs)`, returning full
  prompt-plus-continuation ids and applying the received stacks at every scoring step. The pipeline always passes
  `session=` (a `SteeredSession` carrying the generation's steering entries); drivers issue their rollouts through
  it, and `model` is None on backends without a live model.
- **all categories**: optional `steer()` for one-time preparation and `cleanup()` for releasing resources. The
  pipeline passes `session=` (a `SteeringSession` on the steering backend) into `steer()`; controls that only need
  structural facts read `session.layout` rather than the live model, and fitting call sites accept `session=` for
  capture-backed extraction.

Declare the class attributes the pipeline reads:

- `supports_batching` (default `False`; set `True` only when the control is batch-safe)
- `enabled`
- `RUNTIME_KWARGS_SCHEMA`: a list of `{"name": ...}` entries; declare `scope` on every entry, `"row"` for a
  per-prompt value delivered row-aligned in batched calls or `"call"` for one value per call
- for output controls, `include_in_scoring` and `same_model_forwards`

Backend support is declared through `requirements()`. The default (`IN_PROCESS_TORCH` at generate) is honest for a
new control and keeps it Hugging Face-only; do not widen it speculatively. An `InterventionControl` derives its
requirements from the template: generate offers the intervention-spec alternative exactly when every component has
a wire form (`Intervention.wire_kinds()` reads component and source declarations before `steer()`), and score is
in-process. Components describe their own wire form (`wire_kind` class attribute, `export()` per configuration), and
the equivalence of hooks and specs is pinned by `tests/core/test_spec_hook_equivalence.py`. An output control whose
behavior is sampling-expressible lowers via `export_generation_params()`, a declarative constraint via
`export_constraint()`, and an engine-hosted per-step processor via `export_processor_spec()`.

A control's steer step declares one of four access levels via `steer_access()`: `facts` (layout and tokenizer),
`rollouts` (generate and score through the session), `capture` (hidden states), or `module` (the model as a live
`torch.nn.Module`). Declare the highest rung your steer touches; intervention templates derive it from their
sources, and structural controls are `module` by definition. The pipeline hands your `steer()` a session scoped to
that rung (and the model itself only at `module`) and arranges residency: on an engine backend, module-level steps
run on a temporary in-process model that is freed before the engine starts, with exported artifacts as the handoff.
Do not hold the model past `steer()` unless your generate phase requires `IN_PROCESS_TORCH`. Generate- and
score-phase requirements are unchanged.

A control whose steer step produces fits or other state must also say how it freezes into a `.spipe`, through four
methods:

- `steer_fits()` lists the fit artifacts the step will produce as `(artifact, artifact_class)` pairs. The steer plan
  reads it, and `"calibrated"` artifacts get cross-venue notices.
- `export_state()` returns the steer-time products by logical name (a `SteeringVector`, `Probe`, `ProbeSet`,
  `Memory`, `CheckpointArtifact`, `LoRAArtifact`, tensor, or on-disk `Path`).
- `frozen_form(state)` returns the `(registry method key, constructor kwargs)` of a constructor-valid frozen form,
  which may be the control's own class (`caa`) or another registered method (`activation_adapter`, `load_lora`).
- `fit_identity()` returns the object whose canonical form digests the fit-relevant recipe inputs, so staleness
  detection excludes inert application parameters.

`InterventionControl` derives all four from its template, and the TRL and MergeKit wrappers freeze to `load_lora` /
`load_checkpoint`. A control that produces state and overrides none of them raises `NotFreezableError` at freeze.
Controls whose recipe is their frozen form need nothing.

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
rejects duplicate names. A dependency a control needs goes in core when it installs everywhere the toolkit does without
conflict (xgrammar, trl, peft). It gets its own extra when it conflicts with other dependencies, is platform-limited,
or is very large; import it through `steerability.utils.optional.require("<module>")` at the module boundary and map it
in `OPTIONAL_MODULE_EXTRAS` (`steerability/utils/optional.py`), and discovery then skips the method with an actionable
hint when the dependency is absent instead of failing. A dependency a control runs without (an alternative estimator or
an enhancement) is not declared at all; raise a `ModuleNotFoundError` naming the package and its tested range, as CPO's
`use_dml` does.

### Generics before new machinery

Before writing new components, check the category's `common/` library and compose from it:

- **state**: transforms (`AdditiveTransform`, `ProjectionTransform`, `RotationTransform`,
  `HeadAdditiveTransform`, `NormPreservingTransform`, `AlignmentAdaptiveTransform`); artifact sources
  (`ContrastiveFit`, `SinglePairFit`, `ConditionPointSearch`, `LayerFilteredFit`, `VerifiedPrecomputed`, and the
  PASTA-local `HeadProfile` for rollout-scored head selection; each declares its `access` and `artifact_class`);
  estimators (`MeanDifferenceEstimator`,
  `ContrastiveDirectionEstimator`, `SinglePairEstimator`, `SteeringPlaneEstimator`); gating (`Gate` over an
  `Evidence` and a rule; readouts `AffineReadout`, `CosineReadout`, `ProjectedCosineReadout`, `CallableReadout`;
  rules `SumThreshold`, `PerKeyThreshold`; `gate_from_probe`); selectors (`FixedLayerSelector`,
  `FractionalDepthSelector`, `TopKHeadSelector`, `ConditionPointSelector`); token scopes; `SteeringVector`; and
  `TransformHookRuntime`.
- **output**: `SearchDriver` (propose, score, keep, iterate) and `PhasedDriver` (`Fixed` / `Generated` phase plans);
  processors (`PrefixKeyedProcessor` base, constraint, contrastive mixture, value-guided); sequence scorers
  (`RewardModelScorer`, `MajorityVoteScorer`, `SampleSequenceScorer` over a per-row `SampleScorer`); value
  functions (callable, classifier, reward model, subspace margin); criteria (`StopOnSubstring`, `BudgetTokens`);
  and KV-cache utilities.
- **input**: memories (text, pool); formatters (system prompt, few-shot block, prepend, chat-template slot);
  proposers (LLM meta-prompt, retrieval); scorers; selectors (random, top-k, MMR, dense retrieval); and
  `RolloutBudget` / `ParetoFrontier` utilities.
- **detection**: probes live in `core/internals/probes` (`fit_probe`, `calibrate_bias`, `ProbeSet`, and
  `ProbeSetFit` for fitting deferred to steer time). Prefer these over ad hoc classifiers, and consume their
  decisions through `Probe.as_gate()` for gated interventions or `routed_decoding`'s `Router` (ordered `Route`s
  with `P(name)` predicates over the actions `respond`, `prefix`, `generate`) for routing.

Published methods are frequently presets over generics (`deal` presets `SearchDriver`; `budget_forcing`
presets `PhasedDriver`; `caa` composes an estimator with `AdditiveTransform`). Driver presets map their `Args` onto
the generic base's fields in `_configure()` rather than overriding `__init__`; follow that pattern for new decoding
methods. Before writing a new state control, check whether an `ActivationAdapter` configuration (transform, layer
selector, gate, token scope) already covers the behavior.

### Authoring evaluation tasks

Target-behavior evaluations are ordinary Inspect `Task`s; the toolkit ships no task, scorer, or metric classes of its
own (working examples are defined inside the study notebooks under `examples/notebooks/studies/`).

A task whose samples carry per-sample steering inputs puts them on `Sample.metadata` as `{"runtime_kwargs": {...}}`,
with each value in the consuming control's per-row form, and uses `runtime_kwargs_solver()` from
`steerability/evaluation/solvers.py` as its generation step. Each key must be declared `"row"`-scoped in the
consuming control's `RUNTIME_KWARGS_SCHEMA` (a `"call"`-scoped key is rejected per sample), and a key that no enabled
control of an arm declares is inert on that arm, so the empty baseline shares the task.

Tasks with model-graded scorers take the grader model through their own arguments (`task_args`); the grader is never
the pipeline under evaluation. Controls that consume a per-row reward accept a `SampleScorer`; use
`sample_scorer_from_inspect` to drive them with an Inspect scorer. See
`docs/tutorials/evaluate_steering_pipelines.md` for the authoring guide.

### Testing

Fixtures in `tests/conftest.py` provide a parametrized `device` fixture (`cpu` / `cuda` / `mps`, skipping unavailable
devices), a session-scoped `model_and_tokenizer` fixture over the tiny models in `tests/utils/ci_models.yaml`, mock
controls for every category, and mock model/tokenizer factories.

A new control needs `tests/controls/test_<method>.py` following the existing pattern: a parameter grid expanded with
`build_param_grid()` (from `tests/utils/sweep.py`), then build the control, wrap it in a `SteeringPipeline`, `steer()`,
`generate()`, and assert on the output. Unit-test any new generics directly (`tests/controls/` for control components,
`tests/internals/` for the probes substrate, `tests/core/` for pipeline behavior).

### Code style

- Python 3.12+ with modern typing (`list`, `dict`, `T | None`); line length 120; snake_case variables, PascalCase
  classes, UPPER_SNAKE_CASE constants; descriptive, not overly abbreviated, names.
- Comments describe current functionality only, in lowercase, with two spaces before inline comments
  (`a = 1  # some comment`) and no decorative formatting. Do not narrate edits or prior designs.
- Use a module logger (`logger = logging.getLogger(__name__)`) instead of `print` in library code.
- Keep imports simple; use the optional-dependency guard rather than broad try/except import fallbacks. Import order
  is enforced by isort (black profile) via pre-commit.
- Read structural facts (`hidden_size`, `num_attention_heads`, `head_dim`, `num_hidden_layers`) through
  `text_config(model)` (from `steerability.algorithms.core.internals`), which returns the text sub-config on composite
  multimodal models; never read `model.config.hidden_size` directly, and never default a missing fact to `0`. Resolve
  decoder module paths through `resolve_model_layout(model)` rather than by matching `model.model.layers`.

### Docstrings and documentation

Docstrings use the Google format (`Args:`, `Returns:`, `Raises:`, `Attributes:`); mkdocstrings parses them for the
reference site, and lists render correctly only with a blank line before them. Wrap code identifiers in backticks.

State what the code currently does, with the factual guarantees plainly stated (shapes, dtypes, defaults, side
effects such as in-place mutation, raise conditions, lifecycle constraints). Keep the register neutral (no
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

Each method gets a demonstration notebook: `examples/notebooks/algorithms/` for named methods,
`algorithms/generics/` for config-first controls, `algorithms/wrappers/` for the library wrappers (`trl`,
`mergekit`), `studies/` for use-case studies (run artifacts stay in that subfolder), `recipes/` for composite
workflows. Add an entry to `examples/index.md`.

Conventions: imports in a setup cell; explanation lives in markdown cells rather than code comments; one plot per
cell; no special characters in axis text; f-strings when titles reference variable values. Markdown prose is plain
technical reporting: narrate with "we", keep the register neutral, signpost with connectives ("Note that ...", "For
instance, ..."), write paragraphs rather than bolded bullet lists, and link the paper in the introduction when
demonstrating a published method.

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

1. `steer()` must run before `generate()` or `compute_logprobs()`; it runs once per pipeline, and heavy work belongs
   there, not in control constructors. On engine backends the steer phase runs in residency phases. Stage-venued
   steps (module access, and capture where the engine serves none) run first on a temporary in-process model that
   is freed before the engine boots; session-venued steps then run through the engine session. The pipeline
   model's in-process weights and its engine-served weights never coexist.
2. Steering order is fixed (structural, input, state, output). List order within a category is the composition
   order, preserved within each residency phase: phases run module-first, and the only channel between steers is
   the pipeline model, which only stage-phase controls can touch. For state controls, entry order equals spec op
   order equals worker application order, so an in-process composition and its wire form apply edits in the same
   sequence.
3. The steer phase produces no support verdicts. Each control declares its steer step's model access via
   `steer_access()`, and scoped sessions enforce the declaration on every backend. A control may retain the
   pipeline model beyond `steer()` only if its generate phase requires `IN_PROCESS_TORCH`; on engine backends the
   free protocol verifies the staged weights are gone and raises naming any retaining control.
4. The decode loop does not compose. At most one enabled `DecodingDriver` exists per pipeline, and a driver must
   apply the received `logits_processors` and `stopping_criteria` at every scoring step of every forward pass it
   issues.
5. Logits processors behave as functions of `(prefix_ids, scores)`. Internal state is permitted only as memoization
   keyed on the prefix (subclass `PrefixKeyedProcessor`), and `get_logits_processors` returns fresh instances per
   call.
6. Extra forward passes through the pipeline's own model during decoding are wrapped in `auxiliary_pass()` (from
   `core/utils/auxiliary_pass.py`), and the component declares `same_model_forwards = True`.
7. Hooks exist only inside a session's execution of work (per item, or for the span of a driver decode the
   session hosts). Controls never register hooks and hold no model reference. Hooks travel exclusively as
   `HookEntry` contributions built by the pipeline.
8. Never mutate caller-supplied artifacts (steering vectors, probes, configs); clone before moving devices or
   normalizing.
9. One in-flight generation per control instance. Gate instances embedded in a control's interventions carry
   per-generation decisions, so do not share control instances across concurrently running pipelines.
10. `generate()` returns continuation-only ids by default; never re-slice its result by prompt length.
11. `runtime_kwargs` is a single shared namespace per call. Declare consumed names and their `scope` in
    `RUNTIME_KWARGS_SCHEMA` (a row-scoped kwarg receives a row-aligned sequence in batched calls) and expect shared
    values on name collisions.
12. Declare `supports_batching=True` only when a control is safe under batched prompts; the pipeline and the
    Inspect model provider read it to choose between batched and per-example generation.
13. A control's behavior has exactly one declarative statement (the adapted prompt, a structural artifact, an
    intervention tuple, or exported params/specs). Every backend consumes the highest representation it supports.
    Hooks are per-generation products of the pipeline and specs are per-steer products of it, and no code path
    reconstructs a control's configuration by inspecting another representation of it.
14. Prompt-relative scope kinds (`after_prompt`, `last_k`) are client-side sugar; their wire form inside a driver
    generation is absolute (`from_position` at the generation's original prompt boundary).
15. A control's freezable state is exactly `export_state()`; the frozen form returned by `frozen_form()` is
    constructor-valid for its declared method; and the recipe is never discarded (a frozen `.spipe` entry keeps the
    original args for provenance and `thaw()`).

## Pointers

- `docs/concepts/`: conceptual guides on controls (with each control's `Backends` line), steering pipelines,
  probes, and the `.spipe` format.
- `docs/tutorials/`: step-by-step guides for adding a steering method (with per-category walkthroughs under
  `add_method_by_category/`) and evaluating steering pipelines.
- `examples/notebooks/`: runnable references for every method, the generic controls, and use-case studies.
- `tests/index.md`: test-suite layout and the pattern for adding control tests.
- Hosted documentation: <https://ibm.github.io/steerability/>.
