# Evaluate steering pipelines

The toolkit evaluates steering pipelines on [Inspect AI](https://inspect.aisi.org.uk/) (UK AI
Security Institute) and its benchmark catalog
[`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals). This facilitates the evaluation
of both the target behavior of a pipeline (did instruction following ability improve?) and its
off-target effects (degradation in math ability, coding ability, general knowledge, etc.).

Note that the evaluation is on the entire pipeline rather than the model alone, since a steering
pipeline generally includes modifications to the input/prompt and the decoding process in addition to
model-level modifications (weights, activations). Additionally, evaluation must be done on open-ended
generations rather than logprobs. One of the primary reasons for this is output controls, i.e., a
decoding driver induces a distribution over sequences without a per-token conditional.

## The model provider

The `as_inspect_model` function wraps a steered pipeline as an Inspect model:

```python
from inspect_ai import eval as inspect_eval
from steerability.evaluation.provider import ProviderOptions, as_inspect_model

pipeline.steer()
model = as_inspect_model(pipeline, options=ProviderOptions(max_batch_size=8))
logs = inspect_eval("inspect_evals/gsm8k", model=model, limit=100, temperature=0)
```

where the `ProviderOptions` dataclass contains the provider's configuration:

- `runtime_kwargs`: static runtime kwargs applied to every request
- `chat_template_kwargs`: template kwargs for the messages path
- `max_batch_size`: the batching ceiling
- `default_max_tokens`: the default `max_tokens`
- `reasoning_tags`: the tags used to split thinking from the answer before scoring (`reasoning_tags=None`
  disables the split)
- `on_unsupported_param`: the policy for `GenerateConfig` parameters the pipeline cannot honor (`"raise"` by
  default or `"warn"`)

The provider decides how to deliver prompts to the pipeline when it is constructed. With a chat-templated
tokenizer, prompts are dispatched as `messages=` and every input control participates as it does in
deployment. Base models without a chat template (a common subject of capability measurements) are
evaluated through a text path instead, i.e., each conversation is rendered to plain text and dispatched
as `text=`. On the text path `adapt_messages` never runs (token-level `adapt` still does). Since the same
controls behave differently on the two paths, the provider warns once at construction and records the
path as `prompt_path` in the run provenance.

### Scope

The provider is generation-only. Requests that include tools or tool messages, logprob parameters
(`logprobs`, `top_logprobs`, `prompt_logprobs`), or multimodal content raise an error that
explains the restriction. This limits evaluation to non-agentic tasks, which form the majority of
`inspect_evals`. Note that `GenerateConfig.response_schema` is not translated into a
`constrained_decoding` control because that would inject a control the configuration did not
declare. It follows the unsupported-parameter policy instead.

## Batching and reproducibility

Inspect issues one asynchronous request per sample and keeps many outstanding at once, while the
pipeline runs one (possibly batched) generation at a time. The provider bridges the two by collecting
concurrent requests into batched `pipeline.generate()` calls, filling the next batch while the current
generation runs. It advertises `max_connections` equal to its effective batch ceiling, and Inspect's
`max_samples` defaults to that value, which fills batches exactly. Note that `max_connections` should
not be set below `max_batch_size`.

Batching applies only to arms whose enabled controls all declare `supports_batching=True`, and the
provider otherwise clamps the batch size to 1. Input, state, and structural arms batch. Among output
controls, only `phased_decoding`, `routed_decoding`, and `stopping_rules` declare batch safety (`rad`
and `value_guidance` compute it), and most driver-based arms therefore run one sample at a time. Rows
of one batch need not share a prompt length. A ragged batch (e.g., few-shot with per-row exemplar
draws) is left-packed on the Hugging Face backend, and each row's continuation is predicted from its
own last real token rather than from a trailing pad.

We recommend greedy decoding (`temperature=0`) as the default since it is the norm for capability
benchmarks and avoids seed sensitivity. Which samples are evaluated is fixed by the suite, independent
of batch composition. Under sampling, `seed_scope` in `ProviderOptions` sets how seeds are applied.
The default `"dispatch"` scope seeds each batch as a whole and decodes it in one pass, and the
`"item"` scope derives a seed per row and decodes one row at a time. Bitwise reproducibility of
stochastic sampling is not preserved under concurrency, because a sample's batch membership and row
index depend on the order in which requests arrive. A bitwise-reproducible stochastic run requires
`max_batch_size=1` and Inspect `max_connections=1`. Even greedy outputs can differ across batch
compositions, since padded-batch numerics differ from single-item numerics on some kernels.
Trial-to-trial variation under sampling is therefore measured rather than eliminated, which is the role
of `num_trials` and the per-metric standard error.

## Suites and the runner

An `InspectSuite` specifies a set of tasks evaluated together. `SteeringEval` runs each configuration
(fixed controls, `ControlSpec` sweeps, and the empty baseline arm) over every trial and suite,
building and releasing one GPU-resident pipeline at a time:

```python
from steerability.evaluation.runner import SteeringEval
from steerability.evaluation.suite import InspectSuite

capability = InspectSuite(name="capability", tasks=("inspect_evals/gsm8k",), limit=200)
target = InspectSuite(name="target", tasks=("target_task.py",))

runner = SteeringEval(
    pipelines={"baseline": [], "pasta": [pasta]},
    base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    suites=[capability, target],
    num_trials=3,
    seed=7,
    generate_defaults={"temperature": 0},
    save_dir="runs/exp1",
    display="plain",
)
results = runner.run()
frame = runner.results()
```

File-referenced tasks resolve relative to the working directory. The study notebooks keep a
`task.py` beside the notebook and reference it by an absolute path built from the notebook
directory (`f"{TASK_FILE}@instruction_following"`).

Each suite run goes through `inspect_ai.eval_set`, which provides task retry and log-based resume. The
`.eval` logs under `save_dir/inspect_logs/` are the record of the run, and a re-run completes only the
missing samples of each (configuration, trial, suite) cell. Since `eval_set` matches on task identity
only, a changed protocol (seed, generate defaults, provider options, suites, fit, backend) needs a new
`save_dir` rather than a re-run into the old one. Repetition is trial-based rather than epoch-based,
and with `seed` set, each (configuration, trial) pair derives one seed.

The runner draws a `tqdm` bar over the (configuration, trial, suite) cells (`progress=True` by
default) and logs a summary line and one line per cell at INFO. `display="plain"` streams Inspect's
per-sample progress inside the currently running cell, which is the recommended setting in a notebook.
Note that `inspect_evals` tasks download their datasets from the Hugging Face Hub (some are gated) and
`.eval` logs can be large. Per-sample runtime kwargs are recorded with each model event and should
be kept small.

Every arm and every trial scores the identical sample set per task, either through explicit
`sample_ids` or through `limit=N` over the task's native dataset order. Taking the first `N`
samples is deterministic across arms, which paired comparison requires, but it is a biased
estimate of the full-benchmark score. This means that absolute scores are not directly comparable
to numbers published under other harnesses or logprob-scored protocols. The intended use is a
paired comparison against the baseline arm on identical samples, which is a single pivot on the
results frame:

```python
pivot = frame.pivot_table(index=["suite", "task", "metric"], columns="config", values="value")
deltas = pivot.sub(pivot["baseline"], axis=0)
```

The raw `.eval` logs contain per-sample generations, grades, and finish behavior, which is enough to
trace a drop in score to its cause (e.g., unparseable output rather than a wrong answer).
`SteeringEval.samples_frame` reads these logs into one row per (pipeline, trial, sample) with
per-sample scores joined to the sample metadata, which supports per-instruction-type breakdowns and
paired per-example comparisons. Inspect's log viewer and the `inspect_ai.analysis` dataframes
(`evals_df`, `samples_df`, `events_df`) support sample-level analysis directly.

Tasks with model-graded scorers need a grader model supplied through the task's own arguments
(`task_args`). The grader must be a separate model (an API model or a second local model) and
never the pipeline under evaluation, since self-grading is circular and grader traffic would
compete with evaluation traffic inside the collator. Also note that a local grader shares the GPU
with the pipeline. An API grader is preferable unless memory headroom is planned for both models.

## Authoring target-behavior tasks

Custom target-behavior evaluations are ordinary Inspect tasks, and the toolkit provides no task,
scorer, or metric classes of its own. Two working examples are in the study notebooks. The
`examples/notebooks/studies/commonsense_mcqa/` task defines a shuffled-choice MCQA task with a custom
positional-bias metric, and the `examples/notebooks/studies/instruction_following/` task passes each
prompt's instruction lines as per-sample runtime kwargs for a PASTA arm and scores every response
with both the strict IFEval checker and a local reward model loaded inside the scorer.

Controls that take per-generation parameters receive them through two tiers of runtime kwargs.
Static kwargs (`ProviderOptions.runtime_kwargs`) apply to every request. They suit catalog tasks,
whose datasets contain no steering columns, and any kwarg that is a property of the arm rather than
the sample. Per-sample kwargs are stored in `Sample.metadata` and delivered by the provided
`runtime_kwargs_solver`, which performs the sample's generation in place of a bare `generate()` in
the solver chain:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from steerability.evaluation.solvers import runtime_kwargs_solver

@task
def target_qa() -> Task:
    samples = [
        Sample(
            input="Answer with the city name only. Which city is the Eiffel Tower in?",
            target="Paris",
            metadata={"runtime_kwargs": {"substrings": ["Answer with the city name only."]}},
        ),
    ]
    return Task(dataset=MemoryDataset(samples), solver=[runtime_kwargs_solver()], scorer=includes())
```

The provider interprets every runtime kwarg, on either tier, against the arm's enabled controls. A
key declared `"row"`-scoped (a per-prompt value) reaches the control as one value per prompt row.
Per-sample values are collated row by row across a batched dispatch, and a static value is broadcast
to every row. A key declared `"call"`-scoped (one value per generate call) is passed through
unchanged and may only be delivered statically. A key that no enabled control of the arm declares is
dropped from the call and logged once per provider, which allows one task to contain the steering
inputs of every arm in an experiment, including the empty baseline. For PASTA's `substrings` the
per-row form is one `list[str]`, on both tiers. Tasks without the solver, including the entire
`inspect_evals` catalog, receive static kwargs only.

## Inspect scorers as rewards inside controls

Controls that optimize or rerank against a per-row score (PRewrite, CPO, GEPA, `best_of_n`,
`search_decoding`) consume a `SampleScorer`, a callable `(response, row) -> float` where the row
contains `"input"`, optionally `"reference"`, and any other dataset columns.
`sample_scorer_from_inspect` adapts any Inspect scorer into that form:

```python
from inspect_ai.scorer import model_graded_fact
from steerability.evaluation.scorers import sample_scorer_from_inspect

row_scorer = sample_scorer_from_inspect(model_graded_fact(model="openai/gpt-4o-mini"))
prewrite = PRewrite(initial_instruction="...", dev_set=dev_rows, row_scorer=row_scorer)
```

The adapter bridges Inspect's asynchronous scorers into synchronous control code. It works from plain
synchronous code, from inside the provider's dispatch thread, and from inside a running asyncio
event loop (a notebook), where it applies the same `nest_asyncio2` re-entry that Inspect uses.
Inside a running trio task it raises an error instead, since re-entry is impossible there. Note that a
model-graded scorer used this way runs grader traffic from inside a control's `steer()` or decode
loop. We recommend running optimizers with model-graded rewards from scripts.

The `PRewrite` example above rewards at steer time, from a fixed development set. A reranking
driver instead rewards at generate time, once per sample, and its `SampleScorer` therefore needs that
sample's row. The `SearchDriver` presets (`DeAL`, `best_of_n`, `search_decoding`) read a `reward_params`
runtime kwarg for this, declared `"row"`-scoped, and `SampleSequenceScorer` merges it into the row
the scorer sees (`{"input": prompt, **reward_params}`). We store each sample's reference on
`Sample.metadata` as one mapping and deliver it with `runtime_kwargs_solver`, exactly as for PASTA's
`substrings`:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from steerability.evaluation.scorers import sample_scorer_from_inspect
from steerability.evaluation.solvers import runtime_kwargs_solver
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.output_control.common.scorers.sample import SampleSequenceScorer

row_scorer = sample_scorer_from_inspect(includes())          # reads row["reference"]
control = BestOfN(n=8, scorer=SampleSequenceScorer(row_scorer))

@task
def reranked_qa() -> Task:
    samples = [
        Sample(
            input="Which city is the Eiffel Tower in?",
            target="Paris",
            metadata={"runtime_kwargs": {"reward_params": {"reference": "Paris"}}},
        ),
    ]
    return Task(dataset=MemoryDataset(samples), solver=[runtime_kwargs_solver()], scorer=includes())
```

Since the collator refuses one runtime-kwarg name on both tiers, an arm that passes per-sample
references through `reward_params` cannot also pass per-arm reward hyperparameters under the same
name. Put those in the scorer's constructor instead.
