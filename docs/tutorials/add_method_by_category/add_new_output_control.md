# Adding an output control method

Output control methods constrain or transform what leaves the decoder.

## Config first, subclass second

The first design decision is config first, subclass second. Before writing a class, check whether the method is an
assignment of a config of one of the [generic controls](../../concepts/controls.md#generic-controls). Most output
methods from the literature map onto one of them:

- a method that reshapes the next-token distribution from a per-candidate score is a [`ValueGuidance`](../../concepts/controls.md#generic-controls) config (FUDGE, ARGS, RAD, SASA)
- one that mixes weighted full-vocabulary log-prob sources is a [`ContrastiveGuidance`](../../concepts/controls.md#generic-controls) config (DExperts, contrastive decoding, proxy-tuning)
- one that changes the shape of the search (propose, score, keep, iterate) is a [`SearchDecoding`](../../concepts/controls.md#generic-controls) config (best-of-N, self-consistency, DeAL)
- one that splices forced and generated segments is a [`PhasedDecoding`](../../concepts/controls.md#generic-controls) config (budget forcing, response prefill, thinking intervention)
- one that stops on a substring, token, or budget is a [`StoppingRules`](../../concepts/controls.md#generic-controls) config

If so, implement the method as a config, not a class. When a config earns a name through use, promote it with a small preset
subclass over the generic that maps its named args onto the generic's fields (the pattern the named methods already
follow, with `BestOfN` over `SearchDecoding`'s shape and `BudgetForcing` over `PhasedDecoding`'s):

```python
class BestOfN(SearchDecoding):
    """Sample n continuations, return the scorer's argmax (rejection sampling)."""
    Args = BestOfNArgs          # fields: n, scorer

    def _configure(self):
        self.num_candidates = self.n
        self.keep_k = 1
        self.max_iterations = 1
        self.segment_len = None
        self.propose_mode = "sample"
        self.tokenizer = None
```

Write a full control class only when the method needs behavior no config expresses: a new candidate policy, a new
value/source/scorer component, or a bespoke decode loop.

## Contribute or drive?

If you are writing a class, output controls participate through one of two mechanisms, and the next design
decision is choosing which:

- **Contribute**: supply logits processors and/or stopping criteria. The pipeline composes every step-level control's
  processors in `controls`-list order into one list (and likewise for stopping criteria), then hands both lists to
  whichever driver owns the loop. A step-level control never runs the decode loop itself and therefore composes with
  other step-level controls and with a driver. **Override**: `get_logits_processors` and/or `get_stopping_criteria`.
- **Drive**: own the decode loop. A driver subclasses `DecodingDriver` and implements `decode(...)`, applying the
  composed processors and stopping criteria in every forward pass it issues. Since the loop does not compose, a pipeline admits at most one
  enabled driver. With none, decoding defaults to the model's own `generate`. **Override**: `decode`.

As a rule of thumb, if the method reshapes the next-token distribution one step at a time (reward shifts, contrastive
mixtures, constraint masks), it is a step-level control. If it changes the shape of the search (lookahead, re-ranking,
phased generation, best-of-N), it is a driver.

Both modes may also implement `steer()` (one-time preparation, e.g., loading a reward model) and `cleanup()` (release
those resources). Each method is a package directory with `args.py`, `control.py`, and a `STEERING_METHOD` export in
`__init__.py` that the registry discovers:

```python
from .control import KeywordBooster
from .args import KeywordBoosterArgs

STEERING_METHOD = {
    "category": "output_control",
    "name": "keyword_booster",
    "control": KeywordBooster,
    "args": KeywordBoosterArgs,
}
```

## Contribute: logits processors

`KeywordBooster` adds a fixed bias to the logits of a set of keyword tokens at every step, making those words more
likely. It edits the distribution one step at a time and is therefore a step-level control.

The args dataclass declares the hyperparameters. The keyword strings are tied to the prompt and supplied at inference
time, arriving via `runtime_kwargs` rather than the constructor. The control declares the name it consumes in
`RUNTIME_KWARGS_SCHEMA`. All controls read from the one `runtime_kwargs` dict, and the pipeline warns at `steer()`
time when two controls declare the same name.

```python
from dataclasses import dataclass, field
from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class KeywordBoosterArgs(BaseArgs):
    boost: float = field(
        default=5.0,
        metadata={"help": "Additive logit bias applied to each keyword token."},
    )

    def __post_init__(self):
        if self.boost < 0:
            raise ValueError("`boost` must be non-negative.")
```

The control returns a fresh processor from `get_logits_processors` on every call, since the hook is invoked once per
`generate()`/`compute_logprobs()` to isolate per-generation state. A processor is any callable
`(input_ids, scores) -> scores` following the Hugging Face `LogitsProcessor` convention:

```python
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.keyword_booster.args import KeywordBoosterArgs


class KeywordBooster(OutputControl):
    """Adds a fixed logit bias to a set of keyword tokens at every decoding step."""

    Args = KeywordBoosterArgs
    RUNTIME_KWARGS_SCHEMA = [{"name": "keywords"}]

    tokenizer: PreTrainedTokenizer | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__) -> PreTrainedModel:
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        runtime_kwargs = runtime_kwargs or {}
        keywords = runtime_kwargs.get("keywords", [])
        keyword_ids = [
            token_id
            for word in keywords
            for token_id in self.tokenizer.encode(word, add_special_tokens=False)
        ]

        def _boost(prefix_ids, scores):
            scores = scores.clone()
            for token_id in keyword_ids:
                scores[:, token_id] += self.boost
            return scores

        return [_boost]  # fresh instance per call
```

Because it only contributes, `KeywordBooster` composes freely. For instance, `controls=[KeywordBooster(...), DeAL(...)]`
applies the boost inside every DeAL rollout, and `controls=[KeywordBooster(...)]` alone runs under the default
`model.generate` loop.

!!! note "Processor purity"
    A processor must behave as a function of `(prefix_ids, scores)`. Drivers may restart, rewind, or reorder sequences
    (segment search re-enters from a shorter frontier and beam search permutes rows), and `compute_logprobs` replays
    prefixes teacher-forced. Any internal state must therefore be memoization keyed on the prefix. Subclass
    [`PrefixKeyedProcessor`](../../reference/algorithms/output_control/common.md) to get this contract mechanically.
    It calls your `reset_state(input_ids)` whenever the observed prefix no longer extends the last one.

By default a step-level control's logits edits also apply during `compute_logprobs`, and scoring therefore reflects
the steered distribution. Set `include_in_scoring = False` (a class attribute) to opt out when the per-position cost is prohibitive.

## Drive: a decoding driver

`ShortestOfN` samples N continuations and returns the shortest one. It changes the shape of the search and is
therefore a driver. A driver receives the composed `logits_processors` / `stopping_criteria` as explicit parameters
and must apply them in every forward pass it issues. Delegating to
`model.generate(..., logits_processor=..., stopping_criteria=...)` satisfies this. The helper `stack_generate_kwargs` builds those two kwargs, including each only when non-empty.

```python
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.output_control.base import DecodingDriver, stack_generate_kwargs


@dataclass
class ShortestOfNArgs(BaseArgs):
    n: int = field(default=4, metadata={"help": "Number of candidates to sample."})

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("`n` must be >= 1.")


class ShortestOfN(DecodingDriver):
    """Samples `n` continuations and returns the shortest (fewest non-pad tokens)."""

    Args = ShortestOfNArgs

    tokenizer: PreTrainedTokenizer | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__) -> PreTrainedModel:
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, **gen_kwargs) -> torch.Tensor:
        if input_ids.size(0) != 1:
            raise NotImplementedError("ShortestOfN handles one prompt at a time (batch size 1).")

        extra = stack_generate_kwargs(logits_processors, stopping_criteria)  # apply the composed processors and stopping criteria
        kwargs = dict(gen_kwargs)  # merge first so the driver's settings win without duplicate-kwarg errors
        kwargs.update({"do_sample": True, "num_return_sequences": self.n})
        candidates = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **extra,
            **kwargs,
        )

        prompt_len = input_ids.size(1)
        pad_id = self.tokenizer.pad_token_id
        lengths = [int((row[prompt_len:] != pad_id).sum()) for row in candidates]
        best = int(torch.tensor(lengths).argmin())
        return candidates[best].unsqueeze(0)  # full sequence: prompt + continuation
```

!!! note "The driver contract"
    `logits_processors` and `stopping_criteria` are the composed lists for this generation, and a driver must apply
    them in every forward pass. The `gen_kwargs` reaching `decode` never contain `logits_processor` or
    `stopping_criteria`, since the pipeline removes caller-supplied ones and composes them into these lists. `decode`
    returns the full sequence ids (prompt plus continuation), and the pipeline strips the prompt prefix. The pipeline
    also passes `session=`, a `SteeredSession` that contains this generation's control entries. Resolving the rollout
    callable with `resolve_generate_callable(model, runtime_kwargs, session=session)` makes the driver's rollouts run
    steered on any backend. A driver can override `max_rollouts_per_query()` to declare an upper bound on the
    continuations it generates per input row (`ShortestOfN` returns `self.n`). The default returns `None`, meaning no
    static bound.

## Prefer the `common` library

Most methods do not start from scratch. The [`output_control.common`](../../reference/algorithms/output_control/common.md)
library factors the category into reusable components, and the methods in the toolkit are thin recipes over them:

- `ValueGuidedProcessor` (step-level candidate scoring): `RAD`, `SASA`.
- `ContrastiveMixtureProcessor` (mix full-vocabulary logit sources): `DExperts`, `ContrastiveDecoding`.
- `SearchDriver` (propose, score, keep top-k, iterate): `DeAL`, `BestOfN`.
- `PhasedDriver` (forced/generated segments with boundary rules): `BudgetForcing`.

A driver built on `SearchDriver` or `PhasedDriver` is a preset. It declares an `Args` dataclass, calls
`OutputControl.__init__` from its own `__init__`, and overrides `_configure()` to map its mirrored args onto the
generic base's fields, never bypassing the parent constructor. See `deal/control.py` and `budget_forcing/control.py`
for the pattern. An argument-free control (no hyperparameters) sets `Args = None` and takes no constructor arguments.

When adding a component to `common`, follow its naming convention. Within a `common/<family>/` folder, the primary
class in `<name>.py` is `<Name><FamilySingular>` (for example `values/classifier.py` defines `ClassifierValue`), and
the family base is in `base.py`. Top-level `common/*.py` modules (such as `candidates.py` and `criteria.py`) are
collection or helper modules exempt from the suffix rule.

## Running the control

Either mode is instantiated and added to a pipeline the same way:

```python
from steerability.algorithms.output_control.keyword_booster.control import KeywordBooster
from steerability.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

keyword_booster = KeywordBooster(boost=6.0)

pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[keyword_booster],
    device_map="auto",
)
pipeline.steer()

prompt = "Explain linear algebra in two sentences."
chat = pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = pipeline.tokenizer(chat, return_tensors="pt").to(pipeline.model.device)

output = pipeline.generate(
    input_ids=inputs.input_ids,
    runtime_kwargs={"keywords": ["matrix", "vector"]},
    max_new_tokens=50,
    do_sample=True,
)
print(pipeline.tokenizer.decode(output[0], skip_special_tokens=True))

# different keywords can be supplied at inference time, without re-steering
output = pipeline.generate(
    input_ids=inputs.input_ids,
    runtime_kwargs={"keywords": ["eigenvalue", "determinant"]},
    max_new_tokens=50,
    do_sample=True,
)
print(pipeline.tokenizer.decode(output[0], skip_special_tokens=True))
```
