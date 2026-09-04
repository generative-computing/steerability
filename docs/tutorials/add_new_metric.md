# Adding your own metric

Evaluation metrics are intended to be consumed by use cases. This guide illustrates how to add new metrics. Broadly,
metrics are of two categories:

- Generic metrics: metrics that can be called from any use case.
- Custom metrics:  metrics that are intended to be called from a specific use case (e.g., question answering)

Depending on the metric category, structure your files in `aisteer360/evaluation/metrics` as follows:
```
aisteer360/
└── evaluation/
    └── metrics/
        ├── custom/
        │   └── <my_use_case>/
        │       └── <custom_metric_name>.py
        └── generic/
            └── <generic_metric_name>.py
```

Implementation of a new metric is the same regardless of the metric's category. Both generic and custom metrics can be
one of two types:

- standard: subclasses `Metric` from `aisteer360.evaluation.metrics.base`
- LLM-as-a-judge: subclasses `LLMJudgeMetric` from `aisteer360.evaluation.metrics.base_judge`

All metrics compute scores using at minimum a `response`, with an optional field `prompt`. Any other necessary arguments
can be passed into the metric's `compute` method via `kwargs`.


## Implementing a standard metric

Standard metrics are any metric that require completely custom `compute` logic. Any unstructured computation can be
implemented as a function of `responses`, `prompts`, and `kwargs`. Any necessary parameter initialization should be
added to the metric's constructor (`__init__`).

Below is an example implementation of a `DistinctN` metric (for computing unigrams, bigrams, etc.).

```python
from itertools import islice
from typing import Any

from aisteer360.evaluation.metrics.base import Metric


class DistinctN(Metric):
    """Corpus-level Distinct-n (Li et al., 2015).

    Distinct-n = (# unique n-grams) / (# total n-grams)

    Args:
        n (int, optional): Size of the n-gram.

    Li, J., Galley, M., Brockett, C., Gao, J. and Dolan, B., 2015.
    A diversity-promoting objective function for neural conversation models.
    arXiv preprint arXiv:1510.03055.
    """

    def __init__(self, n: int = 2):
        super().__init__()
        self.n = n

    def _ngrams(self, tokens: list[str]):
        return zip(*(islice(tokens, i, None) for i in range(self.n)))

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        total_ngrams = 0
        unique_ngrams: set[tuple[str, ...]] = set()

        for response in responses:
            response = response.lower()
            tokens = response.split()
            grams = list(self._ngrams(tokens))
            total_ngrams += len(grams)
            unique_ngrams.update(grams)

        score = len(unique_ngrams) / total_ngrams if total_ngrams else 0.0
        return {
            f"distinct_{self.n}": score
        }
```

The above metric is called as follows:

```python
from aisteer360.evaluation.metrics.generic.distinct_n import DistinctN

responses = [
    "I love exploring new places.",
    "I love exploring new places.",
    "Traveling is my passion."
]

unigram = DistinctN(n=1)

unigrams = unigram.compute(responses=responses)
```


## Implementing an LLM-as-a-judge metric

To facilitate evaluation of more complex quantities, the toolkit provides a base class for LLM-as-a-judge metrics
(`LLMJudgeMetric`) that extends the `Metric` class. Judge generation runs through the execution backend seam, so a judge
works on the in-process Hugging Face backend, the offline vLLM engine, and a vLLM server with no judge-specific code.

Configuration is declarative. A judge subclass sets its prompt template and scale as class attributes; a constructor
keyword overrides the class attribute per instance. The prompt template must contain a `{response}` placeholder (and the
scale bounds `{lower_bound}` / `{upper_bound}` when the built-in structured format instructions reference them), and may
contain a `{prompt}` placeholder. For instance, the `Factuality` metric uses the `response` (the model's answer) and the
`prompt` (the question).

```python
from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric


_PROMPT = """\
You are a careful fact-checker.

Considering only verifiable facts, rate the response's factual accuracy with respect to the prompt on a scale from
{lower_bound} (completely incorrect) to {upper_bound} (fully correct).

PROMPT:
{prompt}

RESPONSE:
{response}

What is your score?
"""


class Factuality(LLMJudgeMetric):
    """Judge factual correctness of an answer to a question."""

    prompt_template = _PROMPT
    scale = (1, 5)
```

A judge is configured by a model reference and a backend, never by a live model object. Pass the judge model at
construction with `model=` (in-process Hugging Face by default) or with `backend=` for a specific backend. Generation
parameters are given in the normalized vocabulary via `gen_kwargs`; `n` is the multi-sample knob (scores are averaged
across the `n` candidates), and unknown keys raise.

```python
from aisteer360.algorithms.core.execution import BackendSpec
from aisteer360.evaluation.metrics.generic.relevance import Relevance

# in-process Hugging Face judge, sampling three candidates per response
answer_relevance = Relevance(
    model="meta-llama/Llama-3.2-3B-Instruct",
    gen_kwargs={"temperature": 0.8, "n": 3},
)

# the same judge on the offline vLLM engine, or a vLLM server carrying base_url
vllm_relevance = Relevance(backend=BackendSpec(kind="vllm", model="meta-llama/Llama-3.2-3B-Instruct"))

# run the metric
questions = ["What is the capital of Ireland?"]
answers = ["Dublin."]
scores = answer_relevance(responses=answers, prompts=questions)
```

Backends are cached by spec, so two metrics configured with equal specs share one loaded judge. Model placement and
dtype travel as spec options (given as plain data), e.g.
`BackendSpec(kind="huggingface", model=..., options={"device_map": "cuda:1", "hf_model_kwargs": {"torch_dtype": "bfloat16"}})`.

The cache is released and emptied with `release_metric_backends()` (from `aisteer360.evaluation.metrics`).
`Benchmark.run()` calls it when the run finishes or fails; outside a benchmark the caller releases when done. A metric
resolves its backend per `compute()`, so it works again after a release (the next call boots the engine again). The
offline vLLM engine is one-per-process, so a `vllm` judge alongside a `vllm` steering pipeline is unsupported; run the
judge on `huggingface`, or point one side at a server with `BackendSpec(kind="vllm-serve", ...)`.

### Extra template fields

A template placeholder beyond the built-ins (`response`, `prompt`, `lower_bound`, `upper_bound`) is extracted at
construction and resolved per item from the keyword arguments `compute` receives. Each extra field's value must be a
sequence aligned with `responses`, or a scalar (broadcast to every item). This lets a judge grade against per-item
context without a custom judge loop.

```python
_PROMPT = """\
Rate, from {lower_bound} to {upper_bound}, how well the RESPONSE answers the QUESTION given the CONTEXT.

QUESTION:
{question}

CONTEXT:
{context}

RESPONSE:
{response}

What is your score?
"""


class Groundedness(LLMJudgeMetric):
    """Judge how well a response is grounded in a supplied context."""

    prompt_template = _PROMPT
    scale = (1, 5)


groundedness = Groundedness(model="meta-llama/Llama-3.2-3B-Instruct")
scores = groundedness(
    responses=["Dublin is the capital."],
    question=["What is the capital of Ireland?"],
    context=["Ireland's capital city is Dublin."],  # aligned with responses
)
```

For non-numeric judgments (e.g. a yes/no decision), set `structured_output = False` and provide a `parser` that maps the
decoded response to a float; see the TruthfulQA `Truthfulness` and `Informativeness` metrics for a binary example.

To call metrics, please see the tutorial on [adding your own use case](add_new_use_case.md).
