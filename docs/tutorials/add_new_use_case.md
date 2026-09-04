# Adding your own use case

Use cases define tasks for a model and specify how performance on that task (via the model's generations) is
measured. A use case instance is intended to be consumed by a benchmark. Please see the
[tutorial for adding your own benchmark](add_new_benchmark.md) for instructions on how to run a use case.

For the purposes of this tutorial, we will focus on a simple multiple-choice QA task, which we term `CommonsenseMCQA`,
based on the [CommonsenseQA dataset](https://huggingface.co/datasets/tau/commonsense_qa).

## Setup

The only required file to create a use case is `use_case.py`. This file must be placed in a new directory
`<custom_use_case>`, of your choosing, in `aisteer360/evaluation/use_cases`:
```
aisteer360/
└── evaluation/
    └── use_cases/
        └── <custom_use_case>/
            └── use_case.py
```

The `CommonsenseMCQA` use case is located at`commonsense_mcqa/use_case.py`. Every use case is instantiated by providing
`evaluation_data`, the data that the model uses to produce generations, and `evaluation_metrics`, the functions to
evaluate the model's behavior. A use case may declare additional constructor parameters specific to it (e.g.,
`num_shuffling_runs` for `CommonsenseMCQA`); each is declared as a class-level annotation and passed as a keyword. A
bare annotation makes the parameter required, and an annotation with a class-attribute default makes it optional.
Unknown keywords and missing required parameters both raise `TypeError` at construction. For instance,

```python
from aisteer360.evaluation.use_cases.commonsense_mcqa.use_case import CommonsenseMCQA
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_accuracy import MCQAAccuracy
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_positional_bias import MCQAPositionalBias

commonsense_mcqa = CommonsenseMCQA(
    evaluation_data="./data/evaluation_qa.jsonl",
    evaluation_metrics=[
        MCQAAccuracy(),
        MCQAPositionalBias()
    ],
    num_shuffling_runs=20
)
```

Evaluation data should contain any information that is relevant for evaluating the model's performance. For our example
task, this data (stored as a `jsonl` file) contains the following information:

```python
{
    "id": "033b86ec-e7c1-40ac-8c9e-27ebfba41faf",
    "question": "Where would someone keep a grandfather clock?",
    "answer": "house",
    "choices": ["desk", "exhibition hall", "own bedroom", "house", "office building"]
}
```

We've implemented two custom metrics for our use case: `MCQAAccuracy` for evaluating the accuracy statistics of choices
with respect to the ground truth answers, and `MCQAPositionalBias` for measuring how much the model is biased toward
choices in a given position. This tutorial will not go into depth about these metrics; please see their implementations
at `aisteer360/evaluation/metrics/custom/commonsense_mcqa` for details. For details on contributing any new metrics
(either generic metrics or those custom to a use case), please see the
[tutorial on adding your own metric](./add_new_metric.md).


## Defining the use case class

Each use case subclasses the base `UseCase` class (`aisteer/evaluation/use_cases/base.py`), which contains all necessary
initialization logic. Please **do not** write an `__init__` for your custom use case. Instead, declare each use-case
parameter as a class-level annotation, e.g., `num_shuffling_runs: int`. A bare annotation makes the parameter required;
adding a class-attribute default (e.g., `num_shuffling_runs: int = 20`) makes it optional with that default. The base
constructor reads each declared parameter from the keyword arguments and sets it as an instance attribute, so
`num_shuffling_runs` is available at runtime as `self.num_shuffling_runs`. A keyword that is not a declared parameter
raises `TypeError`, as does a missing required parameter. We additionally advise that contributors write validation
logic for their evaluation data (via `validate_evaluation_data`) based on the required columns
(`_EVALUATION_REQ_KEYS`); the base constructor calls it on each retained instance (after shuffling and sampling), so a
schema violation raises `ValueError` at construction with the offending `evaluation_data[<index>]` prefix.

For our example use case:

```python
from aisteer360.evaluation.use_cases.base import UseCase

_EVALUATION_REQ_KEYS = [
    "id",
    "question",
    "answer",
    "choices"
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CommonsenseMCQA(UseCase):
    """
    Commonsense multiple-choice question answering use case.

    """
    num_shuffling_runs: int

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        if "id" not in evaluation_data.keys():
            raise ValueError("The evaluation data must include an 'id' key")

        missing_keys = [col for col in _EVALUATION_REQ_KEYS if col not in evaluation_data.keys()]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if any(
            key not in evaluation_data or evaluation_data[key] is None or
            (isinstance(evaluation_data[key], float) and math.isnan(evaluation_data[key]))
            for key in _EVALUATION_REQ_KEYS
        ):
            raise ValueError("Some required fields are missing or null.")
```

!!! note
    We require that your evaluation data contains a column named `id`, serving to assign a unique identifier to each
    datapoint. This is required by the `Benchmark` class to ensure that any `runtime_kwargs` (any arguments that may be
    required by the controls at inference time; see the [tutorial on adding a benchmark](./add_new_benchmark.md) for
    details) are consistently populated.

Any use case class must define two required methods (`generate` and `evaluate`) and an optional method (`export`).
Implementation of these methods is outlined below.


### Generation via `generate`

The `generate` method produces outputs as a function of the evaluation data (accessible via `self.evaluation_data`). The
generate method must return `generations` as a list of dictionaries (i.e., `list[dict[str, Any]]`). Each dictionary must
contain at minimum a `response` key and can optionally contain a `prompt` key. The dictionary should also contain any
number of keyword args that may be necessary for later computation of metric scores. In other words, `generations`
should contain everything that the use case's evaluate method needs to run its evaluation.


The `generate` method for `CommonsenseMCQA` is defined as follows:
```python
def generate(
    self,
    model_or_pipeline,
    tokenizer,
    gen_kwargs: dict | None = None,
    runtime_overrides: dict[str, dict[str, Any]] | None = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> list[dict[str, Any]]:

    if not self.evaluation_data:
        print('No evaluation data provided.')
        return []
    gen_kwargs = dict(gen_kwargs or {})

    # form prompt data; each shuffled copy inherits its instance's columns
    prompt_data = []
    for instance in self.evaluation_data:
        question = instance['question']
        answer = instance['answer']
        choices = instance['choices']
        # shuffle order of choices for each shuffling run
        for _ in range(self.num_shuffling_runs):

            lines = ["You will be given a multiple-choice question and asked to select from a set of choices."]
            lines += [f"\nQuestion: {question}\n"]

            # shuffle
            choice_order = list(range(len(choices)))
            random.shuffle(choice_order)
            for i, old_idx in enumerate(choice_order):
                lines.append(f"{_LETTERS[i]}. {choices[old_idx]}")

            lines += ["\nPlease only print the letter corresponding to your choice."]
            lines += ["\nAnswer:"]

            prompt_data.append({
                **instance,
                "prompt": "\n".join(lines),
                "reference_answer": _LETTERS[choice_order.index(choices.index(answer))],
            })

    # batch template/generate/decode
    choices = batch_retry_generate(
        prompt_data=prompt_data,
        model_or_pipeline=model_or_pipeline,
        tokenizer=tokenizer,
        parse_fn=self._parse_letter,
        gen_kwargs=gen_kwargs,
        runtime_overrides=runtime_overrides,
        batch_size=batch_size,
    )

    # store
    generations = [
        {
            "response": choice,
            "prompt": prompt_dict["prompt"],
            "question_id": prompt_dict["id"],
            "reference_answer": prompt_dict["reference_answer"],
        }
        for prompt_dict, choice in zip(prompt_data, choices)
    ]

    return generations

@staticmethod
def _parse_letter(response) -> str:
    valid = _LETTERS
    text = re.sub(r"^\s*(assistant|system|user)[:\n ]*", "", response, flags=re.I).strip()
    match = re.search(rf"\b([{valid}])\b", text, flags=re.I)
    return match.group(1).upper() if match else None
```

The `generate` method is designed to be called, via the benchmark class, on either a base (unsteered) model or a
steering pipeline, and thus the "model" object passed into `generate` is referenced via the required argument
`model_or_pipeline`. In addition, the `generate` method requires an associated `tokenizer` and
(optionally) any `gen_kwargs` and `runtime_overrides`. The current `CommonsenseMCQA` use case does not make use of any
`runtime_overrides` (since none of the studied controls in the associated benchmark require inference time arguments);
please see the [instruction following benchmark notebook](../examples/notebooks/benchmarks/instruction_following/instruction_following.ipynb)
for an example of how these overrides are defined and used.

The first step in defining the `generate` method is to construct the prompt data. For the example MCQA task, our goal is
to (robustly) evaluate a model's ability to accurately answer (common sense) multiple choice questions, and thus we
present the same question to the model under various orderings/shufflings of the answers. Each prompt row spreads its
source instance (`**instance`) and then sets the constructed `prompt` (the question) and `reference_answer` for that
shuffle. Spreading the instance means every prompt row carries the instance's own columns, so `runtime_overrides` map
per row (a `runtime_overrides` column resolves against these rows). Constructed keys such as `prompt`,
`reference_answer`, and `thinking` shadow same-named instance columns, so name any override column distinctly from them.

Once the prompt data has been prepared for the use case, it then needs to be passed into the model (or steering
pipeline) to generate responses. We strongly advise that contributors make use of the `batch_retry_generate` helper
function to aid in this process. This function implements conversion to a model's chat template, batch encoding, batch
generation, batch decoding, and parsing (via `parse_fn`), and retry logic for a given list of prompts. For the example
use case, we define the parsing function as a custom `parse_letter` method, such that the model's choices can be
reliably extracted from its response (and stored as `choices`).

For reasoning models, `batch_retry_generate` splits each decoded continuation into a thinking segment and an answer
segment (the `think_tags` parameter, default `("<think>", "</think>")`). The raw text and `parse_fn` see the answer
segment only, so reasoning tokens do not blend into parsing or scoring. To retain the reasoning, pass
`return_thinking=True` and store the returned list under a `"thinking"` column, as the built-in use cases do; pass
`think_tags=None` to disable the split and keep the full continuation.

Lastly, we store each choice under the `response` key along with the prompt, question ID, and reference answer across
all elements of the prompt data.


### Evaluation via `evaluate`

The `evaluate` method defines how to process the model's generations (produced by the `generate` method) via evaluation
metrics. All evaluation metrics that were passed in as the use case's construction are used in the evaluation.

```python
def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:

    eval_data = {
        "responses": [generation["response"] for generation in generations],
        "reference_answers": [generation["reference_answer"] for generation in generations],
        "question_ids": [generation["question_id"] for generation in generations],
    }

    scores = {}
    for metric in self.evaluation_metrics:
        scores[metric.name] = metric(**eval_data)

    return scores
```

A useful pattern for evaluation logic is to first define the necessary quantities across all generations (`eval_data`),
then simply pass these into each metric (via `**eval_data`). Note that for the example use case, the metrics make use of
the question IDs by computing statistics across the shuffled choice order for each question.


### Formatting and exporting via `export`

The `export` method (optional) is useful for storing benchmark evaluations for later plotting or analysis, e.g.,
comparing benchmark results across multiple base models. The `export` method allows the user to specify custom
processing before exporting. In the simplest case, the method can just save the profiles to a `json` file, as is done
in the example use case:

```python
def export(self, profiles: dict[str, Any], save_dir) -> None:

    with open(Path(save_dir) / "profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)
```


---


For a complete example of the `CommonsenseMCQA` use case, please see the implementation located at
`aisteer360/evaluation/use_cases/commonsense_mcqa/use_case.py`. For instructions on how to build an associated benchmark, please
see the [tutorial](./add_new_benchmark.md) and the [notebook](../examples/notebooks/benchmarks/commonsense_mcqa/commonsense_mcqa.ipynb).
