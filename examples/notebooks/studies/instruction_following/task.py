"""Single-instruction Split-IFEval prompts, scored by strict/loose IFEval checking and a reward model.

A balanced set of single-instruction prompts from `ibm-research/Split-IFEval` is selected (a fixed
number per instruction type), each carrying its instruction lines as per-sample runtime kwargs so a
PASTA arm can steer attention onto them. `instruction_checker()` runs the IFEval checker library and
`reward_score()` scores each response with the OpenAssistant DeBERTa reward model;
`runtime_kwargs_solver()` performs the generation, so one task serves both the PASTA arm and the
empty baseline (which ignores the runtime kwargs no enabled control declares).

Heavy imports (torch, transformers, the IFEval checker) live inside the functions that need them, so
the module imports cleanly with no network access for the notebook to read its helper functions.
"""
import random
from typing import Any, Iterable, Mapping, Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, mean, metric, scorer, stderr
from inspect_ai.solver import TaskState

from steerability.evaluation.solvers import runtime_kwargs_solver

SPLIT_IFEVAL_PATH = "ibm-research/Split-IFEval"

DEFAULT_INSTRUCTION_TYPES = (
    "keywords:forbidden_words",
    "detectable_format:number_highlighted_sections",
    "language:response_language",
    "startend:end_checker",
)


def load_records() -> list[dict]:
    """The Split-IFEval train split as a list of plain-dict records (network access).

    Returns:
        One dict per record with keys `key` (int), `prompt` (str), `instruction_id_list`
        (list[str]), `kwargs` (list[dict], one checker-argument dict per instruction), and
        `instructions` (list[str], the instruction lines verbatim as they appear in the prompt).
    """
    from datasets import load_dataset

    dataset = load_dataset(SPLIT_IFEVAL_PATH, split="train")
    return [dict(record) for record in dataset]


def select_records(
    records: Iterable[dict],
    instruction_types: Sequence[str],
    per_type: int,
    sample_seed: int,
) -> list[dict]:
    """Balanced, deterministic selection of single-instruction records.

    Keeps records whose `instruction_id_list` has exactly one entry that is in
    `instruction_types`, groups them by instruction id in dataset order, and draws
    `min(per_type, len(group))` records per group with
    `random.Random(f"{sample_seed}:{instruction_id}")`. String seeding is deterministic across
    platforms and Python versions, so the selection is a pure function of its arguments and
    identical across arms and trials. Groups are emitted in the order of `instruction_types`.

    Args:
        records: An iterable of Split-IFEval records.
        instruction_types: Instruction ids to keep, one group per id, emitted in this order.
        per_type: Maximum number of records drawn per instruction type.
        sample_seed: Base seed for the per-type deterministic draw.

    Returns:
        The selected records, in `instruction_types` order.

    Raises:
        ValueError: If an instruction type has no single-instruction records.
    """
    wanted = set(instruction_types)
    grouped: dict[str, list[dict]] = {instruction_id: [] for instruction_id in instruction_types}
    for record in records:
        instruction_id_list = record["instruction_id_list"]
        if len(instruction_id_list) != 1:
            continue
        instruction_id = instruction_id_list[0]
        if instruction_id in wanted:
            grouped[instruction_id].append(record)

    selected: list[dict] = []
    for instruction_id in instruction_types:
        group = grouped[instruction_id]
        if not group:
            raise ValueError(f"No single-instruction records for instruction type {instruction_id!r}.")
        rng = random.Random(f"{sample_seed}:{instruction_id}")
        selected.extend(rng.sample(group, min(int(per_type), len(group))))
    return selected


def clean_kwargs(kwargs: Sequence[Mapping]) -> list[dict]:
    """Checker kwargs with `None` entries dropped and integral floats coerced to `int`.

    The IFEval checkers reject unknown keyword arguments and some of them use their integer
    arguments in integer contexts, so the dataset's padded, float-valued dicts are normalized
    before they reach `InputExample`. Values that are not integral floats pass through unchanged
    (strings, lists of strings, booleans).

    Args:
        kwargs: One checker-argument mapping per instruction.

    Returns:
        One cleaned dict per instruction.
    """
    cleaned: list[dict] = []
    for entry in kwargs:
        row: dict[str, Any] = {}
        for key, value in entry.items():
            if value is None:
                continue
            if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
                row[key] = int(value)
            else:
                row[key] = value
        cleaned.append(row)
    return cleaned


def to_sample(record: dict) -> Sample:
    """One `Sample` per record, carrying the prompt's instruction lines as PASTA runtime kwargs.

    The `substrings` runtime kwarg is the per-row form PASTA declares (one `list[str]` per
    sample); the strings are verbatim slices of the prompt (they include the leading `"- "` of
    each instruction line), which is what PASTA's offset-mapping locator matches against.

    Args:
        record: A Split-IFEval record.

    Returns:
        The sample, with metadata carrying `prompt`, `instruction_id`, `instruction_id_list`,
        the cleaned `kwargs`, and `runtime_kwargs`.
    """
    return Sample(
        id=record["key"],
        input=record["prompt"],
        metadata={
            "prompt": record["prompt"],
            "instruction_id": record["instruction_id_list"][0],
            "instruction_id_list": list(record["instruction_id_list"]),
            "kwargs": clean_kwargs(record["kwargs"]),
            "runtime_kwargs": {"substrings": list(record["instructions"])},
        },
    )


def profile_records(records: Iterable[dict], exclude_types: Sequence[str]) -> list[dict]:
    """Single-instruction records whose instruction type is not in `exclude_types`.

    The task-agnostic profiling pool: every record with exactly one instruction whose id is not
    an evaluation type, so the profiling set is disjoint from the evaluation set by construction.

    Args:
        records: An iterable of Split-IFEval records.
        exclude_types: Instruction ids to exclude (the evaluation types).

    Returns:
        The matching records, in dataset order.
    """
    excluded = set(exclude_types)
    kept: list[dict] = []
    for record in records:
        instruction_id_list = record["instruction_id_list"]
        if len(instruction_id_list) == 1 and instruction_id_list[0] not in excluded:
            kept.append(record)
    return kept


def to_profile_row(record: dict) -> dict:
    """One `HeadProfile` row per record, carrying the prompt and its instruction lines.

    `"input"` is the prompt and `"substrings"` its instruction lines in PASTA's per-row form;
    `"group"` is the record's single instruction id, which enables the per-group statistics. The
    remaining keys (`"key"`, `"instruction_id_list"`, `"kwargs"`) let the scorer build the IFEval
    `InputExample`.

    Args:
        record: A single-instruction Split-IFEval record.

    Returns:
        The profiling row.
    """
    return {
        "input": record["prompt"],
        "substrings": list(record["instructions"]),
        "group": record["instruction_id_list"][0],
        "key": record["key"],
        "instruction_id_list": list(record["instruction_id_list"]),
        "kwargs": clean_kwargs(record["kwargs"]),
    }


def _follow(response: str, row: Mapping, strict: bool) -> float:
    """1.0 when `response` follows every instruction of `row` under the IFEval checker, else 0.0."""
    from instruction_following_eval.evaluation import InputExample, test_instruction_following

    example = InputExample(
        key=row["key"],
        instruction_id_list=row["instruction_id_list"],
        prompt=row["input"],
        kwargs=row["kwargs"],
    )
    result = test_instruction_following(example, response, strict=strict)
    return float(result.follow_all_instructions)


def strict_follow(response: str, row: Mapping) -> float:
    """Strict IFEval follow score of `response` against a `to_profile_row` row (0.0 or 1.0)."""
    return _follow(response, row, strict=True)


def loose_follow(response: str, row: Mapping) -> float:
    """Loose IFEval follow score of `response` against a `to_profile_row` row (0.0 or 1.0)."""
    return _follow(response, row, strict=False)


def _fraction(flags: Sequence[bool]) -> float:
    """Fraction of `True` in `flags`, or 0.0 when empty."""
    return sum(bool(flag) for flag in flags) / len(flags) if flags else 0.0


@metric
def instruction_metrics() -> Metric:
    """Mean strict/loose accuracy over samples, keyed by the four accuracy names.

    Prompt-level accuracies are the plain mean of the per-sample boolean, and instruction-level
    accuracies are the micro-average over instructions,
    `sum(fraction * num_instructions) / sum(num_instructions)` (the instruction-level accuracy
    IFEval reports). With one instruction per prompt the two coincide.

    Returns:
        The metric callable, producing `prompt_level_strict`, `inst_level_strict`,
        `prompt_level_loose`, and `inst_level_loose`.
    """
    prompt_keys = ("prompt_level_strict", "prompt_level_loose")
    inst_keys = ("inst_level_strict", "inst_level_loose")

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        values = [sample_score.score.value for sample_score in scores]
        result: dict[str, float] = {}
        for key in prompt_keys:
            result[key] = float(sum(bool(value[key]) for value in values) / len(values)) if values else 0.0
        for key in inst_keys:
            total = sum(float(value["num_instructions"]) for value in values)
            weighted = sum(float(value[key]) * float(value["num_instructions"]) for value in values)
            result[key] = float(weighted / total) if total else 0.0
        return result

    return compute


@scorer(metrics=[instruction_metrics()])
def instruction_checker() -> Scorer:
    """Strict and loose IFEval instruction checking, with scalar per-sample values.

    Runs the IFEval checker library (`instruction_following_eval`, the fork
    `inspect_evals[ifeval]` installs) over the sample's completion. The checker exposes
    `InputExample(key, instruction_id_list, prompt, kwargs)` with `kwargs` a `list[dict]`, and
    `test_instruction_following(example, response, strict=...)` returning an object with
    `follow_all_instructions` and `follow_instruction_list`. Every per-sample value is a scalar:
    the prompt-level keys are booleans and the instruction-level keys are the fraction of
    instructions followed, so a per-sample frame can read each key directly. With one instruction
    per prompt the prompt-level and instruction-level values coincide; both are kept so the task
    stays valid for multi-instruction selections.

    Returns:
        The scorer, whose score value is a dict with keys `prompt_level_strict`,
        `inst_level_strict`, `prompt_level_loose`, `inst_level_loose`, and `num_instructions`.

    Raises:
        ModuleNotFoundError: If the IFEval checker library is not installed.
    """
    try:
        from instruction_following_eval.evaluation import InputExample, ensure_nltk_resource, test_instruction_following
    except ImportError as error:
        raise ModuleNotFoundError(
            "The IFEval checker library is required for instruction_checker(); install it via: "
            'pip install "inspect_evals[ifeval]"'
        ) from error

    ensure_nltk_resource()

    async def score(state: TaskState, target: Target) -> Score:
        example = InputExample(
            key=state.sample_id,
            instruction_id_list=state.metadata["instruction_id_list"],
            prompt=state.metadata["prompt"],
            kwargs=state.metadata["kwargs"],
        )
        strict = test_instruction_following(example, state.output.completion, strict=True)
        loose = test_instruction_following(example, state.output.completion, strict=False)
        return Score(
            value={
                "prompt_level_strict": bool(strict.follow_all_instructions),
                "inst_level_strict": _fraction(strict.follow_instruction_list),
                "prompt_level_loose": bool(loose.follow_all_instructions),
                "inst_level_loose": _fraction(loose.follow_instruction_list),
                "num_instructions": len(strict.follow_instruction_list),
            },
            answer=state.output.completion,
        )

    return score


_REWARD_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


def _device() -> str:
    """`"cuda"` when available, else `"mps"`, else `"cpu"`."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_reward_model(model_id: str, device: str):
    """Tokenizer and model for `model_id` on `device`, cached per process.

    Loads in float32 (DeBERTa-v3 misbehaves in bfloat16). Inspect instantiates the scorer once
    per `eval_set` call, so the cache keeps one model resident across the run's many cells.

    Args:
        model_id: Hugging Face model id of the sequence-classification reward model.
        device: Device string to place the model on.

    Returns:
        The `(tokenizer, model)` pair.
    """
    cache_key = (model_id, device)
    if cache_key not in _REWARD_MODELS:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32)
        model.to(device)
        model.eval()
        _REWARD_MODELS[cache_key] = (tokenizer, model)
    return _REWARD_MODELS[cache_key]


@scorer(metrics=[mean(), stderr()])
def reward_score(model_id: str, max_length: int = 1024) -> Scorer:
    """Reward-model score of each response, as the raw scalar logit.

    Scores the `(prompt, response)` pair with the OpenAssistant DeBERTa-v3 reward model
    (`AutoModelForSequenceClassification`, single-logit head). The question is the raw prompt
    before chat templating (`state.input_text`) and the response is the completion. One forward
    pass per sample; the model is cached per process and shares the GPU with the pipeline.

    Args:
        model_id: Hugging Face model id of the reward model.
        max_length: Tokenizer truncation length for the concatenated pair.

    Returns:
        The scorer, whose score value is the scalar reward.
    """
    async def score(state: TaskState, target: Target) -> Score:
        import torch

        tokenizer, model = _load_reward_model(model_id, _device())
        inputs = tokenizer(
            state.input_text,
            state.output.completion,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_length),
        ).to(model.device)
        with torch.inference_mode():
            value = model(**inputs).logits[0, 0].item()
        return Score(value=float(value), answer=state.output.completion)

    return score


@task
def instruction_following(
    instruction_types: Sequence[str] = DEFAULT_INSTRUCTION_TYPES,
    per_type: int = 12,
    sample_seed: int = 123,
    reward_model: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
    reward_max_length: int = 1024,
) -> Task:
    """Single-instruction Split-IFEval prompts with strict instruction checking and a reward score.

    Selects a balanced set of single-instruction prompts (`per_type` per instruction type),
    delivers each prompt's instruction lines to any consuming control (e.g. PASTA) through the
    per-sample runtime kwargs, and scores every response with the strict/loose IFEval checker and
    the reward model.

    Args:
        instruction_types: Instruction ids to draw single-instruction prompts from.
        per_type: Maximum number of prompts per instruction type.
        sample_seed: Base seed for the deterministic per-type selection.
        reward_model: Hugging Face model id of the reward model.
        reward_max_length: Tokenizer truncation length for the reward model.

    Returns:
        The task, scored by `instruction_checker()` and `reward_score()`, with the per-sample
        runtime kwargs delivered by `runtime_kwargs_solver()`.
    """
    records = select_records(load_records(), list(instruction_types), int(per_type), int(sample_seed))
    return Task(
        dataset=MemoryDataset([to_sample(record) for record in records], name="split_ifeval"),
        solver=[runtime_kwargs_solver()],
        scorer=[instruction_checker(), reward_score(reward_model, max_length=int(reward_max_length))],
    )
