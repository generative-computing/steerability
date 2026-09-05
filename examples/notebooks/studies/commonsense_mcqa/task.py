"""CommonsenseQA under deterministic choice shuffling, with a positional-bias metric.

Each CommonsenseQA validation question is expanded into `num_shuffling_runs` samples, one per
deterministic shuffle of its answer choices, so accuracy and positional bias are measured over
repeated presentations of the same question. The task is otherwise Inspect-native:
`multiple_choice()` formats and generates, `choice()` parses and scores, and the shipped
`accuracy()` / `stderr()` metrics are joined by the custom `positional_bias()` metric below.
"""
import random
from typing import Any, Callable

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Metric, SampleScore, accuracy, choice, metric, stderr
from inspect_ai.solver import MultipleChoiceTemplate, multiple_choice

LETTERS = "ABCDEFGH"
CSQA_PATH = "tau/commonsense_qa"


def format_example(question: str, choices: list[str]) -> str:
    """Render one question exactly as `multiple_choice()` presents it.

    Uses the single-answer template the task's solver renders, with choices as `A) text` lines
    and the letter list comma-joined, so steering data built from this function (few-shot
    exemplars, preference pairs) matches the evaluation-time prompt character for character.

    Args:
        question: The question text.
        choices: Answer options in presentation order.

    Returns:
        The fully rendered prompt.
    """
    options = "\n".join(f"{LETTERS[i]}) {text}" for i, text in enumerate(choices))
    letters = ",".join(LETTERS[: len(choices)])
    return MultipleChoiceTemplate.SINGLE_ANSWER.format(
        question=question, choices=options, letters=letters,
    )


def shuffled_samples(num_shuffling_runs: int, shuffle_seed: int) -> Callable[[dict[str, Any]], list[Sample]]:
    """A `sample_fields` mapper expanding one record into shuffled presentations.

    Each of the `num_shuffling_runs` samples bakes one deterministic shuffle of the record's
    choices into `Sample.choices`, with the target set to the correct slot's letter under that
    shuffle. Shuffles seed `random.Random` with the string `f"{shuffle_seed}:{id}:{run}"`, so
    the dataset is a pure function of the task arguments (string seeding is deterministic
    across platforms and Python versions). Records without a single-letter in-range answer key
    (present in the dataset's test split) expand to an empty list and are skipped.

    Args:
        num_shuffling_runs: Number of shuffled presentations per question.
        shuffle_seed: Base seed for the per-sample shuffles.

    Returns:
        The mapper, for `hf_dataset(sample_fields=...)`.
    """

    def to_samples(record: dict[str, Any]) -> list[Sample]:
        choices = list(record["choices"]["text"])
        answer_key = record["answerKey"]
        if len(answer_key) != 1 or answer_key not in LETTERS[: len(choices)]:
            return []
        answer_index = LETTERS.index(answer_key)
        samples: list[Sample] = []
        for run in range(num_shuffling_runs):
            rng = random.Random(f"{shuffle_seed}:{record['id']}:{run}")
            order = list(range(len(choices)))
            rng.shuffle(order)
            samples.append(Sample(
                id=f"{record['id']}:run{run}",
                input=record["question"],
                choices=[choices[i] for i in order],
                target=LETTERS[order.index(answer_index)],
                metadata={
                    "question_id": record["id"],
                    "run": run,
                    "num_choices": len(choices),
                },
            ))
        return samples

    return to_samples


@metric
def positional_bias() -> Metric:
    """Mean total variation distance between chosen-slot distributions and uniform.

    For each question, the empirical distribution of the model's chosen slot across that
    question's shuffling runs (read from the `choice` scorer's `Score.answer`) is compared to
    the uniform distribution via total variation distance, `0.5 * sum_i |p_i - 1/n|`, and the
    distances are averaged over questions. Answers that are not a single in-range letter are
    excluded; questions with no parsed answer are skipped; the metric is NaN when no question
    qualifies. The range is `[0, 1 - 1/n]`, and 0 means no positional preference.

    Note the finite-sample floor: with a finite number of runs over `n` slots, even a chooser
    with no positional preference has a strictly positive expected distance (its empirical
    histogram cannot be exactly uniform), so small values should be read against that floor
    rather than against zero.

    Returns:
        The metric callable.
    """

    def compute(scores: list[SampleScore]) -> float:
        slot_counts: dict[Any, list[int]] = {}
        for sample_score in scores:
            sample_metadata = sample_score.sample_metadata or {}
            question_id = sample_metadata.get("question_id")
            num_choices = sample_metadata.get("num_choices")
            answer = (sample_score.score.answer or "").strip()
            if question_id is None or not num_choices:
                continue
            if len(answer) != 1 or answer not in LETTERS[: int(num_choices)]:
                continue
            counts = slot_counts.setdefault(question_id, [0] * int(num_choices))
            counts[LETTERS.index(answer)] += 1

        per_question: list[float] = []
        for counts in slot_counts.values():
            total = sum(counts)
            if total == 0:
                continue
            uniform = 1.0 / len(counts)
            per_question.append(0.5 * sum(abs(count / total - uniform) for count in counts))
        if not per_question:
            return float("nan")
        return float(sum(per_question) / len(per_question))

    return compute


@task
def commonsense_mcqa(
    num_questions: int = 50,
    num_shuffling_runs: int = 20,
    shuffle_seed: int = 0,
) -> Task:
    """CommonsenseQA validation questions under deterministic choice shuffling.

    Loads the first `num_questions` validation records (the `hf_dataset` limit counts records
    before the expanding mapper runs, so it counts questions rather than samples) and expands
    each into `num_shuffling_runs` shuffled presentations.

    Args:
        num_questions: Number of validation questions to load.
        num_shuffling_runs: Shuffled presentations per question.
        shuffle_seed: Base seed for the deterministic shuffles.

    Returns:
        The task, scored by `choice()` with `accuracy`, `stderr`, and `positional_bias`.
    """
    dataset = hf_dataset(
        CSQA_PATH,
        split="validation",
        sample_fields=shuffled_samples(int(num_shuffling_runs), int(shuffle_seed)),
        limit=int(num_questions),
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
        metrics=[accuracy(), stderr(), positional_bias()],
    )
