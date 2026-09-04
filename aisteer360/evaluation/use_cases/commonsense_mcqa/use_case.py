import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import (
    DEFAULT_EVAL_BATCH_SIZE,
    batch_retry_generate,
    log_truncation_count,
    output_record_fields,
)

logger = logging.getLogger(__name__)

_EVALUATION_REQ_KEYS = [
    "question",
    "answer",
    "choices"
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CommonsenseMCQA(UseCase):
    """Commonsense MCQA evaluation use case.

    Evaluates model's ability to answer commonsense questions via accuracy on the CommonsenseMCQA dataset
    ([https://huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)). Supports
    answer choice shuffling across multiple runs to reduce position bias and improve evaluation robustness.

    The evaluation data should contain questions with multiple choice options where models are asked to respond with
    only the letter (A, B, C, etc.) corresponding to their chosen answer.

    Attributes:
        num_shuffling_runs: Number of times to shuffle answer choices for each question to mitigate position bias effects.
    """
    num_shuffling_runs: int

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        """Validates that evaluation data contains required fields for MCQA evaluation.

        Ensures each data instance has the necessary keys and non-null values for the evaluation.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with question, answer choices, and correct answer information.

        Raises:
            ValueError: If required keys ('id', 'question', 'answer', 'choices') are missing or if any required fields contain null/NaN values.
        """
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

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Generates model responses for multiple-choice questions with shuffled answer orders.

        Creates prompts for each question with shuffled answer choices, generates model responses, and parses the
        outputs to extract letter choices. Repeats the process multiple times with different answer orderings to reduce
        positional bias.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline instance to use for generation.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters.
            runtime_overrides: Optional runtime parameter overrides for steering controls, keyed by control class name
                as ``{control_class_name: {variable: column_name}}``; each column resolves against the prompt rows.
            batch_size: Generation batch size.
            kwargs: Optional keyword arguments. A `trial_seed` value seeds a private
                `random.Random` for choice shuffling, so a trial's answer orderings are
                reproducible; without it, shuffling uses the module-global `random`.

        Returns:
            List of generation dictionaries, each containing:

                - "response": Parsed letter choice (A, B, C, etc.) or None if not parseable
                - "prompt": Full prompt text sent to the model
                - "question_id": Identifier from the original evaluation data
                - "reference_answer": Correct letter choice for this shuffled ordering
                - "thinking": Reasoning segment split from the continuation, or None if no think tag
                    is present. This constructed key shadows any same-named instance column.

        Note:

        - The number of returned generations will be `len(evaluation_data)` * `num_shuffling_runs` due to answer choice shuffling.
        """

        if not self.evaluation_data:
            logger.warning("No evaluation data provided.")
            return []
        gen_kwargs = dict(gen_kwargs or {})

        trial_seed = kwargs.get("trial_seed")
        rng = random.Random(trial_seed) if trial_seed is not None else random

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
                rng.shuffle(choice_order)
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
        choices, _, outputs, thinking = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            parse_fn=self._parse_letter,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            return_outputs=True,
            return_thinking=True,
            batch_size=batch_size
        )

        log_truncation_count(outputs)

        # store
        generations = [
            {
                "response": choice,
                "prompt": prompt_dict["prompt"],
                "question_id": prompt_dict["id"],
                "reference_answer": prompt_dict["reference_answer"],
                "thinking": think,
                **output_record_fields(output, tokenizer),
            }
            for prompt_dict, choice, output, think in zip(prompt_data, choices, outputs, thinking)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against reference answers using configured metrics.

        Extracts responses and reference answers from generations and computes scores using all evaluation metrics
        specified during initialization.

        Args:
            generations: List of generation dictionaries returned by the `generate()` method, each containing response,
                reference_answer, and question_id fields.

        Returns:
            Dictionary of scores keyed by `metric_name`
        """
        eval_data = {
            "responses": [generation["response"] for generation in generations],
            "reference_answers": [generation["reference_answer"] for generation in generations],
            "question_ids": [generation["question_id"] for generation in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)

        return scores

    def export(self, profiles: dict[str, Any], save_dir) -> None:
        """Exports evaluation profiles to (tabbed) JSON format."""

        with open(Path(save_dir) / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _parse_letter(response) -> str:
        """Extracts the letter choice from model's generation.

        Parses model output to find the first valid letter (A-Z) that represents the model's choice.

        Args:
            response: Raw text response from the model.

        Returns:
            Single uppercase letter (A, B, C, etc.) representing the model's choice, or None if no valid letter choice could be parsed.
        """
        valid = _LETTERS
        text = re.sub(r"^\s*(assistant|system|user)[:\n ]*", "", response, flags=re.I).strip()
        match = re.search(rf"\b([{valid}])\b", text, flags=re.I)
        return match.group(1).upper() if match else None
