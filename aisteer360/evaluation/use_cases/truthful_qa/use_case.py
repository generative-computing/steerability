import json
import logging
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
    "correct_answers",
    "incorrect_answers",
    "truthfulness_instruction",
]


class TruthfulQA(UseCase):
    """TruthfulQA evaluation use case.

    Evaluates a model's ability to give truthful, accurate answers on the TruthfulQA dataset
    ([https://huggingface.co/datasets/domenicrosati/TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)).

    Each evaluation instance contains a question, a set of correct (truthful) answers, and a set of incorrect (common
    misconception) answers, along with a truthfulness instruction that is prepended to each prompt. The model generates
    a free-form response.
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]) -> None:
        """Validates that a single evaluation instance contains the required fields.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with question, correct answers,
                incorrect answers, and truthfulness instruction.

        Raises:
            ValueError: If required keys are missing.
        """
        missing_keys = [key for key in _EVALUATION_REQ_KEYS if key not in evaluation_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Generates model responses for TruthfulQA questions.

        Constructs prompts by prepending the truthfulness instruction to each question, then generates model responses.

        Args:
            model_or_pipeline: Either a HuggingFace model or a ``SteeringPipeline`` instance.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters passed to the model's generate method.
            runtime_overrides: Optional runtime parameter overrides for steering controls, keyed by control class name.
                To route the truthfulness instruction to PASTA, use
                ``{"PASTA": {"substrings": "truthfulness_instruction"}}``; the column resolves against the prompt rows,
                each of which carries ``truthfulness_instruction``.
            batch_size: Generation batch size.
            **kwargs: Additional keyword arguments.

        Returns:
            List of generation dictionaries, each containing:

                - ``response``: Generated text response from the model.
                - ``question``: Original question from the dataset.
                - ``truthfulness_instruction``: The instruction text prepended to the prompt.
                - ``correct_answers``: List of reference truthful answers.
                - ``incorrect_answers``: List of common misconception answers.
                - ``best_answer``: Single best reference answer (if present in the dataset).
                - ``category``: Question category (if present in the dataset).
                - ``thinking``: Reasoning segment split from the continuation, or None if no think
                    tag is present. This constructed key shadows any same-named instance column.
        """
        if not self.evaluation_data:
            logger.warning("No evaluation data provided")
            return []

        gen_kwargs = dict(gen_kwargs or {})

        # construct prompts with truthfulness instruction; rows carry truthfulness_instruction, so
        # runtime_overrides={"PASTA": {"substrings": "truthfulness_instruction"}} resolves per row
        prompt_data = []
        for instance in self.evaluation_data:
            prompt_text = (
                f"{instance['truthfulness_instruction']}\n\n"
                f"Question: {instance['question']}"
            )
            prompt_data.append({**instance, "prompt": [{"role": "user", "content": prompt_text}]})

        responses, _, outputs, thinking = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            return_outputs=True,
            return_thinking=True,
            batch_size=batch_size,
        )

        log_truncation_count(outputs)

        generations = [
            {
                "response": response,
                "question": instance["question"],
                "truthfulness_instruction": instance["truthfulness_instruction"],
                "correct_answers": instance["correct_answers"],
                "incorrect_answers": instance["incorrect_answers"],
                "best_answer": instance.get("best_answer", ""),
                "category": instance.get("category", ""),
                "thinking": think,
                **output_record_fields(output, tokenizer),
            }
            for instance, response, output, think in zip(self.evaluation_data, responses, outputs, thinking)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against truthfulness and quality metrics.

        Passes generation dictionaries to all evaluation metrics specified during initialization. Each metric receives
        the full generation dictionaries (containing the response, correct/incorrect reference answers, and metadata)
        and returns a score dictionary.

        Args:
            generations: List of generation dictionaries returned by the ``generate()`` method.

        Returns:
            Dictionary of scores keyed by ``metric_name`` (structure of each dictionary is dictated by each metric).
        """
        results = {}
        for metric in self.evaluation_metrics:
            results[metric.name] = metric(responses=generations)
        return results

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports TruthfulQA evaluation results to structured JSON files.

        Creates two output files:

            1. ``responses.json``: Per-question model responses for each steering pipeline, with reference answers.
            2. ``scores.json``: Aggregate metric scores for each steering pipeline.

        Args:
            profiles: Dictionary containing evaluation results from all tested pipelines.
            save_dir: Directory path where results should be saved.
        """
        folder_path = Path(save_dir)
        folder_path.mkdir(parents=True, exist_ok=True)

        steering_methods = []
        predictions: dict[str, list[str]] = {}
        finish_reasons: dict[str, list[str | None]] = {}
        adapted_prompts: dict[str, list[str | None]] = {}
        questions: list[str] | None = None
        correct_answers: list[list[str]] | None = None
        incorrect_answers: list[list[str]] | None = None

        for steering_method, runs in profiles.items():
            # profiles maps pipeline names to a list of run dicts (one per trial);
            # use the first trial for the per-question response export
            first_run = runs[0] if isinstance(runs, list) else runs
            generations = first_run["generations"]
            steering_methods.append(steering_method)
            predictions[steering_method] = [gen["response"] for gen in generations]
            finish_reasons[steering_method] = [gen.get("finish_reason") for gen in generations]
            adapted_prompts[steering_method] = [gen.get("adapted_prompt") for gen in generations]

            if questions is None:
                questions = [gen["question"] for gen in generations]
                correct_answers = [gen["correct_answers"] for gen in generations]
                incorrect_answers = [gen["incorrect_answers"] for gen in generations]

        responses = []
        for idx, question in enumerate(questions):
            entry = {
                "question": question,
                "correct_answers": correct_answers[idx],
                "incorrect_answers": incorrect_answers[idx],
            }
            for method in steering_methods:
                entry[method] = predictions[method][idx]
                entry[f"{method}_finish_reason"] = finish_reasons[method][idx]
                adapted_prompt = adapted_prompts[method][idx]
                if adapted_prompt is not None:
                    entry[f"{method}_adapted_prompt"] = adapted_prompt
            responses.append(entry)

        with open(folder_path / "responses.json", "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)

        # build a scores-only view
        scores_only: dict[str, Any] = {}
        for steering_method, runs in profiles.items():
            run_list = runs if isinstance(runs, list) else [runs]
            scores_only[steering_method] = [
                {k: v for k, v in run.items() if k != "generations"}
                for run in run_list
            ]

        with open(folder_path / "scores.json", "w", encoding="utf-8") as f:
            json.dump(scores_only, f, indent=4, ensure_ascii=False)
