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
    "prompt",
    "instructions",
    "instruction_id_list",
    "kwargs",
]


class InstructionFollowing(UseCase):
    """Instruction following evaluation use case using the IFEval dataset.

    Evaluates model ability to follow specific instructions by testing adherence to various formatting, content, and
    structural constraints. Uses the IFEval dataset which contains prompts with explicit instructions that models must
    follow precisely.

    The evaluation focuses on whether models can follow instructions like formatting requirements (e.g., "respond in
    exactly 3 sentences"), content constraints (e.g., "include the word 'fantastic' twice"), and structural
    requirements (e.g., "use bullet points", "write in JSON format").

    Attributes:
        evaluation_data: List of instances containing prompts and instruction metadata.
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]) -> None:
        """Validates that evaluation data contains required fields for instruction following evaluation.

        Ensures each data instance has the necessary keys for the evaluation.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with prompt, instructions, and metadata.

        Raises:
            ValueError: If required keys ('prompt', 'instructions', 'instruction_id_list', 'kwargs') are missing.
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
        **kwargs
    ) -> list[dict[str, Any]]:
        """Generates model responses for instruction following prompts.

        Processes evaluation data to create chat-formatted prompts and generates model responses.

        The constructed chat ``"prompt"`` key on each prompt row shadows the instance's raw ``"prompt"`` string
        column, so a ``runtime_overrides`` column must be named distinctly from ``"prompt"`` to reach the raw text.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline instance to use for generation.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters passed to the model's generate method.
            runtime_overrides: Optional runtime parameter overrides for steering controls, keyed by control class name
                as ``{control_class_name: {variable: column_name}}``; each column resolves against the prompt rows.
            batch_size: Generation batch size.

        Returns:
            List of generation dictionaries, each containing:

                - "response": Generated text response from the model
                - "prompt": Original instruction following prompt
                - "instructions": List of specific instructions the model should follow
                - "instruction_id_list": Identifiers for each instruction type
                - "kwargs": Additional metadata for instruction evaluation
                - "thinking": Reasoning segment split from the continuation, or None if no think tag
                    is present. This constructed key shadows any same-named instance column.
        """
        if not self.evaluation_data:
            logger.warning("No evaluation data provided")
            return []
        gen_kwargs = dict(gen_kwargs or {})

        # form prompt data; the constructed chat "prompt" shadows the instance's raw "prompt" column
        prompt_data = []
        for instance in self.evaluation_data:
            prompt_data.append({**instance, "prompt": [{"role": "user", "content": instance["prompt"]}]})

        responses, _, outputs, thinking = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            return_outputs=True,
            return_thinking=True,
            batch_size=batch_size
        )

        log_truncation_count(outputs)

        generations = [
            {
                "response": response,
                "prompt": eval_data["prompt"],
                "instructions": eval_data["instructions"],
                "instruction_id_list": eval_data["instruction_id_list"],
                "kwargs": eval_data["kwargs"],
                "thinking": think,
                **output_record_fields(output, tokenizer),
            }
            for eval_data, response, output, think in zip(self.evaluation_data, responses, outputs, thinking)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against instruction requirements using configured metrics.

        Passes generation dictionaries to all evaluation metrics specified during initialization.

        Args:
            generations: List of generation dictionaries returned by the `generate()` method, each containing
                response, prompt, instructions, instruction_id_list, and kwargs fields.

        Returns:
            Dictionary of scores keyed by `metric_name`.
        """
        results = {}
        for metric in self.evaluation_metrics:
            results[metric.name] = metric(responses=generations)
        return results

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports instruction following evaluation results to structured JSON files.

        Creates two output files:

            1. `responses.json`: Contains model responses for each steering method
            2. `scores.json`: Contains strict metric scores for each steering method

        Args:
            profiles: Dictionary containing evaluation results from all tested pipelines.
            save_dir: Directory path where results should be saved.
        """
        folder_path = Path(save_dir)
        folder_path.mkdir(parents=True, exist_ok=True)
        steering_methods, predictions, follow_instructions, finish_reasons, adapted_prompts = [], {}, {}, {}, {}
        inputs = None

        for steering_method, runs in profiles.items():
            # profiles maps pipeline names to a list of run dicts (one per trial); use the first trial for the
            # per-question response export
            first_run = runs[0] if isinstance(runs, list) else runs
            generations = first_run["generations"]
            steering_methods.append(steering_method)
            predictions[steering_method] = [gen["response"] for gen in generations]
            finish_reasons[steering_method] = [gen.get("finish_reason") for gen in generations]
            adapted_prompts[steering_method] = [gen.get("adapted_prompt") for gen in generations]

            # get instruction following details from the StrictInstruction metric
            evaluations = first_run.get("evaluations", {})
            if "StrictInstruction" in evaluations:
                follow_instructions[steering_method] = evaluations[
                    "StrictInstruction"
                ].get("follow_all_instructions", [])
            if not inputs:
                inputs = [gen["prompt"] for gen in generations]

        responses = []
        for idx, prompt in enumerate(inputs):
            response = {"prompt": prompt}
            for method in steering_methods:
                response[method] = predictions[method][idx]
                response[f"{method}_instr_follow"] = follow_instructions[method][idx]
                response[f"{method}_finish_reason"] = finish_reasons[method][idx]
                adapted_prompt = adapted_prompts[method][idx]
                if adapted_prompt is not None:
                    response[f"{method}_adapted_prompt"] = adapted_prompt
            responses.append(response)

        with open(folder_path / "responses.json", "w") as f:
            json.dump(responses, f, indent=4)

        # build a scores-only view (everything except the bulky per-example generations)
        scores_only: dict[str, Any] = {}
        for steering_method, runs in profiles.items():
            run_list = runs if isinstance(runs, list) else [runs]
            scores_only[steering_method] = [
                {k: v for k, v in run.items() if k != "generations"}
                for run in run_list
            ]

        with open(folder_path / "scores.json", "w") as f:
            json.dump(scores_only, f, indent=4)
