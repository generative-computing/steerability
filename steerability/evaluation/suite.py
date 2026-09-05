"""Inspect task sets runnable against one steered pipeline."""
from steerability.utils.optional import require

require("inspect_ai")
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlparse
from urllib.request import url2pathname

from inspect_ai import eval_set

from steerability.evaluation.provider import ProviderOptions, as_inspect_model

if TYPE_CHECKING:
    from inspect_ai.model import Model

    from steerability.algorithms.core.steering_pipeline import SteeringPipeline


@dataclass(frozen=True, slots=True)
class InspectSuite:
    """A named set of Inspect tasks evaluated together against one pipeline.

    Every arm and every trial evaluated with the same suite scores the identical sample set per
    task: explicit `sample_ids` pass through, and `limit` otherwise selects the first N samples in
    the task's native dataset order. First-N is deterministic across arms, which is what paired
    comparison needs; it is a biased estimate of the full-benchmark score.

    Tasks with model-graded scorers need a grader model supplied through the task's own arguments
    (`task_args`). The grader must be an explicit, separate model (an API model or a second local
    model), never the pipeline under evaluation, since self-grading is circular and grader traffic
    would compete with evaluation traffic inside the provider's collator.

    Attributes:
        name: Namespace key in results (e.g. `"capability"`, `"target"`).
        tasks: Task references (e.g. `("inspect_evals/gsm8k", "my_pkg/target_qa")`).
        limit: Per-task sample cap (the first N in dataset order), or None for the full dataset.
        sample_ids: Explicit per-task sample selection, keyed by task name; a task with an entry
            ignores `limit`. Keys must be a subset of `tasks`.
        task_args: Task arguments forwarded to every task (e.g. a grader model reference).
        generate_overrides: `GenerateConfig` overrides for this suite, applied over the runner's
            `generate_defaults`.
        retry_attempts: `eval_set` task retry budget.
    """

    name: str
    tasks: tuple[str, ...]
    limit: int | None = None
    sample_ids: Mapping[str, tuple] | None = None
    task_args: Mapping[str, Any] = field(default_factory=dict)
    generate_overrides: Mapping[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")
        if not self.tasks:
            raise ValueError("tasks must be non-empty.")
        if self.limit is not None and int(self.limit) < 1:
            raise ValueError(f"limit must be >= 1; got {self.limit}.")
        if self.sample_ids is not None:
            unknown = sorted(set(self.sample_ids) - set(self.tasks))
            if unknown:
                raise ValueError(f"sample_ids names task(s) not in this suite: {unknown}.")

    def run(
        self,
        pipeline: "SteeringPipeline",
        *,
        log_dir: str | Path,
        options: ProviderOptions | None = None,
        base_seed: int | None = None,
        model_name: str = "steering-pipeline",
        generate_defaults: Mapping[str, Any] | None = None,
        display: str = "none",
        score: bool = True,
        model_roles: Mapping[str, "str | Model"] | None = None,
    ) -> dict:
        """Run every task in the suite against `pipeline` through `eval_set`.

        One provider is built per call and discarded when it returns; `eval_set` retries reuse
        the same model object within the call. `log_dir` must be dedicated to this suite run;
        `eval_set` resumes completed samples from its logs, so re-running with the same `log_dir`
        completes only the missing work. With `sample_ids` set, each task runs in its own
        `eval_set` call under a per-task subdirectory of `log_dir`.

        Args:
            pipeline: The steered pipeline to evaluate.
            log_dir: Directory for the `.eval` logs, dedicated to this suite run.
            options: Provider options forwarded to `as_inspect_model`.
            base_seed: Seed forwarded to `as_inspect_model` (attached to sampling dispatches whose
                config carries no seed).
            model_name: Bare model name; renders as `steerability/<model_name>` in logs.
            generate_defaults: `GenerateConfig` defaults applied under this suite's
                `generate_overrides`.
            display: Inspect's `display` option (`"full"`, `"conversation"`, `"rich"`, `"plain"`,
                `"log"`, `"none"`), forwarded unvalidated so `eval_set` reports an unknown value.
                `"plain"` is recommended inside a sweep, since one `eval_set` call runs per cell
                and the full display would redraw per cell.
            score: Whether `eval_set` scores the logs. The default `True` scores as usual;
                `False` produces logs carrying samples and outputs but no scores, for generation
                that is scored afterwards from the logs. `SteeringEval` frames assume scored logs.
            model_roles: Inspect model roles forwarded to `eval_set`, mapping a role name to a
                model reference or a live `Model`. A registry task resolving a grader through
                `get_model(role="grader")` is given one here. As with `task_args`, the grader must
                never be the pipeline under evaluation.

        Returns:
            A mapping from task name to a result dict with keys:

                - `"metrics"`: Mapping from `"<scorer>/<metric>"` to the metric value
                    (`stderr` is an ordinary metric and is keyed like any other).
                - `"n"`: Number of completed samples.
                - `"log"`: Path to the task's `.eval` log, relative to `log_dir` where possible.

        Raises:
            RuntimeError: If `eval_set` reports failure after its retries; the message names the
                failed tasks when the logs identify them and `unknown` otherwise. No partial
                results are returned.
        """
        log_dir = Path(log_dir)
        model = as_inspect_model(
            pipeline, options=options, base_seed=base_seed, model_name=model_name,
        )
        generate_config = {**(generate_defaults or {}), **dict(self.generate_overrides)}
        common: dict[str, Any] = dict(
            model=model,
            task_args=dict(self.task_args),
            epochs=1,
            max_tasks=1,
            display=display,
            retry_attempts=self.retry_attempts,
            score=score,
            **generate_config,
        )
        if model_roles is not None:
            common["model_roles"] = dict(model_roles)

        logs: list = []
        failed: list[str] = []
        succeeded = True
        if self.sample_ids is None:
            success, task_logs = eval_set(
                list(self.tasks), log_dir=str(log_dir), limit=self.limit, **common,
            )
            logs.extend(task_logs)
            if not success:
                succeeded = False
                failed.extend(sorted({log.eval.task for log in task_logs if log.status != "success"}))
        else:
            # per-task sample_ids cannot be expressed in one eval_set call; each task gets its own
            # dedicated log subdirectory
            for task in self.tasks:
                task_log_dir = log_dir / task.replace("/", "_")
                selection: dict[str, Any] = {}
                if task in self.sample_ids:
                    selection["sample_id"] = list(self.sample_ids[task])
                else:
                    selection["limit"] = self.limit
                success, task_logs = eval_set(
                    [task], log_dir=str(task_log_dir), **selection, **common,
                )
                logs.extend(task_logs)
                if not success:
                    succeeded = False
                    failed.extend(sorted({log.eval.task for log in task_logs if log.status != "success"}))
        if not succeeded:
            raise RuntimeError(
                f"Inspect eval_set failed for task(s): {', '.join(failed) or 'unknown'} "
                f"(logs under {log_dir})."
            )
        return _flatten_logs(logs, log_dir)


def _flatten_logs(logs: list, log_dir: Path) -> dict:
    """Reduce eval logs to plain JSON keyed by task, with metrics keyed as `<scorer>/<metric>`."""
    results: dict[str, dict] = {}
    for log in logs:
        metrics: dict[str, Any] = {}
        scores = log.results.scores if log.results is not None else []
        for score in scores:
            for metric_name, metric in score.metrics.items():
                metrics[f"{score.name}/{metric_name}"] = metric.value
        location = _log_location(str(log.location), log_dir)
        results[log.eval.task] = {
            "metrics": metrics,
            "n": log.results.completed_samples if log.results is not None else 0,
            "log": location,
        }
    return results


def _log_location(location: str, log_dir: Path) -> str:
    """Normalize an Inspect log location for storage in the results record.

    Inspect reports a local log location as a `file:` URI. A local path is stripped of the scheme
    and returned relative to `log_dir` when possible, so the record stays portable if the run
    directory moves. A remote location (e.g. an `s3://` URI) is returned unchanged.

    Args:
        location: The `location` attribute of an Inspect `EvalLog` (a `file:` URI, a plain local
            path, or a remote URI).
        log_dir: The directory the task's logs were written under.

    Returns:
        A path relative to `log_dir` for a local log, otherwise the location unchanged.
    """
    parsed = urlparse(location)
    if parsed.scheme not in ("", "file"):
        return location
    path = url2pathname(parsed.path) if parsed.scheme == "file" else location
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, log_dir)
        except ValueError:
            return path
    return path
