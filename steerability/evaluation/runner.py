"""Sweep runner: configurations x trials x suites over one base model.

`SteeringEval` owns what Inspect cannot: pipeline lifecycle and GPU discipline. Inspect treats
models as cheap concurrent handles to endpoints; a steered pipeline is a GPU-resident object that
must be built, steered, evaluated, and released sequentially, so the runner evaluates one pipeline
at a time and never passes more than one pipeline-backed model to one `eval` or `eval_set` call.

There is no results checkpoint: the `.eval` logs under `save_dir/inspect_logs/` are the store, and
`eval_set` resumes each (configuration, trial, suite) cell from them at sample granularity. `eval_set`
matches task identity only (task, task args, model name); the runner's seed, generate defaults, provider
options, fit, and backend are not part of it, so a changed protocol needs a new `save_dir` rather than a
re-run into the old one. Each result row's `provenance` entry records what actually ran.

Three frames reshape a completed run: `results()` gives the tidy one-row-per-metric frame,
`runs_frame` pivots it to one row per (pipeline, trial) with one column per metric and per swept
argument, and `samples_frame` reads the `.eval` logs to give one row per (pipeline, trial, sample)
with per-sample scores joined to sample metadata. `summarize_runs` aggregates the per-trial frame
into the summary form the plotting layer consumes.
"""
import datetime
import importlib.metadata
import logging
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import pandas
from tqdm.auto import tqdm

import steerability
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.identity import derive_trial_seed
from steerability.algorithms.core.sweeps import PipelineFactory, expand_configurations, preflight
from steerability.algorithms.core.utils.controls import runtime_kwargs_schema
from steerability.utils.rendering import has_chat_template

if TYPE_CHECKING:
    from steerability.evaluation.provider import ProviderOptions
    from steerability.evaluation.suite import InspectSuite

logger = logging.getLogger(__name__)

_RESULTS_COLUMNS = (
    "config", "config_id", "trial", "seed", "suite", "task", "scorer", "metric", "value", "n", "log",
)


def _package_version(name: str) -> str | None:
    """Installed version of a package, or None when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class SteeringEval:
    """Evaluate steering pipeline configurations on Inspect suites, over trials, sequentially.

    Configurations expand from `pipelines` (fixed controls, `ControlSpec` sweeps, and the empty
    baseline arm); a pre-flight support check runs over every configuration before any model or
    engine work. Each configuration is built and steered once, then every trial runs every suite
    against it before the pipeline is released; each suite run builds and discards its own
    provider, named by the configuration's `config_id`. Repetition is trial-based: with `seed`
    set, each (configuration, trial) derives one seed, attached to sampling dispatches whose
    config carries no seed of its own. Inspect epochs are not used.

    Attributes:
        pipelines: Mapping from pipeline name to `[]` (the unsteered baseline arm),
            `[Control, ...]`, or `[ControlSpec, ...]`.
        base_model_name_or_path: Hugging Face model ID or local path of the base model.
        suites: The `InspectSuite`s every configuration and trial runs.
        backend: Backend forwarded to the pipelines (a `BackendSpec` or a known kind name).
        fit: Fit venue policy forwarded to the pipelines.
        hf_model_kwargs: Load-time kwargs for in-process model loads.
        device_map: Device placement for in-process model loads.
        trust_remote_code: Trust remote code when loading tokenizers.
        num_trials: Trials per configuration; a completion target, not part of run identity.
        seed: Base seed deriving one seed per (configuration, trial), or None.
        provider_options: `ProviderOptions` forwarded to every suite run (static runtime kwargs,
            batching ceiling, reasoning split).
        generate_defaults: `GenerateConfig` defaults applied under each suite's overrides.
        on_unsupported: `"raise"` (default) fails the run with one aggregate error on any
            unsupported configuration; `"skip"` runs the supported ones with a warning.
        save_dir: Directory holding the `.eval` logs; when None, logs go to a
            fresh temporary directory and the run cannot be resumed.
        progress: Draw a `tqdm` bar over the (configuration, trial, suite) cells. The same
            information is logged at INFO regardless, so script users see it without the bar.
        display: Inspect's per-sample `display` mode, forwarded to every suite run (`"none"` by
            default, `"plain"` recommended in a sweep). Presentation only; not part of run
            identity.
    """

    def __init__(
        self,
        pipelines: dict[str, list],
        base_model_name_or_path: str | Path,
        suites: "Sequence[InspectSuite]",
        *,
        backend: BackendSpec | str | None = None,
        fit: Literal["auto", "in_process"] = "auto",
        hf_model_kwargs: dict | None = None,
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
        num_trials: int = 1,
        seed: int | None = None,
        provider_options: "ProviderOptions | None" = None,
        generate_defaults: Mapping[str, Any] | None = None,
        on_unsupported: Literal["raise", "skip"] = "raise",
        save_dir: str | Path | None = None,
        progress: bool = True,
        display: str = "none",
    ) -> None:
        if not isinstance(pipelines, dict):
            raise TypeError(f"pipelines must be a dict; got {type(pipelines).__name__}.")
        if int(num_trials) < 1:
            raise ValueError(f"num_trials must be >= 1; got {num_trials}.")
        if on_unsupported not in ("raise", "skip"):
            raise ValueError(f"on_unsupported must be 'raise' or 'skip'; got {on_unsupported!r}.")
        suites = list(suites)
        if not suites:
            raise ValueError("suites must be non-empty.")
        suite_names = [suite.name for suite in suites]
        duplicates = sorted({name for name in suite_names if suite_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Suite names must be distinct; duplicated: {duplicates}.")

        self.pipelines = pipelines
        self.base_model_name_or_path = base_model_name_or_path
        self.suites = suites
        self.backend = backend
        self.fit = fit
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.num_trials = int(num_trials)
        self.seed = seed
        self.provider_options = provider_options
        self.generate_defaults = dict(generate_defaults) if generate_defaults is not None else None
        self.on_unsupported = on_unsupported
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.progress = bool(progress)
        self.display = display
        self._results: dict[str, list[dict]] | None = None
        self._log_root: Path | None = None

    def run(self) -> dict[str, list[dict]]:
        """Run every configuration x trial x suite cell, sequentially, resuming from the logs.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has
            keys:

                - `"trial_id"`: Integer trial index.
                - `"seed"`: The trial's derived seed, or None when no base seed was set.
                - `"config_id"`: The configuration's canonical identifier.
                - `"params"`: Mapping from spec name to resolved constructor kwargs, or an empty
                    dict for fixed and baseline configurations.
                - `"suites"`: Mapping from suite name to that suite's flattened results, with log
                    paths relative to `save_dir`.
                - `"provenance"`: Versions, backend kind, `prompt_path`, the shared-base
                    fingerprint when one exists, and a timestamp.

        Raises:
            RuntimeError: If any configuration is unsupported and `on_unsupported="raise"` (one
                aggregate error before any model or engine work), or a suite's `eval_set` fails
                after its retries.

        Warns:
            UserWarning: If a static runtime kwarg in `provider_options` is declared by no
                configuration in the evaluation, or `seed` is set while no `temperature` is
                configured in `generate_defaults` or any suite's `generate_overrides` (trial
                seeds are attached to sampling dispatches only).
        """
        points = list(expand_configurations(
            self.pipelines, base_model_name_or_path=self.base_model_name_or_path,
        ))

        failures: list[str] = []
        skipped: set[tuple[str, str]] = set()
        for point in points:
            messages = preflight(
                [point], base_model_name_or_path=self.base_model_name_or_path,
                backend=self.backend, fit=self.fit,
            )
            if messages:
                failures.extend(messages)
                skipped.add((point.pipeline_name, point.config_id))
        if failures:
            if self.on_unsupported == "raise":
                raise RuntimeError(
                    "Unsupported pipeline configuration(s):\n" + "\n".join(failures)
                )
            for line in failures:
                logger.warning("Skipping unsupported configuration: %s", line)

        if self.seed is not None:
            configured = set(self.generate_defaults or {})
            for suite in self.suites:
                configured.update(suite.generate_overrides)
            if "temperature" not in configured:
                warnings.warn(
                    "seed is set but no temperature is configured in generate_defaults or any suite's "
                    "generate_overrides; trial seeds are attached to sampling dispatches only, so the derived "
                    "seeds will not be attached. Pass generate_defaults={'temperature': 0} for greedy decoding "
                    "or an explicit sampling temperature.",
                    UserWarning,
                )

        if self.save_dir is not None:
            save_dir = self.save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(tempfile.mkdtemp(prefix="steering-eval-"))
            logger.info("No save_dir was given; logs go to %s and the run cannot be resumed.", save_dir)
        self._log_root = save_dir

        versions = {
            "toolkit_version": getattr(steerability, "__version__", "unknown"),
            "inspect_ai_version": _package_version("inspect-ai"),
            "inspect_evals_version": _package_version("inspect-evals"),
        }
        factory = PipelineFactory(
            self.base_model_name_or_path,
            backend=self.backend,
            fit=self.fit,
            hf_model_kwargs=self.hf_model_kwargs,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        results: dict[str, list[dict]] = {name: [] for name in self.pipelines}
        active = [point for point in points if (point.pipeline_name, point.config_id) not in skipped]

        static_keys = set(self.provider_options.runtime_kwargs) if self.provider_options is not None else set()
        if static_keys:
            declared: set[str] = set()
            for point in active:
                declared.update(runtime_kwargs_schema(point.controls_factory()))
            undeclared = sorted(static_keys - declared)
            if undeclared:
                warnings.warn(
                    f"Static runtime kwarg(s) {undeclared} are declared by no configuration in this evaluation and "
                    "will be inert on every arm.",
                    UserWarning,
                )

        total_cells = len(active) * self.num_trials * len(self.suites)
        logger.info(
            "Evaluating %d configuration(s) x %d trial(s) x %d suite(s) = %d cell(s); logs under %s.",
            len(active), self.num_trials, len(self.suites), total_cells, save_dir,
        )
        bar = tqdm(
            total=total_cells, disable=not self.progress, desc="steering eval", unit="cell",
            dynamic_ncols=True,
        )
        cell_index = 0
        try:
            for point in active:
                label = f"{point.pipeline_name}/{point.config_id}"
                bar.set_postfix_str(f"steering {label}")
                steer_started = time.monotonic()
                with factory.steered(point.controls_factory()) as pipeline:
                    logger.info("Steered %s in %.0fs.", label, time.monotonic() - steer_started)
                    prompt_path = "messages" if has_chat_template(pipeline.tokenizer) else "text"
                    for trial_id in range(self.num_trials):
                        trial_seed = (
                            derive_trial_seed(self.seed, point.config_id, trial_id)
                            if self.seed is not None else None
                        )
                        suite_results: dict[str, dict] = {}
                        for suite in self.suites:
                            cell_index += 1
                            bar.set_postfix_str(f"{label} trial {trial_id} {suite.name}")
                            log_dir = (
                                save_dir / "inspect_logs" / point.config_id
                                / f"trial_{trial_id}" / suite.name
                            )
                            started = time.monotonic()
                            flattened = suite.run(
                                pipeline,
                                log_dir=log_dir,
                                options=self.provider_options,
                                base_seed=trial_seed,
                                model_name=point.config_id,
                                generate_defaults=self.generate_defaults,
                                display=self.display,
                            )
                            completed = sum(int(result["n"]) for result in flattened.values())
                            logger.info(
                                "Cell %d/%d %s trial=%d suite=%s: %d sample(s) in %.0fs.",
                                cell_index, total_cells, label, trial_id, suite.name, completed,
                                time.monotonic() - started,
                            )
                            bar.update(1)
                            cell = log_dir.relative_to(save_dir)
                            for task_result in flattened.values():
                                if not Path(task_result["log"]).is_absolute():
                                    task_result["log"] = str(cell / task_result["log"])
                            suite_results[suite.name] = flattened
                        results[point.pipeline_name].append({
                            "trial_id": trial_id,
                            "seed": trial_seed,
                            "config_id": point.config_id,
                            "params": {
                                name: dict(kwargs) for name, kwargs in (point.params or {}).items()
                            },
                            "suites": suite_results,
                            "provenance": {
                                **versions,
                                "backend": factory.backend_kind,
                                "prompt_path": prompt_path,
                                "shared_base_fingerprint": factory.shared_base_fingerprint,
                                "recorded_utc": datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat(),
                            },
                        })
        finally:
            bar.close()
            factory.release()
        self._results = results
        return results

    def results(self) -> pandas.DataFrame:
        """The last `run()`'s results as one row per (config, trial, suite, task, scorer/metric).

        `runs_frame` (module-level, or the `runs_frame` method for swept-parameter columns)
        pivots this frame to one row per (pipeline, trial), and `summarize_runs` aggregates
        trials into the `{metric}_mean` / `{metric}_std` / `{metric}_sem` summary rows the
        plotting layer (`steerability.evaluation.plotting`) consumes. `samples_frame` (module-level,
        or the `samples_frame` method) reads the `.eval` logs for per-sample scores and paired
        per-sample deltas.

        Returns:
            A `pandas.DataFrame` with columns `config`, `config_id`, `trial`, `seed`, `suite`,
            `task`, `scorer`, `metric`, `value`, `n`, and `log`.

        Raises:
            RuntimeError: If `run()` has not been called.
        """
        if self._results is None:
            raise RuntimeError("results() requires a completed run(); call run() first.")
        rows: list[dict[str, Any]] = []
        for config_name, runs in self._results.items():
            for run in runs:
                for suite_name, tasks in run["suites"].items():
                    for task_name, task_result in tasks.items():
                        for key, value in task_result["metrics"].items():
                            scorer_name, _, metric_name = key.partition("/")
                            rows.append({
                                "config": config_name,
                                "config_id": run["config_id"],
                                "trial": run["trial_id"],
                                "seed": run["seed"],
                                "suite": suite_name,
                                "task": task_name,
                                "scorer": scorer_name,
                                "metric": metric_name,
                                "value": value,
                                "n": task_result["n"],
                                "log": task_result["log"],
                            })
        return pandas.DataFrame(rows, columns=list(_RESULTS_COLUMNS))

    def runs_frame(
        self,
        metrics: Mapping[str, str],
        *,
        params: Mapping[str, tuple[str, str]] | None = None,
        suite: str | None = None,
        task: str | None = None,
    ) -> pandas.DataFrame:
        """The last `run()`'s per-trial metric values, one row per (pipeline, trial).

        Pivots `results()` through the module-level `runs_frame` and, when `params` names
        swept constructor arguments as `column -> (spec name, argument name)`, attaches each
        as a column keyed on `config_id`, read from the run records' resolved parameters.
        Rows of configurations that do not sweep the argument (the baseline arm, fixed
        pipelines) receive NaN. A column whose values are all numeric is returned with a
        numeric dtype; non-numeric values (strings, lists) are kept raw.

        Args:
            metrics: Mapping from output column name to a metric key, either
                `"scorer/metric"` (e.g. `"choice/accuracy"`) or a bare metric name when
                unambiguous. Must be non-empty.
            params: Optional mapping from output column name to `(spec name, argument name)`.
            suite: Suite to select; required when the results span several.
            task: Task to select; required when the results span several.

        Returns:
            The wide frame with columns `pipeline`, `config_id`, `trial_id`, `seed`, one
            column per `metrics` entry, and one column per `params` entry.

        Raises:
            RuntimeError: If `run()` has not been called.
            ValueError: If `metrics` is empty, or the suite/task selection is empty or
                ambiguous.
            KeyError: If a metric key matches nothing, or a bare metric name is ambiguous.
        """
        # the bare name resolves to the module-level runs_frame (name lookup skips class attributes)
        frame = runs_frame(self.results(), metrics, suite=suite, task=task)
        for column, (spec_name, argument) in (params or {}).items():
            mapped = frame["config_id"].map(self._sweep_param_map(spec_name, argument))
            converted = pandas.to_numeric(mapped, errors="coerce")
            # keep raw values for non-numeric arguments; otherwise take the numeric dtype
            frame[column] = mapped if (converted.isna() & mapped.notna()).any() else converted
        return frame

    def samples_frame(
        self,
        scores: Mapping[str, str],
        *,
        metadata_keys: Sequence[str] = (),
        params: Mapping[str, tuple[str, str]] | None = None,
        suite: str | None = None,
        task: str | None = None,
        include_text: bool = False,
    ) -> pandas.DataFrame:
        """The last `run()`'s per-sample scores, one row per (pipeline, trial, sample).

        Reads the `.eval` logs written under the run's log directory (the `save_dir`, or the
        temporary directory when none was given) through the module-level `samples_frame`, and,
        when `params` names swept constructor arguments as `column -> (spec name, argument name)`,
        attaches each as a column keyed on `config_id`, exactly as `runs_frame` does. Rows of
        configurations that do not sweep the argument (the baseline arm, fixed pipelines) receive
        NaN. A column whose values are all numeric is returned with a numeric dtype; non-numeric
        values are kept raw.

        Args:
            scores: Mapping from output column name to a score key, either `"scorer"` for a
                scalar-valued score or `"scorer/key"` for one key of a dict-valued score. Must
                be non-empty.
            metadata_keys: Sample metadata entries to carry as columns.
            params: Optional mapping from output column name to `(spec name, argument name)`.
            suite: Suite to select; required when the results span several.
            task: Task to select; required when the results span several.
            include_text: Add the sample input and the model completion as `input` and
                `completion` columns.

        Returns:
            The frame with columns `pipeline`, `config_id`, `trial_id`, `seed`, `suite`, `task`,
            `sample_id`, the metadata columns, the score columns, optionally `input` and
            `completion`, and one column per `params` entry.

        Raises:
            RuntimeError: If `run()` has not been called.
            ValueError: If `scores` is empty, the selection is empty, or the results span several
                suites or tasks without a selector.
            KeyError: If a score key names a scorer or dict key absent from a sample.
        """
        if self._results is None or self._log_root is None:
            raise RuntimeError("samples_frame() requires a completed run(); call run() first.")
        frame = samples_frame(
            self._results, self._log_root, scores=scores, metadata_keys=metadata_keys,
            suite=suite, task=task, include_text=include_text,
        )
        for column, (spec_name, argument) in (params or {}).items():
            mapped = frame["config_id"].map(self._sweep_param_map(spec_name, argument))
            converted = pandas.to_numeric(mapped, errors="coerce")
            # keep raw values for non-numeric arguments; otherwise take the numeric dtype
            frame[column] = mapped if (converted.isna() & mapped.notna()).any() else converted
        return frame

    def _sweep_param_map(self, spec_name: str, argument: str) -> dict[str, Any]:
        """`config_id -> value` for one swept constructor argument, from the run records."""
        mapping: dict[str, Any] = {}
        for runs in (self._results or {}).values():
            for run in runs:
                spec_params = (run.get("params") or {}).get(spec_name)
                if spec_params is not None and argument in spec_params:
                    mapping[run["config_id"]] = spec_params[argument]
        return mapping


def runs_frame(
    results: pandas.DataFrame,
    metrics: Mapping[str, str],
    *,
    suite: str | None = None,
    task: str | None = None,
) -> pandas.DataFrame:
    """One row per (pipeline, trial) with one column per requested metric.

    Pivots the tidy `SteeringEval.results()` frame into the wide per-trial form that
    `summarize_runs` and the plotting layer (`steerability.evaluation.plotting`) consume. The
    `config` column is renamed `pipeline` and `trial` is renamed `trial_id`.

    Args:
        results: The frame returned by `SteeringEval.results()`.
        metrics: Mapping from output column name to a metric key, either `"scorer/metric"`
            (e.g. `"choice/accuracy"`) or a bare metric name when unambiguous. Must be
            non-empty.
        suite: Suite to select; required when the frame holds several.
        task: Task to select; required when the frame holds several.

    Returns:
        The wide frame with columns `pipeline`, `config_id`, `trial_id`, `seed`, and one
        column per `metrics` entry, sorted by (pipeline, config_id, trial_id).

    Raises:
        ValueError: If `metrics` is empty, the selection is empty, the frame spans several
            suites or tasks without a selector, or a metric key selects duplicate
            (config, trial) rows.
        KeyError: If a metric key matches nothing, or a bare metric name is ambiguous.
    """
    if not metrics:
        raise ValueError("metrics must name at least one output column.")
    frame = results
    if suite is not None:
        frame = frame[frame["suite"] == suite]
    if task is not None:
        frame = frame[frame["task"] == task]
    if frame.empty:
        raise ValueError("No rows match the requested suite/task selection.")
    if suite is None and frame["suite"].nunique() > 1:
        raise ValueError(f"Results span several suites {sorted(frame['suite'].unique())}; pass suite=.")
    if task is None and frame["task"].nunique() > 1:
        raise ValueError(f"Results span several tasks {sorted(frame['task'].unique())}; pass task=.")

    frame = frame.assign(_key=frame["scorer"].astype(str) + "/" + frame["metric"].astype(str))
    index_cols = ["config", "config_id", "trial", "seed"]
    wide: pandas.DataFrame | None = None
    for column, key in metrics.items():
        selected = frame[frame["_key"] == key] if "/" in key else frame[frame["metric"] == key]
        if selected.empty:
            available = sorted(frame["_key"].unique())
            raise KeyError(f"Metric {key!r} not found in results; available: {available}.")
        if "/" not in key and selected["_key"].nunique() > 1:
            raise KeyError(
                f"Metric name {key!r} is ambiguous ({sorted(selected['_key'].unique())}); "
                "use the 'scorer/metric' form."
            )
        if selected.duplicated(subset=index_cols).any():
            raise ValueError(f"Metric {key!r} has duplicate (config, trial) rows; narrow the selection.")
        series = selected.set_index(index_cols)["value"].rename(column)
        wide = series.to_frame() if wide is None else wide.join(series, how="outer")

    return (
        wide.reset_index()
        .rename(columns={"config": "pipeline", "trial": "trial_id"})
        .sort_values(["pipeline", "config_id", "trial_id"], ignore_index=True)
    )


_LETTER_GRADES = frozenset({"C", "I", "P", "N"})

_SAMPLES_COLUMNS = ("pipeline", "config_id", "trial_id", "seed", "suite", "task", "sample_id")


def _read_eval_log(path: str | Path):
    """Read one `.eval` log, resolving samples. Imported lazily so `runner` stays Inspect-free.

    Args:
        path: Filesystem path to the `.eval` log.

    Returns:
        The `EvalLog`, with `samples` populated.
    """
    from steerability.utils.optional import require

    require("inspect_ai")
    from inspect_ai.log import read_eval_log

    return read_eval_log(str(path))


def _score_to_column(value: Any, converter) -> Any:
    """Convert a scalar/letter score value through `converter`, keeping any other value raw.

    Numbers, booleans, and the `C`/`I`/`P`/`N` letter grades are scalar and pass through the
    `value_to_float` converter; dicts, lists, other strings, and None are returned unchanged.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return converter(value)
    if isinstance(value, str) and value in _LETTER_GRADES:
        return converter(value)
    return value


def samples_frame(
    results: Mapping[str, list[dict]],
    log_root: str | Path,
    *,
    scores: Mapping[str, str],
    metadata_keys: Sequence[str] = (),
    suite: str | None = None,
    task: str | None = None,
    include_text: bool = False,
) -> pandas.DataFrame:
    """One row per (pipeline, trial, sample) with per-sample scores read from the eval logs.

    Reads each run record's task logs under `log_root` (the runner's `save_dir`) and flattens
    their samples. `scores` maps an output column to a score key, either `"scorer"` for a
    scalar-valued score or `"scorer/key"` for one key of a dict-valued score; scalar values
    (numbers, booleans, and the `C`/`I`/`P`/`N` letter grades) are converted through
    `inspect_ai.scorer.value_to_float()`, and any other value is kept raw. `metadata_keys` names
    sample metadata entries to carry as columns. `include_text` adds the sample input and the
    model completion.

    Args:
        results: The `SteeringEval.run()` mapping from pipeline name to run records.
        log_root: Directory the run wrote its logs under (the runner's `save_dir`); each record's
            log path is resolved against it.
        scores: Mapping from output column name to a score key. Must be non-empty.
        metadata_keys: Sample metadata entries to carry as columns.
        suite: Suite to select; required when the results span several.
        task: Task to select; required when the results span several.
        include_text: Add the sample input and the model completion as `input` and `completion`
            columns.

    Returns:
        A `pandas.DataFrame` with columns `pipeline`, `config_id`, `trial_id`, `seed`, `suite`,
        `task`, `sample_id`, the metadata columns, the score columns, and optionally `input` and
        `completion`, sorted by (pipeline, config_id, trial_id, sample_id).

    Raises:
        ValueError: If `scores` is empty, the selection is empty, or the results span several
            suites or tasks without a selector.
        KeyError: If a score key names a scorer or dict key absent from a sample.
    """
    if not scores:
        raise ValueError("scores must name at least one output column.")
    from inspect_ai.scorer import value_to_float

    converter = value_to_float()
    log_root = Path(log_root)

    cells: list[tuple[str, dict, str, dict]] = []
    for pipeline_name, runs in results.items():
        for run in runs:
            for suite_name, tasks in run["suites"].items():
                if suite is not None and suite_name != suite:
                    continue
                for task_name, task_result in tasks.items():
                    if task is not None and task_name != task:
                        continue
                    cells.append((pipeline_name, run, suite_name, task_name))

    suites_present = sorted({suite_name for _, _, suite_name, _ in cells})
    tasks_present = sorted({task_name for _, _, _, task_name in cells})
    if not cells:
        raise ValueError("No rows match the requested suite/task selection.")
    if suite is None and len(suites_present) > 1:
        raise ValueError(f"Results span several suites {suites_present}; pass suite=.")
    if task is None and len(tasks_present) > 1:
        raise ValueError(f"Results span several tasks {tasks_present}; pass task=.")

    rows: list[dict[str, Any]] = []
    for pipeline_name, run, suite_name, task_name in cells:
        log_path = run["suites"][suite_name][task_name]["log"]
        log_path = Path(log_path)
        if not log_path.is_absolute():
            log_path = log_root / log_path
        log = _read_eval_log(log_path)
        for sample in log.samples or []:
            sample_scores = sample.scores or {}
            metadata = sample.metadata or {}
            row: dict[str, Any] = {
                "pipeline": pipeline_name,
                "config_id": run["config_id"],
                "trial_id": run["trial_id"],
                "seed": run["seed"],
                "suite": suite_name,
                "task": task_name,
                "sample_id": sample.id,
            }
            for key in metadata_keys:
                row[key] = metadata.get(key)
            for column, score_key in scores.items():
                scorer_name, _, sub_key = score_key.partition("/")
                if scorer_name not in sample_scores:
                    raise KeyError(
                        f"Score key {score_key!r} names scorer {scorer_name!r}, absent from sample "
                        f"{sample.id!r}; present: {sorted(sample_scores)}."
                    )
                value = sample_scores[scorer_name].value
                if sub_key:
                    if not isinstance(value, Mapping) or sub_key not in value:
                        available = sorted(value) if isinstance(value, Mapping) else value
                        raise KeyError(
                            f"Score key {score_key!r} names dict key {sub_key!r}, absent from "
                            f"scorer {scorer_name!r} on sample {sample.id!r}; present: {available}."
                        )
                    value = value[sub_key]
                row[column] = _score_to_column(value, converter)
            if include_text:
                row["input"] = sample.input
                row["completion"] = sample.output.completion
            rows.append(row)

    columns = (
        list(_SAMPLES_COLUMNS)
        + list(metadata_keys)
        + list(scores)
        + (["input", "completion"] if include_text else [])
    )
    frame = pandas.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["pipeline", "config_id", "trial_id", "sample_id"], ignore_index=True,
    )


def summarize_runs(
    runs: pandas.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("pipeline", "config_id"),
    param_cols: Sequence[str] = (),
) -> pandas.DataFrame:
    """Aggregate per-trial rows into `{metric}_mean` / `{metric}_std` / `{metric}_sem` rows.

    Produces the summary-frame contract the plotting layer (`steerability.evaluation.plotting`)
    consumes; the plots read `{metric}_mean` and `{metric}_std`, and `{metric}_sem` (the
    standard error of the mean over trials) is carried for tabular reporting. Adds `n_trials`
    (the non-null count of the first metric) and carries each `param_cols` entry with its
    first value per group; `param_cols` entries absent from `runs` are ignored. A
    single-trial group's std and sem are 0.0 rather than NaN, so the plotting layer draws a
    zero-length error bar.

    Args:
        runs: The wide per-trial frame from `runs_frame`.
        metric_cols: Metric column names to aggregate. Must be non-empty.
        group_cols: Grouping columns defining one configuration.
        param_cols: Per-configuration columns (e.g. a swept argument) to carry through.

    Returns:
        One row per group with the aggregated columns.

    Raises:
        ValueError: If `metric_cols` is empty.
    """
    if not metric_cols:
        raise ValueError("metric_cols must name at least one metric column.")
    group_cols = list(group_cols)
    param_cols = [col for col in param_cols if col in runs.columns]
    aggregations: dict[str, tuple[str, str]] = {}
    for metric_col in metric_cols:
        aggregations[f"{metric_col}_mean"] = (metric_col, "mean")
        aggregations[f"{metric_col}_std"] = (metric_col, "std")
        aggregations[f"{metric_col}_sem"] = (metric_col, "sem")
    aggregations["n_trials"] = (metric_cols[0], "count")
    for col in param_cols:
        aggregations[col] = (col, "first")
    summary = runs.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    spread_cols = [f"{metric_col}_{stat}" for metric_col in metric_cols for stat in ("std", "sem")]
    summary[spread_cols] = summary[spread_cols].fillna(0.0)
    return summary
