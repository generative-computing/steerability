"""Data processing utilities for benchmark profiles."""

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Conversion to json-safe format.

    - primitives: pass through
    - Path: str(path)
    - mappings: recurse, stringify keys
    - sequences: recurse on elements
    - numpy scalars/arrays: convert to Python / list, then recurse
    - everything else: repr(obj)
    """
    from pathlib import Path as _Path

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, _Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())

    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    if callable(obj):
        return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"

    return repr(obj)


def flatten_profiles(
    profiles: dict[str, list[dict[str, Any]]],
    metric_accessors: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Flatten nested benchmark profiles into a single DataFrame with one row per run.

    Works for both fixed-control and ControlSpec-based pipelines. Each row represents
    a single trial of a single configuration. Every run dict must carry a `config_id` (as
    produced by `Benchmark.run()`); a run dict without it raises `KeyError`.

    Args:
        profiles: Output from `Benchmark.run()`. Maps pipeline names to lists of run dicts.
        metric_accessors: Optional mapping from column name to (metric_name, key) tuples
            for extracting specific metric values. For example:
            `{"accuracy": ("MCQAAccuracy", "trial_mean"), "reward": ("RewardScore", "mean_reward")}`
            If None, no metric columns are added (use `extract_metric` separately).

    Returns:
        DataFrame with columns:

            - `pipeline`: Name of the steering pipeline.
            - `trial_id`: Trial index within the configuration.
            - `config_id`: The run's recorded configuration identifier (`"baseline"` for the unsteered pipeline).
            - `params`: The full params dict (for ControlSpec runs) or empty dict.
            - `_run`: Reference to the original run dict (for downstream access).
            - Additional columns for each entry in `metric_accessors`.

    Example:
        >>> profiles = benchmark.run()
        >>> df = flatten_profiles(profiles, metric_accessors={
        ...     "accuracy": ("MCQAAccuracy", "trial_mean"),
        ... })
        >>> df.groupby("pipeline")["accuracy"].mean()
    """
    rows = []
    for pipeline_name, runs in profiles.items():
        for run in runs:
            params = run.get("params", {}) or {}
            config_id = run["config_id"]

            row = {
                "pipeline": pipeline_name,
                "trial_id": run.get("trial_id", 0),
                "config_id": config_id,
                "params": params,
                "_run": run,
            }

            # extract requested metrics
            if metric_accessors:
                evals = run.get("evaluations", {}) or {}
                for col_name, (metric_name, key) in metric_accessors.items():
                    metric_dict = evals.get(metric_name, {}) or {}
                    row[col_name] = metric_dict.get(key, np.nan)

            rows.append(row)

    return pd.DataFrame(rows)


def hash_params(params: dict[str, Any]) -> str:
    """Short stable hash of a params dict, for grouping configurations in analysis.

    Stable across processes for JSON-serializable values and for callables (serialized by
    `__qualname__`). Any other object falls back to `str(obj)` and is only as stable as that
    string; a repr containing a memory address defeats cross-process matching. Checkpoint
    identity does not use this function (see the identity design); this is the analysis-side
    grouping hash.

    Args:
        params: The params dict to hash.

    Returns:
        An 8-character hex digest.
    """
    def _default(obj: Any) -> str:
        if callable(obj):
            return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"
        return str(obj)

    serialized = json.dumps(params, sort_keys=True, default=_default)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def extract_metric(
    run: dict[str, Any],
    metric_name: str,
    key: str,
    default: Any = np.nan,
) -> Any:
    """Extract a specific metric value from a run dictionary.

    Args:
        run: A single run dictionary from benchmark profiles.
        metric_name: Name of the metric (e.g., "MCQAAccuracy", "StrictInstruction").
        key: Key within the metric's result dict (e.g., "trial_mean", "strict_prompt_accuracy").
        default: Value to return if the metric or key is not found.

    Returns:
        The metric value, or `default` if not found.

    Example:
        >>> acc = extract_metric(run, "MCQAAccuracy", "trial_mean")
    """
    evals = run.get("evaluations", {}) or {}
    metric_dict = evals.get(metric_name, {}) or {}
    return metric_dict.get(key, default)


def extract_param(
    run: dict[str, Any],
    spec_name: str,
    param_name: str,
    default: Any = None,
) -> Any:
    """Extract a specific parameter value from a run's params.

    Args:
        run: A single run dictionary from benchmark profiles.
        spec_name: Name of the ControlSpec (or control class name).
        param_name: Name of the parameter within that spec.
        default: Value to return if the spec or param is not found.

    Returns:
        The parameter value, or `default` if not found.

    Example:
        >>> alpha = extract_param(run, "PASTA", "alpha")
    """
    params = run.get("params", {}) or {}
    spec_params = params.get(spec_name, {}) or {}
    return spec_params.get(param_name, default)


def summarize_by_config(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics across trials for each configuration.

    Args:
        df: DataFrame from `flatten_profiles` with metric columns.
        metric_cols: List of column names containing metric values to aggregate.
        group_cols: Columns to group by. Defaults to `["pipeline", "config_id"]`.

    Returns:
        DataFrame with one row per configuration, containing:

            - Group columns
            - `n_trials`: Number of trials in the group
            - For each metric: `{metric}_mean` and `{metric}_std`

    Example:
        >>> df = flatten_profiles(profiles, {"acc": ("Accuracy", "mean")})
        >>> summary = summarize_by_config(df, metric_cols=["acc"])
    """
    if group_cols is None:
        group_cols = ["pipeline", "config_id"]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        result = {"n_trials": len(g)}
        for col in metric_cols:
            result[f"{col}_mean"] = g[col].mean()
            result[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        return pd.Series(result)

    return df.groupby(group_cols, sort=False).apply(agg_group, include_groups=False).reset_index()


def get_param_values(
    df: pd.DataFrame,
    spec_name: str,
    param_name: str,
) -> pd.Series:
    """Extract a parameter value as a Series from the params column.

    Useful for adding swept parameter values as columns for analysis.

    Args:
        df: DataFrame from `flatten_profiles`.
        spec_name: Name of the ControlSpec.
        param_name: Name of the parameter.

    Returns:
        Series of parameter values aligned with the DataFrame index.

    Example:
        >>> df = flatten_profiles(profiles)
        >>> df["alpha"] = get_param_values(df, "PASTA", "alpha")
    """
    return df["params"].apply(
        lambda p: (p.get(spec_name, {}) or {}).get(param_name, None)
    )


def build_per_example_df(
    run: dict[str, Any],
    generation_fields: list[str] | None = None,
    metric_lists: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a per-example DataFrame from a single run.

    Useful for analyzing individual examples across configurations.

    Args:
        run: A single run dictionary from benchmark profiles.
        generation_fields: Fields to extract from each generation dict.
            Defaults to `["prompt", "response"]`.
        metric_lists: Mapping from column name to (metric_name, list_key) for
            per-example metric values stored as lists. For example:
            `{"followed": ("StrictInstruction", "follow_all_instructions")}`

    Returns:
        DataFrame with one row per example, containing:

            - `idx`: Example index
            - Requested generation fields
            - Requested per-example metric values

    Example:
        >>> example_df = build_per_example_df(
        ...     run,
        ...     generation_fields=["prompt", "response"],
        ...     metric_lists={"followed": ("StrictInstruction", "follow_all_instructions")}
        ... )
    """
    if generation_fields is None:
        generation_fields = ["prompt", "response"]

    generations = run.get("generations", [])
    evals = run.get("evaluations", {}) or {}

    # pre-extract metric lists
    metric_data: dict[str, list] = {}
    if metric_lists:
        for col_name, (metric_name, list_key) in metric_lists.items():
            metric_dict = evals.get(metric_name, {}) or {}
            metric_data[col_name] = metric_dict.get(list_key, [None] * len(generations))

    rows = []
    for i, gen in enumerate(generations):
        row = {"idx": i}
        for field in generation_fields:
            row[field] = gen.get(field)
        for col_name, values in metric_data.items():
            row[col_name] = values[i] if i < len(values) else None
        rows.append(row)

    return pd.DataFrame(rows)

def per_example_config_means(
    profiles: dict[str, list[dict[str, Any]]],
    metric_lists: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Compute per-example score means across trials for each (pipeline, config).

    For benchmarks with multiple trials per configuration, this averages each
    example's per-trial scores to produce a stable per-example estimate. Every run dict must carry
    a ``config_id`` (as produced by ``Benchmark.run()``); a run dict without it raises ``KeyError``.

    Args:
        profiles: Output from ``Benchmark.run()``. Maps pipeline names to lists of run dicts.
        metric_lists: Mapping from column name to ``(metric_name, list_key)`` for
            per-example metric values stored as lists. For example:
            ``{"truthful": ("Truthfulness", "scores"), "informative": ("Informativeness", "scores")}``

    Returns:
        DataFrame with columns:

            - ``pipeline``: Name of the steering pipeline.
            - ``config_id``: Configuration identifier.
            - ``idx``: Example index.
            - One column per entry in ``metric_lists``, containing the trial mean.

    Example:
        >>> means = per_example_config_means(profiles, {
        ...     "truthful": ("Truthfulness", "scores"),
        ...     "informative": ("Informativeness", "scores"),
        ... })
        >>> means.groupby(["pipeline", "config_id"])["truthful"].mean()
    """
    from collections import defaultdict

    # accumulate per-trial scores for each (pipeline, config, idx)
    accum: dict[tuple[str, str], dict[int, dict[str, list]]] = {}

    for pipeline_name, runs in profiles.items():
        run_list = runs if isinstance(runs, list) else [runs]
        for run in run_list:
            config_id = run["config_id"]
            key = (pipeline_name, config_id)
            if key not in accum:
                accum[key] = defaultdict(lambda: {col: [] for col in metric_lists})

            evals = run.get("evaluations", {}) or {}
            score_lists = {}
            for col_name, (metric_name, list_key) in metric_lists.items():
                metric_dict = evals.get(metric_name, {}) or {}
                score_lists[col_name] = metric_dict.get(list_key, [])

            n_examples = max((len(v) for v in score_lists.values()), default=0)
            for idx in range(n_examples):
                for col_name, scores in score_lists.items():
                    if idx < len(scores):
                        accum[key][idx][col_name].append(scores[idx])

    # collapse to means
    rows = []
    for (pipeline_name, config_id), examples in accum.items():
        for idx, col_scores in sorted(examples.items()):
            row = {"pipeline": pipeline_name, "config_id": config_id, "idx": idx}
            for col_name, values in col_scores.items():
                row[col_name] = np.mean(values) if values else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_config(
    summary: pd.DataFrame,
    pipeline: str,
    optimize: str,
    constraint_col: str | None = None,
    constraint_min: float | None = None,
) -> pd.Series:
    """Select the best configuration for a pipeline from a summary table.

    Picks the configuration that maximizes ``optimize`` subject to an optional
    minimum-value constraint on another column. Falls back to unconstrained
    selection if no configuration satisfies the constraint.

    Args:
        summary: DataFrame from ``summarize_by_config`` with metric summary columns.
        pipeline: Pipeline name to filter on.
        optimize: Column name to maximize (e.g., ``"truthfulness_mean"``).
        constraint_col: Optional column name for the minimum-value constraint.
        constraint_min: Minimum acceptable value for ``constraint_col``.

    Returns:
        Series for the best configuration row.

    Raises:
        ValueError: If no rows match the given pipeline name.

    Example:
        >>> best = select_best_config(
        ...     summary, "pasta_deal",
        ...     optimize="truthfulness_mean",
        ...     constraint_col="informativeness_mean",
        ...     constraint_min=0.88,
        ... )
        >>> best["config_id"]
    """
    subset = summary[summary["pipeline"] == pipeline]
    if subset.empty:
        raise ValueError(f"No rows found for pipeline '{pipeline}'")

    if constraint_col is not None and constraint_min is not None:
        viable = subset[subset[constraint_col] >= constraint_min]
        if not viable.empty:
            subset = viable

    return subset.loc[subset[optimize].idxmax()]


def get_generation_field(
    profiles: dict[str, list[dict[str, Any]]],
    pipeline: str,
    config_id: str,
    idx: int,
    field: str = "response",
    trial_id: int = 0,
) -> Any:
    """Retrieve a generation field from a specific (pipeline, config, example, trial).

    Useful for displaying representative responses alongside aggregated metrics. Every run dict
    must carry a ``config_id`` (as produced by ``Benchmark.run()``); a run dict without it raises
    ``KeyError``.

    Args:
        profiles: Output from ``Benchmark.run()``.
        pipeline: Pipeline name.
        config_id: The run's recorded ``config_id`` (``"baseline"`` for the unsteered pipeline).
        idx: Example index within the generation list.
        field: Field name to extract from the generation dict. Defaults to ``"response"``.
        trial_id: Which trial to pull from when multiple trials share a config.
            Defaults to ``0`` (first trial).

    Returns:
        The requested field value.

    Raises:
        KeyError: If the pipeline is not found in profiles.
        StopIteration: If no run matches the given ``config_id`` and ``trial_id``.

    Example:
        >>> resp = get_generation_field(profiles, "pasta_deal", "a1b2c3d4", idx=5)
    """
    run_list = profiles[pipeline]
    run_list = run_list if isinstance(run_list, list) else [run_list]

    match_count = 0
    for run in run_list:
        run_config = run["config_id"]
        if run_config == config_id:
            if match_count == trial_id:
                return run["generations"][idx].get(field)
            match_count += 1

    raise StopIteration(
        f"No run found for pipeline='{pipeline}', config_id='{config_id}', trial_id={trial_id}"
    )
