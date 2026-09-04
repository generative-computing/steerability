"""
Tests for evaluation utilities.

Tests cover:

- Data utilities (flatten_profiles, summarize_by_config, etc.)
- Visualization utilities (plot functions)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aisteer360.evaluation.utils.data_utils import (
    build_per_example_df,
    extract_metric,
    extract_param,
    flatten_profiles,
    get_param_values,
    summarize_by_config,
    to_jsonable,
)


# Test Fixtures
@pytest.fixture
def sample_profiles_fixed():
    """Sample profiles from a fixed-control benchmark run."""
    return {
        "baseline": [
            {
                "trial_id": 0,
                "generations": [
                    {"prompt": "Q1?", "response": "A", "reference_answer": "A"},
                    {"prompt": "Q2?", "response": "B", "reference_answer": "B"},
                ],
                "evaluations": {
                    "Accuracy": {"mean": 0.8, "std": 0.1},
                    "Reward": {"mean_reward": 1.5, "rewards": [1.2, 1.8]},
                },
                "params": {},
                "config_id": "baseline",
            },
            {
                "trial_id": 1,
                "generations": [
                    {"prompt": "Q1?", "response": "A", "reference_answer": "A"},
                    {"prompt": "Q2?", "response": "C", "reference_answer": "B"},
                ],
                "evaluations": {
                    "Accuracy": {"mean": 0.7, "std": 0.15},
                    "Reward": {"mean_reward": 1.3, "rewards": [1.1, 1.5]},
                },
                "params": {},
                "config_id": "baseline",
            },
        ],
        "steered": [
            {
                "trial_id": 0,
                "generations": [
                    {"prompt": "Q1?", "response": "A", "reference_answer": "A"},
                    {"prompt": "Q2?", "response": "B", "reference_answer": "B"},
                ],
                "evaluations": {
                    "Accuracy": {"mean": 0.9, "std": 0.05},
                    "Reward": {"mean_reward": 1.7, "rewards": [1.6, 1.8]},
                },
                "params": {},
                "config_id": "baseline",
            },
        ],
    }


@pytest.fixture
def sample_profiles_spec():
    """Sample profiles from a ControlSpec-based benchmark run."""
    return {
        "baseline": [
            {
                "trial_id": 0,
                "generations": [{"prompt": "Q1?", "response": "A"}],
                "evaluations": {"Accuracy": {"mean": 0.5}},
                "params": {},
                "config_id": "baseline",
            },
        ],
        "alpha_sweep": [
            {
                "trial_id": 0,
                "generations": [{"prompt": "Q1?", "response": "A"}],
                "evaluations": {"Accuracy": {"mean": 0.6}},
                "params": {"PASTA": {"alpha": 5.0, "layers": [8, 9]}},
                "config_id": "cfg_alpha5",
            },
            {
                "trial_id": 1,
                "generations": [{"prompt": "Q1?", "response": "B"}],
                "evaluations": {"Accuracy": {"mean": 0.65}},
                "params": {"PASTA": {"alpha": 5.0, "layers": [8, 9]}},
                "config_id": "cfg_alpha5",
            },
            {
                "trial_id": 0,
                "generations": [{"prompt": "Q1?", "response": "A"}],
                "evaluations": {"Accuracy": {"mean": 0.7}},
                "params": {"PASTA": {"alpha": 10.0, "layers": [8, 9]}},
                "config_id": "cfg_alpha10",
            },
            {
                "trial_id": 1,
                "generations": [{"prompt": "Q1?", "response": "A"}],
                "evaluations": {"Accuracy": {"mean": 0.75}},
                "params": {"PASTA": {"alpha": 10.0, "layers": [8, 9]}},
                "config_id": "cfg_alpha10",
            },
        ],
    }


@pytest.fixture
def sample_run_with_per_example_metrics():
    """Sample run with per-example metric lists."""
    return {
        "trial_id": 0,
        "generations": [
            {"prompt": "Q1?", "response": "A", "instruction_id": "type_a"},
            {"prompt": "Q2?", "response": "B", "instruction_id": "type_b"},
            {"prompt": "Q3?", "response": "C", "instruction_id": "type_a"},
        ],
        "evaluations": {
            "StrictInstruction": {
                "strict_prompt_accuracy": 0.67,
                "follow_all_instructions": [True, False, True],
            },
            "RewardScore": {
                "mean_reward": 1.5,
                "rewards": [1.2, 1.8, 1.5],
            },
        },
        "params": {"PASTA": {"alpha": 5.0}},
        "config_id": "cfg_alpha5",
    }


# to_jsonable Tests
class TestToJsonable:
    """Tests for to_jsonable function."""

    def test_primitive_passthrough(self):
        """Test that primitives pass through unchanged."""
        assert to_jsonable("string") == "string"
        assert to_jsonable(42) == 42
        assert to_jsonable(3.14) == 3.14
        assert to_jsonable(True) is True
        assert to_jsonable(None) is None

    def test_path_to_string(self):
        """Test that Path objects become strings."""
        result = to_jsonable(Path("/some/path"))
        assert result == "/some/path"
        assert isinstance(result, str)

    def test_numpy_scalar(self):
        """Test numpy scalar conversion."""
        assert to_jsonable(np.float64(3.14)) == 3.14
        assert to_jsonable(np.int32(42)) == 42
        assert isinstance(to_jsonable(np.float64(3.14)), float)

    def test_numpy_array(self):
        """Test numpy array conversion."""
        arr = np.array([1, 2, 3])
        result = to_jsonable(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_object_array_recurses(self):
        """Test nested objects in numpy arrays become JSON-safe."""
        arr = np.array(
            [Path("models/base"), {"layers": {1, 2}}],
            dtype=object,
        )

        result = to_jsonable(arr)
        encoded = json.dumps(result)

        assert result[0] == "models/base"
        assert sorted(result[1]["layers"]) == [1, 2]
        assert json.loads(encoded) == result

    def test_nested_dict(self):
        """Test nested dict conversion."""
        data = {"a": np.float64(1.0), "b": {"c": np.array([1, 2])}}
        result = to_jsonable(data)
        assert result == {"a": 1.0, "b": {"c": [1, 2]}}

    def test_list_conversion(self):
        """Test list conversion."""
        data = [np.float64(1.0), np.array([2, 3]), "string"]
        result = to_jsonable(data)
        assert result == [1.0, [2, 3], "string"]

    def test_non_json_type_repr(self):
        """Test that non-JSON types become repr strings."""

        class CustomClass:
            def __repr__(self):
                return "CustomClass()"

        result = to_jsonable(CustomClass())
        assert result == "CustomClass()"


# flatten_profiles Tests
class TestFlattenProfiles:
    """Tests for flatten_profiles function."""

    def test_basic_flattening(self, sample_profiles_fixed):
        """Test basic profile flattening."""
        df = flatten_profiles(sample_profiles_fixed)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 baseline + 1 steered
        assert "pipeline" in df.columns
        assert "trial_id" in df.columns
        assert "config_id" in df.columns
        assert "params" in df.columns
        assert "_run" in df.columns

    def test_flattening_with_metric_accessors(self, sample_profiles_fixed):
        """Test flattening with metric extraction."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={
                "accuracy": ("Accuracy", "mean"),
                "reward": ("Reward", "mean_reward"),
            },
        )

        assert "accuracy" in df.columns
        assert "reward" in df.columns
        assert df[df["pipeline"] == "baseline"]["accuracy"].iloc[0] == 0.8

    def test_flattening_preserves_run_reference(self, sample_profiles_fixed):
        """Test that _run column contains original run dict."""
        df = flatten_profiles(sample_profiles_fixed)

        first_run = df.iloc[0]["_run"]
        assert isinstance(first_run, dict)
        assert "generations" in first_run
        assert "evaluations" in first_run

    def test_flattening_spec_profiles(self, sample_profiles_spec):
        """Test flattening ControlSpec-based profiles."""
        df = flatten_profiles(sample_profiles_spec)

        # 1 baseline + 4 alpha_sweep
        assert len(df) == 5

        # the two recorded config ids flow through, one per alpha value
        alpha_sweep_df = df[df["pipeline"] == "alpha_sweep"]
        config_ids = set(alpha_sweep_df["config_id"].unique())
        assert config_ids == {"cfg_alpha5", "cfg_alpha10"}

    def test_flattening_missing_metric(self, sample_profiles_fixed):
        """Test flattening with missing metric returns NaN."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={"nonexistent": ("NonexistentMetric", "value")},
        )

        assert "nonexistent" in df.columns
        assert df["nonexistent"].isna().all()

    def test_flattening_empty_profiles(self):
        """Test flattening empty profiles."""
        df = flatten_profiles({})
        assert len(df) == 0

    def test_flattening_baseline_config_id(self, sample_profiles_fixed):
        """Test that baseline runs get 'baseline' config_id."""
        df = flatten_profiles(sample_profiles_fixed)
        baseline_df = df[df["pipeline"] == "baseline"]
        assert (baseline_df["config_id"] == "baseline").all()


# extract_metric Tests
class TestExtractMetric:
    """Tests for extract_metric function."""

    def test_extract_existing_metric(self, sample_run_with_per_example_metrics):
        """Test extracting existing metric value."""
        result = extract_metric(
            sample_run_with_per_example_metrics,
            "StrictInstruction",
            "strict_prompt_accuracy",
        )
        assert result == 0.67

    def test_extract_missing_metric(self, sample_run_with_per_example_metrics):
        """Test extracting missing metric returns default."""
        result = extract_metric(
            sample_run_with_per_example_metrics,
            "NonexistentMetric",
            "value",
        )
        assert np.isnan(result)

    def test_extract_missing_key(self, sample_run_with_per_example_metrics):
        """Test extracting missing key returns default."""
        result = extract_metric(
            sample_run_with_per_example_metrics,
            "StrictInstruction",
            "nonexistent_key",
        )
        assert np.isnan(result)

    def test_extract_with_custom_default(self, sample_run_with_per_example_metrics):
        """Test extracting with custom default value."""
        result = extract_metric(
            sample_run_with_per_example_metrics,
            "NonexistentMetric",
            "value",
            default=-1,
        )
        assert result == -1


# extract_param Tests
class TestExtractParam:
    """Tests for extract_param function."""

    def test_extract_existing_param(self, sample_run_with_per_example_metrics):
        """Test extracting existing parameter value."""
        result = extract_param(
            sample_run_with_per_example_metrics,
            "PASTA",
            "alpha",
        )
        assert result == 5.0

    def test_extract_missing_spec(self, sample_run_with_per_example_metrics):
        """Test extracting from missing spec returns default."""
        result = extract_param(
            sample_run_with_per_example_metrics,
            "NonexistentSpec",
            "param",
        )
        assert result is None

    def test_extract_missing_param(self, sample_run_with_per_example_metrics):
        """Test extracting missing param returns default."""
        result = extract_param(
            sample_run_with_per_example_metrics,
            "PASTA",
            "nonexistent_param",
        )
        assert result is None

    def test_extract_with_custom_default(self, sample_run_with_per_example_metrics):
        """Test extracting with custom default."""
        result = extract_param(
            sample_run_with_per_example_metrics,
            "PASTA",
            "nonexistent_param",
            default="custom_default",
        )
        assert result == "custom_default"


# summarize_by_config Tests
class TestSummarizeByConfig:
    """Tests for summarize_by_config function."""

    def test_basic_summarization(self, sample_profiles_fixed):
        """Test basic summarization across trials."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        summary = summarize_by_config(df, metric_cols=["accuracy"])

        assert "accuracy_mean" in summary.columns
        assert "accuracy_std" in summary.columns
        assert "n_trials" in summary.columns

    def test_summarization_computes_mean(self, sample_profiles_fixed):
        """Test that summarization computes correct mean."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        summary = summarize_by_config(df, metric_cols=["accuracy"])

        baseline_summary = summary[summary["pipeline"] == "baseline"]
        # Baseline has trials with 0.8 and 0.7
        assert baseline_summary["accuracy_mean"].iloc[0] == pytest.approx(0.75, rel=1e-6)

    def test_summarization_computes_std(self, sample_profiles_fixed):
        """Test that summarization computes correct std."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        summary = summarize_by_config(df, metric_cols=["accuracy"])

        baseline_summary = summary[summary["pipeline"] == "baseline"]
        # std of [0.8, 0.7] with ddof=1
        expected_std = np.std([0.8, 0.7], ddof=1)
        assert baseline_summary["accuracy_std"].iloc[0] == pytest.approx(expected_std, rel=1e-6)

    def test_summarization_single_trial_zero_std(self, sample_profiles_fixed):
        """Test that single trial produces zero std."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        summary = summarize_by_config(df, metric_cols=["accuracy"])

        steered_summary = summary[summary["pipeline"] == "steered"]
        assert steered_summary["accuracy_std"].iloc[0] == 0.0

    def test_summarization_custom_group_cols(self, sample_profiles_spec):
        """Test summarization with custom group columns."""
        df = flatten_profiles(
            sample_profiles_spec,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        df["alpha"] = get_param_values(df, "PASTA", "alpha")

        # Filter to just alpha_sweep pipeline for this test
        df_sweep = df[df["pipeline"] == "alpha_sweep"]

        summary = summarize_by_config(
            df_sweep,
            metric_cols=["accuracy"],
            group_cols=["pipeline", "alpha"],
        )

        # Should have 2 groups: alpha=5.0, alpha=10.0
        assert len(summary) == 2

    def test_summarization_multiple_metrics(self, sample_profiles_fixed):
        """Test summarization with multiple metrics."""
        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={
                "accuracy": ("Accuracy", "mean"),
                "reward": ("Reward", "mean_reward"),
            },
        )
        summary = summarize_by_config(df, metric_cols=["accuracy", "reward"])

        assert "accuracy_mean" in summary.columns
        assert "accuracy_std" in summary.columns
        assert "reward_mean" in summary.columns
        assert "reward_std" in summary.columns


# get_param_values Tests
class TestGetParamValues:
    """Tests for get_param_values function."""

    def test_extract_param_series(self, sample_profiles_spec):
        """Test extracting parameter values as a Series."""
        df = flatten_profiles(sample_profiles_spec)
        alpha_series = get_param_values(df, "PASTA", "alpha")

        assert isinstance(alpha_series, pd.Series)
        assert len(alpha_series) == len(df)

    def test_extract_param_values(self, sample_profiles_spec):
        """Test that extracted values are correct."""
        df = flatten_profiles(sample_profiles_spec)
        alpha_series = get_param_values(df, "PASTA", "alpha")

        alpha_sweep_mask = df["pipeline"] == "alpha_sweep"
        assert alpha_series[alpha_sweep_mask].dropna().isin([5.0, 10.0]).all()

    def test_extract_missing_spec_returns_none(self, sample_profiles_spec):
        """Test that missing spec returns None values."""
        df = flatten_profiles(sample_profiles_spec)
        missing_series = get_param_values(df, "NonexistentSpec", "param")

        assert missing_series.isna().all() or (missing_series == None).all()  # noqa: E711

    def test_extract_baseline_returns_none(self, sample_profiles_spec):
        """Test that baseline (no params) returns None."""
        df = flatten_profiles(sample_profiles_spec)
        alpha_series = get_param_values(df, "PASTA", "alpha")

        baseline_mask = df["pipeline"] == "baseline"
        baseline_alphas = alpha_series[baseline_mask]
        assert baseline_alphas.isna().all() or (baseline_alphas == None).all()  # noqa: E711


# build_per_example_df Tests
class TestBuildPerExampleDf:
    """Tests for build_per_example_df function."""

    def test_basic_per_example_df(self, sample_run_with_per_example_metrics):
        """Test building basic per-example DataFrame."""
        df = build_per_example_df(sample_run_with_per_example_metrics)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "idx" in df.columns
        assert "prompt" in df.columns
        assert "response" in df.columns

    def test_custom_generation_fields(self, sample_run_with_per_example_metrics):
        """Test extracting custom generation fields."""
        df = build_per_example_df(
            sample_run_with_per_example_metrics,
            generation_fields=["prompt", "response", "instruction_id"],
        )

        assert "instruction_id" in df.columns
        assert df["instruction_id"].tolist() == ["type_a", "type_b", "type_a"]

    def test_metric_lists_extraction(self, sample_run_with_per_example_metrics):
        """Test extracting per-example metric lists."""
        df = build_per_example_df(
            sample_run_with_per_example_metrics,
            metric_lists={
                "followed": ("StrictInstruction", "follow_all_instructions"),
                "reward": ("RewardScore", "rewards"),
            },
        )

        assert "followed" in df.columns
        assert "reward" in df.columns
        assert df["followed"].tolist() == [True, False, True]
        assert df["reward"].tolist() == [1.2, 1.8, 1.5]

    def test_missing_metric_list(self, sample_run_with_per_example_metrics):
        """Test handling missing metric list."""
        df = build_per_example_df(
            sample_run_with_per_example_metrics,
            metric_lists={"missing": ("NonexistentMetric", "values")},
        )

        assert "missing" in df.columns
        assert df["missing"].isna().all() or (df["missing"] == None).all()  # noqa: E711

    def test_empty_generations(self):
        """Test handling empty generations."""
        run = {
            "trial_id": 0,
            "generations": [],
            "evaluations": {},
            "params": {},
            "config_id": "baseline",
        }
        df = build_per_example_df(run)

        assert len(df) == 0

    def test_default_generation_fields(self, sample_run_with_per_example_metrics):
        """Test default generation fields (prompt, response)."""
        df = build_per_example_df(sample_run_with_per_example_metrics)

        assert "prompt" in df.columns
        assert "response" in df.columns
        # instruction_id should NOT be included by default
        assert "instruction_id" not in df.columns


# Visualization Tests (basic import and structure tests)
class TestVizUtilsImport:
    """Tests for viz_utils import and basic functionality."""

    def test_viz_utils_importable(self):
        """Test that viz_utils can be imported."""
        try:
            from aisteer360.evaluation.utils import viz_utils

            assert hasattr(viz_utils, "plot_metric_by_config")
            assert hasattr(viz_utils, "plot_tradeoff_scatter")
            assert hasattr(viz_utils, "plot_metric_heatmap")
            assert hasattr(viz_utils, "plot_comparison_bars")
            assert hasattr(viz_utils, "plot_pareto_frontier")
            assert hasattr(viz_utils, "create_tradeoff_figure")
        except ImportError:
            pytest.skip("matplotlib not installed, skipping viz tests")


def _has_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
class TestVizUtils:
    """Tests for visualization utilities (requires matplotlib)."""

    def test_plot_metric_by_config(self, sample_profiles_spec):
        """Test plot_metric_by_config creates figure."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_metric_by_config

        df = flatten_profiles(
            sample_profiles_spec,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )
        df["alpha"] = get_param_values(df, "PASTA", "alpha")
        summary = summarize_by_config(
            df,
            metric_cols=["accuracy"],
            group_cols=["pipeline", "config_id", "alpha"],
        )

        # Filter to non-baseline
        summary = summary[summary["pipeline"] != "baseline"]

        ax = plot_metric_by_config(summary, metric="accuracy", x_col="alpha")

        assert ax is not None
        plt.close("all")

    def test_plot_tradeoff_scatter(self, sample_profiles_fixed):
        """Test plot_tradeoff_scatter creates figure."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_tradeoff_scatter

        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={
                "accuracy": ("Accuracy", "mean"),
                "reward": ("Reward", "mean_reward"),
            },
        )
        summary = summarize_by_config(df, metric_cols=["accuracy", "reward"])

        ax = plot_tradeoff_scatter(
            summary,
            x_metric="accuracy",
            y_metric="reward",
        )

        assert ax is not None
        plt.close("all")

    def test_plot_comparison_bars(self):
        """Test plot_comparison_bars creates figure."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_comparison_bars

        comparison_df = pd.DataFrame({
            "group": ["A", "B", "C"],
            "metric1": [0.5, 0.6, 0.7],
            "metric2": [0.3, 0.4, 0.5],
        })

        ax = plot_comparison_bars(
            comparison_df,
            metric_cols=["metric1", "metric2"],
            group_col="group",
        )

        assert ax is not None
        plt.close("all")

    def test_plot_pareto_frontier(self, sample_profiles_fixed):
        """Test plot_pareto_frontier creates figure."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_pareto_frontier

        df = flatten_profiles(
            sample_profiles_fixed,
            metric_accessors={
                "accuracy": ("Accuracy", "mean"),
                "reward": ("Reward", "mean_reward"),
            },
        )
        summary = summarize_by_config(df, metric_cols=["accuracy", "reward"])

        ax, points = plot_pareto_frontier(
            summary,
            x_metric="accuracy",
            y_metric="reward",
        )

        assert ax is not None
        assert isinstance(points, list)
        plt.close("all")

    def test_plot_tradeoff_with_fixed_pipelines(self):
        """Test plot_tradeoff with fixed_pipelines parameter."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_tradeoff

        # Create test data
        swept = pd.DataFrame({
            "k_positive": [1, 5, 10],
            "accuracy_mean": [0.4, 0.5, 0.55],
            "accuracy_std": [0.02, 0.03, 0.02],
            "positional_bias_mean": [0.1, 0.08, 0.07],
            "positional_bias_std": [0.01, 0.01, 0.01],
        })

        baseline = pd.DataFrame({
            "accuracy_mean": [0.35],
            "accuracy_std": [0.02],
            "positional_bias_mean": [0.12],
            "positional_bias_std": [0.01],
        })

        dpo = pd.DataFrame({
            "accuracy_mean": [0.7],
            "accuracy_std": [0.03],
            "positional_bias_mean": [0.05],
            "positional_bias_std": [0.01],
        })

        # Test with compare_to_pipelines as list of tuples
        ax = plot_tradeoff(
            swept=swept,
            x_metric="accuracy",
            y_metric="positional_bias",
            sweep_col="k_positive",
            compare_to_pipelines=[
                ("baseline", baseline),
                ("DPO-LoRA", dpo),
            ],
        )

        assert ax is not None
        # Check that legend contains both fixed pipelines
        legend_texts = [t.get_text() for t in ax.legend_.get_texts()]
        assert "baseline" in legend_texts
        assert "DPO-LoRA" in legend_texts
        plt.close("all")

    def test_plot_tradeoff_backward_compatible(self):
        """Test plot_tradeoff backward compatibility with baseline parameter."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import plot_tradeoff

        swept = pd.DataFrame({
            "k_positive": [1, 5],
            "accuracy_mean": [0.4, 0.5],
            "accuracy_std": [0.02, 0.03],
            "positional_bias_mean": [0.1, 0.08],
            "positional_bias_std": [0.01, 0.01],
        })

        baseline = pd.DataFrame({
            "accuracy_mean": [0.35],
            "accuracy_std": [0.02],
            "positional_bias_mean": [0.12],
            "positional_bias_std": [0.01],
        })

        # Test with only baseline (backward compatible)
        ax = plot_tradeoff(
            swept=swept,
            x_metric="accuracy",
            y_metric="positional_bias",
            sweep_col="k_positive",
            baseline=baseline,
        )

        assert ax is not None
        # Check that legend contains baseline
        legend_texts = [t.get_text() for t in ax.legend_.get_texts()]
        assert "baseline" in legend_texts
        plt.close("all")

    def test_create_tradeoff_figure(self, sample_profiles_spec):
        """Test create_tradeoff_figure creates multi-panel figure."""
        import matplotlib.pyplot as plt

        from aisteer360.evaluation.utils.viz_utils import create_tradeoff_figure

        # Create profiles with two metrics
        profiles = {
            "baseline": [
                {
                    "trial_id": 0,
                    "generations": [],
                    "evaluations": {"Accuracy": {"mean": 0.5}, "Reward": {"mean": 1.0}},
                    "params": {},
                    "config_id": "baseline",
                }
            ],
            "sweep": [
                {
                    "trial_id": 0,
                    "generations": [],
                    "evaluations": {"Accuracy": {"mean": 0.6}, "Reward": {"mean": 1.2}},
                    "params": {"CTRL": {"alpha": 5.0}},
                    "config_id": "cfg_ctrl5",
                },
                {
                    "trial_id": 0,
                    "generations": [],
                    "evaluations": {"Accuracy": {"mean": 0.7}, "Reward": {"mean": 1.4}},
                    "params": {"CTRL": {"alpha": 10.0}},
                    "config_id": "cfg_ctrl10",
                },
            ],
        }

        df = flatten_profiles(
            profiles,
            metric_accessors={
                "accuracy": ("Accuracy", "mean"),
                "reward": ("Reward", "mean"),
            },
        )
        df["alpha"] = get_param_values(df, "CTRL", "alpha")
        summary = summarize_by_config(
            df,
            metric_cols=["accuracy", "reward"],
            group_cols=["pipeline", "config_id", "alpha"],
        )

        fig = create_tradeoff_figure(
            summary,
            x_metric="accuracy",
            y_metric="reward",
            sweep_col="alpha",
            baseline_pipeline="baseline",
        )

        assert fig is not None
        assert len(fig.axes) >= 3  # Three panels (plus colorbar)
        plt.close("all")


# Integration Tests
class TestUtilsIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_analysis_workflow(self, sample_profiles_spec):
        """Test complete analysis workflow from profiles to summary."""
        # 1. Flatten profiles
        df = flatten_profiles(
            sample_profiles_spec,
            metric_accessors={"accuracy": ("Accuracy", "mean")},
        )

        # 2. Extract swept parameter
        df["alpha"] = get_param_values(df, "PASTA", "alpha")

        # 3. Summarize by configuration (use default group_cols for all pipelines)
        summary = summarize_by_config(
            df,
            metric_cols=["accuracy"],
        )

        # Verify structure - should have 3 groups: baseline + 2 alpha configs
        assert len(summary) == 3
        assert "accuracy_mean" in summary.columns
        assert "accuracy_std" in summary.columns
        assert "n_trials" in summary.columns

        # 4. For alpha comparison, filter to alpha_sweep and re-summarize with alpha column
        df_sweep = df[df["pipeline"] == "alpha_sweep"]
        sweep_summary = summarize_by_config(
            df_sweep,
            metric_cols=["accuracy"],
            group_cols=["pipeline", "config_id", "alpha"],
        )

        # Verify alpha=10.0 has higher accuracy than alpha=5.0
        alpha_5 = sweep_summary[sweep_summary["alpha"] == 5.0]["accuracy_mean"].iloc[0]
        alpha_10 = sweep_summary[sweep_summary["alpha"] == 10.0]["accuracy_mean"].iloc[0]
        assert alpha_10 > alpha_5

    def test_per_example_analysis_workflow(self, sample_run_with_per_example_metrics):
        """Test per-example analysis workflow."""
        # Build per-example DataFrame
        df = build_per_example_df(
            sample_run_with_per_example_metrics,
            generation_fields=["prompt", "response", "instruction_id"],
            metric_lists={
                "followed": ("StrictInstruction", "follow_all_instructions"),
                "reward": ("RewardScore", "rewards"),
            },
        )

        # Analyze by instruction type
        by_type = df.groupby("instruction_id").agg({
            "followed": "mean",
            "reward": "mean",
        })

        assert len(by_type) == 2  # type_a and type_b
        assert by_type.loc["type_a", "followed"] == 1.0  # Both type_a followed
        assert by_type.loc["type_b", "followed"] == 0.0  # type_b did not follow
