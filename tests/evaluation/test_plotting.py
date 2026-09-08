"""Smoke tests for `steerability.evaluation.plotting`: every public function renders on the Agg
backend under warnings-as-errors, the multi-configuration reference guard raises, and the Pareto
direction handling is pinned.
"""
import warnings

import pandas
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from steerability.evaluation import plotting  # noqa: E402


def _summary() -> pandas.DataFrame:
    """A small swept summary frame: three few-shot configurations with mean/std/sem columns."""
    return pandas.DataFrame({
        "pipeline": ["few_shot_sweep"] * 3,
        "config_id": ["cfg_k1", "cfg_k5", "cfg_k10"],
        "k_positive": [1, 5, 10],
        "accuracy_mean": [0.55, 0.62, 0.68],
        "accuracy_std": [0.03, 0.02, 0.04],
        "accuracy_sem": [0.015, 0.010, 0.020],
        "positional_bias_mean": [0.20, 0.15, 0.12],
        "positional_bias_std": [0.02, 0.01, 0.02],
        "positional_bias_sem": [0.010, 0.005, 0.010],
    })


def _baseline() -> pandas.DataFrame:
    """A one-configuration reference summary frame."""
    return pandas.DataFrame({
        "pipeline": ["baseline"],
        "config_id": ["baseline"],
        "accuracy_mean": [0.45],
        "accuracy_std": [0.02],
        "accuracy_sem": [0.010],
        "positional_bias_mean": [0.10],
        "positional_bias_std": [0.01],
        "positional_bias_sem": [0.005],
    })


def _per_trial() -> pandas.DataFrame:
    """Per-trial rows aligned with the swept summary, for the scatter overlays."""
    return pandas.DataFrame({
        "pipeline": ["few_shot_sweep"] * 6,
        "config_id": ["cfg_k1", "cfg_k1", "cfg_k5", "cfg_k5", "cfg_k10", "cfg_k10"],
        "k_positive": [1, 1, 5, 5, 10, 10],
        "accuracy": [0.53, 0.57, 0.60, 0.64, 0.66, 0.70],
        "positional_bias": [0.19, 0.21, 0.14, 0.16, 0.11, 0.13],
    })


class TestSmokeAllPublicFunctions:
    def test_all_nine_render_without_warnings(self, tmp_path):
        summary = _summary()
        baseline = _baseline()
        per_trial = _per_trial()
        refs = [("baseline", baseline)]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plotting.apply_plot_style()

            plotting.plot_metric_by_config(
                summary, metric="accuracy", baseline_value=0.45, baseline_std=0.02,
                save_path=tmp_path / "metric_by_config.png",
            )
            plotting.plot_metric_by_config(
                summary, metric="accuracy", save_path=tmp_path / "metric_by_config_no_baseline.png",
            )
            plotting.plot_tradeoff_scatter(
                summary, x_metric="accuracy", y_metric="positional_bias",
                color_col="k_positive", label_col="k_positive", compare_to_pipelines=refs,
                per_trial_data=per_trial, show_pareto=True, maximize_y=False,
                save_path=tmp_path / "tradeoff_scatter.png",
            )
            plotting.plot_comparison_bars(
                summary.assign(label=summary["config_id"]),
                metric_cols=["accuracy_mean", "positional_bias_mean"], group_col="label",
                save_path=tmp_path / "comparison_bars.png",
            )
            plotting.plot_sensitivity(
                summary, metric="accuracy", sweep_col="k_positive",
                compare_to_pipelines=refs, per_trial_data=per_trial,
                save_path=tmp_path / "sensitivity.png",
            )
            plotting.plot_tradeoff(
                summary, x_metric="accuracy", y_metric="positional_bias", sweep_col="k_positive",
                compare_to_pipelines=refs, per_trial_data=per_trial,
                maximize_y=False, save_path=tmp_path / "tradeoff.png",
            )
            full = pandas.concat([baseline, summary], ignore_index=True)
            plotting.create_tradeoff_figure(
                full, x_metric="accuracy", y_metric="positional_bias", sweep_col="k_positive",
                save_path=tmp_path / "tradeoff_figure.png",
            )
            plotting.plot_pareto_frontier(
                summary, x_metric="accuracy", y_metric="positional_bias",
                maximize_y=False, save_path=tmp_path / "pareto.png",
            )

            pytest.importorskip("seaborn")
            pivot = summary.pivot_table(index="pipeline", columns="k_positive", values="accuracy_mean")
            plotting.plot_metric_heatmap(pivot, save_path=tmp_path / "heatmap.png")

        for name in [
            "metric_by_config.png", "metric_by_config_no_baseline.png", "tradeoff_scatter.png",
            "comparison_bars.png", "sensitivity.png", "tradeoff.png", "tradeoff_figure.png",
            "pareto.png", "heatmap.png",
        ]:
            assert (tmp_path / name).exists()


class TestReferenceGuard:
    def test_multi_config_reference_raises(self):
        multi = pandas.DataFrame({
            "pipeline": ["arm", "arm"],
            "config_id": ["a", "b"],
            "accuracy_mean": [0.5, 0.6],
            "accuracy_std": [0.01, 0.01],
            "positional_bias_mean": [0.1, 0.1],
            "positional_bias_std": [0.01, 0.01],
        })
        with pytest.raises(ValueError, match="configurations"):
            plotting.plot_sensitivity(
                _summary(), metric="accuracy", sweep_col="k_positive",
                compare_to_pipelines=[("swept", multi)],
            )


class TestParetoDirection:
    def test_maximize_y_changes_frontier(self):
        summary = _summary()
        maximized = plotting._compute_pareto_points(
            summary, "accuracy", "positional_bias", maximize_x=True, maximize_y=True,
        )
        minimized = plotting._compute_pareto_points(
            summary, "accuracy", "positional_bias", maximize_x=True, maximize_y=False,
        )
        assert maximized != minimized
