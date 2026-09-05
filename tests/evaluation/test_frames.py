"""Tests for the reshaping bridge in `steerability.evaluation.runner`: `runs_frame`,
`summarize_runs`, and `SteeringEval.runs_frame`.

Pure pandas, no models and no Inspect imports; the runner-method test drives a duck-typed stub
suite (mirroring `tests/evaluation/test_runner.py`) and assigns synthetic run records directly.
"""
import math

import pandas
import pytest

from steerability.evaluation.runner import SteeringEval, runs_frame, summarize_runs

_RESULTS_COLUMNS = [
    "config", "config_id", "trial", "seed", "suite", "task", "scorer", "metric", "value", "n", "log",
]


def make_results(rows: list[dict], scorer: str = "choice") -> pandas.DataFrame:
    """Build a tidy results frame with the exact `results()` columns from compact metric rows.

    Each entry in `rows` supplies `config`, `config_id`, `trial`, and a `metrics` mapping from
    metric name to value; suite/task/seed/n/log default to fixed values.
    """
    records: list[dict] = []
    for row in rows:
        for metric_name, value in row["metrics"].items():
            records.append({
                "config": row["config"],
                "config_id": row["config_id"],
                "trial": row["trial"],
                "seed": row.get("seed", 0),
                "suite": row.get("suite", "mcqa"),
                "task": row.get("task", "commonsense_mcqa"),
                "scorer": row.get("scorer", scorer),
                "metric": metric_name,
                "value": value,
                "n": row.get("n", 4),
                "log": row.get("log", "one.eval"),
            })
    return pandas.DataFrame(records, columns=_RESULTS_COLUMNS)


def _two_config_results() -> pandas.DataFrame:
    return make_results([
        {"config": "baseline", "config_id": "baseline", "trial": 0, "metrics": {"accuracy": 0.40, "positional_bias": 0.10}},
        {"config": "baseline", "config_id": "baseline", "trial": 1, "metrics": {"accuracy": 0.50, "positional_bias": 0.12}},
        {"config": "few_shot_sweep", "config_id": "cfg_k1", "trial": 0, "metrics": {"accuracy": 0.60, "positional_bias": 0.20}},
        {"config": "few_shot_sweep", "config_id": "cfg_k1", "trial": 1, "metrics": {"accuracy": 0.70, "positional_bias": 0.22}},
    ])


class TestRunsFrame:
    def test_pivots_to_one_row_per_pipeline_trial(self):
        frame = runs_frame(
            _two_config_results(),
            {"accuracy": "choice/accuracy", "positional_bias": "choice/positional_bias"},
        )
        assert list(frame.columns) == [
            "pipeline", "config_id", "trial_id", "seed", "accuracy", "positional_bias",
        ]
        assert len(frame) == 4  # 2 configs x 2 trials

    def test_pivoted_cell_matches_tidy_source(self):
        frame = runs_frame(_two_config_results(), {"accuracy": "choice/accuracy"})
        cell = frame[(frame["pipeline"] == "few_shot_sweep") & (frame["trial_id"] == 1)]
        assert cell["accuracy"].iloc[0] == pytest.approx(0.70)

    def test_bare_metric_name_accepted_when_unambiguous(self):
        frame = runs_frame(_two_config_results(), {"accuracy": "accuracy"})
        assert set(frame.columns) == {"pipeline", "config_id", "trial_id", "seed", "accuracy"}
        assert frame["accuracy"].tolist() == [0.40, 0.50, 0.60, 0.70]

    def test_unknown_metric_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            runs_frame(_two_config_results(), {"missing": "choice/missing"})

    def test_ambiguous_bare_metric_name_raises_keyerror(self):
        first = _two_config_results()
        second = first.copy()
        second["scorer"] = "match"
        combined = pandas.concat([first, second], ignore_index=True)
        with pytest.raises(KeyError, match="ambiguous"):
            runs_frame(combined, {"accuracy": "accuracy"})

    def test_several_suites_without_selector_raises(self):
        first = _two_config_results()
        second = first.copy()
        second["suite"] = "other"
        combined = pandas.concat([first, second], ignore_index=True)
        with pytest.raises(ValueError, match="suites"):
            runs_frame(combined, {"accuracy": "choice/accuracy"})

    def test_suite_selector_narrows_to_one_suite(self):
        first = _two_config_results()
        second = first.copy()
        second["suite"] = "other"
        combined = pandas.concat([first, second], ignore_index=True)
        frame = runs_frame(combined, {"accuracy": "choice/accuracy"}, suite="mcqa")
        assert len(frame) == 4

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="No rows match"):
            runs_frame(_two_config_results(), {"accuracy": "choice/accuracy"}, suite="nonexistent")

    def test_empty_metrics_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            runs_frame(_two_config_results(), {})


class TestSummarizeRuns:
    def test_mean_std_sem_n_against_ground_truth(self):
        runs = runs_frame(
            _two_config_results(),
            {"accuracy": "choice/accuracy", "positional_bias": "choice/positional_bias"},
        )
        summary = summarize_runs(runs, ["accuracy", "positional_bias"])
        baseline = summary[summary["pipeline"] == "baseline"].iloc[0]

        assert baseline["accuracy_mean"] == pytest.approx(0.45)
        std = pandas.Series([0.40, 0.50]).std()  # sample std, ddof=1
        assert baseline["accuracy_std"] == pytest.approx(std)
        assert baseline["accuracy_sem"] == pytest.approx(std / math.sqrt(2))
        assert baseline["n_trials"] == 2

    def test_absent_param_cols_ignored(self):
        runs = runs_frame(_two_config_results(), {"accuracy": "choice/accuracy"})
        summary = summarize_runs(runs, ["accuracy"], param_cols=["k_positive"])
        assert "k_positive" not in summary.columns

    def test_single_trial_std_and_sem_are_zero(self):
        results = make_results([
            {"config": "dpo", "config_id": "dpo", "trial": 0, "metrics": {"accuracy": 0.8}},
        ])
        runs = runs_frame(results, {"accuracy": "choice/accuracy"})
        summary = summarize_runs(runs, ["accuracy"])
        assert summary["accuracy_std"].iloc[0] == 0.0
        assert summary["accuracy_sem"].iloc[0] == 0.0

    def test_empty_metric_cols_raises(self):
        runs = runs_frame(_two_config_results(), {"accuracy": "choice/accuracy"})
        with pytest.raises(ValueError, match="at least one"):
            summarize_runs(runs, [])

    def test_param_cols_carried_through(self):
        runs = runs_frame(_two_config_results(), {"accuracy": "choice/accuracy"})
        runs["k_positive"] = [float("nan"), float("nan"), 5.0, 5.0]
        summary = summarize_runs(runs, ["accuracy"], param_cols=["k_positive"])
        swept = summary[summary["pipeline"] == "few_shot_sweep"].iloc[0]
        assert swept["k_positive"] == 5.0


class _StubSuite:
    """Duck-typed suite; `SteeringEval.runs_frame` only reads `self._results`, so `run()` is unused."""

    name = "mcqa"
    tasks = ("commonsense_mcqa",)

    def run(self, *args, **kwargs):  # pragma: no cover - not exercised here
        raise AssertionError("run() should not be called in these tests")


def _synthetic_records() -> dict[str, list[dict]]:
    """Run records shaped like `SteeringEval._results`: a baseline arm and a swept few-shot arm."""

    def suites(accuracy: float) -> dict:
        return {"mcqa": {"commonsense_mcqa": {
            "metrics": {"choice/accuracy": accuracy, "choice/positional_bias": 0.1},
            "n": 4, "log": "one.eval",
        }}}

    return {
        "baseline": [
            {"trial_id": 0, "seed": 1, "config_id": "baseline", "params": {}, "suites": suites(0.4)},
        ],
        "few_shot_sweep": [
            {"trial_id": 0, "seed": 2, "config_id": "cfg_k1",
             "params": {"FewShot": {"k_positive": 1, "k_negative": 0}}, "suites": suites(0.6)},
            {"trial_id": 0, "seed": 3, "config_id": "cfg_k5",
             "params": {"FewShot": {"k_positive": 5, "k_negative": 0}}, "suites": suites(0.7)},
        ],
    }


class TestSteeringEvalRunsFrame:
    def _runner(self) -> SteeringEval:
        return SteeringEval({"baseline": [], "few_shot_sweep": []}, "test-model", [_StubSuite()])

    def test_swept_params_attach_by_config_id(self):
        runner = self._runner()
        runner._results = _synthetic_records()
        frame = runner.runs_frame(
            {"accuracy": "choice/accuracy"},
            params={"k_positive": ("FewShot", "k_positive")},
        )
        swept = frame[frame["pipeline"] == "few_shot_sweep"].sort_values("config_id")
        assert swept["k_positive"].tolist() == [1, 5]

    def test_baseline_rows_get_nan(self):
        runner = self._runner()
        runner._results = _synthetic_records()
        frame = runner.runs_frame(
            {"accuracy": "choice/accuracy"},
            params={"k_positive": ("FewShot", "k_positive")},
        )
        baseline = frame[frame["pipeline"] == "baseline"]
        assert baseline["k_positive"].isna().all()

    def test_numeric_swept_column_is_numeric_dtype(self):
        runner = self._runner()
        runner._results = _synthetic_records()
        frame = runner.runs_frame(
            {"accuracy": "choice/accuracy"},
            params={"k_positive": ("FewShot", "k_positive")},
        )
        assert pandas.api.types.is_numeric_dtype(frame["k_positive"])

    def test_list_valued_swept_argument_kept_raw(self):
        records = _synthetic_records()
        records["few_shot_sweep"][0]["params"]["FewShot"]["target_modules"] = ["q_proj", "v_proj"]
        records["few_shot_sweep"][1]["params"]["FewShot"]["target_modules"] = ["q_proj"]
        runner = self._runner()
        runner._results = records
        frame = runner.runs_frame(
            {"accuracy": "choice/accuracy"},
            params={"modules": ("FewShot", "target_modules")},
        )
        swept = frame[frame["pipeline"] == "few_shot_sweep"].sort_values("config_id")
        assert swept["modules"].tolist() == [["q_proj", "v_proj"], ["q_proj"]]
        assert frame["modules"].dtype == object

    def test_raises_before_run(self):
        runner = self._runner()
        with pytest.raises(RuntimeError, match="run"):
            runner.runs_frame({"accuracy": "choice/accuracy"})
