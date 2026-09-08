"""Tests for `steerability.evaluation.runner.samples_frame` and the `SteeringEval.samples_frame`
method, over stubbed eval logs.

Pure pandas; `runner._read_eval_log` is monkeypatched to return `SimpleNamespace` logs (mirroring
how `tests/evaluation/test_suite.py` stubs `eval_set`), so no `.eval` files are read and no model
runs.
"""
from types import SimpleNamespace

import pandas
import pytest

import steerability.evaluation.runner as runner_module
from steerability.evaluation.runner import SteeringEval, samples_frame


def _sample(sample_id, scores, metadata=None, *, completion="", text=""):
    """One stub `EvalSample`: id, a scorer-name -> stub `Score` mapping, metadata, and output."""
    return SimpleNamespace(
        id=sample_id,
        scores={name: SimpleNamespace(value=value) for name, value in scores.items()},
        metadata=metadata or {},
        input=text,
        output=SimpleNamespace(completion=completion),
    )


def _suites(log_path: str, suite: str = "ifeval", task: str = "task") -> dict:
    return {suite: {task: {"metrics": {}, "n": 1, "log": log_path}}}


def _stub_logs(monkeypatch, logs: dict[str, SimpleNamespace]) -> None:
    """Route `_read_eval_log` to `logs`, keyed by the log path's final component."""

    def fake_read(path):
        return logs[str(path).rsplit("/", 1)[-1]]

    monkeypatch.setattr(runner_module, "_read_eval_log", fake_read)


def _two_arm_results() -> dict[str, list[dict]]:
    """A baseline arm (one trial) and a swept PASTA arm (one trial), one suite/task each."""
    return {
        "baseline": [
            {"trial_id": 0, "seed": 1, "config_id": "baseline", "params": {},
             "suites": _suites("a.eval")},
        ],
        "pasta_sweep": [
            {"trial_id": 0, "seed": 2, "config_id": "cfg_a20",
             "params": {"PASTA": {"alpha": 20.0}}, "suites": _suites("b.eval")},
        ],
    }


class TestSamplesFrame:
    def test_one_row_per_pipeline_trial_sample(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[
                _sample(1, {"reward_score": 0.5}),
                _sample(2, {"reward_score": 1.5}),
            ]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 2.5})]),
        })
        frame = samples_frame(_two_arm_results(), "/logs", scores={"reward": "reward_score"})
        assert list(frame.columns) == [
            "pipeline", "config_id", "trial_id", "seed", "suite", "task", "sample_id", "reward",
        ]
        assert len(frame) == 3  # 2 baseline samples + 1 pasta sample
        assert frame.sort_values("pipeline")["reward"].tolist() == [0.5, 1.5, 2.5]

    def test_scalar_and_dict_key_scores(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[
                _sample(1, {"checker": {"prompt_level_strict": 0.0}, "reward_score": -1.0}),
            ]),
            "b.eval": SimpleNamespace(samples=[
                _sample(1, {"checker": {"prompt_level_strict": 1.0}, "reward_score": 3.0}),
            ]),
        })
        frame = samples_frame(
            _two_arm_results(), "/logs",
            scores={"followed": "checker/prompt_level_strict", "reward": "reward_score"},
        )
        assert frame.set_index("pipeline")["followed"].to_dict() == {"baseline": 0.0, "pasta_sweep": 1.0}
        assert frame.set_index("pipeline")["reward"].to_dict() == {"baseline": -1.0, "pasta_sweep": 3.0}

    def test_letter_and_bool_values_converted_through_value_to_float(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"choice": "C", "flag": True})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"choice": "I", "flag": False})]),
        })
        frame = samples_frame(
            _two_arm_results(), "/logs", scores={"grade": "choice", "flag": "flag"},
        )
        by_pipeline = frame.set_index("pipeline")
        assert by_pipeline["grade"].to_dict() == {"baseline": 1.0, "pasta_sweep": 0.0}
        assert by_pipeline["flag"].to_dict() == {"baseline": 1.0, "pasta_sweep": 0.0}

    def test_non_scalar_value_kept_raw(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"tokens": ["a", "b"]})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"tokens": ["c"]})]),
        })
        frame = samples_frame(_two_arm_results(), "/logs", scores={"tokens": "tokens"})
        assert frame.sort_values("pipeline")["tokens"].tolist() == [["a", "b"], ["c"]]
        assert frame["tokens"].dtype == object

    def test_metadata_keys_carried_as_columns(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[
                _sample(1, {"reward_score": 0.0}, {"instruction_id": "startend:end_checker"}),
            ]),
            "b.eval": SimpleNamespace(samples=[
                _sample(1, {"reward_score": 0.0}, {"instruction_id": "language:response_language"}),
            ]),
        })
        frame = samples_frame(
            _two_arm_results(), "/logs",
            scores={"reward": "reward_score"}, metadata_keys=["instruction_id"],
        )
        assert "instruction_id" in frame.columns
        assert set(frame["instruction_id"]) == {"startend:end_checker", "language:response_language"}

    def test_include_text_adds_input_and_completion(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[
                _sample(1, {"reward_score": 0.0}, completion="answer one", text="prompt one"),
            ]),
            "b.eval": SimpleNamespace(samples=[
                _sample(1, {"reward_score": 0.0}, completion="answer two", text="prompt two"),
            ]),
        })
        frame = samples_frame(
            _two_arm_results(), "/logs", scores={"reward": "reward_score"}, include_text=True,
        )
        assert {"input", "completion"} <= set(frame.columns)
        row = frame[frame["pipeline"] == "baseline"].iloc[0]
        assert row["input"] == "prompt one"
        assert row["completion"] == "answer one"

    def test_relative_log_path_resolved_under_log_root(self, monkeypatch):
        seen: list[str] = []

        def fake_read(path):
            seen.append(str(path))
            return SimpleNamespace(samples=[])

        monkeypatch.setattr(runner_module, "_read_eval_log", fake_read)
        results = {"baseline": [
            {"trial_id": 0, "seed": 1, "config_id": "baseline", "params": {},
             "suites": _suites("inspect_logs/baseline/trial_0/ifeval/a.eval")},
        ]}
        samples_frame(results, "/logs", scores={"reward": "reward_score"})
        assert seen == ["/logs/inspect_logs/baseline/trial_0/ifeval/a.eval"]

    def test_empty_scores_raises(self, monkeypatch):
        _stub_logs(monkeypatch, {})
        with pytest.raises(ValueError, match="at least one"):
            samples_frame(_two_arm_results(), "/logs", scores={})

    def test_unknown_scorer_raises_keyerror(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 0.0})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 0.0})]),
        })
        with pytest.raises(KeyError, match="missing"):
            samples_frame(_two_arm_results(), "/logs", scores={"x": "missing"})

    def test_unknown_dict_key_raises_keyerror(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"checker": {"prompt_level_strict": 1.0}})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"checker": {"prompt_level_strict": 1.0}})]),
        })
        with pytest.raises(KeyError, match="absent"):
            samples_frame(_two_arm_results(), "/logs", scores={"x": "checker/nope"})

    def test_several_suites_without_selector_raises(self, monkeypatch):
        _stub_logs(monkeypatch, {"a.eval": SimpleNamespace(samples=[])})
        results = {"baseline": [
            {"trial_id": 0, "seed": 1, "config_id": "baseline", "params": {},
             "suites": {**_suites("a.eval", suite="one"), **_suites("a.eval", suite="two")}},
        ]}
        with pytest.raises(ValueError, match="suites"):
            samples_frame(results, "/logs", scores={"reward": "reward_score"})

    def test_suite_selector_narrows(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 0.5})]),
        })
        results = {"baseline": [
            {"trial_id": 0, "seed": 1, "config_id": "baseline", "params": {},
             "suites": {**_suites("a.eval", suite="one"), **_suites("a.eval", suite="two")}},
        ]}
        frame = samples_frame(results, "/logs", scores={"reward": "reward_score"}, suite="one")
        assert frame["suite"].unique().tolist() == ["one"]


class _StubSuite:
    """Duck-typed suite; the method reads only `self._results` / `self._log_root`."""

    name = "ifeval"
    tasks = ("task",)

    def run(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise AssertionError("run() should not be called in these tests")


class TestSteeringEvalSamplesFrame:
    def _runner(self) -> SteeringEval:
        runner = SteeringEval({"baseline": [], "pasta_sweep": []}, "test-model", [_StubSuite()])
        runner._results = _two_arm_results()
        runner._log_root = "/logs"
        return runner

    def test_swept_params_attach_by_config_id(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 0.5})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 2.5})]),
        })
        frame = self._runner().samples_frame(
            {"reward": "reward_score"}, params={"alpha": ("PASTA", "alpha")},
        )
        assert frame.loc[frame["pipeline"] == "pasta_sweep", "alpha"].tolist() == [20.0]
        assert pandas.api.types.is_numeric_dtype(frame["alpha"])

    def test_baseline_rows_get_nan(self, monkeypatch):
        _stub_logs(monkeypatch, {
            "a.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 0.5})]),
            "b.eval": SimpleNamespace(samples=[_sample(1, {"reward_score": 2.5})]),
        })
        frame = self._runner().samples_frame(
            {"reward": "reward_score"}, params={"alpha": ("PASTA", "alpha")},
        )
        assert frame.loc[frame["pipeline"] == "baseline", "alpha"].isna().all()

    def test_raises_before_run(self):
        runner = SteeringEval({"baseline": []}, "test-model", [_StubSuite()])
        with pytest.raises(RuntimeError, match="run"):
            runner.samples_frame({"reward": "reward_score"})
