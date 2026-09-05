"""Tests for `InspectSuite` over a mocked `eval_set`: call shape, per-task selection, failure
handling, and result flattening."""
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

import steerability.evaluation.suite as suite_module
from steerability.evaluation.suite import InspectSuite
from tests.evaluation.conftest import StubSteeringPipeline


def _log(task, metrics_by_scorer, *, status="success", n=3, location="logs/one.eval"):
    scores = [
        SimpleNamespace(
            name=scorer_name,
            scorer=scorer_name,
            metrics={
                metric_name: SimpleNamespace(name=metric_name, value=value)
                for metric_name, value in metric_values.items()
            },
        )
        for scorer_name, metric_values in metrics_by_scorer.items()
    ]
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(task=task),
        results=SimpleNamespace(scores=scores, completed_samples=n),
        location=location,
    )


@pytest.fixture
def recorded_eval_set(monkeypatch):
    """Replace `eval_set` with a recorder returning configurable logs."""
    calls = []
    plan = {"logs": {}, "success": True}

    def fake_eval_set(tasks, *, log_dir, **kwargs):
        calls.append({"tasks": list(tasks), "log_dir": log_dir, **kwargs})
        logs = [plan["logs"].get(task, _log(task, {"match": {"accuracy": 1.0}})) for task in tasks]
        return plan["success"], logs

    monkeypatch.setattr(suite_module, "eval_set", fake_eval_set)
    return calls, plan


class TestValidation:
    def test_empty_tasks_raise(self):
        with pytest.raises(ValueError, match="tasks"):
            InspectSuite(name="capability", tasks=())

    def test_bad_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            InspectSuite(name="capability", tasks=("t",), limit=0)

    def test_sample_ids_must_name_suite_tasks(self):
        with pytest.raises(ValueError, match="other"):
            InspectSuite(name="capability", tasks=("t",), sample_ids={"other": (1,)})


class TestRun:
    def test_eval_set_call_shape(self, recorded_eval_set, tmp_path):
        calls, _ = recorded_eval_set
        suite = InspectSuite(
            name="capability", tasks=("a", "b"), limit=25,
            task_args={"grader_model": "openai/x"}, generate_overrides={"max_tokens": 64},
            retry_attempts=5,
        )
        results = suite.run(
            StubSteeringPipeline(), log_dir=tmp_path, model_name="cfg-1",
            generate_defaults={"temperature": 0, "max_tokens": 16},
        )
        (call,) = calls
        assert call["tasks"] == ["a", "b"]
        assert call["log_dir"] == str(tmp_path)
        assert call["max_tasks"] == 1
        assert call["epochs"] == 1
        assert call["display"] == "none"
        assert call["retry_attempts"] == 5
        assert call["limit"] == 25
        assert call["task_args"] == {"grader_model": "openai/x"}
        assert call["temperature"] == 0
        assert call["max_tokens"] == 64  # suite overrides beat runner defaults
        assert str(call["model"]) == "steerability/cfg-1"
        assert set(results) == {"a", "b"}

    def test_score_defaults_true_and_forwards_false(self, recorded_eval_set, tmp_path):
        calls, _ = recorded_eval_set
        suite = InspectSuite(name="capability", tasks=("a",))
        suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert calls[-1]["score"] is True
        suite.run(StubSteeringPipeline(), log_dir=tmp_path, score=False)
        assert calls[-1]["score"] is False

    def test_model_roles_omitted_by_default_and_forwarded_when_given(self, recorded_eval_set, tmp_path):
        calls, _ = recorded_eval_set
        suite = InspectSuite(name="capability", tasks=("a",))
        suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert "model_roles" not in calls[-1]
        suite.run(StubSteeringPipeline(), log_dir=tmp_path, model_roles={"grader": "openai/x"})
        assert calls[-1]["model_roles"] == {"grader": "openai/x"}

    def test_display_defaults_to_none_and_forwards_when_given(self, recorded_eval_set, tmp_path):
        calls, _ = recorded_eval_set
        suite = InspectSuite(name="capability", tasks=("a",))
        suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert calls[-1]["display"] == "none"
        suite.run(StubSteeringPipeline(), log_dir=tmp_path, display="plain")
        assert calls[-1]["display"] == "plain"

    def test_sample_ids_run_one_eval_set_per_task(self, recorded_eval_set, tmp_path):
        calls, _ = recorded_eval_set
        suite = InspectSuite(
            name="capability", tasks=("pkg/a", "b"), limit=10, sample_ids={"pkg/a": (1, 2)},
        )
        suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert len(calls) == 2
        first, second = calls
        assert first["tasks"] == ["pkg/a"]
        assert first["sample_id"] == [1, 2]
        assert "limit" not in first
        assert first["log_dir"] == str(tmp_path / "pkg_a")
        assert second["tasks"] == ["b"]
        assert second["limit"] == 10
        assert second["log_dir"] == str(tmp_path / "b")

    def test_failure_raises_naming_failed_tasks(self, recorded_eval_set, tmp_path):
        _, plan = recorded_eval_set
        plan["success"] = False
        plan["logs"]["b"] = _log("b", {}, status="error")
        suite = InspectSuite(name="capability", tasks=("a", "b"))
        with pytest.raises(RuntimeError, match="failed for task\\(s\\): b"):
            suite.run(StubSteeringPipeline(), log_dir=tmp_path)

    def test_failure_without_failed_logs_raises_naming_unknown(self, recorded_eval_set, tmp_path):
        _, plan = recorded_eval_set
        plan["success"] = False  # every returned log reports success
        suite = InspectSuite(name="capability", tasks=("a",))
        with pytest.raises(RuntimeError, match="unknown"):
            suite.run(StubSteeringPipeline(), log_dir=tmp_path)

    def test_flattening_keys_metrics_by_scorer_including_stderr(self, recorded_eval_set, tmp_path):
        _, plan = recorded_eval_set
        plan["logs"]["a"] = _log(
            "a",
            {"match": {"accuracy": 0.81, "stderr": 0.04}, "grader": {"mean": 0.5}},
            n=100,
            location=str(tmp_path / "a.eval"),
        )
        suite = InspectSuite(name="capability", tasks=("a",))
        results = suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert results["a"]["metrics"] == {
            "match/accuracy": 0.81, "match/stderr": 0.04, "grader/mean": 0.5,
        }
        assert results["a"]["n"] == 100
        assert results["a"]["log"] == "a.eval"  # relative to log_dir

    def test_file_uri_location_relativized_against_log_dir(self, recorded_eval_set, tmp_path):
        _, plan = recorded_eval_set
        plan["logs"]["a"] = _log(
            "a", {"match": {"accuracy": 1.0}}, location=f"file:{tmp_path / 'a.eval'}",
        )
        suite = InspectSuite(name="capability", tasks=("a",))
        results = suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert results["a"]["log"] == "a.eval"

    def test_remote_uri_location_kept_verbatim(self, recorded_eval_set, tmp_path):
        _, plan = recorded_eval_set
        plan["logs"]["a"] = _log(
            "a", {"match": {"accuracy": 1.0}}, location="s3://bucket/runs/a.eval",
        )
        suite = InspectSuite(name="capability", tasks=("a",))
        results = suite.run(StubSteeringPipeline(), log_dir=tmp_path)
        assert results["a"]["log"] == "s3://bucket/runs/a.eval"
