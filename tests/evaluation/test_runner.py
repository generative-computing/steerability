"""Tests for `SteeringEval` over stub suites: run shape, results frame, log-dir layout, resume,
sequential execution, pre-flight policy, and factory discipline under a raising suite.

The runner itself imports no Inspect symbols at runtime, so these tests run against duck-typed
stub suites and the tiny hub-free models.
"""
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.core.specs import ControlSpec
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.evaluation.runner import SteeringEval
from tests.conftest import MockInputControl, MockStateControl

SUITE_CALLS: list[dict] = []


@dataclass(frozen=True, slots=True)
class _StubSuite:
    """Duck-typed suite recording every `run()` call into the module-level `SUITE_CALLS`."""

    name: str = "capability"
    tasks: tuple[str, ...] = ("stub/task",)
    generate_overrides: Mapping[str, Any] = field(default_factory=dict)
    fail: bool = field(default=False)

    def run(self, pipeline, *, log_dir, options=None, base_seed=None,
            model_name="steering-pipeline", generate_defaults=None, display="none") -> dict:
        SUITE_CALLS.append({
            "suite": self.name,
            "pipeline": pipeline,
            "log_dir": str(log_dir),
            "options": options,
            "base_seed": base_seed,
            "model_name": model_name,
            "generate_defaults": generate_defaults,
            "display": display,
        })
        if self.fail:
            raise RuntimeError("suite exploded")
        return {
            "stub/task": {
                "metrics": {"match/accuracy": 0.5, "match/stderr": 0.1},
                "n": 4,
                "log": "one.eval",
            }
        }


class _RaisingStateControl(MockStateControl):
    def requirements(self) -> Requirements:
        return Requirements(generate=needs(Capability.INTERVENTION_SPECS))


class _CleanupControl(MockInputControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


@pytest.fixture(autouse=True)
def reset_suite_calls():
    SUITE_CALLS.clear()


@pytest.fixture
def tiny_base(monkeypatch):
    """Serve the shared base from tiny hub-free models."""
    from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

    def fake_ensure(self):
        if self._base_model is None:
            self._base_model = tiny_llama()
            tokenizer = wordlevel_tokenizer()
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            self._base_tokenizer = tokenizer

    monkeypatch.setattr(PipelineFactory, "_ensure_base_model", fake_ensure)


def _runner(pipelines, suites=None, **kwargs) -> SteeringEval:
    kwargs.setdefault("progress", False)
    return SteeringEval(
        pipelines, "test-model", suites if suites is not None else [_StubSuite()], **kwargs,
    )


class TestRunShapeAndResults:
    def test_run_shape_and_results_frame(self, tiny_base, tmp_path):
        runner = _runner(
            {"baseline": [], "steered": [MockInputControl()]},
            num_trials=2, seed=7, save_dir=tmp_path,
        )
        results = runner.run()

        assert set(results) == {"baseline", "steered"}
        assert [run["trial_id"] for run in results["baseline"]] == [0, 1]
        baseline_run = results["baseline"][0]
        assert baseline_run["config_id"] == "baseline"
        assert baseline_run["seed"] is not None
        assert baseline_run["params"] == {}
        assert baseline_run["suites"]["capability"]["stub/task"]["metrics"]["match/accuracy"] == 0.5
        assert baseline_run["provenance"]["backend"] == "huggingface"
        assert baseline_run["provenance"]["prompt_path"] == "messages"

        frame = runner.results()
        assert list(frame.columns) == [
            "config", "config_id", "trial", "seed", "suite", "task", "scorer", "metric",
            "value", "n", "log",
        ]
        assert len(frame) == 2 * 2 * 2  # configs x trials x metrics
        assert set(frame["metric"]) == {"accuracy", "stderr"}
        assert set(frame["scorer"]) == {"match"}

    def test_results_before_run_raises(self):
        with pytest.raises(RuntimeError, match="run\\(\\)"):
            _runner({"baseline": []}).results()

    def test_log_dir_layout_and_relative_log_paths(self, tiny_base, tmp_path):
        runner = _runner({"baseline": []}, num_trials=1, save_dir=tmp_path)
        results = runner.run()
        (call,) = SUITE_CALLS
        assert call["log_dir"] == str(tmp_path / "inspect_logs" / "baseline" / "trial_0" / "capability")
        assert results["baseline"][0]["suites"]["capability"]["stub/task"]["log"] == str(
            "inspect_logs/baseline/trial_0/capability/one.eval"
        )

    def test_one_provider_name_per_config_and_trial_seeds(self, tiny_base, tmp_path):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        runner = _runner({"sweep": [spec]}, num_trials=2, seed=3, save_dir=tmp_path)
        results = runner.run()
        config_ids = [run["config_id"] for run in results["sweep"]]
        assert len(SUITE_CALLS) == 4
        assert [call["model_name"] for call in SUITE_CALLS] == config_ids
        seeds = {(call["model_name"], call["base_seed"]) for call in SUITE_CALLS}
        assert len(seeds) == 4  # distinct per (config, trial)

    def test_sequential_execution_one_pipeline_at_a_time(self, tiny_base, tmp_path):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        runner = _runner({"sweep": [spec]}, save_dir=tmp_path)
        runner.run()
        first, second = SUITE_CALLS
        assert first["pipeline"] is not second["pipeline"]
        # the first configuration's pipeline was released before the second ran
        assert first["pipeline"] != second["pipeline"]

    def test_temp_dir_when_no_save_dir(self, tiny_base):
        runner = _runner({"baseline": []})
        results = runner.run()
        assert "inspect_logs" in SUITE_CALLS[0]["log_dir"]
        assert results["baseline"][0]["suites"]["capability"]["stub/task"]["n"] == 4


class TestProgressAndDisplay:
    def test_display_forwarded_to_suites(self, tiny_base, tmp_path):
        _runner({"baseline": []}, save_dir=tmp_path, display="plain").run()
        assert all(call["display"] == "plain" for call in SUITE_CALLS)

    def test_display_defaults_to_none(self, tiny_base, tmp_path):
        _runner({"baseline": []}, save_dir=tmp_path).run()
        assert all(call["display"] == "none" for call in SUITE_CALLS)

    def test_progress_summary_and_cell_lines_logged(self, tiny_base, tmp_path, caplog):
        with caplog.at_level("INFO", logger="steerability.evaluation.runner"):
            _runner(
                {"baseline": [], "steered": [MockInputControl()]},
                num_trials=2, save_dir=tmp_path,
            ).run()
        messages = [record.message for record in caplog.records]
        assert any("= 4 cell(s)" in message for message in messages)  # 2 configs x 2 trials x 1 suite
        assert any("Cell 1/4" in message for message in messages)


class TestResume:
    def test_same_identity_resumes(self, tiny_base, tmp_path):
        _runner({"baseline": []}, save_dir=tmp_path).run()
        _runner({"baseline": []}, save_dir=tmp_path).run()  # no refusal

    def test_raised_num_trials_resumes(self, tiny_base, tmp_path):
        _runner({"baseline": []}, num_trials=1, save_dir=tmp_path).run()
        _runner({"baseline": []}, num_trials=2, save_dir=tmp_path).run()  # completes only the missing trial

    def test_missing_save_dir_is_created(self, tiny_base, tmp_path):
        # the runner creates save_dir; eval_set (stubbed out here) creates the inspect_logs subtree
        save_dir = tmp_path / "fresh"
        assert not save_dir.exists()
        _runner({"baseline": []}, save_dir=save_dir).run()
        assert save_dir.is_dir()


class TestPreflightPolicy:
    def test_raise_aggregates_before_any_work(self, tiny_base, tmp_path):
        spec = ControlSpec(control_cls=_RaisingStateControl, vars={"scale_factor": [0.5, 1.0]})
        runner = _runner({"sweep": [spec]}, save_dir=tmp_path)
        with pytest.raises(RuntimeError, match="_RaisingStateControl"):
            runner.run()
        assert SUITE_CALLS == []

    def test_skip_runs_supported_points_only(self, tiny_base, tmp_path, caplog):
        runner = _runner(
            {"good": [MockInputControl()], "gated": [_RaisingStateControl()]},
            on_unsupported="skip", save_dir=tmp_path,
        )
        with caplog.at_level("WARNING", logger="steerability.evaluation.runner"):
            results = runner.run()
        assert len(results["good"]) == 1
        assert results["gated"] == []
        assert any("Skipping unsupported configuration" in r.getMessage() for r in caplog.records)


class TestFactoryDiscipline:
    def test_cleanup_runs_when_a_suite_raises(self, tiny_base, tmp_path):
        control = _CleanupControl()
        runner = _runner({"arm": [control]}, suites=[_StubSuite(fail=True)], save_dir=tmp_path)
        with pytest.raises(RuntimeError, match="suite exploded"):
            runner.run()
        assert control.cleanup_calls == 1


class TestSuiteFailure:
    def test_eval_set_failure_propagates_and_records_no_cell(self, tiny_base, tmp_path, monkeypatch):
        pytest.importorskip("inspect_ai")
        import steerability.evaluation.suite as suite_module
        from steerability.evaluation.suite import InspectSuite

        monkeypatch.setattr(suite_module, "eval_set", lambda tasks, **kwargs: (False, []))
        suite = InspectSuite(name="capability", tasks=("stub/task",))
        runner = _runner({"baseline": []}, suites=[suite], save_dir=tmp_path)
        with pytest.raises(RuntimeError, match="eval_set failed"):
            runner.run()
        with pytest.raises(RuntimeError, match="run\\(\\)"):
            runner.results()


class TestSeedWithoutTemperature:
    def test_seed_without_temperature_warns(self, tiny_base, tmp_path):
        runner = _runner({"baseline": []}, seed=7, save_dir=tmp_path)
        with pytest.warns(UserWarning, match="temperature"):
            runner.run()

    def test_seed_with_greedy_default_does_not_warn(self, tiny_base, tmp_path):
        runner = _runner({"baseline": []}, seed=7, save_dir=tmp_path, generate_defaults={"temperature": 0})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run()
        assert not any("temperature" in str(w.message) for w in caught)

    def test_seed_with_suite_override_does_not_warn(self, tiny_base, tmp_path):
        suite = _StubSuite(generate_overrides={"temperature": 0.7})
        runner = _runner({"baseline": []}, suites=[suite], seed=7, save_dir=tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run()
        assert not any("temperature" in str(w.message) for w in caught)


class TestStaticRuntimeKwargAudit:
    def test_static_runtime_kwarg_declared_by_no_configuration_warns(self, tiny_base, tmp_path):
        pytest.importorskip("inspect_ai")
        from steerability.evaluation.provider import ProviderOptions

        runner = _runner(
            {"baseline": [], "steered": [MockInputControl()]},
            save_dir=tmp_path,
            provider_options=ProviderOptions(runtime_kwargs={"substrings": ["x"]}),
        )
        with pytest.warns(UserWarning, match="inert on every arm"):
            runner.run()

    def test_static_runtime_kwarg_declared_by_one_configuration_does_not_warn(self, tiny_base, tmp_path):
        pytest.importorskip("inspect_ai")
        from steerability.evaluation.provider import ProviderOptions

        class _RowConsumer(MockInputControl):
            RUNTIME_KWARGS_SCHEMA = [{"name": "substrings", "type": "list[str]", "scope": "row"}]

        runner = _runner(
            {"baseline": [], "steered": [_RowConsumer()]},
            save_dir=tmp_path,
            provider_options=ProviderOptions(runtime_kwargs={"substrings": ["x"]}),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run()
        assert not any("inert on every arm" in str(w.message) for w in caught)


class TestConstructorValidation:
    def test_bad_arguments_raise(self):
        with pytest.raises(TypeError, match="pipelines"):
            SteeringEval([], "m", [_StubSuite()])
        with pytest.raises(ValueError, match="num_trials"):
            _runner({"baseline": []}, num_trials=0)
        with pytest.raises(ValueError, match="suites"):
            _runner({"baseline": []}, suites=[])
        with pytest.raises(ValueError, match="distinct"):
            _runner({"baseline": []}, suites=[_StubSuite(), _StubSuite()])
        with pytest.raises(ValueError, match="on_unsupported"):
            _runner({"baseline": []}, on_unsupported="ignore")
