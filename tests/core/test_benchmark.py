"""
Tests for the `Benchmark` runner and `ControlSpec`.

Tests cover:

- Benchmark initialization and defaults
- Pipeline dispatch: baseline, fixed controls, structural controls, and `ControlSpec` sweeps
- Trial loops and the structure of run dictionaries
- Checkpointing: incremental saves, resume-and-skip, corrupted checkpoints, and interrupted
  sweeps
- Export and control cleanup
- `ControlSpec.iter_points` and `ControlSpec.resolve_params`

Model loading is replaced at the Hugging Face boundary (`Benchmark._ensure_base_model` and
the loader classes used by `SteeringPipeline`); the benchmark, pipeline, and spec logic under
test are the package implementations.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from aisteer360.algorithms.core.execution.contracts import Capability, Requirements, needs
from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.evaluation.benchmark import _IDENTITY_META_FIELDS, Benchmark, UnsupportedBenchmarkError
from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.identity import derive_trial_seed
from tests.conftest import (
    MockAccuracyMetric,
    MockInputControl,
    MockScoreMetric,
    MockStateControl,
    MockStructuralControl,
    MockUseCase,
    create_mock_model,
    create_mock_tokenizer,
)


@pytest.fixture
def mock_base_model(monkeypatch):
    """Replace `Benchmark._ensure_base_model` with a mock installer.

    Returns:
        A list receiving one entry per `_ensure_base_model` invocation, usable as a call
        counter.
    """
    calls = []

    def fake_ensure(self):
        calls.append(1)
        if self._base_model is None:
            self._base_model = create_mock_model()
            self._base_tokenizer = create_mock_tokenizer()

    monkeypatch.setattr(Benchmark, "_ensure_base_model", fake_ensure)
    return calls


@pytest.fixture
def patched_pipeline_loaders(monkeypatch):
    """Replace the Hugging Face loader classes used by `SteeringPipeline` with mocks.

    Returns:
        A tuple `(model_loader, tokenizer_loader, model, tokenizer)`.
    """
    model = create_mock_model()
    model.to.return_value = model
    tokenizer = create_mock_tokenizer()

    model_loader = MagicMock()
    model_loader.from_pretrained.return_value = model
    tokenizer_loader = MagicMock()
    tokenizer_loader.from_pretrained.return_value = tokenizer

    monkeypatch.setattr(
        "aisteer360.algorithms.core.steering_pipeline.AutoModelForCausalLM", model_loader
    )
    monkeypatch.setattr(
        "aisteer360.algorithms.core.steering_pipeline.AutoTokenizer", tokenizer_loader
    )
    return model_loader, tokenizer_loader, model, tokenizer


def _make_use_case(evaluation_data) -> MockUseCase:
    return MockUseCase(
        evaluation_data=evaluation_data,
        evaluation_metrics=[MockAccuracyMetric(), MockScoreMetric()],
    )


# Benchmark Initialization Tests
class TestBenchmarkInitialization:
    """Tests for `Benchmark` construction."""

    def test_basic_initialization(self, sample_evaluation_data):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        assert benchmark.use_case is use_case
        assert benchmark.base_model_name_or_path == "test-model"
        assert benchmark.num_trials == 1
        assert benchmark.gen_kwargs == {}
        assert benchmark.hf_model_kwargs == {}
        assert benchmark.runtime_overrides is None
        assert benchmark.save_dir is None

    def test_num_trials_coerced_to_int(self, sample_evaluation_data):
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=2.0,
        )
        assert benchmark.num_trials == 2
        assert isinstance(benchmark.num_trials, int)

    def test_save_dir_coerced_to_path(self, sample_evaluation_data, tmp_path):
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=str(tmp_path),
        )
        assert isinstance(benchmark.save_dir, Path)
        assert benchmark.save_dir == tmp_path

    def test_multiple_pipelines_preserved(self, sample_evaluation_data):
        pipelines = {
            "baseline": [],
            "steered": [MockInputControl()],
        }
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines=pipelines,
        )
        assert set(benchmark.steering_pipelines.keys()) == {"baseline", "steered"}


class TestBenchmarkConstructorValidation:
    """The constructor rejects malformed arguments before any run."""

    def test_non_use_case_rejected(self, sample_evaluation_data):
        with pytest.raises(TypeError, match="use_case must be a UseCase"):
            Benchmark(
                use_case=object(),
                base_model_name_or_path="test-model",
                steering_pipelines={"baseline": []},
            )

    def test_non_dict_steering_pipelines_rejected(self, sample_evaluation_data):
        with pytest.raises(TypeError, match="steering_pipelines must be a dict"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines=[],
            )

    def test_non_list_pipeline_value_rejected(self, sample_evaluation_data):
        with pytest.raises(TypeError, match="must be a list, tuple, or None"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines={"bad": MockInputControl()},
            )

    def test_negative_num_trials_rejected(self, sample_evaluation_data):
        with pytest.raises(ValueError, match="num_trials must be >= 0"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines={"baseline": []},
                num_trials=-1,
            )

    def test_zero_batch_size_rejected(self, sample_evaluation_data):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines={"baseline": []},
                batch_size=0,
            )


# Benchmark Run Tests
class TestBenchmarkRunBaseline:
    """Tests for baseline (unsteered) pipelines."""

    def test_baseline_uses_shared_base_model(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        assert mock_base_model == [1]
        assert len(use_case._generate_calls) == 1
        call = use_case._generate_calls[0]
        # the baseline now runs through an empty SteeringPipeline sharing the base model
        pipeline = call["model_or_pipeline"]
        assert isinstance(pipeline, SteeringPipeline)
        assert pipeline.model is benchmark._base_model
        assert pipeline.input_controls == []
        assert pipeline.state_controls == []
        assert pipeline.output_controls == []
        assert call["tokenizer"] is benchmark._base_tokenizer
        assert profiles["baseline"][0]["params"] == {}

    def test_none_pipeline_treated_as_baseline(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": None},
        )

        profiles = benchmark.run()

        assert len(profiles["baseline"]) == 1
        pipeline = use_case._generate_calls[0]["model_or_pipeline"]
        assert isinstance(pipeline, SteeringPipeline)
        assert pipeline.model is benchmark._base_model

    def test_multiple_trials(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=3,
        )

        profiles = benchmark.run()

        assert len(use_case._generate_calls) == 3
        assert [run["trial_id"] for run in profiles["baseline"]] == [0, 1, 2]

    def test_zero_trials_yields_no_runs(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=0,
        )

        profiles = benchmark.run()

        assert profiles["baseline"] == []
        assert use_case._generate_calls == []


class TestBenchmarkRunControls:
    """Tests for pipelines with concrete controls."""

    def test_input_control_pipeline_shares_base_model(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        control = MockInputControl(num_examples=2)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [control]},
        )

        benchmark.run()

        pipeline = use_case._generate_calls[0]["model_or_pipeline"]
        assert isinstance(pipeline, SteeringPipeline)
        assert pipeline._is_steered
        assert pipeline.model is benchmark._base_model
        assert pipeline.input_controls == [control]

    def test_state_control_pipeline(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        control = MockStateControl(target_layers=[0])
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [control]},
        )

        benchmark.run()

        pipeline = use_case._generate_calls[0]["model_or_pipeline"]
        assert pipeline.state_controls == [control]
        assert control.model is benchmark._base_model

    def test_structural_control_loads_fresh_pipeline(
            self, sample_evaluation_data, mock_base_model, patched_pipeline_loaders
    ):
        """A structural control builds a `SteeringPipeline` from the base checkpoint rather
        than reusing the shared base model."""
        model_loader, _, loaded_model, _ = patched_pipeline_loaders
        use_case = _make_use_case(sample_evaluation_data)
        control = MockStructuralControl()
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"structural": [control]},
        )

        benchmark.run()

        model_loader.from_pretrained.assert_called_once()
        assert model_loader.from_pretrained.call_args.args == ("test-model",)

        pipeline = use_case._generate_calls[0]["model_or_pipeline"]
        assert isinstance(pipeline, SteeringPipeline)
        assert pipeline._is_steered
        assert pipeline.model is loaded_model
        assert pipeline.model is not benchmark._base_model
        assert control._steer_called

    def test_run_dict_structure(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl()]},
        )

        profiles = benchmark.run()

        run = profiles["steered"][0]
        assert set(run.keys()) == {
            "trial_id", "generations", "evaluations", "params", "config_id", "seed", "provenance"
        }
        assert run["trial_id"] == 0
        assert len(run["generations"]) == len(sample_evaluation_data)

    def test_evaluations_keyed_by_metric_name(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        evaluations = profiles["baseline"][0]["evaluations"]
        assert set(evaluations.keys()) == {"MockAccuracyMetric", "MockScoreMetric"}
        assert "accuracy" in evaluations["MockAccuracyMetric"]
        assert evaluations["MockScoreMetric"] == {"score": 0.5}

    def test_generation_arguments_passed_through(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        gen_kwargs = {"max_new_tokens": 7}
        runtime_overrides = {"MockInputControl": {"key": "value"}}
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            batch_size=4,
        )

        benchmark.run()

        call = use_case._generate_calls[0]
        assert call["gen_kwargs"] == gen_kwargs
        assert call["runtime_overrides"] == runtime_overrides
        assert call["kwargs"]["batch_size"] == 4


# ControlSpec Sweep Tests
class TestBenchmarkSpecs:
    """Tests for pipelines defined by `ControlSpec` sweeps."""

    def test_single_spec_grid(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        profiles = benchmark.run()

        params = [run["params"] for run in profiles["sweep"]]
        assert params == [
            {"MockInputControl": {"num_examples": 1}},
            {"MockInputControl": {"num_examples": 2}},
        ]
        assert len(use_case._generate_calls) == 2

    def test_spec_name_keys_params(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(
            control_cls=MockInputControl, vars={"num_examples": [3]}, name="few_shot"
        )
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        profiles = benchmark.run()

        assert profiles["sweep"][0]["params"] == {"few_shot": {"num_examples": 3}}

    def test_fixed_params_merged_into_points(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "p_"},
            vars={"num_examples": [1]},
        )
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        profiles = benchmark.run()

        assert profiles["sweep"][0]["params"] == {
            "MockInputControl": {"prefix": "p_", "num_examples": 1}
        }

    def test_multiple_specs_cartesian_product(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        input_spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        state_spec = ControlSpec(control_cls=MockStateControl, vars={"scale_factor": [0.5, 1.0]})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [input_spec, state_spec]},
        )

        profiles = benchmark.run()

        assert len(profiles["sweep"]) == 4
        for run in profiles["sweep"]:
            assert set(run["params"].keys()) == {"MockInputControl", "MockStateControl"}

    def test_spec_without_vars_runs_once(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(control_cls=MockInputControl, params={"num_examples": 5})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        profiles = benchmark.run()

        assert len(profiles["sweep"]) == 1
        assert profiles["sweep"][0]["params"] == {"MockInputControl": {"num_examples": 5}}

    def test_trials_run_per_configuration(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
            num_trials=2,
        )

        profiles = benchmark.run()

        assert len(profiles["sweep"]) == 4
        assert [run["trial_id"] for run in profiles["sweep"]] == [0, 1, 0, 1]

    def test_mixed_spec_and_concrete_raises(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "mixed": [
                    ControlSpec(control_cls=MockInputControl),
                    MockStateControl(),
                ]
            },
        )

        with pytest.raises(TypeError, match="mixes ControlSpec"):
            benchmark.run()


# Checkpointing Tests
class TestBenchmarkCheckpointing:
    """Tests for incremental checkpointing and resume."""

    def test_checkpoint_converts_nested_numpy_objects(self, sample_evaluation_data, tmp_path):
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )
        profiles = {
            "baseline": [
                {
                    "params": np.array(
                        [Path("models/base"), {"layers": {1, 2}}],
                        dtype=object,
                    )
                }
            ]
        }

        benchmark._save_checkpoint(profiles)

        with (tmp_path / "checkpoint.json").open(encoding="utf-8") as handle:
            saved = json.load(handle)
        saved_params = saved["profiles"]["baseline"][0]["params"]
        assert saved_params[0] == "models/base"
        assert sorted(saved_params[1]["layers"]) == [1, 2]

    def test_checkpoint_written(self, sample_evaluation_data, mock_base_model, tmp_path):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        profiles = benchmark.run()

        checkpoint_path = tmp_path / "checkpoint.json"
        assert checkpoint_path.exists()
        with open(checkpoint_path) as f:
            saved = json.load(f)
        assert "version" not in saved
        assert saved["meta"]["format"] == 3
        assert set(_IDENTITY_META_FIELDS) <= set(saved["meta"].keys())
        assert set(saved["profiles"]) == {"baseline"}
        assert len(saved["profiles"]["baseline"]) == len(profiles["baseline"])
        assert saved["profiles"]["baseline"][0]["params"] == {}
        assert saved["profiles"]["baseline"][0]["config_id"] == "baseline"

    def test_resume_skips_completed_configurations(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        first_use_case = _make_use_case(sample_evaluation_data)
        Benchmark(
            use_case=first_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        ).run()
        assert mock_base_model == [1]

        second_use_case = _make_use_case(sample_evaluation_data)
        profiles = Benchmark(
            use_case=second_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        ).run()

        assert second_use_case._generate_calls == []
        assert mock_base_model == [1]  # the fast path skips model loading entirely
        assert len(profiles["baseline"]) == 1

    def test_corrupted_checkpoint_starts_fresh(
            self, sample_evaluation_data, mock_base_model, tmp_path, caplog
    ):
        (tmp_path / "checkpoint.json").write_text("{not valid json")
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            profiles = benchmark.run()

        assert any("Could not read checkpoint" in r.getMessage() for r in caplog.records)
        assert len(use_case._generate_calls) == 1
        assert len(profiles["baseline"]) == 1

    def test_interrupted_sweep_resumes_remaining_configurations(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        """A sweep that fails mid-way keeps its completed configurations; a subsequent run
        executes only the remaining ones."""

        class _FailOnSecondGenerate(MockUseCase):
            def generate(self, *args, **kwargs):
                if len(self._generate_calls) >= 1:
                    raise RuntimeError("interrupted")
                return super().generate(*args, **kwargs)

        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        failing_use_case = _FailOnSecondGenerate(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        interrupted = Benchmark(
            use_case=failing_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
            save_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="interrupted"):
            interrupted.run()

        with open(tmp_path / "checkpoint.json") as f:
            partial = json.load(f)
        assert len(partial["profiles"]["sweep"]) == 1
        assert partial["profiles"]["sweep"][0]["params"] == {"MockInputControl": {"num_examples": 1}}

        # the resumed run reuses the same use-case class so the checkpoint's identity meta matches;
        # a fresh instance succeeds on its single (remaining) generate call
        resumed_use_case = _FailOnSecondGenerate(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        profiles = Benchmark(
            use_case=resumed_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [ControlSpec(control_cls=MockInputControl,
                                                      vars={"num_examples": [1, 2]})]},
            save_dir=tmp_path,
        ).run()

        assert len(resumed_use_case._generate_calls) == 1
        assert len(profiles["sweep"]) == 2
        assert [run["params"]["MockInputControl"]["num_examples"] for run in profiles["sweep"]] == [1, 2]


# Export and Cleanup Tests
class TestBenchmarkExportAndCleanup:
    """Tests for use-case export and control cleanup."""

    def test_export_writes_profiles(self, sample_evaluation_data, mock_base_model, tmp_path):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        benchmark.run()

        profiles_path = tmp_path / "profiles.json"
        assert profiles_path.exists()
        with open(profiles_path) as f:
            exported = json.load(f)
        assert "baseline" in exported

    def test_export_failure_is_swallowed(
            self, sample_evaluation_data, mock_base_model, tmp_path, caplog
    ):
        """A failing `export()` does not abort the run and leaves the checkpoint intact."""

        class _FailingExportUseCase(MockUseCase):
            def export(self, profiles, save_dir):
                raise RuntimeError("export failed")

        use_case = _FailingExportUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            profiles = benchmark.run()

        assert any("Incremental export failed" in r.getMessage() for r in caplog.records)
        assert len(profiles["baseline"]) == 1
        assert (tmp_path / "checkpoint.json").exists()

    def test_control_cleanup_called_after_run(self, sample_evaluation_data, mock_base_model):
        class _CleanupInputControl(MockInputControl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cleaned = False

            def cleanup(self):
                self._cleaned = True

        control = _CleanupInputControl()
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [control]},
        )

        benchmark.run()

        assert control._cleaned


# Shared-base fingerprint guard, structural isolation, and default export
class _MutatingStateControl(MockStateControl):
    """State control whose `steer` perturbs a shared-model parameter in place."""

    def steer(self, model, tokenizer=None, **kwargs):
        super().steer(model, tokenizer=tokenizer, **kwargs)
        with torch.no_grad():
            first_param = next(model.parameters())
            first_param.add_(1.0)
        return model


@pytest.fixture
def fingerprintable_base(monkeypatch):
    """Patch `_ensure_base_model` to install a real tiny model and record its fingerprint.

    Returns:
        A dict with `"invocations"` (one entry per `_ensure_base_model` call) and `"loads"` (one entry
        per actual model load). A reload after a dropped base shows as a second `"loads"` entry.
    """
    from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
    from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

    record = {"invocations": [], "loads": []}

    def fake_ensure(self):
        record["invocations"].append(1)
        if self._base_model is None:
            self._base_model = tiny_llama()
            tokenizer = wordlevel_tokenizer()
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            self._base_tokenizer = tokenizer
            self._base_fingerprint = model_fingerprint(self._base_model)
            record["loads"].append(1)

    monkeypatch.setattr(Benchmark, "_ensure_base_model", fake_ensure)
    return record


class TestSharedBaseFingerprintGuard:
    """The tripwire detects shared-base mutation, warns naming the control, and reloads a clean base."""

    def test_mutating_sweep_warns_and_reloads(self, sample_evaluation_data, fingerprintable_base, caplog):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(control_cls=_MutatingStateControl, vars={"scale_factor": [0.5, 1.0]})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            benchmark.run()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Shared base model changed" in m and "_MutatingStateControl" in m for m in messages)
        # the mutated base is dropped after the first config, so the second config reloads a clean base
        assert fingerprintable_base["loads"] == [1, 1]

    def test_clean_sweep_does_not_warn(self, sample_evaluation_data, fingerprintable_base, caplog):
        use_case = _make_use_case(sample_evaluation_data)
        spec = ControlSpec(control_cls=MockStateControl, vars={"scale_factor": [0.5, 1.0]})
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            benchmark.run()

        assert not any("Shared base model changed" in r.getMessage() for r in caplog.records)
        assert fingerprintable_base["loads"] == [1]  # the clean base is loaded once and reused across configs


class TestStructuralIsolation:
    """Structural-only pipelines load their own model and never touch the shared base."""

    def test_structural_only_never_loads_shared_base(
            self, sample_evaluation_data, mock_base_model, patched_pipeline_loaders
    ):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"structural": [MockStructuralControl()]},
        )

        benchmark.run()

        assert mock_base_model == []  # _ensure_base_model never called
        assert benchmark._base_model is None


class TestBenchmarkDefaultExport:
    """A use case that does not override `export` gets the benchmark's default `profiles.json`."""

    def test_default_export_writes_profiles_json(self, sample_evaluation_data, mock_base_model, tmp_path):
        class _NoExportUseCase(MockUseCase):
            pass

        _NoExportUseCase.export = UseCase.export  # ensure no override is inherited from MockUseCase

        use_case = _NoExportUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        benchmark.run()

        profiles_path = tmp_path / "profiles.json"
        assert profiles_path.exists()
        with open(profiles_path) as f:
            exported = json.load(f)
        assert "baseline" in exported


# Use Case Data Handling Tests
class TestUseCaseDataHandling:
    """Tests for evaluation-data handling in the use case."""

    def test_empty_evaluation_data_warns(self):
        with pytest.warns(UserWarning):
            MockUseCase(evaluation_data=[], evaluation_metrics=[MockAccuracyMetric()])

    def test_num_samples_limits_data(self, large_evaluation_data):
        use_case = MockUseCase(
            evaluation_data=large_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
            num_samples=5,
        )
        assert len(use_case.evaluation_data) == 5


# ControlSpec Tests
class TestControlSpecIterPoints:
    """Tests for `ControlSpec.iter_points`."""

    def test_no_vars_yields_single_empty_point(self):
        spec = ControlSpec(control_cls=MockInputControl)
        assert list(spec.iter_points({})) == [{}]

    def test_grid_cartesian_product_order(self):
        spec = ControlSpec(
            control_cls=MockInputControl,
            vars={"a": [1, 2], "b": [10, 20]},
        )
        points = list(spec.iter_points({}))
        assert points == [
            {"a": 1, "b": 10},
            {"a": 1, "b": 20},
            {"a": 2, "b": 10},
            {"a": 2, "b": 20},
        ]

    def test_empty_value_sequence_yields_no_points(self):
        spec = ControlSpec(control_cls=MockInputControl, vars={"a": []})
        assert list(spec.iter_points({})) == []

    def test_sequence_vars_passthrough(self):
        param_dicts = [{"a": 1}, {"a": 2, "b": 3}]
        spec = ControlSpec(control_cls=MockInputControl, vars=param_dicts)
        points = list(spec.iter_points({}))
        assert points == param_dicts
        assert points[0] is not param_dicts[0]  # yielded dicts are copies

    def test_callable_vars_receives_context(self):
        spec = ControlSpec(
            control_cls=MockInputControl,
            vars=lambda context: [{"got": context["pipeline_name"]}],
        )
        points = list(spec.iter_points({"pipeline_name": "sweep"}))
        assert points == [{"got": "sweep"}]

    def test_random_mapping_is_seed_deterministic_subset(self):
        spec = ControlSpec(
            control_cls=MockInputControl,
            vars={"a": [1, 2, 3], "b": [10, 20, 30]},
            search_strategy="random",
            num_samples=4,
            seed=7,
        )
        first = list(spec.iter_points({}))
        second = list(spec.iter_points({}))

        assert first == second
        assert len(first) == 4
        grid = [{"a": a, "b": b} for a in [1, 2, 3] for b in [10, 20, 30]]
        assert all(point in grid for point in first)

    def test_random_num_samples_at_least_grid_size_yields_full_grid(self):
        spec = ControlSpec(
            control_cls=MockInputControl,
            vars={"a": [1, 2]},
            search_strategy="random",
            num_samples=10,
            seed=0,
        )
        assert list(spec.iter_points({})) == [{"a": 1}, {"a": 2}]

    def test_random_sequence_subset(self):
        param_dicts = [{"a": i} for i in range(5)]
        spec = ControlSpec(
            control_cls=MockInputControl,
            vars=param_dicts,
            search_strategy="random",
            num_samples=2,
            seed=3,
        )
        points = list(spec.iter_points({}))
        assert len(points) == 2
        assert all(point in param_dicts for point in points)


class TestControlSpecResolveParams:
    """Tests for `ControlSpec.resolve_params`."""

    def test_merges_fixed_params_and_chosen(self):
        spec = ControlSpec(control_cls=MockInputControl, params={"prefix": "p_"})
        resolved = spec.resolve_params(chosen={"num_examples": 2}, context={})
        assert resolved == {"prefix": "p_", "num_examples": 2}

    def test_chosen_overrides_fixed_params(self):
        spec = ControlSpec(control_cls=MockInputControl, params={"num_examples": 1})
        resolved = spec.resolve_params(chosen={"num_examples": 9}, context={})
        assert resolved == {"num_examples": 9}

    def test_original_params_unmodified(self):
        params = {"num_examples": 1}
        spec = ControlSpec(control_cls=MockInputControl, params=params)
        spec.resolve_params(chosen={"prefix": "x"}, context={})
        assert params == {"num_examples": 1}

    def test_callable_param_values_receive_search_params(self):
        """Callable fixed-param values are resolved with a context that includes the chosen
        search point under `"search_params"`."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": lambda context: f"n{context['search_params']['num_examples']}_"},
        )
        resolved = spec.resolve_params(chosen={"num_examples": 4}, context={"pipeline_name": "p"})
        assert resolved == {"prefix": "n4_", "num_examples": 4}


# Provenance and versioned-envelope tests
class TestCheckpointEnvelope:
    """The checkpoint is a versioned envelope; identity-mismatch refuses, other files are overwritten."""

    def test_run_dicts_carry_provenance_fields(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl()]},
            seed=11,
        )

        run = benchmark.run()["steered"][0]
        assert run["config_id"] != "baseline"
        assert run["seed"] == derive_trial_seed(11, run["config_id"], 0)
        assert set(run["provenance"]) == {"backend", "model_fingerprint", "toolkit_version"}
        assert run["provenance"]["backend"] == "huggingface"

    def test_non_envelope_file_is_ignored_and_overwritten(
            self, sample_evaluation_data, mock_base_model, tmp_path, caplog
    ):
        # a bare profiles dict, the old format's shape
        (tmp_path / "checkpoint.json").write_text(json.dumps({"baseline": [{"trial_id": 0}]}))
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            profiles = benchmark.run()

        assert any("not a checkpoint envelope" in r.getMessage() for r in caplog.records)
        assert len(use_case._generate_calls) == 1  # ran fresh, not resumed
        assert len(profiles["baseline"]) == 1
        with open(tmp_path / "checkpoint.json") as f:
            rewritten = json.load(f)
        assert rewritten["meta"]["format"] == 3  # the old content is gone
        assert set(rewritten["profiles"]) == {"baseline"}

    def test_prior_format_envelope_refuses_naming_format(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        # a well-shaped envelope from an earlier checkpoint format refuses loudly, so runs the
        # user may want to finish on the old toolkit version are preserved
        (tmp_path / "checkpoint.json").write_text(json.dumps({
            "version": 2,
            "meta": {"model": "test-model", "backend": {"kind": "huggingface"}},
            "profiles": {"baseline": [{"trial_id": 0}]},
        }))
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="format was None, now 3"):
            benchmark.run()
        with open(tmp_path / "checkpoint.json") as f:
            preserved = json.load(f)
        assert preserved["profiles"] == {"baseline": [{"trial_id": 0}]}  # nothing overwritten

    @pytest.mark.parametrize("field", _IDENTITY_META_FIELDS)
    def test_identity_mismatch_refuses_naming_field(
            self, sample_evaluation_data, mock_base_model, tmp_path, field
    ):
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )
        benchmark.run()

        with open(tmp_path / "checkpoint.json") as f:
            envelope = json.load(f)
        envelope["meta"][field] = "mutated-identity-value"
        (tmp_path / "checkpoint.json").write_text(json.dumps(envelope))

        resumed = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
        )
        with pytest.raises(ValueError, match=field):
            resumed.run()

    def test_chat_template_kwargs_changes_gen_kwargs_digest(
            self, sample_evaluation_data, mock_base_model
    ):
        # two gen_kwargs differing only in chat_template_kwargs get distinct checkpoint identities
        thinking_off = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            gen_kwargs={"max_new_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}},
        )
        thinking_on = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            gen_kwargs={"max_new_tokens": 8, "chat_template_kwargs": {"enable_thinking": True}},
        )
        off_digest = thinking_off._checkpoint_meta()["gen_kwargs_digest"]
        on_digest = thinking_on._checkpoint_meta()["gen_kwargs_digest"]
        assert off_digest != on_digest

    def test_checkpoint_every_trial_grows_on_disk_per_trial(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        counts = []

        class _RecordingUseCase(MockUseCase):
            def generate(self, *args, **kwargs):
                result = super().generate(*args, **kwargs)
                path = tmp_path / "checkpoint.json"
                if path.exists():
                    with open(path) as f:
                        counts.append(len(json.load(f)["profiles"].get("baseline", [])))
                else:
                    counts.append(0)  # first trial runs before any save
                return result

        use_case = _RecordingUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=3,
            checkpoint_every="trial",
        ).run()

        # each generate observes the file before its own trial was recorded: 0, 1, 2
        assert counts == [0, 1, 2]
        with open(tmp_path / "checkpoint.json") as f:
            saved = json.load(f)
        assert len(saved["profiles"]["baseline"]) == 3

    def test_checkpoint_every_config_writes_once_per_config(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        counts = []

        class _RecordingUseCase(MockUseCase):
            def generate(self, *args, **kwargs):
                result = super().generate(*args, **kwargs)
                path = tmp_path / "checkpoint.json"
                if path.exists():
                    with open(path) as f:
                        counts.append(len(json.load(f)["profiles"].get("baseline", [])))
                else:
                    counts.append(-1)
                return result

        use_case = _RecordingUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )
        Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=3,
            checkpoint_every="config",
        ).run()

        # no per-trial write: every trial sees no file yet (config write happens after all trials)
        assert counts == [-1, -1, -1]


class TestTrialGranularResume:
    """Resume completes only missing trials; a complete config performs zero loads."""

    def test_raising_num_trials_runs_only_delta(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        first_use_case = _make_use_case(sample_evaluation_data)
        Benchmark(
            use_case=first_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=1,
        ).run()
        assert len(first_use_case._generate_calls) == 1

        second_use_case = _make_use_case(sample_evaluation_data)
        profiles = Benchmark(
            use_case=second_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=3,
        ).run()

        assert len(second_use_case._generate_calls) == 2  # only trials 1 and 2
        assert [run["trial_id"] for run in profiles["baseline"]] == [0, 1, 2]

    def test_complete_config_performs_zero_loads(
            self, sample_evaluation_data, mock_base_model, tmp_path
    ):
        Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=2,
        ).run()
        mock_base_model.clear()

        second_use_case = _make_use_case(sample_evaluation_data)
        Benchmark(
            use_case=second_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            save_dir=tmp_path,
            num_trials=2,
        ).run()

        assert second_use_case._generate_calls == []
        assert mock_base_model == []  # a complete config never loads the base

    def test_run_pipeline_return_matches_record_channel(
            self, sample_evaluation_data, mock_base_model
    ):
        recorded = []
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=2,
        )
        returned = benchmark._run_pipeline(
            [], specs=None, params=None, existing_runs=recorded, record=recorded.append,
        )
        assert [run["trial_id"] for run in returned] == [0, 1]
        assert returned == recorded  # the two-channel contract


class TestFixedPipelineIdentity:
    """Differently configured fixed controls under one name produce different config ids."""

    def test_distinct_config_ids_for_distinct_fixed_controls(self, sample_evaluation_data, mock_base_model):
        first = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl(num_examples=1)]},
        ).run()
        second = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl(num_examples=9)]},
        ).run()

        assert first["steered"][0]["config_id"] != second["steered"][0]["config_id"]


class TestSeededTrials:
    """A benchmark seed derives one seed per (config, trial), threaded into gen_kwargs and use-case kwargs."""

    def test_seed_recorded_and_threaded(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            seed=7,
            num_trials=2,
        )

        profiles = benchmark.run()
        config_id = profiles["baseline"][0]["config_id"]
        expected = [derive_trial_seed(7, config_id, t) for t in range(2)]
        assert [run["seed"] for run in profiles["baseline"]] == expected
        assert expected[0] != expected[1]
        for call, seed in zip(use_case._generate_calls, expected):
            assert call["gen_kwargs"]["seed"] == seed
            assert call["kwargs"]["trial_seed"] == seed

    def test_no_seed_injects_nothing(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        profiles = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        ).run()

        assert profiles["baseline"][0]["seed"] is None
        call = use_case._generate_calls[0]
        assert "seed" not in call["gen_kwargs"]
        assert "trial_seed" not in call["kwargs"]

    def test_seed_and_gen_kwargs_seed_conflict_raises(self, sample_evaluation_data):
        with pytest.raises(ValueError, match="not both"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines={"baseline": []},
                seed=1,
                gen_kwargs={"seed": 2},
            )

    def test_commonsense_shuffle_determinism(self, monkeypatch):
        from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_accuracy import MCQAAccuracy
        from aisteer360.evaluation.use_cases.commonsense_mcqa.use_case import CommonsenseMCQA

        recorded_prompts = []

        class _StubPipeline:
            supports_batching = True
            tokenizer = None

            def generate(self, *args, **kwargs):
                raise AssertionError("generation is stubbed at the batch layer")

        def fake_batch_retry_generate(prompt_data, **kwargs):
            recorded_prompts.append([row["reference_answer"] for row in prompt_data])
            n = len(prompt_data)
            return ["A"] * n, ["A"] * n, [None] * n, [None] * n

        monkeypatch.setattr(
            "aisteer360.evaluation.use_cases.commonsense_mcqa.use_case.batch_retry_generate",
            fake_batch_retry_generate,
        )

        data = [{"id": "q1", "question": "Q?", "answer": "4", "choices": ["4", "5", "6", "7"]}]
        use_case = CommonsenseMCQA(
            evaluation_data=data, evaluation_metrics=[MCQAAccuracy()], num_shuffling_runs=5,
        )

        use_case.generate(model_or_pipeline=_StubPipeline(), tokenizer=None, trial_seed=42)
        use_case.generate(model_or_pipeline=_StubPipeline(), tokenizer=None, trial_seed=42)
        use_case.generate(model_or_pipeline=_StubPipeline(), tokenizer=None, trial_seed=99)

        assert recorded_prompts[0] == recorded_prompts[1]  # same seed -> identical orderings
        assert recorded_prompts[0] != recorded_prompts[2]  # different seed -> different orderings


# Backend passthrough tests
class _RecordingPipeline:
    """Recording stand-in for `SteeringPipeline` used by the backend tests.

    Records the construction kwargs of every instance and provides the surface the benchmark
    touches: `check()` (always ok), `steer()`, `tokenizer`, and empty control-category lists for
    cleanup.
    """

    instances: list["_RecordingPipeline"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = object()
        self.tokenizer = object()
        self.device = None
        self.structural_controls = []
        self.input_controls = []
        self.state_controls = []
        self.output_controls = []
        self.release_calls = 0
        _RecordingPipeline.instances.append(self)

    def check(self):
        report = MagicMock()
        report.ok = True
        report.failures = ()
        return report

    def steer(self):
        self._is_steered = True

    def release_backends(self):
        self.release_calls += 1


@pytest.fixture
def recording_pipeline(monkeypatch):
    """Replace `SteeringPipeline` in the benchmark module with a recording stand-in.

    Returns:
        The list of constructed `_RecordingPipeline` instances (cleared per test).
    """
    _RecordingPipeline.instances = []
    monkeypatch.setattr("aisteer360.evaluation.benchmark.SteeringPipeline", _RecordingPipeline)
    return _RecordingPipeline.instances


class TestBackendPassthrough:
    """`backend`/`fit` are forwarded; non-HF kinds never load the shared base."""

    def test_vllm_backend_never_loads_shared_base_and_forwards_kind(
            self, sample_evaluation_data, mock_base_model, recording_pipeline
    ):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl()]},
            backend="vllm",
            fit="in_process",
        )

        benchmark.run()

        assert mock_base_model == []  # shared base never loaded on a non-HF backend kind
        # one probe pipeline (pre-flight) + one build pipeline
        assert len(recording_pipeline) == 2
        for instance in recording_pipeline:
            assert instance.kwargs["backend"] == "vllm"
            assert instance.kwargs["fit"] == "in_process"
            assert "lazy_init" not in instance.kwargs

    def test_unknown_backend_kind_raises_type_error(self, sample_evaluation_data):
        with pytest.raises(TypeError, match="backend must be a BackendSpec"):
            Benchmark(
                use_case=_make_use_case(sample_evaluation_data),
                base_model_name_or_path="test-model",
                steering_pipelines={"baseline": []},
                backend="not-a-real-kind",
            )

    def test_default_backend_uses_shared_model_path(self, sample_evaluation_data, mock_base_model):
        use_case = _make_use_case(sample_evaluation_data)
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl()]},
        )

        benchmark.run()

        assert mock_base_model == [1]  # shared-base path active by default
        pipeline = use_case._generate_calls[0]["model_or_pipeline"]
        assert pipeline.model is benchmark._base_model


class TestBenchmarkReleasesBackends:
    """The benchmark releases each configuration's backends, including when a trial raises."""

    @pytest.fixture
    def recording_release(self, monkeypatch):
        """Wrap `SteeringPipeline.release_backends` with a counter that still calls through."""
        calls = []
        original = SteeringPipeline.release_backends

        def wrapper(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(SteeringPipeline, "release_backends", wrapper)
        return calls

    def test_release_called_once_per_configuration(
        self, sample_evaluation_data, mock_base_model, recording_release
    ):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        benchmark.run()

        assert len(recording_release) == 2  # one per swept configuration

    def test_release_called_when_a_trial_raises(
        self, sample_evaluation_data, mock_base_model, recording_release
    ):
        class _FailingUseCase(MockUseCase):
            def generate(self, *args, **kwargs):
                raise RuntimeError("trial boom")

        benchmark = Benchmark(
            use_case=_FailingUseCase(
                evaluation_data=sample_evaluation_data,
                evaluation_metrics=[MockAccuracyMetric()],
            ),
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [MockInputControl()]},
        )

        with pytest.raises(RuntimeError, match="trial boom"):
            benchmark.run()

        assert len(recording_release) == 1  # released in the finally despite the failure


# Pre-flight support tests
class _UnsupportedControl(MockStateControl):
    """State control requiring an atom the implicit Hugging Face backend never advertises."""

    def requirements(self) -> Requirements:
        return Requirements(generate=needs(Capability.INTERVENTION_SPECS))


class TestPreflight:
    """Pre-flight checks every sweep point before any model or engine work."""

    def test_raise_aggregates_and_loads_nothing(self, sample_evaluation_data, mock_base_model):
        spec = ControlSpec(control_cls=_UnsupportedControl, vars={"scale_factor": [0.5, 1.0]})
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        with pytest.raises(UnsupportedBenchmarkError) as excinfo:
            benchmark.run()

        message = str(excinfo.value)
        assert "sweep" in message
        assert "_UnsupportedControl" in message
        assert "generate" in message  # core's verdict text names the phase
        assert mock_base_model == []

    def test_skip_runs_supported_points_only(self, sample_evaluation_data, mock_base_model, tmp_path, caplog):
        # one supported point (MockInputControl) and one unsupported point (_UnsupportedControl)
        supported = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1]}, name="ok")
        unsupported = ControlSpec(control_cls=_UnsupportedControl, vars={"scale_factor": [0.5]}, name="bad")
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"good": [supported], "gated": [unsupported]},
            save_dir=tmp_path,
            on_unsupported="skip",
        )

        with caplog.at_level("WARNING", logger="aisteer360.evaluation.benchmark"):
            profiles = benchmark.run()

        assert len(profiles["good"]) == 1
        assert profiles["gated"] == []  # skipped point produced no runs
        assert any("Skipping unsupported configuration" in r.getMessage() for r in caplog.records)
        with open(tmp_path / "checkpoint.json") as f:
            saved = json.load(f)
        assert saved["profiles"]["gated"] == []  # no checkpoint entry for the skipped point

    def test_preflight_enumerates_executed_config_ids(self, sample_evaluation_data, mock_base_model):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        benchmark = Benchmark(
            use_case=_make_use_case(sample_evaluation_data),
            base_model_name_or_path="test-model",
            steering_pipelines={"sweep": [spec]},
        )

        profiles = benchmark.run()

        executed = {(("sweep",), run["config_id"]) for run in profiles["sweep"]}
        # re-derive the config ids the pre-flight would enumerate
        preflight_ids = set()
        for specs, params, factory in benchmark._iter_config_points("sweep", [spec]):
            controls = factory()
            preflight_ids.add((("sweep",), benchmark._config_id(specs=specs, params=params, controls=controls)))
        assert {cid for (_, cid) in executed} == {cid for (_, cid) in preflight_ids}
