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

from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.evaluation.benchmark import Benchmark
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
        assert call["model_or_pipeline"] is benchmark._base_model
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
        assert use_case._generate_calls[0]["model_or_pipeline"] is benchmark._base_model

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
        assert set(run.keys()) == {"trial_id", "generations", "evaluations", "params"}
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
        saved_params = saved["baseline"][0]["params"]
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
        assert set(saved.keys()) == {"baseline"}
        assert len(saved["baseline"]) == len(profiles["baseline"])
        assert saved["baseline"][0]["params"] == {}

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
        assert len(partial["sweep"]) == 1
        assert partial["sweep"][0]["params"] == {"MockInputControl": {"num_examples": 1}}

        resumed_use_case = _make_use_case(sample_evaluation_data)
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
