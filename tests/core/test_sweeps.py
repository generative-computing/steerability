"""Tests for the core sweep layer: configuration expansion, pre-flight, and the pipeline factory.

Covers `expand_configurations` (mixing, duplicate names, product order, fresh instantiation,
`config_id` stability, the baseline point), `preflight` verdict messages, and `PipelineFactory`
(shared-base reuse, the drop before a structural point and reload after, the fingerprint tripwire
with an intentionally mutating control, and the `finally` discipline under a raising body).
Runs hub-free on mocks and the tiny randomly-initialized models.
"""
from unittest.mock import MagicMock

import pytest
import torch

from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.core.specs import ControlSpec
from steerability.algorithms.core.sweeps import PipelineFactory, expand_configurations, preflight
from tests.conftest import (
    MockInputControl,
    MockStateControl,
    MockStructuralControl,
    create_mock_model,
    create_mock_tokenizer,
)

BASE = "test-model"


def _points(pipelines):
    return list(expand_configurations(pipelines, base_model_name_or_path=BASE))


class TestExpandConfigurations:
    def test_baseline_point(self):
        (point,) = _points({"baseline": []})
        assert point.pipeline_name == "baseline"
        assert point.config_id == "baseline"
        assert point.descriptor == {"controls": []}
        assert point.specs is None and point.params is None
        assert point.controls_factory() == []

    def test_none_pipeline_treated_as_baseline(self):
        (point,) = _points({"baseline": None})
        assert point.config_id == "baseline"

    def test_fixed_pipeline_reuses_instances(self):
        control = MockInputControl(prefix="a")
        (point,) = _points({"fixed": [control]})
        assert point.controls_factory() is point.controls_factory()
        assert point.controls_factory()[0] is control
        assert point.specs is None and point.params is None
        assert point.config_id != "baseline"

    def test_mixed_spec_and_fixed_raises(self):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1]})
        with pytest.raises(TypeError, match="mixes ControlSpec and fixed controls"):
            _points({"mixed": [spec, MockInputControl()]})

    def test_duplicate_resolved_names_raise(self):
        specs = [
            ControlSpec(control_cls=MockInputControl, params={"prefix": "a"}),
            ControlSpec(control_cls=MockInputControl, params={"prefix": "b"}),
        ]
        with pytest.raises(ValueError, match="distinct `name="):
            _points({"sweep": specs})

    def test_cartesian_product_order_and_params(self):
        specs = [
            ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]}, name="first"),
            ControlSpec(control_cls=MockStateControl, vars={"scale_factor": [0.5, 1.0]}, name="second"),
        ]
        points = _points({"sweep": specs})
        assert len(points) == 4
        combos = [
            (point.params["first"]["num_examples"], point.params["second"]["scale_factor"])
            for point in points
        ]
        assert combos == [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0)]

    def test_spec_factory_instantiates_fresh_per_call(self):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [3]})
        (point,) = _points({"sweep": [spec]})
        first, second = point.controls_factory(), point.controls_factory()
        assert first[0] is not second[0]
        assert first[0].num_examples == 3

    def test_config_id_stable_across_expansions(self):
        spec = ControlSpec(control_cls=MockInputControl, vars={"num_examples": [1, 2]})
        first = [point.config_id for point in _points({"sweep": [spec]})]
        second = [point.config_id for point in _points({"sweep": [spec]})]
        assert first == second
        assert len(set(first)) == 2

    def test_callable_vars_receive_context(self):
        seen = {}

        def space(context):
            seen.update(context)
            yield {"num_examples": 5}

        spec = ControlSpec(control_cls=MockInputControl, vars=space)
        (point,) = _points({"sweep": [spec]})
        assert seen["pipeline_name"] == "sweep"
        assert seen["base_model_name_or_path"] == BASE
        assert point.params["MockInputControl"] == {"num_examples": 5}

    def test_spec_without_points_counts_as_one(self):
        spec = ControlSpec(control_cls=MockInputControl, params={"prefix": "x"})
        points = _points({"sweep": [spec]})
        assert len(points) == 1
        assert points[0].params["MockInputControl"] == {"prefix": "x"}


class _UnsupportedStateControl(MockStateControl):
    """State control requiring an atom the implicit Hugging Face backend never advertises."""

    def requirements(self) -> Requirements:
        return Requirements(generate=needs(Capability.INTERVENTION_SPECS))


class TestPreflight:
    def test_supported_points_yield_no_messages(self):
        points = _points({"good": [MockInputControl()], "baseline": []})
        assert preflight(points, base_model_name_or_path=BASE, backend=None, fit="auto") == []

    def test_unsupported_point_message_names_pipeline_and_config_id(self):
        spec = ControlSpec(control_cls=_UnsupportedStateControl, vars={"scale_factor": [0.5, 1.0]})
        points = _points({"sweep": [spec]})
        messages = preflight(points, base_model_name_or_path=BASE, backend=None, fit="auto")
        assert len(messages) == 2
        for point, message in zip(points, messages):
            assert message.startswith(f"sweep [{point.config_id}] _UnsupportedStateControl (generate)")


@pytest.fixture
def tiny_base(monkeypatch):
    """Patch `_ensure_base_model` to install a real tiny model and record loads."""
    from steerability.algorithms.core.internals.fingerprint import model_fingerprint
    from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

    record = {"loads": []}

    def fake_ensure(self):
        if self._base_model is None:
            self._base_model = tiny_llama()
            tokenizer = wordlevel_tokenizer()
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            self._base_tokenizer = tokenizer
            self._base_fingerprint = model_fingerprint(self._base_model)
            record["loads"].append(1)

    monkeypatch.setattr(PipelineFactory, "_ensure_base_model", fake_ensure)
    return record


@pytest.fixture
def patched_pipeline_loaders(monkeypatch):
    """Replace the Hugging Face loader classes used by `SteeringPipeline` with mocks."""
    model = create_mock_model()
    model.to.return_value = model
    tokenizer = create_mock_tokenizer()
    model_loader = MagicMock()
    model_loader.from_pretrained.return_value = model
    tokenizer_loader = MagicMock()
    tokenizer_loader.from_pretrained.return_value = tokenizer
    monkeypatch.setattr("steerability.algorithms.core.steering_pipeline.AutoModelForCausalLM", model_loader)
    monkeypatch.setattr("steerability.algorithms.core.steering_pipeline.AutoTokenizer", tokenizer_loader)
    return model_loader, tokenizer_loader, model, tokenizer


class _MutatingStateControl(MockStateControl):
    """State control whose `steer` perturbs a shared-model parameter in place."""

    def steer(self, model, tokenizer=None, **kwargs):
        super().steer(model, tokenizer=tokenizer, **kwargs)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return model


class _CleanupRecordingControl(MockInputControl):
    """Input control recording `cleanup` invocations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


class TestPipelineFactory:
    def test_shared_base_reused_across_configurations(self, tiny_base):
        factory = PipelineFactory(BASE)
        with factory.steered([MockInputControl()]) as first:
            first_model = first.model
        with factory.steered([MockInputControl()]) as second:
            assert second.model is first_model
        assert tiny_base["loads"] == [1]
        assert factory.shared_base_fingerprint is not None

    def test_structural_point_drops_shared_base_and_next_point_reloads(
        self, tiny_base, patched_pipeline_loaders
    ):
        factory = PipelineFactory(BASE)
        with factory.steered([MockInputControl()]):
            pass
        assert factory.shared_base_fingerprint is not None
        with factory.steered([MockStructuralControl()]):
            assert factory._base_model is None  # dropped before the structural point steered
        with factory.steered([MockInputControl()]):
            pass
        assert tiny_base["loads"] == [1, 1]

    def test_fingerprint_tripwire_warns_and_quarantines(self, tiny_base, caplog):
        factory = PipelineFactory(BASE)
        with caplog.at_level("WARNING", logger="steerability.algorithms.core.sweeps"):
            with factory.steered([_MutatingStateControl(scale_factor=0.5)]):
                pass
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Shared base model changed" in message and "_MutatingStateControl" in message
            for message in messages
        )
        assert factory._base_model is None  # quarantined
        with factory.steered([MockInputControl()]):
            pass
        assert tiny_base["loads"] == [1, 1]  # clean base reloaded

    def test_clean_configuration_does_not_trip(self, tiny_base, caplog):
        factory = PipelineFactory(BASE)
        with caplog.at_level("WARNING", logger="steerability.algorithms.core.sweeps"):
            with factory.steered([MockStateControl()]):
                pass
        assert not any("Shared base model changed" in record.getMessage() for record in caplog.records)
        assert tiny_base["loads"] == [1]

    def test_finally_discipline_under_raising_body(self, tiny_base, monkeypatch):
        released = []
        from steerability.algorithms.core.steering_pipeline import SteeringPipeline
        original_release = SteeringPipeline.release_backends
        monkeypatch.setattr(
            SteeringPipeline, "release_backends",
            lambda self: (released.append(1), original_release(self))[1],
        )
        control = _CleanupRecordingControl()
        factory = PipelineFactory(BASE)
        with pytest.raises(RuntimeError, match="task failed"):
            with factory.steered([control]):
                raise RuntimeError("task failed")
        assert control.cleanup_calls == 1
        assert released == [1]

    def test_release_drops_shared_base(self, tiny_base):
        factory = PipelineFactory(BASE)
        with factory.steered([]):
            pass
        assert factory.shared_base_fingerprint is not None
        factory.release()
        assert factory.shared_base_fingerprint is None
        assert factory._base_model is None

    def test_backend_kind_default(self):
        assert PipelineFactory(BASE).backend_kind == "huggingface"
        assert PipelineFactory(BASE, backend="vllm").backend_kind == "vllm"
