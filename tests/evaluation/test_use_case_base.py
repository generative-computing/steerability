"""Tests for `UseCase` construction: declared parameters, data loading, validation, and defaults.

Covers the class-level-annotation parameter mechanism (required vs optional, mutable-default copying,
ClassVar / underscore / method exclusion, mixin non-contribution, re-annotated base names), `.json` /
`.jsonl` loading and rejection of non-mapping data, seed-deterministic shuffle and `num_samples`
limiting, per-item `validate_evaluation_data` running after sampling and carrying the index, metric
type checking and duplicate-name warning, empty-data warning, and the default no-op `export`.
"""
import json
from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from aisteer360.evaluation.metrics.base import Metric
from aisteer360.evaluation.use_cases.base import UseCase


class _Dummy(Metric):
    """Trivial metric; name defaults to the class name unless overridden."""

    def __init__(self, name: str | None = None, **extras):
        super().__init__(**extras)
        if name is not None:
            self.name = name

    def compute(self, responses, prompts=None, **kwargs):
        return {"n": len(responses)}


class _Base(UseCase):
    """Concrete use case with `generate`/`evaluate` stubs, for construction tests."""

    def generate(self, model_or_pipeline, tokenizer, gen_kwargs=None, runtime_overrides=None, **kwargs):
        return []

    def evaluate(self, generations):
        return {}


class _RequiredParam(_Base):
    shuffling_runs: int


class _OptionalParam(_Base):
    threshold: float = 0.5


class _MutableDefault(_Base):
    tags: list = ["a", "b"]


class _ClassVarAnnotated(_Base):
    marker: ClassVar[str] = "not-a-parameter"
    real_param: int = 3


class _StringizedClassVar(_Base):
    marker: "ClassVar[int]" = 7
    real_param: int = 3


class _MethodAnnotated(_Base):
    helper: Any = None  # a real optional parameter, default None

    def helper(self):  # noqa: F811 - a method shadows the annotation; not a parameter
        return 1


class _ReAnnotatesBaseName(_Base):
    num_samples: int = 99  # re-annotating a base __init__ name must not create a parameter


class _Mixin:
    extra: int = 123  # a plain mixin does not subclass UseCase, so contributes no parameter


class _WithMixin(_Mixin, _Base):
    own: int = 1


def _one_row() -> list[dict]:
    return [{"id": "q1", "value": 1}]


class TestDeclaredParameters:
    def test_required_parameter_supplied_and_mirrored(self):
        use_case = _RequiredParam(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], shuffling_runs=4)
        assert use_case.shuffling_runs == 4

    def test_missing_required_raises_naming_it(self):
        with pytest.raises(TypeError, match="shuffling_runs"):
            _RequiredParam(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])

    def test_optional_default_applied(self):
        use_case = _OptionalParam(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])
        assert use_case.threshold == 0.5

    def test_optional_overridable(self):
        use_case = _OptionalParam(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], threshold=0.9)
        assert use_case.threshold == 0.9

    def test_unknown_keyword_raises_listing_declared_set(self):
        with pytest.raises(TypeError, match="typo"):
            _OptionalParam(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], typo=1)

    def test_unknown_keyword_when_nothing_declared(self):
        with pytest.raises(TypeError) as info:
            _Base(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], anything=1)
        assert "anything" in str(info.value)
        assert "declared parameters are []" in str(info.value)

    def test_classvar_annotation_is_not_a_parameter(self):
        use_case = _ClassVarAnnotated(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], real_param=5)
        assert use_case.real_param == 5
        with pytest.raises(TypeError, match="marker"):
            _ClassVarAnnotated(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], marker="x")

    def test_stringized_classvar_is_not_a_parameter(self):
        with pytest.raises(TypeError, match="marker"):
            _StringizedClassVar(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], marker=1)

    def test_underscore_annotation_is_not_a_parameter(self):
        class _Underscore(_Base):
            _hidden: int = 1

        with pytest.raises(TypeError, match="_hidden"):
            _Underscore(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], _hidden=2)

    def test_method_annotation_is_not_a_parameter(self):
        # the annotated name resolves to a method, so the callable rule skips it
        with pytest.raises(TypeError, match="helper"):
            _MethodAnnotated(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], helper=lambda: 2)

    def test_reannotating_base_init_name_creates_no_parameter(self):
        use_case = _ReAnnotatesBaseName(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])
        # num_samples remains the base __init__ parameter, defaulting to keep-all
        assert len(use_case.evaluation_data) == 1

    def test_mixin_contributes_no_parameter(self):
        use_case = _WithMixin(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], own=2)
        assert use_case.own == 2
        with pytest.raises(TypeError, match="extra"):
            _WithMixin(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()], own=2, extra=1)

    def test_mutable_default_copied_per_instance(self):
        first = _MutableDefault(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])
        second = _MutableDefault(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])
        first.tags.append("c")
        assert second.tags == ["a", "b"]
        assert _MutableDefault.tags == ["a", "b"]


class TestDataLoading:
    def test_in_memory_items_copied(self):
        source = [{"id": "q1"}]
        use_case = _Base(evaluation_data=source, evaluation_metrics=[_Dummy()])
        use_case.evaluation_data[0]["id"] = "mutated"
        assert source[0]["id"] == "q1"

    def test_json_loading(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"id": "q1"}, {"id": "q2"}]))
        use_case = _Base(evaluation_data=str(path), evaluation_metrics=[_Dummy()])
        assert [row["id"] for row in use_case.evaluation_data] == ["q1", "q2"]

    def test_jsonl_loading(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": "q1"}\n\n{"id": "q2"}\n')
        use_case = _Base(evaluation_data=str(path), evaluation_metrics=[_Dummy()])
        assert [row["id"] for row in use_case.evaluation_data] == ["q1", "q2"]

    def test_non_sequence_rejected(self):
        with pytest.raises(TypeError, match="sequence of mappings or a path"):
            _Base(evaluation_data=42, evaluation_metrics=[_Dummy()])

    def test_non_mapping_items_rejected(self):
        with pytest.raises(TypeError, match="must contain mappings"):
            _Base(evaluation_data=[1, 2, 3], evaluation_metrics=[_Dummy()])


class TestShuffleAndSample:
    def test_shuffle_is_seed_deterministic(self):
        data = [{"id": f"q{i}"} for i in range(10)]
        first = _Base(evaluation_data=data, evaluation_metrics=[_Dummy()], shuffle=True, seed=7)
        second = _Base(evaluation_data=data, evaluation_metrics=[_Dummy()], shuffle=True, seed=7)
        assert [r["id"] for r in first.evaluation_data] == [r["id"] for r in second.evaluation_data]

    def test_num_samples_limits(self):
        data = [{"id": f"q{i}"} for i in range(10)]
        use_case = _Base(evaluation_data=data, evaluation_metrics=[_Dummy()], num_samples=3)
        assert len(use_case.evaluation_data) == 3


class TestPerItemValidation:
    def test_validation_carries_index_and_runs_after_sampling(self):
        class _NeedsFlag(_Base):
            def validate_evaluation_data(self, instance: Mapping[str, Any]) -> None:
                if "flag" not in instance:
                    raise ValueError("missing 'flag'")

        # only the first two survive num_samples; the third (invalid) is never validated
        data = [{"id": "q0", "flag": 1}, {"id": "q1"}, {"id": "q2"}]
        with pytest.raises(ValueError, match=r"evaluation_data\[1\]: missing 'flag'"):
            _NeedsFlag(evaluation_data=data, evaluation_metrics=[_Dummy()])

    def test_validation_skips_sampled_out_invalid_rows(self):
        class _NeedsFlag(_Base):
            def validate_evaluation_data(self, instance: Mapping[str, Any]) -> None:
                if "flag" not in instance:
                    raise ValueError("missing 'flag'")

        data = [{"id": "q0", "flag": 1}, {"id": "q1"}]  # second is invalid but sampled out
        use_case = _NeedsFlag(evaluation_data=data, evaluation_metrics=[_Dummy()], num_samples=1)
        assert len(use_case.evaluation_data) == 1


class TestMetricsAndWarnings:
    def test_non_metric_rejected(self):
        with pytest.raises(TypeError, match="must be of type `Metric`"):
            _Base(evaluation_data=_one_row(), evaluation_metrics=["not a metric"])

    def test_duplicate_metric_name_warns(self):
        with pytest.warns(UserWarning, match="Duplicate metric name"):
            _Base(evaluation_data=_one_row(), evaluation_metrics=[_Dummy(name="M"), _Dummy(name="M")])

    def test_empty_data_warns(self):
        with pytest.warns(UserWarning, match="evaluation data"):
            _Base(evaluation_data=[], evaluation_metrics=[_Dummy()])


class TestDefaultExport:
    def test_default_export_writes_nothing(self, tmp_path):
        use_case = _Base(evaluation_data=_one_row(), evaluation_metrics=[_Dummy()])
        use_case.export({"pipeline": []}, str(tmp_path))
        assert list(tmp_path.iterdir()) == []
