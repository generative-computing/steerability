"""Layout guards for the consolidated data specs and the `state_control.common` module split.

`LabeledExamples` and `as_labeled_examples` live in `core/internals/data.py` alongside
`ContrastivePairs`/`as_contrastive_pairs`. The `state_control.common` and `output_control.common`
packages re-export them from that single definition, and `output_control.common.specs` no longer
exists. `state_control.common.specs` holds the intervention IR only; fit configuration lives in
`fit_specs.py` and the wire compiler in `lowering.py`. These tests pin the identity of the
re-exports, the module layout, the widening of `as_labeled_examples` over `ContrastivePairs`, and
ITI's per-method rejection of `ContrastivePairs`.
"""
import importlib

import pytest

from aisteer360.algorithms.core.internals.data import ContrastivePairs, LabeledExamples, as_labeled_examples


class TestReExportIdentity:
    """The re-exports resolve to the single core definition (same object)."""

    def test_labeled_examples_identity_across_common_packages(self):
        from aisteer360.algorithms.output_control.common import LabeledExamples as output_labeled
        from aisteer360.algorithms.state_control.common import LabeledExamples as state_labeled

        assert output_labeled is LabeledExamples
        assert state_labeled is LabeledExamples

    def test_as_labeled_examples_identity_across_common_packages(self):
        from aisteer360.algorithms.output_control.common import as_labeled_examples as output_fn
        from aisteer360.algorithms.state_control.common import as_labeled_examples as state_fn

        assert output_fn is as_labeled_examples
        assert state_fn is as_labeled_examples


class TestAsLabeledExamplesWidening:
    """`as_labeled_examples` accepts a `ContrastivePairs`, a `LabeledExamples`, and a mapping."""

    def test_accepts_contrastive_pairs(self):
        pairs = ContrastivePairs(positives=["a", "b"], negatives=["c", "d"])
        labeled = as_labeled_examples(pairs)
        assert isinstance(labeled, LabeledExamples)
        assert list(labeled.positives) == ["a", "b"]
        assert list(labeled.negatives) == ["c", "d"]

    def test_accepts_contrastive_pairs_from_state_common_reexport(self):
        from aisteer360.algorithms.state_control.common import ContrastivePairs as state_pairs

        pairs = state_pairs(positives=["p"], negatives=["n"])
        labeled = as_labeled_examples(pairs)
        assert isinstance(labeled, LabeledExamples)

    def test_accepts_labeled_examples_identity(self):
        labeled = LabeledExamples(positives=["p1", "p2", "p3"], negatives=["n1"])
        assert as_labeled_examples(labeled) is labeled

    def test_accepts_mapping(self):
        labeled = as_labeled_examples({"positives": ["p"], "negatives": ["n1", "n2"]})
        assert isinstance(labeled, LabeledExamples)
        assert list(labeled.positives) == ["p"]
        assert list(labeled.negatives) == ["n1", "n2"]

    def test_labeled_examples_allows_unequal_lengths(self):
        labeled = LabeledExamples(positives=["p1", "p2", "p3"], negatives=["n1"])
        assert len(labeled.positives) == 3
        assert len(labeled.negatives) == 1


class TestITIRejectsContrastivePairs:
    """ITI keeps its per-method rejection of `ContrastivePairs` (above the generic converter)."""

    def test_iti_args_raises_on_contrastive_pairs(self):
        from aisteer360.algorithms.state_control.iti.args import ITIArgs

        with pytest.raises(TypeError, match="ITI requires LabeledExamples, not ContrastivePairs"):
            ITIArgs(data=ContrastivePairs(positives=["a"], negatives=["b"]))


class TestModuleRemoval:
    """`output_control.common.specs` is gone; `state_control.common.specs` no longer holds the moved names."""

    def test_output_common_specs_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("aisteer360.algorithms.output_control.common.specs")

    def test_state_common_specs_has_no_moved_names(self):
        state_specs = importlib.import_module("aisteer360.algorithms.state_control.common.specs")
        assert not hasattr(state_specs, "LabeledExamples")
        assert not hasattr(state_specs, "ContrastivePairs")
        assert not hasattr(state_specs, "as_labeled_examples")


class TestCommonSpecsSplit:
    """`state_control.common.specs` holds the IR; fit configuration and the wire compiler live beside it."""

    def test_specs_has_no_moved_names(self):
        state_specs = importlib.import_module("aisteer360.algorithms.state_control.common.specs")
        for name in (
            "VectorTrainSpec",
            "ConditionSearchSpec",
            "Comparator",
            "CompMode",
            "normalize_comparator",
            "lower_interventions",
            "artifact_id_for",
            "ScopeKindLiteral",
        ):
            assert not hasattr(state_specs, name)

    def test_fit_specs_holds_the_fit_configuration(self):
        fit_specs = importlib.import_module("aisteer360.algorithms.state_control.common.fit_specs")
        for name in (
            "Comparator",
            "CompMode",
            "VectorTrainSpec",
            "ConditionSearchSpec",
        ):
            assert hasattr(fit_specs, name)

    def test_lowering_holds_the_wire_compiler(self):
        lowering = importlib.import_module("aisteer360.algorithms.state_control.common.lowering")
        assert hasattr(lowering, "lower_interventions")
        assert hasattr(lowering, "artifact_id_for")

    def test_common_reexports_are_the_fit_specs_definitions(self):
        common = importlib.import_module("aisteer360.algorithms.state_control.common")
        fit_specs = importlib.import_module("aisteer360.algorithms.state_control.common.fit_specs")
        for name in ("Comparator", "CompMode", "ConditionSearchSpec", "VectorTrainSpec"):
            assert getattr(common, name) is getattr(fit_specs, name)

    def test_token_scope_scope_kind_is_the_specs_definition(self):
        specs = importlib.import_module("aisteer360.algorithms.state_control.common.specs")
        token_scope = importlib.import_module("aisteer360.algorithms.state_control.common.token_scope")
        assert token_scope.ScopeKind is specs.ScopeKind
