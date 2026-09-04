"""
Tests for control-composition and tokenizer utilities.

Tests cover:

- `merge_controls` categorization, defaults, ordering, and error handling
- `ensure_pad_token`
"""
from unittest.mock import MagicMock

import pytest

from aisteer360.algorithms.core.utils.controls import merge_controls
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import DecodingDriver
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.utils.tokenization import ensure_pad_token
from tests.conftest import MockInputControl, MockOutputControl, MockStateControl, MockStructuralControl


# merge_controls Tests
class TestMergeControlsEmpty:
    """Tests for merge_controls with empty or minimal input."""

    def test_empty_list_returns_empty_categories(self):
        """An empty list yields an empty list per category; the identity element of every
        category's fold is the empty sequence."""
        result = merge_controls([])

        assert result == {
            "input_controls": [],
            "structural_controls": [],
            "state_controls": [],
            "output_controls": [],
        }

    def test_empty_iterable_returns_empty_categories(self):
        """Any empty iterable yields empty categories."""
        result = merge_controls(iter([]))
        assert result["input_controls"] == []


class TestMergeControlsSingleCategory:
    """Tests for merge_controls with single-category controls."""

    def test_single_input_control(self):
        control = MockInputControl()
        result = merge_controls([control])

        assert result["input_controls"] == [control]
        # other categories stay empty
        assert result["structural_controls"] == []
        assert result["state_controls"] == []
        assert result["output_controls"] == []

    def test_single_structural_control(self):
        control = MockStructuralControl()
        result = merge_controls([control])

        assert result["structural_controls"] == [control]
        assert result["input_controls"] == []

    def test_single_state_control(self):
        control = MockStateControl()
        result = merge_controls([control])

        assert isinstance(result["state_controls"][0], MockStateControl)
        assert result["state_controls"] == [control]

    def test_single_output_control(self):
        control = MockOutputControl()
        result = merge_controls([control])

        assert result["output_controls"] == [control]


class TestMergeControlsMultipleCategories:
    """Tests for merge_controls with controls from multiple categories."""

    def test_two_different_categories(self):
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        result = merge_controls([input_ctrl, state_ctrl])

        assert result["input_controls"] == [input_ctrl]
        assert result["state_controls"] == [state_ctrl]
        assert result["structural_controls"] == []
        assert result["output_controls"] == []

    def test_all_four_categories(self):
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()
        output_ctrl = MockOutputControl()

        result = merge_controls([input_ctrl, structural_ctrl, state_ctrl, output_ctrl])

        assert result["input_controls"] == [input_ctrl]
        assert result["structural_controls"] == [structural_ctrl]
        assert result["state_controls"] == [state_ctrl]
        assert result["output_controls"] == [output_ctrl]

    def test_order_independent_across_categories(self):
        """Cross-category ordering does not change per-category results."""
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        result1 = merge_controls([input_ctrl, state_ctrl])
        result2 = merge_controls([state_ctrl, input_ctrl])

        assert result1["input_controls"] == result2["input_controls"]
        assert result1["state_controls"] == result2["state_controls"]


class TestMergeControlsMultiplicityAndErrors:
    """Tests for within-category multiplicity and error handling."""

    def test_multiple_input_controls_returned_in_order(self):
        ctrl1 = MockInputControl()
        ctrl2 = MockInputControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["input_controls"] == [ctrl1, ctrl2]

    def test_multiple_structural_controls_returned_in_order(self):
        ctrl1 = MockStructuralControl()
        ctrl2 = MockStructuralControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["structural_controls"] == [ctrl1, ctrl2]

    def test_multiple_state_controls_returned_in_order(self):
        ctrl1 = MockStateControl()
        ctrl2 = MockStateControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["state_controls"] == [ctrl1, ctrl2]

    def test_multiple_output_controls_returned_in_order(self):
        ctrl1 = MockOutputControl()
        ctrl2 = MockOutputControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["output_controls"] == [ctrl1, ctrl2]

    def test_multiple_decoding_drivers_raises(self):
        """Two enabled decoding drivers raise (the decode loop does not compose)."""

        class DriverA(DecodingDriver):
            def decode(self, *args, **kwargs):
                raise NotImplementedError

        class DriverB(DecodingDriver):
            def decode(self, *args, **kwargs):
                raise NotImplementedError

        with pytest.raises(ValueError, match="decoding drivers"):
            merge_controls([DriverA(), DriverB()])

    def test_step_level_control_plus_driver_allowed(self):
        """A step-level control alongside a single driver is accepted, in encounter order."""

        class Driver(DecodingDriver):
            def decode(self, *args, **kwargs):
                raise NotImplementedError

        step_level_control = MockOutputControl()
        driver = Driver()

        result = merge_controls([step_level_control, driver])
        assert result["output_controls"] == [step_level_control, driver]

    def test_unknown_control_type_raises(self):
        class UnknownControl:
            pass

        with pytest.raises(TypeError, match="Unknown control type"):
            merge_controls([UnknownControl()])

    def test_duplicate_instance_error_includes_control_name(self):
        ctrl = MockInputControl()

        with pytest.raises(ValueError) as exc_info:
            merge_controls([ctrl, ctrl])

        assert "MockInputControl" in str(exc_info.value)


class TestMergeControlsWithSubclasses:
    """Tests for merge_controls with control subclasses."""

    def test_subclass_recognized_as_parent_category(self):
        class CustomInputControl(InputControl):
            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

        control = CustomInputControl()
        result = merge_controls([control])

        assert isinstance(result["input_controls"][0], CustomInputControl)

    def test_different_subclasses_of_same_category_compose_in_order(self):
        class CustomInput1(InputControl):
            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

        class CustomInput2(InputControl):
            def adapt(self, input_ids, runtime_kwargs=None):
                return input_ids

        first, second = CustomInput1(), CustomInput2()
        result = merge_controls([first, second])
        assert result["input_controls"] == [first, second]


# ensure_pad_token Tests
class TestEnsurePadToken:
    """Tests for the `ensure_pad_token` utility."""

    def test_sets_pad_token_when_none(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 1
        tokenizer.eos_token = "</s>"

        result = ensure_pad_token(tokenizer)

        assert result.pad_token_id == 1
        assert result.pad_token == "</s>"

    def test_preserves_existing_pad_token(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token_id = 1

        result = ensure_pad_token(tokenizer)

        assert result.pad_token_id == 0
        assert result.pad_token == "<pad>"

    def test_returns_same_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        result = ensure_pad_token(tokenizer)

        assert result is tokenizer


# Control Type Detection Tests
class TestControlTypeDetection:
    """Tests for category detection in merge_controls."""

    def test_detects_input_control(self):
        control = MockInputControl()
        result = merge_controls([control])
        assert "input_controls" in result
        assert result["input_controls"] == [control]

    def test_detects_by_inheritance(self):
        control = MockInputControl()
        assert isinstance(control, InputControl)

        result = merge_controls([control])
        assert result["input_controls"] == [control]


# Edge Cases
class TestMergeControlsEdgeCases:
    """Edge-case tests for merge_controls."""

    def test_generator_input(self):
        def control_generator():
            yield MockInputControl()
            yield MockStateControl()

        result = merge_controls(control_generator())
        assert isinstance(result["input_controls"][0], MockInputControl)
        assert isinstance(result["state_controls"][0], MockStateControl)

    def test_single_element_list(self):
        control = MockInputControl()
        result = merge_controls([control])
        assert result["input_controls"] == [control]

    def test_preserves_control_state(self):
        control = MockInputControl(prefix="test_", num_examples=5)
        result = merge_controls([control])

        assert result["input_controls"][0].prefix == "test_"
        assert result["input_controls"][0].num_examples == 5

    def test_controls_not_modified(self):
        control = MockInputControl()
        original_enabled = control.enabled

        merge_controls([control])

        assert control.enabled == original_enabled
