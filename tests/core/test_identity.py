"""Tests for canonical configuration identity and trial-seed derivation.

Covers `canonical_value` over primitives, paths, numpy, tensors (content-addressed, device- and
grad-independent, bfloat16), dataclasses, mappings/sequences/sets, callables, and unhandled
objects; the descriptor builders for fixed controls and specs; the purity of `config_digest`; and
the purity and distinctness of `derive_trial_seed`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.identity import (
    canonical_value,
    config_descriptor_from_controls,
    config_descriptor_from_specs,
    config_digest,
    derive_trial_seed,
    qualname,
)
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.input_control.base import InputControl


@dataclass
class _IdentityArgs(BaseArgs):
    """Arguments for the local identity-test control."""
    alpha: float = 1.0
    label: str = "x"


class _ArgControl(InputControl):
    """Input control carrying a real `Args` dataclass, for descriptor tests."""
    Args = _IdentityArgs

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


class _ArgFreeControl(InputControl):
    """Arg-free input control (`Args = None`), contributing empty params."""
    Args = None

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


class _Spec:
    """Duck-typed stand-in for a `ControlSpec` (only `control_cls` and `name` are read)."""

    def __init__(self, control_cls, name=None):
        self.control_cls = control_cls
        self.name = name


class TestCanonicalValue:
    """Tests for `canonical_value`."""

    def test_primitives_pass_through(self):
        assert canonical_value(None) is None
        assert canonical_value("s") == "s"
        assert canonical_value(3) == 3
        assert canonical_value(3.5) == 3.5
        assert canonical_value(True) is True

    def test_path_becomes_string(self):
        assert canonical_value(Path("/a/b")) == "/a/b"

    def test_numpy_scalar_and_array(self):
        assert canonical_value(np.float64(2.5)) == 2.5
        assert canonical_value(np.array([1, 2, 3])) == [1, 2, 3]
        assert canonical_value(np.array([[1.5, 2.0], [3.0, 4.0]])) == [[1.5, 2.0], [3.0, 4.0]]

    def test_object_array_elements_follow_element_rules(self):
        arr = np.array([Path("models/base"), {"layers": {2, 1}}, np.int64(3)], dtype=object)
        form = canonical_value(arr)
        assert form == ["models/base", {"layers": [1, 2]}, 3]
        assert form == canonical_value([Path("models/base"), {"layers": {1, 2}}, 3])
        assert len(config_digest({"params": form})) == 12

    def test_nested_arrays_recurse(self):
        outer = np.empty(2, dtype=object)
        outer[0] = np.array([1, 2])
        outer[1] = np.array([Path("a")], dtype=object)
        assert canonical_value(outer) == [[1, 2], ["a"]]

    def test_mapping_key_order_irrelevant(self):
        assert canonical_value({"a": 1, "b": 2}) == canonical_value({"b": 2, "a": 1})

    def test_sequence_order_matters(self):
        assert canonical_value([1, 2, 3]) != canonical_value([3, 2, 1])
        assert canonical_value((1, 2)) == canonical_value([1, 2])

    def test_set_order_irrelevant(self):
        assert canonical_value({1, 2, 3}) == canonical_value({3, 1, 2})
        assert canonical_value(frozenset({1, 2})) == canonical_value({2, 1})

    def test_equal_tensors_equal_digest_one_change_differs(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        c = torch.tensor([[1.0, 2.0], [3.0, 4.5]])
        assert canonical_value(a) == canonical_value(b)
        assert canonical_value(a) != canonical_value(c)

    def test_tensor_device_and_grad_do_not_participate(self):
        plain = torch.tensor([1.0, 2.0, 3.0])
        with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert canonical_value(plain) == canonical_value(with_grad)

    def test_bfloat16_tensor_canonicalizes(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        form = canonical_value(tensor)
        assert "__tensor__" in form
        assert form["__tensor__"]["dtype"] == "torch.bfloat16"

    def test_dataclass_traversal(self):
        pairs = ContrastivePairs(positives=["p1"], negatives=["n1"])
        form = canonical_value(pairs)
        assert form["__dataclass__"] == qualname(ContrastivePairs)
        assert set(form["fields"]) == {"positives", "negatives", "prompts"}
        assert form["fields"]["positives"] == ["p1"]

    def test_callable_qualname_form(self):
        assert canonical_value(len).startswith("callable:")

    def test_unknown_object_yields_type_token(self):
        class _Weird:
            pass

        assert canonical_value(_Weird()) == {"__type__": qualname(_Weird)}


class TestConfigDescriptors:
    """Tests for the descriptor builders."""

    def test_fixed_controls_distinguish_configuration(self):
        first = config_descriptor_from_controls([_ArgControl(alpha=1.0)])
        second = config_descriptor_from_controls([_ArgControl(alpha=2.0)])
        assert first != second
        assert config_digest(first) != config_digest(second)

    def test_arg_free_control_contributes_empty_params(self):
        descriptor = config_descriptor_from_controls([_ArgFreeControl()])
        entry = descriptor["controls"][0]
        assert entry["params"] == {}
        assert entry["control"] == qualname(_ArgFreeControl)
        assert entry["enabled"] is True

    def test_specs_keyed_by_name_in_list_order(self):
        specs = [_Spec(_ArgControl, name="first"), _Spec(_ArgFreeControl, name="second")]
        params = {"first": {"alpha": 3.0}, "second": {}}
        descriptor = config_descriptor_from_specs(specs, params)
        controls = descriptor["controls"]
        assert [entry["control"] for entry in controls] == [
            qualname(_ArgControl), qualname(_ArgFreeControl),
        ]
        assert controls[0]["params"] == {"alpha": 3.0}


class TestConfigDigest:
    """Tests for `config_digest`."""

    def test_pure_function(self):
        left = config_descriptor_from_controls([_ArgControl(alpha=1.0, label="y")])
        right = config_descriptor_from_controls([_ArgControl(alpha=1.0, label="y")])
        assert config_digest(left) == config_digest(right)
        assert len(config_digest(left)) == 12

    def test_object_array_param_digests_like_its_list_form(self):
        specs = [_Spec(_ArgControl, name="ctl")]
        as_array = config_descriptor_from_specs(
            specs, {"ctl": {"alpha": 1.0, "paths": np.array([Path("a"), Path("b")], dtype=object)}}
        )
        as_list = config_descriptor_from_specs(specs, {"ctl": {"alpha": 1.0, "paths": [Path("a"), Path("b")]}})
        assert config_digest(as_array) == config_digest(as_list)


class TestDeriveTrialSeed:
    """Tests for `derive_trial_seed`."""

    def test_pure(self):
        assert derive_trial_seed(7, "cfg", 0) == derive_trial_seed(7, "cfg", 0)

    def test_distinct_across_trial_and_config(self):
        assert derive_trial_seed(7, "cfg", 0) != derive_trial_seed(7, "cfg", 1)
        assert derive_trial_seed(7, "cfg-a", 0) != derive_trial_seed(7, "cfg-b", 0)
        assert derive_trial_seed(1, "cfg", 0) != derive_trial_seed(2, "cfg", 0)
