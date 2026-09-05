"""Tests for the shared `BaseControl` constructor/lifecycle, parametrized over all four categories.

Pins the construction contract: the null-argument guard and its message, args-field
mirroring (with reachability via `self.args`), the `@property`-name skip in every category, the
`_configure()` hook firing on both the null and non-null paths in every category, the class-attribute
defaults.
"""
from dataclasses import dataclass

import pytest

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.state_control.base import StateControl
from steerability.algorithms.structural_control.base import StructuralControl


@dataclass
class _Args(BaseArgs):
    """Minimal args dataclass shared by the concrete test subclasses."""
    prefix: str = "p"
    count: int = 3


def _implement_abstracts(namespace: dict, base: type) -> None:
    """Add trivial implementations for a base's abstract methods to a class namespace."""
    if "adapt" in base.__abstractmethods__:
        namespace["adapt"] = lambda self, input_ids, runtime_kwargs=None: input_ids
    if "steer" in base.__abstractmethods__:
        namespace["steer"] = lambda self, model=None, tokenizer=None, **kwargs: model
    if "get_hooks" in base.__abstractmethods__:
        namespace["get_hooks"] = lambda self, input_ids, runtime_kwargs=None, **kwargs: {
            "pre": [], "forward": [], "backward": []
        }


def _concrete(base: type, *, args=_Args, extra: dict | None = None) -> type:
    """Build a minimal concrete subclass of `base` with `Args = args` and abstract methods filled."""
    namespace: dict = {"Args": args}
    _implement_abstracts(namespace, base)
    if extra:
        namespace.update(extra)
    return type(f"_Concrete{base.__name__}", (base,), namespace)


CATEGORIES = [
    pytest.param(InputControl, False, id="input"),
    pytest.param(StructuralControl, True, id="structural"),
    pytest.param(StateControl, False, id="state"),
    pytest.param(OutputControl, False, id="output"),
]


@pytest.mark.parametrize("base,supports_batching_default", CATEGORIES)
class TestConstruction:
    def test_null_control_constructs_bare_and_rejects_arguments(self, base, supports_batching_default):
        cls = _concrete(base, args=None)  # Args = None -> arg-free control
        cls()  # bare construction succeeds
        with pytest.raises(TypeError, match="accepts no constructor arguments"):
            cls("x")
        with pytest.raises(TypeError, match="accepts no constructor arguments"):
            cls(prefix="x")

    def test_fields_mirror_onto_instance_and_stay_on_args(self, base, supports_batching_default):
        cls = _concrete(base)
        control = cls(prefix="hello", count=7)
        assert control.prefix == "hello"
        assert control.count == 7
        assert control.args.prefix == "hello"  # raw value reachable via self.args
        assert control.args.count == 7

    def test_property_named_field_survives_and_raw_value_on_args(self, base, supports_batching_default):
        cls = _concrete(base, extra={"prefix": property(lambda self: "computed")})
        control = cls(prefix="raw", count=1)
        assert control.prefix == "computed"  # the property answers, not clobbered by setattr
        assert control.args.prefix == "raw"   # the raw value stays reachable on self.args
        assert control.count == 1             # non-property fields still mirror

    def test_configure_runs_on_non_null_path(self, base, supports_batching_default):
        def _configure(self):
            self.configured = True

        cls = _concrete(base, extra={"_configure": _configure})
        control = cls(prefix="a", count=2)
        assert control.configured is True

    def test_configure_runs_on_null_path(self, base, supports_batching_default):
        def _configure(self):
            self.configured = True

        cls = _concrete(base, args=None, extra={"_configure": _configure})
        control = cls()
        assert control.configured is True

    def test_class_attribute_defaults(self, base, supports_batching_default):
        cls = _concrete(base)
        control = cls(prefix="a", count=2)
        assert control.supports_batching is supports_batching_default
        assert control.enabled is True
        assert callable(control.cleanup)
        assert control.cleanup() is None  # no-op


def test_all_categories_subclass_base_control():
    for base in (InputControl, StructuralControl, StateControl, OutputControl):
        assert issubclass(base, BaseControl)
