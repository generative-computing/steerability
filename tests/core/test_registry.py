"""Tests for steering-method discovery in `steerability.algorithms.core.registry`.

Each case builds a synthetic `fakepkg` package tree under `tmp_path`, puts it on
`sys.path`, and crawls it with the parameterized `_crawl_methods` signature so the
real package is never touched. The `synthetic_env` fixture snapshots and restores the
module-global `REGISTRY`, `sys.path`, and `sys.modules` so cases do not bleed.

Covers the registry failure modes plus the happy paths:

    - well-formed export (with an extra key, mirroring MergeKit) -> registered
    - absent recognized optional dependency -> INFO skip with extra hint
    - absent unrecognized module -> WARNING skip
    - internal `ModuleNotFoundError` -> RegistryError
    - non-ImportError at import (tripwire) -> RegistryError naming the module
    - malformed export -> RegistryError
    - duplicate name within a category -> RegistryError
    - no export -> silent skip

A second group covers `register_method`, the explicit registration route for control classes
defined outside the toolkit tree, including a save/load round trip through a `.spipe` bundle.
"""
import logging
import sys
import textwrap
from dataclasses import dataclass, field

import pytest

import steerability.algorithms.core.registry as registry
from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.registry import (
    RegistryError,
    _crawl_methods,
    method_key_for,
    register_method,
    resolve_method_key,
)
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.spipe import SPipe, SpipeFormatError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def _make_category(pkg_root, category="state_control"):
    """Create `fakepkg/`, `fakepkg/algorithms/`, and `fakepkg/algorithms/<category>/` inits."""
    _write(pkg_root / "__init__.py", "")
    _write(pkg_root / "algorithms" / "__init__.py", "")
    _write(pkg_root / "algorithms" / category / "__init__.py", "")
    return pkg_root / "algorithms" / category


def _crawl(pkg_root):
    _crawl_methods(root=pkg_root / "algorithms", package_prefix="fakepkg.algorithms")


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    """Isolate a synthetic package tree: restore REGISTRY, sys.path, and sys.modules after."""
    pkg_root = tmp_path / "fakepkg"

    registry_snapshot = {category: bucket.copy() for category, bucket in registry.REGISTRY.items()}
    modules_snapshot = set(sys.modules)

    registry.REGISTRY.clear()
    monkeypatch.syspath_prepend(str(tmp_path))

    yield pkg_root

    registry.REGISTRY.clear()
    registry.REGISTRY.update(registry_snapshot)
    for name in list(sys.modules):
        if name == "fakepkg" or name.startswith("fakepkg."):
            if name not in modules_snapshot:
                del sys.modules[name]


def test_well_formed_method_registered_with_extra_key_tolerated(synthetic_env):
    """A well-formed export (with an extra 'category' key) is registered."""
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "good" / "__init__.py",
        """
        class GoodControl:
            pass

        STEERING_METHOD = {
            "category": "state_control",
            "name": "good_method",
            "control": GoodControl,
            "args": None,
        }
        """,
    )

    _crawl(synthetic_env)

    assert "good_method" in registry.REGISTRY.get("state_control", {})
    method = registry.REGISTRY["state_control"]["good_method"]
    assert method.category == "state"
    assert method.name == "good_method"
    assert method.args_cls is None


def test_absent_recognized_optional_dependency_skipped_with_info(synthetic_env, monkeypatch, caplog):
    """An absent module present in the extras map -> INFO skip with hint."""
    monkeypatch.setitem(registry.OPTIONAL_MODULE_EXTRAS, "totally_fake_optional", "fakeextra")
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "opt" / "__init__.py",
        """
        import totally_fake_optional  # noqa: F401
        STEERING_METHOD = {"name": "opt", "control": object, "args": None}
        """,
    )

    with caplog.at_level(logging.INFO, logger=registry.logger.name):
        _crawl(synthetic_env)

    assert "opt" not in registry.REGISTRY.get("state_control", {})
    assert any(
        record.levelno == logging.INFO
        and "totally_fake_optional" in record.getMessage()
        and 'steerability[fakeextra]' in record.getMessage()
        for record in caplog.records
    )


def test_absent_unrecognized_module_skipped_with_warning(synthetic_env, caplog):
    """An absent module not in the extras map -> WARNING skip."""
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "weird" / "__init__.py",
        """
        import a_module_nobody_declared_xyz  # noqa: F401
        STEERING_METHOD = {"name": "weird", "control": object, "args": None}
        """,
    )

    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        _crawl(synthetic_env)

    assert "weird" not in registry.REGISTRY.get("state_control", {})
    assert any(
        record.levelno == logging.WARNING and "a_module_nobody_declared_xyz" in record.getMessage()
        for record in caplog.records
    )


def test_internal_module_not_found_raises(synthetic_env):
    """A missing module *inside the package prefix* raises RegistryError."""
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "broken" / "__init__.py",
        """
        from fakepkg.algorithms.nope import Thing  # noqa: F401
        STEERING_METHOD = {"name": "broken", "control": object, "args": None}
        """,
    )

    with pytest.raises(RegistryError, match="fakepkg.algorithms"):
        _crawl(synthetic_env)


def test_tripwire_typeerror_raises_naming_module(synthetic_env):
    """A non-ImportError at import raises RegistryError naming the module path."""
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "tripwire" / "__init__.py",
        """
        raise TypeError("subclass tripwire")
        """,
    )

    with pytest.raises(RegistryError) as excinfo:
        _crawl(synthetic_env)

    assert "fakepkg.algorithms.state_control.tripwire" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "export_src, match",
    [
        ('STEERING_METHOD = {"control": object, "args": None}', "missing keys"),
        ('STEERING_METHOD = {"name": "", "control": object, "args": None}', "non-empty str"),
        ('STEERING_METHOD = {"name": "m", "control": 123, "args": None}', "must be a class"),
        ('STEERING_METHOD = {"name": "m", "control": object, "args": 123}', "class or None"),
        ('STEERING_METHOD = ["not", "a", "dict"]', "must be a dict"),
    ],
)
def test_malformed_export_raises(synthetic_env, export_src, match):
    """Malformed exports raise RegistryError."""
    category_dir = _make_category(synthetic_env)
    _write(category_dir / "bad" / "__init__.py", export_src + "\n")

    with pytest.raises(RegistryError, match=match):
        _crawl(synthetic_env)


def test_duplicate_name_within_category_raises(synthetic_env):
    """Two packages exporting the same name in one category raise RegistryError."""
    category_dir = _make_category(synthetic_env)
    for pkg in ("first", "second"):
        _write(
            category_dir / pkg / "__init__.py",
            """
            STEERING_METHOD = {"name": "dup", "control": object, "args": None}
            """,
        )

    with pytest.raises(RegistryError, match="dup"):
        _crawl(synthetic_env)


def test_no_export_silently_skipped(synthetic_env, caplog):
    """A package with no STEERING_METHOD is skipped with no log noise above DEBUG."""
    category_dir = _make_category(synthetic_env)
    _write(
        category_dir / "plain" / "__init__.py",
        """
        VALUE = 1
        """,
    )

    with caplog.at_level(logging.INFO, logger=registry.logger.name):
        _crawl(synthetic_env)

    assert registry.REGISTRY.get("state_control", {}) == {}
    registry_records = [record for record in caplog.records if record.name == registry.logger.name]
    assert registry_records == []


# register_method: controls defined outside the toolkit tree
@dataclass
class ExternalPrefixArgs(BaseArgs):
    """Arguments for `ExternalPrefixControl`."""

    marker: str = field(default="[external]", metadata={"help": "Text prepended to the last user turn."})


class ExternalPrefixControl(InputControl):
    """Input control defined outside the toolkit tree, prepending a marker to the last user turn."""

    Args = ExternalPrefixArgs

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def adapt_messages(self, messages, runtime_kwargs=None):
        adapted = []
        for chat in messages:
            turns = [dict(turn) for turn in chat]
            for turn in reversed(turns):
                if turn.get("role") == "user":
                    turn["content"] = f"{self.marker}\n\n{turn['content']}"
                    break
            adapted.append(turns)
        return adapted


class ExternalOtherControl(InputControl):
    """A second out-of-tree input control, used to test name collisions."""

    Args = None

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


@pytest.fixture
def clean_registry():
    """Restore every registry bucket after a test that registers methods."""
    snapshot = {category: bucket.copy() for category, bucket in registry.REGISTRY.items()}
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(snapshot)


def test_register_method_registers_under_category_bucket(clean_registry):
    """A registered class resolves both ways and its record carries the bare category."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)

    assert method_key_for(ExternalPrefixControl) == "input_control/external_prefix"
    method = resolve_method_key("input_control/external_prefix")
    assert method.control_cls is ExternalPrefixControl
    assert method.args_cls is ExternalPrefixArgs
    assert method.category == "input"
    assert method.name == "external_prefix"


def test_register_method_accepts_suffixed_category(clean_registry):
    """'input_control' and 'input' name the same bucket."""
    register_method("input_control", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)
    assert "external_prefix" in registry.REGISTRY["input_control"]

    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)
    assert list(registry.REGISTRY["input_control"]).count("external_prefix") == 1


def test_register_method_is_idempotent(clean_registry):
    """Re-registering the same class under the same name is a no-op."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)
    first = registry.REGISTRY["input_control"]["external_prefix"]

    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)
    assert registry.REGISTRY["input_control"]["external_prefix"] is first


def test_register_method_rejects_taken_name(clean_registry):
    """A different class under a taken name raises, naming the name."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)

    with pytest.raises(RegistryError, match="external_prefix"):
        register_method("input", "external_prefix", ExternalOtherControl, None)


def test_register_method_rejects_second_key_for_one_class(clean_registry):
    """One class registers under one key."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)

    with pytest.raises(RegistryError, match="already registered"):
        register_method("input", "external_prefix_alias", ExternalPrefixControl, ExternalPrefixArgs)


def test_register_method_rejects_unknown_category(clean_registry):
    with pytest.raises(RegistryError, match="Unknown steering category"):
        register_method("sideways", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)


def test_register_method_rejects_wrong_base_class(clean_registry):
    """An input control cannot register in the state category."""
    with pytest.raises(RegistryError, match="must subclass StateControl"):
        register_method("state", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)


def test_register_method_rejects_args_mismatch(clean_registry):
    with pytest.raises(RegistryError, match="must be"):
        register_method("input", "external_prefix", ExternalPrefixControl, None)


def test_registered_method_round_trips_through_spipe(tmp_path, clean_registry):
    """An externally registered control saves frozen (model-free) and loads back."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)

    pipeline = SteeringPipeline(
        model_name_or_path=TINY_MODEL,
        controls=[ExternalPrefixControl(marker="[house style]")],
    )
    saved = pipeline.to_spipe(freeze=True).save(tmp_path / "external.spipe")

    rebuilt = SPipe.load(saved).pipeline()
    assert len(rebuilt.input_controls) == 1
    control = rebuilt.input_controls[0]
    assert isinstance(control, ExternalPrefixControl)
    assert control.marker == "[house style]"


def test_loading_unregistered_key_points_at_register_method(tmp_path, clean_registry):
    """A bundle naming an unregistered method fails with a message naming the fix."""
    register_method("input", "external_prefix", ExternalPrefixControl, ExternalPrefixArgs)
    pipeline = SteeringPipeline(
        model_name_or_path=TINY_MODEL,
        controls=[ExternalPrefixControl(marker="[house style]")],
    )
    saved = pipeline.to_spipe(freeze=True).save(tmp_path / "external.spipe")

    del registry.REGISTRY["input_control"]["external_prefix"]

    with pytest.raises(SpipeFormatError, match="register_method"):
        SPipe.load(saved).pipeline()
