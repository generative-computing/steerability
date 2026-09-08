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
"""
import logging
import sys
import textwrap

import pytest

import steerability.algorithms.core.registry as registry
from steerability.algorithms.core.registry import RegistryError, _crawl_methods


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
