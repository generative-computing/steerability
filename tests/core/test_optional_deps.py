"""Tests for the optional-dependency guard and its consistency with `pyproject.toml`.

`test_optional_map_matches_pyproject` is the guardrail that keeps `OPTIONAL_MODULE_EXTRAS`
and the packaging metadata from drifting: every mapped module must live in the extra it
names and must not appear in the core `[project.dependencies]`.
"""
import tomllib
from pathlib import Path

import pytest

from steerability.utils.optional import OPTIONAL_MODULE_EXTRAS, require

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_require_returns_installed_module():
    """`require` returns the module object for an installed package."""
    import os

    assert require("os") is os


def test_require_missing_module_raises_naming_package():
    """`require` raises ModuleNotFoundError (an ImportError) naming the missing package.

    The error must stay a `ModuleNotFoundError` with `name` preserved so the registry can
    classify optional-dependency skips by module name; it remains an `ImportError` subclass so
    callers guarding on `ImportError` keep working.
    """
    with pytest.raises(ImportError, match="definitely_not_installed_xyz") as excinfo:
        require("definitely_not_installed_xyz")

    assert isinstance(excinfo.value, ModuleNotFoundError)
    assert excinfo.value.name == "definitely_not_installed_xyz"


def test_require_missing_optional_names_extra():
    """A mapped module's error message carries the `steerability[<extra>]` install hint."""
    try:
        require("mergekit")
    except ImportError as exc:
        assert 'steerability[merging]' in str(exc)
    else:
        pytest.skip("mergekit installed; can't test the missing-optional hint path.")


def test_optional_map_matches_pyproject():
    """Every mapped module is in its declared extra and absent from core deps."""
    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    core_dependencies = project.get("dependencies", [])
    extras = project.get("optional-dependencies", {})

    for module_name, extra_name in OPTIONAL_MODULE_EXTRAS.items():
        assert extra_name in extras, f"{module_name!r} maps to undeclared extra {extra_name!r}"

        # requirement strings use hyphenated distribution names; module names use underscores
        normalized_name = module_name.replace("_", "-")
        requirements = extras[extra_name]
        assert any(normalized_name in requirement.replace("_", "-") for requirement in requirements), (
            f"extra {extra_name!r} does not require {module_name!r}: {requirements}"
        )

        assert not any(
            normalized_name in requirement.replace("_", "-") for requirement in core_dependencies
        ), (
            f"{module_name!r} must not appear in core [project.dependencies]"
        )


def test_all_extra_is_eval():
    """`all` is every extra that coexists on every platform; today that is eval alone."""
    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    declared = set(pyproject["project"]["optional-dependencies"]["all"])

    assert declared == {"steerability[eval]"}
