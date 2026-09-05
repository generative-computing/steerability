"""CPU-only tests for the offline vLLM engine-boot environment policy.

These import the torch-free `steerability.backends.vllm.environment` module directly and need
neither vLLM nor a GPU.
"""
import os

import pytest

from steerability.backends.vllm.environment import (
    FLASHINFER_SAMPLER_VARIABLE,
    HOOK_WORKER_VARIABLE,
    engine_boot_environment,
    engine_environment,
    serve_environment,
)

ALL_VARIABLES = (HOOK_WORKER_VARIABLE, FLASHINFER_SAMPLER_VARIABLE)


@pytest.fixture
def clean_environ(monkeypatch):
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_plain_engine_only_defaults_the_sampler():
    forced, defaults = engine_boot_environment(hook_plugin=False)
    assert forced == {}
    assert defaults == {FLASHINFER_SAMPLER_VARIABLE: "0"}


def test_hook_plugin_engine_forces_worker():
    forced, defaults = engine_boot_environment(hook_plugin=True)
    assert forced == {HOOK_WORKER_VARIABLE: "unified"}
    assert "VLLM_USE_V2_MODEL_RUNNER" not in forced
    assert defaults == {FLASHINFER_SAMPLER_VARIABLE: "0"}


def test_applies_and_restores_unset(clean_environ):
    forced, defaults = engine_boot_environment(hook_plugin=True)
    with engine_environment(forced, defaults) as applied:
        assert os.environ[HOOK_WORKER_VARIABLE] == "unified"
        assert os.environ[FLASHINFER_SAMPLER_VARIABLE] == "0"
        assert set(applied) == set(ALL_VARIABLES)
    for name in ALL_VARIABLES:
        assert name not in os.environ


def test_explicit_caller_setting_wins(clean_environ, monkeypatch):
    monkeypatch.setenv(FLASHINFER_SAMPLER_VARIABLE, "1")
    forced, defaults = engine_boot_environment(hook_plugin=False)
    with engine_environment(forced, defaults) as applied:
        assert os.environ[FLASHINFER_SAMPLER_VARIABLE] == "1"
        assert applied == {}
    assert os.environ[FLASHINFER_SAMPLER_VARIABLE] == "1"


def test_forced_overrides_and_restores(clean_environ, monkeypatch):
    monkeypatch.setenv(HOOK_WORKER_VARIABLE, "probe_hidden_states")
    forced, defaults = engine_boot_environment(hook_plugin=True)
    with engine_environment(forced, defaults):
        assert os.environ[HOOK_WORKER_VARIABLE] == "unified"
    assert os.environ[HOOK_WORKER_VARIABLE] == "probe_hidden_states"


def test_restores_on_exception(clean_environ):
    forced, defaults = engine_boot_environment(hook_plugin=True)
    with pytest.raises(RuntimeError):
        with engine_environment(forced, defaults):
            raise RuntimeError("boom")
    for name in ALL_VARIABLES:
        assert name not in os.environ


def test_serve_environment_fills_defaults_and_forces_worker():
    assert serve_environment(True, base={"PATH": "/bin"}) == {
        "PATH": "/bin",
        FLASHINFER_SAMPLER_VARIABLE: "0",
        HOOK_WORKER_VARIABLE: "unified",
    }
    assert serve_environment(False, base={}) == {FLASHINFER_SAMPLER_VARIABLE: "0"}


def test_serve_environment_caller_default_wins_forced_overrides():
    base = {FLASHINFER_SAMPLER_VARIABLE: "1", HOOK_WORKER_VARIABLE: "probe_hidden_states"}
    result = serve_environment(True, base=base)
    assert result[FLASHINFER_SAMPLER_VARIABLE] == "1"
    assert result[HOOK_WORKER_VARIABLE] == "unified"
    assert base == {FLASHINFER_SAMPLER_VARIABLE: "1", HOOK_WORKER_VARIABLE: "probe_hidden_states"}


def test_serve_environment_matches_engine_environment(clean_environ):
    forced, defaults = engine_boot_environment(hook_plugin=True)
    with engine_environment(forced, defaults) as applied:
        written = {name: os.environ[name] for name in applied}
    assert written == serve_environment(True, base={})
