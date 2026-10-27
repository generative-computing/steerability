"""Engine-free tests: a `VLLMBackend.__init__` failure after the boot releases the engine and stops
its engine thread, and a released backend refuses new requests."""
import sys
import types

import pytest

from steerability.algorithms.core.execution import BackendSpec
from steerability.backends.vllm import VLLMBackend
from steerability.backends.vllm.engine_thread import EngineThread


class _FakeEngine:
    instances: list = []

    def __init__(self, model_ref, engine_kwargs):
        self.model_ref = model_ref
        self.engine_kwargs = engine_kwargs
        self.shutdown_calls = 0
        _FakeEngine.instances.append(self)

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def fake_engine(monkeypatch):
    """Boot a fake engine through the engine thread, with no vLLM and no hub lookups."""
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    _FakeEngine.instances.clear()

    async def construct(model_ref, engine_kwargs):
        return _FakeEngine(model_ref, engine_kwargs)

    monkeypatch.setattr("steerability.backends.vllm.backend._construct_engine", construct)
    monkeypatch.setattr("steerability.backends.vllm.backend._reject_encoder_decoder", lambda *a, **k: None)
    yield _FakeEngine
    _FakeEngine.instances.clear()


def test_post_boot_failure_releases_engine(fake_engine, monkeypatch):
    # the realistic failure after the boot is tokenizer resolution
    def failing(source, trust_remote_code=False):
        raise OSError("no such tokenizer")

    monkeypatch.setattr("steerability.backends.vllm.backend._client_tokenizer", failing)
    created = []
    original_init = EngineThread.__init__

    def recording_init(self, *, name):
        original_init(self, name=name)
        created.append(self)

    monkeypatch.setattr(EngineThread, "__init__", recording_init)
    with pytest.raises(OSError, match="no such tokenizer"):
        VLLMBackend(BackendSpec(kind="vllm", model="tiny"))
    (engine,) = fake_engine.instances
    assert engine.shutdown_calls == 1
    (engine_thread,) = created
    assert engine_thread.closed


def test_engine_logging_defaults_and_caller_overrides(fake_engine, monkeypatch):
    monkeypatch.setattr("steerability.backends.vllm.backend._client_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr("steerability.backends.vllm.backend._config_layout", lambda *a, **k: None)
    backend = VLLMBackend(BackendSpec(kind="vllm", model="tiny"))
    overridden = VLLMBackend(BackendSpec(
        kind="vllm", model="tiny",
        options={"engine_kwargs": {"enable_log_requests": True, "disable_log_stats": True}},
    ))
    try:
        default_kwargs, override_kwargs = (engine.engine_kwargs for engine in fake_engine.instances)
        assert default_kwargs["enable_log_requests"] is False
        # stats logging keeps vLLM's default (on), which reports preemptions
        assert "disable_log_stats" not in default_kwargs
        assert override_kwargs["enable_log_requests"] is True
        assert override_kwargs["disable_log_stats"] is True
    finally:
        backend.release()
        overridden.release()


def test_released_backend_refuses_requests(fake_engine, monkeypatch):
    monkeypatch.setattr("steerability.backends.vllm.backend._client_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr("steerability.backends.vllm.backend._config_layout", lambda *a, **k: None)
    backend = VLLMBackend(BackendSpec(kind="vllm", model="tiny"))
    session = backend.open_session()
    backend.release()
    backend.release()

    (engine,) = fake_engine.instances
    assert engine.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="was released"):
        backend.open_session()
    with pytest.raises(RuntimeError, match="was released"):
        backend._submit_request({"prompt_token_ids": [1]}, types.SimpleNamespace(output_kind=None))
    assert session.closed is False
