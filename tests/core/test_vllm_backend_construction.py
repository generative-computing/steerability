"""Engine-free test: a `VLLMBackend.__init__` failure after the boot releases the engine."""
import sys
import types

import pytest

from steerability.algorithms.core.execution import BackendSpec
from steerability.backends.vllm import VLLMBackend


class _FakeLLM:
    instances: list = []

    def __init__(self, model, **kwargs):
        self.shutdown_calls = 0
        _FakeLLM.instances.append(self)

    def shutdown(self):
        self.shutdown_calls += 1


def test_post_boot_failure_releases_engine(monkeypatch):
    module = types.ModuleType("vllm")
    module.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", module)
    _FakeLLM.instances.clear()
    # hermetic: no hub lookups, and the realistic failure (tokenizer resolution) raises
    monkeypatch.setattr("steerability.backends.vllm.backend._reject_encoder_decoder", lambda *a, **k: None)

    def failing(source, trust_remote_code=False):
        raise OSError("no such tokenizer")

    monkeypatch.setattr("steerability.backends.vllm.backend._client_tokenizer", failing)
    with pytest.raises(OSError, match="no such tokenizer"):
        VLLMBackend(BackendSpec(kind="vllm", model="tiny"))
    (engine,) = _FakeLLM.instances
    assert engine.shutdown_calls == 1
