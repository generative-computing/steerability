"""Tests for `CancelToken` and `GenerationCancelled`, and for the in-process session's cancellation
granularity (the token is checked before each decode; a running `model.generate` finishes)."""
import logging
import threading

import pytest
import torch

from steerability.algorithms.core.execution import (
    CancelToken,
    GenerationCancelled,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
    ScoringItem,
)
from steerability.backends.huggingface import HFBackend
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class TestCancelToken:

    def test_cancel_is_idempotent_and_runs_each_callback_once(self):
        token = CancelToken()
        calls = []
        token.on_cancel(lambda: calls.append("first"))
        token.on_cancel(lambda: calls.append("second"))
        assert not token.cancelled
        token.cancel()
        token.cancel()
        assert token.cancelled
        assert calls == ["first", "second"]

    def test_callbacks_run_on_the_cancelling_thread(self):
        token = CancelToken()
        threads = []
        token.on_cancel(lambda: threads.append(threading.get_ident()))
        canceller = threading.Thread(target=token.cancel)
        canceller.start()
        canceller.join(5)
        assert threads == [canceller.ident]

    def test_callback_registered_after_cancellation_runs_at_once(self):
        token = CancelToken()
        token.cancel()
        calls = []
        token.on_cancel(lambda: calls.append("late"))
        assert calls == ["late"]

    def test_callback_exceptions_are_logged_and_do_not_propagate(self, caplog):
        token = CancelToken()
        calls = []

        def failing():
            raise RuntimeError("callback failed")

        token.on_cancel(failing)
        token.on_cancel(lambda: calls.append("after"))
        with caplog.at_level(logging.WARNING, logger="steerability.algorithms.core.execution.cancellation"):
            token.cancel()
        assert calls == ["after"]
        assert any("callback raised" in record.getMessage() for record in caplog.records)

    def test_raise_if_cancelled(self):
        token = CancelToken()
        token.raise_if_cancelled()
        token.cancel()
        with pytest.raises(GenerationCancelled, match="cancelled"):
            token.raise_if_cancelled()

    def test_generation_cancelled_is_a_runtime_error(self):
        assert issubclass(GenerationCancelled, RuntimeError)


@pytest.fixture
def exclusive_backend():
    model = tiny_llama()
    tokenizer = wordlevel_tokenizer()
    from steerability.algorithms.core.execution import BackendSpec

    spec = BackendSpec(kind="huggingface", model=None)
    return HFBackend.adopt(spec, lambda: model, lambda: tokenizer)


def _items(count):
    return [
        GenerationItem(prompt=PreparedPrompt.from_token_ids(torch.tensor([[0, 3 + index, 4]])), seed=index)
        for index in range(count)
    ]


class TestExclusiveSessionCancellation:

    def test_cancelled_token_raises_before_any_decode(self, exclusive_backend, monkeypatch):
        model = exclusive_backend._model_provider()
        decodes = []
        original = model.generate
        monkeypatch.setattr(model, "generate", lambda *args, **kwargs: decodes.append(1) or original(*args, **kwargs))
        token = CancelToken()
        token.cancel()
        with exclusive_backend.open_session() as session:
            with pytest.raises(GenerationCancelled):
                session.generate(_items(1), GenerationParams(max_new_tokens=2, greedy=True), cancel_token=token)
        assert decodes == []

    def test_serial_path_stops_between_items(self, exclusive_backend, monkeypatch):
        model = exclusive_backend._model_provider()
        token = CancelToken()
        decodes = []
        original = model.generate

        def decode_then_cancel(*args, **kwargs):
            decodes.append(1)
            result = original(*args, **kwargs)
            token.cancel()
            return result

        monkeypatch.setattr(model, "generate", decode_then_cancel)
        with exclusive_backend.open_session() as session:
            with pytest.raises(GenerationCancelled):
                # distinct item seeds decode the items one at a time
                session.generate(_items(3), GenerationParams(max_new_tokens=2, greedy=True), cancel_token=token)
        assert decodes == [1]

    def test_batched_path_checks_once_before_the_decode(self, exclusive_backend):
        token = CancelToken()
        token.cancel()
        items = [GenerationItem(prompt=PreparedPrompt.from_token_ids(torch.tensor([[0, 3, 4]]))) for _ in range(2)]
        with exclusive_backend.open_session() as session:
            with pytest.raises(GenerationCancelled):
                session.generate(items, GenerationParams(max_new_tokens=2, greedy=True), cancel_token=token)

    def test_score_checks_the_token(self, exclusive_backend):
        token = CancelToken()
        token.cancel()
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids(torch.tensor([[0, 3, 4]])), ref_output_ids=torch.tensor([[5]]),
        )
        with exclusive_backend.open_session() as session:
            with pytest.raises(GenerationCancelled):
                session.score([item], GenerationParams(), cancel_token=token)

    def test_uncancelled_token_generates_normally(self, exclusive_backend):
        with exclusive_backend.open_session() as session:
            results = session.generate(
                _items(2), GenerationParams(max_new_tokens=2, greedy=True), cancel_token=CancelToken(),
            )
        assert len(results) == 2
