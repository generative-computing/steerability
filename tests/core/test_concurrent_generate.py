"""Concurrent `generate()` calls: the `supports_concurrent_generate` truth table, the prompt lock
(prompt phases never overlap; dispatches overlap only when concurrent generation is supported), and
the reserved `cancel_token` keyword (validation, abort on cancellation, and forwarding to a
driver's rollouts).

Engine-kind pipelines run against a recording backend registered by monkeypatching
`resolve_backend_class`, so no vLLM installation or engine is needed.
"""
import threading
import time

import pytest
import torch

from steerability.algorithms.core.execution import (
    BackendSpec,
    CancelToken,
    GenerationCancelled,
    ItemResult,
    ModelAccess,
    ModelFacts,
    Requirements,
)
from steerability.algorithms.core.execution.session_utils import session_generate
from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.base import DecodingDriver
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VLLM_SPEC = BackendSpec(kind="vllm", model="tiny-engine-model")


class _RecordingSession:
    """An engine session double that records each call's thread and wall-clock interval.

    A call blocks until its backend's `release` event is set (when one is set on the backend),
    or until its `cancel_token` is cancelled, whichever comes first.
    """

    def __init__(self, backend):
        self._backend = backend

    @property
    def tokenizer(self):
        return self._backend.tokenizer

    @property
    def layout(self) -> ModelFacts:
        return ModelFacts(
            num_layers=2, hidden_size=16, num_attention_heads=2, head_dim=8, dtype="float32",
            model_fingerprint=None, model_type="llama", fingerprint_grade="config",
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def close(self):
        return None

    def generate(self, items, params, *, cancel_token=None):
        backend = self._backend
        started = time.monotonic()
        backend.tokens_seen.append(cancel_token)
        if cancel_token is not None:
            cancel_token.on_cancel(backend.abort_seen.set)
        deadline = started + backend.hold_seconds
        while time.monotonic() < deadline:
            if cancel_token is not None and cancel_token.cancelled:
                backend.calls.append((threading.get_ident(), started, time.monotonic(), "cancelled"))
                raise GenerationCancelled("cancelled")
            if backend.release is not None and backend.release.is_set():
                break
            time.sleep(0.01)
        backend.calls.append((threading.get_ident(), started, time.monotonic(), "done"))
        return [
            ItemResult(index=index, output=Output(
                output_ids=torch.tensor([[3, 1]]), adapted_input_ids=item.prompt.token_ids, finish_reasons=("eos",),
            ))
            for index, item in enumerate(items)
        ]

    def score(self, items, params, *, cancel_token=None):
        return torch.zeros(len(items), items[0].ref_output_ids.shape[-1])


class _RecordingBackend:
    """An engine-kind backend double; its sessions record calls (see `_RecordingSession`)."""

    hold_seconds = 0.3
    release: threading.Event | None = None
    calls: list = []
    tokens_seen: list = []
    abort_seen = threading.Event()

    def __init__(self, spec, artifacts=()):
        self.spec = spec
        self.tokenizer = wordlevel_tokenizer()
        self._discovery = None

    @classmethod
    def capabilities_for_spec(cls, spec):
        from steerability.backends.vllm import VLLMBackend

        return VLLMBackend.capabilities_for_spec(spec)

    def negotiated_capabilities(self):
        return self.capabilities_for_spec(self.spec)

    def open_session(self):
        return _RecordingSession(self)

    def stage_artifacts(self, payloads):
        return None

    def release(self):
        return None

    @classmethod
    def reset(cls):
        cls.hold_seconds = 0.3
        cls.release = None
        cls.calls = []
        cls.tokens_seen = []
        cls.abort_seen = threading.Event()


@pytest.fixture
def recording_engine(monkeypatch):
    import steerability.algorithms.core.execution.backend as backend_module
    import steerability.algorithms.core.steering_pipeline as pipeline_module

    original = backend_module.resolve_backend_class

    def resolver(spec):
        if spec.kind == "huggingface":
            return original(spec)
        return _RecordingBackend

    monkeypatch.setattr(backend_module, "resolve_backend_class", resolver)
    monkeypatch.setattr(pipeline_module, "resolve_backend_class", resolver)
    _RecordingBackend.reset()
    yield _RecordingBackend
    _RecordingBackend.reset()


class _OverlapProbe(InputControl):
    """A token-level input control that records whether two `adapt` calls ever overlapped."""

    Args = None
    supports_batching = True

    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0

    def adapt(self, input_ids, runtime_kwargs=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        self.active -= 1
        return input_ids

    def requirements(self):
        return Requirements()

    def steer_access(self):
        return ModelAccess.FACTS


class _ProbeDriver(DecodingDriver):
    """A driver that rolls out once through the session it receives and records that session."""

    Args = None

    def _configure(self):
        self.seen_session = None

    def requirements(self):
        return Requirements()

    def steer_access(self):
        return ModelAccess.FACTS

    def decode(self, input_ids, attention_mask, model, logits_processors, stopping_criteria,
               runtime_kwargs, session=None, **gen_kwargs):
        self.seen_session = session
        return session_generate(session, input_ids, attention_mask, **gen_kwargs)


def _engine_pipeline(controls=()):
    pipeline = SteeringPipeline(controls=list(controls), backend=VLLM_SPEC, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    return pipeline


def _generate_in_threads(pipeline, count=2, **gen_kwargs):
    errors = []

    def call(index):
        try:
            pipeline.generate(input_ids=torch.tensor([[0, 3 + index, 4]]), max_new_tokens=2, **gen_kwargs)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=call, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return errors


def _overlap(first, second) -> bool:
    return first[1] < second[2] and second[1] < first[2]


class TestSupportsConcurrentGenerate:
    """The truth table, read from the spec before `steer()` (no backend is constructed)."""

    def test_huggingface_backend_is_false(self):
        pipeline = SteeringPipeline(model=tiny_llama(), tokenizer=wordlevel_tokenizer())
        assert pipeline.supports_concurrent_generate is False

    def test_vllm_spec_without_a_driver_is_true(self):
        pipeline = SteeringPipeline(backend=VLLM_SPEC)
        assert pipeline.supports_concurrent_generate is True

    def test_vllm_serve_spec_without_a_driver_is_true(self):
        spec = BackendSpec(kind="vllm-serve", model="m", options={"base_url": "http://localhost:1"})
        assert SteeringPipeline(backend=spec).supports_concurrent_generate is True

    def test_enabled_driver_is_false(self):
        pipeline = SteeringPipeline(controls=[_ProbeDriver()], backend=VLLM_SPEC)
        assert pipeline.supports_concurrent_generate is False

    def test_disabled_driver_is_true(self):
        driver = _ProbeDriver()
        driver.enabled = False
        pipeline = SteeringPipeline(controls=[driver], backend=VLLM_SPEC)
        assert pipeline.supports_concurrent_generate is True

    def test_steered_pipeline_reads_the_negotiated_capabilities(self, recording_engine):
        pipeline = _engine_pipeline()
        assert pipeline.supports_concurrent_generate is True


class TestPromptLock:

    def test_dispatches_overlap_and_prompt_phases_do_not(self, recording_engine):
        probe = _OverlapProbe()
        pipeline = _engine_pipeline([probe])
        assert pipeline.supports_concurrent_generate

        errors = _generate_in_threads(pipeline, count=3)

        assert errors == []
        assert probe.max_active == 1
        calls = recording_engine.calls
        assert len(calls) == 3
        assert len({thread for thread, *_ in calls}) == 3
        assert any(_overlap(first, second) for i, first in enumerate(calls) for second in calls[i + 1:])

    def test_unsupported_pipeline_serializes_dispatches(self, recording_engine):
        pipeline = _engine_pipeline([_ProbeDriver()])
        assert not pipeline.supports_concurrent_generate

        errors = _generate_in_threads(pipeline, count=2)

        assert errors == []
        first, second = recording_engine.calls
        assert not _overlap(first, second)

    def test_in_process_pipeline_serializes_calls(self, monkeypatch):
        from steerability.backends.huggingface import ExclusiveSession

        intervals = []
        original = ExclusiveSession.generate

        def recording(self, items, params, *, cancel_token=None):
            started = time.monotonic()
            time.sleep(0.2)
            result = original(self, items, params, cancel_token=cancel_token)
            intervals.append((None, started, time.monotonic()))
            return result

        monkeypatch.setattr(ExclusiveSession, "generate", recording)
        pipeline = SteeringPipeline(model=tiny_llama(), tokenizer=wordlevel_tokenizer())
        pipeline.steer()
        errors = _generate_in_threads(pipeline, count=2, do_sample=False)

        assert errors == []
        assert len(intervals) == 2
        assert not _overlap(*intervals)

    def test_compute_logprobs_runs_under_the_lock(self, recording_engine):
        pipeline = _engine_pipeline()
        holder_started = threading.Event()
        holder_release = threading.Event()

        def hold_lock():
            with pipeline._prompt_lock:
                holder_started.set()
                holder_release.wait(5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert holder_started.wait(5)
        finished = threading.Event()

        def score():
            pipeline.compute_logprobs(torch.tensor([[0, 3, 4]]), ref_output_ids=torch.tensor([[5]]))
            finished.set()

        scorer = threading.Thread(target=score)
        scorer.start()
        assert not finished.wait(0.3)
        holder_release.set()
        assert finished.wait(5)
        holder.join(5)
        scorer.join(5)


class TestCancelTokenOnGenerate:

    def test_wrong_type_raises_type_error(self, recording_engine):
        pipeline = _engine_pipeline()
        with pytest.raises(TypeError, match="cancel_token must be a CancelToken"):
            pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=object())

    def test_token_reaches_the_session(self, recording_engine):
        pipeline = _engine_pipeline()
        token = CancelToken()
        pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=token)
        assert recording_engine.tokens_seen == [token]

    def test_cancelling_aborts_the_session_call(self, recording_engine):
        recording_engine.hold_seconds = 30.0
        pipeline = _engine_pipeline()
        token = CancelToken()
        errors = []

        def call():
            try:
                pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=token)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=call)
        started = time.monotonic()
        thread.start()
        time.sleep(0.2)
        token.cancel()
        thread.join(10)

        assert not thread.is_alive()
        assert time.monotonic() - started < 5
        assert len(errors) == 1 and isinstance(errors[0], GenerationCancelled)
        assert recording_engine.abort_seen.is_set()

    def test_cancelled_token_before_the_call_raises(self, recording_engine):
        pipeline = _engine_pipeline()
        token = CancelToken()
        token.cancel()
        with pytest.raises(GenerationCancelled):
            pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=token)

    def test_steered_session_forwards_the_token_to_a_driver_rollout(self, recording_engine):
        driver = _ProbeDriver()
        pipeline = _engine_pipeline([driver])
        token = CancelToken()
        pipeline.generate(input_ids=torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=token)
        assert driver.seen_session.cancel_token is token
        assert recording_engine.tokens_seen == [token]

    def test_generate_methods_accept_the_token(self, recording_engine):
        pipeline = _engine_pipeline()
        token = CancelToken()
        pipeline.generate_tokens(torch.tensor([[0, 3, 4]]), max_new_tokens=2, cancel_token=token)
        assert recording_engine.tokens_seen == [token]
