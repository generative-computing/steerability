"""Tests for the request dispatchers.

The lock-leader collator tests cover batching under deterministic gating, batch keys, runtime-kwargs
collation, poison isolation, cancellation, strict serialization, and loop reuse. The per-request
dispatcher tests cover one call per request, the concurrency limiter, error isolation, request seeds, and
cancellation of a request's generation.
"""
import threading

import anyio
import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.execution import CancelToken
from steerability.evaluation.batching import LockLeaderCollator, PerRequestDispatcher, request_seed
from tests.evaluation.conftest import StubSteeringPipeline


def _collator(pipeline=None, *, max_batch_size=4, declared_scopes=None,
              static_runtime_kwargs=None, prompt_path="messages") -> tuple[LockLeaderCollator, StubSteeringPipeline]:
    pipeline = pipeline if pipeline is not None else StubSteeringPipeline()
    collator = LockLeaderCollator(
        pipeline,
        max_batch_size=max_batch_size,
        prompt_path=prompt_path,
        declared_scopes={"spans": "row"} if declared_scopes is None else declared_scopes,
        static_runtime_kwargs=static_runtime_kwargs or {},
    )
    return collator, pipeline


def _admit(collator, prompt, *, gen_kwargs=None, per_sample=None, num_choices=1):
    return collator.admit(prompt, gen_kwargs or {"max_new_tokens": 4}, per_sample or {}, num_choices)


class TestAdmission:
    def test_call_scoped_per_sample_key_raises(self):
        collator, _ = _collator(declared_scopes={"spans": "row", "canned_responses": "call"})
        with pytest.raises(ValueError, match="declared 'call'-scoped.*ProviderOptions.runtime_kwargs"):
            _admit(collator, [{"role": "user", "content": "q"}], per_sample={"canned_responses": {"a": "b"}})

    def test_undeclared_per_sample_key_is_inert(self, caplog):
        collator, _ = _collator()
        with caplog.at_level("INFO", logger="steerability.evaluation.batching"):
            first = _admit(collator, [{"role": "user", "content": "q"}], per_sample={"other": 1, "spans": ["x"]})
            _admit(collator, [{"role": "user", "content": "q2"}], per_sample={"other": 2, "spans": ["y"]})
        assert first.per_sample_runtime_kwargs == {"spans": ["x"]}
        assert collator.inert_runtime_kwargs == frozenset({"other"})
        inert_lines = [r for r in caplog.records if "is inert on this arm" in r.getMessage()]
        assert len(inert_lines) == 1

    def test_inert_per_sample_key_does_not_split_batch_keys(self):
        collator, _ = _collator()
        with_inert = _admit(collator, [{"role": "user", "content": "a"}], per_sample={"other": 1})
        without = _admit(collator, [{"role": "user", "content": "b"}], per_sample={})
        assert with_inert.batch_key == without.batch_key

    def test_key_in_both_tiers_raises(self):
        collator, _ = _collator(static_runtime_kwargs={"spans": ["x"]})
        with pytest.raises(ValueError, match="both per sample.*and statically"):
            _admit(collator, [{"role": "user", "content": "q"}], per_sample={"spans": ["a"]})

    def test_key_in_both_tiers_raises_even_when_undeclared(self):
        collator, _ = _collator(declared_scopes={}, static_runtime_kwargs={"spans": ["x"]})
        with pytest.raises(ValueError, match="both per sample.*and statically"):
            _admit(collator, [{"role": "user", "content": "q"}], per_sample={"spans": ["a"]})

    def test_closed_collator_refuses_admission(self):
        collator, _ = _collator()
        collator.close()
        assert collator.closed
        with pytest.raises(RuntimeError, match="closed"):
            _admit(collator, [{"role": "user", "content": "q"}])


class TestBatchKeys:
    def test_equal_config_and_key_set_share_a_key(self):
        collator, _ = _collator()
        first = _admit(collator, "a", gen_kwargs={"max_new_tokens": 4}, per_sample={"spans": ["x"]})
        second = _admit(collator, "b", gen_kwargs={"max_new_tokens": 4}, per_sample={"spans": ["y", "z"]})
        assert first.batch_key == second.batch_key  # values are excluded from the key

    def test_different_gen_kwargs_split_keys(self):
        collator, _ = _collator()
        first = _admit(collator, "a", gen_kwargs={"max_new_tokens": 4})
        second = _admit(collator, "b", gen_kwargs={"max_new_tokens": 8})
        assert first.batch_key != second.batch_key

    def test_different_per_sample_key_sets_split_keys(self):
        collator, _ = _collator(declared_scopes={"spans": "row", "targets": "row"})
        first = _admit(collator, "a", per_sample={"spans": ["x"]})
        second = _admit(collator, "b", per_sample={"targets": ["y"]})
        assert first.batch_key != second.batch_key

    def test_multi_candidate_keys_are_isolated(self):
        collator, _ = _collator()
        single = _admit(collator, "a", num_choices=1)
        multi = _admit(collator, "b", num_choices=3)
        assert single.batch_key != multi.batch_key


def _run_concurrent(collator, records, *, release: threading.Event, cancel_indices=frozenset(),
                    cancel_delay=0.2, release_delay=0.4):
    """Serve `records` concurrently; release the stub's gate after they enqueue.

    Returns (results, errors) aligned with `records`; a cancelled record holds the string
    "cancelled" in its error slot.
    """
    results: list = [None] * len(records)
    errors: list = [None] * len(records)
    scopes: dict[int, anyio.CancelScope] = {}

    async def serve(index):
        with anyio.CancelScope() as scope:
            scopes[index] = scope
            try:
                results[index] = await collator.serve(records[index])
            except Exception as error:
                errors[index] = error
        if scope.cancelled_caught:
            errors[index] = "cancelled"

    async def main():
        async with anyio.create_task_group() as tg:
            for index in range(len(records)):
                tg.start_soon(serve, index)
            if cancel_indices:
                await anyio.sleep(cancel_delay)
                for index in cancel_indices:
                    scopes[index].cancel()
            await anyio.sleep(release_delay)
            release.set()

    anyio.run(main)
    return results, errors


class TestLeaderProtocol:
    def test_gated_requests_batch_with_row_aligned_kwargs(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=3)
        records = [
            _admit(collator, [{"role": "user", "content": f"q{i}"}], per_sample={"spans": [f"s{i}"]})
            for i in range(5)
        ]
        results, errors = _run_concurrent(collator, records, release=release)

        assert all(error is None for error in errors)
        assert all(result is not None for result in results)
        outputs = {int(result.output_ids[0, 0]) for result in results}
        assert len(outputs) == 5  # each record resolved to its own output
        sizes = sorted(len(call["messages"]) for call in pipeline.calls)
        assert sum(sizes) == 5
        assert max(sizes) == 3  # one dispatch carried min(K, max_batch_size)
        full = next(call for call in pipeline.calls if len(call["messages"]) == 3)
        rows = [conversation[0]["content"] for conversation in full["messages"]]
        assert full["runtime_kwargs"]["spans"] == [[f"s{row[1:]}"] for row in rows]  # row-aligned

    def test_seeded_requests_share_one_dispatch(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=4)
        seeded = {"max_new_tokens": 4, "seed": 42, "seed_scope": "dispatch"}
        records = [
            _admit(collator, [{"role": "user", "content": f"q{i}"}], gen_kwargs=seeded)
            for i in range(4)
        ]
        results, errors = _run_concurrent(collator, records, release=release)
        assert all(error is None for error in errors)
        assert len(pipeline.calls) == 1  # one seeded dispatch over all four prompts
        assert len(pipeline.calls[0]["messages"]) == 4
        assert pipeline.calls[0]["gen_kwargs"]["seed_scope"] == "dispatch"

    def test_supports_batching_false_dispatches_singletons(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline(supports_batching=False)
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=1)
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(4)]
        results, errors = _run_concurrent(collator, records, release=release)
        assert all(error is None for error in errors)
        assert [len(call["messages"]) for call in pipeline.calls] == [1, 1, 1, 1]

    def test_waiter_cancelled_while_queued_never_dispatches(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=1)  # leader takes only itself
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(3)]
        results, errors = _run_concurrent(collator, records, release=release, cancel_indices={2})

        assert errors[2] == "cancelled"
        dispatched = [call["messages"][0][0]["content"] for call in pipeline.calls]
        assert "q2" not in dispatched
        assert results[0] is not None and results[1] is not None

    def test_leader_cancelled_mid_flight_completes_cobatched_records(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=4)
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(3)]
        # index 0 becomes the leader (first to enqueue); cancel it while its dispatch is gated;
        # the in-flight generation is awaited rather than abandoned, so every co-batched record
        # still completes (whether the leader's own task then observes the cancellation depends on
        # checkpoint placement)
        results, errors = _run_concurrent(collator, records, release=release, cancel_indices={0})

        assert results[1] is not None and results[2] is not None
        assert records[0].output is not None  # the leader's own record completed in the thread
        assert len(pipeline.calls) == 1  # one dispatch carried all three; nothing re-dispatched

    def test_reuse_across_two_anyio_run_loops(self):
        collator, pipeline = _collator()

        async def one_request(tag):
            record = _admit(collator, [{"role": "user", "content": tag}])
            return await collator.serve(record)

        first = anyio.run(one_request, "first")
        second = anyio.run(one_request, "second")
        assert first is not None and second is not None
        assert len(pipeline.calls) == 2


class TestStaticTier:
    def test_static_row_scoped_value_broadcasts_per_row(self):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=4, static_runtime_kwargs={"spans": ["x"]})
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(3)]
        results, errors = _run_concurrent(collator, records, release=release)

        assert all(error is None for error in errors)
        (call,) = pipeline.calls
        assert len(call["messages"]) == 3
        assert call["runtime_kwargs"]["spans"] == [["x"], ["x"], ["x"]]

    def test_static_call_scoped_value_passes_through(self):
        pipeline = StubSteeringPipeline()
        artifact = {"a": "b"}
        collator, _ = _collator(
            pipeline, declared_scopes={"canned_responses": "call"},
            static_runtime_kwargs={"canned_responses": artifact},
        )

        async def main():
            record = _admit(collator, [{"role": "user", "content": "q"}])
            return await collator.serve(record)

        anyio.run(main)
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["canned_responses"] is artifact

    def test_undeclared_static_key_is_inert(self, caplog):
        pipeline = StubSteeringPipeline()
        with caplog.at_level("INFO", logger="steerability.evaluation.batching"):
            collator, _ = _collator(pipeline, declared_scopes={}, static_runtime_kwargs={"other": 1})
            inert_lines = [r for r in caplog.records if "is inert on this arm" in r.getMessage()]
            assert len(inert_lines) == 1

            async def main():
                record = _admit(collator, [{"role": "user", "content": "q"}])
                return await collator.serve(record)

            anyio.run(main)
        (call,) = pipeline.calls
        assert "other" not in call["runtime_kwargs"]
        assert collator.inert_runtime_kwargs == frozenset({"other"})

    def test_static_row_scoped_singleton_multi_candidate_broadcasts_one_row(self):
        pipeline = StubSteeringPipeline()
        collator, _ = _collator(pipeline, static_runtime_kwargs={"spans": ["x"]})

        async def main():
            record = _admit(collator, [{"role": "user", "content": "q"}], num_choices=3)
            return await collator.serve(record)

        anyio.run(main)
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["spans"] == [["x"]]


class TestPoisonIsolation:
    def test_shape_failure_reruns_serially_and_warns_once(self, caplog):
        release = threading.Event()
        pipeline = StubSteeringPipeline()
        pipeline.gate = release
        pipeline.fail_above_batch_size = 1  # every member succeeds serially
        collator, _ = _collator(pipeline, max_batch_size=4)
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(3)]
        with caplog.at_level("WARNING", logger="steerability.evaluation.batching"):
            results, errors = _run_concurrent(collator, records, release=release)
        assert all(error is None for error in errors)
        assert all(result is not None for result in results)
        warnings_seen = [
            record for record in caplog.records
            if "succeeded serially" in record.getMessage()
        ]
        assert len(warnings_seen) == 1

    def test_poison_sample_fails_alone(self):
        release = threading.Event()

        class PoisonPipeline(StubSteeringPipeline):
            def generate(self, *, messages=None, text=None, runtime_kwargs=None, return_output=True, **kw):
                prompts = messages if messages is not None else text
                if any(conversation[0]["content"] == "poison" for conversation in prompts):
                    self.calls.append({"messages": messages, "text": text,
                                       "runtime_kwargs": runtime_kwargs, "gen_kwargs": dict(kw)})
                    if self.gate is not None and not self._gate_used:
                        self._gate_used = True
                        self.gate.wait(10)
                    raise ValueError("bad sample")
                return super().generate(messages=messages, text=text, runtime_kwargs=runtime_kwargs,
                                        return_output=return_output, **kw)

        pipeline = PoisonPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=4)
        prompts = ["ok0", "poison", "ok1"]
        records = [_admit(collator, [{"role": "user", "content": prompt}]) for prompt in prompts]
        results, errors = _run_concurrent(collator, records, release=release)
        assert results[0] is not None and results[2] is not None
        assert isinstance(errors[1], ValueError) and "bad sample" in str(errors[1])

    def test_short_output_list_leaves_no_record_unresolved(self):
        class ShortPipeline(StubSteeringPipeline):
            def generate(self, **kw):
                return super().generate(**kw)[:-1]  # one Output fewer than prompts

        release = threading.Event()
        pipeline = ShortPipeline()
        pipeline.gate = release
        collator, _ = _collator(pipeline, max_batch_size=4)
        records = [_admit(collator, [{"role": "user", "content": f"q{i}"}]) for i in range(3)]
        _, errors = _run_concurrent(collator, records, release=release)
        assert not any(record.output is None and record.error is None for record in records)
        assert all(isinstance(error, RuntimeError) and "output" in str(error) for error in errors)

    def test_singleton_failure_lands_on_its_record(self):
        class FailingPipeline(StubSteeringPipeline):
            def generate(self, **kw):
                raise RuntimeError("boom")

        collator, _ = _collator(FailingPipeline(), max_batch_size=4)

        async def main():
            record = _admit(collator, [{"role": "user", "content": "q"}])
            return await collator.serve(record)

        with pytest.raises(RuntimeError, match="boom"):
            anyio.run(main)


def _dispatcher(pipeline=None, *, max_concurrency=4, declared_scopes=None, static_runtime_kwargs=None,
                prompt_path="messages") -> tuple[PerRequestDispatcher, StubSteeringPipeline]:
    pipeline = pipeline if pipeline is not None else StubSteeringPipeline(supports_concurrent_generate=True)
    dispatcher = PerRequestDispatcher(
        pipeline,
        max_concurrency=max_concurrency,
        prompt_path=prompt_path,
        declared_scopes={"spans": "row"} if declared_scopes is None else declared_scopes,
        static_runtime_kwargs=static_runtime_kwargs or {},
    )
    return dispatcher, pipeline


class _BlockingPipeline(StubSteeringPipeline):
    """A stub whose `generate` records its interval and blocks until released or cancelled."""

    def __init__(self, *, release: threading.Event | None = None, hold_seconds: float = 0.3):
        super().__init__(supports_concurrent_generate=True)
        self.release = release
        self.hold_seconds = hold_seconds
        self.intervals: list[tuple[float, float]] = []
        self.active = 0
        self.peak = 0
        self._counter = threading.Lock()

    def generate(self, **kwargs):
        import time

        with self._counter:
            self.active += 1
            self.peak = max(self.peak, self.active)
        started = time.monotonic()
        token = kwargs.get("cancel_token")
        try:
            deadline = started + self.hold_seconds
            while time.monotonic() < deadline:
                if token is not None and token.cancelled:
                    from steerability.algorithms.core.execution import GenerationCancelled

                    raise GenerationCancelled("cancelled")
                if self.release is not None and self.release.is_set():
                    break
                time.sleep(0.01)
            return super().generate(**kwargs)
        finally:
            self.intervals.append((started, time.monotonic()))
            with self._counter:
                self.active -= 1


def _serve_all(dispatcher, records):
    results: list = [None] * len(records)
    errors: list = [None] * len(records)

    async def serve(index):
        try:
            results[index] = await dispatcher.serve(records[index])
        except Exception as error:
            errors[index] = error

    async def main():
        async with anyio.create_task_group() as tg:
            for index in range(len(records)):
                tg.start_soon(serve, index)

    anyio.run(main)
    return results, errors


class TestPerRequestDispatcher:
    def test_every_record_dispatches_as_its_own_single_prompt_call(self):
        pipeline = _BlockingPipeline(hold_seconds=0.3)
        dispatcher, _ = _dispatcher(pipeline, max_concurrency=8)
        records = [_admit(dispatcher, [{"role": "user", "content": f"q{i}"}], per_sample={"spans": [f"s{i}"]})
                   for i in range(4)]
        results, errors = _serve_all(dispatcher, records)

        assert errors == [None] * 4
        assert all(result is not None for result in results)
        assert len(pipeline.calls) == 4
        # a batch of one conversation per call; the dispatcher does not wait for other records
        assert all(len(call["messages"]) == 1 for call in pipeline.calls)
        assert pipeline.peak == 4
        dispatched = sorted(call["messages"][0][0]["content"] for call in pipeline.calls)
        assert dispatched == ["q0", "q1", "q2", "q3"]
        spans = sorted(call["runtime_kwargs"]["spans"][0][0] for call in pipeline.calls)
        assert spans == ["s0", "s1", "s2", "s3"]  # per-sample values arrive as one-element lists

    def test_limiter_bounds_the_calls_in_flight(self):
        pipeline = _BlockingPipeline(hold_seconds=0.2)
        dispatcher, _ = _dispatcher(pipeline, max_concurrency=2)
        records = [_admit(dispatcher, [{"role": "user", "content": f"q{i}"}]) for i in range(5)]
        _, errors = _serve_all(dispatcher, records)
        assert errors == [None] * 5
        assert pipeline.peak == 2

    def test_each_record_receives_its_own_error(self):
        class PoisonPipeline(StubSteeringPipeline):
            def generate(self, *, messages=None, **kwargs):
                (conversation,) = messages
                if conversation[0]["content"] == "poison":
                    raise ValueError("bad sample")
                return super().generate(messages=messages, **kwargs)

        dispatcher, _ = _dispatcher(PoisonPipeline(supports_concurrent_generate=True))
        records = [_admit(dispatcher, [{"role": "user", "content": prompt}]) for prompt in ("ok", "poison", "ok2")]
        results, errors = _serve_all(dispatcher, records)
        assert results[0] is not None and results[2] is not None
        assert isinstance(errors[1], ValueError) and "bad sample" in str(errors[1])
        assert errors[0] is None and errors[2] is None

    def test_static_and_per_sample_kwargs_partition_like_the_collator(self, caplog):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        artifact = {"a": "b"}
        with caplog.at_level("INFO", logger="steerability.evaluation.batching"):
            dispatcher, _ = _dispatcher(
                pipeline,
                declared_scopes={"spans": "row", "canned_responses": "call"},
                static_runtime_kwargs={"canned_responses": artifact, "other": 1},
            )
        record = _admit(dispatcher, [{"role": "user", "content": "q"}], per_sample={"spans": ["x"], "unused": 2})
        _serve_all(dispatcher, [record])
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["canned_responses"] is artifact
        assert call["runtime_kwargs"]["spans"] == [["x"]]
        assert "other" not in call["runtime_kwargs"] and "unused" not in call["runtime_kwargs"]
        assert dispatcher.inert_runtime_kwargs == frozenset({"other", "unused"})
        with pytest.raises(ValueError, match="both per sample"):
            _admit(dispatcher, [{"role": "user", "content": "q"}], per_sample={"canned_responses": {}})
        scoped, _ = _dispatcher(declared_scopes={"canned_responses": "call"})
        with pytest.raises(ValueError, match="declared 'call'-scoped"):
            _admit(scoped, [{"role": "user", "content": "q"}], per_sample={"canned_responses": {}})

    def test_static_row_scoped_value_is_one_row(self):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        dispatcher, _ = _dispatcher(pipeline, static_runtime_kwargs={"spans": ["x"]})
        _serve_all(dispatcher, [_admit(dispatcher, [{"role": "user", "content": "q"}])])
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["spans"] == [["x"]]

    def test_multi_candidate_request_dispatches_a_bare_conversation(self):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        dispatcher, _ = _dispatcher(pipeline)
        conversation = [{"role": "user", "content": "q"}]
        results, errors = _serve_all(dispatcher, [_admit(dispatcher, conversation, num_choices=3)])
        assert errors == [None]
        (call,) = pipeline.calls
        assert call["messages"] == conversation
        assert call["gen_kwargs"]["n"] == 3
        assert results[0].output_ids.shape[0] == 3

    def test_closed_dispatcher_refuses_admission(self):
        dispatcher, _ = _dispatcher()
        dispatcher.close()
        assert dispatcher.closed
        with pytest.raises(RuntimeError, match="closed"):
            _admit(dispatcher, [{"role": "user", "content": "q"}])

    def test_max_concurrency_is_validated(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            _dispatcher(max_concurrency=0)

    def test_a_cancel_token_reaches_pipeline_generate(self):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        dispatcher, _ = _dispatcher(pipeline)
        record = _admit(dispatcher, [{"role": "user", "content": "q"}])
        _serve_all(dispatcher, [record])
        (call,) = pipeline.calls
        token = call["gen_kwargs"]["cancel_token"]
        assert isinstance(token, CancelToken)
        assert token is record.cancel_token and not token.cancelled

    def test_cancelling_the_serving_task_cancels_the_token(self):
        import time

        pipeline = _BlockingPipeline(hold_seconds=30.0)
        dispatcher, _ = _dispatcher(pipeline)
        record = _admit(dispatcher, [{"role": "user", "content": "q"}])
        outcome = {}

        async def main():
            with anyio.move_on_after(0.3) as scope:
                await dispatcher.serve(record)
            outcome["cancelled"] = scope.cancelled_caught
            outcome["returned_at"] = time.monotonic()

        started = time.monotonic()
        anyio.run(main)
        assert outcome["cancelled"] is True
        assert outcome["returned_at"] - started < 5  # the task returned at once
        assert record.cancel_token.cancelled
        deadline = time.monotonic() + 5
        while not pipeline.intervals and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pipeline.intervals  # the worker thread observed the token and ended
        assert record.output is None

    def test_reuse_across_two_anyio_run_loops(self):
        dispatcher, pipeline = _dispatcher()

        async def one_request(tag):
            return await dispatcher.serve(_admit(dispatcher, [{"role": "user", "content": tag}]))

        assert anyio.run(one_request, "first") is not None
        assert anyio.run(one_request, "second") is not None
        assert len(pipeline.calls) == 2


class TestRequestSeeds:
    @staticmethod
    def _seed_of(prompt, *, seed=7, per_sample=None):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        dispatcher, _ = _dispatcher(pipeline)
        record = _admit(dispatcher, prompt, gen_kwargs={"max_new_tokens": 4, "seed": seed, "seed_scope": "dispatch"},
                        per_sample=per_sample or {})
        _serve_all(dispatcher, [record])
        (call,) = pipeline.calls
        return call["gen_kwargs"]["seed"]

    def test_seed_is_derived_from_the_base_seed_and_the_request(self):
        prompt = [{"role": "user", "content": "q"}]
        assert self._seed_of(prompt) == request_seed(7, prompt, {})
        assert self._seed_of(prompt) != 7

    def test_seed_is_deterministic_and_shared_by_identical_requests(self):
        prompt = [{"role": "user", "content": "q"}]
        assert self._seed_of(prompt) == self._seed_of([{"role": "user", "content": "q"}])

    def test_distinct_prompts_and_base_seeds_give_distinct_seeds(self):
        first = self._seed_of([{"role": "user", "content": "a"}])
        second = self._seed_of([{"role": "user", "content": "b"}])
        third = self._seed_of([{"role": "user", "content": "a"}], seed=8)
        assert len({first, second, third}) == 3

    def test_per_sample_kwargs_enter_the_seed(self):
        prompt = [{"role": "user", "content": "q"}]
        assert self._seed_of(prompt, per_sample={"spans": ["x"]}) != self._seed_of(prompt, per_sample={"spans": ["y"]})

    def test_unseeded_request_attaches_no_seed(self):
        pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
        dispatcher, _ = _dispatcher(pipeline)
        _serve_all(dispatcher, [_admit(dispatcher, [{"role": "user", "content": "q"}])])
        (call,) = pipeline.calls
        assert "seed" not in call["gen_kwargs"]

    def test_seed_is_independent_of_arrival_order(self):
        prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]

        def seeds_in(order):
            pipeline = StubSteeringPipeline(supports_concurrent_generate=True)
            dispatcher, _ = _dispatcher(pipeline)
            records = [_admit(dispatcher, prompts[i], gen_kwargs={"max_new_tokens": 4, "seed": 3}) for i in order]
            _serve_all(dispatcher, records)
            return {call["messages"][0][0]["content"]: call["gen_kwargs"]["seed"] for call in pipeline.calls}

        assert seeds_in([0, 1, 2, 3]) == seeds_in([3, 1, 0, 2])
