"""Tests for the lock-leader collator: batching under deterministic gating, batch keys, runtime-
kwargs collation, poison isolation, cancellation, strict serialization, and loop reuse."""
import threading

import anyio
import pytest

pytest.importorskip("inspect_ai")

from steerability.evaluation.batching import LockLeaderCollator
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
