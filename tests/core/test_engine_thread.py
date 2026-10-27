"""Tests for `EngineThread`, the event-loop thread that runs the offline vLLM engine's coroutines.

The class has no vLLM dependency, so these tests run everywhere: concurrent submission from several
threads onto one loop, result and exception delivery, cancellation reaching the coroutine, and the
close lifecycle.
"""
import asyncio
import threading
import time

import pytest

from steerability.backends.vllm.engine_thread import EngineClosedError, EngineThread


@pytest.fixture
def engine_thread():
    thread = EngineThread(name="test-engine")
    thread.start()
    yield thread
    thread.close()


def test_submissions_from_several_threads_run_concurrently_on_one_loop(engine_thread):
    running = 0
    peak = 0
    loops = set()

    async def work(value):
        nonlocal running, peak
        loops.add(asyncio.get_running_loop())
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.2)
        running -= 1
        return value * 2

    results = {}

    def caller(value):
        results[value] = engine_thread.submit(work(value)).result(5)

    threads = [threading.Thread(target=caller, args=(value,)) for value in range(4)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {0: 0, 1: 2, 2: 4, 3: 6}
    assert len(loops) == 1
    assert peak == 4
    assert time.monotonic() - started < 0.6  # the four sleeps overlapped


def test_run_returns_the_result_or_raises_the_coroutine_error(engine_thread):
    async def ok():
        return "done"

    async def fail():
        raise KeyError("missing")

    assert engine_thread.run(ok()) == "done"
    with pytest.raises(KeyError, match="missing"):
        engine_thread.run(fail())


def test_cancelling_a_future_cancels_the_coroutine(engine_thread):
    started = threading.Event()
    observed = threading.Event()

    async def long_running():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            observed.set()
            raise

    future = engine_thread.submit(long_running())
    assert started.wait(5)
    assert future.cancel()
    assert observed.wait(5)


def test_cancel_pending_cancels_only_unfinished_futures(engine_thread):
    async def quick():
        return 1

    async def slow():
        await asyncio.sleep(30)

    done = engine_thread.submit(quick())
    assert done.result(5) == 1
    pending = engine_thread.submit(slow())
    engine_thread.cancel_pending()
    assert pending.cancelled() or pending.exception(5) is not None
    assert done.result() == 1
    assert not engine_thread.closed


def test_close_cancels_pending_work_and_is_idempotent():
    thread = EngineThread(name="closing-engine")
    thread.start()
    observed = threading.Event()
    started = threading.Event()

    async def long_running():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            observed.set()
            raise

    future = thread.submit(long_running())
    assert started.wait(5)
    thread.close()
    thread.close()
    assert thread.closed
    assert future.cancelled()
    assert observed.is_set()
    assert not any(t.name == "closing-engine" and t.is_alive() for t in threading.enumerate())


def test_submit_and_run_raise_after_close():
    thread = EngineThread(name="closed-engine")
    thread.start()
    thread.close()

    async def never():
        return None

    coroutine = never()
    with pytest.raises(EngineClosedError):
        thread.submit(coroutine)
    assert coroutine.cr_frame is None  # the refused coroutine was closed, not left pending
    with pytest.raises(EngineClosedError):
        thread.run(never())


def test_run_raises_engine_closed_when_closed_under_the_call(engine_thread):
    started = threading.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(30)

    errors = []

    def caller():
        try:
            engine_thread.run(long_running())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=caller)
    thread.start()
    assert started.wait(5)
    engine_thread.close()
    thread.join(5)
    assert len(errors) == 1 and isinstance(errors[0], EngineClosedError)


def test_start_twice_raises(engine_thread):
    with pytest.raises(RuntimeError, match="already started"):
        engine_thread.start()


def test_submit_before_start_raises():
    thread = EngineThread(name="unstarted-engine")

    async def never():
        return None

    with pytest.raises(EngineClosedError):
        thread.submit(never())
    thread.close()
