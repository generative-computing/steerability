"""The event-loop thread that runs the offline vLLM engine's coroutines.

`EngineThread` runs one asyncio event loop on a daemon thread. The offline backend constructs its
`AsyncLLM` engine on that loop and submits every engine operation to it as a coroutine, which lets
callers on any thread share one engine. The module has no vLLM dependency; the backend owns what runs on
the loop.
"""
import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EngineClosedError(RuntimeError):
    """The engine thread was closed (the backend released) before a request completed."""


class EngineThread:
    """An asyncio event loop on a daemon thread, the single place engine coroutines run.

    `submit` and `run` accept coroutines from any thread. Every future `submit` returns is tracked
    until it completes, which lets `cancel_pending` and `close` cancel the work still in flight. Once
    closed, the thread accepts no further submissions.

    Args:
        name: Thread name, for logs.
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pending: set[concurrent.futures.Future] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether `close()` has been called."""
        return self._closed

    def start(self) -> None:
        """Create the event loop and run it on a new daemon thread.

        Raises:
            EngineClosedError: If the thread was closed.
            RuntimeError: If the thread was already started.
        """
        with self._lock:
            if self._closed:
                raise EngineClosedError(f"Engine thread {self._name} is closed.")
            if self._thread is not None:
                raise RuntimeError(f"Engine thread {self._name} is already started.")
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(target=run_loop, name=self._name, daemon=True)
            thread.start()
            ready.wait()
            self._loop = loop
            self._thread = thread

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future:
        """Schedule `coro` on the loop and return a future for its result.

        Cancelling the returned future cancels the coroutine on the loop, which then sees
        `asyncio.CancelledError` at its current await.

        Args:
            coro: The coroutine to run.

        Returns:
            A `concurrent.futures.Future` that resolves with the coroutine's result or exception.

        Raises:
            EngineClosedError: If the thread is closed or was never started; `coro` is closed
                without running.
        """
        with self._lock:
            if self._closed or self._loop is None:
                coro.close()
                raise EngineClosedError(f"Engine thread {self._name} is closed.")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._pending.add(future)
        future.add_done_callback(self._forget)
        return future

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run `coro` on the loop and block the calling thread until it completes.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine's result.

        Raises:
            EngineClosedError: If the thread is closed, or is closed while `coro` runs.
            Exception: The coroutine's own exception, re-raised on the calling thread.
        """
        future = self.submit(coro)
        try:
            return future.result()
        except concurrent.futures.CancelledError as error:
            raise EngineClosedError(f"Engine thread {self._name} was closed while the call ran.") from error

    def cancel_pending(self) -> None:
        """Cancel every submitted future that is not yet done."""
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.cancel()

    def close(self) -> None:
        """Cancel pending work, stop the loop, join the thread, and close the loop.

        Close is idempotent and thread-safe. Afterwards `submit` and `run` raise
        `EngineClosedError`. Must not be called from a coroutine running on the loop.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        if loop is None or thread is None:
            return
        if thread is threading.current_thread():
            raise RuntimeError(f"Engine thread {self._name} cannot close itself from its own loop.")
        loop.call_soon_threadsafe(self._drain_and_stop, loop)
        thread.join()
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            logger.debug("Async generator shutdown on %s failed.", self._name, exc_info=True)
        loop.close()

    @staticmethod
    def _drain_and_stop(loop: asyncio.AbstractEventLoop) -> None:
        """Cancel the tasks still on the loop and stop it once their cancellation has run."""
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
        if not tasks:
            loop.stop()
            return

        async def wait_then_stop() -> None:
            await asyncio.wait(tasks, timeout=5.0)
            loop.stop()

        loop.create_task(wait_then_stop())

    def _forget(self, future: concurrent.futures.Future) -> None:
        with self._lock:
            self._pending.discard(future)
