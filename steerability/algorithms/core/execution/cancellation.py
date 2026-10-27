"""Cooperative cancellation for one generation call: `CancelToken` and `GenerationCancelled`.

A caller creates one token per `SteeringPipeline.generate()` call (the reserved `cancel_token`
keyword) and cancels it from any thread. The pipeline forwards the token to the session that runs
the call, and each session honors it at the granularity its docstring states.
"""
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class GenerationCancelled(RuntimeError):
    """A generation stopped because its `CancelToken` was cancelled before it completed."""


class CancelToken:
    """A thread-safe, one-way cancellation signal for one generation call.

    `cancel()` is idempotent and runs every registered callback once, on the cancelling thread. A
    callback registered after cancellation runs at once, on the registering thread. Callback
    exceptions are logged and do not propagate. Sessions honor the token as their docstrings state:
    the offline vLLM session aborts the call's engine requests, and the in-process and serve
    sessions stop between items and let a running decode or request finish.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._callbacks: list[Callable[[], None]] = []

    def cancel(self) -> None:
        """Cancel the token and run the registered callbacks; later calls do nothing."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            self._run(callback)

    @property
    def cancelled(self) -> bool:
        """Whether `cancel()` has been called."""
        return self._cancelled

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a zero-argument callback to run once when the token is cancelled.

        Args:
            callback: The callable to run. It runs at once when the token is already cancelled.
        """
        with self._lock:
            if not self._cancelled:
                self._callbacks.append(callback)
                return
        self._run(callback)

    def raise_if_cancelled(self) -> None:
        """Raise `GenerationCancelled` when the token is cancelled.

        Raises:
            GenerationCancelled: If `cancel()` has been called.
        """
        if self._cancelled:
            raise GenerationCancelled("The generation was cancelled through its CancelToken.")

    @staticmethod
    def _run(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.warning("A CancelToken callback raised; continuing.", exc_info=True)
