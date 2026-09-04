"""Per-item seed derivation, bounded fan-out, transport retries, and the partial-batch error.

Shared machinery for request-building sessions. Everything here is backend-agnostic and runs
without any optional dependency.
"""
import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aisteer360.algorithms.core.execution.payloads import ItemResult

logger = logging.getLogger(__name__)

_SEED_MASK = (1 << 63) - 1


def derive_item_seed(base_seed: int, operation_id: str, item_index: int) -> int:
    """Derive one item's sampling seed from a base seed, an operation id, and the item index.

    The derivation is `SHA-256("{base_seed}:{operation_id}:{item_index}")` truncated to the
    first eight bytes (big-endian) and masked to 63 bits, so the result is a stable non-negative
    integer accepted by every backend sampler. Distinct item indices under one operation yield
    distinct streams while the whole fan-out stays reproducible from `base_seed`.

    Args:
        base_seed: The caller-supplied seed.
        operation_id: Identifier of the logical operation (stable across reruns).
        item_index: Position of the item in the submitted sequence.

    Returns:
        The derived seed in `[0, 2**63)`.
    """
    digest = hashlib.sha256(f"{base_seed}:{operation_id}:{item_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


class TransportError(RuntimeError):
    """A transport-level failure (connection, timeout, or server-side 5xx) that is safe to retry.

    Application-level rejections (validation errors, unknown parameters) are not transport
    errors and must not be wrapped in this type.
    """


class PartialBatchError(RuntimeError):
    """Raised when some items of a fan-out failed after retries while others succeeded.

    Attributes:
        results: The successful per-item results, in item order (`ItemResult`s for generation,
            per-item score rows for scoring).
        failures: `(item_index, exception)` pairs for the failed items, in item order.
    """

    def __init__(
        self,
        results: Sequence[ItemResult] | Sequence[Any],
        failures: Sequence[tuple[int, Exception]],
    ) -> None:
        self.results = tuple(results)
        self.failures = tuple(failures)
        summary = "; ".join(
            f"item {index}: {type(error).__name__}: {error}" for index, error in self.failures
        )
        super().__init__(
            f"{len(self.failures)} of {len(self.results) + len(self.failures)} items failed "
            f"after retries ({summary}). Successful results are on `results`; re-issue the "
            "indices on `failed_indices`."
        )

    @property
    def failed_indices(self) -> tuple[int, ...]:
        """Indices of the failed items, re-issuable as a remainder batch."""
        return tuple(index for index, _ in self.failures)


def with_transport_retries(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `fn`, retrying `TransportError`s with exponential backoff.

    Only `TransportError` triggers a retry; every other exception propagates immediately, since
    an application-level rejection will not change on resubmission.

    Args:
        fn: Zero-argument callable issuing one request.
        max_attempts: Total attempts including the first.
        backoff_base: Sleep before attempt `k` (1-based retries) is `backoff_base * 2**(k-1)`.
        sleep: Sleep function (injectable for tests).

    Returns:
        `fn()`'s return value.

    Raises:
        TransportError: The last attempt's error when every attempt failed.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except TransportError as error:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            logger.debug("Transport error (%s); retrying in %.2fs.", error, delay)
            sleep(delay)


def run_bounded(
    tasks: Sequence[Callable[[], Any]],
    max_concurrency: int,
) -> list[Any | Exception]:
    """Run `tasks` concurrently with at most `max_concurrency` in flight.

    Args:
        tasks: Zero-argument callables, one per item.
        max_concurrency: Maximum number of concurrently running tasks (at least 1).

    Returns:
        One entry per task in task order: the task's return value, or the exception it raised.
    """
    if not tasks:
        return []
    max_workers = max(1, min(int(max_concurrency), len(tasks)))
    if max_workers == 1:
        results: list[Any | Exception] = []
        for task in tasks:
            try:
                results.append(task())
            except Exception as error:
                results.append(error)
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        gathered: list[Any | Exception] = []
        for future in futures:
            error = future.exception()
            gathered.append(future.result() if error is None else error)
        return gathered
