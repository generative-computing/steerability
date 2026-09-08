"""Adapt Inspect scorers into per-row `SampleScorer` rewards for steering controls.

The adapter lets an Inspect scorer (`includes()`, `match()`, `model_graded_fact(...)`) drive
controls that consume a `SampleScorer` (prompt optimizers, sequence rerankers) while `algorithms/`
stays free of Inspect imports. A model-graded scorer used this way runs grader traffic from inside
a control's `steer()` or decode loop; when the grader is a local model it shares the GPU with the
pipeline, so prefer API graders or size headroom accordingly.
"""
from steerability.utils.optional import require

require("inspect_ai")  # anyio, sniffio, and nest_asyncio2 arrive through the inspect extra
import asyncio
from typing import Any, Callable, Mapping

import anyio
import anyio.from_thread
import sniffio
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Scorer, Target, value_to_float
from inspect_ai.solver import TaskState

from steerability.algorithms.core.scoring import SampleScorer


async def _probe() -> None:
    return None


def _run_coroutine_fn(coroutine_fn) -> Any:
    """Run an async zero-argument callable from synchronous code.

    Three contexts are detected, in order. Inside an anyio worker thread (such as the batching
    collator's dispatch thread), the coroutine is scheduled on the running event loop through
    `anyio.from_thread.run`, blocking the worker until it returns. With an asyncio event loop
    running in the current thread (a notebook, or a script that is itself async), `nest_asyncio2`
    re-entry is applied once and the coroutine runs through `asyncio.run`. With no running loop,
    the coroutine runs through `anyio.run`.

    Raises:
        RuntimeError: If a trio task is running in the current thread; re-entry is impossible
            there, so use the Inspect scorer directly.
    """
    try:
        anyio.from_thread.run(_probe)
        in_worker_thread = True
    except RuntimeError:
        in_worker_thread = False
    if in_worker_thread:
        return anyio.from_thread.run(coroutine_fn)
    try:
        library = sniffio.current_async_library()
    except sniffio.AsyncLibraryNotFoundError:
        library = None
    if library == "asyncio":
        import nest_asyncio2 as nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coroutine_fn())
    if library == "trio":
        raise RuntimeError(
            "sample_scorer_from_inspect was called from inside a running trio task; loop re-entry "
            "is impossible there. Use the Inspect scorer directly from async code."
        )
    return anyio.run(coroutine_fn)


def sample_scorer_from_inspect(
    scorer: Scorer,
    *,
    target_key: str = "reference",
    to_float: Callable | None = None,
    model_name: str = "steerability/sample-scorer",
) -> SampleScorer:
    """Adapt an Inspect scorer into a per-row `SampleScorer`.

    Each call builds a standalone `TaskState` from the row (`row["input"]` as the user turn, the
    response as the assistant turn, `row[target_key]` as the target, the row as metadata), runs the
    scorer, and converts `Score.value` to a float.

    Args:
        scorer: The Inspect scorer to adapt.
        target_key: Row key holding the reference target; a missing or None value scores against
            an empty target.
        to_float: Converts `Score.value` to a float; defaults to
            `inspect_ai.scorer.value_to_float()` (maps `"C"`/`"I"`/`"P"`/`"N"` and numerics).
        model_name: Model name recorded on the constructed state and output.

    Returns:
        A `SampleScorer` mapping `(response, row)` to a float.
    """
    convert = to_float if to_float is not None else value_to_float()
    state_model = ModelName(model_name)

    def score(response: str, row: Mapping[str, Any]) -> float:
        query = str(row.get("input", ""))
        target = Target(row.get(target_key) or "")
        state = TaskState(
            model=state_model,
            sample_id=0,
            epoch=0,
            input=query,
            messages=[ChatMessageUser(content=query), ChatMessageAssistant(content=response)],
            target=target,
            output=ModelOutput.from_content(model_name, response),
            metadata=dict(row),
        )

        async def run_scorer():
            return await scorer(state, target)

        result = _run_coroutine_fn(run_scorer)
        if result is None:
            raise ValueError(
                f"The Inspect scorer returned no Score for input {query!r}; a SampleScorer must "
                "produce a float for every row."
            )
        return float(convert(result.value))

    return score
