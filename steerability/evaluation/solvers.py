"""Inspect solvers for steering-pipeline evaluation."""
from steerability.utils.optional import require

require("inspect_ai")
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def runtime_kwargs_solver(key: str = "runtime_kwargs") -> Solver:
    """Deliver `sample.metadata[key]` (a dict) to the steering-pipeline provider for this sample.

    The per-sample runtime kwargs travel with the request on `GenerateConfig.extra_body`, so they
    are recorded in the eval log's model events alongside the rest of the config; keep the values
    JSON-plain and modest in size. Each value must be in the consuming control's per-row form (for
    PASTA `substrings`, one `list[str]` per sample). Each key must be declared `"row"`-scoped by the
    controls that consume it (a `"call"`-scoped key is rejected per sample), and a key that no
    enabled control of the arm declares is inert on that arm, so one task can serve every arm of an
    experiment, including the empty baseline.

    This solver performs the sample's generation itself, so it takes the place of a bare
    `generate()` in the task's solver chain (typically as the last solver); do not chain both, or
    the sample generates twice.

    Args:
        key: The `Sample.metadata` key holding the sample's runtime-kwargs dict.

    Returns:
        The solver.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(
            state, extra_body={"runtime_kwargs": dict(state.metadata.get(key) or {})},
        )
    return solve
