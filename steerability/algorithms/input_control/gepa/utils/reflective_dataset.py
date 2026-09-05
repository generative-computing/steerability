"""Builder of reflective records from rollout outputs and feedback (single system prompt)."""
from __future__ import annotations


def build_records(queries: list[str], outputs: list, feedback: list) -> list[dict]:
    """Build reflective records from rendered queries and their rollout outputs/feedback.

    Produces one record per item with shape
    `{"Inputs": <rendered query string>, "Generated Output": ..., "Feedback": ...}`.

    `Inputs` is the query the task model actually received (i.e. `format_query(row)`), not
    the raw training row -- the gold target must not appear here; it reaches reflection via
    `Feedback` when the user's `feedback_fn` includes it.

    Args:
        queries: Per-instance rendered query strings shown to the task model.
        outputs: Per-instance task-model outputs.
        feedback: Per-instance textual feedback.

    Returns:
        A list of records, length equal to `len(queries)`.
    """
    return [
        {"Inputs": q, "Generated Output": out, "Feedback": fb}
        for q, out, fb in zip(queries, outputs, feedback)
    ]
