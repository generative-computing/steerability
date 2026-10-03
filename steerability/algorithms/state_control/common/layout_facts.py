"""Structural model facts for steer-time preparation.

State controls consume structural facts (layer count, dtype, hidden size) from the steering
session's `ModelFacts` so preparation works the same whether the steering backend holds a live
model or only a layout. With a loaded model and no session, the facts come from the builder that
the Hugging Face session uses (`model_facts`). Module-path resolution stays out of this module;
hook module names are resolved from the module tree at `get_hooks()` time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from steerability.algorithms.core.execution.payloads import ModelFacts
from steerability.algorithms.core.internals.facts import model_facts

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from steerability.algorithms.core.execution.backend import SteeringSession


def resolve_layout(model: PreTrainedModel | None = None, session: SteeringSession | None = None) -> ModelFacts:
    """Structural facts from the session's layout, else derived from the loaded model by
    `model_facts`.

    Args:
        model: A live model, consulted only when `session` is None.
        session: A `SteeringSession` whose `layout` property carries the facts.

    Returns:
        The structural `ModelFacts`.

    Raises:
        ValueError: If neither a session nor a model is available.
    """
    if session is not None:
        return session.layout
    if model is None:
        raise ValueError(
            "Structural facts require a steering session (session.layout) or a live model; "
            "vector-supplied configurations may steer with model=None only when a session is given."
        )
    return model_facts(model)
