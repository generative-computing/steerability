"""Marking for forward passes through the pipeline's model that are not generation steps.

Output-control components that forward the pipeline's own model during decoding wrap those calls in
`auxiliary_pass()`. State-control accounting consults `current_auxiliary_pass()` to keep auxiliary
passes out of condition scoring, gate updates, and fallback position counting. `aligned` declares
whether the pass's token positions lie on the current generation's coordinate axis (a prefix of the
trajectory, or a prefix plus a candidate next token) or on a detached sequence such as a
transformed prompt.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class AuxiliaryPassInfo:
    """Describes the auxiliary pass currently in flight.

    Attributes:
        aligned: True when the pass's token positions lie on the current generation's coordinate
            axis. False for a detached sequence.
    """

    aligned: bool


_CURRENT: contextvars.ContextVar[AuxiliaryPassInfo | None] = contextvars.ContextVar(
    "auxiliary_pass", default=None
)


@contextlib.contextmanager
def auxiliary_pass(*, aligned: bool = True):
    """Mark model forwards issued inside the block as auxiliary (not generation steps)."""
    token = _CURRENT.set(AuxiliaryPassInfo(aligned=aligned))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_auxiliary_pass() -> AuxiliaryPassInfo | None:
    """The in-flight auxiliary pass marker, or None during ordinary generation passes."""
    return _CURRENT.get()
