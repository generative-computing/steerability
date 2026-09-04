"""Backend implementations of the execution seam.

Each package implements the `Backend` and `SteeringSession` protocols from
`aisteer360.algorithms.core.execution` for one backend family. Specs resolve to these classes
through `aisteer360.algorithms.core.execution.backend`; nothing in `aisteer360.algorithms`
imports this package at module level.
"""
from aisteer360.backends.huggingface import ExclusiveSession, HFBackend

__all__ = ["ExclusiveSession", "HFBackend"]
