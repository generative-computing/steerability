"""Backend implementations of the execution seam.

Each package implements the `Backend` and `SteeringSession` protocols from
`steerability.algorithms.core.execution` for one backend family. Specs resolve to these classes
through `steerability.algorithms.core.execution.backend`; nothing in `steerability.algorithms`
imports this package at module level.
"""
from steerability.backends.huggingface import ExclusiveSession, HFBackend

__all__ = ["ExclusiveSession", "HFBackend"]
