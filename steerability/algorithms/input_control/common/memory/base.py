"""The structural contract for input-control state."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Method-owned, serializable state.

    Methods place their persistent state behind this Protocol so that the framework can save/load it without
    knowing the concrete type. Subclass or define a new class that satisfies this Protocol; see `TextMemory`
    for the primary usage.
    """

    def save(self, path: Path) -> None:
        ...

    @classmethod
    def load(cls, path: Path) -> "Memory":
        ...
