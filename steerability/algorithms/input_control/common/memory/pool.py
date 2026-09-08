"""Typed pool with parallel-indexed metadata."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class PoolMemory(Generic[T]):
    """Typed pool of items with optional parallel-indexed metadata.

    `items[i]` has `metadata[k][i]` for each metadata key `k`. All metadata lists must have
    `len() == len(items)`. None is used as a placeholder for missing per-item metadata.

    Stored as pickle by default (T may be arbitrary). For text-only pools, methods can implement their
    own JSON path.
    """
    items: list[T] = field(default_factory=list)
    metadata: dict[str, list[Any]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: T, **per_item_meta: Any) -> None:
        self.items.append(item)
        new_index = len(self.items) - 1
        # extend supplied metadata keys (back-filling earlier indices with None)
        for key, value in per_item_meta.items():
            if key not in self.metadata:
                self.metadata[key] = [None] * new_index
            self.metadata[key].append(value)
        # backfill any pre-existing keys not supplied this call
        for key, lst in self.metadata.items():
            if len(lst) < len(self.items):
                lst.append(None)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"items": self.items, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: Path) -> "PoolMemory":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(items=data["items"], metadata=data["metadata"])
