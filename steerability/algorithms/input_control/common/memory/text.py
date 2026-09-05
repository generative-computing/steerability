"""Named text slots, JSON-serializable."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TextMemory:
    """Named text slots.

    Slot values are str or list[str]; nested structures should use a custom Memory implementation instead.

    Example:
        memory = TextMemory(slots={"instruction": "...", "examples": [...]})
    """
    slots: dict[str, str | list[str]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.slots[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.slots[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.slots

    def get(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.slots, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TextMemory":
        return cls(slots=json.loads(Path(path).read_text()))
