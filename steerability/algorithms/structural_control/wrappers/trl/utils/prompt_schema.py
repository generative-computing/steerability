from typing import Any

from datasets import Dataset

_PROMPT_KEYS = ["prompt", "question", "query", "input", "instruction"]
_MESSAGES_KEYS = ["messages", "conversations"]  # prompt-only conversational keys


def _first_present(record: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _to_plain_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def standardize_prompt_dataset(dataset: Dataset, *, drop_unknown_columns: bool = True) -> Dataset:
    if not isinstance(dataset, Dataset):
        raise TypeError("standardize_prompt_dataset expects a datasets.Dataset")

    column_names = set(dataset.column_names)
    has_prompt = any(k in column_names for k in _PROMPT_KEYS)
    if not has_prompt:
        raise ValueError("Prompt dataset is missing a prompt-like column (e.g., 'prompt', 'question').")

    def to_prompt_row(example: dict[str, Any]) -> dict[str, str]:
        prompt_val = _first_present(example, _PROMPT_KEYS)
        if prompt_val is None:
            raise ValueError("Example lacks a prompt-like field after key resolution.")
        return {"prompt": _to_plain_string(prompt_val).strip()}

    standardized = dataset.map(
        to_prompt_row,
        remove_columns=[c for c in dataset.column_names if c not in _PROMPT_KEYS],
    )

    if drop_unknown_columns:
        columns_to_drop = [c for c in standardized.column_names if c != "prompt"]
        if columns_to_drop:
            standardized = standardized.remove_columns(columns_to_drop)

    return standardized
