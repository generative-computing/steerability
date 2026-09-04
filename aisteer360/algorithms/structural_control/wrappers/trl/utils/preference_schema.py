from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset

_PROMPT_KEYS = ["prompt", "question", "query", "input"]
_CHOSEN_KEYS = ["chosen", "chosen_response", "preferred", "pos", "accepted", "answer_chosen"]
_REJECTED_KEYS = ["rejected", "rejected_response", "dispreferred", "neg", "answer_rejected"]
_MESSAGES_KEYS = ["messages", "conversations"]


def _first_present(d: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _is_message_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, Mapping) and "role" in item and "content" in item for item in value)
    )


def _last_assistant_content(messages: Sequence[Mapping[str, Any]], column: str) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message["content"]).strip()
    raise ValueError(
        f"Column '{column}' is a conversation with no assistant turn; cannot resolve a completion string."
    )


def _resolve_completion(value: Any, column: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if _is_message_list(value):
        return _last_assistant_content(value, column)
    raise TypeError(
        f"Column '{column}' has unsupported type {type(value).__name__}; expected a string or a list of "
        "role/content messages. Map the dataset to string columns before calling this function."
    )


def _resolve_prompt(value: Any, column: str) -> str:
    if isinstance(value, str):
        return value.strip()
    raise TypeError(
        f"Column '{column}' has unsupported type {type(value).__name__}; expected a string. Map the dataset "
        "to string columns before calling this function."
    )


def standardize_preference_dataset(
    dataset: Dataset,
    drop_unknown_columns: bool = True,
) -> Dataset:
    """Return a dataset with exactly `{'prompt', 'chosen', 'rejected'}` as plain strings.

    Key aliases are resolved (e.g. `question` to `prompt`, `preferred` to `chosen`), extra columns are
    dropped, and any coexisting `messages`/`conversations` column is removed so TRL sees a single schema.

    Column values are resolved by type:

    - A string passes through stripped.
    - For the chosen/rejected columns, a value that is a sequence of role/content mappings resolves to the
        content of the last message whose role is `assistant`, stripped. Earlier turns of a multi-turn
        completion are not folded into the prompt.

    Args:
        dataset: A `datasets.Dataset` carrying prompt/chosen/rejected columns (or their aliases).
        drop_unknown_columns: When True, drop every column other than `prompt`, `chosen`, and `rejected`
            from the result.

    Returns:
        A `datasets.Dataset` with columns `prompt`, `chosen`, and `rejected`, all plain strings.

    Raises:
        TypeError: If `dataset` is not a `datasets.Dataset`; or if the `prompt` value is not a string; or if
            a chosen/rejected value is neither a string nor a list of role/content messages.
        ValueError: If a required prompt/chosen/rejected column (or alias) is absent; or if a chosen/rejected
            conversation has no assistant turn.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("standardize_preference_dataset expects a datasets.Dataset")

    column_names = set(dataset.column_names)

    has_prompt = any(k in column_names for k in _PROMPT_KEYS)
    has_chosen = any(k in column_names for k in _CHOSEN_KEYS)
    has_rejected = any(k in column_names for k in _REJECTED_KEYS)

    if not (has_prompt and has_chosen and has_rejected):
        missing = []
        if not has_prompt: missing.append("prompt")
        if not has_chosen: missing.append("chosen")
        if not has_rejected: missing.append("rejected")
        raise ValueError(
            f"Preference dataset is missing required fields: {missing}. "
            "Provide columns equivalent to prompt/chosen/rejected, or map them before calling this function."
        )

    # build a canonical mapping function
    def to_preference_row(example: dict[str, Any]) -> dict[str, str]:
        prompt_val = _first_present(example, _PROMPT_KEYS)
        chosen_val = _first_present(example, _CHOSEN_KEYS)
        rejected_val = _first_present(example, _REJECTED_KEYS)

        if prompt_val is None or chosen_val is None or rejected_val is None:
            raise ValueError("Example lacks one of prompt/chosen/rejected after key resolution.")

        return {
            "prompt": _resolve_prompt(prompt_val, "prompt"),
            "chosen": _resolve_completion(chosen_val, "chosen"),
            "rejected": _resolve_completion(rejected_val, "rejected"),
        }

    # apply mapping
    standardized = dataset.map(
        to_preference_row,
        remove_columns=[c for c in dataset.column_names if c not in (_PROMPT_KEYS + _CHOSEN_KEYS + _REJECTED_KEYS)],
    )

    # optionally drop stray 'messages' / 'conversations' columns; TRL rejects mixed schemas
    if drop_unknown_columns:
        columns_to_drop = [c for c in standardized.column_names if c not in {"prompt", "chosen", "rejected"}]
        if columns_to_drop:
            standardized = standardized.remove_columns(columns_to_drop)

    return standardized
