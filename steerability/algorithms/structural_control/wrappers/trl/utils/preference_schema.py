import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from steerability.utils.rendering import PromptFormat, has_chat_template, render_for_model

_PROMPT_KEYS = ["prompt", "question", "query", "input"]
_CHOSEN_KEYS = ["chosen", "chosen_response", "preferred", "pos", "accepted", "answer_chosen"]
_REJECTED_KEYS = ["rejected", "rejected_response", "dispreferred", "neg", "answer_rejected"]
_MESSAGES_KEYS = ["messages", "conversations"]

_BOUNDARY_CHECK_ROWS = 8


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


def _warn_on_broken_prompt_boundary(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> None:
    """Warn once when a sampled row's tokenized `prompt` does not prefix its tokenized `prompt + chosen`.

    TRL concatenates `prompt` and each completion verbatim, so a boundary that re-tokenizes across the
    seam trains on token sequences whose prompt region differs from the standalone prompt. Only the
    first `_BOUNDARY_CHECK_ROWS` rows are checked, and at most one `UserWarning` is emitted.
    """
    for row in dataset.select(range(min(len(dataset), _BOUNDARY_CHECK_ROWS))):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        joint_ids = tokenizer(row["prompt"] + row["chosen"], add_special_tokens=False)["input_ids"]
        if joint_ids[: len(prompt_ids)] != prompt_ids:
            warnings.warn(
                "Preference rows tokenize with a broken prompt/completion boundary: the tokenized "
                "prompt is not a prefix of the tokenized prompt + chosen (TRL concatenates the two "
                "verbatim). Likely cause: a missing separator between prompt and completion, or a "
                "prompt that should be rendered through the model's chat template. Add a separator "
                "to the data, or pass prompt_format='chat_prompt'.",
                UserWarning,
            )
            return


def standardize_preference_dataset(
    dataset: Dataset,
    drop_unknown_columns: bool = True,
    *,
    prompt_format: PromptFormat = "raw",
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    """Return a dataset with exactly `{'prompt', 'chosen', 'rejected'}` as plain strings.

    Key aliases are resolved (e.g. `question` to `prompt`, `preferred` to `chosen`), extra columns are
    dropped, and any coexisting `messages`/`conversations` column is removed so TRL sees a single schema.

    Column values are resolved by type:

    - A string passes through stripped.
    - For the chosen/rejected columns, a value that is a sequence of role/content mappings resolves to the
        content of the last message whose role is `assistant`, stripped. Earlier turns of a multi-turn
        completion are not folded into the prompt.

    With a non-`"raw"` `prompt_format`, each resolved prompt is additionally rendered through the
    tokenizer's chat template as a user turn with the assistant generation prompt appended, producing the
    prompt exactly as inference renders it; the chosen/rejected completions stay bare strings that TRL
    concatenates after the assistant turn.

    When `tokenizer` is provided, a sample of the standardized rows is checked for a broken
    prompt/completion boundary (the tokenized `prompt` not being a prefix of the tokenized
    `prompt + chosen`), which typically indicates a missing separator or a prompt that needs the chat
    template.

    Args:
        dataset: A `datasets.Dataset` carrying prompt/chosen/rejected columns (or their aliases).
        drop_unknown_columns: When True, drop every column other than `prompt`, `chosen`, and `rejected`
            from the result.
        prompt_format: `"raw"` keeps prompts verbatim; `"chat_prompt"` or `"chat_completion"` renders each
            prompt through the tokenizer's chat template with the generation prompt appended.
        tokenizer: Tokenizer used for chat-template rendering and the boundary check. Required for a
            non-`"raw"` `prompt_format`.

    Returns:
        A `datasets.Dataset` with columns `prompt`, `chosen`, and `rejected`, all plain strings.

    Raises:
        TypeError: If `dataset` is not a `datasets.Dataset`; or if the `prompt` value is not a string; or if
            a chosen/rejected value is neither a string nor a list of role/content messages.
        ValueError: If a required prompt/chosen/rejected column (or alias) is absent; if a chosen/rejected
            conversation has no assistant turn; if `prompt_format` is not one of `"raw"`,
            `"chat_completion"`, `"chat_prompt"`; or if a non-`"raw"` `prompt_format` is requested without
            a tokenizer or with a tokenizer that has no chat template.

    Warns:
        UserWarning: Once, when `tokenizer` is provided and a sampled row's tokenized `prompt` is not a
            prefix of its tokenized `prompt + chosen`.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("standardize_preference_dataset expects a datasets.Dataset")

    if prompt_format not in ("raw", "chat_completion", "chat_prompt"):
        raise ValueError(
            f"prompt_format must be 'raw', 'chat_completion', or 'chat_prompt'; got {prompt_format!r}."
        )
    if prompt_format != "raw":
        if tokenizer is None:
            raise ValueError(
                f"prompt_format={prompt_format!r} requires a tokenizer to render the chat template."
            )
        if not has_chat_template(tokenizer):
            raise ValueError(
                f"prompt_format={prompt_format!r} was requested but the tokenizer has no chat template; "
                "pass a chat-templated tokenizer or use prompt_format='raw'."
            )

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

        prompt = _resolve_prompt(prompt_val, "prompt")
        if prompt_format != "raw":
            prompt = render_for_model(tokenizer, prompt=prompt, mode="chat_prompt")

        return {
            "prompt": prompt,
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

    if tokenizer is not None:
        _warn_on_broken_prompt_boundary(standardized, tokenizer)

    return standardized
