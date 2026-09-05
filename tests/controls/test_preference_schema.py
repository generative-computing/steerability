"""Unit tests for `standardize_preference_dataset`.

The function normalizes preference datasets to plain-string `prompt`/`chosen`/`rejected` columns. These
tests pin its handling of both string columns and conversational (`ultrafeedback_binarized`-shaped) columns,
the type/value errors it raises for unsupported inputs, the `prompt_format` chat-template rendering, and the
prompt/completion boundary warning. No model is loaded; the tokenizer-dependent tests use the hub-free
word-level tokenizer.
"""
from __future__ import annotations

import warnings

import pytest
from datasets import Dataset

from steerability.algorithms.structural_control.wrappers.trl.utils.preference_schema import (
    standardize_preference_dataset,
)
from tests.utils.tiny_models import wordlevel_tokenizer

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def test_string_columns_pass_through_and_extras_dropped():
    dataset = Dataset.from_list(
        [
            {"prompt": "  Why is the sky blue?  ", "chosen": " Rayleigh scattering. ", "rejected": " Magic. ", "extra": 1},
        ]
    )

    standardized = standardize_preference_dataset(dataset)

    assert set(standardized.column_names) == {"prompt", "chosen", "rejected"}
    row = standardized[0]
    assert row["prompt"] == "Why is the sky blue?"
    assert row["chosen"] == "Rayleigh scattering."
    assert row["rejected"] == "Magic."


def test_ultrafeedback_shaped_row_resolves_assistant_contents():
    prompt = "how can i develop a habit of drawing daily"
    chosen_answer = "Set a specific time and start small."
    rejected_answer = "Just wait for inspiration."
    dataset = Dataset.from_list(
        [
            {
                "prompt": prompt,
                "prompt_id": "abc123",
                "chosen": [
                    {"content": prompt, "role": "user"},
                    {"content": chosen_answer, "role": "assistant"},
                ],
                "rejected": [
                    {"content": prompt, "role": "user"},
                    {"content": rejected_answer, "role": "assistant"},
                ],
                "messages": [{"content": prompt, "role": "user"}],
                "score_chosen": 9.0,
                "score_rejected": 3.0,
            }
        ]
    )

    standardized = standardize_preference_dataset(dataset)

    assert set(standardized.column_names) == {"prompt", "chosen", "rejected"}
    row = standardized[0]
    assert row["prompt"] == prompt
    assert row["chosen"] == chosen_answer
    assert row["rejected"] == rejected_answer
    assert not row["chosen"].startswith("[{'content'")
    assert not row["rejected"].startswith("[{'content'")


def test_last_assistant_turn_is_selected():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "hi",
                "chosen": [
                    {"content": "hi", "role": "user"},
                    {"content": "first", "role": "assistant"},
                    {"content": "and then?", "role": "user"},
                    {"content": "second", "role": "assistant"},
                ],
                "rejected": [
                    {"content": "hi", "role": "user"},
                    {"content": "bad", "role": "assistant"},
                ],
            }
        ]
    )

    row = standardize_preference_dataset(dataset)[0]
    assert row["chosen"] == "second"
    assert row["rejected"] == "bad"


def test_conversation_without_assistant_turn_raises_value_error():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "hi",
                "chosen": [{"content": "hi", "role": "user"}],
                "rejected": "fine",
            }
        ]
    )

    with pytest.raises(ValueError, match="chosen"):
        standardize_preference_dataset(dataset)


def test_non_string_non_conversational_completion_raises_type_error():
    dataset = Dataset.from_list(
        [
            {"prompt": "hi", "chosen": ["a", "b"], "rejected": "fine"},
        ]
    )

    with pytest.raises(TypeError, match="chosen"):
        standardize_preference_dataset(dataset)


def test_integer_completion_raises_type_error():
    dataset = Dataset.from_list(
        [
            {"prompt": "hi", "chosen": 7, "rejected": "fine"},
        ]
    )

    with pytest.raises(TypeError, match="chosen"):
        standardize_preference_dataset(dataset)


def test_non_string_prompt_raises_type_error():
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"content": "hi", "role": "user"}],
                "chosen": "good",
                "rejected": "bad",
            }
        ]
    )

    with pytest.raises(TypeError, match="prompt"):
        standardize_preference_dataset(dataset)


def test_prompt_format_raw_matches_default():
    rows = [
        {"prompt": "  Why is the sky blue?  ", "chosen": " Rayleigh scattering. ", "rejected": " Magic. "},
    ]
    default = standardize_preference_dataset(Dataset.from_list(rows))
    raw = standardize_preference_dataset(Dataset.from_list(rows), prompt_format="raw")
    assert default.to_dict() == raw.to_dict()


def test_unknown_prompt_format_raises():
    dataset = Dataset.from_list([{"prompt": "hi", "chosen": "a", "rejected": "b"}])

    with pytest.raises(ValueError, match="prompt_format"):
        standardize_preference_dataset(dataset, prompt_format="templated")


def test_chat_prompt_renders_prompt_through_template():
    tokenizer = wordlevel_tokenizer()
    tokenizer.chat_template = CHATML_TEMPLATE
    dataset = Dataset.from_list([{"prompt": "the cat:", "chosen": "sat on", "rejected": "ran fast"}])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        standardized = standardize_preference_dataset(dataset, prompt_format="chat_prompt", tokenizer=tokenizer)

    row = standardized[0]
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert "the cat:" in row["prompt"]
    assert row["chosen"] == "sat on"
    assert row["rejected"] == "ran fast"

    # the rendered prompt is a token prefix of prompt + chosen, so the boundary check passes silently
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    joint_ids = tokenizer(row["prompt"] + row["chosen"], add_special_tokens=False)["input_ids"]
    assert joint_ids[: len(prompt_ids)] == prompt_ids
    assert not [w for w in record if "boundary" in str(w.message)]


def test_chat_prompt_without_chat_template_raises():
    tokenizer = wordlevel_tokenizer()
    assert tokenizer.chat_template is None
    dataset = Dataset.from_list([{"prompt": "hi", "chosen": "a", "rejected": "b"}])

    with pytest.raises(ValueError, match="chat template"):
        standardize_preference_dataset(dataset, prompt_format="chat_prompt", tokenizer=tokenizer)


def test_chat_prompt_without_tokenizer_raises():
    dataset = Dataset.from_list([{"prompt": "hi", "chosen": "a", "rejected": "b"}])

    with pytest.raises(ValueError, match="tokenizer"):
        standardize_preference_dataset(dataset, prompt_format="chat_prompt")


def test_glued_boundary_warns_once_under_raw():
    tokenizer = wordlevel_tokenizer()
    # "the cat" + "sat on" concatenates to "the catsat on", so the tokenized prompt is not a prefix
    dataset = Dataset.from_list(
        [
            {"prompt": "the cat", "chosen": "sat on", "rejected": "ran"},
            {"prompt": "the dog", "chosen": "ran fast", "rejected": "sat"},
        ]
    )

    with pytest.warns(UserWarning, match="prompt_format") as record:
        standardize_preference_dataset(dataset, tokenizer=tokenizer)
    assert len([w for w in record if "boundary" in str(w.message)]) == 1


def test_cleanly_separated_pair_does_not_warn():
    tokenizer = wordlevel_tokenizer()
    # the ":" pre-tokenizes on its own, so "the cat:" stays a token prefix of "the cat:sat on"
    dataset = Dataset.from_list([{"prompt": "the cat:", "chosen": "sat on", "rejected": "ran"}])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        standardize_preference_dataset(dataset, tokenizer=tokenizer)
    assert not [w for w in record if "boundary" in str(w.message)]
