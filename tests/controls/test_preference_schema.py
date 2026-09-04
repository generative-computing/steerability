"""Unit tests for `standardize_preference_dataset`.

The function normalizes preference datasets to plain-string `prompt`/`chosen`/`rejected` columns. These
tests pin its handling of both string columns and conversational (`ultrafeedback_binarized`-shaped) columns,
along with the type/value errors it raises for unsupported inputs. No model is loaded.
"""
from __future__ import annotations

import pytest
from datasets import Dataset

from aisteer360.algorithms.structural_control.wrappers.trl.utils.preference_schema import standardize_preference_dataset


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
