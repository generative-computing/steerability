"""Tests for the pure helpers in the instruction-following study's `task.py`.

CPU only, no network: the study's `task.py` is loaded by path and its pure functions
(`select_records`, `clean_kwargs`, `to_sample`) are exercised with synthetic records. No test
loads the Split-IFEval dataset, the IFEval checker, or the reward model.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

TASK_FILE = (
    Path(__file__).resolve().parents[2]
    / "examples" / "notebooks" / "studies" / "instruction_following" / "task.py"
)


@pytest.fixture(scope="module")
def task_module():
    """The study's `task.py`, loaded by path (as the notebook loads it)."""
    spec = importlib.util.spec_from_file_location("instruction_following_task", TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(key: int, instruction_id: str) -> dict:
    """One synthetic single-instruction record."""
    return {
        "key": key,
        "prompt": f"prompt {key}",
        "instruction_id_list": [instruction_id],
        "kwargs": [{}],
        "instructions": [f"- follow {instruction_id}"],
    }


def _pool(counts: dict[str, int]) -> list[dict]:
    """A flat record pool with `counts[instruction_id]` single-instruction records per type."""
    records: list[dict] = []
    key = 0
    for instruction_id, count in counts.items():
        for _ in range(count):
            records.append(_record(key, instruction_id))
            key += 1
    return records


class TestSelectRecords:
    def test_balanced_selection_size(self, task_module):
        records = _pool({"A": 5, "B": 4})
        selected = task_module.select_records(records, ["A", "B"], per_type=3, sample_seed=7)
        assert len(selected) == 6  # 3 per type

    def test_min_per_type_when_group_smaller(self, task_module):
        records = _pool({"A": 5, "B": 2})
        selected = task_module.select_records(records, ["A", "B"], per_type=4, sample_seed=7)
        assert len(selected) == 6  # 4 from A, only 2 available from B

    def test_only_single_instruction_records_of_requested_types(self, task_module):
        records = _pool({"A": 3})
        records.append({
            "key": 90, "prompt": "multi", "instruction_id_list": ["A", "B"],
            "kwargs": [{}, {}], "instructions": ["- x", "- y"],
        })
        records.append(_record(91, "C"))  # single, but off-type
        selected = task_module.select_records(records, ["A"], per_type=10, sample_seed=1)
        assert len(selected) == 3
        assert all(len(r["instruction_id_list"]) == 1 for r in selected)
        assert all(r["instruction_id_list"][0] == "A" for r in selected)
        assert {90, 91}.isdisjoint({r["key"] for r in selected})

    def test_groups_emitted_in_instruction_type_order(self, task_module):
        records = _pool({"A": 2, "B": 2})
        selected = task_module.select_records(records, ["B", "A"], per_type=2, sample_seed=3)
        assert [r["instruction_id_list"][0] for r in selected] == ["B", "B", "A", "A"]

    def test_deterministic_across_calls(self, task_module):
        records = _pool({"A": 8, "B": 8})
        first = task_module.select_records(records, ["A", "B"], per_type=4, sample_seed=11)
        second = task_module.select_records(records, ["A", "B"], per_type=4, sample_seed=11)
        assert [r["key"] for r in first] == [r["key"] for r in second]

    def test_seed_changes_selection(self, task_module):
        records = _pool({"A": 20})
        first = task_module.select_records(records, ["A"], per_type=5, sample_seed=1)
        second = task_module.select_records(records, ["A"], per_type=5, sample_seed=2)
        assert [r["key"] for r in first] != [r["key"] for r in second]

    def test_empty_type_raises(self, task_module):
        records = _pool({"A": 3})
        with pytest.raises(ValueError, match="ZZZ"):
            task_module.select_records(records, ["A", "ZZZ"], per_type=1, sample_seed=1)


class TestCleanKwargs:
    def test_drops_none_and_coerces_integral_float(self, task_module):
        cleaned = task_module.clean_kwargs([
            {"num_highlights": 2.0, "num_bullets": None, "language": "fr"},
        ])
        assert cleaned == [{"num_highlights": 2, "language": "fr"}]

    def test_passes_strings_lists_and_bools_through(self, task_module):
        cleaned = task_module.clean_kwargs([
            {"forbidden_words": ["a", "b"], "keyword": "x", "flag": True, "ratio": 1.5},
        ])
        assert cleaned == [{"forbidden_words": ["a", "b"], "keyword": "x", "flag": True, "ratio": 1.5}]

    def test_one_dict_per_instruction(self, task_module):
        cleaned = task_module.clean_kwargs([{"a": 1.0}, {"b": None, "c": "y"}])
        assert cleaned == [{"a": 1}, {"c": "y"}]


class TestToSample:
    def test_substrings_under_runtime_kwargs(self, task_module):
        sample = task_module.to_sample(_record(42, "startend:end_checker"))
        assert sample.id == 42
        assert sample.input == "prompt 42"
        assert sample.metadata["runtime_kwargs"]["substrings"] == ["- follow startend:end_checker"]

    def test_metadata_carries_instruction_id_and_cleaned_kwargs(self, task_module):
        record = _record(1, "language:response_language")
        record["kwargs"] = [{"language": "en", "num_words": 3.0, "unused": None}]
        sample = task_module.to_sample(record)
        assert sample.metadata["instruction_id"] == "language:response_language"
        assert sample.metadata["instruction_id_list"] == ["language:response_language"]
        assert sample.metadata["kwargs"] == [{"language": "en", "num_words": 3}]
