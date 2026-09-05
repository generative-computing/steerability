"""Tests for the Inspect-scorer-to-SampleScorer adapter and its sync bridge."""
import asyncio

import anyio
import anyio.to_thread
import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Score, accuracy, includes, match, scorer

from steerability.evaluation.scorers import sample_scorer_from_inspect


class TestSampleScorerFromInspect:
    def test_includes_from_sync_context(self):
        score = sample_scorer_from_inspect(includes())
        assert score("the answer is Paris", {"input": "capital?", "reference": "Paris"}) == 1.0
        assert score("the answer is London", {"input": "capital?", "reference": "Paris"}) == 0.0

    def test_match_from_sync_context(self):
        score = sample_scorer_from_inspect(match())
        assert score("Paris", {"input": "capital?", "reference": "Paris"}) == 1.0

    def test_custom_target_key_and_missing_target(self):
        score = sample_scorer_from_inspect(includes(), target_key="gold")
        assert score("Paris is nice", {"input": "q", "gold": "Paris"}) == 1.0
        # a missing target scores against ""; includes() treats the empty target as contained
        assert score("anything", {"input": "q"}) == 1.0

    def test_custom_to_float(self):
        score = sample_scorer_from_inspect(includes(), to_float=lambda value: 42.0)
        assert score("Paris", {"input": "q", "reference": "Paris"}) == 42.0

    def test_row_travels_as_metadata(self):
        seen = {}

        @scorer(metrics=[accuracy()])
        def metadata_scorer():
            async def run(state, target):
                seen.update(state.metadata)
                return Score(value=1.0)
            return run

        score = sample_scorer_from_inspect(metadata_scorer())
        score("response", {"input": "q", "reference": "r", "extra_column": 7})
        assert seen["extra_column"] == 7

    def test_none_score_raises(self):
        @scorer(metrics=[accuracy()])
        def silent_scorer():
            async def run(state, target):
                return None
            return run

        score = sample_scorer_from_inspect(silent_scorer())
        with pytest.raises(ValueError, match="no Score"):
            score("response", {"input": "q", "reference": "r"})

    def test_scorer_errors_propagate(self):
        @scorer(metrics=[accuracy()])
        def failing_scorer():
            async def run(state, target):
                raise RuntimeError("grader down")
            return run

        score = sample_scorer_from_inspect(failing_scorer())
        with pytest.raises(RuntimeError, match="grader down"):
            score("response", {"input": "q", "reference": "r"})

    def test_from_inside_running_asyncio_loop(self):
        score = sample_scorer_from_inspect(includes())

        async def in_loop():
            return score("Paris", {"input": "q", "reference": "Paris"})

        assert asyncio.run(in_loop()) == 1.0

    def test_from_inside_anyio_worker_thread(self):
        score = sample_scorer_from_inspect(includes())

        def in_thread():
            return score("Paris", {"input": "q", "reference": "Paris"})

        async def main():
            return await anyio.to_thread.run_sync(in_thread)

        assert anyio.run(main) == 1.0
