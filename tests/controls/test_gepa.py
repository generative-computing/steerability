"""Tests for GEPA — reflective prompt optimization (single-system-prompt variant)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.input_control.common.pareto import ParetoFrontier
from aisteer360.algorithms.input_control.gepa import GEPA, GEPAArgs
from aisteer360.algorithms.input_control.gepa.utils import pareto_sampling
from aisteer360.algorithms.input_control.gepa.utils.pool import CandidatePool
from aisteer360.algorithms.input_control.gepa.utils.reflective_dataset import build_records
from aisteer360.algorithms.input_control.gepa.utils.reflective_meta_prompt import render_records

TINY_LM = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_lm():
    model = AutoModelForCausalLM.from_pretrained(TINY_LM, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LM, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestGEPAArgs:
    def test_minimal_valid(self):
        args = GEPAArgs(
            seed_instruction="x",
            train_set=[{"input": "a"}],
            row_scorer=lambda out, row: 0.0,
        )
        assert args.seed_instruction == "x"

    def test_empty_seed_instruction_raises(self):
        with pytest.raises(ValueError, match="seed_instruction"):
            GEPAArgs(
                seed_instruction="",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
            )

    def test_non_string_seed_instruction_raises(self):
        with pytest.raises(ValueError, match="seed_instruction"):
            GEPAArgs(
                seed_instruction=123,  # type: ignore[arg-type]
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
            )

    def test_empty_trainset_raises(self):
        with pytest.raises(ValueError, match="train_set"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[],
                row_scorer=lambda out, row: 0.0,
            )

    def test_no_row_scorer_raises(self):
        with pytest.raises(ValueError, match="row_scorer"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
            )

    def test_non_callable_row_scorer_raises(self):
        with pytest.raises(ValueError, match="row_scorer"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer="not callable",
            )

    def test_reflection_lm_without_tokenizer_raises(self):
        with pytest.raises(ValueError, match="reflection_lm"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
                reflection_lm=object(),
                reflection_tokenizer=None,
            )

    def test_reflection_tokenizer_without_lm_raises(self):
        with pytest.raises(ValueError, match="reflection_lm"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
                reflection_lm=None,
                reflection_tokenizer=object(),
            )

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="budget"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
                budget=0,
            )

    def test_invalid_minibatch_raises(self):
        with pytest.raises(ValueError, match="minibatch_size"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
                minibatch_size=0,
            )

    def test_invalid_pareto_set_size_raises(self):
        with pytest.raises(ValueError, match="pareto_set_size"):
            GEPAArgs(
                seed_instruction="x",
                train_set=[{"input": "a"}],
                row_scorer=lambda out, row: 0.0,
                pareto_set_size=0,
            )


class TestBuildRecords:
    def test_zips_rendered_queries_outputs_feedback(self):
        queries = ["Q1", "Q2"]
        records = build_records(queries, ["o1", "o2"], ["f1", "f2"])
        assert records == [
            {"Inputs": "Q1", "Generated Output": "o1", "Feedback": "f1"},
            {"Inputs": "Q2", "Generated Output": "o2", "Feedback": "f2"},
        ]

    def test_empty(self):
        assert build_records([], [], []) == []

    def test_inputs_is_rendered_query_not_gold_target(self):
        # Inputs must be the rendered query, never the raw row holding the gold target.
        records = build_records(["Q"], ["A"], ["fb"])
        assert records[0]["Inputs"] == "Q"
        rendered = render_records(records)
        assert "ZZZSENTINEL" not in rendered


class TestCandidatePool:
    def test_add_returns_sequential_indices(self):
        pool = CandidatePool()
        i0 = pool.add("0", score_row=[1.0, 0.0])
        i1 = pool.add("1", score_row=[0.0, 1.0])
        assert (i0, i1) == (0, 1)
        assert pool.scores.shape == (2, 2)
        assert pool.candidates == ["0", "1"]

    def test_best_index_matches_row_means(self):
        pool = CandidatePool()
        pool.add("0", score_row=[0.2, 0.2])
        pool.add("1", score_row=[0.9, 0.9])
        pool.add("2", score_row=[0.5, 0.5])
        assert pool.best_index() == 1

    def test_frontier_built_from_scores(self):
        pool = CandidatePool()
        pool.add("0", score_row=[1.0, 0.0])
        pool.add("1", score_row=[0.0, 1.0])
        frontier = pool.frontier()
        assert isinstance(frontier, ParetoFrontier)


class TestParetoSampling:
    def test_returns_index_in_range(self):
        scores = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        frontier = ParetoFrontier(scores)
        rng = random.Random(0)
        idx = pareto_sampling.sample(frontier, 3, rng=rng)
        assert 0 <= idx < 3

    def test_never_returns_compromise_candidate(self):
        # candidate 2 is non-dominated but wins no instance; it must never be sampled.
        scores = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        frontier = ParetoFrontier(scores)
        rng = random.Random(7)
        picks = {pareto_sampling.sample(frontier, 3, rng=rng) for _ in range(200)}
        assert 2 not in picks
        assert picks == {0, 1}

    def test_single_pool_returns_zero(self):
        scores = np.array([[1.0, 1.0]])
        frontier = ParetoFrontier(scores)
        idx = pareto_sampling.sample(frontier, 1)
        assert idx == 0


class TestGEPASteer:
    def test_runs_end_to_end(self, tiny_lm):
        model, tokenizer = tiny_lm
        train = [{"input": f"q{i}", "reference": "r"} for i in range(6)]
        gepa = GEPA(
            seed_instruction="be helpful",
            train_set=train,
            row_scorer=lambda out, row: 0.5,
            budget=15,
            minibatch_size=2,
            pareto_set_size=3,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
            seed=0,
        )
        gepa.steer(model=model, tokenizer=tokenizer)
        assert gepa.memory is not None
        assert set(gepa.memory.slots.keys()) == {"instruction"}
        assert isinstance(gepa.memory["instruction"], str)
        assert len(gepa.memory["instruction"]) > 0

    def test_reflection_records_omit_gold_target(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm

        # capture every rendered records string the reflection proposer receives.
        seen_contexts: list[str] = []

        def capturing_propose(self, seed, n=1, context=None):
            seen_contexts.append((context or {}).get("records", ""))
            return ["be concise"]

        from aisteer360.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
        monkeypatch.setattr(LLMMetaPromptProposer, "propose", capturing_propose)

        # gold target lives in a distinctive sentinel field; format_query returns only the input.
        train = [{"input": f"Q{i}", "gold": "ZZZSENTINEL"} for i in range(6)]
        gepa = GEPA(
            seed_instruction="be helpful",
            train_set=train,
            row_scorer=lambda out, row: 0.5,
            format_query=lambda row: row["input"],
            budget=20,
            minibatch_size=2,
            pareto_set_size=3,
            seed=0,
        )
        gepa.steer(model=model, tokenizer=tokenizer)

        assert seen_contexts, "expected the reflection proposer to be called at least once"
        for rendered in seen_contexts:
            assert "ZZZSENTINEL" not in rendered
            assert "Q" in rendered  # the rendered query did reach the reflector

    def test_tiny_pareto_set_size_falls_back_without_error(self, tiny_lm):
        model, tokenizer = tiny_lm
        train = [{"input": f"q{i}"} for i in range(3)]
        gepa = GEPA(
            seed_instruction="be helpful",
            train_set=train,
            row_scorer=lambda out, row: 0.5,
            budget=12,
            minibatch_size=2,
            pareto_set_size=10,  # larger than the train_set -> overlap fallback
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
            seed=0,
        )
        gepa.steer(model=model, tokenizer=tokenizer)
        assert gepa.memory["instruction"]

    def test_progress_callback_fires_seed_and_iteration_events(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm

        # force the proposer to emit a long instruction and score on instruction length, so the
        # first mutation is strictly accepted -> at least one per-iteration "accept" event fires.
        def fake_propose(self, seed, n=1, context=None):
            return ["x" * 200]

        from aisteer360.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
        monkeypatch.setattr(LLMMetaPromptProposer, "propose", fake_propose)

        def scored_run(self, task_lm, instruction, batch, *, with_feedback):
            outputs = [""] * len(batch)
            scores = [min(1.0, len(instruction) / 200.0)] * len(batch)
            feedback = [f"score={s:.4f}" for s in scores] if with_feedback else None
            return outputs, scores, feedback

        monkeypatch.setattr(GEPA, "_run", scored_run)

        trace: list[dict] = []
        gepa = GEPA(
            seed_instruction="x",
            train_set=[{"input": f"q{i}"} for i in range(6)],
            row_scorer=lambda out, row: 0.0,
            budget=20,
            minibatch_size=2,
            pareto_set_size=3,
            progress_callback=trace.append,
            seed=0,
        )
        gepa.steer(model=model, tokenizer=tokenizer)

        assert trace[0]["event"] == "seed"
        assert trace[0]["step"] == 0
        assert trace[0]["proposed"] == "x"  # full (untruncated) seed text
        iteration_events = [r for r in trace[1:] if r["event"] in {"accept", "reject"}]
        assert iteration_events, "expected at least one per-iteration event"
        assert any(r["event"] == "accept" for r in iteration_events)
        expected_keys = {
            "step", "event", "parent_idx", "parent_score", "candidate_score",
            "accepted", "pool_size", "best_mean", "proposed",
        }
        for record in trace:
            assert expected_keys <= set(record.keys())


class TestGEPAMetaPrompt:
    def test_default_contains_domain_fact_directive_and_fenced_instruction(self):
        from aisteer360.algorithms.input_control.gepa.utils import reflective_meta_prompt
        text = reflective_meta_prompt.GEPA_DEFAULT
        assert "niche and domain specific factual information" in text
        assert "within ``` blocks" in text


class TestGEPAImprovementAcceptance:
    """Drives the genetic loop into the accept path via a length-rewarding scorer and a fixed proposer."""

    def test_strict_improvement_drives_instruction_toward_target(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        target_instruction = (
            "Answer the question precisely.\n\n"
            "Follow these rules:\n- cite the relevant fact\n- never speculate\n\n"
            "Domain note: capital cities are proper nouns."
        )

        # force the proposer to emit the long target instruction, and (below) score on instruction
        # length so the target strictly beats the short seed -> it is accepted and retained as best.
        def fake_propose(self, seed, n=1, context=None):
            return [target_instruction]

        from aisteer360.algorithms.input_control.common.proposers.llm_meta_prompt import LLMMetaPromptProposer
        monkeypatch.setattr(LLMMetaPromptProposer, "propose", fake_propose)

        gepa = GEPA(
            seed_instruction="x",
            train_set=[{"input": f"q{i}"} for i in range(6)],
            row_scorer=lambda out, row: 0.0,  # unused; _run is monkeypatched to score on instruction length
            budget=40,
            minibatch_size=2,
            pareto_set_size=3,
            seed=0,
        )

        # score by instruction length so the long target strictly beats the short seed and is accepted.
        def scored_run(self, task_lm, instruction, batch, *, with_feedback):
            outputs = [""] * len(batch)
            scores = [min(1.0, len(instruction) / 200.0)] * len(batch)
            feedback = [f"score={s:.4f}" for s in scores] if with_feedback else None
            return outputs, scores, feedback

        monkeypatch.setattr(GEPA, "_run", scored_run)

        gepa.steer(model=model, tokenizer=tokenizer)
        assert gepa.memory["instruction"] == target_instruction
        assert "\n\n" in gepa.memory["instruction"]


class TestGEPABudget:
    def test_total_rollouts_within_budget(self, tiny_lm):
        model, tokenizer = tiny_lm
        train = [{"input": f"q{i}"} for i in range(8)]
        calls: list[int] = []
        gepa = GEPA(
            seed_instruction="be helpful",
            train_set=train,
            row_scorer=lambda out, row: 0.5,
            budget=12,
            minibatch_size=2,
            pareto_set_size=2,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
            seed=0,
        )

        original_run = GEPA._run

        def counting_run(self, task_lm, instruction, batch, *, with_feedback):
            calls.append(len(batch))
            return original_run(self, task_lm, instruction, batch, with_feedback=with_feedback)

        gepa._run = counting_run.__get__(gepa, GEPA)
        gepa.steer(model=model, tokenizer=tokenizer)
        assert sum(calls) <= gepa.budget


class TestGEPAAdapt:
    def _make_steered(self, tiny_lm):
        model, tokenizer = tiny_lm
        train = [{"input": f"q{i}"} for i in range(4)]
        gepa = GEPA(
            seed_instruction="be brief",
            train_set=train,
            row_scorer=lambda out, row: 0.5,
            budget=8,
            minibatch_size=2,
            pareto_set_size=2,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
            seed=0,
        )
        gepa.steer(model=model, tokenizer=tokenizer)
        return gepa

    def test_adapt_messages_injects_system_prompt(self, tiny_lm):
        gepa = self._make_steered(tiny_lm)
        adapted = gepa.adapt_messages([[{"role": "user", "content": "?"}]])
        assert adapted is not None
        assert adapted[0][0]["role"] == "system"
        assert adapted[0][0]["content"] == gepa.memory["instruction"]

    def test_adapt_messages_no_runtime_kwargs_needed(self, tiny_lm):
        gepa = self._make_steered(tiny_lm)
        adapted = gepa.adapt_messages([[{"role": "user", "content": "?"}]], runtime_kwargs=None)
        assert adapted[0][0]["content"] == gepa.memory["instruction"]

    def test_before_steer_raises(self):
        gepa = GEPA(
            seed_instruction="x",
            train_set=[{"input": "a"}],
            row_scorer=lambda out, row: 0.0,
        )
        with pytest.raises(RuntimeError, match="before .steer"):
            gepa.adapt_messages([[{"role": "user", "content": "?"}]])

    def test_adapt_before_steer_raises(self):
        gepa = GEPA(
            seed_instruction="x",
            train_set=[{"input": "a"}],
            row_scorer=lambda out, row: 0.0,
        )
        with pytest.raises(RuntimeError, match="before .steer"):
            gepa.adapt(torch.tensor([[1, 2, 3]]))


class TestGEPASessionOnlySteer:
    """The steer phase completes with model=None against a session-only fake (ROLLOUTS)."""

    def test_steer_completes_with_model_none(self):
        from tests.utils.runtime_helpers import ScriptedSession
        from tests.utils.tiny_models import wordlevel_tokenizer

        tokenizer = wordlevel_tokenizer()

        def fake_generate(input_ids=None, attention_mask=None, **gen_kwargs):
            continuation = torch.full((input_ids.size(0), 2), 4, dtype=torch.long)
            return torch.cat([input_ids, continuation], dim=1)

        gepa = GEPA(
            seed_instruction="seed instruction",
            train_set=[{"input": "the cat"}, {"input": "the dog"}],
            row_scorer=lambda out, row: float(len(out)),
            budget=4,
            minibatch_size=1,
            pareto_set_size=1,
            seed=0,
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
            proposer_gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        gepa.steer(model=None, tokenizer=tokenizer, session=ScriptedSession(fake_generate, tokenizer=tokenizer))
        assert gepa.memory is not None
        assert len(gepa.memory["instruction"]) > 0
