"""Unit tests for steerability/algorithms/input_control/common/."""
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.input_control.common import ParetoFrontier, RolloutBudget
from steerability.algorithms.input_control.common.formatters import (
    ChatTemplateSlotFormatter,
    FewShotBlockFormatter,
    PrependTextFormatter,
    SystemPromptFormatter,
)
from steerability.algorithms.input_control.common.memory import Memory, PoolMemory, TextMemory
from steerability.algorithms.input_control.common.proposers import (
    BaseProposer,
    LLMMetaPromptProposer,
    RetrievalProposer,
    parse_concise_instruction,
    parse_fenced_or_whole,
    parse_whole,
)
from steerability.algorithms.input_control.common.scorers import BaseScorer, TaskEvaluationScorer
from steerability.algorithms.input_control.common.selectors import (
    BaseSelector,
    DenseRetrievalSelector,
    MMRSelector,
    RandomSelector,
    TopKSelector,
)


class _CallableScorer(BaseScorer):
    """Wraps a (prompt, query) -> float callable. Test-only replacement for RewardModelScorer."""

    def __init__(self, fn):
        self.fn = fn

    def score(self, prompts, queries=None):
        if queries is not None and len(queries) != len(prompts):
            raise ValueError(
                f"queries (len={len(queries)}) must align with prompts (len={len(prompts)})."
            )
        return [
            float(self.fn(p, queries[i] if queries else None))
            for i, p in enumerate(prompts)
        ]


class TestTextMemory:
    def test_basic_dict_access(self):
        memory = TextMemory(slots={"instruction": "be concise"})
        assert memory["instruction"] == "be concise"
        assert "instruction" in memory
        assert memory.get("missing", "default") == "default"

    def test_setitem(self):
        memory = TextMemory()
        memory["instruction"] = "act helpful"
        assert memory["instruction"] == "act helpful"

    def test_save_load_roundtrip(self, tmp_path):
        memory = TextMemory(slots={
            "instruction": "abc",
            "examples": ["one", "two"],
        })
        path = tmp_path / "mem.json"
        memory.save(path)
        loaded = TextMemory.load(path)
        assert loaded.slots == memory.slots

    def test_satisfies_memory_protocol(self):
        assert isinstance(TextMemory(), Memory)


class TestPoolMemory:
    def test_empty_pool(self):
        pool = PoolMemory()
        assert len(pool) == 0
        assert pool.items == []

    def test_add_with_metadata(self):
        pool = PoolMemory[dict]()
        pool.add({"input": "a", "output": "b"}, polarity="pos")
        pool.add({"input": "c", "output": "d"}, polarity="neg")
        assert len(pool) == 2
        assert pool.metadata["polarity"] == ["pos", "neg"]

    def test_add_backfills_missing_metadata(self):
        pool = PoolMemory[dict]()
        pool.add({"x": 1})  # no metadata
        pool.add({"x": 2}, polarity="pos")  # introduces a new key
        # earlier index gets back-filled with None
        assert pool.metadata["polarity"] == [None, "pos"]

    def test_add_with_unset_metadata_padded_to_length(self):
        pool = PoolMemory[dict]()
        pool.add({"x": 1}, polarity="pos", score=0.5)
        pool.add({"x": 2}, polarity="neg")  # omit score
        assert pool.metadata["score"] == [0.5, None]

    def test_save_load_roundtrip(self, tmp_path):
        pool = PoolMemory[dict]()
        pool.add({"input": "q", "output": "a"}, polarity="pos")
        pool.add({"input": "r", "output": "b"}, polarity="neg")
        path = tmp_path / "pool.pkl"
        pool.save(path)
        loaded = PoolMemory.load(path)
        assert loaded.items == pool.items
        assert loaded.metadata == pool.metadata

    def test_satisfies_memory_protocol(self):
        assert isinstance(PoolMemory(), Memory)


class TestRolloutBudget:
    def test_initial_state(self):
        b = RolloutBudget(10)
        assert bool(b) is True
        assert b.remaining == 10

    def test_charge_decrements_remaining(self):
        b = RolloutBudget(10)
        b.charge(3)
        assert b.remaining == 7
        assert bool(b) is True

    def test_exhausted_is_falsy(self):
        b = RolloutBudget(5)
        b.charge(5)
        assert bool(b) is False
        assert b.remaining == 0

    def test_overcharge_is_clamped_to_zero(self):
        b = RolloutBudget(5)
        b.charge(10)
        assert b.remaining == 0
        assert bool(b) is False

    def test_negative_max_raises(self):
        with pytest.raises(ValueError):
            RolloutBudget(-1)

    def test_negative_charge_raises(self):
        b = RolloutBudget(10)
        with pytest.raises(ValueError):
            b.charge(-1)


class TestParetoFrontier:
    def test_per_instance_best_max(self):
        # 3 candidates, 4 instances; each row is a candidate's score per instance
        scores = np.array([
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        frontier = ParetoFrontier(scores)
        np.testing.assert_array_equal(frontier.per_instance_best, [1.0, 1.0, 0.5, 1.0])

    def test_per_instance_winners(self):
        scores = np.array([
            [1.0, 0.5],
            [1.0, 0.0],  # ties with candidate 0 on instance 0
            [0.0, 1.0],
        ])
        frontier = ParetoFrontier(scores)
        winners = frontier.per_instance_winners
        assert winners[0] == {0, 1}
        assert winners[1] == {2}

    def test_non_dominated_includes_unique_winners(self):
        # candidate 0 wins instance 0, candidate 1 wins instance 1, candidate 2 strictly dominated
        scores = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        frontier = ParetoFrontier(scores)
        nd = frontier.non_dominated
        assert 0 in nd
        assert 1 in nd
        assert 2 not in nd

    def test_coverage_counts_per_instance_wins(self):
        scores = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        frontier = ParetoFrontier(scores)
        assert frontier.coverage(0) == 2
        assert frontier.coverage(1) == 1

    def test_minimize_inverts_winners(self):
        scores = np.array([
            [1.0, 5.0],
            [2.0, 0.0],
        ])
        frontier_max = ParetoFrontier(scores)
        frontier_min = ParetoFrontier(scores, minimize=True)
        assert frontier_max.per_instance_winners[0] == {1}
        assert frontier_min.per_instance_winners[0] == {0}
        np.testing.assert_array_equal(frontier_max.per_instance_best, [2.0, 5.0])
        np.testing.assert_array_equal(frontier_min.per_instance_best, [1.0, 0.0])

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            ParetoFrontier(np.array([1.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ParetoFrontier(np.empty((0, 3)))

    def test_illumination_set_excludes_compromise_candidate(self):
        # candidate 0 wins instance 0, candidate 1 wins instance 1; candidate 2 is non-dominated
        # but wins no instance (a compromise) and must be excluded.
        scores = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        frontier = ParetoFrontier(scores)
        assert frontier.illumination_set == [(0, 1), (1, 1)]

    def test_illumination_set_prunes_dominated_tie(self):
        # candidate 1 ties on instance 0 but is dominated by candidate 0 overall.
        scores = np.array([
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        frontier = ParetoFrontier(scores)
        assert frontier.illumination_set == [(0, 2)]


class TestSystemPromptFormatter:
    def test_inserts_when_no_system(self):
        f = SystemPromptFormatter()
        memory = TextMemory(slots={"instruction": "be helpful"})
        out = f.apply_to_messages([[{"role": "user", "content": "hi"}]], memory)
        assert out[0][0] == {"role": "system", "content": "be helpful"}
        assert out[0][1]["role"] == "user"

    def test_replaces_existing_system(self):
        f = SystemPromptFormatter()
        memory = TextMemory(slots={"instruction": "new"})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]],
            memory,
        )
        assert out[0][0]["content"] == "new"
        assert len(out[0]) == 2

    def test_handles_batch(self):
        f = SystemPromptFormatter()
        memory = TextMemory(slots={"instruction": "x"})
        out = f.apply_to_messages(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            memory,
        )
        assert len(out) == 2
        assert all(chat[0]["content"] == "x" for chat in out)

    def test_missing_instruction_raises(self):
        f = SystemPromptFormatter()
        with pytest.raises(TypeError):
            f.apply_to_messages([[{"role": "user", "content": "hi"}]], TextMemory())

    def test_default_mode_is_replace(self):
        # backward compatibility: the no-arg formatter replaces the leading system message
        f = SystemPromptFormatter()
        assert f.mode == "replace"
        memory = TextMemory(slots={"instruction": "new"})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]],
            memory,
        )
        assert out[0][0]["content"] == "new"

    def test_prepend_merges_ahead_of_existing(self):
        f = SystemPromptFormatter(mode="prepend", separator=" | ")
        memory = TextMemory(slots={"instruction": "ctx"})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]],
            memory,
        )
        assert out[0][0]["content"] == "ctx | old"
        assert len(out[0]) == 2

    def test_append_merges_after_existing(self):
        f = SystemPromptFormatter(mode="append", separator=" | ")
        memory = TextMemory(slots={"instruction": "ctx"})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]],
            memory,
        )
        assert out[0][0]["content"] == "old | ctx"
        assert len(out[0]) == 2

    @pytest.mark.parametrize("mode", ["replace", "prepend", "append"])
    def test_no_system_message_identical_across_modes(self, mode):
        # with no leading system message, a merge with nothing is a set: one new system message at position 0
        f = SystemPromptFormatter(mode=mode, separator=" | ")
        memory = TextMemory(slots={"instruction": "ctx"})
        out = f.apply_to_messages([[{"role": "user", "content": "hi"}]], memory)[0]
        assert out[0] == {"role": "system", "content": "ctx"}
        assert out[1]["role"] == "user"

    @pytest.mark.parametrize("mode", ["replace", "prepend", "append"])
    def test_input_messages_not_mutated(self, mode):
        f = SystemPromptFormatter(mode=mode, separator=" | ")
        memory = TextMemory(slots={"instruction": "ctx"})
        original = [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]]
        _ = f.apply_to_messages(original, memory)
        assert original == [[{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            SystemPromptFormatter(mode="merge")

    def test_apply_to_ids_batch_roundtrips(self, tiny_lm):
        # apply_to_ids must handle a 2-row batch (batch_decode), returning two rows without raising
        model, tokenizer = tiny_lm
        f = SystemPromptFormatter()
        memory = TextMemory(slots={"instruction": "be brief"})
        enc = tokenizer(["first question", "second question here"], return_tensors="pt", padding=True)
        with pytest.warns(UserWarning):
            out = f.apply_to_ids(enc["input_ids"], memory, tokenizer)
        assert out.shape[0] == 2
        for row in out:
            tokenizer.decode(row.tolist(), skip_special_tokens=True)  # round-trips without raising


class TestPrependTextFormatter:
    def test_prepends_to_first_user_message(self):
        f = PrependTextFormatter(separator=" | ")
        memory = TextMemory(slots={"text": "ctx"})
        out = f.apply_to_messages([[{"role": "user", "content": "hi"}]], memory)
        assert out[0][0]["content"] == "ctx | hi"

    def test_creates_user_message_if_absent(self):
        f = PrependTextFormatter()
        memory = TextMemory(slots={"text": "ctx"})
        out = f.apply_to_messages([[{"role": "system", "content": "s"}]], memory)
        assert any(m.get("role") == "user" and "ctx" in m.get("content", "") for m in out[0])

    def test_resolve_text_present(self):
        f = PrependTextFormatter()
        assert f._resolve_text(TextMemory(slots={"text": "ctx"})) == "ctx"

    def test_resolve_text_missing_raises(self):
        f = PrependTextFormatter()
        with pytest.raises(TypeError):
            f._resolve_text(TextMemory())

    def test_resolve_text_non_str_raises(self):
        f = PrependTextFormatter()
        with pytest.raises(TypeError):
            f._resolve_text(TextMemory(slots={"text": 123}))

    def test_default_target_is_first_user(self):
        # backward compatibility: the no-arg formatter targets the first user turn
        f = PrependTextFormatter(separator=" | ")
        assert f.target == "first_user"
        memory = TextMemory(slots={"text": "ctx"})
        chat = [[{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"},
                 {"role": "user", "content": "b"}]]
        out = f.apply_to_messages(chat, memory)[0]
        assert out[0]["content"] == "ctx | a"
        assert out[2]["content"] == "b"

    def test_target_last_user(self):
        f = PrependTextFormatter(separator=" | ", target="last_user")
        memory = TextMemory(slots={"text": "ctx"})
        chat = [[{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"},
                 {"role": "user", "content": "b"}]]
        out = f.apply_to_messages(chat, memory)[0]
        assert out[0]["content"] == "a"
        assert out[2]["content"] == "ctx | b"

    def test_target_all_user(self):
        f = PrependTextFormatter(separator=" | ", target="all_user")
        memory = TextMemory(slots={"text": "ctx"})
        chat = [[{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"},
                 {"role": "user", "content": "b"}]]
        out = f.apply_to_messages(chat, memory)[0]
        assert out[0]["content"] == "ctx | a"
        assert out[1]["content"] == "x"  # assistant turn untouched
        assert out[2]["content"] == "ctx | b"

    @pytest.mark.parametrize("target", ["first_user", "last_user", "all_user"])
    def test_no_user_message_appends_user_turn(self, target):
        f = PrependTextFormatter(target=target)
        memory = TextMemory(slots={"text": "ctx"})
        out = f.apply_to_messages([[{"role": "system", "content": "s"}]], memory)[0]
        assert any(m.get("role") == "user" and "ctx" in m.get("content", "") for m in out)

    @pytest.mark.parametrize("target", ["first_user", "last_user", "all_user"])
    def test_input_messages_not_mutated(self, target):
        f = PrependTextFormatter(separator=" | ", target=target)
        memory = TextMemory(slots={"text": "ctx"})
        original = [[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]]
        _ = f.apply_to_messages(original, memory)
        assert original == [[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]]

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target"):
            PrependTextFormatter(target="middle_user")


class TestChatTemplateSlotFormatter:
    def test_substitutes_named_slot(self):
        f = ChatTemplateSlotFormatter()
        memory = TextMemory(slots={"guidelines": "be polite"})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "follow: {{guidelines}}"}]],
            memory,
        )
        assert out[0][0]["content"] == "follow: be polite"

    def test_renders_list_as_bullets(self):
        f = ChatTemplateSlotFormatter()
        memory = TextMemory(slots={"items": ["a", "b", "c"]})
        out = f.apply_to_messages(
            [[{"role": "user", "content": "{{items}}"}]],
            memory,
        )
        rendered = out[0][0]["content"]
        assert "a" in rendered and "b" in rendered and "c" in rendered

    def test_unknown_placeholder_left_unchanged(self):
        f = ChatTemplateSlotFormatter()
        memory = TextMemory(slots={"x": "y"})
        out = f.apply_to_messages([[{"role": "user", "content": "{{missing}}"}]], memory)
        assert out[0][0]["content"] == "{{missing}}"


class TestFewShotBlockFormatter:
    def test_inserts_single_system_block(self):
        f = FewShotBlockFormatter()
        memory = TextMemory(slots={
            "examples": [
                {"input": "1+1", "output": "2"},
                {"input": "2+2", "output": "4"},
            ],
        })
        out = f.apply_to_messages([[{"role": "user", "content": "3+3?"}]], memory)
        # one system block + the original user turn
        assert len(out[0]) == 2
        assert out[0][0]["role"] == "system"
        block = out[0][0]["content"]
        for fragment in ("Input: 1+1", "Output: 2", "Input: 2+2", "Output: 4"):
            assert fragment in block
        assert block.index("1+1") < block.index("2+2")
        assert out[0][-1]["role"] == "user"
        assert out[0][-1]["content"] == "3+3?"

    def test_preserves_existing_system_message(self):
        f = FewShotBlockFormatter()
        memory = TextMemory(slots={"examples": [{"input": "a", "output": "b"}]})
        out = f.apply_to_messages(
            [[{"role": "system", "content": "be brief"}, {"role": "user", "content": "?"}]],
            memory,
        )
        # original system message remains in place; the few-shot block is inserted after it
        assert out[0][0] == {"role": "system", "content": "be brief"}
        assert out[0][1]["role"] == "system"
        assert "Input: a" in out[0][1]["content"]
        assert "Output: b" in out[0][1]["content"]
        assert out[0][-1] == {"role": "user", "content": "?"}

    def test_negative_examples_get_negative_header(self):
        f = FewShotBlockFormatter()
        memory = TextMemory(slots={
            "examples": [{"input": "x", "output": "y", "_polarity": "negative"}],
        })
        out = f.apply_to_messages([[{"role": "user", "content": "z"}]], memory)
        assert FewShotBlockFormatter.DEFAULT_NEGATIVE_HEADER in out[0][0]["content"]

    def test_directive_inserted_as_system(self):
        f = FewShotBlockFormatter()
        memory = TextMemory(slots={
            "directive": "be careful",
            "examples": [{"input": "a", "output": "b"}],
        })
        out = f.apply_to_messages([[{"role": "user", "content": "?"}]], memory)
        assert out[0][0]["role"] == "system"
        assert "be careful" in out[0][0]["content"]

    def test_empty_examples_returns_unchanged(self):
        f = FewShotBlockFormatter()
        memory = TextMemory(slots={"examples": []})
        original = [[{"role": "user", "content": "hi"}]]
        out = f.apply_to_messages(original, memory)
        assert out == original


class TestCallableScorer:
    def test_scores_each_prompt(self):
        # reward = length of the prompt
        scorer = _CallableScorer(lambda p, q: float(len(p)))
        scores = scorer.score(["a", "abc", "ab"])
        assert scores == [1.0, 3.0, 2.0]

    def test_query_alignment_required(self):
        scorer = _CallableScorer(lambda p, q: 0.0)
        with pytest.raises(ValueError):
            scorer.score(["a", "b"], queries=[{"x": 1}])

    def test_passes_queries_through(self):
        seen = []
        scorer = _CallableScorer(lambda p, q: (seen.append((p, q)), 0.0)[1])
        scorer.score(["a"], queries=[{"k": "v"}])
        assert seen == [("a", {"k": "v"})]

    def test_satisfies_base_scorer(self):
        scorer = _CallableScorer(lambda p, q: 0.0)
        assert isinstance(scorer, BaseScorer)


def _constant_scorer(value: float = 0.5):
    """Trivial per-row scorer returning a fixed score regardless of input."""
    def score(response, row):
        return value
    return score


@pytest.fixture(scope="module")
def tiny_lm():
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestTaskEvaluationScorerSmoke:
    def test_runs_end_to_end(self, tiny_lm):
        model, tokenizer = tiny_lm
        scorer = TaskEvaluationScorer(
            task_lm=model,
            tokenizer=tokenizer,
            dev_set=[{"input": "hello"}, {"input": "world"}],
            row_scorer=_constant_scorer(value=0.7),
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        scores = scorer.score(["be brief", "be concise"])
        assert scores == [0.7, 0.7]

    def test_max_dev_size_respected(self, tiny_lm):
        model, tokenizer = tiny_lm
        captured = []

        def capture_scorer(response, row):
            captured.append(row["input"])
            return 1.0

        scorer = TaskEvaluationScorer(
            task_lm=model,
            tokenizer=tokenizer,
            dev_set=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
            row_scorer=capture_scorer,
            gen_kwargs={"max_new_tokens": 1, "do_sample": False},
            max_dev_size=2,
        )
        scorer.score(["x"])
        assert captured == ["a", "b"]

    def test_batched_dev_rows_deterministic(self, tiny_lm):
        """Greedy generation on the batched path should be reproducible."""
        model, tokenizer = tiny_lm

        def response_length_scorer(response, row):
            return float(len(response))

        scorer = TaskEvaluationScorer(
            task_lm=model,
            tokenizer=tokenizer,
            dev_set=[{"input": "hello"}, {"input": "world"}, {"input": "test"}],
            row_scorer=response_length_scorer,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        prompts = ["be helpful", "be brief"]

        first = scorer.score(prompts)
        second = scorer.score(prompts)
        assert first == second
        assert len(first) == len(prompts)
        assert all(isinstance(s, float) for s in first)

    def test_padding_side_restored(self, tiny_lm):
        """Tokenizer.padding_side should be restored to its original value after scoring."""
        model, tokenizer = tiny_lm
        original = tokenizer.padding_side
        try:
            tokenizer.padding_side = "right"
            scorer = TaskEvaluationScorer(
                task_lm=model,
                tokenizer=tokenizer,
                dev_set=[{"input": "a"}, {"input": "b"}],
                row_scorer=_constant_scorer(value=0.5),
                gen_kwargs={"max_new_tokens": 1, "do_sample": False},
            )
            scorer.score(["x"])
            assert tokenizer.padding_side == "right"
        finally:
            tokenizer.padding_side = original


class TestRetrievalProposer:
    def test_returns_top_n(self):
        class FakeEncoder:
            def encode(self, text):
                return text

        class FakeIndex:
            def __init__(self, items):
                self.items = items

            def query(self, embedding, top_k):
                return self.items[:top_k]

        proposer = RetrievalProposer(
            encoder=FakeEncoder(),
            corpus_index=FakeIndex([
                {"text": "a", "score": 1.0},
                {"text": "b", "score": 0.9},
                {"text": "c", "score": 0.5},
            ]),
        )
        out = proposer.propose("query", n=2)
        assert len(out) == 2
        assert [item["text"] for item in out] == ["a", "b"]

    def test_n_above_kmax_raises(self):
        class FakeEncoder:
            def encode(self, text):
                return text

        class FakeIndex:
            def query(self, embedding, top_k):
                return []

        proposer = RetrievalProposer(encoder=FakeEncoder(), corpus_index=FakeIndex(), k_max=5)
        with pytest.raises(ValueError):
            proposer.propose("q", n=10)

    def test_satisfies_base_proposer(self):
        class FakeEncoder:
            def encode(self, text):
                return text

        class FakeIndex:
            def query(self, embedding, top_k):
                return []

        proposer = RetrievalProposer(encoder=FakeEncoder(), corpus_index=FakeIndex())
        assert isinstance(proposer, BaseProposer)


class TestLLMMetaPromptProposerSmoke:
    def test_n_candidates_returned(self, tiny_lm):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 1.0},
        )
        candidates = proposer.propose(seed="be helpful", n=2)
        assert len(candidates) == 2
        assert all(isinstance(c, str) for c in candidates)

    def test_parse_fn_splits_response(self, tiny_lm):
        model, tokenizer = tiny_lm

        # parser claims each response yields three pieces (split on whitespace, max 3)
        def parse(response):
            tokens = response.split()[:3]
            return tokens or [response]

        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="x: {seed}",
            parse_fn=parse,
            gen_kwargs={"max_new_tokens": 6, "do_sample": False},
        )
        candidates = proposer.propose(seed="hi", n=2)
        assert len(candidates) <= 2

    def test_context_keys_available_in_template(self, tiny_lm):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="task={task}; seed={seed}",
            gen_kwargs={"max_new_tokens": 1, "do_sample": False},
            # the tiny double's model vocab exceeds its tokenizer's, so the sampled id can
            # decode to nothing; map any response (even empty) to one candidate so the test
            # exercises template substitution rather than the double's decode luck
            parse_fn=lambda response: [response or "<empty>"],
        )
        # if the format substitution fails, .format will KeyError
        out = proposer.propose(seed="x", n=1, context={"task": "rewrite"})
        assert len(out) == 1


class TestParseConciseInstruction:
    """Pure unit tests for parse_concise_instruction (no model). Validated case matrix (Doc 3 §5)."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Explain the reasoning behind your response, in the form of a short essay, with "
                "supporting explanations, evidence, and examples.\n\nHere's an attempt at refining",
                "Explain the reasoning behind your response, in the form of a short essay, with "
                "supporting explanations, evidence, and examples.",
            ),
            ("Refined prompt:\nBe concise and direct.", "Be concise and direct."),
            ('"Be concise."', "Be concise."),
            ("Be concise. Here is an attempt at", "Be concise."),
            ("Be concise.", "Be concise."),
            ("Reply with the city name only", "Reply with the city name only"),
            ("   ", ""),
            ("Here is a refined version of the prompt:\n\nBe concise and direct.", "Be concise and direct."),
            (
                "Refined prompt:\n\nReply with only the answer.\n\nThis version preserves intent.",
                "Reply with only the answer.",
            ),
            ("Sure, here's a refined version of the prompt.\n\nBe direct.", "Be direct."),
            ("Sure!\n\nHere's the refined prompt:\n\nBe terse.", "Be terse."),
            ("Refined prompt:", ""),
            ("Here is a refined version of the prompt:", ""),
            ("Here's an attempt at refining", ""),
            ("Sure!", ""),
            (
                "Here is the rule you must always follow: never speculate.",
                "Here is the rule you must always follow: never speculate.",
            ),
            ("Answer in one word. Do not explain.", "Answer in one word. Do not explain."),
            ('"Reply with a clear, complete answer. This version improves', "Reply with a clear, complete answer."),
            (
                '"Reply with a clear answer.\n\nThis version preserves the original intent."',
                "Reply with a clear answer.",
            ),
            ('"Be concise. Here is an attempt at', "Be concise."),
            ('Reply briefly."', "Reply briefly."),
            ("Don't speculate. Answer plainly.", "Don't speculate. Answer plainly."),
            ("'Tis the season, isn't it.", "'Tis the season, isn't it."),
            ('Answer with "yes" or "no" only.', 'Answer with "yes" or "no" only.'),
            ("1. Provide a concise and accurate answer", "Provide a concise and accurate answer"),
            (
                "1. Provide a concise and accurate answer to the question.",
                "Provide a concise and accurate answer to the question.",
            ),
            ("2) Reply with only the answer.", "Reply with only the answer."),
            ("- Be brief and factual.", "Be brief and factual."),
            ("10. Answer plainly and briefly.", "Answer plainly and briefly."),
            ("1. Refined prompt:\nBe concise and direct.", "Be concise and direct."),
            ("1.", ""),
            ("Concise", ""),
            (
                "Provide a thorough and well-structured answer that carefully explains the reasoning "
                "behind each point while drawing on relevant background knowledge and concrete examples "
                "so the reader can follow the full chain of logic from the initial premises through the "
                "intermediate steps and finally to the conclusion without losing track of how the "
                "various ideas connect to one another and",
                "",
            ),
            (
                "Provide a thorough and well-structured answer that carefully explains the reasoning "
                "behind each point while drawing on relevant background knowledge and concrete examples "
                "so the reader can follow the chain of logic to the conclusion.",
                "Provide a thorough and well-structured answer that carefully explains the reasoning "
                "behind each point while drawing on relevant background knowledge and concrete examples "
                "so the reader can follow the chain of logic to the conclusion.",
            ),
        ],
    )
    def test_matrix(self, text, expected):
        result = parse_concise_instruction(text)
        assert result == ([expected] if expected else [])


class TestParseWhole:
    def test_strips_and_wraps(self):
        assert parse_whole("  hello world  ") == ["hello world"]

    def test_empty_returns_empty_list(self):
        assert parse_whole("   ") == []
        assert parse_whole("") == []

    def test_preserves_internal_structure(self):
        text = "line one\n\nline two\n- bullet"
        assert parse_whole(f"  {text}  ") == [text]


class TestParseFencedOrWhole:
    def test_preserves_multi_paragraph_bulleted_content(self):
        text = (
            "First paragraph with detail.\n\n"
            "Second paragraph.\n\n"
            "Key rules:\n- always cite sources\n- never speculate\n\n"
            "Final paragraph wrapping up."
        )
        out = parse_fenced_or_whole(text)
        assert out == [text]
        # all four paragraphs survive
        assert out[0].count("\n\n") == 3
        assert "- always cite sources" in out[0]

    def test_extracts_first_fenced_block(self):
        text = (
            "Here is the new instruction:\n"
            "```\nDo the task carefully.\n\nUse domain facts.\n```\n"
            "Hope this helps!\n```\nignored second block\n```"
        )
        out = parse_fenced_or_whole(text)
        assert out == ["Do the task carefully.\n\nUse domain facts."]

    def test_strips_leading_lead_in_line(self):
        text = "Here is the revised prompt:\n\nBe thorough and structured.\n\nExplain each step."
        out = parse_fenced_or_whole(text)
        assert out == ["Be thorough and structured.\n\nExplain each step."]

    def test_unterminated_fence_takes_rest(self):
        text = "Sure:\n```\nThe instruction continues to the very end with no closing fence"
        out = parse_fenced_or_whole(text)
        assert out == ["The instruction continues to the very end with no closing fence"]

    def test_strips_language_tag_after_fence(self):
        text = "```text\nActual instruction body.\n```"
        out = parse_fenced_or_whole(text)
        assert out == ["Actual instruction body."]

    def test_empty_returns_empty_list(self):
        assert parse_fenced_or_whole("   ") == []
        assert parse_fenced_or_whole("") == []

    def test_no_word_cap(self):
        long_text = " ".join(["word"] * 200) + " end."
        out = parse_fenced_or_whole(long_text)
        assert out == [long_text]


class TestProposeDropsRejectedCandidates:
    def test_preamble_only_candidate_dropped(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            parse_fn=parse_concise_instruction,
        )
        monkeypatch.setattr(
            proposer,
            "_sample_responses",
            lambda rendered, n: [
                "Refined prompt:",
                "Here is a refined version of the prompt:\n\nBe direct.",
            ],
        )
        # one response is a preamble-only reject; the survivor is returned for n=1
        assert proposer.propose(seed="x", n=1) == ["Be direct."]

    def test_resamples_when_first_batch_all_rejected(self, tiny_lm, monkeypatch):
        """The first sampling round yields only parser-rejected responses; a later round supplies a
        usable candidate. `propose` must resample rather than return an empty list."""
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            parse_fn=parse_concise_instruction,
            max_attempts=4,
        )
        rounds = [
            ["Refined prompt:", "Sure!"],  # round 1: all rejected
            ["Be direct and concise."],  # round 2: one survivor
        ]
        calls = {"n": 0}

        def fake_sample(rendered, n):
            batch = rounds[calls["n"]] if calls["n"] < len(rounds) else []
            calls["n"] += 1
            return batch

        monkeypatch.setattr(proposer, "_sample_responses", fake_sample)
        assert proposer.propose(seed="x", n=1) == ["Be direct and concise."]
        assert calls["n"] == 2  # resampled exactly once after the all-reject round

    def test_returns_fewer_than_n_when_budget_exhausted(self, tiny_lm, monkeypatch):
        """If every round is fully rejected, `propose` returns an empty list after `max_attempts`
        rounds rather than looping forever."""
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            parse_fn=parse_concise_instruction,
            max_attempts=3,
        )
        calls = {"n": 0}

        def fake_sample(rendered, n):
            calls["n"] += 1
            return ["Refined prompt:"]  # always rejected

        monkeypatch.setattr(proposer, "_sample_responses", fake_sample)
        assert proposer.propose(seed="x", n=2) == []
        assert calls["n"] == 3  # bounded by max_attempts


class TestProposerChatTemplateMode:
    def test_default_routes_through_chat_template(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        calls = []
        original = tokenizer.apply_chat_template

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(proposer.tokenizer, "apply_chat_template", spy)
        proposer.propose(seed="be helpful", n=1)
        assert len(calls) >= 1

    def test_use_chat_template_false_never_wraps(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
            use_chat_template=False,
        )
        calls = []
        original = tokenizer.apply_chat_template

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(proposer.tokenizer, "apply_chat_template", spy)
        proposer.propose(seed="be helpful", n=1)
        assert calls == []

    def test_chat_mode_false_without_template(self):
        class _StubTokenizer:
            chat_template = None

        for flag in (None, True, False):
            proposer = LLMMetaPromptProposer.__new__(LLMMetaPromptProposer)
            proposer.tokenizer = _StubTokenizer()
            proposer.use_chat_template = flag
            assert proposer._chat_mode() is False


class TestProposerGenerationHygiene:
    def test_generate_receives_mask_and_pad_id(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        captured = []
        original = model.generate

        def spy(input_ids=None, **kwargs):
            captured.append(kwargs)
            return original(input_ids, **kwargs)

        monkeypatch.setattr(model, "generate", spy)
        proposer.propose(seed="be helpful", n=1)
        assert captured
        assert captured[0].get("attention_mask") is not None
        assert captured[0].get("pad_token_id") is not None

    def test_caller_supplied_pad_id_not_overridden(self, tiny_lm, monkeypatch):
        model, tokenizer = tiny_lm
        proposer = LLMMetaPromptProposer(
            llm=model,
            tokenizer=tokenizer,
            meta_prompt_template="Rewrite: {seed}",
            gen_kwargs={"max_new_tokens": 2, "do_sample": False, "pad_token_id": 123},
        )
        captured = []
        original = model.generate

        def spy(input_ids=None, **kwargs):
            captured.append(kwargs)
            return original(input_ids, **kwargs)

        monkeypatch.setattr(model, "generate", spy)
        proposer.propose(seed="be helpful", n=1)
        assert captured[0].get("pad_token_id") == 123


class TestRandomSelector:
    def test_returns_k_items(self):
        sel = RandomSelector(seed=0)
        out = sel.select(list(range(10)), k=3)
        assert len(out) == 3
        assert all(x in range(10) for x in out)

    def test_no_replacement(self):
        sel = RandomSelector(seed=0)
        out = sel.select([1, 2, 3, 4, 5], k=5)
        assert sorted(out) == [1, 2, 3, 4, 5]

    def test_k_exceeds_pool_returns_all(self):
        sel = RandomSelector(seed=0)
        out = sel.select([1, 2, 3], k=10)
        assert sorted(out) == [1, 2, 3]


class TestTopKSelector:
    def test_picks_highest_scores(self):
        # scorer = string length
        scorer = _CallableScorer(lambda p, q: float(len(p)))
        sel = TopKSelector(scorer=scorer)
        out = sel.select(["a", "abcdef", "abc", "ab"], k=2)
        # the two highest-length items are "abcdef" (6) and "abc" (3)
        assert out == ["abcdef", "abc"]

    def test_k_exceeds_pool(self):
        scorer = _CallableScorer(lambda p, q: 1.0)
        sel = TopKSelector(scorer=scorer)
        out = sel.select(["a", "b"], k=5)
        assert len(out) == 2

    def test_satisfies_base_selector(self):
        scorer = _CallableScorer(lambda p, q: 0.0)
        assert isinstance(TopKSelector(scorer=scorer), BaseSelector)


class TestMMRSelector:
    def test_diversity_param_effect(self):
        # crude letter-level encoder; each letter -> a one-hot, lowercased
        class LetterEncoder:
            def encode(self, text):
                vec = np.zeros(26, dtype=float)
                for ch in text.lower():
                    if "a" <= ch <= "z":
                        vec[ord(ch) - ord("a")] += 1
                return vec

        items = ["apple", "banana", "apricot"]  # apple ~ apricot share many letters
        sel_relevance = MMRSelector(encoder=LetterEncoder(), lambda_param=1.0)
        sel_diverse = MMRSelector(encoder=LetterEncoder(), lambda_param=0.0)

        out_rel = sel_relevance.select(items, query="apple", k=2)
        out_div = sel_diverse.select(items, query="apple", k=2)

        # pure relevance picks apple first; pure diversity ranks differently
        assert out_rel[0] == "apple"
        assert out_div != out_rel

    def test_lambda_out_of_range_raises(self):
        class E:
            def encode(self, text):
                return np.zeros(3)
        with pytest.raises(ValueError):
            MMRSelector(encoder=E(), lambda_param=1.5)


class TestDenseRetrievalSelector:
    def test_picks_nearest(self):
        # 2-D embeddings on the unit circle: cos similarity now actually ranks neighbors.
        class CircleEncoder:
            def encode(self, text):
                idx = float(text)
                angle = idx * (np.pi / 6)  # 30° per step
                return np.array([np.cos(angle), np.sin(angle)])

        sel = DenseRetrievalSelector(encoder=CircleEncoder())
        out = sel.select(items=["1", "2", "3", "4", "5"], query="3", k=2)
        assert out[0] == "3"  # 3 itself is most similar to 3
        assert out[1] in {"2", "4"}  # nearest neighbors

    def test_embedding_key_uses_precomputed(self):
        # encoder should not be called for items that already carry an embedding
        class FailEncoder:
            def __init__(self):
                self.called = 0

            def encode(self, text):
                self.called += 1
                return np.zeros(2)

        encoder = FailEncoder()
        sel = DenseRetrievalSelector(encoder=encoder, embedding_key="emb")
        items = [
            {"text": "a", "emb": np.array([1.0, 0.0])},
            {"text": "b", "emb": np.array([0.0, 1.0])},
        ]
        out = sel.select(items=items, query=np.array([1.0, 0.0]), k=1)
        assert out[0]["text"] == "a"
        # encoder only ever called for the query path; never for items
        assert encoder.called == 0

    def test_invalid_similarity_raises(self):
        class E:
            def encode(self, text):
                return np.zeros(2)
        with pytest.raises(ValueError):
            DenseRetrievalSelector(encoder=E(), similarity="invalid")

    def test_query_required(self):
        class E:
            def encode(self, text):
                return np.zeros(2)
        sel = DenseRetrievalSelector(encoder=E())
        with pytest.raises(ValueError):
            sel.select(["a"], k=1)


class TestGenerateWithSystemPrompt:
    def test_smoke_returns_one_per_query(self, tiny_lm):
        from steerability.algorithms.input_control.common.generation import generate_with_system_prompt
        model, tokenizer = tiny_lm
        out = generate_with_system_prompt(
            model, tokenizer, "be brief", ["hello", "world", "test"],
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        assert len(out) == 3
        assert all(isinstance(o, str) for o in out)

    def test_empty_queries_returns_empty(self, tiny_lm):
        from steerability.algorithms.input_control.common.generation import generate_with_system_prompt
        model, tokenizer = tiny_lm
        assert generate_with_system_prompt(model, tokenizer, "x", []) == []

    def test_padding_side_restored(self, tiny_lm):
        from steerability.algorithms.input_control.common.generation import generate_with_system_prompt
        model, tokenizer = tiny_lm
        original = tokenizer.padding_side
        try:
            tokenizer.padding_side = "right"
            generate_with_system_prompt(
                model, tokenizer, "x", ["a", "b"],
                gen_kwargs={"max_new_tokens": 1, "do_sample": False},
            )
            assert tokenizer.padding_side == "right"
        finally:
            tokenizer.padding_side = original
