"""Tests for CPO — causal prompt optimization."""
from __future__ import annotations

import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.cpo import CPO, CPOArgs
from steerability.algorithms.input_control.cpo.control import CPOMemory
from steerability.algorithms.input_control.cpo.utils import causal_reward, refinement_meta_prompt

TINY_LM = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# not in the CI model set; tests that load it are marked `network`
TINY_BERT = "hf-internal-testing/tiny-random-BertModel"


def _constant_scorer(value: float = 0.5):
    def score(response, row):
        return value
    return score


@pytest.fixture(scope="module")
def tiny_lm():
    model = AutoModelForCausalLM.from_pretrained(TINY_LM, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LM, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture(scope="module")
def offline_rows():
    return [
        {"query": "q1", "prompt": "be brief", "score": 0.5},
        {"query": "q1", "prompt": "be polite", "score": 0.7},
        {"query": "q2", "prompt": "be brief", "score": 0.4},
        {"query": "q2", "prompt": "be polite", "score": 0.6},
        {"query": "q3", "prompt": "be brief", "score": 0.3},
        {"query": "q3", "prompt": "be polite", "score": 0.8},
    ]


class TestCPOArgs:
    def test_minimal_offline_data(self, offline_rows):
        args = CPOArgs(seed_prompt="x", offline_data=offline_rows)
        assert args.seed_prompt == "x"

    def test_train_dataset_requires_scorer_and_lm(self):
        with pytest.raises(ValueError, match="row_scorer"):
            CPOArgs(seed_prompt="x", train_dataset=[{"input": "a"}], prompt_lm="model")
        with pytest.raises(ValueError, match="prompt_lm"):
            CPOArgs(seed_prompt="x", train_dataset=[{"input": "a"}], row_scorer=_constant_scorer())

    def test_neither_offline_nor_train_raises(self):
        with pytest.raises(ValueError, match="offline_data"):
            CPOArgs(seed_prompt="x")

    def test_prompt_lm_requires_its_tokenizer(self, offline_rows):
        with pytest.raises(ValueError, match="`prompt_tokenizer` is required with `prompt_lm`"):
            CPOArgs(seed_prompt="x", offline_data=offline_rows, prompt_lm=object())

    def test_prompt_tokenizer_without_prompt_lm_raises(self, offline_rows):
        with pytest.raises(ValueError, match="`prompt_tokenizer` was given without `prompt_lm`"):
            CPOArgs(seed_prompt="x", offline_data=offline_rows, prompt_tokenizer=object())


class _StubProposerLM:
    """A proposer LM double that appends fixed token ids to every prompt."""

    device = torch.device("cpu")

    def __init__(self, continuation_ids):
        self._continuation = torch.tensor([continuation_ids])

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return torch.cat([input_ids, self._continuation.expand(input_ids.size(0), -1)], dim=1)


class _PreferNonSeedScorer:
    """A causal scorer double that scores the seed prompt below every other candidate."""

    def __init__(self, seed_prompt):
        self._seed_prompt = seed_prompt

    def score(self, candidates, queries):
        return [0.0 if candidate == self._seed_prompt else 1.0 for candidate in candidates]


class TestProposerTokenizer:
    def test_the_proposer_decodes_with_prompt_tokenizer_and_prepends_its_survivor(self, monkeypatch):
        from tests.utils.tiny_models import reasoning_tag_tokenizer, tiny_llama, wordlevel_tokenizer

        monkeypatch.setattr("steerability.algorithms.input_control.cpo.control.TextEncoder", lambda *a, **k: None)
        prompt_tokenizer = reasoning_tag_tokenizer(words=("plan", "answer", "x"))
        plan_answer = prompt_tokenizer("plan answer", add_special_tokens=False)["input_ids"]
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=[{"query": "q", "prompt": "p", "score": 1.0}],
            prompt_lm=_StubProposerLM(plan_answer),
            prompt_tokenizer=prompt_tokenizer,
            memory=CPOMemory(causal_scorer=_PreferNonSeedScorer("be helpful")),
            rounds=1,
            candidates_per_parent=1,
            retained_per_round=1,
        )
        pipeline_tokenizer = wordlevel_tokenizer()
        SteeringPipeline(controls=[cpo], model=tiny_llama(num_layers=2, hidden=16, heads=2),
                         tokenizer=pipeline_tokenizer).steer()
        assert cpo._proposer.tokenizer is prompt_tokenizer

        chat = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "what is 2+2"}]
        adapted = cpo.adapt_messages([chat])
        assert adapted[0][0] == {"role": "system", "content": "plan answer\n\nBe kind."}
        assert chat[0]["content"] == "Be kind."

    def test_empty_seed_raises(self, offline_rows):
        with pytest.raises(ValueError, match="seed_prompt"):
            CPOArgs(seed_prompt="", offline_data=offline_rows)

    def test_invalid_search_params_raise(self, offline_rows):
        with pytest.raises(ValueError):
            CPOArgs(seed_prompt="x", offline_data=offline_rows, rounds=0)


@pytest.mark.network
class TestCausalRewardScorer:
    def test_train_and_score_gbr_path(self, offline_rows):
        scorer = causal_reward.train(
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            seed_prompt="be helpful",
            use_dml=False,
        )
        assert scorer.mode == "gbr"
        scores = scorer.score(["be brief", "be polite"], queries=[{"text": "q1"}, {"text": "q1"}])
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_save_load_roundtrip(self, offline_rows, tmp_path):
        scorer = causal_reward.train(
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            seed_prompt="be helpful",
            use_dml=False,
        )
        path = tmp_path / "scorer.pkl"
        scorer.save(path)
        loaded = causal_reward.CausalRewardScorer.load(path, encoder=scorer.encoder)
        a = scorer.score(["be brief"], queries=[{"text": "q1"}])
        b = loaded.score(["be brief"], queries=[{"text": "q1"}])
        assert a == b

    def test_use_dml_true_without_econml_raises(self, offline_rows):
        # without econml, an explicit use_dml=True raises rather than silently falling back
        if causal_reward._try_import_dml() is not None:
            pytest.skip("econml installed; can't test the missing-dep error path.")
        with pytest.raises(ImportError, match="econml"):
            causal_reward.train(
                offline_data=offline_rows,
                embedding_model=TINY_BERT,
                pca_query_dim=4,
                pca_prompt_dim=4,
                seed_prompt="be helpful",
                use_dml=True,
            )


class TestCPOSteer:
    @pytest.mark.network
    def test_with_offline_data(self, tiny_lm, offline_rows):
        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=2,
            retained_per_round=2,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        assert isinstance(cpo.memory, CPOMemory)
        assert cpo.memory.causal_scorer.mode in ("gbr", "dml")

    def test_without_data_raises(self):
        with pytest.raises(ValueError):
            CPO(seed_prompt="x")


@pytest.mark.network
class TestCPOTreeSearchAndCache:
    def test_adapt_messages_caches_per_query(self, tiny_lm, offline_rows):
        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=2,
            retained_per_round=2,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        # first call populates the cache
        cpo.adapt_messages([[{"role": "user", "content": "hello"}]])
        assert len(cpo.memory.query_cache) == 1
        cached_value = next(iter(cpo.memory.query_cache.values()))

        # second call with the same query should hit the cache
        cpo.adapt_messages([[{"role": "user", "content": "hello"}]])
        assert len(cpo.memory.query_cache) == 1
        assert next(iter(cpo.memory.query_cache.values())) == cached_value

        # different query -> new cache entry
        cpo.adapt_messages([[{"role": "user", "content": "different question"}]])
        assert len(cpo.memory.query_cache) == 2

    def test_cache_disabled(self, tiny_lm, offline_rows):
        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=2,
            retained_per_round=2,
            cache_queries=False,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()
        cpo.adapt_messages([[{"role": "user", "content": "hello"}]])
        assert cpo.memory.query_cache == {}


@pytest.mark.network
class TestCPOCacheKeyNormalizer:
    """`cache_key_normalizer` lets near-duplicate queries share a cache entry; the default is unchanged."""

    @staticmethod
    def _build(normalizer, offline_rows):
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=2,
            retained_per_round=2,
            cache_key_normalizer=normalizer,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        return cpo

    def test_normalizer_merges_near_duplicates(self, tiny_lm, offline_rows, monkeypatch):
        model, tokenizer = tiny_lm
        normalize = lambda q: " ".join(q.split()).lower()
        cpo = self._build(normalize, offline_rows)
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        calls = {"n": 0}
        original = cpo._tree_search

        def counting_tree_search(query):
            calls["n"] += 1
            return original(query)

        monkeypatch.setattr(cpo, "_tree_search", counting_tree_search)

        cpo.adapt_messages([[{"role": "user", "content": "Reverse the list"}]])
        cpo.adapt_messages([[{"role": "user", "content": "  reverse   the LIST  "}]])

        # both queries normalize to the same key: one search, one cache entry
        assert calls["n"] == 1
        assert len(cpo.memory.query_cache) == 1

    def test_default_keeps_near_duplicates_distinct(self, tiny_lm, offline_rows, monkeypatch):
        model, tokenizer = tiny_lm
        cpo = self._build(None, offline_rows)  # default: raw-query hashing
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        calls = {"n": 0}
        original = cpo._tree_search

        def counting_tree_search(query):
            calls["n"] += 1
            return original(query)

        monkeypatch.setattr(cpo, "_tree_search", counting_tree_search)

        cpo.adapt_messages([[{"role": "user", "content": "Reverse the list"}]])
        cpo.adapt_messages([[{"role": "user", "content": "  reverse   the LIST  "}]])

        # distinct raw queries -> two searches, two cache entries
        assert calls["n"] == 2
        assert len(cpo.memory.query_cache) == 2


@pytest.mark.network
class TestCausalRewardVarianceWarning:
    def test_warns_on_constant_scores(self, offline_rows):
        constant_rows = [{**row, "score": 0.7} for row in offline_rows]
        with pytest.warns(UserWarning, match="constant"):
            causal_reward.train(
                offline_data=constant_rows,
                embedding_model=TINY_BERT,
                pca_query_dim=4,
                pca_prompt_dim=4,
                seed_prompt="be helpful",
                use_dml=False,
            )

    def test_warns_on_near_saturated_scores(self):
        # 20 rows: 19 at 0.0, 1 at 1.0 -> top value covers 95%
        rows = [
            {"query": f"q{i}", "prompt": f"prompt {i}", "score": 0.0}
            for i in range(19)
        ]
        rows.append({"query": "q19", "prompt": "prompt 19", "score": 1.0})
        with pytest.warns(UserWarning, match="saturated"):
            causal_reward.train(
                offline_data=rows,
                embedding_model=TINY_BERT,
                pca_query_dim=4,
                pca_prompt_dim=4,
                seed_prompt="be helpful",
                use_dml=False,
            )

    def test_silent_on_varied_scores(self, offline_rows):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            causal_reward.train(
                offline_data=offline_rows,
                embedding_model=TINY_BERT,
                pca_query_dim=4,
                pca_prompt_dim=4,
                seed_prompt="be helpful",
                use_dml=False,
            )
        assert not any(
            isinstance(w.message, UserWarning)
            and ("constant" in str(w.message) or "saturated" in str(w.message))
            for w in caught
        )


class TestCPODefaultTemplate:
    def test_renders_without_stray_format_fields(self):
        rendered = refinement_meta_prompt.CPO_DEFAULT.format(seed="Answer the question.")
        assert "Answer the question." in rendered
        assert "{" not in rendered
        assert "}" not in rendered


@pytest.mark.network
class TestCPOMemoryRoundTrip:
    def test_save_load(self, tiny_lm, offline_rows, tmp_path):
        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=2,
            retained_per_round=2,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        cpo.adapt_messages([[{"role": "user", "content": "hi"}]])
        cpo.memory.save(tmp_path / "mem")
        loaded = CPOMemory.load(tmp_path / "mem", encoder=cpo.memory.causal_scorer.encoder)
        assert loaded.query_cache == cpo.memory.query_cache


class TestCPOTrustRemoteCode:
    """CPO is the documented exception: `trust_remote_code` defaults to True for the paper-default encoder."""

    def test_args_default_true(self, offline_rows):
        assert CPOArgs(seed_prompt="x", offline_data=offline_rows).trust_remote_code is True

    def test_args_opt_out_false(self, offline_rows):
        args = CPOArgs(seed_prompt="x", offline_data=offline_rows, trust_remote_code=False)
        assert args.trust_remote_code is False


class TestCPOBackendPosture:
    """D12: the proposer binds once at steer; the module configuration is declared."""

    @pytest.mark.network
    def test_unset_prompt_lm_never_reads_a_pipeline_attribute_at_adapt(self, tiny_lm, offline_rows):
        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=offline_rows,
            embedding_model=TINY_BERT,
            pca_query_dim=4,
            pca_prompt_dim=4,
            rounds=1,
            candidates_per_parent=1,
            retained_per_round=1,
            proposer_gen_kwargs={"max_new_tokens": 4, "do_sample": True, "temperature": 0.9},
        )
        pipeline = SteeringPipeline(controls=[cpo], model=model, tokenizer=tokenizer)
        pipeline.steer()

        # the proposer bound the model at steer; adaptation consults no pipeline attribute
        pipeline.model = None
        del pipeline
        adapted = cpo.adapt_messages([[{"role": "user", "content": "what is 2+2"}]])
        assert adapted is not None
        assert adapted[0][0]["role"] == "system"

    def test_module_configuration_verdict_on_engine(self):
        from steerability.algorithms.core.execution import BackendSpec, ModelAccess

        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=[{"query": "q", "prompt": "p", "score": 1.0}],
            embedding_model=TINY_BERT,
        )
        assert cpo.steer_access() is ModelAccess.MODULE
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[cpo])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        (failure,) = report.failures_for("generate")
        assert failure.message == (
            "CPO is unsupported at generate on backend kind 'vllm': missing IN_PROCESS_TORCH; "
            "set prompt_lm to run CPO's per-query search off the pipeline model."
        )

    def test_aux_prompt_lm_configuration_is_supported_on_engines(self, tiny_lm):
        from steerability.algorithms.core.execution import BackendSpec, ModelAccess

        model, tokenizer = tiny_lm
        cpo = CPO(
            seed_prompt="be helpful",
            offline_data=[{"query": "q", "prompt": "p", "score": 1.0}],
            embedding_model=TINY_BERT,
            prompt_lm=model,
            prompt_tokenizer=tokenizer,
        )
        assert cpo.steer_access() is ModelAccess.ROLLOUTS
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[cpo])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.supported("generate")
        (step,) = report.plan.steps
        assert step.venue == "session"
