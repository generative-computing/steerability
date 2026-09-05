"""Tests for the RAD (Reward-Augmented Decoding) output control.

Covers the end-to-end steer/generate loop over the CI models, args validation, reward-score
extraction (`score_index` / `score_transform`), the cached/stateless equivalence (including a prefix
rewind), the degrade path when the cached preconditions fail, and the lifecycle posture.

Hub-free component tests build tiny sequence classifiers via config classes saved to `tmp_path` and
share the wordlevel tokenizer, so the reward model and the language model share a vocabulary and the
cached path is exercised (`tests/utils/tiny_models.py`).
"""
import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GraniteConfig,
    GraniteForCausalLM,
    GraniteMoeHybridConfig,
    GraniteMoeHybridForCausalLM,
    LlamaConfig,
    LlamaForSequenceClassification,
)

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.granite_heads import (
    GraniteForSequenceClassification,
    GraniteMoeHybridForSequenceClassification,
)
from steerability.algorithms.output_control.common.loading import load_sequence_classifier
from steerability.algorithms.output_control.common.processors.value_guided import ValueStepRecord
from steerability.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext
from steerability.algorithms.output_control.common.values.reward_model import (
    CachedRewardModelValue,
    RewardModelValue,
    extract_score,
)
from steerability.algorithms.output_control.rad.args import RADArgs
from steerability.algorithms.output_control.rad.control import RAD
from tests.utils.sweep import build_param_grid
from tests.utils.tiny_models import reasoning_tag_tokenizer, tiny_llama, wordlevel_tokenizer

VOCAB = 100

RAD_GRID = {
    "beta": [0.0, 2.0, 50.0],
    "top_k": [2, 20],
    "invert": [False, True],
}


def _decoder_classifier(tmp_path, num_labels=2, vocab=VOCAB):
    """A hub-free decoder-only (Llama) sequence classifier sharing the wordlevel vocabulary."""
    cfg = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=vocab,
        num_labels=num_labels, pad_token_id=2,
    )
    clf = LlamaForSequenceClassification(cfg).eval()
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    clf.save_pretrained(str(tmp_path))
    return str(tmp_path)


def _encoder_classifier(tmp_path, vocab=VOCAB):
    """A hub-free encoder (Bert) sequence classifier sharing the wordlevel vocabulary.

    Bert's forward accepts no `past_key_values` / `cache_position`, so RAD's cached preconditions
    fail and it degrades to the stateless value.
    """
    cfg = BertConfig(
        vocab_size=vocab, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, num_labels=2, pad_token_id=2,
    )
    clf = BertForSequenceClassification(cfg).eval()
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    clf.save_pretrained(str(tmp_path))
    return str(tmp_path)


def _granite_causal_lm_path(tmp_path, vocab=VOCAB):
    """A hub-free `GraniteForCausalLM` checkpoint sharing the wordlevel vocabulary."""
    cfg = GraniteConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=vocab, pad_token_id=2,
    )
    GraniteForCausalLM(cfg).eval().save_pretrained(str(tmp_path))
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    return str(tmp_path)


def _granitemoehybrid_causal_lm_path(tmp_path, vocab=VOCAB):
    """A hub-free `GraniteMoeHybridForCausalLM` checkpoint with all-attention layers.

    All `layer_types` are `"attention"` (no Mamba, no experts), the shape the `granite-4.0-350m`
    reward backbone has, so the decoder-only cached path is available.
    """
    num_layers = 2
    cfg = GraniteMoeHybridConfig(
        hidden_size=16, intermediate_size=32, shared_intermediate_size=32,
        num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=vocab, pad_token_id=2, num_local_experts=1, num_experts_per_tok=1,
        layer_types=["attention"] * num_layers,
        mamba_expand=1, mamba_n_heads=4, mamba_d_state=8, mamba_d_conv=2,
    )
    GraniteMoeHybridForCausalLM(cfg).eval().save_pretrained(str(tmp_path))
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    return str(tmp_path)


def _pipeline(control, model, tokenizer):
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


class _ForceValue(BaseCandidateValue):
    """A value that scores one target candidate id at 1.0 and every other at 0.0."""

    def __init__(self, target: int):
        self.target = target
        self.supports_batching = True

    def score(self, ctx: StepContext) -> torch.Tensor:
        return (ctx.candidate_ids == self.target).float()


# end-to-end
@pytest.mark.parametrize("conf", build_param_grid(RAD_GRID))
def test_rad_end_to_end(tmp_path, conf):
    """RAD steers and generates on every param combo, returning a well-shaped tensor."""
    torch.manual_seed(0)
    rm_path = _decoder_classifier(tmp_path)
    model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    tokenizer = wordlevel_tokenizer()

    rad = RAD(reward_model_id=rm_path, beta=conf["beta"], top_k=conf["top_k"], invert=conf["invert"])
    pipeline = _pipeline(rad, model, tokenizer)

    prompt = tokenizer("the cat", return_tensors="pt").input_ids
    for do_sample in (False, True):
        out = pipeline.generate(input_ids=prompt, max_new_tokens=4, do_sample=do_sample, eos_token_id=None)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2 and out.size(0) == 1


def test_rad_beta_shifts_distribution(tmp_path):
    """With a deterministic value, beta=0 leaves scores unshifted while a large beta reranks them."""
    rm_path = _decoder_classifier(tmp_path)
    model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    tokenizer = wordlevel_tokenizer()
    prefix = torch.tensor([[0, 4, 5]])
    scores = torch.randn(1, VOCAB)

    rad_zero = RAD(reward_model_id=rm_path, beta=0.0, top_k=VOCAB)
    _pipeline(rad_zero, model, tokenizer)
    rad_zero._value = _ForceValue(target=7)
    out_zero = rad_zero.get_logits_processors(prefix, {})[0](prefix, scores.clone())

    rad_big = RAD(reward_model_id=rm_path, beta=100.0, top_k=VOCAB)
    _pipeline(rad_big, model, tokenizer)
    rad_big._value = _ForceValue(target=7)
    out_big = rad_big.get_logits_processors(prefix, {})[0](prefix, scores.clone())

    # beta=0 preserves the base ranking; a dominating beta forces the target to the top
    assert torch.argmax(out_zero) == torch.argmax(scores)
    assert torch.argmax(out_big).item() == 7


# args validation
class TestRADArgsValidation:
    def test_missing_beta_raises(self):
        with pytest.raises(TypeError):
            RAD(reward_model_id="x")

    def test_missing_reward_model_id_raises(self):
        with pytest.raises(TypeError):
            RAD(beta=1.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            RAD(reward_model_id="x", beta=-1.0)

    def test_top_k_below_one_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            RAD(reward_model_id="x", beta=1.0, top_k=0)

    def test_bad_score_transform_raises(self):
        with pytest.raises(ValueError, match="score_transform"):
            RAD(reward_model_id="x", beta=1.0, score_transform="nope")

    def test_empty_reward_model_id_raises(self):
        with pytest.raises(ValueError, match="reward_model_id"):
            RAD(reward_model_id="", beta=1.0)


# score extraction
class TestScoreExtraction:
    def _output(self, logits):
        class _Out:
            pass
        out = _Out()
        out.logits = torch.tensor(logits)
        return out

    def test_none_selects_column(self):
        out = self._output([[1.0, 2.0, 3.0]])
        assert torch.allclose(extract_score(out, 0, "none"), torch.tensor([1.0]))
        assert torch.allclose(extract_score(out, 2, "none"), torch.tensor([3.0]))

    def test_sigmoid_then_select(self):
        out = self._output([[0.0, 10.0]])
        assert torch.allclose(extract_score(out, 0, "sigmoid"), torch.tensor([0.5]), atol=1e-5)
        assert extract_score(out, 1, "sigmoid").item() == pytest.approx(1.0, abs=1e-3)

    def test_softmax_then_select(self):
        out = self._output([[1.0, 1.0]])
        assert torch.allclose(extract_score(out, 0, "softmax"), torch.tensor([0.5]), atol=1e-5)

    def test_raw_tensor_without_logits_attr(self):
        raw = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(extract_score(raw, 1, "none"), torch.tensor([6.0]))

    def test_reward_model_value_shape(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path, num_labels=3)
        from steerability.algorithms.output_control.common.loading import load_sequence_classifier
        rm, rm_tok = load_sequence_classifier(rm_path, device="cpu")
        value = RewardModelValue(rm, rm_tok, score_index=1, score_transform="softmax", shared_vocab=True)
        ctx = StepContext(
            prefix_ids=torch.tensor([[0, 4, 5]]),
            candidate_ids=torch.tensor([[6, 7, 8, 9]]),
            lm_tokenizer=wordlevel_tokenizer(),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        scores = value.score(ctx)
        assert scores.shape == (1, 4)


# cached / stateless equivalence
class TestCachedEquivalence:
    def _values(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        from steerability.algorithms.output_control.common.loading import load_sequence_classifier
        rm, rm_tok = load_sequence_classifier(rm_path, device="cpu")
        cached = CachedRewardModelValue(rm, rm_tok, score_index=0, score_transform="none")
        stateless = RewardModelValue(rm, rm_tok, score_index=0, score_transform="none", shared_vocab=True)
        return cached, stateless

    def _ctx(self, prefix, candidates):
        return StepContext(
            prefix_ids=prefix,
            candidate_ids=candidates,
            lm_tokenizer=wordlevel_tokenizer(),
            attention_mask=torch.ones_like(prefix),
        )

    def test_multi_step_equivalence(self, tmp_path):
        cached, stateless = self._values(tmp_path)
        candidates = torch.tensor([[6, 7, 8, 9, 3]])
        prefix = torch.tensor([[0, 4]])
        for extra in (5, 6, 7):  # extend the prefix step by step
            prefix = torch.cat([prefix, torch.tensor([[extra]])], dim=1)
            ctx = self._ctx(prefix, candidates)
            torch.testing.assert_close(cached.score(ctx), stateless.score(ctx), atol=1e-5, rtol=1e-5)

    def test_rewind_equivalence(self, tmp_path):
        cached, stateless = self._values(tmp_path)
        candidates = torch.tensor([[6, 7, 8]])
        long_prefix = torch.tensor([[0, 4, 5, 6, 7]])
        short_prefix = torch.tensor([[0, 4, 5]])  # rewind to a shorter prefix
        cached.score(self._ctx(long_prefix, candidates))
        torch.testing.assert_close(
            cached.score(self._ctx(short_prefix, candidates)),
            stateless.score(self._ctx(short_prefix, candidates)),
            atol=1e-5, rtol=1e-5,
        )

    def test_pad_id_candidate_scores_the_candidate_on_both_paths(self, tmp_path):
        """A candidate equal to the classifier's pad id (eos after `train_prefix_reward_model`) is scored at
        its own position on both paths rather than pooled back onto the prefix."""
        rm_path = _decoder_classifier(tmp_path)
        rm, rm_tok = load_sequence_classifier(rm_path, device="cpu")
        pad_id = rm.config.pad_token_id
        assert pad_id is not None
        cached = CachedRewardModelValue(rm, rm_tok, score_index=0, score_transform="none")
        stateless = RewardModelValue(rm, rm_tok, score_index=0, score_transform="none", shared_vocab=True)
        prefix = torch.tensor([[0, 4, 5, 6]])
        candidates = torch.tensor([[7, pad_id, 8]])
        ctx = self._ctx(prefix, candidates)
        stateless_scores = stateless.score(ctx)
        torch.testing.assert_close(cached.score(ctx), stateless_scores, atol=1e-5, rtol=1e-5)
        with torch.inference_mode():
            prefix_only = rm(input_ids=prefix, attention_mask=torch.ones_like(prefix)).logits[0, 0]
        assert not torch.isclose(stateless_scores[0, 1], prefix_only, atol=1e-6)


# degrade path
def test_rad_degrades_on_encoder_reward_model(tmp_path):
    """An encoder reward model fails the cached preconditions; RAD warns once and still generates."""
    rm_path = _encoder_classifier(tmp_path)
    model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    tokenizer = wordlevel_tokenizer()

    rad = RAD(reward_model_id=rm_path, beta=5.0, efficient=True)
    with pytest.warns(UserWarning, match="decoder-only"):
        pipeline = _pipeline(rad, model, tokenizer)

    assert isinstance(rad._value, RewardModelValue)
    assert rad._value.shared_vocab is True  # degraded to the exact shared-vocab stateless value

    prompt = tokenizer("the cat", return_tensors="pt").input_ids
    gen_out = pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert isinstance(gen_out, torch.Tensor)


# posture
class TestRADPosture:
    def test_supports_batching_cached_false(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=rm_path, beta=1.0, efficient=True)
        _pipeline(rad, model, wordlevel_tokenizer())
        assert isinstance(rad._value, CachedRewardModelValue)
        assert rad.supports_batching is False

    def test_supports_batching_stateless_true(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=rm_path, beta=1.0, efficient=False)
        _pipeline(rad, model, wordlevel_tokenizer())
        assert isinstance(rad._value, RewardModelValue)
        assert rad.supports_batching is True

    def test_steer_returns_none(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=rm_path, beta=1.0)
        assert rad.steer(model, tokenizer=wordlevel_tokenizer()) is None

    def test_unsteered_raises(self, tmp_path):
        rad = RAD(reward_model_id="x", beta=1.0)
        with pytest.raises(RuntimeError, match="steer"):
            rad.get_logits_processors(torch.tensor([[0, 4, 5]]), {})


# granite sequence-classification heads
class TestGraniteHeads:
    """Granite causal-LM checkpoints load as scalar-head classifiers and back RAD's cached path."""

    def test_granite_loads_as_classifier(self, tmp_path):
        path = _granite_causal_lm_path(tmp_path)
        model, _ = load_sequence_classifier(path, device="cpu", hf_model_kwargs={"num_labels": 1})
        assert isinstance(model, GraniteForSequenceClassification)
        assert model.config.num_labels == 1

    def test_granitemoehybrid_loads_as_classifier(self, tmp_path):
        path = _granitemoehybrid_causal_lm_path(tmp_path)
        model, _ = load_sequence_classifier(path, device="cpu", hf_model_kwargs={"num_labels": 1})
        assert isinstance(model, GraniteMoeHybridForSequenceClassification)
        assert model.config.num_labels == 1

    @pytest.mark.parametrize(
        "builder", [_granite_causal_lm_path, _granitemoehybrid_causal_lm_path]
    )
    def test_granite_reward_model_reaches_cached_path(self, tmp_path, builder):
        """A Granite classifier sharing the LM vocabulary resolves the cached path and steers."""
        rm_path = builder(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        rad = RAD(reward_model_id=rm_path, beta=5.0, top_k=8, score_transform="sigmoid", efficient=True)
        pipeline = _pipeline(rad, model, tokenizer)
        assert rad.scoring_path == "cached"

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert isinstance(out, torch.Tensor)

    def test_cached_matches_stateless_on_granite(self, tmp_path):
        """The Granite cached forward reproduces the stateless shared-vocab scores (TestCachedEquivalence)."""
        rm_path = _granitemoehybrid_causal_lm_path(tmp_path)
        rm, rm_tok = load_sequence_classifier(rm_path, device="cpu", hf_model_kwargs={"num_labels": 1})
        cached = CachedRewardModelValue(rm, rm_tok, score_index=0, score_transform="sigmoid")
        stateless = RewardModelValue(rm, rm_tok, score_index=0, score_transform="sigmoid", shared_vocab=True)
        candidates = torch.tensor([[6, 7, 8, 9, 3]])
        prefix = torch.tensor([[0, 4]])
        for extra in (5, 6, 7):
            prefix = torch.cat([prefix, torch.tensor([[extra]])], dim=1)
            ctx = StepContext(
                prefix_ids=prefix, candidate_ids=candidates,
                lm_tokenizer=wordlevel_tokenizer(), attention_mask=torch.ones_like(prefix),
            )
            torch.testing.assert_close(cached.score(ctx), stateless.score(ctx), atol=1e-4, rtol=1e-4)

    def test_sequence_classifier_class_resolution(self, tmp_path, monkeypatch):
        """Granite families resolve to the toolkit heads; covered configs resolve to the auto class."""
        from transformers import AutoModelForSequenceClassification

        from steerability.algorithms.output_control.common import granite_heads

        granite = granite_heads.sequence_classifier_class(_granite_causal_lm_path(tmp_path / "g"))
        hybrid = granite_heads.sequence_classifier_class(_granitemoehybrid_causal_lm_path(tmp_path / "h"))
        llama = granite_heads.sequence_classifier_class(_decoder_classifier(tmp_path / "l"))
        assert granite is GraniteForSequenceClassification
        assert hybrid is GraniteMoeHybridForSequenceClassification
        assert llama is AutoModelForSequenceClassification

        # a head shipped by transformers wins over the toolkit head
        class _Covers:
            def __contains__(self, item):
                return True

        monkeypatch.setattr(granite_heads, "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING", _Covers())
        covered = granite_heads.sequence_classifier_class(_granite_causal_lm_path(tmp_path / "g2"))
        assert covered is AutoModelForSequenceClassification


# value trace
class TestValueTrace:
    """A caller-owned `value_trace` list receives one record per scored step."""

    def test_schema_declares_value_trace(self):
        names = [entry["name"] for entry in RAD.RUNTIME_KWARGS_SCHEMA]
        assert "value_trace" in names

    def test_trace_records_each_step(self, tmp_path):
        torch.manual_seed(0)
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        top_k = 5

        rad = RAD(reward_model_id=rm_path, beta=5.0, top_k=top_k, score_transform="sigmoid")
        pipeline = _pipeline(rad, model, tokenizer)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        trace: list = []
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=4, do_sample=False, eos_token_id=None,
            runtime_kwargs={"value_trace": trace},
        )

        assert len(trace) == 4
        for step, record in enumerate(trace):
            assert isinstance(record, ValueStepRecord)
            assert record.candidate_ids.shape == (1, top_k)
            assert record.candidate_scores.shape == (1, top_k)
            assert record.values.shape == (1, top_k)
            assert record.normalized.shape == (1, top_k)
            assert torch.all(record.normalized >= 0) and torch.all(record.normalized <= 1)
            chosen = out[0, step].item()
            assert chosen in record.candidate_ids[0].tolist()


# scoring path
class TestScoringPath:
    def test_none_before_steer(self):
        rad = RAD(reward_model_id="x", beta=1.0)
        assert rad.scoring_path is None

    def test_cached_for_decoder_classifier(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=rm_path, beta=1.0, efficient=True)
        _pipeline(rad, model, wordlevel_tokenizer())
        assert rad.scoring_path == "cached"

    def test_stateless_when_not_efficient(self, tmp_path):
        rm_path = _decoder_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=rm_path, beta=1.0, efficient=False)
        _pipeline(rad, model, wordlevel_tokenizer())
        assert rad.scoring_path == "stateless"

    def test_text_for_mismatched_vocabulary(self, tmp_path):
        """A classifier whose tokenizer differs from the LM tokenizer uses the text path."""
        cfg = LlamaConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=VOCAB,
            num_labels=2, pad_token_id=2,
        )
        LlamaForSequenceClassification(cfg).eval().save_pretrained(str(tmp_path))
        reasoning_tag_tokenizer().save_pretrained(str(tmp_path))  # a distinct vocabulary
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(reward_model_id=str(tmp_path), beta=1.0, efficient=True)
        _pipeline(rad, model, wordlevel_tokenizer())
        assert rad.scoring_path == "text"
