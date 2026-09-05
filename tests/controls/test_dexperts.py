"""Behavior / math tests for DExperts (output multiplicity design, P4).

Hub-free: auxiliary expert / anti-expert models are second `tiny_llama` instances saved to tmp_path
(shared vocab by construction).
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from steerability.algorithms.output_control.dexperts.control import DExperts
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _save_aux(tmp_path, name):
    """Save a tiny causal LM + wordlevel tokenizer to a subdir; return its path."""
    path = tmp_path / name
    model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    tokenizer = wordlevel_tokenizer()
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return str(path)


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestDExpertsConfig:
    def test_requires_both_models(self):
        with pytest.raises(ValueError):
            DExperts(expert_name_or_path="x")
        with pytest.raises(ValueError):
            DExperts(anti_expert_name_or_path="y")

    def test_is_step_level_control_not_driver(self):
        from steerability.algorithms.output_control.base import DecodingDriver, OutputControl
        dex = DExperts(expert_name_or_path="x", anti_expert_name_or_path="y", alpha=0.5)
        assert isinstance(dex, OutputControl)
        assert not isinstance(dex, DecodingDriver)

    def test_include_in_scoring_default_true(self):
        assert DExperts.include_in_scoring is True

    def test_unsteered_raises(self):
        dex = DExperts(expert_name_or_path="x", anti_expert_name_or_path="y")
        with pytest.raises(RuntimeError, match="steer"):
            dex.get_logits_processors(torch.tensor([[0, 3, 4]]), {})


class TestDExpertsMath:
    def test_mixture_matches_reference(self, tmp_path):
        expert_path = _save_aux(tmp_path, "expert")
        anti_path = _save_aux(tmp_path, "anti")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        dex = DExperts(expert_name_or_path=expert_path, anti_expert_name_or_path=anti_path, alpha=2.0)
        pipeline, model, tokenizer = _pipeline([dex], model=model, tokenizer=tokenizer)

        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        proc = dex.get_logits_processors(prefix, {})[0]
        assert isinstance(proc, ContrastiveMixtureProcessor)
        out = proc(prefix, scores.clone())

        # reference: log p_base + alpha * log p_expert - alpha * log p_anti_expert
        base_lp = torch.log_softmax(scores, dim=-1)
        expert_lp = dex._expert_source.logprobs(prefix)
        anti_lp = dex._anti_expert_source.logprobs(prefix)
        expected = base_lp + 2.0 * expert_lp - 2.0 * anti_lp
        assert torch.allclose(out, expected, atol=1e-4)

    def test_fresh_processor_per_call(self, tmp_path):
        expert_path = _save_aux(tmp_path, "expert")
        anti_path = _save_aux(tmp_path, "anti")
        dex = DExperts(expert_name_or_path=expert_path, anti_expert_name_or_path=anti_path, alpha=1.0)
        _pipeline([dex], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB))
        p1 = dex.get_logits_processors(torch.tensor([[0, 3]]), {})[0]
        p2 = dex.get_logits_processors(torch.tensor([[0, 3]]), {})[0]
        assert p1 is not p2
        # but the loaded sources persist across calls
        assert p1.sources[0][0] is p2.sources[0][0]


class TestVocabGuardrail:
    def test_vocab_mismatch_raises(self, tmp_path):
        # aux model with a different vocab than the base model -> clear error, no silent mapping
        from steerability.algorithms.output_control.common.logit_sources import AuxModelSource

        path = tmp_path / "mismatched"
        tiny_llama(num_layers=2, hidden=16, heads=2, vocab=64).save_pretrained(str(path))
        wordlevel_tokenizer().save_pretrained(str(path))

        base_model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        source = AuxModelSource(str(path), base_tokenizer=wordlevel_tokenizer())
        with pytest.raises(ValueError, match="vocab mismatch"):
            source.prepare(model=base_model, tokenizer=wordlevel_tokenizer())


class TestDExpertsEndToEnd:
    def test_generate(self, tmp_path):
        expert_path = _save_aux(tmp_path, "expert")
        anti_path = _save_aux(tmp_path, "anti")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        dex = DExperts(expert_name_or_path=expert_path, anti_expert_name_or_path=anti_path, alpha=1.0)
        pipeline, model, tokenizer = _pipeline([dex], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=4, do_sample=False, eos_token_id=None)
        assert out.ndim == 2
        assert out.size(1) == 4

    def test_cleanup_releases_sources(self, tmp_path):
        expert_path = _save_aux(tmp_path, "expert")
        anti_path = _save_aux(tmp_path, "anti")
        dex = DExperts(expert_name_or_path=expert_path, anti_expert_name_or_path=anti_path, alpha=1.0)
        _pipeline([dex], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB))
        assert dex._expert_source is not None
        dex.cleanup()
        assert dex._expert_source is None
        assert dex._anti_expert_source is None
