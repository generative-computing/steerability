"""Behavior / math tests for ContrastiveDecoding (output multiplicity design, P4).

Hub-free: the amateur model is a second `tiny_llama` instance saved to tmp_path (shared vocab).
"""
import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from aisteer360.algorithms.output_control.contrastive_decoding.control import ContrastiveDecoding
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _save_aux(tmp_path, name):
    path = tmp_path / name
    tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB).save_pretrained(str(path))
    wordlevel_tokenizer().save_pretrained(str(path))
    return str(path)


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestConfig:
    def test_requires_amateur(self):
        with pytest.raises(ValueError):
            ContrastiveDecoding()

    def test_alpha_range_validated(self):
        with pytest.raises(ValueError):
            ContrastiveDecoding(amateur_name_or_path="x", alpha=1.5)
        with pytest.raises(ValueError):
            ContrastiveDecoding(amateur_name_or_path="x", alpha=-0.1)

    def test_is_step_level_control_not_driver(self):
        from aisteer360.algorithms.output_control.base import DecodingDriver, OutputControl
        cd = ContrastiveDecoding(amateur_name_or_path="x")
        assert isinstance(cd, OutputControl)
        assert not isinstance(cd, DecodingDriver)

    def test_unsteered_raises(self):
        cd = ContrastiveDecoding(amateur_name_or_path="x")
        with pytest.raises(RuntimeError, match="steer"):
            cd.get_logits_processors(torch.tensor([[0, 3, 4]]), {})


class TestMath:
    def test_expert_minus_amateur_with_alpha_mask(self, tmp_path):
        amateur_path = _save_aux(tmp_path, "amateur")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        cd = ContrastiveDecoding(amateur_name_or_path=amateur_path, alpha=0.1, base_weight=1.0, amateur_weight=1.0)
        pipeline, model, tokenizer = _pipeline([cd], model=model, tokenizer=tokenizer)

        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        proc = cd.get_logits_processors(prefix, {})[0]
        assert isinstance(proc, ContrastiveMixtureProcessor)
        out = proc(prefix, scores.clone())

        # reference: base_lp - amateur_lp, then plausibility mask keep = p_base >= alpha * max p_base
        base_lp = torch.log_softmax(scores, dim=-1)
        amateur_lp = cd._amateur_source.logprobs(prefix)
        mixed = base_lp - amateur_lp
        base_probs = base_lp.exp()
        threshold = 0.1 * base_probs.max(dim=-1, keepdim=True).values
        keep = base_probs >= threshold
        expected = mixed.masked_fill(~keep, float("-inf"))
        assert torch.allclose(out, expected, atol=1e-4, equal_nan=True)
        # some tokens should be masked out (alpha filter is active)
        assert torch.isinf(out).any()

    def test_alpha_mask_confines_output_to_plausible_set(self, tmp_path):
        amateur_path = _save_aux(tmp_path, "amateur")
        cd = ContrastiveDecoding(amateur_name_or_path=amateur_path, alpha=0.5)
        _pipeline([cd], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB))
        prefix = torch.tensor([[0, 3]])
        scores = torch.randn(1, VOCAB)
        proc = cd.get_logits_processors(prefix, {})[0]
        out = proc(prefix, scores.clone())
        base_probs = torch.log_softmax(scores, dim=-1).exp()
        keep = base_probs >= 0.5 * base_probs.max()
        # masked positions are exactly the implausible ones
        assert torch.all(torch.isinf(out[~keep]))
        assert torch.all(torch.isfinite(out[keep]))


class TestEndToEnd:
    def test_generate(self, tmp_path):
        amateur_path = _save_aux(tmp_path, "amateur")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        cd = ContrastiveDecoding(amateur_name_or_path=amateur_path, alpha=0.1)
        pipeline, model, tokenizer = _pipeline([cd], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=4, do_sample=False, eos_token_id=None)
        assert out.ndim == 2
        assert out.size(1) == 4

    def test_cleanup(self, tmp_path):
        amateur_path = _save_aux(tmp_path, "amateur")
        cd = ContrastiveDecoding(amateur_name_or_path=amateur_path)
        _pipeline([cd], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB))
        assert cd._amateur_source is not None
        cd.cleanup()
        assert cd._amateur_source is None
