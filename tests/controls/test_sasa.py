"""Tests for the SASA output control: args validation, candidate policies, the paired-fit path,
data normalization, the value trace, and the frozen-form round trip.

Hub-free, using the tiny Llama model and the WordLevel tokenizer from `tests/utils/tiny_models.py`.
"""
import pytest
import torch

from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.candidates import select_candidates
from steerability.algorithms.output_control.sasa.args import SASAArgs
from steerability.algorithms.output_control.sasa.control import SASA
from steerability.spipe import SPipe
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100

# a minimal chat template over the WordLevel vocabulary; the rendered text stays inside the vocab so
# it tokenizes cleanly (same shape as tests/controls/test_residual_norm_calibration.py)
_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}sat {% endif %}"
)


def _chat_tokenizer():
    tok = wordlevel_tokenizer()
    tok.chat_template = _CHAT_TEMPLATE
    return tok


class TestArgsValidation:
    def test_beta_must_be_non_negative(self):
        with pytest.raises(ValueError, match="beta"):
            SASAArgs(beta=-1.0)

    def test_top_p_policy_requires_valid_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            SASAArgs(candidate_policy="top_p")
        with pytest.raises(ValueError, match="top_p"):
            SASAArgs(candidate_policy="top_p", top_p=1.5)

    def test_top_p_policy_rejects_top_k(self):
        with pytest.raises(ValueError, match="does not use top_k"):
            SASAArgs(candidate_policy="top_p", top_p=0.9, top_k=10)

    def test_top_k_policy_requires_valid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            SASAArgs(candidate_policy="top_k")
        with pytest.raises(ValueError, match="top_k"):
            SASAArgs(candidate_policy="top_k", top_k=0)

    def test_top_k_policy_rejects_top_p(self):
        with pytest.raises(ValueError, match="does not use top_p"):
            SASAArgs(candidate_policy="top_k", top_k=10, top_p=0.9)

    def test_surviving_policy_rejects_top_p_and_top_k(self):
        with pytest.raises(ValueError, match="does not use top_p or top_k"):
            SASAArgs(candidate_policy="surviving", top_p=0.9)
        with pytest.raises(ValueError, match="does not use top_p or top_k"):
            SASAArgs(candidate_policy="surviving", top_k=10)

    def test_chat_completion_requires_paired_data(self):
        with pytest.raises(ValueError, match="chat_completion"):
            SASAArgs(prompt_format="chat_completion", gen_wv_data={"pos": ["a"], "neg": ["b"]})
        with pytest.raises(ValueError, match="chat_completion"):
            SASAArgs(
                prompt_format="chat_completion",
                gen_wv_data=LabeledExamples(positives=["a"], negatives=["b"]),
            )

    def test_chat_completion_accepts_paired_data(self):
        SASAArgs(
            prompt_format="chat_completion",
            gen_wv_data={"pos": ["a"], "neg": ["b"], "prompts": ["q"]},
        )
        SASAArgs(
            prompt_format="chat_completion",
            gen_wv_data=ContrastivePairs(positives=["a"], negatives=["b"], prompts=["q"]),
        )

    def test_missing_data_raises_at_steer(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        sasa = SASA(beta=1.0)  # neither gen_wv_data nor wv_path
        pipeline = SteeringPipeline(controls=[sasa], model=model, tokenizer=tokenizer)
        with pytest.raises(ValueError, match="gen_wv_data.*wv_path"):
            pipeline.steer()


class TestDictNormalization:
    def _fit(self, gen_wv_data, prompt_format="raw", tokenizer=None):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = tokenizer or wordlevel_tokenizer()
        sasa = SASA(beta=1.0, gen_wv_data=gen_wv_data, prompt_format=prompt_format)
        sasa.model = model
        sasa.tokenizer = tokenizer
        return sasa

    def test_pos_neg_dict_becomes_labeled_examples(self):
        sasa = self._fit({"pos": ["the cat sat", "the dog ran"], "neg": ["mat on fast", "span attention"]})
        resolved = sasa._resolve_fit_data()
        assert isinstance(resolved, LabeledExamples)
        assert list(resolved.positives) == ["the cat sat", "the dog ran"]

    def test_prompts_dict_becomes_contrastive_pairs(self):
        sasa = self._fit(
            {"pos": ["the cat sat"], "neg": ["mat on fast"], "prompts": ["the dog"]},
            prompt_format="chat_completion",
            tokenizer=_chat_tokenizer(),
        )
        resolved = sasa._resolve_fit_data()
        assert isinstance(resolved, ContrastivePairs)
        assert list(resolved.prompts) == ["the dog"]

    def test_gen_wv_length_truncates_each_class(self):
        sasa = self._fit({"pos": ["a", "b", "c"], "neg": ["d", "e", "f"]})
        sasa.gen_wv_length = 2
        resolved = sasa._resolve_fit_data()
        assert len(resolved.positives) == 2 and len(resolved.negatives) == 2


class TestTopPCandidatePolicy:
    def test_trace_candidate_count_matches_nucleus_and_respects_max_candidates(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        sasa = SASA(
            beta=1.0,
            gen_wv_data={
                "pos": ["the cat sat", "the dog ran", "the cat ran on"],
                "neg": ["mat on fast", "span attention", "fast mat sat"],
            },
            candidate_policy="top_p",
            top_p=0.9,
            max_candidates=8,
        )
        pipeline = SteeringPipeline(controls=[sasa], model=model, tokenizer=tokenizer)
        pipeline.steer()

        trace: list = []
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(
            input_ids=prompt,
            max_new_tokens=6,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            eos_token_id=None,
            runtime_kwargs={"value_trace": trace},
        )
        assert len(trace) == 6

        for record in trace:
            k = record.candidate_ids.size(1)
            assert k <= 8  # clamped to max_candidates
            # the recorded pre-shift scores are the raw (pre-warper) logits at that step, so the
            # candidate set is the nucleus of the raw logits clamped to max_candidates
            raw_scores = record.candidate_scores
            full = torch.full((1, VOCAB), float("-inf"))
            full.scatter_(1, record.candidate_ids, raw_scores)
            nucleus_ids, _ = select_candidates(full, "surviving")
            assert nucleus_ids.size(1) == k

    def test_non_candidate_logits_unchanged(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        sasa = SASA(beta=5.0, wv_path=None, candidate_policy="top_p", top_p=0.5)
        from steerability.algorithms.core.internals.probes.probe import Probe

        sasa.model = model
        sasa.tokenizer = tokenizer
        sasa.probe = Probe(
            model_type="test", location="layer_output", pooling="last",
            layer_ids=[1], weights={1: torch.randn(16)}, bias=0.0,
        )

        prefix = torch.tensor([[0, 3, 4]])
        attention_mask = torch.ones_like(prefix)
        scores = torch.randn(1, VOCAB)
        proc = sasa.get_logits_processors(prefix, {}, attention_mask=attention_mask)[0]
        cand_ids, _ = select_candidates(scores, "top_p", p=0.5)

        out = proc(prefix, scores.clone())
        non_candidate = torch.ones(VOCAB, dtype=torch.bool)
        non_candidate[cand_ids.reshape(-1)] = False
        assert torch.allclose(out[0, non_candidate], scores[0, non_candidate])


class TestChatCompletionFit:
    def test_fit_and_generate_on_chat_template(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = _chat_tokenizer()
        sasa = SASA(
            beta=2.0,
            prompt_format="chat_completion",
            candidate_policy="top_p",
            top_p=0.9,
            gen_wv_data=ContrastivePairs(
                positives=["the cat sat", "the dog ran", "the cat ran on"],
                negatives=["mat on fast", "span attention", "fast mat sat"],
                prompts=["the dog", "the mat", "the cat"],
            ),
        )
        pipeline = SteeringPipeline(controls=[sasa], model=model, tokenizer=tokenizer)
        pipeline.steer()

        assert sasa.probe.layer_ids == [1]  # final decoder layer of a 2-layer model
        assert sasa.probe.location == "layer_output" and sasa.probe.pooling == "last"

        out = pipeline.generate(
            messages=[{"role": "user", "content": "the cat"}],
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
            return_output=True,
        )
        assert out.output_ids.size(1) == 4


class TestFrozenFormRoundTrip:
    def test_new_fields_survive_freeze(self, tmp_path):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        sasa = SASA(
            beta=3.0,
            gen_wv_data={
                "pos": ["the cat sat", "the dog ran", "the cat ran on"],
                "neg": ["mat on fast", "span attention", "fast mat sat"],
            },
            candidate_policy="top_p",
            top_p=0.85,
            max_candidates=16,
        )
        pipeline = SteeringPipeline(controls=[sasa], model=model, tokenizer=tokenizer,
                                    model_name_or_path="tiny")
        pipeline.steer()
        saved = pipeline.to_spipe().save(tmp_path / "sasa.spipe")

        rebuilt = SPipe.load(saved).pipeline()
        frozen = rebuilt.output_controls[0]
        assert frozen.candidate_policy == "top_p"
        assert frozen.top_p == 0.85
        assert frozen.max_candidates == 16
        assert frozen.beta == 3.0

        rebuilt.model, rebuilt.tokenizer = model, tokenizer
        rebuilt.steer()  # loads the frozen probe via wv_path, no refit
        assert torch.allclose(frozen.probe.weights[1], sasa.probe.weights[1])
