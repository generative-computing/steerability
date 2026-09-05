"""Behavior tests for BestOfN (output multiplicity design, P4).

Hub-free, using the tiny-model fixtures and the lazy-init `_pipeline` pattern.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.output_control.common.scorers.majority_vote import MajorityVoteScorer
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestBestOfNConfig:
    def test_preset_maps_onto_search_driver(self):
        bon = BestOfN(n=5, scorer=lambda p, c, params: [0.0] * len(c))
        assert bon.num_candidates == 5
        assert bon.keep_k == 1
        assert bon.max_iterations == 1
        assert bon.propose_mode == "sample"

    def test_is_decoding_driver(self):
        from steerability.algorithms.output_control.base import DecodingDriver
        assert isinstance(BestOfN(n=2, scorer=lambda p, c, params: [0.0] * len(c)), DecodingDriver)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            BestOfN(n=0, scorer=lambda p, c, params: [])
        with pytest.raises(TypeError):
            BestOfN(n=4, scorer="not callable")


class TestBestOfNBehavior:
    def test_returns_argmax_of_scored_samples(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        seen = {}

        # scorer rewards the continuation containing "mat"; record what it saw
        def scorer(prompt, continuations, params):
            seen["continuations"] = continuations
            return [float(c.count("mat")) for c in continuations]

        bon = BestOfN(n=6, scorer=scorer)
        pipeline, model, tokenizer = _pipeline([bon], model=model, tokenizer=tokenizer)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=5, do_sample=True, eos_token_id=None)
        assert out.ndim == 2
        # the scorer saw n candidates
        assert len(seen["continuations"]) == 6

    def test_single_iteration_one_scorer_call(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        calls = []

        def scorer(prompt, continuations, params):
            calls.append(len(continuations))
            return [float(i) for i in range(len(continuations))]

        bon = BestOfN(n=4, scorer=scorer)
        pipeline, model, tokenizer = _pipeline([bon], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=True, eos_token_id=None)
        # one iteration -> exactly one scorer call over n candidates
        assert calls == [4]

    def test_step_level_control_steers_every_sample(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        class _ForceToken(OutputControl):
            Args = None

            def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
                def _force(prefix_ids, scores):
                    out = torch.full_like(scores, float("-inf"))
                    out[:, 7] = 0.0
                    return out
                return [_force]

        bon = BestOfN(n=3, scorer=lambda p, c, params: [0.0] * len(c))
        pipeline, model, tokenizer = _pipeline([_ForceToken(), bon], model=model, tokenizer=tokenizer)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=3, do_sample=True, return_full_sequence=True, eos_token_id=None,
        )
        continuation = out[:, prompt.size(1):]
        assert torch.all(continuation == 7)

    def test_batch_gt_one_raises(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        bon = BestOfN(n=2, scorer=lambda p, c, params: [0.0] * len(c))
        pipeline, model, tokenizer = _pipeline([bon], model=model, tokenizer=tokenizer)
        prompts = tokenizer(["the cat", "the dog"], return_tensors="pt", padding=True).input_ids
        with pytest.raises(NotImplementedError):
            pipeline.generate(input_ids=prompts, max_new_tokens=3, do_sample=True, eos_token_id=None)


class TestSelfConsistencyRecipe:
    def test_majority_vote_scorer_pairs_with_best_of_n(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        bon = BestOfN(n=4, scorer=MajorityVoteScorer())
        pipeline, model, tokenizer = _pipeline([bon], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=True, eos_token_id=None)
        assert out.ndim == 2
