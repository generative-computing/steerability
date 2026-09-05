"""Rollout-token accounting: `SteeredSession.generate` accumulation, the pipeline's per-row
split onto `Output.generated_tokens`, and the driverless path leaving it None.

Hub-free, using the tiny-model fixtures. Driver rollouts pass through the session wrapper, so the
count scales with the number of rollouts a driver issues rather than the text it returns.
"""
import torch

from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline, _split_generated_tokens
from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
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


def _zero_scorer(prompt, continuations, params):
    return [0.0] * len(continuations)


class TestSplitHelper:
    def test_even_split_sums_to_total(self):
        assert _split_generated_tokens(12, 4) == [3, 3, 3, 3]

    def test_remainder_lands_on_the_first_row(self):
        parts = _split_generated_tokens(10, 4)
        assert parts == [3, 3, 2, 2]
        assert sum(parts) == 10

    def test_none_total_yields_none_per_row(self):
        assert _split_generated_tokens(None, 3) == [None, None, None]

    def test_zero_rows_is_empty(self):
        assert _split_generated_tokens(7, 0) == []


class TestDriverPath:
    def test_generated_tokens_scales_with_rollouts(self):
        # a two-iteration search over 4 candidates keeping 1 rolls out far more than it returns.
        # a single input_ids prompt returns one Output per row, so index the first row
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        prompt = tokenizer("the cat", return_tensors="pt").input_ids

        search = SearchDecoding(
            scorer=_zero_scorer, segment_len=3, num_candidates=4, keep_k=1, max_iterations=2,
            propose_mode="sample",
        )
        search_pipeline, _, _ = _pipeline([search], model=model, tokenizer=tokenizer)
        search_out = search_pipeline.generate(
            input_ids=prompt, max_new_tokens=6, do_sample=True, eos_token_id=None, return_output=True,
        )[0]
        assert search_out.generated_tokens is not None
        returned = int((search_out.output_ids != tokenizer.pad_token_id).sum().item())
        # the count meters every rollout (4 candidates over up to 2 iterations), far more than the
        # single continuation the driver returns
        assert search_out.generated_tokens >= 4 * returned

    def test_driverless_path_leaves_generated_tokens_none(self):
        pipeline, _, tokenizer = _pipeline([])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=4, do_sample=False, eos_token_id=None, return_output=True,
        )[0]
        assert out.generated_tokens is None
