"""Tests for `infer_attention_mask_from_ids` (WS2) and the `prepare_inputs` interior-eos path.

The utility replaces the token-identity `ids != pad_id` heuristic: interior occurrences of the pad id
(which equals eos for tokenizers without a dedicated pad token) must NOT be masked, because chat
templates legitimately place the eos mid-prompt. All tests run CPU-only and offline.
"""
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.utils.generation import PromptWarnings, prepare_inputs
from aisteer360.utils.tokenization import infer_attention_mask_from_ids
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PAD = 2  # <pad> id in wordlevel_tokenizer


class TestInferAttentionMask:
    def test_no_pad_row_all_ones(self):
        ids = torch.tensor([[3, 4, 5, 6]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [[1, 1, 1, 1]]

    def test_right_padding_masked(self):
        ids = torch.tensor([[3, 4, PAD, PAD]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [[1, 1, 0, 0]]

    def test_left_padding_masked(self):
        ids = torch.tensor([[PAD, PAD, 3, 4]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [[0, 0, 1, 1]]

    def test_interior_pad_id_preserved(self):
        # a pad-id token between real tokens is a real token (pad == eos mid-prompt), not padding
        ids = torch.tensor([[PAD, 3, PAD, 4, PAD]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [[0, 1, 1, 1, 0]]  # only the leading/trailing pad runs are zeroed

    def test_all_pad_row_all_ones(self):
        ids = torch.tensor([[PAD, PAD, PAD]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [[1, 1, 1]]  # degenerate; safest is all-attend

    def test_pad_token_id_none_all_ones(self):
        ids = torch.tensor([[3, 4, 5]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=None)
        assert mask.tolist() == [[1, 1, 1]]

    def test_batched_mixed_padding(self):
        ids = torch.tensor([
            [3, 4, PAD, PAD],   # right pad
            [PAD, 5, PAD, 6],   # left pad + interior pad preserved
            [PAD, PAD, PAD, PAD],  # all pad
        ])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]

    def test_dtype_and_device(self):
        ids = torch.tensor([[3, PAD, 4]])
        mask = infer_attention_mask_from_ids(ids, pad_token_id=PAD)
        assert mask.dtype == torch.long
        assert mask.device == ids.device


class TestPrepareInputsInteriorEos:
    """`prepare_inputs` must not mask an interior eos when pad == eos and no mask is supplied."""

    def _steered_pipeline(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        tokenizer = wordlevel_tokenizer()
        # force pad == eos, the hazardous configuration ensure_pad_token would create
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pipeline = SteeringPipeline(model=model, tokenizer=tokenizer)
        pipeline.steer()
        return pipeline, tokenizer

    def test_interior_eos_not_masked(self):
        pipeline, tokenizer = self._steered_pipeline()
        eos = tokenizer.eos_token_id
        # eos appears mid-sequence; with pad == eos, a token-identity mask would wrongly zero it
        ids = torch.tensor([[3, 4, eos, 5, 6]])
        steered_ids, mask = prepare_inputs(
            ids, None,
            input_controls=pipeline.input_controls,
            tokenizer=pipeline.tokenizer,
            device=pipeline.model.device,
            runtime_kwargs=None,
            warnings_state=PromptWarnings(),
        )
        # no interior zero: the eos at position 2 is kept as a real token
        assert mask.tolist() == [[1, 1, 1, 1, 1]]

    def test_trailing_eos_pad_still_masked(self):
        pipeline, tokenizer = self._steered_pipeline()
        eos = tokenizer.eos_token_id
        # a trailing run of eos/pad is genuine right-padding and should be masked
        ids = torch.tensor([[3, 4, 5, eos, eos]])
        _, mask = prepare_inputs(
            ids, None,
            input_controls=pipeline.input_controls,
            tokenizer=pipeline.tokenizer,
            device=pipeline.model.device,
            runtime_kwargs=None,
            warnings_state=PromptWarnings(),
        )
        assert mask.tolist() == [[1, 1, 1, 0, 0]]
