"""Tests for the `Output` generation record and per-row finish-reason inference.

Covers `infer_finish_reasons` per-row semantics (including the pad-equals-eos configuration), the
`Output` dataclass fields, `decode` round-tripping, and the module home (importable from `core`
and `core.output`).
"""
import pytest
import torch

from steerability.algorithms.core.output import Output, infer_finish_reasons
from tests.utils.tiny_models import wordlevel_tokenizer


class TestInferFinishReasons:
    """Per-row `"length"`/`"eos"`/None labeling from right-padded generated tokens."""

    def test_mixed_batch_eos_and_length(self):
        # row 0: 2 real tokens ending on eos (7), right-padded to width 3 with pad (2)
        # row 1: 3 real tokens, hits the cap
        new_tokens = torch.tensor([[5, 7, 2], [5, 6, 8]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 3}, eos_token_id=7, pad_token_id=2
        )
        assert reasons == ["eos", "length"]

    def test_pad_equals_eos_configuration(self):
        # pad == eos == 1; the first stripped trailing token was the genuine EOS
        new_tokens = torch.tensor([[5, 6, 1, 1], [5, 6, 7, 8]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 4}, eos_token_id=1, pad_token_id=1
        )
        assert reasons == ["eos", "length"]

    def test_pad_equals_eos_generate_rows(self):
        # rows as `generate` returns them under pad == eos == 1 and max_new_tokens=4: an early eos
        # followed by pads, an eos at the ceiling, and a row at the ceiling without eos
        new_tokens = torch.tensor([[5, 1, 1, 1], [5, 6, 7, 1], [5, 6, 7, 8]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 4}, eos_token_id=1, pad_token_id=1
        )
        assert reasons == ["eos", "eos", "length"]

    def test_pad_equals_eos_row_at_the_ceiling_padded_to_a_longer_row(self):
        # a decoding driver can return a row longer than the ceiling (row 0); row 1 reached the
        # ceiling and is padded to row 0's width, and row 2 stopped on eos before the ceiling
        new_tokens = torch.tensor([[5, 6, 7, 8, 9, 10], [5, 6, 7, 8, 1, 1], [5, 6, 1, 1, 1, 1]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 4}, eos_token_id=1, pad_token_id=1
        )
        assert reasons == ["length", "length", "eos"]

    def test_no_max_new_tokens_length_unreachable(self):
        new_tokens = torch.tensor([[5, 6, 7], [5, 6, 8]])
        reasons = infer_finish_reasons(new_tokens, {}, eos_token_id=7, pad_token_id=2)
        assert reasons == ["eos", None]
        assert "length" not in reasons

    def test_exact_boundary_is_length(self):
        # n == max_new_tokens -> "length"
        new_tokens = torch.tensor([[5, 6, 8]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 3}, eos_token_id=7, pad_token_id=2
        )
        assert reasons == ["length"]

    def test_zero_length_row_is_none(self):
        # entire row is padding -> length 0 -> None
        new_tokens = torch.tensor([[2, 2, 2]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 5}, eos_token_id=7, pad_token_id=2
        )
        assert reasons == [None]

    def test_eos_token_id_as_list(self):
        new_tokens = torch.tensor([[5, 6, 9, 2], [5, 6, 7, 2]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 4}, eos_token_id=[7, 9], pad_token_id=2
        )
        assert reasons == ["eos", "eos"]

    def test_eos_token_id_none(self):
        new_tokens = torch.tensor([[5, 6, 7, 2]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 10}, eos_token_id=None, pad_token_id=2
        )
        assert reasons == [None]

    def test_pad_token_id_none_no_stripping(self):
        # no pad id -> full width is the continuation; last token in eos set -> "eos"
        new_tokens = torch.tensor([[5, 6, 7]])
        reasons = infer_finish_reasons(
            new_tokens, {"max_new_tokens": 10}, eos_token_id=7, pad_token_id=None
        )
        assert reasons == ["eos"]


class TestOutputFields:
    """The dataclass carries its declared fields and decodes."""

    def test_fields_present(self):
        out = Output(
            output_ids=torch.tensor([[1, 2]]),
            adapted_input_ids=torch.tensor([[3, 4]]),
            finish_reasons=("length",),
        )
        assert out.finish_reasons == ("length",)
        assert out.adapted_input_ids is not None

    def test_one_reason_per_row_is_required(self):
        with pytest.raises(ValueError, match="finish_reasons has 1 entries for 2 rows"):
            Output(output_ids=torch.zeros((2, 3), dtype=torch.long), finish_reasons=("eos",))

    def test_output_ids_must_be_two_dimensional(self):
        with pytest.raises(ValueError, match=r"output_ids must be \[batch, seq\]"):
            Output(output_ids=torch.zeros((3,), dtype=torch.long), finish_reasons=("eos",))

    def test_reasons_are_known_values_or_none(self):
        with pytest.raises(ValueError, match="'halt'"):
            Output(output_ids=torch.zeros((1, 3), dtype=torch.long), finish_reasons=("halt",))
        Output(output_ids=torch.zeros((2, 3), dtype=torch.long), finish_reasons=(None, "stop"))

    def test_finish_reasons_is_a_required_keyword(self):
        with pytest.raises(TypeError):
            Output(output_ids=torch.zeros((1, 3), dtype=torch.long))

    def test_an_output_without_rows_constructs(self):
        out = Output(output_ids=torch.zeros((0, 0), dtype=torch.long), finish_reasons=())
        assert out.finish_reasons == ()

    def test_the_singular_field_is_gone(self):
        out = Output(output_ids=torch.zeros((1, 2), dtype=torch.long), finish_reasons=("eos",))
        assert not hasattr(out, "finish_reason")

    def test_decode_round_trips(self):
        tokenizer = wordlevel_tokenizer()
        ids = tokenizer(["the cat sat"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        out = Output(output_ids=ids, finish_reasons=(None,))
        decoded = out.decode(tokenizer)
        assert decoded == ["the cat sat"]


class TestModuleHome:
    """`Output` lives at `core.output` and is re-exported from `core`."""

    def test_importable_from_core(self):
        from steerability.algorithms.core import Output as CoreOutput

        assert CoreOutput is Output
