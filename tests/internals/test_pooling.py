"""Tests for token, span, and position aggregation (`core/internals/pooling.py`).

Every helper must be attention-mask-driven: left-padded batches are the norm, and "last" always
means the last real token.
"""
import pytest
import torch

from steerability.algorithms.core.internals.pooling import (
    aggregate_condition_hidden,
    get_last_token_positions,
    masked_mean,
    pool_over_spans,
    select_at_positions,
    select_spans,
)


def _hidden(N, T, H):
    return torch.arange(N * T * H, dtype=torch.float32).reshape(N, T, H)


class TestMaskedMean:
    def test_ignores_pad_positions(self):
        hidden = _hidden(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
        poisoned = hidden.clone()
        poisoned[mask == 0] = 1e6
        expected = torch.stack([hidden[0, :2].mean(dim=0), hidden[1, 1:].mean(dim=0)])
        torch.testing.assert_close(masked_mean(poisoned, mask), expected)

    def test_none_mask_means_all_positions(self):
        hidden = _hidden(2, 4, 3)
        torch.testing.assert_close(masked_mean(hidden, None), hidden.mean(dim=1))


class TestAggregateConditionHidden:
    def test_mean_matches_masked_mean(self):
        hidden = _hidden(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 0], [0, 0, 1, 1]])
        torch.testing.assert_close(
            aggregate_condition_hidden(hidden, "mean", mask), masked_mean(hidden, mask)
        )

    def test_last_selects_last_real_token(self):
        hidden = _hidden(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])  # right- and left-padded rows
        out = aggregate_condition_hidden(hidden, "last", mask)
        expected = torch.stack([hidden[0, 2], hidden[1, 3]])
        torch.testing.assert_close(out, expected)

    def test_last_without_mask_uses_final_position(self):
        hidden = _hidden(2, 4, 3)
        torch.testing.assert_close(aggregate_condition_hidden(hidden, "last", None), hidden[:, -1, :])

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="comparison mode"):
            aggregate_condition_hidden(_hidden(1, 2, 3), "median", None)  # type: ignore[arg-type]

    def test_empty_row_raises(self):
        mask = torch.tensor([[1, 1], [0, 0]])
        with pytest.raises(ValueError, match="no real tokens"):
            aggregate_condition_hidden(_hidden(2, 2, 3), "last", mask)


class TestSelectSpans:
    def test_all_excludes_pads_on_both_sides(self):
        am = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
        enc = {"input_ids": torch.zeros(2, 5, dtype=torch.long), "attention_mask": am}
        assert select_spans(enc, None, "all") == [(0, 3), (2, 5)]

    def test_suffix_only_skips_prompt_length(self):
        am = torch.tensor([[1, 1, 1, 1, 0]])
        enc = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "attention_mask": am}
        prompt_enc = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        assert select_spans(enc, prompt_enc, "suffix-only") == [(2, 4)]

    def test_last_token_is_pad_side_agnostic(self):
        am = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
        enc = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": am}
        assert select_spans(enc, None, "last_token") == [(2, 3), (3, 4)]

    def test_no_mask_falls_back_to_full_width(self):
        enc = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        assert select_spans(enc, None, "all") == [(0, 4)]
        assert select_spans(enc, None, "last_token") == [(3, 4)]

    def test_unknown_accumulate_raises(self):
        enc = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        with pytest.raises(ValueError, match="select_spans does not support"):
            select_spans(enc, None, "bogus")


class TestPoolOverSpans:
    def test_mean_over_span(self):
        hidden = _hidden(1, 4, 2)
        torch.testing.assert_close(pool_over_spans(hidden, [(1, 3)]), hidden[:, 1:3, :].mean(dim=1))

    def test_degenerate_span_falls_back_to_last_token(self):
        hidden = _hidden(1, 4, 2)
        torch.testing.assert_close(pool_over_spans(hidden, [(3, 3)]), hidden[:, -1, :])


class TestPositionHelpers:
    def test_last_positions_left_and_right_padded(self):
        am = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        positions = get_last_token_positions(am, 4, 2)
        assert positions.tolist() == [1, 3]

    def test_last_positions_none_mask(self):
        positions = get_last_token_positions(None, 5, 3)
        assert positions.tolist() == [4, 4, 4]

    def test_select_at_positions_gathers_per_row(self):
        hidden = _hidden(2, 4, 3)
        positions = torch.tensor([1, 3])
        out = select_at_positions(hidden, positions)
        torch.testing.assert_close(out, torch.stack([hidden[0, 1], hidden[1, 3]]))
