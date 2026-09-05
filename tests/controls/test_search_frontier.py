"""Finished-beam detection in `Frontier` over right-padded beams."""
import pytest
import torch

from steerability.algorithms.output_control.common.drivers.frontier import Frontier


def test_padded_beam_after_eos_is_finished():
    beams = torch.tensor([[1, 2, 5, 9, 0, 0], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=2, eos_token_id=9, input_length=2, max_new_tokens=None, pad_token_id=0)
    step = frontier.keep(beams, [0.9, 0.1])
    assert step.finished_flags == [True, False]


def test_pad_equals_eos_flags_padded_beam_finished():
    beams = torch.tensor([[1, 2, 5, 9, 9, 9], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=2, eos_token_id=9, input_length=2, max_new_tokens=None, pad_token_id=9)
    step = frontier.keep(beams, [0.9, 0.1])
    assert step.finished_flags == [True, False]


def test_budget_flag_fires_at_max_new_tokens():
    beams = torch.tensor([[1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=1, eos_token_id=9, input_length=2, max_new_tokens=4, pad_token_id=0)
    step = frontier.keep(beams, [0.5])
    assert step.finished_flags == [True]


def test_padding_does_not_count_toward_budget():
    beams = torch.tensor([[1, 2, 5, 6, 0, 0]])
    frontier = Frontier(keep_k=1, eos_token_id=9, input_length=2, max_new_tokens=4, pad_token_id=0)
    step = frontier.keep(beams, [0.5])
    assert step.finished_flags == [False]


def test_list_eos_accepts_every_member():
    beams = torch.tensor([[1, 2, 5, 7, 0, 0], [1, 2, 5, 9, 0, 0], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=3, eos_token_id=[7, 9], input_length=2, max_new_tokens=None, pad_token_id=0)
    step = frontier.keep(beams, [0.9, 0.8, 0.1])
    assert step.finished_flags == [True, True, False]


def test_best_ids_tracks_best_score_across_calls():
    frontier = Frontier(keep_k=1, eos_token_id=None, input_length=2, max_new_tokens=None, pad_token_id=0)
    first = torch.tensor([[1, 2, 5, 6]])
    second = torch.tensor([[1, 2, 7, 8]])
    frontier.keep(first, [0.9])
    frontier.keep(second, [0.4])
    assert frontier.best_score == pytest.approx(0.9)
    assert torch.equal(frontier.best_ids, first[0])
