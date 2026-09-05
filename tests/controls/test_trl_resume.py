"""The resume_from_checkpoint path reaches trainer.train() for SFT/DPO/GRPO and is rejected by PPO.

Model-free and CPU-only: the trainer class in each mixin module is replaced with a recording stub
that captures the resume_from_checkpoint value passed to train(), the model is a plain
torch.nn.Linear, and the tokenizer is the conftest mock. No training runs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import DPO
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import base_mixin as dpo_mixin
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer import GRPO
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer import base_mixin as grpo_mixin
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer import PPO
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer import SFT
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer import base_mixin as sft_mixin


class _RecordingTrainer:
    calls: list = []

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.accelerator = SimpleNamespace(unwrap_model=lambda m: m)

    def train(self, resume_from_checkpoint=None, **kwargs):
        type(self).calls.append(resume_from_checkpoint)

    def save_model(self, output_dir):
        pass


@pytest.fixture
def recorder(monkeypatch):
    """A recording trainer with a cleared call log, installed into every TRL mixin module."""
    _RecordingTrainer.calls = []
    monkeypatch.setattr(sft_mixin, "SFTTrainer", _RecordingTrainer)
    monkeypatch.setattr(dpo_mixin, "DPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(grpo_mixin, "GRPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(dpo_mixin, "standardize_preference_dataset", lambda dataset, **kwargs: dataset)
    return _RecordingTrainer


def _model():
    return torch.nn.Linear(2, 2)


def _reward_stub(prompts, completions, **kwargs):
    return [0.0] * len(completions)


def test_sft_passes_resume_path(recorder, mock_tokenizer, tmp_path):
    control = SFT(
        train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]),
        resume_from_checkpoint="ckpt",
        output_dir=str(tmp_path),
        load_best_model_at_end=False,
        training_args={"use_cpu": True},
    )
    control.steer(_model(), tokenizer=mock_tokenizer)
    assert recorder.calls == ["ckpt"]


def test_dpo_passes_resume_path(recorder, mock_tokenizer, tmp_path):
    control = DPO(
        train_dataset=Dataset.from_list(
            [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        ),
        resume_from_checkpoint="ckpt",
        output_dir=str(tmp_path),
        load_best_model_at_end=False,
        training_args={"use_cpu": True},
    )
    control.steer(_model(), tokenizer=mock_tokenizer)
    assert recorder.calls == ["ckpt"]


def test_grpo_passes_resume_path(recorder, mock_tokenizer, tmp_path):
    control = GRPO(
        train_dataset=Dataset.from_list([{"prompt": "p"}]),
        reward_funcs=[_reward_stub],
        num_generations=2,
        per_device_train_batch_size=2,
        resume_from_checkpoint="ckpt",
        output_dir=str(tmp_path),
        training_args={"use_cpu": True},
    )
    control.steer(_model(), tokenizer=mock_tokenizer)
    assert recorder.calls == ["ckpt"]


def test_unset_resume_records_none(recorder, mock_tokenizer, tmp_path):
    control = SFT(
        train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]),
        output_dir=str(tmp_path),
        load_best_model_at_end=False,
        training_args={"use_cpu": True},
    )
    control.steer(_model(), tokenizer=mock_tokenizer)
    assert recorder.calls == [None]


def test_ppo_rejects_resume_before_trainer(mock_tokenizer, tmp_path):
    control = PPO(
        train_dataset=Dataset.from_list([{"prompt": "p"}]),
        reward_model_name_or_path="x",
        resume_from_checkpoint="ckpt",
        output_dir=str(tmp_path),
        training_args={"use_cpu": True},
    )
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        control.steer(_model(), tokenizer=mock_tokenizer)
