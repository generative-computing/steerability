"""TRL structural controls release the staged model before `steer()` returns.

On an engine backend the pipeline drops its own reference to the staged in-process model and
verifies that no control still holds it (`verify_stage_released`), so a control that serves its
result off an exported artifact must not retain the model past `steer()`. These tests pin that
contract directly on each TRL mixin, backend-free: run a no-train `steer()` on a tiny model, drop
every local reference, and assert the model is collectable (its weakref dies after `gc.collect()`,
the same signal the free protocol uses) and that no `nn.Module` remains on the control.

Construction happens even on the no-train path (the TRL config is built before the train guard), so
`training_args={"use_cpu": True, ...}` keeps config construction valid on CPU-only machines.
"""
from __future__ import annotations

import gc
import weakref

import pytest
import torch

from aisteer360.algorithms.structural_control.wrappers.trl.apotrainer import APO
from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer import DPO
from aisteer360.algorithms.structural_control.wrappers.trl.grpotrainer import GRPO
from aisteer360.algorithms.structural_control.wrappers.trl.ppotrainer import PPO
from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer import SFT
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

# CPU-only, no mixed precision, and no best-model reload (which requires a save strategy); keeps the
# TRL config constructible on the no-train path everywhere.
CPU_TRAINING_ARGS = {"use_cpu": True, "bf16": False, "fp16": False}


def _reward_stub(prompts, completions, **kwargs):
    return [float(len(c)) for c in completions]


@pytest.fixture(scope="session")
def tiny_reward_dir(tmp_path_factory):
    """A saved tiny sequence-classification model, loadable as PPO's reward/value reference.

    `vocab_size` matches the word-level tokenizer's length so PPO's reward/value vocab check passes,
    and `num_labels=1` matches the wrapper's `from_pretrained(num_labels=1)` so the scoring head is
    not reinitialized.
    """
    from transformers import LlamaConfig, LlamaForSequenceClassification

    path = tmp_path_factory.mktemp("tiny-reward")
    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=len(wordlevel_tokenizer()),
        num_labels=1,
        pad_token_id=2,
    )
    LlamaForSequenceClassification(config).save_pretrained(path)
    return str(path)


def _make_control(name, tiny_reward_dir):
    if name == "SFT":
        return SFT(train_dataset=None, load_best_model_at_end=False, training_args=dict(CPU_TRAINING_ARGS))
    if name == "DPO":
        return DPO(train_dataset=None, load_best_model_at_end=False, training_args=dict(CPU_TRAINING_ARGS))
    if name == "APO":
        return APO(train_dataset=None, load_best_model_at_end=False, training_args=dict(CPU_TRAINING_ARGS))
    if name == "GRPO":
        return GRPO(
            train_dataset=None,
            reward_funcs=[_reward_stub],
            num_generations=2,
            per_device_train_batch_size=2,
            training_args=dict(CPU_TRAINING_ARGS),
        )
    if name == "PPO":
        return PPO(
            train_dataset=None,
            reward_model_name_or_path=tiny_reward_dir,
            load_best_model_at_end=False,
            training_args=dict(CPU_TRAINING_ARGS),
        )
    raise AssertionError(f"unknown control {name}")


@pytest.mark.parametrize("name", ["SFT", "DPO", "APO", "GRPO", "PPO"])
def test_no_train_steer_releases_the_model(name, tiny_reward_dir):
    control = _make_control(name, tiny_reward_dir)
    tokenizer = wordlevel_tokenizer()

    model = tiny_llama()
    ref = weakref.ref(model)

    returned = control.steer(model, tokenizer=tokenizer)
    assert returned is not None

    # drop every local strong reference, then force collection (mirroring verify_stage_released)
    del model, returned
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # the model is collectable once every local reference is dropped, which is exactly what the
    # pipeline's free protocol checks after it releases its own reference (see verify_stage_released)
    assert ref() is None, f"{name} leaked the staged model (weakref still alive after gc)"
    # blanket: catches a retained trainer / ref model / value model, which is a different object than
    # the staged policy and so invisible to an identity check, but the same residency leak
    module_attrs = [key for key, value in vars(control).items() if isinstance(value, torch.nn.Module)]
    assert module_attrs == [], f"{name} retained nn.Module attribute(s): {module_attrs}"
