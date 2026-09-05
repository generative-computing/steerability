"""Validation of the TRL wrappers' `training_args` composition.

`training_args` is forwarded verbatim to the installed TRL config classes, so every key a wrapper
emits must be a field of the config it targets, and the convenience fields must lose to an explicit
`training_args` entry of the same name. The audit test is the regression guard against argument-name
drift between the toolkit and TRL: it enumerates every emitted key against the live config classes,
so a future TRL release that renames or removes a field fails here.

Model-free and CPU-only; no training runs.
"""
from __future__ import annotations

import dataclasses

import pytest
import transformers
import trl
from datasets import Dataset
from trl import DPOConfig, GRPOConfig, SFTConfig

from steerability.algorithms.core.identity import config_descriptor_from_controls, config_digest
from steerability.algorithms.structural_control.wrappers.trl.apotrainer.args import APOArgs
from steerability.algorithms.structural_control.wrappers.trl.args import TRLArgs
from steerability.algorithms.structural_control.wrappers.trl.base_mixin import resolve_config_kwargs
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import DPO
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer.args import DPOArgs
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer.args import GRPOArgs
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.args import PPOArgs
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.args import SFTArgs
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT

try:
    from trl.experimental.ppo import PPOConfig

    _PPO_CONFIG_AVAILABLE = True
except ImportError:
    PPOConfig = None
    _PPO_CONFIG_AVAILABLE = False


def _reward_stub(prompts, completions, **kwargs):
    return [0.0] * len(completions)


class TestResolveConfigKwargs:
    def test_unknown_key_raises_with_generic_message(self):
        # T1: an unknown key is rejected; the message names the config class, the installed TRL
        # version, and the offending key, and carries no retired-name replacement text
        with pytest.raises(ValueError) as excinfo:
            resolve_config_kwargs(DPOConfig, {"rpo_alpha": 1.0, "beta": 0.1})
        message = str(excinfo.value)
        assert "rpo_alpha" in message
        assert "DPOConfig" in message
        assert trl.__version__ in message

    def test_known_keys_pass_through_and_drop_none(self):
        # T2: known keys are kept in order and None values are dropped
        assert resolve_config_kwargs(DPOConfig, {"beta": 0.2, "seed": None}) == {"beta": 0.2}


def _audit_cases():
    cases = [
        (SFTArgs, SFTConfig, {}),
        (DPOArgs, DPOConfig, {}),
        (APOArgs, DPOConfig, {}),
        (
            GRPOArgs,
            GRPOConfig,
            {
                "reward_funcs": [_reward_stub],
                "num_generations": 2,
                "per_device_train_batch_size": 2,
            },
        ),
    ]
    ppo_case = (PPOArgs, PPOConfig, {"reward_model_name_or_path": "x"})
    cases.append(
        pytest.param(
            *ppo_case,
            marks=pytest.mark.skipif(not _PPO_CONFIG_AVAILABLE, reason="trl.experimental.ppo not importable"),
        )
    )
    return cases


class TestArgsAudit:
    @pytest.mark.parametrize("args_cls, config_cls, minimal_kwargs", _audit_cases())
    def test_every_emitted_key_is_a_config_field(self, args_cls, config_cls, minimal_kwargs):
        # T3: the regression guard for argument-name drift; every emitted key must be a live field
        args = args_cls(**minimal_kwargs)
        allowed = {field.name for field in dataclasses.fields(config_cls)}
        unknown = sorted(key for key in args.training_args if key not in allowed)
        assert not unknown, f"{args_cls.__name__} emits keys not on {config_cls.__name__}: {unknown}"

    def test_base_training_args_are_transformers_fields(self):
        allowed = {field.name for field in dataclasses.fields(transformers.TrainingArguments)}
        unknown = sorted(key for key in TRLArgs().training_args if key not in allowed)
        assert not unknown, f"TRLArgs emits keys not on TrainingArguments: {unknown}"


class TestDPOLossFields:
    def test_list_loss_and_weights_carry_through(self):
        # T4
        args = DPOArgs(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.5])
        assert args.training_args["loss_type"] == ["sigmoid", "sft"]
        assert args.training_args["loss_weights"] == [1.0, 0.5]

    def test_mismatched_weights_length_raises(self):
        with pytest.raises(ValueError, match="one weight per loss_type"):
            DPOArgs(loss_type=["sigmoid", "sft"], loss_weights=[1.0])

    def test_string_loss_passes_through_unchanged(self):
        assert DPOArgs(loss_type="ipo").training_args["loss_type"] == "ipo"


class TestPrecedence:
    def test_training_args_beta_overrides_field(self):
        # T5
        assert DPOArgs(beta=0.1, training_args={"beta": 0.7}).training_args["beta"] == 0.7

    def test_training_args_loss_type_overrides_field(self):
        args = DPOArgs(training_args={"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0]})
        assert args.training_args["loss_type"] == ["sigmoid", "sft"]

    def test_sft_max_length_override(self):
        assert SFTArgs(max_length=64, training_args={"max_length": 32}).training_args["max_length"] == 32


class TestRemovedAndRenamedFields:
    def test_dpo_rejects_max_prompt_length(self):
        # T6
        with pytest.raises(TypeError):
            DPOArgs(max_prompt_length=1)

    def test_sft_max_length_field(self):
        assert SFTArgs(max_length=64).training_args["max_length"] == 64

    def test_apo_list_loss_allowed(self):
        assert APOArgs(loss_type=["apo_zero", "sft"], loss_weights=[1.0, 1.0])

    def test_apo_rejects_non_apo_loss(self):
        with pytest.raises(ValueError, match="apo_zero.*apo_down"):
            APOArgs(loss_type="sigmoid")


class TestGRPOArgsDropsMaxPromptLength:
    def test_no_max_prompt_length_field(self):
        with pytest.raises(TypeError):
            GRPOArgs(reward_funcs=[_reward_stub], max_prompt_length=32, num_generations=2, per_device_train_batch_size=2)

    def test_not_emitted_in_training_args(self):
        args = GRPOArgs(reward_funcs=[_reward_stub], num_generations=2, per_device_train_batch_size=2)
        assert "max_prompt_length" not in args.training_args


class TestConfigIdentity:
    def test_loss_type_form_changes_digest_and_encodes(self):
        # T7: the anchored recipe is a distinct configuration from plain sigmoid, and both encode
        sigmoid = DPO(train_dataset=None, loss_type="sigmoid")
        anchored = DPO(train_dataset=None, loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0])
        sigmoid_digest = config_digest(config_descriptor_from_controls([sigmoid]))
        anchored_digest = config_digest(config_descriptor_from_controls([anchored]))
        assert sigmoid_digest != anchored_digest


class TestResumeFromCheckpoint:
    def test_field_lands_in_training_args(self):
        assert SFTArgs(resume_from_checkpoint="ckpt").training_args["resume_from_checkpoint"] == "ckpt"

    def test_training_args_entry_overrides_field(self):
        args = SFTArgs(resume_from_checkpoint="a", training_args={"resume_from_checkpoint": "b"})
        assert args.training_args["resume_from_checkpoint"] == "b"

    def test_unset_is_dropped_from_config_kwargs(self):
        assert "resume_from_checkpoint" not in resolve_config_kwargs(SFTConfig, SFTArgs().training_args)

    def test_fit_identity_ignores_resume_path(self):
        dataset = Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}])
        plain = SFT(train_dataset=dataset).fit_identity()
        resumed = SFT(train_dataset=dataset, resume_from_checkpoint="ckpt").fit_identity()
        assert plain == resumed
