"""LoRA targets declared as module-name suffixes are scoped to the resolved decoder stack.

`lora_target_pattern` turns PEFT's list form (matched by suffix over every module of the model) into
one full-match regex over the decoder layers, so an adapter injected by the TRL wrappers attaches to
the text decoder and not to a multimodal wrapper's vision tower. The tests are CPU-only and hub-free:
models come from `tests/utils/tiny_models.py` and no training runs.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch.nn as nn
from datasets import Dataset

from steerability.algorithms.core.internals import lora_target_pattern
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import DPO
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import base_mixin as dpo_mixin
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer import GRPO
from steerability.algorithms.structural_control.wrappers.trl.grpotrainer import base_mixin as grpo_mixin
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer import PPO
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer import base_mixin as ppo_mixin
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer import SFT
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer import base_mixin as sft_mixin
from tests.utils.tiny_models import tiny_gemma3_conditional, tiny_gpt2, tiny_llama, tiny_lora

LLAMA_PATTERN = r"model\.layers\.\d+\.(?:.*\.)?(?:q_proj|v_proj)"
GPT2_PATTERN = r"transformer\.h\.\d+\.(?:.*\.)?(?:c_attn)"
GEMMA_PATTERN = r"model\.language_model\.layers\.\d+\.(?:.*\.)?(?:q_proj|v_proj)"


def matched(model: nn.Module, pattern: str) -> set[str]:
    """The module names of `model` that `pattern` full-matches, as PEFT selects them."""
    return {name for name, _ in model.named_modules() if re.fullmatch(pattern, name)}


def wrap_vision_projections(model: nn.Module) -> nn.Module:
    """Replace every `q_proj`/`v_proj` under the vision tower with a non-`nn.Linear` wrapper.

    Simulates Gemma 4's `Gemma4ClippableLinear`, an `nn.Module` that wraps an `nn.Linear` and that
    PEFT's LoRA layer construction rejects. The model is mutated in place and returned.
    """

    class _Wrapper(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.linear = linear

        def forward(self, hidden_states):
            return self.linear(hidden_states)

    tower = model.get_submodule("model.vision_tower")
    for name, module in list(tower.named_modules()):
        for attribute in ("q_proj", "v_proj"):
            child = getattr(module, attribute, None)
            if isinstance(child, nn.Linear):
                setattr(module, attribute, _Wrapper(child))
    assert any(
        isinstance(module, _Wrapper) for _, module in model.named_modules()
    ), "the tiny vision tower carries no q_proj/v_proj to wrap"
    return model


def lora_module_names(model: nn.Module) -> set[str]:
    """The names of the modules carrying an injected LoRA adapter."""
    return {name for name, module in model.named_modules() if hasattr(module, "lora_A")}


class TestResolvedPattern:
    def test_text_only_decoder(self):
        model = tiny_llama()
        pattern = lora_target_pattern(["q_proj", "v_proj"], model)
        assert pattern == LLAMA_PATTERN
        assert matched(model, pattern) == {
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(4)
            for projection in ("q_proj", "v_proj")
        }

    def test_gpt2(self):
        model = tiny_gpt2()
        pattern = lora_target_pattern(["c_attn"], model)
        assert pattern == GPT2_PATTERN
        assert matched(model, pattern) == {f"transformer.h.{layer}.attn.c_attn" for layer in range(4)}

    def test_composite_multimodal_wrapper_excludes_the_vision_tower(self):
        model = tiny_gemma3_conditional()
        pattern = lora_target_pattern(["q_proj", "v_proj"], model)
        assert pattern == GEMMA_PATTERN

        selected = matched(model, pattern)
        assert selected == {
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(4)
            for projection in ("q_proj", "v_proj")
        }
        assert not any("vision_tower" in name for name in selected)

        # the declared suffix list would have selected the vision tower's projections
        suffix_selected = {
            name
            for name, _ in model.named_modules()
            if name.endswith(".q_proj") or name.endswith(".v_proj")
        }
        assert any("vision_tower" in name for name in suffix_selected)

    def test_peft_wrapped_model_pattern_is_relative_to_the_inner_model(self):
        pattern = lora_target_pattern(["q_proj", "v_proj"], tiny_lora(tiny_llama()))
        assert pattern == LLAMA_PATTERN
        assert "base_model.model." not in pattern

    def test_tuple_is_scoped_like_a_list(self):
        assert lora_target_pattern(("q_proj",), tiny_llama()) == r"model\.layers\.\d+\.(?:.*\.)?(?:q_proj)"


class TestPassthrough:
    def test_regex_string_passes_through_unchanged(self):
        declared = r"model\.layers\.\d+\.self_attn\.q_proj"
        assert lora_target_pattern(declared, tiny_llama()) is declared

    def test_none_passes_through_unchanged(self):
        assert lora_target_pattern(None, tiny_llama()) is None

    def test_unknown_architecture_warns_and_returns_the_list(self):
        with pytest.warns(UserWarning, match="could not be scoped to a decoder stack"):
            result = lora_target_pattern(["q_proj"], nn.Linear(2, 2))
        assert result == ["q_proj"]


class TestPeftInjection:
    """A suffix list reaches a module PEFT cannot adapt; the scoped pattern does not."""

    def test_suffix_list_selects_an_unsupported_vision_module(self):
        from peft import LoraConfig, get_peft_model

        model = wrap_vision_projections(tiny_gemma3_conditional())
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))

    def test_scoped_pattern_injects_on_the_decoder_only(self):
        from peft import LoraConfig, get_peft_model

        model = wrap_vision_projections(tiny_gemma3_conditional())
        pattern = lora_target_pattern(["q_proj", "v_proj"], model)
        adapted = get_peft_model(model, LoraConfig(r=2, target_modules=pattern))

        assert lora_module_names(adapted) == {
            f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(4)
            for projection in ("q_proj", "v_proj")
        }

    def test_saved_adapter_records_the_pattern_and_reloads_onto_the_decoder(self, tmp_path):
        from peft import LoraConfig, PeftModel, get_peft_model

        pattern = lora_target_pattern(["q_proj", "v_proj"], tiny_gemma3_conditional())
        get_peft_model(
            tiny_gemma3_conditional(), LoraConfig(r=2, target_modules=pattern)
        ).save_pretrained(tmp_path)

        saved = json.loads((tmp_path / "adapter_config.json").read_text())
        assert saved["target_modules"] == pattern

        reloaded = PeftModel.from_pretrained(tiny_gemma3_conditional(), str(tmp_path), torch_device="cpu")
        assert lora_module_names(reloaded) == {
            f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(4)
            for projection in ("q_proj", "v_proj")
        }


class _RecordingTrainer:
    """A trainer stub capturing the `peft_config` its mixin passed in."""

    configs: list = []

    def __init__(self, model=None, peft_config=None, **kwargs):
        self.model = model
        self.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
        type(self).configs.append(peft_config)

    def train(self, resume_from_checkpoint=None, **kwargs):
        pass

    def save_model(self, output_dir):
        pass


class _ScoringModelStub(nn.Module):
    """A sequence-classification stand-in for PPO's reward and value models."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)


@pytest.fixture
def recorder(monkeypatch):
    """The recording trainer, with a cleared log, installed into every TRL mixin module."""
    _RecordingTrainer.configs = []
    monkeypatch.setattr(sft_mixin, "SFTTrainer", _RecordingTrainer)
    monkeypatch.setattr(dpo_mixin, "DPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(grpo_mixin, "GRPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(ppo_mixin, "PPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(dpo_mixin, "standardize_preference_dataset", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(
        ppo_mixin.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, path, **kwargs: _ScoringModelStub(vocab_size=10_000)),
    )
    return _RecordingTrainer


def _reward_stub(prompts, completions, **kwargs):
    return [0.0] * len(completions)


def _controls(tmp_path):
    """One control per TRL wrapper, each configured for a LoRA run that trains nothing real."""
    common = dict(use_peft=True, output_dir=str(tmp_path), training_args={"use_cpu": True})
    return {
        "sft": SFT(
            train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]),
            load_best_model_at_end=False,
            **common,
        ),
        "dpo": DPO(
            train_dataset=Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}]),
            load_best_model_at_end=False,
            **common,
        ),
        "grpo": GRPO(
            train_dataset=Dataset.from_list([{"prompt": "p"}]),
            reward_funcs=[_reward_stub],
            num_generations=2,
            per_device_train_batch_size=2,
            **common,
        ),
        "ppo": PPO(
            train_dataset=Dataset.from_list([{"prompt": "p"}]),
            reward_model_name_or_path="reward/path",
            **common,
        ),
    }


class TestMixinWiring:
    """Every TRL wrapper hands the trainer a config scoped to the resolved decoder stack."""

    @pytest.mark.parametrize("wrapper", ["sft", "dpo", "grpo", "ppo"])
    def test_peft_config_targets_the_decoder_stack(self, wrapper, recorder, mock_tokenizer, tmp_path):
        control = _controls(tmp_path)[wrapper]
        control.steer(tiny_gemma3_conditional(), tokenizer=mock_tokenizer)

        assert len(recorder.configs) == 1
        assert recorder.configs[0].target_modules == GEMMA_PATTERN

    @pytest.mark.parametrize("wrapper", ["sft", "dpo", "grpo", "ppo"])
    def test_declared_target_modules_are_unchanged(self, wrapper, recorder, mock_tokenizer, tmp_path):
        control = _controls(tmp_path)[wrapper]
        control.steer(tiny_gemma3_conditional(), tokenizer=mock_tokenizer)

        assert control.args.target_modules == ["q_proj", "v_proj"]
        assert control.fit_identity()["target_modules"] == ["q_proj", "v_proj"]
        assert control.lora_kwargs["target_modules"] == ["q_proj", "v_proj"]

    def test_peft_off_yields_no_config(self, recorder, mock_tokenizer, tmp_path):
        control = SFT(
            train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]),
            output_dir=str(tmp_path),
            load_best_model_at_end=False,
            training_args={"use_cpu": True},
        )
        control.steer(tiny_llama(), tokenizer=mock_tokenizer)

        assert recorder.configs == [None]

    def test_declared_regex_reaches_the_trainer_unchanged(self, recorder, mock_tokenizer, tmp_path):
        declared = r"model\.layers\.\d+\.self_attn\.q_proj"
        control = SFT(
            train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]),
            use_peft=True,
            target_modules=declared,
            output_dir=str(tmp_path),
            load_best_model_at_end=False,
            training_args={"use_cpu": True},
        )
        control.steer(tiny_llama(), tokenizer=mock_tokenizer)

        assert recorder.configs[0].target_modules == declared


class TestArgsValidation:
    @pytest.mark.parametrize("declared", [[], "", [1, 2], 7])
    def test_rejected_target_modules(self, declared):
        with pytest.raises(ValueError, match="target_modules"):
            SFT(use_peft=True, target_modules=declared)

    def test_regex_string_is_accepted(self):
        control = SFT(use_peft=True, target_modules=r"model\..*q_proj")
        assert control.args.target_modules == r"model\..*q_proj"
        assert control.lora_kwargs["target_modules"] == r"model\..*q_proj"
