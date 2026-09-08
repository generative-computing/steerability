"""DPO wrapper smoke test for the templated preference path.

Runs a one-step LoRA DPO fit with `prompt_format="chat_prompt"` on a tiny hub model (CPU), over both
the single sigmoid loss and the `["sigmoid", "sft"]` list form, and asserts the standardized prompts
carry the chat template with a clean prompt/completion token boundary, the run trains, and the result
freezes to a `load_lora` entry. Learned behavior is not asserted.
"""
from __future__ import annotations

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.structural_control.wrappers.trl.dpotrainer import DPO

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.mark.parametrize(
    "loss_kwargs",
    [
        {"loss_type": "sigmoid"},
        {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0]},
    ],
    ids=["sigmoid", "sigmoid+sft"],
)
def test_chat_prompt_dpo_trains_and_freezes_to_load_lora(tmp_path, loss_kwargs):
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_TEMPLATE

    control = DPO(
        train_dataset=Dataset.from_list(
            [
                {"prompt": "Is the sky blue?", "chosen": "ANSWER: yes", "rejected": "ANSWER: no"},
                {"prompt": "Is grass red?", "chosen": "ANSWER: no", "rejected": "ANSWER: yes"},
            ]
        ),
        prompt_format="chat_prompt",
        output_dir=str(tmp_path / "adapter"),
        use_peft=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=100,
        load_best_model_at_end=False,
        training_args={"max_steps": 1, "use_cpu": True, "bf16": False, "fp16": False},
        **loss_kwargs,
    )
    trained = control.steer(model, tokenizer=tokenizer)
    assert trained is not None

    # standardized prompts are chat-rendered and token-prefix their prompt + chosen concatenation
    row = control.train_dataset[0]
    assert row["prompt"] != "Is the sky blue?"
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    joint_ids = tokenizer(row["prompt"] + row["chosen"], add_special_tokens=False)["input_ids"]
    assert joint_ids[: len(prompt_ids)] == prompt_ids

    assert any((tmp_path / "adapter").iterdir())
    state = control.export_state()
    assert isinstance(state["artifact"], LoRAArtifact)
    method, _ = control.frozen_form(state)
    assert method == "structural_control/load_lora"
