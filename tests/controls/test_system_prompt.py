import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.system_prompt.args import SystemPromptArgs
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.spipe import SPipe
from steerability.utils.rendering import has_chat_template
from tests.utils.sweep import build_param_grid

CONTROL_SYS = "|CONTROL_SYS|"
SETTING_SYS = "|SETTING_SYS|"
SEPARATOR = " "

MODE_GRID = {
    "mode": ["prepend", "append", "replace"],
    "n_turns": [1, 2],
}


# args validation (no model)

def test_args_empty_text_raises():
    with pytest.raises(ValueError, match="non-empty"):
        SystemPromptArgs(text="")


def test_args_non_str_text_raises():
    with pytest.raises(TypeError, match="text must be a str"):
        SystemPromptArgs(text=123)


def test_args_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        SystemPromptArgs(text="ok", mode="merge")


def test_args_non_str_separator_raises():
    with pytest.raises(TypeError, match="separator must be a str"):
        SystemPromptArgs(text="ok", separator=1)


def test_args_defaults():
    args = SystemPromptArgs(text=CONTROL_SYS)
    assert args.mode == "prepend"
    assert args.separator == "\n\n"


def test_control_construction_promotes_fields():
    control = SystemPrompt(text=CONTROL_SYS, mode="append", separator=SEPARATOR)
    assert control.text == CONTROL_SYS
    assert control.mode == "append"
    assert control.separator == SEPARATOR


# message adaptation (no model, tokenizer-only steer)

@pytest.mark.parametrize("conf", build_param_grid(MODE_GRID))
def test_adapt_messages_with_existing_system(model_and_tokenizer, conf):
    _, tokenizer = model_and_tokenizer
    control = SystemPrompt(text=CONTROL_SYS, mode=conf["mode"], separator=SEPARATOR)
    control.steer(tokenizer=tokenizer)

    if conf["n_turns"] == 1:
        chat = [
            {"role": "system", "content": SETTING_SYS},
            {"role": "user", "content": "first question"},
        ]
    else:
        chat = [
            {"role": "system", "content": SETTING_SYS},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "an answer"},
            {"role": "user", "content": "second question"},
        ]

    out = control.adapt_messages([chat])[0]
    system_messages = [m for m in out if m["role"] == "system"]
    assert len(system_messages) == 1
    content = system_messages[0]["content"]

    if conf["mode"] == "prepend":
        assert content == f"{CONTROL_SYS}{SEPARATOR}{SETTING_SYS}"
    elif conf["mode"] == "append":
        assert content == f"{SETTING_SYS}{SEPARATOR}{CONTROL_SYS}"
    else:
        assert content == CONTROL_SYS
        assert SETTING_SYS not in content


@pytest.mark.parametrize("mode", ["prepend", "append", "replace"])
def test_adapt_messages_no_existing_system(model_and_tokenizer, mode):
    _, tokenizer = model_and_tokenizer
    control = SystemPrompt(text=CONTROL_SYS, mode=mode, separator=SEPARATOR)
    control.steer(tokenizer=tokenizer)
    out = control.adapt_messages([[{"role": "user", "content": "q"}]])[0]
    assert out[0] == {"role": "system", "content": CONTROL_SYS}
    assert out[1]["role"] == "user"


def test_adapt_messages_batch_independent(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    control = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    control.steer(tokenizer=tokenizer)
    batch = [
        [{"role": "system", "content": "one"}, {"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]
    out = control.adapt_messages(batch)
    assert out[0][0]["content"] == f"{CONTROL_SYS}{SEPARATOR}one"
    assert out[1][0]["content"] == CONTROL_SYS


# pipeline integration over the model/device grid

def test_pipeline_generates(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    model = base_model.to(device)

    control = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    out = pipeline.generate(
        messages=[{"role": "system", "content": SETTING_SYS}, {"role": "user", "content": "hi"}],
        max_new_tokens=8,
        do_sample=False,
    )
    assert isinstance(out, str)
    assert len(out) >= 0  # generation ran without error


def test_pipeline_prepend_preserves_setting_prompt(model_and_tokenizer, device: torch.device):
    """Under prepend, both the control text and the setting's system prompt land in the templated prompt,
    with the control text ahead of the setting prompt."""
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    model = base_model.to(device)

    control = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    output = pipeline.generate(
        messages=[{"role": "system", "content": SETTING_SYS}, {"role": "user", "content": "is the sky green?"}],
        max_new_tokens=4,
        do_sample=False,
        return_output=True,
    )
    prompt_text = tokenizer.decode(output.adapted_input_ids[0].tolist(), skip_special_tokens=True)
    assert CONTROL_SYS in prompt_text, f"control text missing: {prompt_text!r}"
    assert SETTING_SYS in prompt_text, f"setting prompt missing: {prompt_text!r}"
    assert prompt_text.index(CONTROL_SYS) < prompt_text.index(SETTING_SYS)


def test_composes_with_few_shot(model_and_tokenizer):
    """The a2-shaped stack: SystemPrompt merges ahead of the setting's system prompt, and FewShot inserts its
    example block afterward without deleting the setting's prompt. Verified over the folded message adaptation
    (as the pipeline chains input controls in list order), since two consecutive system messages exceed what the
    tiny CI chat templates accept."""
    _, tokenizer = model_and_tokenizer
    system_prompt = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    few_shot = FewShot(positive_example_pool=[{"input": "hey", "output": "Good afternoon."}], k_positive=1)
    system_prompt.steer(tokenizer=tokenizer)
    few_shot.steer(tokenizer=tokenizer)

    chat = [{"role": "system", "content": SETTING_SYS}, {"role": "user", "content": "hi"}]
    adapted = [chat]
    for control in (system_prompt, few_shot):
        adapted = control.adapt_messages(adapted)

    out = adapted[0]
    leading_system = out[0]
    assert leading_system["role"] == "system"
    assert leading_system["content"] == f"{CONTROL_SYS}{SEPARATOR}{SETTING_SYS}"
    assert any("Good afternoon." in m["content"] for m in out if m["role"] == "system")


# token path (no chat structure)

def test_token_path_sets_instruction(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    control = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    with pytest.warns(UserWarning):
        output = pipeline.generate(
            text="the answer is",
            max_new_tokens=4,
            do_sample=False,
            return_output=True,
        )
    prompt_text = tokenizer.decode(output.adapted_input_ids[0].tolist(), skip_special_tokens=True)
    assert CONTROL_SYS in prompt_text, f"instruction missing from token-path prompt: {prompt_text!r}"


def test_adapt_before_steer_raises():
    control = SystemPrompt(text=CONTROL_SYS)
    with pytest.raises(RuntimeError, match="steer"):
        control.adapt([1, 2, 3])


# spipe round trip

def test_spipe_roundtrip_recipe_only(tmp_path, model_and_tokenizer):
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    control = SystemPrompt(text=CONTROL_SYS, mode="prepend", separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    spipe = pipeline.to_spipe()
    config_id_before = spipe.config_id
    assert spipe.code_dependent is False

    saved = spipe.save(tmp_path / "system_prompt.spipe")
    loaded = SPipe.load(saved)  # no allow_code
    assert loaded.code_dependent is False
    assert loaded.config_id == config_id_before

    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = base_model, tokenizer
    rebuilt.steer()
    rebuilt_control = rebuilt.input_controls[0]
    assert rebuilt_control.text == CONTROL_SYS
    assert rebuilt_control.mode == "prepend"
    assert rebuilt.generate(
        messages=[{"role": "system", "content": SETTING_SYS}, {"role": "user", "content": "hi"}],
        max_new_tokens=3,
        do_sample=False,
    ) is not None
