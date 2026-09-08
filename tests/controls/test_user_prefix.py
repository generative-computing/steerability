import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.user_prefix.args import UserPrefixArgs
from steerability.algorithms.input_control.user_prefix.control import UserPrefix
from steerability.spipe import SPipe
from steerability.utils.rendering import has_chat_template
from tests.utils.sweep import build_param_grid

MARKER = "|HONEST_ONLY|"
SEPARATOR = " "

PLACEMENT_GRID = {
    "placement": ["last_user", "first_user", "all_user"],
    "n_turns": [1, 2],
}


# args validation (no model)

def test_args_empty_text_raises():
    with pytest.raises(ValueError, match="non-empty"):
        UserPrefixArgs(text="")


def test_args_non_str_text_raises():
    with pytest.raises(TypeError, match="text must be a str"):
        UserPrefixArgs(text=123)


def test_args_bad_placement_raises():
    with pytest.raises(ValueError, match="placement"):
        UserPrefixArgs(text="ok", placement="middle")


def test_args_bad_separator_raises():
    with pytest.raises(TypeError, match="separator must be a str"):
        UserPrefixArgs(text="ok", separator=1)


def test_args_defaults():
    args = UserPrefixArgs(text=MARKER)
    assert args.separator == "\n\n"
    assert args.placement == "last_user"


def test_control_construction_promotes_fields():
    control = UserPrefix(text=MARKER, separator=SEPARATOR, placement="all_user")
    assert control.text == MARKER
    assert control.separator == SEPARATOR
    assert control.placement == "all_user"


# message adaptation (no model, tokenizer-only steer)

@pytest.mark.parametrize("conf", build_param_grid(PLACEMENT_GRID))
def test_adapt_messages_places_marker(model_and_tokenizer, conf):
    _, tokenizer = model_and_tokenizer
    control = UserPrefix(text=MARKER, separator=SEPARATOR, placement=conf["placement"])
    control.steer(tokenizer=tokenizer)

    if conf["n_turns"] == 1:
        chat = [{"role": "user", "content": "first question"}]
        user_positions = [0]
    else:
        chat = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "an answer"},
            {"role": "user", "content": "second question"},
        ]
        user_positions = [0, 2]

    adapted = control.adapt_messages([chat])
    assert adapted is not None
    out = adapted[0]

    placement = conf["placement"]
    if placement == "first_user":
        expected_marked = {user_positions[0]}
    elif placement == "last_user":
        expected_marked = {user_positions[-1]}
    else:
        expected_marked = set(user_positions)

    for idx in user_positions:
        content = out[idx]["content"]
        if idx in expected_marked:
            assert content.startswith(MARKER + SEPARATOR), f"turn {idx} should carry the marker: {content!r}"
        else:
            assert MARKER not in content, f"turn {idx} should be unchanged: {content!r}"


def test_adapt_messages_no_user_turn_appends(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    control = UserPrefix(text=MARKER)
    control.steer(tokenizer=tokenizer)
    out = control.adapt_messages([[{"role": "system", "content": "sys"}]])[0]
    assert any(m.get("role") == "user" and MARKER in m.get("content", "") for m in out)


def test_adapt_messages_batch_independent(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    control = UserPrefix(text=MARKER, separator=SEPARATOR, placement="last_user")
    control.steer(tokenizer=tokenizer)
    batch = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]
    out = control.adapt_messages(batch)
    assert out[0][0]["content"] == f"{MARKER}{SEPARATOR}a"
    assert out[1][0]["content"] == f"{MARKER}{SEPARATOR}b"


# pipeline integration over the model/device grid

def test_pipeline_generates(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    model = base_model.to(device)

    control = UserPrefix(text=MARKER, separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    out = pipeline.generate(
        messages=[{"role": "user", "content": "where is the eiffel tower?"}],
        max_new_tokens=8,
        do_sample=False,
    )
    assert isinstance(out, str)
    assert len(out) >= 0  # generation ran without error


def test_pipeline_composes_with_few_shot(model_and_tokenizer, device: torch.device):
    """FewShot's directive (a system turn) and UserPrefix's marker (a user turn) both land in the templated prompt."""
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    model = base_model.to(device)

    directive = "Answer honestly."
    controls = [
        FewShot(directive=directive),
        UserPrefix(text=MARKER, separator=SEPARATOR),
    ]
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()

    output = pipeline.generate(
        messages=[{"role": "user", "content": "is the sky green?"}],
        max_new_tokens=4,
        do_sample=False,
        return_output=True,
    )
    prompt_text = tokenizer.decode(output.adapted_input_ids[0].tolist(), skip_special_tokens=True)
    assert directive in prompt_text, f"directive missing from templated prompt: {prompt_text!r}"
    assert MARKER in prompt_text, f"marker missing from templated prompt: {prompt_text!r}"


# token path (no chat structure)

def test_token_path_prefixes_marker(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    control = UserPrefix(text=MARKER, separator=SEPARATOR)
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()

    output = pipeline.generate(
        text="the answer is",
        max_new_tokens=4,
        do_sample=False,
        return_output=True,
    )
    prefix_ids = tokenizer.encode(MARKER + SEPARATOR, add_special_tokens=False)
    head = output.adapted_input_ids[0].tolist()[: len(prefix_ids)]
    assert head == prefix_ids, f"encoded marker should head the token stream; got {head!r} vs {prefix_ids!r}"


def test_adapt_before_steer_raises():
    control = UserPrefix(text=MARKER)
    with pytest.raises(RuntimeError, match="steer"):
        control.adapt([1, 2, 3])


# spipe round trip

def test_spipe_roundtrip_recipe_only(tmp_path, model_and_tokenizer):
    base_model, tokenizer = model_and_tokenizer
    if not has_chat_template(tokenizer):
        pytest.skip("model has no chat template; messages= path requires one")
    control = UserPrefix(text=MARKER, separator=SEPARATOR, placement="last_user")
    pipeline = SteeringPipeline(controls=[control], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    spipe = pipeline.to_spipe()
    config_id_before = spipe.config_id
    assert spipe.code_dependent is False

    saved = spipe.save(tmp_path / "user_prefix.spipe")
    loaded = SPipe.load(saved)  # no allow_code
    assert loaded.code_dependent is False
    assert loaded.config_id == config_id_before

    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = base_model, tokenizer
    rebuilt.steer()
    rebuilt_control = rebuilt.input_controls[0]
    assert rebuilt_control.text == MARKER
    assert rebuilt_control.placement == "last_user"
    assert rebuilt.generate(
        messages=[{"role": "user", "content": "hi"}], max_new_tokens=3, do_sample=False
    ) is not None
