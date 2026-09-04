import random
import warnings

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.common.formatters.few_shot_block import FewShotBlockFormatter
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.few_shot.control import FewShot
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Classify the sentiment of the following sentence as Positive or Negative.\n"
    "Sentence: I loved the cinematography but the plot was thin."
)

POS_POOL = [
    {"input": "The service was excellent and the food was great.", "label": "Positive"},
    {"input": "What an amazing performance; I had a wonderful time!", "label": "Positive"},
]
NEG_POOL = [
    {"input": "The device kept crashing and the battery died fast.", "label": "Negative"},
    {"input": "Terrible support; I regret this purchase.", "label": "Negative"},
]

FEWSHOT_GRID = {
    "mode": ["runtime", "pool", "none"],
    "k_positive": [1, 2],
    "k_negative": [0, 1],
    "selector": ["random"],
    "use_negative_runtime": [False, True],
}


def _runtime_kwargs_from_conf(conf):
    if conf["mode"] != "runtime":
        return {}
    pos = [{"input": "I am thrilled with the results.", "label": "Positive"}]
    neg = [{"input": "This was a waste of time.", "label": "Negative"}] if conf["use_negative_runtime"] else []
    runtime_kwargs = {}
    if pos:
        runtime_kwargs["positive_examples"] = pos
    if neg:
        runtime_kwargs["negative_examples"] = neg
    return runtime_kwargs


@pytest.mark.parametrize("conf", build_param_grid(FEWSHOT_GRID))
def test_few_shot(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that FewShot adapts prompts and generates on every model/device/param combo. Also sanity-check that the
    adapted prompt length increases when examples or a directive are provided.
    """
    # deterministic selector behavior
    random.seed(0)
    torch.manual_seed(0)

    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build FewShot control based on the mode
    kwargs = {
        "directive": "Follow the schema. Classify correctly using the demonstrations",
        "selector": conf["selector"],
    }

    if conf["mode"] == "pool":
        kwargs.update(
            dict(
                positive_example_pool=POS_POOL,
                negative_example_pool=NEG_POOL if conf["k_negative"] > 0 else None,
                k_positive=conf["k_positive"],
                k_negative=conf["k_negative"],
            )
        )
    elif conf["mode"] == "runtime":
        # no pools; runtime examples will be provided via runtime_kwargs
        kwargs.update(dict(k_positive=None, k_negative=None))
    else:  # "none"; deliberately provide no pools and no runtime examples (but directive is still set)
        kwargs.update(dict(k_positive=None, k_negative=None))

    fewshot = FewShot(**kwargs)

    # pipeline
    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()

    # prepare inputs & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = _runtime_kwargs_from_conf(conf)

    # sanity check
    adapted = fewshot.adapt(prompt_ids, runtime_kwargs=runtime_kwargs)

    # handle tensor/list shapes consistently
    if isinstance(adapted, torch.Tensor):
        adapted_len = adapted.size(-1) if adapted.ndim > 1 else adapted.size(0)
        orig_len = prompt_ids.size(-1)
    else:
        adapted_len = len(adapted)
        orig_len = len(tokenizer.encode(PROMPT_TEXT, add_special_tokens=False))

    # with the directive set, all modes (runtime, pool, none) should expand the prompt
    assert adapted_len > orig_len, "FewShot should prepend a directive/examples block and increase prompt length"

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"


PROMPT_TEXT_SHORT = "Hello world"
PROMPT_TEXT_SHORT_2 = "Goodbye moon"


@pytest.mark.parametrize("input_format", ["tensor_1d", "tensor_2d", "list_flat", "list_nested"])
def test_few_shot_batch_formats(model_and_tokenizer, device: torch.device, input_format: str):
    """Test that FewShot correctly handles various input formats and preserves format on output."""
    random.seed(42)
    torch.manual_seed(42)

    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    fewshot = FewShot(
        directive="Follow these examples",
        positive_example_pool=POS_POOL,
        k_positive=1,
    )

    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()


    # prepare input in the specified format
    tokens_1 = tokenizer.encode(PROMPT_TEXT_SHORT, add_special_tokens=False)
    tokens_2 = tokenizer.encode(PROMPT_TEXT_SHORT_2, add_special_tokens=False)

    if input_format == "tensor_1d":
        input_ids = torch.tensor(tokens_1, device=device)
    elif input_format == "tensor_2d":
        input_ids = torch.tensor([tokens_1], device=device)
    elif input_format == "list_flat":
        input_ids = tokens_1
    else:  # list_nested
        input_ids = [tokens_1, tokens_2]

    adapted = fewshot.adapt(input_ids, runtime_kwargs={})

    # verify output format matches input format
    if input_format == "tensor_1d":
        assert isinstance(adapted, torch.Tensor), "Expected tensor output for tensor input"
        assert adapted.ndim == 1, "Expected 1D tensor for 1D tensor input"
        assert adapted.device.type == device.type, "Device type should be preserved"
        assert len(adapted) > len(tokens_1), "Adapted should be longer with examples prepended"

    elif input_format == "tensor_2d":
        assert isinstance(adapted, torch.Tensor), "Expected tensor output for tensor input"
        assert adapted.ndim == 2, "Expected 2D tensor for 2D tensor input"
        assert adapted.device.type == device.type, "Device type should be preserved"
        assert adapted.size(0) == 1, "Batch size should be preserved"
        assert adapted.size(1) > len(tokens_1), "Adapted should be longer with examples prepended"

    elif input_format == "list_flat":
        assert isinstance(adapted, list), "Expected list output for list input"
        assert isinstance(adapted[0], int), "Expected flat list of ints for flat list input"
        assert len(adapted) > len(tokens_1), "Adapted should be longer with examples prepended"

    else:  # list_nested
        assert isinstance(adapted, list), "Expected list output for list input"
        assert isinstance(adapted[0], list), "Expected nested list for nested list input"
        assert len(adapted) == 2, "Batch size should be preserved"
        # both sequences should be padded to same length
        assert len(adapted[0]) == len(adapted[1]), "Batched sequences should be padded to same length"
        assert len(adapted[0]) > len(tokens_1), "Adapted should be longer with examples prepended"


def test_few_shot_1d_tensor_bug_regression(model_and_tokenizer, device: torch.device):
    """Regression test: 1D tensor input should decode correctly, not grab a single token ID."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    fewshot = FewShot(
        directive="Test directive",
        positive_example_pool=POS_POOL,
        k_positive=1,
    )

    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()


    # create 1D tensor input
    tokens = tokenizer.encode(PROMPT_TEXT_SHORT, add_special_tokens=False)
    input_ids_1d = torch.tensor(tokens, device=device)

    # this would fail before the fix: .tolist()[0] grabbed a single int instead of the sequence
    adapted = fewshot.adapt(input_ids_1d, runtime_kwargs={})

    # the adapted output should contain the original text (decoded correctly)
    decoded = tokenizer.decode(adapted.tolist(), skip_special_tokens=True)
    assert PROMPT_TEXT_SHORT in decoded, "Original prompt text should appear in adapted output"


def test_adapt_messages_inserts_single_system_block(model_and_tokenizer, device: torch.device):
    """`adapt_messages` should insert a single system message containing the directive and example blocks."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    fewshot = FewShot(
        directive="follow examples",
        positive_example_pool=POS_POOL,
        k_positive=2,
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()

    messages = [[{"role": "user", "content": "How was the movie?"}]]
    adapted = fewshot.adapt_messages(messages)
    assert adapted is not None
    chat = adapted[0]
    # one system message (directive + examples block) followed by the original user message
    assert len(chat) == 2
    assert chat[0]["role"] == "system"
    assert "follow examples" in chat[0]["content"]
    assert FewShotBlockFormatter.DEFAULT_POSITIVE_HEADER in chat[0]["content"]
    assert chat[-1]["role"] == "user"
    assert chat[-1]["content"] == "How was the movie?"


def test_adapt_messages_returns_none_when_nothing_configured(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    fewshot = FewShot()
    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()
    out = fewshot.adapt_messages([[{"role": "user", "content": "?"}]])
    assert out is None


def test_selector_accepts_instance(model_and_tokenizer, device: torch.device):
    """`selector=` should accept a BaseSelector instance directly (not just a string name)."""
    from aisteer360.algorithms.input_control.common.selectors.random import RandomSelector

    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    instance = RandomSelector(seed=42)
    fewshot = FewShot(
        positive_example_pool=POS_POOL,
        k_positive=1,
        selector=instance,
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()
    # the resolved selector is the same instance we passed in
    assert fewshot._selector is instance


def test_unknown_selector_name_raises(model_and_tokenizer, device: torch.device):
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)
    fewshot = FewShot(
        positive_example_pool=POS_POOL,
        k_positive=1,
        selector="nonexistent",
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="Unknown selector"):
        pipeline.steer()


def test_few_shot_missing_pad_token_raises(model_and_tokenizer, device: torch.device):
    """Test that missing pad_token_id raises RuntimeError instead of silently using 0."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    fewshot = FewShot(
        directive="Test directive",
        positive_example_pool=POS_POOL,
        k_positive=1,
    )

    pipeline = SteeringPipeline(controls=[fewshot], model=model, tokenizer=tokenizer)
    pipeline.steer()


    tokens = tokenizer.encode(PROMPT_TEXT_SHORT, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)

    # temporarily remove pad_token_id to simulate missing pad token
    original_pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token_id = None
    try:
        with pytest.raises(RuntimeError, match="pad_token_id"):
            fewshot.adapt(input_ids, runtime_kwargs={})
    finally:
        tokenizer.pad_token_id = original_pad_token_id


# New content-level tests for the redesigned formatter / control


def _decode_adapted_text(fewshot: FewShot, tokenizer, prompt: str, runtime_kwargs: dict | None = None) -> str:
    """Helper: run adapt() on `prompt` and return the decoded adapted text."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    adapted = fewshot.adapt(input_ids, runtime_kwargs=runtime_kwargs or {})
    if isinstance(adapted, torch.Tensor):
        if adapted.ndim == 2:
            adapted = adapted[0]
        return tokenizer.decode(adapted.tolist(), skip_special_tokens=True)
    if isinstance(adapted[0], list):
        adapted = adapted[0]
    return tokenizer.decode(adapted, skip_special_tokens=True)


def test_content_survives_schema_agnostic_keys(model_and_tokenizer, device: torch.device):
    """Pool examples with arbitrary keys ({question, answer}) should appear in the adapted prompt."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    pool = [
        {"question": "What is 2 plus 2?", "answer": "Four."},
        {"question": "Capital of France?", "answer": "Paris."},
    ]

    fewshot = FewShot(
        directive="Answer concisely.",
        positive_example_pool=pool,
        k_positive=2,
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    decoded = _decode_adapted_text(fewshot, tokenizer, "How tall is Everest?")
    for ex in pool:
        assert ex["question"] in decoded, f"Example question text missing: {ex['question']!r}"
        assert ex["answer"] in decoded, f"Example answer text missing: {ex['answer']!r}"


def test_default_headers_present(model_and_tokenizer, device: torch.device):
    """With one positive and one negative example, both default headers should appear, and the negative
    header should precede the negative example body."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    fewshot = FewShot(
        positive_example_pool=[{"input": "Great food.", "label": "Positive"}],
        negative_example_pool=[{"input": "Awful service.", "label": "Negative"}],
        k_positive=1,
        k_negative=1,
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    decoded = _decode_adapted_text(fewshot, tokenizer, "Was the meal good?")
    assert FewShotBlockFormatter.DEFAULT_POSITIVE_HEADER in decoded
    assert FewShotBlockFormatter.DEFAULT_NEGATIVE_HEADER in decoded
    neg_header_idx = decoded.index(FewShotBlockFormatter.DEFAULT_NEGATIVE_HEADER)
    neg_body_idx = decoded.index("Awful service.")
    assert neg_header_idx < neg_body_idx, "Negative header should precede the negative example body"


def test_custom_headers_propagate(model_and_tokenizer, device: torch.device):
    """Custom headers passed through the formatter should appear in the adapted prompt; defaults must not."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    fewshot = FewShot(
        formatter=FewShotBlockFormatter("@@P@@", "@@N@@"),
        positive_example_pool=[{"input": "Great food.", "label": "Positive"}],
        negative_example_pool=[{"input": "Awful service.", "label": "Negative"}],
        k_positive=1,
        k_negative=1,
    )
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    decoded = _decode_adapted_text(fewshot, tokenizer, "Was the meal good?")
    assert "@@P@@" in decoded
    assert "@@N@@" in decoded
    assert FewShotBlockFormatter.DEFAULT_POSITIVE_HEADER not in decoded
    assert FewShotBlockFormatter.DEFAULT_NEGATIVE_HEADER not in decoded


def test_directive_only_via_adapt_messages(model_and_tokenizer, device: torch.device):
    """A directive on its own (no example pools) should produce a leading system message via adapt_messages."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    fewshot = FewShot(directive="be concise")
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    out = fewshot.adapt_messages([[{"role": "user", "content": "x"}]])
    assert out is not None
    chat = out[0]
    assert chat[0]["role"] == "system"
    assert "be concise" in chat[0]["content"]


def test_directive_only_via_adapt(model_and_tokenizer, device: torch.device):
    """A directive on its own should also expand a 2-D tensor input via adapt()."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    fewshot = FewShot(directive="be concise")
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    input_ids = tokenizer("Tell me about cats.", return_tensors="pt").input_ids
    adapted = fewshot.adapt(input_ids, runtime_kwargs={})
    assert isinstance(adapted, torch.Tensor)
    assert adapted.size(-1) > input_ids.size(-1)
    decoded = tokenizer.decode(adapted[0].tolist(), skip_special_tokens=True)
    assert "be concise" in decoded


def test_no_examples_no_directive_warns_and_passes_through(model_and_tokenizer, device: torch.device):
    """With nothing configured, adapt() should warn and return the input unchanged."""
    base_model, tokenizer = model_and_tokenizer
    _ = base_model.to(device)

    fewshot = FewShot()
    pipeline = SteeringPipeline(controls=[fewshot], model=base_model, tokenizer=tokenizer)
    pipeline.steer()

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapted = fewshot.adapt(input_ids, runtime_kwargs={})
    messages = [str(w.message) for w in caught]
    assert any("nothing to inject" in m for m in messages), f"Expected warning not emitted; got {messages!r}"
    assert torch.equal(adapted, input_ids), "Input should be returned unchanged when nothing is configured"


def test_render_path_parity():
    """`apply_to_messages` and `apply_to_ids` should render the same content from the same memory."""
    formatter = FewShotBlockFormatter()
    memory = TextMemory(slots={
        "directive": "be concise",
        "examples": [
            {"question": "Q1?", "answer": "A1.", "_polarity": "positive"},
            {"question": "Q2?", "answer": "A2.", "_polarity": "negative"},
        ],
    })

    # message-side: extract the system content
    chats = formatter.apply_to_messages([[{"role": "user", "content": "x"}]], memory)
    chat = chats[0]
    assert chat[0]["role"] == "system"
    message_text = chat[0]["content"]

    # token-side: build using a stub tokenizer to capture the encoded string
    captured: dict[str, str] = {}

    class _StubTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            captured["text"] = text
            # return a dummy non-empty token sequence; values don't matter for this test
            return [1, 2, 3]

    user_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    out_ids = formatter.apply_to_ids(user_ids, memory, _StubTokenizer())
    assert out_ids.size(1) == user_ids.size(1) + 3
    ids_text = captured["text"].rstrip("\n")

    # both paths share the directive, both default headers, and both example bodies in the same order
    for expected in (
        "be concise",
        FewShotBlockFormatter.DEFAULT_POSITIVE_HEADER,
        FewShotBlockFormatter.DEFAULT_NEGATIVE_HEADER,
        "Question: Q1?",
        "Answer: A1.",
        "Question: Q2?",
        "Answer: A2.",
    ):
        assert expected in message_text, f"Missing in message path: {expected!r}"
        assert expected in ids_text, f"Missing in ids path: {expected!r}"

    assert message_text.index("Q1?") < message_text.index("Q2?")
    assert ids_text.index("Q1?") < ids_text.index("Q2?")
