"""Tests for the shared example renderer (`core/internals/render.py`).

The parity test fails if steering-vector extraction and inference produce different
prompt-region token ids. The remaining tests cover `render_for_model`'s
three modes and `render_contrastive`'s mode resolution / fallbacks.
"""
import logging

import pytest
from transformers import AutoTokenizer

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.render import render_contrastive
from steerability.utils.rendering import encode_for_model, render_for_model
from tests.utils.load_ci_models import get_models

# minimal Jinja chat template used as a fallback when no CI model ships one
_STUB_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "[{{ message['role'] | upper }}] {{ message['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}[ASSISTANT] {% endif %}"
)


def _first_chat_template_model() -> str | None:
    """Return a CI model id whose tokenizer has a chat template, or None."""
    for model_id in get_models().values():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            continue
        if getattr(tok, "chat_template", None) is not None:
            return model_id
    return None


@pytest.fixture(scope="session")
def chat_tokenizer():
    """A tokenizer that has a chat template.

    Prefers a CI model that ships a chat template; otherwise attaches a minimal
    Jinja template (and a BOS token) to the first loadable CI tokenizer.
    """
    model_id = _first_chat_template_model()
    if model_id is not None:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        models = list(get_models().values())
        tok = None
        for candidate in models:
            try:
                tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                break
            except Exception:
                continue
        if tok is None:
            pytest.skip("No CI tokenizer could be loaded.")
        tok.chat_template = _STUB_CHAT_TEMPLATE

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="session")
def raw_tokenizer():
    """A tokenizer with NO chat template, for exercising the raw fallback path."""
    for model_id in get_models().values():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            continue
        if getattr(tok, "chat_template", None) is None:
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
            return tok
    pytest.skip("No CI tokenizer without a chat template is available.")


# Parity test
def test_extraction_inference_prompt_parity(chat_tokenizer):
    """Extraction and inference must produce identical prompt-region token ids."""
    prompt = "What is the capital of France?"

    inf = render_for_model(chat_tokenizer, prompt=prompt, mode="chat_prompt")
    inf_ids = chat_tokenizer(inf, add_special_tokens=False)["input_ids"]

    ext = render_for_model(chat_tokenizer, prompt=prompt, completion=" Paris", mode="chat_completion")
    ext_ids = chat_tokenizer(ext, add_special_tokens=False)["input_ids"]

    # prompt prefix must match the inference rendering exactly
    assert ext_ids[: len(inf_ids)] == inf_ids
    # completion token(s) follow the prompt prefix
    assert len(ext_ids) > len(inf_ids)
    # no doubled BOS
    if chat_tokenizer.bos_token_id is not None:
        assert inf_ids.count(chat_tokenizer.bos_token_id) <= 1


# Chat-modality ≡ encode_for_model, and the buggy render+retokenize = +1 BOS (WS5).
def test_chat_modality_equals_encode_for_model(chat_tokenizer):
    """The pipeline chat modality (`apply_chat_template(tokenize=True)`) must produce the same token
    ids as `encode_for_model(mode="chat_prompt")` — the single source of truth."""
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]

    # transformers v5: `apply_chat_template(tokenize=True)` returns a `BatchEncoding`
    chat_ids = chat_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    efm_ids = encode_for_model(chat_tokenizer, prompt=prompt, mode="chat_prompt")["input_ids"]
    assert chat_ids == efm_ids


def test_buggy_render_then_retokenize_adds_one_bos(chat_tokenizer):
    """Rendering to a string then re-tokenizing with default add_special_tokens=True adds exactly one
    extra leading BOS relative to the correct (encode_for_model) path — the notebook's original bug."""
    prompt = "What is the capital of France?"
    text = chat_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    buggy_ids = chat_tokenizer(text)["input_ids"]  # default add_special_tokens=True
    fixed_ids = encode_for_model(chat_tokenizer, prompt=prompt, mode="chat_prompt")["input_ids"]

    # only meaningful for tokenizers that auto-prepend BOS
    if chat_tokenizer.bos_token_id is not None and \
            chat_tokenizer("x")["input_ids"][0] == chat_tokenizer.bos_token_id:
        assert len(buggy_ids) == len(fixed_ids) + 1
        assert buggy_ids[0] == chat_tokenizer.bos_token_id
        assert buggy_ids[1:] == fixed_ids


# render_for_model unit tests
class TestRenderForModel:
    def test_raw_concatenates(self, chat_tokenizer):
        out = render_for_model(chat_tokenizer, prompt="abc", completion="def", mode="raw")
        assert out == "abcdef"

    def test_chat_prompt_has_no_completion(self, chat_tokenizer):
        out = render_for_model(chat_tokenizer, prompt="hello", completion=" world", mode="chat_prompt")
        # completion is ignored in chat_prompt mode
        assert "world" not in out
        assert "hello" in out

    def test_chat_completion_appends_completion(self, chat_tokenizer):
        prompt_only = render_for_model(chat_tokenizer, prompt="hello", mode="chat_prompt")
        with_comp = render_for_model(chat_tokenizer, prompt="hello", completion=" world", mode="chat_completion")
        # chat_completion is the chat_prompt render with the completion appended
        assert with_comp == prompt_only + " world"

    def test_raw_fallback_when_no_chat_template(self, raw_tokenizer, caplog):
        with caplog.at_level(logging.WARNING):
            out = render_for_model(raw_tokenizer, prompt="abc", completion="def", mode="chat_prompt")
        assert out == "abcdef"
        assert any("no chat_template" in r.message or "chat_template" in r.message for r in caplog.records)


# render_contrastive unit tests
class TestRenderContrastive:
    def test_chat_completion_without_prompts_falls_back_to_raw(self, chat_tokenizer, caplog):
        data = ContrastivePairs(positives=["yes"], negatives=["no"])
        with caplog.at_level(logging.WARNING):
            rendered = render_contrastive(chat_tokenizer, data, "chat_completion")
        assert rendered.effective_mode == "raw"
        assert rendered.add_special_tokens is True
        assert rendered.pos_texts == ["yes"]
        assert rendered.prompt_texts is None
        assert any("chat_completion" in r.message for r in caplog.records)

    def test_chat_completion_with_prompts(self, chat_tokenizer):
        data = ContrastivePairs(positives=[" A"], negatives=[" B"], prompts=["Q?"])
        rendered = render_contrastive(chat_tokenizer, data, "chat_completion")
        assert rendered.effective_mode == "chat_completion"
        assert rendered.add_special_tokens is False
        # prompt_texts are the user-turn-only renders, and each pos/neg extends them
        assert rendered.prompt_texts is not None
        assert rendered.pos_texts[0] == rendered.prompt_texts[0] + " A"
        assert rendered.neg_texts[0] == rendered.prompt_texts[0] + " B"

    def test_chat_prompt_ignores_prompts(self, chat_tokenizer):
        data = ContrastivePairs(positives=["hello"], negatives=["world"], prompts=["IGNORED"])
        rendered = render_contrastive(chat_tokenizer, data, "chat_prompt")
        assert rendered.effective_mode == "chat_prompt"
        assert rendered.add_special_tokens is False
        assert rendered.prompt_texts is None
        # each positive/negative is rendered as a standalone user turn
        assert "hello" in rendered.pos_texts[0]
        assert "IGNORED" not in rendered.pos_texts[0]

    def test_raw_with_prompts_concatenates(self, chat_tokenizer):
        data = ContrastivePairs(positives=["A"], negatives=["B"], prompts=["Q "])
        rendered = render_contrastive(chat_tokenizer, data, "raw")
        assert rendered.effective_mode == "raw"
        assert rendered.add_special_tokens is True
        assert rendered.pos_texts == ["Q A"]
        assert rendered.neg_texts == ["Q B"]
        assert rendered.prompt_texts == ["Q "]

    def test_chat_modes_fall_back_to_raw_without_chat_template(self, raw_tokenizer, caplog):
        data = ContrastivePairs(positives=["A"], negatives=["B"], prompts=["Q "])
        with caplog.at_level(logging.WARNING):
            rendered = render_contrastive(raw_tokenizer, data, "chat_prompt")
        assert rendered.effective_mode == "raw"
        assert rendered.add_special_tokens is True
