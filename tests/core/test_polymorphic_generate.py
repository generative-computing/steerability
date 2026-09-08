"""Tests for the explicit-modality dispatch of `SteeringPipeline.generate`.

Covers the keyword surface (`text=`, `messages=`, `input_ids=`), the positional-text convenience,
the error catalog (E1-E12), and the return semantics.
"""
import warnings

import pytest
import torch

from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def pipeline():
    p = SteeringPipeline(model_name_or_path=TINY_MODEL)
    p.steer()
    return p


class TestKeywordDispatchReturnTypes:
    """Each keyword source returns the default type from the dispatch table, across single/batch."""

    def test_text_str_returns_str(self, pipeline):
        out = pipeline.generate(text="hello", max_new_tokens=2)
        assert isinstance(out, str)

    def test_text_list_returns_list_str(self, pipeline):
        out = pipeline.generate(text=["a", "b"], max_new_tokens=2)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(x, str) for x in out)

    def test_messages_single_returns_str(self, pipeline):
        out = pipeline.generate(messages=[{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert isinstance(out, str)

    def test_messages_batch_returns_list_str(self, pipeline):
        out = pipeline.generate(
            messages=[[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            max_new_tokens=2,
        )
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(x, str) for x in out)

    def test_input_ids_2d_returns_tensor(self, pipeline):
        out = pipeline.generate(input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 1

    def test_input_ids_1d_returns_tensor(self, pipeline):
        out = pipeline.generate(input_ids=torch.tensor([1, 2, 3]), max_new_tokens=2)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 1

    def test_input_ids_list_int_returns_tensor(self, pipeline):
        out = pipeline.generate(input_ids=[1, 2, 3], max_new_tokens=2)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 1

    def test_input_ids_list_list_int_returns_tensor(self, pipeline):
        out = pipeline.generate(input_ids=[[1, 2, 3], [4, 5, 6]], max_new_tokens=2)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 2


class TestPositionalTextConvenience:
    """Positional `str`/`list[str]` is the permanent text convenience surface (no warning)."""

    def test_positional_str_returns_str(self, pipeline):
        out = pipeline.generate("hello", max_new_tokens=2)
        assert isinstance(out, str)

    def test_positional_list_str_returns_list_str(self, pipeline):
        out = pipeline.generate(["a", "b"], max_new_tokens=2)
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)

    def test_positional_str_emits_no_warning(self, pipeline):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.generate("hello", max_new_tokens=2)

    def test_positional_list_str_emits_no_warning(self, pipeline):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.generate(["a", "b"], max_new_tokens=2)


class TestSourceExclusivity:
    """Exactly one prompt source per call (E1/E2)."""

    def test_no_source_raises(self, pipeline):
        with pytest.raises(TypeError, match="requires a prompt"):
            pipeline.generate(max_new_tokens=2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"inputs": "hi", "text": "hi"},
            {"inputs": "hi", "messages": [{"role": "user", "content": "hi"}]},
            {"inputs": "hi", "input_ids": torch.tensor([[1, 2]])},
            {"text": "hi", "messages": [{"role": "user", "content": "hi"}]},
            {"text": "hi", "input_ids": torch.tensor([[1, 2]])},
            {"messages": [{"role": "user", "content": "hi"}], "input_ids": torch.tensor([[1, 2]])},
        ],
    )
    def test_multiple_sources_raise(self, pipeline, kwargs):
        inputs = kwargs.pop("inputs", None)
        args = (inputs,) if inputs is not None else ()
        with pytest.raises(TypeError, match="multiple prompt sources"):
            pipeline.generate(*args, max_new_tokens=2, **kwargs)


class TestTextValidation:
    """Text validation totality (E3/E4)."""

    def test_non_str_element_raises_e3(self, pipeline):
        with pytest.raises(TypeError, match="element 1 is int"):
            pipeline.generate(text=["a", 5], max_new_tokens=2)

    def test_empty_sequence_raises_e4(self, pipeline):
        with pytest.raises(ValueError, match="text= received an empty sequence"):
            pipeline.generate(text=[], max_new_tokens=2)


class TestMessagesValidation:
    """Chat validation totality (E5/E6/E7); membership checked as `Mapping`."""

    def test_empty_conversation_raises_e5(self, pipeline):
        with pytest.raises(ValueError, match="empty conversation or batch"):
            pipeline.generate(messages=[], max_new_tokens=2)

    def test_empty_inner_conversation_raises_e5(self, pipeline):
        with pytest.raises(ValueError, match="empty conversation or batch"):
            pipeline.generate(messages=[[]], max_new_tokens=2)

    def test_inner_non_mapping_raises_e6(self, pipeline):
        with pytest.raises(TypeError, match=r"messages\[0\]\[1\] must be a mapping"):
            pipeline.generate(
                messages=[[{"role": "user", "content": "x"}, "y"]], max_new_tokens=2
            )

    def test_mixed_outer_kinds_raises_e7(self, pipeline):
        with pytest.raises(TypeError, match="mixed element types at the outer level"):
            pipeline.generate(
                messages=[{"role": "user", "content": "x"}, [{"role": "user", "content": "y"}]],
                max_new_tokens=2,
            )

    def test_mapping_subclass_accepted(self, pipeline):
        class ChatMessage(dict):
            pass

        out = pipeline.generate(
            messages=[ChatMessage(role="user", content="hi")], max_new_tokens=2
        )
        assert isinstance(out, str)


class TestTokenValidation:
    """Token validation is tokens-only with no grace period (E8/E9/E10)."""

    def test_str_raises_e10(self, pipeline):
        with pytest.raises(TypeError, match="input_ids= accepts"):
            pipeline.generate(input_ids="hi", max_new_tokens=2)

    def test_chat_raises_e10(self, pipeline):
        with pytest.raises(TypeError, match="input_ids= accepts"):
            pipeline.generate(input_ids=[{"role": "user", "content": "hi"}], max_new_tokens=2)

    def test_list_str_raises_e10(self, pipeline):
        with pytest.raises(TypeError, match="input_ids= accepts"):
            pipeline.generate(input_ids=["a", "b"], max_new_tokens=2)

    def test_3d_tensor_raises_e8(self, pipeline):
        with pytest.raises(ValueError, match="must be 1-D or 2-D; got 3-D"):
            pipeline.generate(input_ids=torch.zeros(1, 1, 3, dtype=torch.long), max_new_tokens=2)

    def test_ragged_nested_lists_raise_e9(self, pipeline):
        with pytest.raises(ValueError, match="must be rectangular"):
            pipeline.generate(input_ids=[[1, 2], [3]], max_new_tokens=2)


class TestAttentionMaskPairing:
    """`attention_mask` is valid only with `input_ids=` (E11), including positional text."""

    def test_text_with_mask_raises_e11(self, pipeline):
        with pytest.raises(TypeError, match="only valid with token input"):
            pipeline.generate(
                text="hi", attention_mask=torch.ones(1, 3, dtype=torch.long), max_new_tokens=1
            )

    def test_messages_with_mask_raises_e11(self, pipeline):
        with pytest.raises(TypeError, match="only valid with token input"):
            pipeline.generate(
                messages=[{"role": "user", "content": "hi"}],
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                max_new_tokens=1,
            )

    def test_input_ids_with_mask_works(self, pipeline):
        ids = torch.tensor([[1, 2, 3]])
        out = pipeline.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=2
        )
        assert isinstance(out, torch.Tensor)

    def test_positional_str_with_mask_raises_e11(self, pipeline):
        with pytest.raises(TypeError, match="only valid with token input"):
            pipeline.generate(
                "hi", attention_mask=torch.ones(1, 3, dtype=torch.long), max_new_tokens=1
            )


class TestPositionalNonTextShapes:
    """Every positional shape other than text raises E12 at the boundary."""

    @pytest.mark.parametrize(
        "positional",
        [
            [{"role": "user", "content": "hi"}],  # single chat
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],  # batch chat
            torch.tensor([1, 2, 3]),  # 1-D tensor
            torch.tensor([[1, 2, 3]]),  # 2-D tensor
            [1, 2, 3],  # list[int]
            [[1, 2, 3], [4, 5, 6]],  # list[list[int]]
            ["a", {"role": "user", "content": "x"}],  # mixed list (not all-str)
        ],
    )
    def test_positional_non_text_raises_e12(self, pipeline, positional):
        with pytest.raises(TypeError, match="positional input to generate\\(\\)") as excinfo:
            pipeline.generate(positional, max_new_tokens=2, do_sample=False)
        message = str(excinfo.value)
        assert "messages=" in message
        assert "input_ids=" in message


class TestReturnOutputFlag:
    """`return_output=True` produces Output(s) regardless of source."""

    def test_token_single_returns_output(self, pipeline):
        out = pipeline.generate(input_ids=torch.tensor([1, 2, 3]), max_new_tokens=2, return_output=True)
        assert isinstance(out, Output)
        assert out.output_ids.shape[0] == 1
        assert out.adapted_input_ids is not None

    def test_text_batch_returns_list_output(self, pipeline):
        out = pipeline.generate(text=["a", "b"], max_new_tokens=2, return_output=True)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(x, Output) for x in out)

    def test_messages_single_returns_output(self, pipeline):
        out = pipeline.generate(
            messages=[{"role": "user", "content": "hi"}], max_new_tokens=2, return_output=True
        )
        assert isinstance(out, Output)

    def test_finish_reason(self, pipeline):
        out = pipeline.generate(text="hi", max_new_tokens=3, return_output=True)
        assert out.finish_reason in ("eos", "length", None)


class _BothEntryPointsControl(InputControl):
    """Implements BOTH adapt_messages (chat path) and adapt (text/token path), like CPO/GEPA/
    PRewrite/FewShot. Counts invocations so tests can assert exactly-once application."""

    supports_batching = True

    def __init__(self, handle_messages: bool = True):
        super().__init__()
        self.handle_messages = handle_messages
        self.adapt_calls = 0
        self.adapt_messages_calls = 0

    def adapt_messages(self, messages, runtime_kwargs=None):
        self.adapt_messages_calls += 1
        if not self.handle_messages:
            return None
        return [
            [{"role": "system", "content": "injected"}] + list(chat)
            for chat in messages
        ]

    def adapt(self, input_ids, runtime_kwargs=None):
        self.adapt_calls += 1
        return input_ids


class TestInputControlAppliedExactlyOnce:
    """Regression tests for the exactly-once input-control contract."""

    def _make_pipeline(self, control):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[control])
        pipeline.steer()
        return pipeline

    def test_chat_input_skips_token_level_adapt(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        pipeline.generate(messages=[{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert control.adapt_messages_calls == 1
        assert control.adapt_calls == 0

    def test_chat_input_falls_through_when_adapt_messages_returns_none(self):
        control = _BothEntryPointsControl(handle_messages=False)
        pipeline = self._make_pipeline(control)
        pipeline.generate(messages=[{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert control.adapt_messages_calls == 1
        assert control.adapt_calls == 1

    def test_text_input_uses_token_level_adapt_only(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.generate(text="hi", max_new_tokens=2)
        assert control.adapt_messages_calls == 0
        assert control.adapt_calls == 1

    def test_token_input_uses_token_level_adapt_only(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.generate(input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert control.adapt_messages_calls == 0
        assert control.adapt_calls == 1

    def test_bypass_warning_fires_for_text_not_messages(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.generate(text="hi", max_new_tokens=2)
        assert any("adapt_messages" in str(w.message) for w in recorded)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.generate(messages=[{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert not any("adapt_messages" in str(w.message) for w in recorded)

    def test_chat_system_prompt_injected_once(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        out = pipeline.generate(
            messages=[{"role": "user", "content": "hi"}], max_new_tokens=2, return_output=True
        )
        prompt_text = pipeline.tokenizer.decode(
            out.adapted_input_ids[0], skip_special_tokens=True
        )
        assert prompt_text.count("injected") == 1


@pytest.fixture(scope="module")
def tiny_pipeline():
    """Hub-free steered pipeline for the return-semantics regression (CPU-only, offline)."""
    torch.manual_seed(0)
    model = tiny_llama(num_layers=2, hidden=16, heads=2)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


class TestReturnSemantics:
    """Token return is continuation-only by default; `return_full_sequence` includes the prompt.

    Guards against re-introducing the notebook bug of slicing a continuation-only result by prompt
    length (which discards generated tokens).
    """

    def test_default_is_continuation_only(self, tiny_pipeline):
        ids = torch.tensor([[3, 4, 5, 6]])
        prompt_len = ids.size(1)
        k = 5
        cont = tiny_pipeline.generate(input_ids=ids, max_new_tokens=k, do_sample=False)
        full = tiny_pipeline.generate(
            input_ids=ids, max_new_tokens=k, do_sample=False, return_full_sequence=True
        )
        assert full.shape[1] == prompt_len + cont.shape[1]
        assert cont.shape[1] == full.shape[1] - prompt_len
        assert torch.equal(full[:, prompt_len:], cont)

    def test_continuation_length_matches_max_new_tokens(self, tiny_pipeline):
        ids = torch.tensor([[3, 4, 5]])
        for k in (1, 3, 6):
            cont = tiny_pipeline.generate(input_ids=ids, max_new_tokens=k, do_sample=False)
            assert cont.shape[1] == k


class TestChatTemplateKwargs:
    """The reserved `chat_template_kwargs` key threads to `apply_chat_template` and is validated."""

    def test_passthrough_reaches_apply_chat_template(self, pipeline, monkeypatch):
        calls = []
        original = pipeline.tokenizer.apply_chat_template

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(pipeline.tokenizer, "apply_chat_template", spy)
        pipeline.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_new_tokens=2,
            do_sample=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert calls, "apply_chat_template was not called"
        seen = calls[-1][1]
        # the four pipeline-owned kwargs and the forwarded kwarg all arrive
        assert seen["return_tensors"] == "pt"
        assert seen["padding"] is True
        assert seen["add_generation_prompt"] is True
        assert seen["return_dict"] is True
        assert seen["enable_thinking"] is False

    def test_key_popped_before_backend_normalization(self, pipeline, monkeypatch):
        seen_gen_kwargs = {}

        def fake_generate(input_ids, attention_mask=None, **gen_kwargs):
            seen_gen_kwargs.update(gen_kwargs)
            return torch.cat([input_ids, input_ids[:, :1]], dim=1)

        script_session_generate(monkeypatch, fake_generate)
        pipeline.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_new_tokens=2,
            do_sample=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert "chat_template_kwargs" not in seen_gen_kwargs

    def test_empty_dict_is_noop(self, pipeline, monkeypatch):
        calls = []
        original = pipeline.tokenizer.apply_chat_template

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(pipeline.tokenizer, "apply_chat_template", spy)
        pipeline.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_new_tokens=2,
            do_sample=False,
            chat_template_kwargs={},
        )
        assert calls
        # an empty mapping adds nothing beyond the four pipeline-owned kwargs
        seen = calls[-1][1]
        assert set(seen) == {"return_tensors", "padding", "add_generation_prompt", "return_dict"}

    def test_non_mapping_raises_typeerror(self, pipeline):
        with pytest.raises(
            TypeError,
            match=r"chat_template_kwargs must be a mapping of chat-template keyword arguments; got list\.",
        ):
            pipeline.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_new_tokens=1,
                chat_template_kwargs=["enable_thinking"],
            )

    def test_pairing_with_text_raises_typeerror(self, pipeline):
        with pytest.raises(
            TypeError,
            match=r"chat_template_kwargs is only valid with chat input \(messages=\); "
            r"text= and input_ids= are already templated or template-free\.",
        ):
            pipeline.generate(
                text="hi", max_new_tokens=1, chat_template_kwargs={"enable_thinking": False}
            )

    def test_pairing_with_input_ids_raises_typeerror(self, pipeline):
        with pytest.raises(
            TypeError,
            match=r"chat_template_kwargs is only valid with chat input \(messages=\); "
            r"text= and input_ids= are already templated or template-free\.",
        ):
            pipeline.generate(
                input_ids=torch.tensor([[1, 2, 3]]),
                max_new_tokens=1,
                chat_template_kwargs={"enable_thinking": False},
            )

    def test_collision_with_pipeline_owned_kwarg_raises_valueerror(self, pipeline):
        with pytest.raises(
            ValueError,
            match=r"chat_template_kwargs may not override pipeline-owned template arguments: "
            r"add_generation_prompt, padding\.",
        ):
            pipeline.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_new_tokens=1,
                chat_template_kwargs={"padding": False, "add_generation_prompt": False},
            )
