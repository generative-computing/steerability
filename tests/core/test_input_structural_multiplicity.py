"""Input- and structural-control multiplicity in `SteeringPipeline`.

Covers the relaxed one-per-category rule for the input and structural categories: input controls
chain in list order across two phases (message-level fold on chat input, then token-level chain),
structural controls thread the model through `steer()` in list order, the post-steer tokenizer
fallback scans `out_path` backwards, the adapt-messages bypass warning names each bypassed control,
and `steer()` warns on overlapping `RUNTIME_KWARGS_SCHEMA` names.

Runs hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer.
"""
import logging
import warnings
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.state_control.base import StateControl
from steerability.algorithms.structural_control.base import StructuralControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

# renders message contents joined by spaces so WordLevel vocab words map to stable ids
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"

# WordLevel vocab ids used as markers: the=3, cat=4, dog=8, attention=11, span=12
THE, CAT, DOG, ATTENTION, SPAN = 3, 4, 8, 11, 12


class _AppendTokenControl(InputControl):
    """Token-only input control; appends a fixed marker token id in `adapt`."""

    Args = None
    supports_batching = True

    def __init__(self, marker_id: int):
        super().__init__()
        self.marker_id = marker_id
        self.adapt_calls = 0
        self.last_output = None

    def adapt(self, input_ids, runtime_kwargs=None):
        self.adapt_calls += 1
        if isinstance(input_ids, list):
            adapted = input_ids + [self.marker_id]
        else:
            marker = torch.full(
                input_ids.shape[:-1] + (1,), self.marker_id, dtype=input_ids.dtype, device=input_ids.device
            )
            adapted = torch.cat([input_ids, marker], dim=-1)
        self.last_output = adapted
        return adapted


class _MessageControl(InputControl):
    """Message-level input control; prepends a system message when `handle` is True."""

    Args = None
    supports_batching = True

    def __init__(self, content: str = "attention", handle: bool = True):
        super().__init__()
        self.content = content
        self.handle = handle
        self.adapt_calls = 0
        self.adapt_messages_calls = 0
        self.seen_messages = None

    def adapt_messages(self, messages, runtime_kwargs=None):
        self.adapt_messages_calls += 1
        self.seen_messages = [list(chat) for chat in messages]
        if not self.handle:
            return None
        return [[{"role": "system", "content": self.content}] + list(chat) for chat in messages]

    def adapt(self, input_ids, runtime_kwargs=None):
        self.adapt_calls += 1
        return input_ids


class _StageStructuralControl(StructuralControl):
    """Structural control returning a fresh sentinel module; records the model it received."""

    Args = None

    def __init__(self):
        super().__init__()
        self.received_model = None
        self.returned_model = None

    def steer(self, model=None, tokenizer=None, **kwargs):
        self.received_model = model
        self.returned_model = nn.Linear(2, 2)
        return self.returned_model


class _OutPathStructuralControl(StructuralControl):
    """Pass-through structural control optionally exposing `args.out_path`."""

    Args = None

    def __init__(self, out_path: str | None = None):
        super().__init__()
        if out_path is not None:
            self.args = MagicMock(out_path=out_path)

    def steer(self, model=None, tokenizer=None, **kwargs):
        return model


def _pipeline(controls, model=None, tokenizer=None):
    """Build a steered lazy-init pipeline on a tiny model with a chat-templated tokenizer."""
    if model is None:
        model = tiny_llama()
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
        tokenizer.chat_template = CHAT_TEMPLATE
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


GEN_KWARGS = dict(max_new_tokens=1, do_sample=False, eos_token_id=None)


# input chaining (token phase)
class TestInputChainOrder:
    def test_list_order_defines_token_chain(self):
        model = tiny_llama()
        a, b = _AppendTokenControl(THE), _AppendTokenControl(CAT)
        pipeline_ab = _pipeline([a, b], model=model)
        out_ab = pipeline_ab.generate(input_ids=torch.tensor([5, 6]), return_output=True, **GEN_KWARGS)
        assert out_ab.adapted_input_ids[0, -2:].tolist() == [THE, CAT]
        assert a.adapt_calls == 1
        assert b.adapt_calls == 1

        c, d = _AppendTokenControl(THE), _AppendTokenControl(CAT)
        pipeline_ba = _pipeline([d, c], model=model)
        out_ba = pipeline_ba.generate(input_ids=torch.tensor([5, 6]), return_output=True, **GEN_KWARGS)
        assert out_ba.adapted_input_ids[0, -2:].tolist() == [CAT, THE]
        assert out_ab.adapted_input_ids[0].tolist() != out_ba.adapted_input_ids[0].tolist()


# mixed phases on chat input
class TestMixedPhasesChat:
    @pytest.mark.parametrize("message_first", [True, False])
    def test_each_control_applied_once_per_phase(self, message_first):
        message_ctrl = _MessageControl(content="attention")
        token_ctrl = _AppendTokenControl(CAT)
        controls = [message_ctrl, token_ctrl] if message_first else [token_ctrl, message_ctrl]
        pipeline = _pipeline(controls)

        out = pipeline.generate(
            messages=[{"role": "user", "content": "the dog"}], return_output=True, **GEN_KWARGS
        )
        adapted = out.adapted_input_ids[0].tolist()

        assert message_ctrl.adapt_messages_calls == 1
        assert message_ctrl.adapt_calls == 0
        assert token_ctrl.adapt_calls == 1
        # the message edit is present, and lands before the token-phase append even for [T, M]
        assert ATTENTION in adapted
        assert adapted.index(ATTENTION) < adapted.index(CAT)


# message-phase fold
class TestMessagePhaseFold:
    def test_second_control_sees_first_controls_output(self):
        first = _MessageControl(content="attention")
        second = _MessageControl(content="span")
        pipeline = _pipeline([first, second])

        out = pipeline.generate(messages=[{"role": "user", "content": "cat"}], return_output=True, **GEN_KWARGS)

        assert second.seen_messages[0][0] == {"role": "system", "content": "attention"}
        assert first.adapt_calls == 0
        assert second.adapt_calls == 0
        adapted = out.adapted_input_ids[0].tolist()
        assert ATTENTION in adapted
        assert SPAN in adapted

    def test_none_return_falls_through_to_token_phase(self):
        declining = _MessageControl(content="attention", handle=False)
        pipeline = _pipeline([declining])

        pipeline.generate(messages=[{"role": "user", "content": "cat"}], **GEN_KWARGS)

        assert declining.adapt_messages_calls == 1
        assert declining.adapt_calls == 1


# bypass warning on text/tensor input
class TestBypassWarning:
    def test_names_only_message_level_controls(self):
        message_ctrl = _MessageControl(content="attention")
        token_ctrl = _AppendTokenControl(CAT)
        pipeline = _pipeline([message_ctrl, token_ctrl])

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.generate(input_ids=torch.tensor([[5, 6]]), **GEN_KWARGS)

        bypass = [w for w in recorded if "adapt_messages" in str(w.message)]
        assert len(bypass) == 1
        assert "_MessageControl" in str(bypass[0].message)
        assert "_AppendTokenControl" not in str(bypass[0].message)


# structural threading
class TestStructuralThreading:
    def test_model_threads_through_stages_in_list_order(self):
        base_model = tiny_llama()
        stage1, stage2 = _StageStructuralControl(), _StageStructuralControl()
        pipeline = SteeringPipeline(controls=[stage1, stage2], model=base_model, tokenizer=wordlevel_tokenizer())
        pipeline.steer()

        assert stage1.received_model is base_model
        assert stage2.received_model is stage1.returned_model
        assert pipeline.model is stage2.returned_model

    def test_order_matters(self):
        base_model = tiny_llama()
        stage1, stage2 = _StageStructuralControl(), _StageStructuralControl()
        pipeline = SteeringPipeline(controls=[stage2, stage1], model=base_model, tokenizer=wordlevel_tokenizer())
        pipeline.steer()

        assert stage2.received_model is base_model
        assert stage1.received_model is stage2.returned_model
        assert pipeline.model is stage1.returned_model


# out_path fallback
class TestOutPathBackwardsScan:
    def test_last_control_with_out_path_wins(self, caplog):
        first = _OutPathStructuralControl(out_path="path/one")
        second = _OutPathStructuralControl(out_path="path/two")
        third = _OutPathStructuralControl(out_path=None)
        pipeline = SteeringPipeline(controls=[first, second, third])

        with caplog.at_level(logging.INFO, logger="steerability.algorithms.core.steering_pipeline"):
            resolved = pipeline._structural_out_path()

        assert str(resolved) == "path/two"
        assert "Multiple structural controls define out_path" in caplog.text

    def test_no_out_path_returns_none(self):
        pipeline = SteeringPipeline(controls=[_OutPathStructuralControl()])
        assert pipeline._structural_out_path() is None

    def test_unresolvable_tokenizer_raises(self):
        pipeline = SteeringPipeline(model=tiny_llama())  # blank name_or_path, no out_path, no tokenizer
        with pytest.raises(RuntimeError, match="Failed to resolve tokenizer"):
            pipeline.steer()


# compute_logprobs parity with the generate chain
class TestComputeLogprobsParity:
    @pytest.mark.parametrize("batched", [True, False])
    def test_full_chain_runs_and_matches_generate_text_path(self, batched):
        model = tiny_llama()
        tokenizer = wordlevel_tokenizer()
        tokenizer.chat_template = CHAT_TEMPLATE

        a, b = _AppendTokenControl(THE), _AppendTokenControl(CAT)
        if not batched:
            b.supports_batching = False
        pipeline = _pipeline([a, b], model=model, tokenizer=tokenizer)
        assert pipeline.supports_batching is batched

        text = "cat sat on"
        prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"]

        pipeline.compute_logprobs(input_ids=prompt_ids.clone(), ref_output_ids=torch.tensor([[8, 9]]))
        assert a.adapt_calls == 1
        assert b.adapt_calls == 1
        logprobs_chain = b.last_output.clone()

        out = pipeline.generate(text, return_output=True, **GEN_KWARGS)
        assert torch.equal(out.adapted_input_ids.cpu(), logprobs_chain.cpu())


# runtime_kwargs overlap warning at steer()
class _SchemaInputControl(_AppendTokenControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "substrings", "type": "list[str]", "required": False, "help": "shared key"},
    ]


class _SchemaStateControl(StateControl):
    Args = None
    supports_batching = True
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "substrings", "type": "list[str]", "required": False, "help": "shared key"},
    ]

    def __init__(self):
        super().__init__()
        self.hooks = {"pre": [], "forward": [], "backward": []}
        self.registered = []

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        return {"pre": [], "forward": [], "backward": []}


class _DistinctSchemaStateControl(_SchemaStateControl):
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "other_key", "type": "list[str]", "required": False, "help": "distinct key"},
    ]


class TestRuntimeKwargsOverlapWarning:
    def test_shared_name_warns_once_naming_both_controls(self):
        controls = [_SchemaInputControl(THE), _SchemaStateControl()]
        pipeline = SteeringPipeline(controls=controls, model=tiny_llama(), tokenizer=wordlevel_tokenizer())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.steer()

        overlap = [w for w in recorded if "runtime_kwargs" in str(w.message)]
        assert len(overlap) == 1
        message = str(overlap[0].message)
        assert "substrings" in message
        assert "_SchemaInputControl" in message
        assert "_SchemaStateControl" in message

    def test_distinct_names_do_not_warn(self):
        controls = [_SchemaInputControl(THE), _DistinctSchemaStateControl()]
        pipeline = SteeringPipeline(controls=controls, model=tiny_llama(), tokenizer=wordlevel_tokenizer())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.steer()

        assert not [w for w in recorded if "runtime_kwargs" in str(w.message)]

    def test_disabled_control_excluded(self):
        disabled = _SchemaStateControl()
        disabled.enabled = False
        controls = [_SchemaInputControl(THE), disabled]
        pipeline = SteeringPipeline(controls=controls, model=tiny_llama(), tokenizer=wordlevel_tokenizer())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.steer()

        assert not [w for w in recorded if "runtime_kwargs" in str(w.message)]
