"""Unit tests for the Inspect model provider over a stub pipeline (no eval runs).

Covers message conversion (roles, structured content, reasoning parts dropped, tool and multimodal
refusals), the `GenerateConfig` classification and its full-field coverage, the
unsupported-parameter policy, the sampling rule, the reserved `extra_body` key, finish-reason
mapping, logprob refusal, stop-string truncation, the reasoning split including the unclosed case,
usage counting under padding, the bare-conversation dispatch for `num_choices > 1`, provider
registration and naming, and the text path.
"""
import warnings

import anyio
import pytest

pytest.importorskip("inspect_ai")

import torch
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
    GenerateConfig,
)

from steerability.algorithms.core.output import Output
from steerability.evaluation.provider import (
    MAPPED_FIELDS,
    POLICY_FIELDS,
    REFUSED_FIELDS,
    UPSTREAM_FIELDS,
    ProviderOptions,
    SteeringPipelineModelAPI,
    as_inspect_model,
)
from tests.evaluation.conftest import CHAT_TEMPLATE, StubControl, StubSteeringPipeline, StubTokenizer, make_output
from tests.utils.tiny_models import reasoning_tag_tokenizer

TOKEN_TAGS = ("<open>", "<close>")


def _token_pipeline(special_tags=TOKEN_TAGS, ordinary_tags=()):
    """A stub pipeline backed by a real tag tokenizer, so token-mode splitting sees real ids."""
    tokenizer = reasoning_tag_tokenizer(special_tags=special_tags, ordinary_tags=ordinary_tags)
    tokenizer.chat_template = CHAT_TEMPLATE
    return StubSteeringPipeline(tokenizer=tokenizer)


def _ids(tokenizer, *words):
    """Encode a whitespace-joined sequence of words and tags into a one-row output-id list."""
    ids = []
    for word in words:
        ids.extend(tokenizer.encode(word, add_special_tokens=False))
    return ids


def _output(row_ids):
    """One `Output` carrying a single candidate row of the given ids."""
    return Output(
        output_ids=torch.tensor([row_ids], dtype=torch.long),
        adapted_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        finish_reason="eos",
        finish_reasons=("eos",),
    )


def _api(pipeline=None, **kwargs) -> SteeringPipelineModelAPI:
    pipeline = pipeline if pipeline is not None else StubSteeringPipeline()
    options = kwargs.pop("options", None)
    return SteeringPipelineModelAPI("stub", pipeline=pipeline, options=options, **kwargs)


def _generate(api, messages, config=GenerateConfig(max_tokens=8, temperature=0)):
    async def main():
        return await api.generate(messages, [], "auto", config)
    return anyio.run(main)


class TestRegistrationAndConstruction:
    def test_model_renders_with_registry_prefix(self):
        model = as_inspect_model(StubSteeringPipeline(), model_name="steering-pipeline")
        assert str(model) == "steerability/steering-pipeline"

    def test_missing_pipeline_raises_actionable_error(self):
        with pytest.raises(ValueError, match="as_inspect_model"):
            SteeringPipelineModelAPI("stub")

    def test_unsteered_pipeline_refused(self):
        pipeline = StubSteeringPipeline()
        pipeline._is_steered = False
        with pytest.raises(ValueError, match="steer"):
            _api(pipeline)

    def test_batching_clamp(self):
        pipeline = StubSteeringPipeline(supports_batching=False)
        api = _api(pipeline, options=ProviderOptions(max_batch_size=8))
        assert api.effective_max_batch == 1
        assert api.max_connections() == 1

    def test_prompt_path_and_chat_template_kwargs_refusal_on_text(self):
        pipeline = StubSteeringPipeline(tokenizer=StubTokenizer(chat_template=None))
        with pytest.warns(UserWarning, match="adapt_messages"):
            api = _api(pipeline)
        assert api.prompt_path == "text"
        with pytest.raises(TypeError, match="chat_template_kwargs"):
            _api(pipeline, options=ProviderOptions(chat_template_kwargs={"enable_thinking": False}))

    def test_hooks(self):
        api = _api(options=ProviderOptions(max_batch_size=3, default_max_tokens=77))
        assert api.max_tokens() == 77
        assert api.max_connections() == 3
        assert api.connection_key().startswith("steerability:")
        assert api.should_retry(RuntimeError()) is False
        assert api.is_auth_failure(RuntimeError()) is False
        assert api.tools_required() is False


class TestProviderOptionsValidation:
    def test_bad_values_raise(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            ProviderOptions(max_batch_size=0)
        with pytest.raises(ValueError, match="default_max_tokens"):
            ProviderOptions(default_max_tokens=0)
        with pytest.raises(ValueError, match="reasoning_tags"):
            ProviderOptions(reasoning_tags=("", "</think>"))
        with pytest.raises(ValueError, match="on_unsupported_param"):
            ProviderOptions(on_unsupported_param="ignore")
        with pytest.raises(ValueError, match="seed_scope"):
            ProviderOptions(seed_scope="whole")
        with pytest.raises(TypeError, match="runtime_kwargs"):
            ProviderOptions(runtime_kwargs=[("a", 1)])

    def test_seed_scope_default_is_dispatch(self):
        assert ProviderOptions().seed_scope == "dispatch"


class TestMessageConversion:
    def test_roles_and_structured_content(self):
        api = _api()
        converted = api._convert_input([
            ChatMessageSystem(content="be brief"),
            ChatMessageUser(content=[ContentText(text="hello "), ContentText(text="world")]),
            ChatMessageAssistant(content=[ContentReasoning(reasoning="hmm"), ContentText(text="hi")]),
            ChatMessageUser(content="again"),
        ])
        assert converted == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]

    def test_tool_message_refused(self):
        api = _api()
        tool_message = ChatMessageTool(content="result", tool_call_id="1", function="f")
        with pytest.raises(NotImplementedError, match="tool"):
            api._convert_input([tool_message])

    def test_assistant_tool_calls_refused(self):
        api = _api()
        message = ChatMessageAssistant(content="x")
        message.tool_calls = [object()]
        with pytest.raises(NotImplementedError, match="tool"):
            api._convert_input([message])

    def test_multimodal_content_refused_naming_type(self):
        api = _api()
        with pytest.raises(NotImplementedError, match="ContentImage"):
            api._convert_input([ChatMessageUser(content=[ContentImage(image="x.png")])])

    def test_tools_refused_at_generate(self):
        api = _api()
        async def main():
            return await api.generate([ChatMessageUser(content="q")], [object()], "auto", GenerateConfig())
        with pytest.raises(NotImplementedError, match="tool"):
            anyio.run(main)


class TestGenerateConfigClassification:
    def test_every_field_is_classified_exactly_once(self):
        all_fields = set(GenerateConfig.model_fields)
        union = MAPPED_FIELDS | UPSTREAM_FIELDS | POLICY_FIELDS | REFUSED_FIELDS
        assert union == all_fields, (
            f"unclassified: {sorted(all_fields - union)}; unknown: {sorted(union - all_fields)}"
        )
        classes = [MAPPED_FIELDS, UPSTREAM_FIELDS, POLICY_FIELDS, REFUSED_FIELDS]
        for i, first in enumerate(classes):
            for second in classes[i + 1:]:
                assert not (first & second)

    def test_logprob_fields_always_refused(self):
        api = _api(options=ProviderOptions(on_unsupported_param="warn"))
        for name in ("logprobs", "top_logprobs", "prompt_logprobs"):
            with pytest.raises(NotImplementedError, match="generation-only"):
                api._map_generate_config(GenerateConfig(**{name: True if name == "logprobs" else 1}))

    def test_policy_raise_and_warn(self):
        raising = _api()
        with pytest.raises(ValueError, match="best_of"):
            raising._map_generate_config(GenerateConfig(best_of=4))
        warning = _api(options=ProviderOptions(on_unsupported_param="warn"))
        with pytest.warns(UserWarning, match="best_of"):
            warning._map_generate_config(GenerateConfig(best_of=4))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # second occurrence does not warn again
            warning._map_generate_config(GenerateConfig(best_of=4))

    def test_upstream_fields_ignored(self):
        api = _api()
        gen_kwargs, _, _ = api._map_generate_config(
            GenerateConfig(
                max_retries=3, timeout=10, stream_idle_timeout=5, system_message="s", cache=True, batch=True
            )
        )
        assert gen_kwargs == {}

    def test_reserved_extra_body_key_stripped_before_policy(self):
        api = _api()
        gen_kwargs, per_sample, _ = api._map_generate_config(
            GenerateConfig(extra_body={"runtime_kwargs": {"spans": ["a"]}})
        )
        assert per_sample == {"spans": ["a"]}
        assert "runtime_kwargs" not in gen_kwargs

    def test_other_extra_body_keys_follow_policy(self):
        api = _api()
        with pytest.raises(ValueError, match="extra_body\\['custom'\\]"):
            api._map_generate_config(GenerateConfig(extra_body={"custom": 1}))


class TestSamplingRule:
    def test_greedy_drops_sampling_knobs_and_seed(self):
        api = _api(base_seed=11)
        gen_kwargs, _, _ = api._map_generate_config(
            GenerateConfig(temperature=0, top_p=0.9, top_k=40, seed=5)
        )
        assert gen_kwargs == {"do_sample": False}

    def test_sampling_forwards_knobs_and_prefers_config_seed(self):
        api = _api(base_seed=11)
        gen_kwargs, _, _ = api._map_generate_config(
            GenerateConfig(temperature=0.7, top_p=0.9, top_k=40, seed=5)
        )
        assert gen_kwargs == {
            "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "seed": 5, "seed_scope": "dispatch",
        }

    def test_sampling_falls_back_to_base_seed(self):
        api = _api(base_seed=11)
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(temperature=0.7))
        assert gen_kwargs["seed"] == 11

    def test_sampling_without_any_seed_attaches_none(self):
        api = _api()
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(temperature=0.7))
        assert "seed" not in gen_kwargs

    def test_unset_temperature_attaches_nothing(self):
        api = _api(base_seed=11)
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(top_p=0.9, seed=5))
        assert gen_kwargs == {}

    def test_max_tokens_and_stop_seqs_map(self):
        api = _api()
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(max_tokens=32, stop_seqs=["END"]))
        assert gen_kwargs == {"max_new_tokens": 32, "stop_strings": ("END",)}

    def test_seed_scope_attached_with_seed_from_options(self):
        api = _api(base_seed=11, options=ProviderOptions(seed_scope="item"))
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(temperature=0.7, seed=5))
        assert gen_kwargs["seed_scope"] == "item"

    def test_seed_scope_defaults_to_dispatch(self):
        api = _api(base_seed=11)
        gen_kwargs, _, _ = api._map_generate_config(GenerateConfig(temperature=0.7))
        assert gen_kwargs["seed_scope"] == "dispatch"

    def test_seed_scope_absent_on_greedy_and_unseeded(self):
        greedy = _api(base_seed=11)
        gen_kwargs, _, _ = greedy._map_generate_config(GenerateConfig(temperature=0, seed=5))
        assert "seed_scope" not in gen_kwargs
        unseeded = _api()
        gen_kwargs, _, _ = unseeded._map_generate_config(GenerateConfig(temperature=0.7))
        assert "seed_scope" not in gen_kwargs


class TestOutputAssembly:
    def test_finish_reason_mapping(self):
        api = _api(options=ProviderOptions(reasoning_tags=None))
        for reason, expected in (("length", "max_tokens"), ("stop", "stop"), ("eos", "stop"), (None, "unknown")):
            output = make_output([[5]], [1, 2], (reason,))
            model_output = api._assemble_model_output(output, stop_strings=())
            assert model_output.choices[0].stop_reason == expected

    def test_stop_string_truncation_applies_to_decoded_text(self):
        pipeline = StubSteeringPipeline(decode_texts=["keep END drop"])
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=None))
        output = make_output([[0]], [1, 2], ("stop",))
        model_output = api._assemble_model_output(output, stop_strings=("END",))
        assert model_output.completion == "keep "

    def test_reasoning_split(self):
        pipeline = StubSteeringPipeline(decode_texts=["<think>plan</think> answer"])
        api = _api(pipeline)
        model_output = api._assemble_model_output(make_output([[0]], [1, 2], ("eos",)), stop_strings=())
        content = model_output.choices[0].message.content
        assert isinstance(content, list)
        assert isinstance(content[0], ContentReasoning) and content[0].reasoning == "plan"
        assert isinstance(content[1], ContentText) and content[1].text == "answer"

    def test_unclosed_reasoning_yields_empty_answer_and_warns(self, caplog):
        pipeline = StubSteeringPipeline(decode_texts=["<think>still thinking"])
        api = _api(pipeline)
        with caplog.at_level("WARNING", logger="steerability.evaluation.provider"):
            model_output = api._assemble_model_output(make_output([[0]], [1, 2], (None,)), stop_strings=())
        content = model_output.choices[0].message.content
        assert content[0].reasoning == "still thinking"
        assert content[1].text == ""
        assert any("thinking" in record.getMessage() for record in caplog.records)

    def test_reasoning_split_disabled(self):
        pipeline = StubSteeringPipeline(decode_texts=["<think>plan</think> answer"])
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=None))
        model_output = api._assemble_model_output(make_output([[0]], [1, 2], ("eos",)), stop_strings=())
        assert model_output.choices[0].message.content == "<think>plan</think> answer"

    def test_usage_counts_non_pad_positions(self):
        api = _api(options=ProviderOptions(reasoning_tags=None))
        output = make_output([[5, 6, 0, 0], [7, 0, 0, 0]], [1, 2, 0], ("eos", "eos"))
        model_output = api._assemble_model_output(output, stop_strings=())
        assert model_output.usage.input_tokens == 2
        assert model_output.usage.output_tokens == 3
        assert model_output.usage.total_tokens == 5
        assert model_output.metadata["returned_output_tokens"] == 3

    def test_generated_tokens_drives_output_usage_when_present(self):
        api = _api(options=ProviderOptions(reasoning_tags=None))
        output = make_output([[5, 6, 0, 0]], [1, 2, 0], ("eos",))
        output.generated_tokens = 40  # a driver rolled out far more than it returned
        model_output = api._assemble_model_output(output, stop_strings=())
        assert model_output.usage.output_tokens == 40
        assert model_output.usage.total_tokens == 42
        # the returned continuation count stays available for scoring/truncation analysis
        assert model_output.metadata["returned_output_tokens"] == 2

    def test_returned_count_used_when_generated_tokens_absent(self):
        api = _api(options=ProviderOptions(reasoning_tags=None))
        output = make_output([[5, 6, 0, 0]], [1, 2, 0], ("eos",))
        model_output = api._assemble_model_output(output, stop_strings=())
        assert output.generated_tokens is None
        assert model_output.usage.output_tokens == 2
        assert model_output.metadata["returned_output_tokens"] == 2


class TestReasoningSplitModes:
    """The token-mode split path and `"auto"` resolution against the pipeline tokenizer."""

    def _reasoning_and_answer(self, content):
        assert isinstance(content, list) and isinstance(content[0], ContentReasoning)
        assert isinstance(content[1], ContentText)
        return content[0].reasoning, content[1].text

    def test_auto_resolves_ordinary_tags_to_text(self):
        api = _api(_token_pipeline(special_tags=(), ordinary_tags=TOKEN_TAGS),
                   options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        assert api._reasoning_split == "text"

    def test_auto_resolves_special_tags_to_tokens(self):
        api = _api(_token_pipeline(), options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        assert api._reasoning_split == "tokens"

    def test_explicit_mode_overrides_auto(self):
        api = _api(_token_pipeline(), options=ProviderOptions(reasoning_tags=TOKEN_TAGS, reasoning_split="text"))
        assert api._reasoning_split == "text"

    def test_none_tags_leave_no_resolved_mode(self):
        api = _api(_token_pipeline(), options=ProviderOptions(reasoning_tags=None))
        assert api._reasoning_split is None

    def test_token_mode_case_i_splits_reasoning_from_answer(self):
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        output = _output(_ids(pipeline.tokenizer, "<open>", "R", "<close>", "A"))
        content = api._assemble_model_output(output, stop_strings=()).choices[0].message.content
        reasoning, answer = self._reasoning_and_answer(content)
        assert reasoning == "R" and answer == "A"
        assert "<open>" not in answer and "<close>" not in answer

    def test_token_mode_close_only_with_opened_at_start(self):
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS, reasoning_opened_at_start=True))
        output = _output(_ids(pipeline.tokenizer, "R", "<close>", "A"))
        model_output = api._assemble_model_output(output, stop_strings=())
        reasoning, answer = self._reasoning_and_answer(model_output.choices[0].message.content)
        assert reasoning == "R" and answer == "A"
        assert model_output.completion == "A"

    def test_token_mode_close_only_without_opened_at_start(self):
        # the close subsequence alone splits the row, as in text mode; the flag matters only for a
        # continuation carrying neither tag
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        output = _output(_ids(pipeline.tokenizer, "R", "<close>", "A"))
        model_output = api._assemble_model_output(output, stop_strings=())
        reasoning, answer = self._reasoning_and_answer(model_output.choices[0].message.content)
        assert reasoning == "R" and answer == "A"
        assert model_output.completion == "A"

    def test_token_mode_unclosed_yields_empty_answer_and_warns(self, caplog):
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS, reasoning_opened_at_start=True))
        output = _output(_ids(pipeline.tokenizer, "R", "plan"))
        with caplog.at_level("WARNING", logger="steerability.evaluation.provider"):
            content = api._assemble_model_output(output, stop_strings=()).choices[0].message.content
        reasoning, answer = self._reasoning_and_answer(content)
        assert reasoning == "R plan" and answer == ""
        assert any("thinking" in record.getMessage() for record in caplog.records)

    def test_token_mode_closed_with_empty_answer_does_not_warn(self, caplog):
        # a channel that closes with no following answer is closed, not unclosed: no warning
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        output = _output(_ids(pipeline.tokenizer, "<open>", "R", "<close>"))
        with caplog.at_level("WARNING", logger="steerability.evaluation.provider"):
            content = api._assemble_model_output(output, stop_strings=()).choices[0].message.content
        reasoning, answer = self._reasoning_and_answer(content)
        assert reasoning == "R" and answer == ""
        assert not any("thinking" in record.getMessage() for record in caplog.records)

    def test_token_mode_stop_string_truncates_answer_only(self):
        # (t1): reasoning is preserved verbatim; only the answer segment is truncated at the stop string
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS))
        output = _output(_ids(pipeline.tokenizer, "<open>", "R", "<close>", "A", "stop", "x"))
        content = api._assemble_model_output(output, stop_strings=("stop",)).choices[0].message.content
        reasoning, answer = self._reasoning_and_answer(content)
        assert reasoning == "R"
        assert answer.split() == ["A"]
        assert "stop" not in answer

    def test_token_mode_reasoning_never_stop_truncated(self):
        # (t2): halted mid-channel with the stop text at the tail; reasoning keeps it, answer empty
        pipeline = _token_pipeline()
        api = _api(pipeline, options=ProviderOptions(reasoning_tags=TOKEN_TAGS, reasoning_opened_at_start=True))
        output = _output(_ids(pipeline.tokenizer, "R", "plan", "stop"))
        content = api._assemble_model_output(output, stop_strings=("stop",)).choices[0].message.content
        reasoning, answer = self._reasoning_and_answer(content)
        assert reasoning.split() == ["R", "plan", "stop"]
        assert answer == ""

    def test_empty_encoding_tag_raises_at_construction(self):
        pipeline = _token_pipeline()
        with pytest.raises(ValueError, match="empty id sequence"):
            _api(pipeline, options=ProviderOptions(reasoning_tags=("<open>", "   "), reasoning_split="auto"))


class TestDispatchShapes:
    def test_multi_candidate_dispatches_bare_conversation(self):
        pipeline = StubSteeringPipeline()
        api = _api(pipeline)
        out = _generate(
            api, [ChatMessageUser(content="q")],
            GenerateConfig(max_tokens=4, temperature=0, num_choices=3),
        )
        assert len(out.choices) == 3
        (call,) = pipeline.calls
        assert isinstance(call["messages"][0], dict)  # one conversation, not a batch
        assert call["gen_kwargs"]["n"] == 3

    def test_single_request_dispatches_as_batch_of_one(self):
        pipeline = StubSteeringPipeline()
        api = _api(pipeline)
        out = _generate(api, [ChatMessageUser(content="q")])
        assert len(out.choices) == 1
        (call,) = pipeline.calls
        assert isinstance(call["messages"][0], list)  # a batch of conversations

    def test_text_path_renders_conversation(self):
        pipeline = StubSteeringPipeline(tokenizer=StubTokenizer(chat_template=None))
        with pytest.warns(UserWarning, match="adapt_messages"):
            api = _api(pipeline)
        _generate(api, [ChatMessageSystem(content="be brief"), ChatMessageUser(content="hello")])
        (call,) = pipeline.calls
        assert call["messages"] is None
        assert call["text"] == ["be brief\n\nhello"]

    def test_static_runtime_kwargs_pass_through_unmutated(self):
        control = StubControl([{"name": "canned_responses", "scope": "call"}])
        pipeline = StubSteeringPipeline(controls=(control,))
        artifact = {"routes": {"a": "b"}}
        api = _api(pipeline, options=ProviderOptions(runtime_kwargs={"canned_responses": artifact}))
        _generate(api, [ChatMessageUser(content="q")])
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["canned_responses"] is artifact

    def test_per_sample_kwarg_without_consumer_is_inert(self):
        pipeline = StubSteeringPipeline()
        api = _api(pipeline)
        _generate(
            api, [ChatMessageUser(content="q")],
            GenerateConfig(max_tokens=4, temperature=0, extra_body={"runtime_kwargs": {"substrings": ["x"]}}),
        )
        (call,) = pipeline.calls
        assert "substrings" not in call["runtime_kwargs"]
        assert api.inert_runtime_kwargs == frozenset({"substrings"})

    def test_per_sample_kwarg_with_row_consumer_is_collated(self):
        control = StubControl([{"name": "substrings", "type": "list[str]", "scope": "row"}])
        pipeline = StubSteeringPipeline(controls=(control,))
        api = _api(pipeline)
        _generate(
            api, [ChatMessageUser(content="q")],
            GenerateConfig(max_tokens=4, temperature=0, extra_body={"runtime_kwargs": {"substrings": ["x"]}}),
        )
        (call,) = pipeline.calls
        assert call["runtime_kwargs"]["substrings"] == [["x"]]

    def test_close_refuses_new_requests(self):
        api = _api()
        api.close()
        with pytest.raises(RuntimeError, match="closed"):
            _generate(api, [ChatMessageUser(content="q")])
