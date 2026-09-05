"""Inspect AI model provider over a steered `SteeringPipeline`.

`as_inspect_model` wraps a steered pipeline as a generation-only Inspect model. Every request flows
through `pipeline.generate()`: on the messages path prompts enter as structured chat messages, so
the pipeline owns chat templating and message-phase input controls fire exactly as in deployment;
tokenizers without a chat template evaluate through the text path instead. Concurrent Inspect
requests collate into batched pipeline calls through the lock-leader collator in `batching`.

The provider is generation-only by design. Logprob-based scoring (including prompt-logprob
perplexity) is structurally unsupported, since a pipeline's decoding-time controls are not
representable in teacher-forced likelihoods; tool calling and multimodal content are unsupported.
Unsupported requests raise at admission with an actionable message.
"""
from steerability.utils.optional import require

require("inspect_ai")
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)

from steerability.algorithms.core.output import Output, truncate_at_stop_strings
from steerability.algorithms.core.utils.controls import runtime_kwargs_schema
from steerability.evaluation.batching import LockLeaderCollator
from steerability.utils.rendering import has_chat_template, render_messages
from steerability.utils.thinking import (
    DEFAULT_THINK_TAGS,
    ThinkingSplit,
    find_subsequence,
    resolve_split_mode,
    split_thinking,
    split_thinking_ids,
)

if TYPE_CHECKING:
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

logger = logging.getLogger(__name__)

# every GenerateConfig field belongs to exactly one class; a unit test pins the coverage so a new
# Inspect field fails at test time instead of being silently dropped
MAPPED_FIELDS: frozenset[str] = frozenset({
    "max_tokens", "stop_seqs", "temperature", "top_p", "top_k", "seed", "num_choices", "extra_body",
})
UPSTREAM_FIELDS: frozenset[str] = frozenset({
    "system_message", "max_retries", "timeout", "attempt_timeout", "stream_idle_timeout", "max_connections",
    "adaptive_connections", "fallback_models", "cache", "cache_prompt", "batch",
    "parallel_tool_calls", "internal_tools", "max_tool_output",
})
POLICY_FIELDS: frozenset[str] = frozenset({
    "best_of", "frequency_penalty", "presence_penalty", "logit_bias", "response_schema",
    "reasoning_effort", "reasoning_tokens", "reasoning_summary", "reasoning_history",
    "reasoning_mode", "verbosity", "effort", "extra_headers", "modalities",
})
REFUSED_FIELDS: frozenset[str] = frozenset({"logprobs", "top_logprobs", "prompt_logprobs"})

RUNTIME_KWARGS_EXTRA_BODY_KEY = "runtime_kwargs"

_STOP_REASON_MAP: dict[str | None, str] = {"length": "max_tokens", "stop": "stop", "eos": "stop"}


@dataclass(frozen=True, slots=True)
class ProviderOptions:
    """Description of one steering-pipeline provider.

    Both `InspectSuite.run` and `SteeringEval` accept it, so the two surfaces cannot drift.

    Attributes:
        runtime_kwargs: Static runtime kwargs applied to every dispatch. A `"call"`-scoped key
            passes through unchanged; a `"row"`-scoped key's value is one row's value in the
            consuming control's per-row form and is broadcast to every row; a key that no enabled
            control declares is inert. Correct for catalog tasks whose datasets carry no steering
            columns, and for any kwarg that is a property of the arm rather than the sample.
        chat_template_kwargs: Forwarded to `apply_chat_template` on the messages path; must be
            None when the tokenizer has no chat template.
        max_batch_size: Collator dispatch ceiling; clamped to 1 when the pipeline does not
            support batching.
        default_max_tokens: Served through `ModelAPI.max_tokens()` when a request sets no
            `max_tokens`.
        reasoning_tags: `(open_tag, close_tag)` pair splitting each generation into a reasoning
            part and an answer part, so scorers grade the answer only; None disables the split.
        reasoning_opened_at_start: Whether the chat template's generation prompt already opened the
            reasoning channel (some thinking-mode templates emit the open tag in the generation
            prompt, as a `<think>\n` prompt tail). A close-only continuation splits into reasoning
            and answer in either mode without the flag; the flag matters only for a continuation
            carrying neither tag (reasoning truncated before the close), which is classified as
            unclosed reasoning when set and as a plain answer otherwise.
        reasoning_split: How the split locates the delimiters, resolved once against the pipeline
            tokenizer. `"text"` splits substrings on the decoded continuation; `"tokens"` splits the
            continuation ids and decodes each segment; `"auto"` (default) picks `"text"` when both
            tags survive `skip_special_tokens=True` under the tokenizer and `"tokens"` otherwise,
            which is the mode that keeps delimiters encoded as special tokens from being stripped
            before the split can see them.
        on_unsupported_param: `"raise"` (default) rejects a request carrying a `GenerateConfig`
            parameter the pipeline surface cannot honor; `"warn"` warns once per parameter per
            provider and ignores it. Silent dropping is not allowed.
        seed_scope: How a seeded sampling dispatch maps a seed onto its items, forwarded to
            `pipeline.generate()`. The default `"dispatch"` decodes a seeded batch in one pass on
            the Hugging Face backend (the batch is reproducible as a whole); `"item"` derives a
            seed per row and decodes rows one at a time. Under the collator per-item
            reproducibility is already unattainable, so `"item"` only serializes the batch without
            protecting anything. Inert on the vLLM backends.
    """

    runtime_kwargs: Mapping[str, Any] = field(default_factory=dict)
    chat_template_kwargs: Mapping[str, Any] | None = None
    max_batch_size: int = 8
    default_max_tokens: int = 1024
    reasoning_tags: tuple[str, str] | None = DEFAULT_THINK_TAGS
    reasoning_opened_at_start: bool = False
    reasoning_split: Literal["auto", "text", "tokens"] = "auto"
    on_unsupported_param: Literal["raise", "warn"] = "raise"
    seed_scope: Literal["item", "dispatch"] = "dispatch"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_kwargs, Mapping):
            raise TypeError(f"runtime_kwargs must be a mapping; got {type(self.runtime_kwargs).__name__}.")
        if self.chat_template_kwargs is not None and not isinstance(self.chat_template_kwargs, Mapping):
            raise TypeError(
                f"chat_template_kwargs must be a mapping or None; got {type(self.chat_template_kwargs).__name__}."
            )
        if int(self.max_batch_size) < 1:
            raise ValueError(f"max_batch_size must be >= 1; got {self.max_batch_size}.")
        if int(self.default_max_tokens) < 1:
            raise ValueError(f"default_max_tokens must be >= 1; got {self.default_max_tokens}.")
        if self.reasoning_tags is not None:
            open_tag, close_tag = self.reasoning_tags
            if not (isinstance(open_tag, str) and open_tag and isinstance(close_tag, str) and close_tag):
                raise ValueError("reasoning_tags must be a pair of non-empty strings, or None.")
        if self.reasoning_split not in ("auto", "text", "tokens"):
            raise ValueError(
                f"reasoning_split must be 'auto', 'text', or 'tokens'; got {self.reasoning_split!r}."
            )
        if self.on_unsupported_param not in ("raise", "warn"):
            raise ValueError(f"on_unsupported_param must be 'raise' or 'warn'; got {self.on_unsupported_param!r}.")
        if self.seed_scope not in ("item", "dispatch"):
            raise ValueError(f"seed_scope must be 'item' or 'dispatch'; got {self.seed_scope!r}.")


@modelapi(name="steerability")
class SteeringPipelineModelAPI(ModelAPI):
    """Generation-only Inspect `ModelAPI` over an in-process steered `SteeringPipeline`.

    Constructed through `as_inspect_model`; the registry call convention is accepted so
    `ModelName` resolution works, but the provider cannot be built from a model name string.

    Args:
        model_name: Bare model name; Inspect renders the model as `steerability/<model_name>`.
        base_url: Accepted for the registry call convention; unused.
        api_key: Accepted for the registry call convention; unused.
        api_key_vars: Accepted for the registry call convention; unused.
        config: Accepted for the registry call convention; per-request configs arrive at
            `generate()`.
        pipeline: The steered `SteeringPipeline` to serve.
        options: Provider options; defaults to `ProviderOptions()`.
        base_seed: Attached to sampling dispatches whose `GenerateConfig` carries no seed.
        **model_args: Accepted for the registry call convention; unused.

    Raises:
        ValueError: If `pipeline` is None (the provider wraps an in-process pipeline and cannot
            be built from a model name; construct it through `as_inspect_model`), or the pipeline
            is not steered.
        TypeError: If `chat_template_kwargs` is set while the tokenizer has no chat template
            (the text path cannot take it).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        *,
        pipeline: "SteeringPipeline | None" = None,
        options: ProviderOptions | None = None,
        base_seed: int | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, config=GenerateConfig())
        if pipeline is None:
            raise ValueError(
                "The 'steerability' provider wraps an in-process SteeringPipeline and cannot be built from "
                "a model name. Construct it through steerability.evaluation.provider.as_inspect_model."
            )
        if not pipeline._is_steered:
            raise ValueError(
                "The pipeline is not steered. Call pipeline.steer() before as_inspect_model(pipeline)."
            )
        self._pipeline = pipeline
        self._options = options if options is not None else ProviderOptions()
        self._base_seed = base_seed
        self._warned_params: set[str] = set()

        self._prompt_path: Literal["messages", "text"] = (
            "messages" if has_chat_template(pipeline.tokenizer) else "text"
        )
        if self._prompt_path == "text":
            if self._options.chat_template_kwargs is not None:
                raise TypeError(
                    "chat_template_kwargs was set but the tokenizer has no chat template; the text "
                    "path cannot take it. Remove the option or use a chat-templated tokenizer."
                )
            warnings.warn(
                "The pipeline tokenizer has no chat template, so prompts are rendered to plain text "
                "and adapt_messages does not fire on this provider (token-level adapt still does).",
                UserWarning,
            )

        self._reasoning_split: Literal["text", "tokens"] | None = None
        self._close_ids: list[int] = []
        if self._options.reasoning_tags is not None:
            if self._options.reasoning_split == "auto":
                self._reasoning_split = resolve_split_mode(pipeline.tokenizer, self._options.reasoning_tags)
            else:
                self._reasoning_split = self._options.reasoning_split
            if self._reasoning_split == "tokens":
                self._close_ids = pipeline.tokenizer.encode(
                    self._options.reasoning_tags[1], add_special_tokens=False
                )

        effective_max_batch = self._options.max_batch_size if pipeline.supports_batching else 1
        if effective_max_batch != self._options.max_batch_size:
            logger.info(
                "Pipeline does not support batching; clamping max_batch_size from %d to 1.",
                self._options.max_batch_size,
            )
        self._effective_max_batch = effective_max_batch

        schema = runtime_kwargs_schema(pipeline.controls)
        declared_scopes = {name: entry["scope"] for name, entry in schema.items()}
        self._collator = LockLeaderCollator(
            pipeline,
            max_batch_size=effective_max_batch,
            prompt_path=self._prompt_path,
            declared_scopes=declared_scopes,
            static_runtime_kwargs=self._options.runtime_kwargs,
            chat_template_kwargs=self._options.chat_template_kwargs,
        )

    @property
    def prompt_path(self) -> Literal["messages", "text"]:
        """`"messages"` when the tokenizer has a chat template, `"text"` otherwise."""
        return self._prompt_path

    @property
    def effective_max_batch(self) -> int:
        """The collator's dispatch ceiling after the batching clamp."""
        return self._effective_max_batch

    @property
    def inert_runtime_kwargs(self) -> frozenset[str]:
        """Runtime kwargs delivered on either tier that no enabled control of the pipeline declares."""
        return self._collator.inert_runtime_kwargs

    async def generate(
        self,
        input: list[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool],
        tools: list,
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Serve one Inspect request through the collated pipeline.

        Raises:
            NotImplementedError: If the request supplies tools, tool messages, multimodal content,
                or logprob parameters.
            ValueError: If a `GenerateConfig` parameter cannot be honored (under the `"raise"`
                policy), or a per-sample runtime kwarg is also supplied statically or is declared
                `"call"`-scoped.
            RuntimeError: If the provider is closed.
        """
        if tools:
            raise NotImplementedError(
                "This task supplies tools, but steering pipelines have no tool-use convention; "
                "agentic evaluation is unsupported. Choose a non-agentic task."
            )
        messages = self._convert_input(input)
        gen_kwargs, per_sample_runtime_kwargs, num_choices = self._map_generate_config(config)
        prompt: Any = messages
        if self._prompt_path == "text":
            prompt = render_messages(self._pipeline.tokenizer, messages)
        record = self._collator.admit(prompt, gen_kwargs, per_sample_runtime_kwargs, num_choices)
        output = await self._collator.serve(record)
        return self._assemble_model_output(output, stop_strings=gen_kwargs.get("stop_strings", ()))

    def _convert_input(self, input: list) -> list[dict[str, str]]:
        """Convert Inspect chat messages to the `[{"role", "content"}, ...]` form the pipeline accepts.

        Raises:
            NotImplementedError: On tool messages, assistant tool calls, or non-text content parts.
        """
        messages: list[dict[str, str]] = []
        for message in input:
            if isinstance(message, ChatMessageTool) or message.role == "tool":
                raise NotImplementedError(
                    "The conversation carries a tool message, but steering pipelines have no tool-use "
                    "convention; agentic evaluation is unsupported. Choose a non-agentic task."
                )
            if message.role == "assistant" and getattr(message, "tool_calls", None):
                raise NotImplementedError(
                    "An assistant message carries tool calls, but steering pipelines have no tool-use "
                    "convention; agentic evaluation is unsupported. Choose a non-agentic task."
                )
            content = message.content
            if isinstance(content, str):
                text = content
            else:
                parts: list[str] = []
                for part in content:
                    if isinstance(part, ContentText):
                        parts.append(part.text)
                    elif isinstance(part, ContentReasoning):
                        logger.debug("Dropping a ContentReasoning part from the conversation history.")
                    else:
                        raise NotImplementedError(
                            f"The conversation carries {type(part).__name__} content, which the pipeline's "
                            "text-only prompt surface cannot represent. Choose a text-only task."
                        )
                text = "".join(parts)
            messages.append({"role": message.role, "content": text})
        return messages

    def _unsupported_param(self, name: str, value: Any) -> None:
        """Apply the unsupported-parameter policy for one non-None config field."""
        message = (
            f"GenerateConfig.{name}={value!r} has no mapping onto the steering-pipeline surface; "
            "honoring it silently is not possible and dropping it would change decoding semantics. "
            "Remove the parameter, or set ProviderOptions(on_unsupported_param='warn') to ignore it."
        )
        if self._options.on_unsupported_param == "raise":
            raise ValueError(message)
        if name not in self._warned_params:
            self._warned_params.add(name)
            warnings.warn(message, UserWarning)

    def _map_generate_config(self, config: GenerateConfig) -> tuple[dict, dict, int]:
        """Classify and map one request's `GenerateConfig` onto pipeline generation kwargs.

        Returns:
            Tuple of (call-scoped gen kwargs, per-sample runtime kwargs, num_choices).

        Raises:
            NotImplementedError: If a logprob parameter is set.
            ValueError: If an unsupported parameter is set under the `"raise"` policy, or the
                reserved `extra_body` key is not a mapping.
        """
        for name in REFUSED_FIELDS:
            if getattr(config, name) is not None:
                raise NotImplementedError(
                    f"GenerateConfig.{name} requests log probabilities, but steering pipelines are "
                    "evaluated generation-only: decoding-time controls are not representable in "
                    "teacher-forced likelihoods. Use generation-based scorers."
                )
        extra_body = dict(config.extra_body or {})
        per_sample = extra_body.pop(RUNTIME_KWARGS_EXTRA_BODY_KEY, None) or {}
        if not isinstance(per_sample, Mapping):
            raise ValueError(
                f"extra_body[{RUNTIME_KWARGS_EXTRA_BODY_KEY!r}] must be a mapping of per-sample runtime "
                f"kwargs; got {type(per_sample).__name__}."
            )
        for name in POLICY_FIELDS:
            if getattr(config, name) is not None:
                self._unsupported_param(name, getattr(config, name))
        for key, value in extra_body.items():
            if value is not None:
                self._unsupported_param(f"extra_body[{key!r}]", value)

        gen_kwargs: dict[str, Any] = {}
        if config.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(config.max_tokens)
        if config.stop_seqs:
            gen_kwargs["stop_strings"] = tuple(config.stop_seqs)

        temperature = config.temperature
        if temperature is None:
            if config.top_p is not None or config.top_k is not None or config.seed is not None:
                logger.debug(
                    "temperature is unset, so the backend's default sampling posture applies; "
                    "top_p/top_k/seed are not attached. Set temperature explicitly for a known posture."
                )
        elif temperature == 0:
            gen_kwargs["do_sample"] = False
            if config.top_p is not None or config.top_k is not None or config.seed is not None:
                logger.debug("Greedy decoding (temperature=0) drops top_p, top_k, and any seed.")
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            if config.top_p is not None:
                gen_kwargs["top_p"] = float(config.top_p)
            if config.top_k is not None:
                gen_kwargs["top_k"] = int(config.top_k)
            seed = config.seed if config.seed is not None else self._base_seed
            if seed is not None:
                gen_kwargs["seed"] = int(seed)
                gen_kwargs["seed_scope"] = self._options.seed_scope

        num_choices = int(config.num_choices) if config.num_choices is not None else 1
        return gen_kwargs, dict(per_sample), num_choices

    def _assemble_model_output(self, output: Output, *, stop_strings: tuple[str, ...]) -> ModelOutput:
        """Map one pipeline `Output` (one row per candidate) onto an Inspect `ModelOutput`.

        Token ids are never modified. When reasoning tags are configured, each row is split into a
        reasoning part and an answer part, and only the answer is stop-string truncated. The split
        runs in the mode resolved at construction:

            - `"text"`: the row is decoded, truncated at the first stop-string occurrence, then
              substring split. Stop-truncation runs before the split, so a stop string that occurs
              inside the reasoning cuts the text mid-thinking and the answer is empty.
            - `"tokens"`: the row ids are split, each segment is decoded, then the answer segment
              (only) is truncated at the first stop-string occurrence. The reasoning segment is
              preserved verbatim, including any stop text or token-boundary overrun it ends with.

        The observable divergence between the modes is confined to token-boundary overrun in the
        answer, since a stop criterion halts generation at the first occurrence. An opened-but-
        unclosed reasoning segment yields an empty answer with everything as reasoning; one warning
        names the count of such rows in this output.
        """
        tokenizer = self._pipeline.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        texts = output.decode(tokenizer)
        num_rows = output.output_ids.size(0)
        reasons = output.finish_reasons
        if reasons is None or len(reasons) != num_rows:
            reasons = (output.finish_reason,) * num_rows

        choices: list[ChatCompletionChoice] = []
        unclosed = 0
        for row, text in enumerate(texts):
            if self._options.reasoning_tags is None:
                content: str | list = truncate_at_stop_strings(text, stop_strings) if stop_strings else text
            else:
                split, closed = self._split_row(text, output.output_ids[row], stop_strings)
                if not closed:
                    unclosed += 1
                if split.thinking is not None:
                    content = [ContentReasoning(reasoning=split.thinking), ContentText(text=split.answer)]
                else:
                    content = split.answer
            choices.append(ChatCompletionChoice(
                message=ChatMessageAssistant(content=content, model=self.model_name, source="generate"),
                stop_reason=_STOP_REASON_MAP.get(reasons[row], "unknown"),
            ))
        if unclosed:
            logger.warning(
                "%d generation(s) opened a thinking segment without closing it (budget spent thinking); "
                "their answer segment is empty.",
                unclosed,
            )

        input_tokens = _count_non_pad(output.adapted_input_ids, pad_token_id)
        returned_output_tokens = _count_non_pad(output.output_ids, pad_token_id)
        # generated_tokens counts every rollout a decoding driver generated, including discarded
        # proposals; the returned continuation count stays available for scoring and truncation
        # analysis. The driverless path leaves generated_tokens None, so usage is unchanged there.
        output_tokens = (
            output.generated_tokens if output.generated_tokens is not None else returned_output_tokens
        )
        usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=usage,
            metadata={"returned_output_tokens": returned_output_tokens},
        )

    def _split_row(
        self, text: str, row_ids: torch.Tensor, stop_strings: tuple[str, ...]
    ) -> tuple[ThinkingSplit, bool]:
        """Split one row into reasoning and answer, per the resolved mode.

        Returns:
            The `ThinkingSplit` and whether the reasoning channel closed (used to count the
            unclosed rows warned about once per output).
        """
        tags = self._options.reasoning_tags
        opened = self._options.reasoning_opened_at_start
        open_tag, close_tag = tags
        if self._reasoning_split == "tokens":
            ids = row_ids.tolist()
            split = split_thinking_ids(ids, self._pipeline.tokenizer, tags, opened_at_start=opened)
            closed = split.thinking is None or find_subsequence(ids, self._close_ids) != -1
            answer = truncate_at_stop_strings(split.answer, stop_strings) if stop_strings else split.answer
            return ThinkingSplit(thinking=split.thinking, answer=answer), closed
        if stop_strings:
            text = truncate_at_stop_strings(text, stop_strings)
        closed = close_tag in text or not (open_tag in text or opened)
        return split_thinking(text, tags, opened_at_start=opened), closed

    def max_tokens(self) -> int | None:
        """Default `max_tokens` filled by Inspect when a request sets none."""
        return self._options.default_max_tokens

    def max_connections(self) -> int:
        """Advertised concurrency; equals the effective batch ceiling so batches self-fill."""
        return self._effective_max_batch

    def connection_key(self) -> str:
        """Each provider instance owns its connection pool."""
        return f"steerability:{id(self)}"

    def should_retry(self, ex: Exception) -> bool:
        """Provider failures are deterministic, never rate limits."""
        return False

    def is_auth_failure(self, ex: Exception) -> bool:
        """The provider performs no authentication."""
        return False

    def tools_required(self) -> bool:
        """The provider never requires tool definitions."""
        return False

    def close(self) -> None:
        """Close the collator; further requests raise at admission."""
        self._collator.close()

    async def aclose(self) -> None:
        """Close the collator; further requests raise at admission."""
        self._collator.close()


def _count_non_pad(ids: torch.Tensor | None, pad_token_id: int | None) -> int:
    """Count non-pad positions in a token-id tensor (best effort where pad equals EOS)."""
    if ids is None:
        return 0
    if pad_token_id is None:
        return int(ids.numel())
    return int((ids != pad_token_id).sum().item())


def as_inspect_model(
    pipeline: "SteeringPipeline",
    *,
    options: ProviderOptions | None = None,
    base_seed: int | None = None,
    model_name: str = "steering-pipeline",
) -> Model:
    """Wrap a steered `SteeringPipeline` as an Inspect `Model`.

    The pipeline is an in-process object; the model is constructed directly rather than through
    the string registry, and renders as `steerability/<model_name>`.

    Args:
        pipeline: The steered pipeline to serve.
        options: Provider options; defaults to `ProviderOptions()`.
        base_seed: Attached to sampling dispatches whose `GenerateConfig` carries no seed.
        model_name: Bare model name for logs and rendering.

    Returns:
        An `inspect_ai.model.Model` usable with `eval` and `eval_set`.

    Raises:
        ValueError: If the pipeline is not steered.
        TypeError: If `options.chat_template_kwargs` is set while the tokenizer has no chat
            template.

    Warns:
        UserWarning: If the tokenizer has no chat template, so prompts render to plain text and
            `adapt_messages` does not fire.
    """
    api = SteeringPipelineModelAPI(
        model_name, pipeline=pipeline, options=options, base_seed=base_seed,
    )
    return Model(api, GenerateConfig())
