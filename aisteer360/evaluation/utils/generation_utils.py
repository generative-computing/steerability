"""Generation utilities for use cases.

Every benchmark generation routes through `SteeringPipeline.generate` with `messages=` (or `text=`
for template-less tokenizers), so the pipeline owns chat templating, tokenization, and padding, and
message-level input controls apply. Runtime-override columns resolve against the prompt rows
themselves, so any subset of rows (a retry batch, an expanded prompt set) stays aligned by
construction.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.output import Output
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.utils.rendering import has_chat_template
from aisteer360.utils.thinking import DEFAULT_THINK_TAGS, split_thinking

logger = logging.getLogger(__name__)

DEFAULT_EVAL_BATCH_SIZE = 8


def output_record_fields(output: Output | None, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Build the per-item observability fields contributed by an `Output` to a generation dict.

    Always includes `"finish_reason"` (the row's `finish_reason`, or None when `output` is None). When
    `output.adapted_input_ids` is present, also includes `"adapted_prompt"`: the row's adapted token IDs with
    `pad_token_id` positions removed, decoded with `skip_special_tokens=False` so chat-template markers are
    preserved. In the pad-equals-eos tokenizer configuration this display string may also drop a genuine trailing
    EOS along with the padding.

    Args:
        output: The `Output` record for one item, or None if generation failed for it.
        tokenizer: Tokenizer used to decode `adapted_input_ids`.

    Returns:
        A dict with `"finish_reason"` and, when available, `"adapted_prompt"`.
    """
    if output is None:
        return {"finish_reason": None}

    fields: dict[str, Any] = {"finish_reason": output.finish_reason}

    if output.adapted_input_ids is not None:
        row = output.adapted_input_ids[0]
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            row = row[row != pad_token_id]
        fields["adapted_prompt"] = tokenizer.decode(row, skip_special_tokens=False)

    return fields


def log_truncation_count(outputs: Sequence[Output | None]) -> None:
    """Log one warning naming how many items in a run ended on `"length"` (truncation)."""
    truncated = sum(1 for output in outputs if output is not None and output.finish_reason == "length")
    if truncated:
        logger.warning(
            "%d of %d generations ended on 'length' (truncated at max_new_tokens); "
            "length-sensitive metrics may be affected.",
            truncated,
            len(outputs),
        )


def log_unclosed_thinking_count(thinking: Sequence[str | None], answers: Sequence[str]) -> None:
    """Log one warning naming how many items opened a thinking segment that never closed."""
    unclosed = sum(
        1 for think, answer in zip(thinking, answers) if think is not None and answer == ""
    )
    if unclosed:
        logger.warning(
            "%d of %d generations opened a thinking segment that never closed (no answer segment); "
            "consider raising max_new_tokens or using budget_forcing.",
            unclosed,
            len(thinking),
        )


def normalize_prompt_conversations(batch: Sequence[dict[str, Any]]) -> list[list[dict]]:
    """One conversation per row: a str prompt becomes a single user turn; a message list passes through.

    Args:
        batch: Prompt rows, each with a `"prompt"` value that is either a `str` or a non-empty list of
            chat-message mappings (each with `"role"` and `"content"`).

    Returns:
        One conversation per row, each a list of message dicts.

    Raises:
        TypeError: If a row's `"prompt"` is neither a `str` nor a list of chat-message dicts.
        ValueError: If a chat message is missing a `"role"` or `"content"` key.
    """
    conversations: list[list[dict]] = []
    for index, item in enumerate(batch):
        prompt = item["prompt"]
        if isinstance(prompt, str):
            conversations.append([{"role": "user", "content": prompt}])
        elif isinstance(prompt, list) and prompt and all(isinstance(message, Mapping) for message in prompt):
            for j, message in enumerate(prompt):
                if "role" not in message or "content" not in message:
                    raise ValueError(f"Prompt {index}: chat message {j} must have 'role' and 'content' keys.")
            conversations.append([dict(message) for message in prompt])
        else:
            raise TypeError(
                f"Prompt {index}: must be a str or a list of chat message dicts; got {type(prompt).__name__}."
            )
    return conversations


def ensure_left_padding(pipeline: SteeringPipeline) -> None:
    """Set left padding on the pipeline tokenizer for decoder-only models.

    The pipeline's `messages=` path tokenizes via `apply_chat_template(padding=True)` on the
    tokenizer's configured side, and the HF session's generate path does not left-normalize
    (only `score` does), so the side must be left before batched uneven prompts. The mutation
    persists on the pipeline's tokenizer, which the use case also holds. No live model (a
    non-HF pipeline) leaves the side unchanged.

    Args:
        pipeline: The steered pipeline whose tokenizer side is normalized.
    """
    config = getattr(getattr(pipeline, "model", None), "config", None)
    if config is None or getattr(config, "is_encoder_decoder", False):
        return
    tokenizer = pipeline.tokenizer
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"


def _map_runtime_overrides(overrides, rows):
    """Resolve one override spec (a column name, or a nested mapping of them) against the prompt rows.

    A column missing from some rows substitutes `[]` for those rows (sparse per-example values);
    a column missing from every row is a misconfiguration and raises.

    Args:
        overrides: A column name (str) or a mapping from variable to column name.
        rows: The prompt rows.

    Returns:
        A per-row value list for a column name, or a mapping from variable to such a list.

    Raises:
        ValueError: If a column name is absent from every row.
    """
    if isinstance(overrides, Mapping):
        return {variable: _map_runtime_overrides(column, rows) for variable, column in overrides.items()}
    column_name = overrides
    if not any(column_name in row for row in rows):
        available = sorted({key for row in rows for key in row})
        raise ValueError(
            f"runtime_overrides column {column_name!r} is missing from every prompt row; "
            f"available columns: {available}."
        )
    return [row.get(column_name, []) for row in rows]


def _build_runtime_kwargs(
    pipeline: SteeringPipeline,
    runtime_overrides: dict[str, dict[str, Any]] | None,
    rows: Sequence[dict[str, Any]],
) -> dict[str, list] | None:
    """Per-variable value lists aligned with `rows`, or None when no override applies.

    `runtime_kwargs` is one namespace per call (the pipeline warns on schema overlaps and treats
    sharing as legal), so two controls mapping one variable to the same override spec share the
    value stream; mapping it to different specs is a genuine conflict and raises.

    Args:
        pipeline: The steered pipeline, whose `controls` are matched by class name against
            `runtime_overrides`.
        runtime_overrides: A mapping from control class name to `{variable: column}`.
        rows: The prompt rows, against which columns resolve.

    Returns:
        A mapping from runtime-kwargs variable to a per-row value list, or None when no override
        applies to any control.

    Raises:
        ValueError: If two controls map one variable to different override specs.
    """
    if not runtime_overrides:
        return None
    runtime_kwargs_by_var: dict[str, list] = {}
    variable_spec: dict[str, tuple[str, Any]] = {}  # variable -> (control class name, raw spec)
    for control in pipeline.controls:
        control_name = type(control).__name__
        overrides = runtime_overrides.get(control_name)
        if not overrides:
            continue
        mapped = _map_runtime_overrides(overrides, rows)
        for variable, values in mapped.items():
            spec = overrides[variable]
            prior = variable_spec.get(variable)
            if prior is not None and prior[1] != spec:
                raise ValueError(
                    f"runtime_kwargs variable {variable!r} is mapped to {prior[1]!r} by {prior[0]} and to "
                    f"{spec!r} by {control_name}; one runtime_kwargs namespace cannot hold two value streams."
                )
            variable_spec[variable] = (control_name, spec)
            runtime_kwargs_by_var[variable] = values
    return runtime_kwargs_by_var or None


def generate_on_pipeline(
    batch: Sequence[dict[str, Any]],
    pipeline: SteeringPipeline,
    gen_kwargs: dict[str, Any] | None = None,
    runtime_overrides: dict[str, dict[str, Any]] | None = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    think_tags: tuple[str, str] | None = DEFAULT_THINK_TAGS,
) -> tuple[list[str], list[Output], list[str | None]]:
    """Generate on a steered pipeline; returns answer texts, aligned `Output` records, and thinking.

    Every chunk routes through `pipeline.generate(messages=...)` (or `text=` when the tokenizer has
    no chat template), so message-level input controls apply, and the pipeline owns templating,
    tokenization, and padding. Override columns resolve against `batch` rows, so any subset of rows
    (a retry batch, an expanded prompt set) stays aligned by construction.

    Each decoded continuation is split into a thinking segment and an answer segment when
    `think_tags` is set. The returned `decoded[i]` is the answer segment, and `thinking[i]` is the
    reasoning segment (or None when no think tag is present). Setting `think_tags=None` disables
    splitting: `decoded[i]` is then the full continuation and every `thinking[i]` is None, so the
    return arity is constant.

    Note that with a template-less tokenizer the pipeline falls back to `text=`; because
    `chat_template_kwargs` is valid only with `messages=`, setting it in `gen_kwargs` for a
    template-less tokenizer raises the pairing `TypeError` from `pipeline.generate`. This is
    intended, since a chat-template kwarg was configured for a model with no chat template.

    Args:
        batch: Prompt rows, each with a `"prompt"` (str or chat-message list) and any override columns.
        pipeline: The steered pipeline to generate on.
        gen_kwargs: Generation parameters forwarded to `pipeline.generate`.
        runtime_overrides: A mapping from control class name to `{variable: column}`; columns resolve
            against `batch`.
        batch_size: Chunk size for generation.
        think_tags: The `(open_tag, close_tag)` pair used to split thinking from the answer, or None
            to disable splitting.

    Returns:
        A tuple `(decoded, records, thinking)`; `decoded[i]` is the answer text for row `i`,
        `records[i]` is its aligned `Output` (carrying `adapted_input_ids` and `finish_reason`), and
        `thinking[i]` is the reasoning segment (`str | None`).

    Raises:
        TypeError: If a prompt is a chat message list but the tokenizer has no chat template.
    """
    ensure_left_padding(pipeline)
    conversations = normalize_prompt_conversations(batch)

    chat = has_chat_template(pipeline.tokenizer)
    if not chat:
        non_string = [i for i, item in enumerate(batch) if not isinstance(item["prompt"], str)]
        if non_string:
            raise TypeError(
                f"Prompt(s) {non_string} are chat message lists but the tokenizer has no chat "
                "template; supply string prompts or a chat-capable tokenizer."
            )

    runtime_kwargs_by_var = _build_runtime_kwargs(pipeline, runtime_overrides, batch)
    gen_kwargs = dict(gen_kwargs or {})
    supports_batching = getattr(pipeline, "supports_batching", False)

    def _generate(convs: list[list[dict]], runtime_kwargs) -> list[Output]:
        source = {"messages": convs} if chat else {"text": [conv[0]["content"] for conv in convs]}
        return pipeline.generate(**source, runtime_kwargs=runtime_kwargs, return_output=True, **gen_kwargs)

    decoded: list[str] = []
    records: list[Output] = []
    thinking: list[str | None] = []
    for start in range(0, len(conversations), batch_size):
        stop = start + batch_size
        chunk = conversations[start:stop]
        chunk_agg = (
            {variable: values[start:stop] for variable, values in runtime_kwargs_by_var.items()}
            if runtime_kwargs_by_var is not None
            else None
        )
        try:
            if supports_batching:
                outputs = _generate(chunk, chunk_agg)
            else:
                per_item = (
                    _runtime_kwargs_to_list(chunk_agg) if chunk_agg is not None else [None] * len(chunk)
                )
                outputs = []
                for conversation, item_kwargs in zip(chunk, per_item):
                    outputs.extend(_generate([conversation], item_kwargs))  # batch-of-one form: list[Output]
        except Exception:
            logger.warning("Generation failed for chunk %d.", start // batch_size, exc_info=True)
            raise
        records.extend(outputs)
        for output in outputs:
            text = output.decode(pipeline.tokenizer)[0]
            if think_tags is None:
                decoded.append(text)
                thinking.append(None)
            else:
                split = split_thinking(text, think_tags)
                decoded.append(split.answer)
                thinking.append(split.thinking)
    return decoded, records, thinking


def _as_pipeline(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringPipeline:
    """Wrap a bare model as an empty steered pipeline (the benchmark's baseline construction)."""
    pipeline = SteeringPipeline(controls=[], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


def batch_retry_generate(
    prompt_data: Sequence[dict[str, Any]],
    model_or_pipeline: PreTrainedModel | SteeringPipeline,
    tokenizer: PreTrainedTokenizerBase,
    gen_kwargs: dict[str, Any] | None = None,
    runtime_overrides: dict[str, dict[str, Any]] | None = None,
    parse_fn: Callable[[str], Any | None] | None = None,
    max_retries: int = 2,
    return_raw: bool = False,
    return_outputs: bool = False,
    return_thinking: bool = False,
    think_tags: tuple[str, str] | None = DEFAULT_THINK_TAGS,
    batch_size: int | None = None,
) -> (
    list[Any]
    | tuple[list[Any], ...]
):
    """Generate on a model or pipeline with optional parsing and retry.

    A bare `PreTrainedModel` is wrapped as an empty steered pipeline; a `SteeringPipeline` is used as
    given. Generation routes through `generate_on_pipeline`, so every path is core's. When `parse_fn`
    is supplied, only the rows whose parse returns None are retried (up to `max_retries` rounds), and
    each retry replaces the raw text, parsed value, `Output`, and thinking segment at that row. Retry
    rows carry their own override columns, so retries are aligned by construction.

    Raw text and `parse_fn` see the answer segment only, with any thinking segment removed by
    `generate_on_pipeline` (`think_tags`). Setting `think_tags=None` disables the split, so raw text
    is the full continuation and every returned thinking value is None.

    Args:
        prompt_data: Prompt rows, each with a `"prompt"` and any override columns.
        model_or_pipeline: A bare model (wrapped) or a steered pipeline.
        tokenizer: Tokenizer used only when wrapping a bare model; the pipeline's own tokenizer is
            authoritative for decoding.
        gen_kwargs: Generation parameters forwarded to the pipeline.
        runtime_overrides: A mapping from control class name to `{variable: column}`; columns resolve
            against `prompt_data` rows.
        parse_fn: Parser applied to each raw (answer) text; a None result marks the row for retry.
        max_retries: Maximum retry rounds for rows that fail to parse.
        return_raw: Return `(parsed, raw)` when `return_outputs` is False.
        return_outputs: Return `(parsed, raw, outputs)` regardless of `return_raw`.
        return_thinking: Append the per-row thinking list as the final element of the returned tuple.
        think_tags: The `(open_tag, close_tag)` pair forwarded to `generate_on_pipeline`, or None to
            disable splitting.
        batch_size: Chunk size; defaults to `DEFAULT_EVAL_BATCH_SIZE`.

    Returns:
        The base shape is `parsed` (default), `(parsed, raw)` (when `return_raw`), or
        `(parsed, raw, outputs)` (when `return_outputs`). When `return_thinking` is True the thinking
        list is appended as the final element of that shape:

            - `return_thinking` only: `(parsed, thinking)`.
            - `return_raw` and `return_thinking`: `(parsed, raw, thinking)`.
            - `return_outputs` and `return_thinking`: `(parsed, raw, outputs, thinking)`.

        `thinking[i]` is `str | None`. Default flags return the base shape unchanged.

    Raises:
        ValueError: If any row is missing the `"prompt"` key.
    """
    missing_prompt = [i for i, item in enumerate(prompt_data) if "prompt" not in item]
    if missing_prompt:
        raise ValueError(f"'prompt' key missing for {len(missing_prompt)} instances")

    batch_size = DEFAULT_EVAL_BATCH_SIZE if batch_size is None else batch_size
    pipeline = (
        model_or_pipeline
        if isinstance(model_or_pipeline, SteeringPipeline)
        else _as_pipeline(model_or_pipeline, tokenizer)
    )

    def _generate(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[Output], list[str | None]]:
        return generate_on_pipeline(
            batch=rows, pipeline=pipeline, gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides, batch_size=batch_size, think_tags=think_tags,
        )

    responses, outputs, thinking = _generate(prompt_data)
    if parse_fn is not None:
        parsed_responses = [parse_fn(response) for response in responses]
        retry_indices = [i for i, value in enumerate(parsed_responses) if value is None]
    else:
        parsed_responses = list(responses)
        retry_indices = []

    tries = 0
    while retry_indices and tries < max_retries:
        retry_raw, retry_outputs, retry_thinking = _generate([prompt_data[i] for i in retry_indices])
        for local_i, global_i in enumerate(retry_indices):
            responses[global_i] = retry_raw[local_i]
            outputs[global_i] = retry_outputs[local_i]
            thinking[global_i] = retry_thinking[local_i]
            parsed_responses[global_i] = parse_fn(retry_raw[local_i])
        retry_indices = [i for i, value in enumerate(parsed_responses) if value is None]
        tries += 1

    if think_tags is not None:
        log_unclosed_thinking_count(thinking, responses)

    if return_outputs:
        base: tuple[list[Any], ...] = (parsed_responses, responses, outputs)
    elif return_raw:
        base = (parsed_responses, responses)
    else:
        base = (parsed_responses,)

    if return_thinking:
        result = base + (thinking,)
        return result if len(result) > 1 else result[0]
    return base if len(base) > 1 else base[0]


def _runtime_kwargs_to_list(flat_dict):
    def find_length(obj):
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            return next(
                (
                    find_length(v)
                    for v in obj.values()
                    if (length := find_length(v)) is not None
                ),
                None,
            )
        return None

    def extract(obj, i):
        if isinstance(obj, list):
            return obj[i]
        if isinstance(obj, dict):
            return {k: extract(v, i) for k, v in obj.items()}
        return obj

    length = find_length(flat_dict)
    return [extract(flat_dict, i) for i in range(length)] if length else []
