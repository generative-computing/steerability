"""LLM-as-a-judge metrics, executed through the backend seam."""
from __future__ import annotations

import json
import re
import string
import warnings
from typing import Any, Callable

from aisteer360.algorithms.core.execution.backend import Backend
from aisteer360.algorithms.core.execution.params import NORMALIZED_PARAM_NAMES, GenerationParams
from aisteer360.algorithms.core.execution.payloads import GenerationItem, PreparedPrompt
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.evaluation.metrics.backend_utils import resolve_metric_backend
from aisteer360.evaluation.metrics.base import Metric
from aisteer360.utils.rendering import has_chat_template

_FORMAT_INSTRUCTIONS = (
    'The output should be a markdown code snippet formatted in the following schema, '
    'including the leading and trailing "```json" and "```":\n\n'
    "```json\n"
    "{{\n"
    '\t"score": float  // A single float between {low} and {high} (inclusive) that rates the prediction.\n'
    "}}\n"
    "```"
)

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_BUILTIN_FIELDS = frozenset({"response", "prompt", "lower_bound", "upper_bound"})


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text, handling optional markdown code fences.

    Tries to find a fenced code block first; falls back to parsing the raw text.

    Args:
        text: Raw LLM response that should contain a JSON object.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    match = _CODE_BLOCK_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from response: {e}")
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object, got {type(result).__name__}")
    return result


def build_structured_parser(scale: tuple[float, float]) -> tuple[str, Callable[[str, tuple[float, float]], float]]:
    """Build format instructions and a parsing function for rating predictions.

    Returns a parser that extracts a `{"score": <float>}` JSON object from the judge model's
    response and clamps the value to the given scale.

    Args:
        scale: A `(low, high)` tuple specifying the valid inclusive range for the score.

    Returns:
        A tuple of `(format_instructions, parse_fn)` where `format_instructions` is the instruction
        string appended to each judge prompt and `parse_fn(text, scale)` returns a clamped float
        score.
    """
    low, high = scale
    format_instructions = _FORMAT_INSTRUCTIONS.format(low=low, high=high)

    def parse_fn(text: str, _: tuple[float, float]) -> float:
        parsed = _extract_json(text)
        if "score" not in parsed:
            raise ValueError(f"JSON missing 'score' key, got keys: {list(parsed.keys())}")
        score = float(parsed["score"])
        return max(low, min(high, score))

    return format_instructions, parse_fn


def _extract_template_fields(template: str) -> set[str]:
    """The named placeholders in `template`, extracted with `string.Formatter().parse`."""
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }


class LLMJudgeMetric(Metric):
    """Base class for LLM-as-a-judge evaluation metrics.

    A judge scores each response with a language model according to natural-language criteria
    stated in a prompt template, returning numerical scores within a configured range. Generation
    runs through the backend seam: rendered prompts become `GenerationItem`s executed by a
    `SteeringSession` on the configured backend, so vLLM offline and vLLM serve judges work with no
    judge-specific backend code, and seeds, `n` fan-out, and stop handling come from the session
    contract.

    Configuration is declarative. The class attributes `prompt_template`, `scale`, `system_prompt`,
    and `structured_output` are overridden by subclasses through assignment, and a constructor
    keyword overrides the class attribute per instance:

        class Factuality(LLMJudgeMetric):
            prompt_template = _PROMPT
            scale = (1, 5)

    The template's placeholders beyond the built-ins (`response`, `prompt`, `lower_bound`,
    `upper_bound`) are extracted at construction and resolved per item from the keyword arguments
    `compute` receives: each such field's value in `kwargs` must be a sequence aligned with
    `responses`, or a scalar (broadcast to every item).

    The judge model is configured by a model reference and a backend, never by a live model object.
    Model placement and dtype travel as spec options; an already-loaded model or engine travels as a
    live `Backend`. Backends are cached by spec (see `resolve_metric_backend`), so a judge and a
    `Perplexity` configured with equal specs share one loaded resource. The backend is resolved per
    `compute()`, so after `release_metric_backends()` the next call constructs it again. On the
    Hugging Face backend each `compute()` opens and closes its own exclusive session, so sharing
    across sequential calls is safe; concurrent `compute()` calls on one shared Hugging Face backend
    are unsupported.

    Args:
        model: Judge model reference (hub id or local path), or None when `backend` carries the
            identity.
        backend: A `BackendSpec`, a backend-kind string (`"huggingface"` or `"vllm"`), a live
            `Backend`, or None (in-process Hugging Face). A bare `"vllm-serve"` string is rejected;
            pass a `BackendSpec` with `base_url`.
        prompt_template: Template string. Must contain `{response}` (and `{lower_bound}` /
            `{upper_bound}` when the structured format instructions reference the bounds), optionally
            `{prompt}`, and any extra fields resolved from `compute` kwargs. Overrides the class
            attribute when given; required (here or as a class attribute).
        scale: Score range as `(min, max)`; scores are clamped to it. Defaults to `(1, 5)`.
        system_prompt: Optional judge system message, used only when the backend tokenizer has a
            chat template.
        structured_output: When True (default), append JSON format instructions and parse with the
            built-in JSON parser. When False, `parser` is required.
        parser: Custom parser mapping the judge's decoded response to a float. Required when
            `structured_output=False`; forbidden when `structured_output=True`.
        batch_size: Number of prompts submitted per session chunk. Defaults to 8.
        max_retries: Maximum re-sample attempts on parse failure. Only meaningful under sampling
            (temperature > 0). Defaults to 5.
        gen_kwargs: Generation parameters in the normalized vocabulary (`NORMALIZED_PARAM_NAMES`).
            Unknown keys raise. `num_return_sequences` is not accepted (`n` is the multi-sample
            knob) and `pad_token_id` is neither accepted nor defaulted.
        name: Metric name; defaults to the class name.

    Raises:
        TypeError: If `prompt_template` is unset after resolution, or a bare `"vllm-serve"` backend
            string is passed.
        ValueError: If `gen_kwargs` carries a key outside the normalized vocabulary; if
            `structured_output=True` and `parser` are both set, or `structured_output=False` without
            a `parser`; if `n > 1` under deterministic decoding; or if the backend/model identity is
            ambiguous.

    Attributes:
        prompt_template: The resolved template.
        scale: The resolved score range.
        system_prompt: The resolved judge system message.
        structured_output: Whether structured JSON output is used.
    """

    prompt_template: str | None = None
    scale: tuple[float, float] = (1, 5)
    system_prompt: str | None = None
    structured_output: bool = True

    def __init__(
        self,
        model: str | None = None,
        *,
        backend: "BackendSpec | str | Backend | None" = None,
        prompt_template: str | None = None,
        scale: tuple[float, float] | None = None,
        system_prompt: str | None = None,
        structured_output: bool | None = None,
        parser: Callable[[str], float] | None = None,
        batch_size: int = 8,
        max_retries: int = 5,
        gen_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        resolved_template = prompt_template if prompt_template is not None else type(self).prompt_template
        if resolved_template is None:
            raise TypeError(
                f"{type(self).__name__} requires `prompt_template`; set it as a class attribute or "
                "pass it to the constructor."
            )
        resolved_scale = tuple(scale) if scale is not None else type(self).scale
        resolved_system_prompt = system_prompt if system_prompt is not None else type(self).system_prompt
        resolved_structured = structured_output if structured_output is not None else type(self).structured_output

        self.scale = resolved_scale
        self.system_prompt = resolved_system_prompt
        self.structured_output = resolved_structured
        self.prompt_template = resolved_template.strip()
        self.batch_size = batch_size
        self.max_retries = max_retries

        field_names = _extract_template_fields(self.prompt_template)
        self._extra_fields = tuple(sorted(field_names - _BUILTIN_FIELDS))
        self._uses_prompt = "prompt" in field_names

        if resolved_structured:
            if parser is not None:
                raise ValueError(
                    "Provide either `structured_output=True` (default) or a custom `parser`, not both. "
                    "When structured_output=True the built-in JSON parser is used."
                )
            self.format_instructions, self.parse_fn = build_structured_parser(self.scale)
        else:
            if parser is None:
                raise ValueError(
                    "structured_output=False requires a `parser` callable: (text: str) -> float."
                )
            self.format_instructions = ""
            self.parse_fn = lambda text, _scale, _p=parser: float(_p(text))

        self._params = self._build_params(gen_kwargs)
        self._model_ref = model
        self._backend_ref = backend
        resolve_metric_backend(model, backend)  # validate the identity now; the backend is re-resolved per compute

    def _build_params(self, gen_kwargs: dict[str, Any] | None) -> GenerationParams:
        """Build the per-compute `GenerationParams` from the normalized `gen_kwargs`.

        Unknown keys raise. A configured temperature of `0.0` renders as `greedy=True` with the
        temperature omitted (the vLLM renderer rejects `greedy=True` with a nonzero temperature).

        Raises:
            ValueError: If `gen_kwargs` carries a key outside `NORMALIZED_PARAM_NAMES`, or `n > 1`
                under deterministic decoding.
        """
        kwargs = dict(gen_kwargs or {})
        unknown = [key for key in kwargs if key not in NORMALIZED_PARAM_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown gen_kwargs key(s) {sorted(unknown)}; the judge accepts only the normalized "
                f"generation vocabulary {', '.join(NORMALIZED_PARAM_NAMES)}."
            )
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_new_tokens", 30)

        temperature = kwargs.get("temperature")
        n = int(kwargs.get("n", 1) or 1)
        if temperature == 0.0 and n > 1:
            raise ValueError(
                "n > 1 requires temperature > 0; deterministic decoding produces identical samples."
            )

        params: dict[str, Any] = {
            key: value for key, value in kwargs.items()
            if key not in ("greedy", "temperature")
        }
        if temperature == 0.0:
            params["greedy"] = True
        else:
            params["temperature"] = temperature
            params["greedy"] = kwargs["greedy"] if "greedy" in kwargs else False
        return GenerationParams(**params)

    @property
    def _backend(self) -> Backend:
        """The configured backend, resolved through the metric cache on each access.

        A cache lookup while the backend is cached; after `release_metric_backends()` the next
        access constructs it again, so a released metric stays usable. A live `Backend` passed at
        construction is returned as is.
        """
        return resolve_metric_backend(self._model_ref, self._backend_ref)

    @property
    def _is_deterministic(self) -> bool:
        return bool(self._params.greedy) and self._params.temperature in (None, 0.0)

    def _resolve_field(self, name: str, kwargs: dict[str, Any], count: int) -> list[Any]:
        """Resolve one extra field to a per-item list of length `count`.

        A sequence value must have length `count`; a scalar broadcasts.

        Raises:
            ValueError: If the field is missing from `kwargs`, or a sequence value is misaligned.
        """
        if name not in kwargs:
            raise ValueError(
                f"Judge template field {name!r} is missing; provide it as a compute() keyword. "
                f"Received keyword(s): {sorted(kwargs)}."
            )
        value = kwargs[name]
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            return [value] * count
        if len(value) != count:
            raise ValueError(
                f"Judge template field {name!r} has length {len(value)}, expected {count} to align "
                "with `responses`."
            )
        return list(value)

    def _render(self, responses: list[str], prompts: list[str] | None, kwargs: dict[str, Any]) -> list[str]:
        """Render the core judge prompt for every response, resolving the D3 extra fields."""
        count = len(responses)
        if self._uses_prompt and prompts is None:
            raise ValueError(
                "The judge template references {prompt} but no `prompts` were provided to compute()."
            )
        extra_values = {name: self._resolve_field(name, kwargs, count) for name in self._extra_fields}

        rendered: list[str] = []
        for index in range(count):
            fields: dict[str, Any] = {
                "response": responses[index],
                "lower_bound": self.scale[0],
                "upper_bound": self.scale[1],
            }
            if prompts is not None:
                fields["prompt"] = prompts[index]
            for name in self._extra_fields:
                fields[name] = extra_values[name][index]
            core = self.prompt_template.format(**fields)
            if self.format_instructions:
                core = f"{core}\n\n{self.format_instructions}"
            rendered.append(core)
        return rendered

    def _prepare_prompt(self, core: str, has_chat: bool) -> PreparedPrompt:
        """One `PreparedPrompt` for a rendered core prompt.

        On a chat-templated tokenizer the core prompt becomes the user turn (preceded by the
        optional system message); otherwise it is submitted as plain text.
        """
        if has_chat:
            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": core})
            return PreparedPrompt.from_messages(messages)
        return PreparedPrompt.from_text(core)

    def _decode_candidates(self, result, tokenizer) -> list[str]:
        """Decode one item's candidate rows to text, one string per candidate."""
        return result.output.decode(tokenizer, skip_special_tokens=True)

    def _parse_or_retry(self, session, prepared: PreparedPrompt, candidates: list[str]) -> list[float]:
        """Parse each candidate to a score, retrying a failed item under sampling."""
        scores: list[float] = []
        for candidate in candidates:
            try:
                scores.append(self.parse_fn(candidate, self.scale))
            except Exception as error:
                if self._is_deterministic:
                    raise ValueError(
                        f"Failed to parse score under deterministic decoding. Raw response: "
                        f"{candidate!r}. Original error: {error}"
                    ) from error
                scores.append(self._retry_score(session, prepared))
        return scores

    def _retry_score(self, session, prepared: PreparedPrompt) -> float:
        """Re-sample the single failed item up to `max_retries`, then return `nan`."""
        tokenizer = getattr(session, "tokenizer", None)
        for _ in range(self.max_retries):
            result = session.generate([GenerationItem(prompt=prepared)], self._params)[0]
            for candidate in self._decode_candidates(result, tokenizer):
                try:
                    return self.parse_fn(candidate, self.scale)
                except Exception:
                    continue
        warnings.warn(
            f"Failed to parse score after {self.max_retries} retries; returning float('nan').",
            UserWarning,
        )
        return float("nan")

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float | list[float]]:
        """Compute LLM judge scores for a list of responses.

        Renders one prompt per response, submits them through one session on the configured backend
        in chunks of `batch_size`, and parses the judge's decoded responses into scores. Under
        `n > 1` the candidate scores of each response are averaged.

        Args:
            responses: Text responses to evaluate.
            prompts: Prompts corresponding to each response, one per item, or None. Referenced by a
                `{prompt}` placeholder; required when the template uses it.
            **kwargs: Per-item values for the template's extra fields; each must be a sequence
                aligned with `responses` or a scalar (broadcast).

        Returns:
            Score statistics with keys:

                - `"mean_score"`: Overall average score across all responses.
                - `"scores"`: Mean score per response (averaged across candidates).
                - `"raw_scores"`: All individual candidate scores per response.

        Raises:
            AssertionError: If `prompts` is provided with a different length than `responses`.
            ValueError: If a `{prompt}` placeholder is used without `prompts`, or an extra field is
                missing or misaligned.
        """
        if prompts is not None and len(prompts) != len(responses):
            raise AssertionError("`responses` and `prompts` must be the same length")

        rendered = self._render(responses, prompts, kwargs)
        if not rendered:
            return {"mean_score": 0.0, "scores": [], "raw_scores": []}

        prompt_scores: list[list[float]] = []
        with self._backend.open_session() as session:
            tokenizer = getattr(session, "tokenizer", None)
            has_chat = tokenizer is not None and has_chat_template(tokenizer)
            prepared = [self._prepare_prompt(core, has_chat) for core in rendered]
            for start in range(0, len(prepared), self.batch_size):
                chunk = prepared[start:start + self.batch_size]
                items = [GenerationItem(prompt=prompt) for prompt in chunk]
                results = session.generate(items, self._params)
                for prompt, result in zip(chunk, results):
                    candidates = self._decode_candidates(result, tokenizer)
                    prompt_scores.append(self._parse_or_retry(session, prompt, candidates))

        mean_per_prompt = [sum(row) / len(row) for row in prompt_scores]
        corpus_mean = sum(mean_per_prompt) / len(mean_per_prompt) if mean_per_prompt else 0.0
        return {
            "mean_score": corpus_mean,
            "scores": mean_per_prompt,
            "raw_scores": prompt_scores,
        }
