"""Normalized generation parameters with one rendering rule per backend family."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

NORMALIZED_PARAM_NAMES: tuple[str, ...] = (
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "greedy",
    "n",
    "repetition_penalty",
    "seed",
    "stop_strings",
    "stop_token_ids",
)

LOWERABLE_PARAM_NAMES: tuple[str, ...] = (
    "stop_strings",
    "stop_token_ids",
    "max_new_tokens",
    "min_new_tokens",
)


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """The sampling-facing subset of generation parameters, normalized across backends.

    Each backend family owns one rendering rule. In-process, the normalized fields render onto
    `model.generate` names and every key in `extra` passes through untouched. On API backends
    the normalized table is exhaustive and unmapped parameters raise, so `extra` is rejected
    there.

    Attributes:
        max_new_tokens: Maximum number of new tokens.
        min_new_tokens: Minimum number of new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        top_k: Top-k sampling cutoff.
        greedy: True forces greedy decoding, False forces sampling, None leaves the backend
            default.
        n: Number of returned candidates per prompt.
        repetition_penalty: Repetition penalty.
        seed: Sampling seed. In-process it renders as a `fork_rng`-scoped `manual_seed` around
            the item's decode; on vLLM it maps to the request seed. Sessions derive a distinct
            per-item seed from this value when an item carries no seed of its own.
        stop_strings: Stop strings, composed by the session as stop rules on both backend
            families. Token ids are returned as generated; the pipeline truncates decoded text
            at the first stop-string occurrence.
        stop_token_ids: Token ids that halt a row once its last generated token is one of them,
            in addition to the tokenizer's EOS.
        extra: Additional keyword arguments passed through unmapped on the in-process arm. A
            normalized field always takes precedence over a same-named key in `extra`.

    Raises:
        ValueError: If `min_new_tokens` exceeds `max_new_tokens` when both are set.
    """

    max_new_tokens: int | None = None
    min_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    greedy: bool | None = None
    n: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop_strings: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.stop_strings, str):
            object.__setattr__(self, "stop_strings", (self.stop_strings,))
        else:
            object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
        object.__setattr__(self, "stop_token_ids", tuple(int(i) for i in self.stop_token_ids))
        if (
            self.min_new_tokens is not None
            and self.max_new_tokens is not None
            and self.min_new_tokens > self.max_new_tokens
        ):
            raise ValueError(
                f"min_new_tokens={self.min_new_tokens} exceeds max_new_tokens={self.max_new_tokens}; the length "
                "bounds are jointly unsatisfiable. Control contributions only tighten bounds, so lower "
                "min_new_tokens or loosen the tightened max_new_tokens."
            )

    @classmethod
    def from_gen_kwargs(cls, **gen_kwargs: Any) -> "GenerationParams":
        """Split keyword arguments into normalized fields and pass-through extras.

        `do_sample` maps onto `greedy` (inverted) and `num_return_sequences` onto `n`; keys named
        exactly like a normalized field bind to it; everything else lands in `extra`.

        Args:
            **gen_kwargs: Generation keyword arguments in `model.generate` vocabulary.

        Returns:
            The normalized `GenerationParams`.
        """
        normalized: dict[str, Any] = {}
        if "do_sample" in gen_kwargs:
            normalized["greedy"] = not gen_kwargs.pop("do_sample")
        if "num_return_sequences" in gen_kwargs:
            normalized["n"] = gen_kwargs.pop("num_return_sequences")
        for name in NORMALIZED_PARAM_NAMES:
            if name in gen_kwargs:
                normalized[name] = gen_kwargs.pop(name)
        return cls(**normalized, extra=gen_kwargs)

    def to_gen_kwargs(self) -> dict[str, Any]:
        """Render the parameters back into `model.generate`-vocabulary keyword arguments.

        This inverts `from_gen_kwargs`, so `greedy` renders as `do_sample` (inverted), `n` as
        `num_return_sequences`, the stop fields and `seed` keep their normalized names, and
        `extra` merges underneath the normalized fields.

        Returns:
            The keyword arguments; `from_gen_kwargs(**params.to_gen_kwargs())` reproduces
            `params`.
        """
        gen_kwargs: dict[str, Any] = dict(self.extra)
        if self.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens
        if self.min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = self.min_new_tokens
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k
        if self.greedy is not None:
            gen_kwargs["do_sample"] = not self.greedy
        if self.n is not None:
            gen_kwargs["num_return_sequences"] = self.n
        if self.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            gen_kwargs["seed"] = self.seed
        if self.stop_strings:
            gen_kwargs["stop_strings"] = self.stop_strings
        if self.stop_token_ids:
            gen_kwargs["stop_token_ids"] = self.stop_token_ids
        return gen_kwargs


def merge_lowered_params(params: GenerationParams, contribution: Mapping[str, Any]) -> GenerationParams:
    """Merge one control's sampling-expressible contribution into `params`.

    Stop strings and stop token ids union with the caller's (caller entries first, duplicates
    dropped); `max_new_tokens` takes the minimum and `min_new_tokens` the maximum of the present
    values, so a control can only tighten the caller's bounds.

    Args:
        params: The caller-derived parameters.
        contribution: Mapping over a subset of `stop_strings`, `stop_token_ids`,
            `max_new_tokens`, and `min_new_tokens`.

    Returns:
        The merged `GenerationParams`.

    Raises:
        ValueError: If `contribution` carries a key outside the lowerable set, or if the tightened
            bounds cross (`min_new_tokens` exceeds `max_new_tokens`), raised on construction of the
            merged instance.
    """
    unknown = [key for key in contribution if key not in LOWERABLE_PARAM_NAMES]
    if unknown:
        raise ValueError(
            f"Control contributed non-lowerable generation parameter(s) {sorted(unknown)}; "
            f"lowerable parameters are {', '.join(LOWERABLE_PARAM_NAMES)}."
        )

    updates: dict[str, Any] = {}
    stop_strings = contribution.get("stop_strings")
    if stop_strings:
        if isinstance(stop_strings, str):
            stop_strings = (stop_strings,)
        merged = list(params.stop_strings)
        merged.extend(text for text in stop_strings if text not in merged)
        updates["stop_strings"] = tuple(merged)
    stop_token_ids = contribution.get("stop_token_ids")
    if stop_token_ids:
        merged_ids = list(params.stop_token_ids)
        merged_ids.extend(int(i) for i in stop_token_ids if int(i) not in merged_ids)
        updates["stop_token_ids"] = tuple(merged_ids)
    max_new = contribution.get("max_new_tokens")
    if max_new is not None:
        updates["max_new_tokens"] = (
            max_new if params.max_new_tokens is None else min(params.max_new_tokens, max_new)
        )
    min_new = contribution.get("min_new_tokens")
    if min_new is not None:
        updates["min_new_tokens"] = (
            min_new if params.min_new_tokens is None else max(params.min_new_tokens, min_new)
        )
    if not updates:
        return params
    return replace(params, **updates)
