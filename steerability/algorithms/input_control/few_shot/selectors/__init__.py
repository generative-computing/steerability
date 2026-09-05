"""Example selectors for few-shot learning prompt adaptation.

Selectors determine which examples are picked from the pool when constructing the few-shot prompt.

Available selectors:

  - `RandomSelector` (re-exported from `common.selectors`): uniform random sampling. Suitable for
    homogeneous pools where any example is roughly as informative as any other.
  - `EPRSelector`: learned dense retriever (Rubin et al. 2021). Constructed by the caller (it requires
    a scoring LM) and passed to `FewShot` via `selector=`.
"""
from __future__ import annotations

from typing import Any

from steerability.algorithms.input_control.common.selectors.base import BaseSelector
from steerability.algorithms.input_control.common.selectors.random import RandomSelector
from steerability.algorithms.input_control.few_shot.selectors.epr import EPRSelector

SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
}


def selector_from_arg(value: BaseSelector | str | None) -> BaseSelector:
    """Resolve a `selector` argument value to a `BaseSelector[dict]` instance.

    Accepts an instance directly, a string name (looked up in the registry), or None (defaults to
    `RandomSelector`).
    """
    if value is None:
        return RandomSelector()
    if isinstance(value, BaseSelector):
        return value
    if isinstance(value, str):
        if value not in SELECTOR_REGISTRY:
            raise ValueError(f"Unknown selector name: {value!r}. Known: {sorted(SELECTOR_REGISTRY)}.")
        return SELECTOR_REGISTRY[value]()
    raise TypeError(f"selector must be BaseSelector, str, or None; got {type(value).__name__}.")


__all__ = ["EPRSelector", "RandomSelector", "SELECTOR_REGISTRY", "selector_from_arg"]
