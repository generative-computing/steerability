"""Helpers for working with control objects: composition/validation and adapt-messages guards."""
import warnings
from collections import defaultdict
from typing import Iterable, Type

from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.output_control.base import DecodingDriver, OutputControl
from steerability.algorithms.state_control.base import StateControl
from steerability.algorithms.structural_control.base import StructuralControl

_CATEGORIES: tuple[Type, ...] = (InputControl, StructuralControl, StateControl, OutputControl)


def merge_controls(
        supplied: Iterable[StructuralControl | StateControl | InputControl | OutputControl]
) -> dict[str, object]:
    """Sort supplied controls by category.

    Every category admits any number of controls, returned as ordered lists (in encounter order)
    under `"input_controls"`, `"structural_controls"`, `"state_controls"`, and `"output_controls"`.
    An omitted category is an empty list; every application of a category is a fold over its
    list, whose identity element is the empty sequence (the prompt threads through the adapt
    chain, the model threads through structural steers, state and output entries accumulate).

    The output category additionally admits at most one enabled `DecodingDriver`; the decode loop
    does not compose. Input controls chain in two phases (message-level, then token-level); see
    `SteeringPipeline.generate` for the per-control application contract.

    Args:
       supplied: List of control instances to organize

    Returns:
       Dict with keys `"input_controls"`, `"structural_controls"`, `"state_controls"`, and
       `"output_controls"`, each an ordered list of controls (empty for unspecified categories).

    Raises:
       ValueError: If the same control instance is supplied more than once, or if more than one
           enabled `DecodingDriver` is supplied
       TypeError: If an unrecognized control type is supplied
    """
    supplied = list(supplied)

    # reject the same control instance supplied twice
    seen_ids: set[int] = set()
    for control in supplied:
        if id(control) in seen_ids:
            raise ValueError(
                f"The same {type(control).__name__} instance was supplied more than once. "
                "To apply a method twice, construct a second instance."
            )
        seen_ids.add(id(control))

    bucket: dict[type, list] = defaultdict(list)
    for control in supplied:
        for category in _CATEGORIES:
            if isinstance(control, category):
                bucket[category].append(control)
                break
        else:
            raise TypeError(f"Unknown control type: {type(control)}")

    # at most one enabled DecodingDriver; the decode loop does not compose
    drivers = [
        control for control in bucket.get(OutputControl, [])
        if isinstance(control, DecodingDriver) and getattr(control, "enabled", True)
    ]
    if len(drivers) > 1:
        names = [type(control).__name__ for control in drivers]
        raise ValueError(
            f"Multiple decoding drivers supplied: {names}. The decode loop does not compose; "
            "keep one DecodingDriver and express the rest as logits processors or stopping criteria."
        )

    return {
        "input_controls": list(bucket.get(InputControl, [])),
        "structural_controls": list(bucket.get(StructuralControl, [])),
        "state_controls": list(bucket.get(StateControl, [])),
        "output_controls": list(bucket.get(OutputControl, [])),
    }


def runtime_kwargs_schema(
        controls: Iterable[StructuralControl | StateControl | InputControl | OutputControl]
) -> dict[str, dict]:
    """Merge the `RUNTIME_KWARGS_SCHEMA` declarations of the enabled controls, by name.

    Each declared entry carries `name` plus optional `type`, `required`, `help`, and `scope`
    fields. `scope` is `"row"` for a per-prompt value (in a batched call the control receives a
    sequence with one element per prompt row, in row order) or `"call"` for one value per
    `generate` call regardless of batch size; an entry without `scope` is `"call"`. Two controls
    may declare one name only when their declarations agree, in which case they share one value at
    inference time.

    Args:
        controls: Control instances whose enabled members' declarations are merged.

    Returns:
        Mapping from declared name to its merged entry, with `scope` normalized to `"row"` or
        `"call"`. For a name declared by several controls, the first declaration's other fields
        are kept.

    Raises:
        ValueError: If an entry's `scope` is neither `"row"` nor `"call"` (naming the control and
            the entry), or if two controls declare one name with a different `scope` or `type`
            (naming both controls).
    """
    merged: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for control in controls:
        if not getattr(control, "enabled", True):
            continue
        control_name = type(control).__name__
        for entry in getattr(control, "RUNTIME_KWARGS_SCHEMA", []):
            name = entry.get("name")
            if not name:
                continue
            scope = entry.get("scope", "call")
            if scope not in ("row", "call"):
                raise ValueError(
                    f"{control_name} declares runtime kwarg {name!r} with invalid scope {scope!r}; "
                    "scope must be 'row' or 'call'."
                )
            if name not in merged:
                merged[name] = {**entry, "scope": scope}
                owners[name] = control_name
                continue
            existing = merged[name]
            if existing["scope"] != scope:
                raise ValueError(
                    f"{owners[name]} and {control_name} declare runtime kwarg {name!r} with different "
                    f"scopes ({existing['scope']!r} vs {scope!r})."
                )
            existing_type = existing.get("type")
            new_type = entry.get("type")
            if existing_type is not None and new_type is not None and existing_type != new_type:
                raise ValueError(
                    f"{owners[name]} and {control_name} declare runtime kwarg {name!r} with different "
                    f"types ({existing_type!r} vs {new_type!r})."
                )
    return merged


def warn_if_adapt_messages_bypassed(input_controls: list[InputControl], already_warned: bool) -> bool:
    """Warn (UserWarning) when any control in `input_controls` overrides `adapt_messages` but the
    caller used tensor/text input, bypassing chat-template tokenization. The warning names each
    bypassed control class. Returns the updated warned-state.

    Args:
        input_controls: The pipeline's input controls, in list order.
        already_warned: Whether the bypass warning has already fired for this pipeline.

    Returns:
        The updated warned-state.
    """
    if already_warned:
        return already_warned
    bypassed = [
        type(control).__name__
        for control in input_controls
        if type(control).adapt_messages is not InputControl.adapt_messages
    ]
    if bypassed:
        warnings.warn(
            f"{', '.join(bypassed)} override(s) `adapt_messages` but received tensor/text input; "
            "the message-level adaptation will not run. Pass `list[dict]` or `list[list[dict]]` "
            "to engage `adapt_messages`.",
            UserWarning,
        )
        return True
    return already_warned
