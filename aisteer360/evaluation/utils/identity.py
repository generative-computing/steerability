"""Canonical configuration identity and trial-seed derivation for benchmarks.

Pure functions with no I/O. Config identity is a digest over the materialized pipeline (control
classes and their full constructor parameters), stable across processes and machines for every
handled value type. The module does not import `ControlSpec`; spec objects are duck-typed on their
`control_cls` and `name` attributes.
"""
import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_TENSOR_TAG = "__tensor__"
_DATACLASS_TAG = "__dataclass__"
_TYPE_TAG = "__type__"


def qualname(obj_type: type) -> str:
    """Fully qualified name of a type, as `<module>.<qualname>`.

    Args:
        obj_type: The type to name.

    Returns:
        The dotted module-plus-qualname string.
    """
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    """Content bytes of a CPU tensor for hashing, handling dtypes numpy lacks.

    A 2-byte dtype without a numpy equivalent (e.g. bfloat16) hashes through a lossless
    `int16` view; any other unsupported dtype casts to float32. The dtype string in the digest
    prefix keeps identity across dtypes.

    Args:
        tensor: A contiguous CPU tensor.

    Returns:
        The raw content bytes.
    """
    try:
        return tensor.numpy().tobytes()
    except TypeError:
        if tensor.element_size() == 2:
            return tensor.view(torch.int16).numpy().tobytes()
        return tensor.to(torch.float32).numpy().tobytes()


def canonical_value(obj: Any, _path: str = "$") -> Any:
    """JSON-serializable canonical form of a constructor-argument value.

    The form is stable across processes and machines for every handled type. Tensor identity is
    content-addressed over dtype, shape, and bytes, with device and `requires_grad` excluded.
    Mapping key order never affects the form, sequence order always does, and set element order
    never does. A callable reduces to its qualified name. An unhandled object type reduces to its
    type qualname (value-blind), logged at debug.

    Args:
        obj: The value to canonicalize.
        _path: Internal breadcrumb naming the position of `obj` within the enclosing structure,
            used only in the debug log for value-blind fallbacks.

    Returns:
        A JSON-serializable canonical representation of `obj`.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().to("cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_payload(tensor))
        return {
            _TENSOR_TAG: {
                "sha256": digest.hexdigest(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            _DATACLASS_TAG: qualname(type(obj)),
            "fields": {
                field.name: canonical_value(getattr(obj, field.name), f"{_path}.{field.name}")
                for field in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, Mapping):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return {str(key): canonical_value(value, f"{_path}[{key!r}]") for key, value in items}
    if isinstance(obj, (set, frozenset)):
        canonical = [canonical_value(item, _path) for item in obj]
        return sorted(canonical, key=lambda value: json.dumps(value, sort_keys=True))
    if isinstance(obj, (list, tuple)):
        return [canonical_value(item, f"{_path}[{i}]") for i, item in enumerate(obj)]
    if callable(obj):
        return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"
    logger.debug("canonical_value: value-blind fallback for %s at %s", qualname(type(obj)), _path)
    return {_TYPE_TAG: qualname(type(obj))}


def config_descriptor_from_specs(specs: Sequence[Any], params: Mapping[str, Mapping[str, Any]]) -> dict:
    """Descriptor for a spec-defined configuration, from the resolved per-spec kwargs.

    Each spec contributes one control entry keyed by its resolved name (its `name`, or its
    `control_cls.__name__`), carrying the canonical form of the resolved constructor kwargs. Entry
    order follows `specs`, so sequence order participates in identity.

    Args:
        specs: The specs of one configuration, in list order; each duck-typed on `control_cls`
            and `name`.
        params: Mapping from resolved spec name to that spec's resolved constructor kwargs.

    Returns:
        A descriptor dict with a `"controls"` list, one entry per spec.
    """
    return {
        "controls": [
            {
                "control": qualname(spec.control_cls),
                "params": canonical_value(dict(params[spec.name or spec.control_cls.__name__])),
                "enabled": True,
            }
            for spec in specs
        ]
    }


def config_descriptor_from_controls(controls: Sequence[Any]) -> dict:
    """Descriptor for a fixed-control configuration, recovered from each control's `args`.

    Construction stores the validated `Args` dataclass as `control.args`; arg-free controls
    (`Args = None`) have no such attribute and contribute empty params. This distinguishes two
    fixed pipelines that differ only in their controls' configuration.

    Args:
        controls: Instantiated controls, in list order.

    Returns:
        A descriptor dict with a `"controls"` list, one entry per control.
    """
    entries = []
    for control in controls:
        args = getattr(control, "args", None)
        entries.append({
            "control": qualname(type(control)),
            "params": canonical_value(args) if args is not None else {},
            "enabled": bool(getattr(control, "enabled", True)),
        })
    return {"controls": entries}


def config_digest(descriptor: Mapping[str, Any]) -> str:
    """A 12-hex-character sha256 of the descriptor's sorted JSON form.

    A pure function of `descriptor`: equal descriptors always digest equal.

    Args:
        descriptor: A JSON-serializable descriptor, e.g. from `config_descriptor_from_controls`.

    Returns:
        The 12-character hex digest.
    """
    serialized = json.dumps(descriptor, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def derive_trial_seed(base_seed: int, config_id: str, trial_id: int) -> int:
    """Deterministic per-(config, trial) seed derived from a benchmark-level base seed.

    A pure function of its three inputs, distinct across `trial_id` values and across `config_id`
    values by construction.

    Args:
        base_seed: The benchmark-level base seed.
        config_id: The configuration identifier.
        trial_id: The trial index.

    Returns:
        A 32-bit non-negative integer seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{config_id}:{trial_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
