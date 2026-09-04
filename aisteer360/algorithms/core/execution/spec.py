"""Frozen backend identity, expressed as `BackendSpec` with canonicalized options."""
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

KNOWN_BACKEND_KINDS: tuple[str, ...] = ("huggingface", "vllm", "vllm-serve")

_MAPPING_TAG = "__mapping__"
_SEQUENCE_TAG = "__sequence__"


def canonicalize_option_value(value: Any) -> Any:
    """Convert one option value to a hashable canonical form.

    Mappings become `("__mapping__", ((key, value), ...))` tuples ordered by key type name and
    key repr, with scalar keys (str, int, bool, float) preserved as-is; sequences become
    `("__sequence__", (value, ...))` tuples; `torch.dtype`, `torch.device`, and `pathlib.Path`
    values become strings (dtypes without the `torch.` prefix); scalars pass through. Any other
    object is rendered as `"<qualname>:<repr>"`, which keeps spec construction total but makes
    the value only as stable as the object's `repr`; options should be plain data.

    Args:
        value: The option value to canonicalize.

    Returns:
        A hashable canonical form of `value`.
    """
    if isinstance(value, Mapping):
        entries = []
        for key, val in value.items():
            canonical_key = key if isinstance(key, (str, int, bool, float)) else str(key)
            entries.append((canonical_key, canonicalize_option_value(val)))
        entries.sort(key=lambda entry: (type(entry[0]).__qualname__, repr(entry[0])))
        return (_MAPPING_TAG, tuple(entries))
    if isinstance(value, (list, tuple)):
        return (_SEQUENCE_TAG, tuple(canonicalize_option_value(item) for item in value))
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"{type(value).__qualname__}:{value!r}"


_ENCODER_DECODER_CACHE: dict[tuple[str, bool], bool] = {}


def _reject_encoder_decoder_if_resolvable(model_ref: str, trust_remote_code: bool) -> None:
    """Reject encoder-decoder models on vLLM specs when the config resolves locally.

    Encoder-decoder execution is in-process only. The check consults only locally available
    files (a local path or the local hub cache); an unresolvable reference passes, and backend
    construction repeats the check authoritatively. Results are memoized per
    `(model_ref, trust_remote_code)`, since equal specs are re-constructed per call.

    Raises:
        ValueError: If the locally resolved config declares an encoder-decoder model.
    """
    key = (model_ref, trust_remote_code)
    is_encoder_decoder = _ENCODER_DECODER_CACHE.get(key)
    if is_encoder_decoder is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_ref, local_files_only=True, trust_remote_code=trust_remote_code,
            )
            is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        except Exception:
            is_encoder_decoder = False
        _ENCODER_DECODER_CACHE[key] = is_encoder_decoder
    if is_encoder_decoder:
        raise ValueError(
            f"Model {model_ref!r} is an encoder-decoder model; encoder-decoder execution is "
            "in-process only. Run this pipeline on the huggingface backend."
        )


def _decanonicalize(value: Any) -> Any:
    """Rebuild plain dicts and lists from a canonical option value.

    Mapping keys keep their original scalar type; sequences rebuild as lists regardless of the
    original sequence type.
    """
    if isinstance(value, tuple) and len(value) == 2 and value[0] == _MAPPING_TAG:
        return {key: _decanonicalize(val) for key, val in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == _SEQUENCE_TAG:
        return [_decanonicalize(item) for item in value[1]]
    return value


@dataclass(frozen=True, slots=True, eq=False)
class BackendSpec:
    """Frozen, hashable identity of one backend configuration.

    The spec hash is the backend identity used for engine caching, benchmark checkpoint keys,
    capability-probe caches, and artifact provenance. Options are canonicalized at construction
    (nested mappings stored as sorted tuples, dtypes as strings, no live objects), so two specs
    built from equal option mappings compare and hash equal regardless of key order. Equality
    and `hash()` follow `spec_hash`, so value type is part of identity (an option value of
    `True` and one of `1` yield different specs).

    Construction also validates the configuration. A `"vllm"`/`"vllm-serve"` spec with the
    vLLM-Hook plugin active (`hook_plugin` option) and speculative decoding configured
    (`speculative_config` option, top-level or under `engine_kwargs`) is rejected, since
    draft-model forwards are unhooked and verification passes break the worker's per-request
    position accounting. A `"vllm"`/`"vllm-serve"` spec naming an encoder-decoder model is
    rejected when the model config resolves locally (encoder-decoder execution is in-process
    only); an unresolvable reference passes here and is re-checked at backend construction.

    Attributes:
        kind: Backend kind, one of `"huggingface"`, `"vllm"`, or `"vllm-serve"`.
        model: Model reference (hub id or local path), or None when the backend adopts an
            already-loaded model.
        options: Canonicalized option mapping, stored as sorted tuples.

    Raises:
        ValueError: If `kind` is unknown, a vLLM spec combines the vLLM-Hook plugin with
            speculative decoding, or a vLLM spec names a locally resolvable encoder-decoder
            model.
        TypeError: If `options` is neither a mapping nor a canonical options tuple.
    """

    kind: str
    model: str | None = None
    options: Any = field(default=())

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_BACKEND_KINDS:
            raise ValueError(
                f"Unknown backend kind {self.kind!r}; known kinds are {', '.join(KNOWN_BACKEND_KINDS)}."
            )
        if isinstance(self.model, Path):
            object.__setattr__(self, "model", str(self.model))
        raw_options = self.options
        if isinstance(raw_options, Mapping):
            object.__setattr__(self, "options", canonicalize_option_value(raw_options))
        elif raw_options == ():
            object.__setattr__(self, "options", canonicalize_option_value({}))
        elif not (isinstance(raw_options, tuple) and len(raw_options) == 2 and raw_options[0] == _MAPPING_TAG):
            raise TypeError(
                f"options must be a mapping or a canonical options tuple; got "
                f"{type(raw_options).__name__}."
            )

        if self.kind in ("vllm", "vllm-serve"):
            if self.get_option("hook_plugin"):
                speculative = (
                    self.get_option("speculative_config")
                    or self.get_option("engine_kwargs", "speculative_config")
                )
                if speculative:
                    raise ValueError(
                        "Speculative decoding cannot be combined with the vLLM-Hook plugin: "
                        "draft-model forwards are unhooked and verification passes break the "
                        "worker's per-request position accounting."
                    )
                if (
                    self.kind == "vllm"
                    and self.get_option("engine_kwargs", "enforce_eager") is False
                ):
                    raise ValueError(
                        "enforce_eager=False cannot be combined with the vLLM-Hook plugin: "
                        "worker hooks do not run under CUDA-graph replay. Drop the option "
                        "(hook_plugin engines default to eager execution) or disable the plugin."
                    )
            if self.model is not None:
                _reject_encoder_decoder_if_resolvable(
                    self.model, bool(self.get_option("trust_remote_code", default=False)),
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackendSpec):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def _identity(self) -> tuple[str, str | None, str]:
        return (self.kind, self.model, repr(self.options))

    def get_option(self, *path: str, default: Any = None) -> Any:
        """Read one option by key path, rebuilding plain dicts and lists on the way out.

        Args:
            *path: Nested option keys, e.g. `get_option("hf_model_kwargs", "attn_implementation")`.
            default: Value returned when any key along the path is absent.

        Returns:
            The option value with mappings rebuilt as plain dicts and sequences as plain lists,
            or `default`.
        """
        value: Any = self.options
        for key in path:
            if not (isinstance(value, tuple) and len(value) == 2 and value[0] == _MAPPING_TAG):
                return default
            entries = dict(value[1])
            if key not in entries:
                return default
            value = entries[key]
        return _decanonicalize(value)

    def options_dict(self) -> dict[str, Any]:
        """Return the options as a plain nested dict (empty when no options were provided)."""
        rebuilt = _decanonicalize(self.options)
        return rebuilt if isinstance(rebuilt, dict) else {}

    @property
    def spec_hash(self) -> str:
        """A 16-character hex digest over `(kind, model, options)`, consistent with `__eq__`
        and stable across processes for plain-data options."""
        digest = hashlib.sha256(repr(self._identity()).encode("utf-8"))
        return digest.hexdigest()[:16]
