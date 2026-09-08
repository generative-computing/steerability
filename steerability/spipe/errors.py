"""Exception types for the `.spipe` serialization format."""
from steerability.algorithms.core.base_control import NotFreezableError

__all__ = [
    "SpipeError",
    "SpipeFormatError",
    "SpipeSaveError",
    "SpipeIntegrityError",
    "SpipeStaleError",
    "SpipeCodeRefError",
    "NotFreezableError",
]


class SpipeError(Exception):
    """Base class for `.spipe` errors."""


class SpipeFormatError(SpipeError):
    """The manifest or archive violates the `spipe/1` format.

    Raised for unsupported format versions, schema violations (missing, mistyped, or unknown
    fields), unknown method keys, and malformed archives.
    """


class SpipeSaveError(SpipeError):
    """A pipeline or value cannot be serialized.

    Raised for unresolvable model references, unregistered control classes, live model or
    tokenizer objects inside args, lambdas and other unnameable callables, reserved `$`-prefixed
    mapping keys, values over the inline size limit, and controls that cannot freeze.
    """


class SpipeIntegrityError(SpipeError):
    """Stored artifact bytes do not match their content-addressed id."""


class SpipeStaleError(SpipeError):
    """A frozen artifact's recorded fit digest does not match the current recipe.

    The recipe's fit-relevant fields were edited after freezing and the pinned artifacts no
    longer correspond to the recipe. Call `thaw()` and re-`steer()`, or pass
    `allow_stale=True` to load anyway.
    """


class SpipeCodeRefError(SpipeError):
    """Decoding requires importing code and `allow_code` was not granted.

    Raised when a manifest contains a `$ref` callable reference, a `$dc` class outside the
    `steerability.` namespace, or a pickle-backed memory artifact, and `load()` was called
    without `allow_code=True`.
    """
