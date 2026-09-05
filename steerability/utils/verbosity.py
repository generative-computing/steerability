"""Opt-in verbosity controls for the `steerability` package logger.

The package attaches a `logging.NullHandler` to its root logger at import, so toolkit logging is
silent by default and never emits "no handlers could be found" warnings. This module exposes the
supported way to turn that logging up or down and to quiet known-noisy third-party libraries. None
of these functions run at import time, and the library never calls them on its own; the package
mutates no global logging state unless a caller asks.

The named module is `verbosity` rather than `logging` so it does not shadow the standard library
`logging` module from inside the package.
"""
from __future__ import annotations

import logging
import os

_ROOT_LOGGER_NAME = "steerability"
_ENV_VAR = "STEERABILITY_VERBOSITY"
_HANDLER_FORMAT = "%(levelname)s %(name)s: %(message)s"

_LEVEL_NAMES: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_env_default_applied = False


def _coerce_level(level: int | str) -> int:
    """Resolve a level name or `logging` constant to an integer level.

    Args:
        level: One of `"debug"`, `"info"`, `"warning"`, `"error"` (case-insensitive) or a
            `logging` level constant.

    Returns:
        The integer logging level.

    Raises:
        ValueError: If a string level name is not recognized.
        TypeError: If `level` is neither a string nor an integer.
    """
    if isinstance(level, str):
        try:
            return _LEVEL_NAMES[level.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown verbosity level {level!r}; expected one of {sorted(_LEVEL_NAMES)} "
                "or a logging level constant."
            ) from None
    if isinstance(level, int):
        return level
    raise TypeError(f"Verbosity level must be a str or int, got {type(level).__name__}.")


def _has_real_handler(logger: logging.Logger) -> bool:
    """Whether `logger` has a handler other than a `NullHandler`."""
    return any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def set_verbosity(level: int | str) -> None:
    """Set the level of the `steerability` logger and attach one stream handler if none is attached.

    Sets the level of the package root logger (`steerability`), so every module logger under it is
    affected. If the package root logger has no handler other than the import-time `NullHandler`, a
    single `StreamHandler` with a plain `%(levelname)s %(name)s: %(message)s` format is attached so
    records reach the console. The function is idempotent: a second call updates the level and does
    not attach a second handler.

    Args:
        level: One of `"debug"`, `"info"`, `"warning"`, `"error"` (case-insensitive) or a
            `logging` level constant.

    Raises:
        ValueError: If a string level name is not recognized.
        TypeError: If `level` is neither a string nor an integer.
    """
    global _env_default_applied
    _env_default_applied = True  # an explicit call supersedes the env default
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(_coerce_level(level))
    if not _has_real_handler(logger):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_HANDLER_FORMAT))
        logger.addHandler(handler)


def get_verbosity() -> int:
    """Return the effective level of the `steerability` logger.

    Applies the `STEERABILITY_VERBOSITY` environment default once (on first library use) before
    reading, so an environment-configured level is reflected without an explicit `set_verbosity`
    call.

    Returns:
        The effective integer level, resolved up the logger hierarchy when the package logger has
        no level of its own.
    """
    _apply_env_default()
    return logging.getLogger(_ROOT_LOGGER_NAME).getEffectiveLevel()


def _apply_env_default() -> None:
    """Apply the `STEERABILITY_VERBOSITY` level once if it is set and no explicit call has been made.

    Reads the environment on first invocation. If `STEERABILITY_VERBOSITY` is set to a recognized level
    name or integer, applies it via `set_verbosity`; if unset, the package logger is left untouched
    (default stays silent). An unrecognized value is ignored. Called by library code on first use
    rather than at import time, so importing `steerability` never inspects the environment.
    """
    global _env_default_applied
    if _env_default_applied:
        return
    _env_default_applied = True
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return
    try:
        level = int(raw)
    except ValueError:
        if raw.lower() not in _LEVEL_NAMES:
            return
        level = raw
    set_verbosity(level)


def quiet_third_party() -> None:
    """Reduce output from known-noisy third-party libraries.

    Sets the `transformers` logging facility to ERROR, disables Hugging Face Hub progress bars, and
    (when importable) sets the `datasets` logging facility to ERROR. Each step is guarded so a
    missing optional dependency is a no-op. This is opt-in and the library never calls it. It does
    not add any `warnings` filter and does not set `TQDM_DISABLE`, so user-owned progress bars and
    Python warnings are left alone.
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError:
        pass

    try:
        from huggingface_hub.utils import logging as hub_logging  # noqa: F401

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass

    try:
        from datasets.utils import logging as datasets_logging

        datasets_logging.set_verbosity_error()
    except ImportError:
        pass
