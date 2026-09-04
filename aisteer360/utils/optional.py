"""Optional-dependency declarations and import guard.

Single source of truth mapping optional third-party modules to the extra that
provides them. The registry consults it for skip messages; feature modules call
`require()` at their import boundary to fail with an actionable hint.
"""
from importlib import import_module
from types import ModuleType

OPTIONAL_MODULE_EXTRAS: dict[str, str] = {
    "mergekit": "merging",
    "econml": "cpo",
    "matplotlib": "plots",
    "seaborn": "plots",
    "vllm": "vllm",
    "vllm_hook_plugins": "vllm",
    "xgrammar": "guided",
}


def require(module_name: str) -> ModuleType:
    """Import `module_name` or raise `ModuleNotFoundError` naming the extra that provides it.

    Args:
        module_name: The importable module name (e.g., ``"mergekit"`` or ``"econml.dml"``).

    Returns:
        The imported module.

    Raises:
        ModuleNotFoundError: If the module is not installed. The message names the
            ``aisteer360[<extra>]`` extra when the top-level module is a declared
            optional dependency, otherwise it names the missing package directly. The
            raised error preserves the missing module's `name`, so the registry can
            classify it, and is an `ImportError` subclass, so callers guarding on
            `ImportError` keep working.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        top_level = module_name.split(".")[0]
        extra = OPTIONAL_MODULE_EXTRAS.get(top_level)
        hint = (
            f'Install it via: pip install "aisteer360[{extra}]"'
            if extra
            else f"Install the '{top_level}' package to use this feature."
        )
        raise ModuleNotFoundError(
            f"'{module_name}' is required for this feature but is not installed. {hint}",
            name=exc.name,
        ) from exc
