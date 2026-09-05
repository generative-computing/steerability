"""
Discovers steering methods at import‑time for cli reference.
"""
import logging
from importlib import import_module
from pathlib import Path

from steerability.utils.optional import OPTIONAL_MODULE_EXTRAS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

REGISTRY: dict[str, dict[str, "SteeringMethod"]] = {}

_REQUIRED_EXPORT_KEYS = ("name", "control", "args")


class SteeringMethod:
    """Container for a discovered steering method's metadata.

    Attributes:
       category: Category name (e.g., "state", "input")
       name: Method name (e.g., "pasta", "few_shot")
       control_cls: The control class implementation
       args_cls: The args dataclass for configuration
    """
    def __init__(self, category: str, name: str, control_cls: type, args_cls: type):
        self.category = category
        self.name = name
        self.control_cls = control_cls
        self.args_cls = args_cls


class RegistryError(RuntimeError):
    """A steering-method package failed to import or exported a malformed STEERING_METHOD."""


def _validate_export(module_path: str, method: object) -> None:
    """Validate the shape of a discovered `STEERING_METHOD` export.

    Requires the keys in `_REQUIRED_EXPORT_KEYS`; extra keys (e.g. `"category"`) are tolerated.

    Raises:
        RegistryError: If `method` is not a dict, is missing a required key, or has a field of
            the wrong type.
    """
    if not isinstance(method, dict):
        raise RegistryError(
            f"{module_path}: STEERING_METHOD must be a dict, got {type(method).__name__}."
        )
    missing = [key for key in _REQUIRED_EXPORT_KEYS if key not in method]
    if missing:
        raise RegistryError(f"{module_path}: STEERING_METHOD missing keys {missing}.")
    if not isinstance(method["name"], str) or not method["name"]:
        raise RegistryError(f"{module_path}: STEERING_METHOD['name'] must be a non-empty str.")
    if not isinstance(method["control"], type):
        raise RegistryError(f"{module_path}: STEERING_METHOD['control'] must be a class.")
    if method["args"] is not None and not isinstance(method["args"], type):
        raise RegistryError(f"{module_path}: STEERING_METHOD['args'] must be a class or None.")


def _crawl_methods(root: Path = ROOT, package_prefix: str = __name__.rsplit(".core.registry", 1)[0]) -> None:
    """Auto-discover all steering methods by recursively crawling the algorithms directory.

    For each top-level category directory (input_control, structural_control, state_control,
    output_control), walks all nested `__init__.py` files and imports any module that exports a
    `STEERING_METHOD` dict. The exported dict is registered under the category's bucket keyed by
    the method name.

    Import failures are classified so that only genuinely-absent optional dependencies are
    skipped; internal import bugs, tripwires, and malformed or duplicate exports fail loudly.

    Args:
        root: Directory whose subdirectories are the steering categories to crawl.
        package_prefix: Import prefix prepended to discovered package paths (parameterized so
            tests can crawl a synthetic package tree).

    Raises:
        RegistryError: On an internal `ModuleNotFoundError`, any other exception raised while
            importing a discovered module, a malformed `STEERING_METHOD` export, or a duplicate
            method name within a category.
    """
    internal_top_level = package_prefix.split(".")[0]

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name in {"core", "__pycache__"}:
            continue
        category = category_dir.name.removesuffix("_control")

        for init_file in sorted(category_dir.rglob("__init__.py")):
            rel_parts = init_file.relative_to(root).parent.parts
            if not rel_parts:
                continue
            module_path = package_prefix + "." + ".".join(rel_parts)

            try:
                module = import_module(module_path)
            except ModuleNotFoundError as exc:
                missing_top_level = (exc.name or "").split(".")[0]
                if missing_top_level and missing_top_level != internal_top_level:
                    extra = OPTIONAL_MODULE_EXTRAS.get(missing_top_level)
                    if extra is not None:
                        logger.info(
                            'Skipping %s: optional dependency %r not installed '
                            '(pip install "steerability[%s]" to enable it).',
                            module_path, missing_top_level, extra,
                        )
                    else:
                        logger.warning(
                            "Skipping %s: module %r is not installed and is not a "
                            "declared optional dependency.",
                            module_path, missing_top_level,
                        )
                    continue
                raise RegistryError(
                    f"Internal import failure while discovering {module_path}: {exc}"
                ) from exc
            except Exception as exc:
                raise RegistryError(
                    f"Failed to import {module_path} during steering-method discovery: {exc}"
                ) from exc

            method = getattr(module, "STEERING_METHOD", None)
            if method is None:
                continue
            _validate_export(module_path, method)

            bucket = REGISTRY.setdefault(category + "_control", {})
            if method["name"] in bucket:
                raise RegistryError(
                    f"Duplicate steering-method name {method['name']!r} in category "
                    f"{category!r} (second definition: {module_path})."
                )
            bucket[method["name"]] = SteeringMethod(
                category, method["name"], method["control"], method["args"]
            )


_crawl_methods()


def method_key_for(control_cls: type) -> str:
    """The registry key `"<category>_control/<name>"` of a registered control class.

    Args:
        control_cls: The control class to look up.

    Returns:
        The method key.

    Raises:
        RegistryError: If `control_cls` is not a registered control class.
    """
    for category, bucket in REGISTRY.items():
        for name, method in bucket.items():
            if method.control_cls is control_cls:
                return f"{category}/{name}"
    raise RegistryError(
        f"{control_cls.__module__}.{control_cls.__qualname__} is not a registered steering "
        "method; register it via a STEERING_METHOD export before serializing it."
    )


def resolve_method_key(key: str) -> SteeringMethod:
    """The `SteeringMethod` registered under a `"<category>_control/<name>"` key.

    Args:
        key: The method key to resolve.

    Returns:
        The registered method.

    Raises:
        RegistryError: If the key is malformed or names no registered method; the message
            lists the registered names of the category (or the known categories).
    """
    category, _, name = key.partition("/")
    bucket = REGISTRY.get(category)
    if not name or bucket is None:
        raise RegistryError(
            f"Unknown method key {key!r}; expected '<category>_control/<name>' with category "
            f"one of {sorted(REGISTRY)}."
        )
    method = bucket.get(name)
    if method is None:
        raise RegistryError(
            f"Unknown method {name!r} in category {category!r}; registered names are "
            f"{sorted(bucket)}."
        )
    return method
