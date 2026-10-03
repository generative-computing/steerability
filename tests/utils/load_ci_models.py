from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent / "ci_models.yaml"

LOCAL_PREFIX = "local:"


def get_models() -> dict[str, str]:
    """Map each CI model tag to a Hub id, or to `local:<builder>` for a model built in process."""
    with open(_YAML_PATH, "r", encoding="utf-8") as filepath:
        return yaml.safe_load(filepath)["models"]


def build_local_model(entry: str):
    """Build the `(model, tokenizer)` pair of a `local:<builder>` entry.

    `<builder>` is the name of a zero-argument function in `tests.utils.tiny_models` that returns
    the pair.
    """
    from tests.utils import tiny_models

    name = entry.removeprefix(LOCAL_PREFIX)
    builder = getattr(tiny_models, name, None)
    if builder is None:
        raise ValueError(f"ci_models.yaml lists an unknown local builder {name!r} (tests.utils.tiny_models.{name}).")
    return builder()
