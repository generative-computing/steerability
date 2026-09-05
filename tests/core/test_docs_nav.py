"""Docs hygiene: every nav target exists and no reference page renders the package root."""
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"


def _nav_targets(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_targets(value)
    elif isinstance(node, list):
        for item in node:
            yield from _nav_targets(item)


def test_nav_targets_exist():
    nav = yaml.safe_load((DOCS / ".nav.yml").read_text())["nav"]
    missing = [target for target in _nav_targets(nav) if not (DOCS / target).exists()]
    assert not missing, f"nav targets not found under docs/: {missing}"


def test_no_reference_page_renders_package_root():
    offenders = [
        str(page.relative_to(REPO))
        for page in (DOCS / "reference").rglob("*.md")
        if re.search(r"^::: steerability\s*$", page.read_text(), re.M)
    ]
    assert not offenders, f"pages rendering the whole package (duplicate anchors): {offenders}"
