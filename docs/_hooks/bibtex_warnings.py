"""Filter mkdocs-bibtex false-positive citation warnings.

mkdocs-bibtex scans every page for bracketed citation blocks and treats an at-prefixed token inside one as a
citation key, warning "Inline reference to unknown key <key>" when the key is not in the bibliography. Rendered
code blocks and Jupyter cell output contain bracketed at-prefixed tokens (Python decorators such as the dataclass,
task, and metric decorators, and the jupyter-widgets model references in widget-state output) that are not
citations, so the warning fires spuriously and aborts `mkdocs build --strict`. This hook drops only that message
for those non-citation keys, leaving every other warning (including genuine missing-citation warnings) intact.
"""
import logging

_ALLOWED_NON_CITATION_KEYS = frozenset({"dataclass", "task", "metric", "torch", "jupyter-widgets"})

_PREFIX = "Inline reference to unknown key "


class _BibtexFalsePositiveFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith(_PREFIX):
            key = message[len(_PREFIX):].strip()
            if key in _ALLOWED_NON_CITATION_KEYS:
                return False
        return True


def on_startup(**kwargs) -> None:
    logging.getLogger("mkdocs.plugins.mkdocs-bibtex").addFilter(_BibtexFalsePositiveFilter())
