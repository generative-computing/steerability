"""Parser strategies for `LLMMetaPromptProposer`.

A parser maps a single decoded LLM response to a list of candidate strings (dropping empties).
Methods opt into a parser explicitly; the proposer default is `parse_whole`. Three strategies are
provided:

  - `parse_whole`: the whole stripped response as a single candidate (default; minimal processing).
  - `parse_fenced_or_whole`: long-instruction parser (GEPA). Extracts the first fenced block when
    present, else the whole text minus a single meta lead-in line; preserves all internal structure.
  - `parse_concise_instruction`: short-system-prompt parser (CPO / PRewrite). Reduces an LLM response
    to a single short instruction paragraph, stripping preamble/enumerators and balancing quotes.
"""
from __future__ import annotations

import re

_LABEL_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,40}:$")
_QUOTE_PAIRS = {('"', '"'), ("'", "'"), ("“", "”"), ("`", "`")}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TERMINAL_RE = re.compile(r"[.!?:]$")
_LEADIN_RE = re.compile(
    r"^(sure|certainly|of course|okay|absolutely|here('s| is)|below is|the following is|i('d| would) be happy)\b",
    re.IGNORECASE,
)
_METAWORD_RE = re.compile(
    r"\b(prompt|version|refin\w*|rewrit\w*|revis\w*|improv\w*|attempt\w*)\b", re.IGNORECASE
)
_ACK_RE = re.compile(r"^(sure|certainly|of course|okay|absolutely|great)[!.…]?$", re.IGNORECASE)
_QUOTE_OPENERS = {o: c for o, c in _QUOTE_PAIRS}
_QUOTE_CLOSERS = {c: o for o, c in _QUOTE_PAIRS}
_ENUMERATOR_RE = re.compile(r"^\s*(\d{1,2}[.)]|[-*•])\s+")
_MAX_UNTERMINATED_WORDS = 25

_FENCE_RE = re.compile(r"```")


def _strip_quotes(para: str) -> str:
    """Strip a matching surrounding pair; otherwise strip an unmatched edge quote whose
    counterpart does not appear anywhere else in the text (contraction-safe)."""
    if len(para) >= 2 and (para[0], para[-1]) in _QUOTE_PAIRS:
        return para[1:-1].strip()
    if para and para[0] in _QUOTE_OPENERS and _QUOTE_OPENERS[para[0]] not in para[1:]:
        para = para[1:].strip()
    if para and para[-1] in _QUOTE_CLOSERS and _QUOTE_CLOSERS[para[-1]] not in para[:-1]:
        para = para[:-1].strip()
    return para


def _is_preamble(paragraph: str) -> bool:
    """Single-line meta-discourse: a cue line, a chatty lead-in about the prompt, or a bare
    acknowledgement. Multi-line paragraphs are never preamble."""
    lines = paragraph.splitlines()
    if len(lines) != 1:
        return False
    line = lines[0].strip()
    if line.endswith(":"):  # "Refined prompt:" / "Here is ...:"
        return True
    if _LEADIN_RE.match(line) and _METAWORD_RE.search(line):  # "Sure, here's a refined version of the prompt."
        return True
    if len(line.split()) <= 4 and _ACK_RE.match(line):  # "Sure!" / "Certainly."
        return True
    return False


def parse_whole(text: str) -> list[str]:
    """Default parser. Return the whole stripped response as a single candidate.

    Returns `[]` when the response is empty or whitespace-only.
    """
    stripped = text.strip()
    return [stripped] if stripped else []


def parse_fenced_or_whole(text: str) -> list[str]:
    """Long-instruction parser (GEPA).

    1. If one or more ``` fenced blocks exist, take the FIRST block's inner content. A fence that
       opens but never closes is treated as running to end-of-text.
    2. Else take the whole text, stripping only a single leading meta lead-in line
       (e.g. "Here is the revised prompt:", "Sure, here is ...:").
    3. Preserve all internal structure: paragraphs, blank lines, bullet lists, length. Do not collapse
       to one paragraph, drop trailing sentences, or cap length.

    Returns `[candidate]` if non-empty, else `[]`.
    """
    if not text.strip():
        return []

    first_fence = _FENCE_RE.search(text)
    if first_fence is not None:
        inner_start = first_fence.end()
        closing_fence = _FENCE_RE.search(text, inner_start)
        if closing_fence is not None:
            inner = text[inner_start:closing_fence.start()]
        else:
            inner = text[inner_start:]
        inner = inner.lstrip("\n").rstrip()
        # drop an optional language tag immediately after the opening fence (e.g. ```text)
        lines = inner.split("\n", 1)
        if lines and re.fullmatch(r"[A-Za-z0-9_+-]{1,20}", lines[0].strip()):
            inner = lines[1] if len(lines) > 1 else ""
        candidate = inner.strip()
        return [candidate] if candidate else []

    paragraphs = re.split(r"(\n\s*\n)", text.strip())
    body = text.strip()
    first_para = paragraphs[0].strip() if paragraphs else ""
    if _is_preamble(first_para):
        # drop the single leading lead-in paragraph, preserve the rest verbatim
        remainder = "".join(paragraphs[1:]).lstrip("\n")
        body = remainder.strip()
    candidate = body.strip()
    return [candidate] if candidate else []


def parse_concise_instruction(text: str) -> list[str]:
    """Short-system-prompt parser (CPO / PRewrite).

    Reduce a free-text LLM response to a single short instruction:

    1. Split into paragraphs; skip leading preamble paragraphs (single-line cue/lead-in/
       acknowledgement) as long as a later paragraph exists; keep the first remaining paragraph.
    2. Strip a leading enumerator/bullet ("1. ", "2) ", "- ") from the kept paragraph.
    3. Drop a leading label line ("Refined prompt:") inside the kept paragraph.
    4. If >= 2 sentences and the last lacks terminal punctuation, drop the trailing fragment.
    5. Balance quotes AFTER fragment-dropping (matching pair, else orphaned edge quote).
    6. Reject degenerate results: preamble-only, fewer than 2 words, or an unterminated
       candidate longer than `_MAX_UNTERMINATED_WORDS` (max_new_tokens truncation junk).

    Returns `[instruction]` for accepted content, else `[]` (so callers drop the candidate).
    """
    raw = text.strip()
    if not raw:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    idx = 0
    while idx < len(paragraphs) - 1 and _is_preamble(paragraphs[idx]):
        idx += 1
    para = paragraphs[idx]
    if _is_preamble(para):
        return []
    para = _ENUMERATOR_RE.sub("", para, count=1)
    lines = para.splitlines()
    if len(lines) > 1 and _LABEL_LINE_RE.match(lines[0].strip()):
        para = "\n".join(lines[1:]).strip()
    parts = _SENTENCE_SPLIT_RE.split(para)
    if len(parts) >= 2 and not _TERMINAL_RE.search(parts[-1].strip()):
        para = " ".join(parts[:-1]).strip()
    para = _strip_quotes(para)
    if not para or _is_preamble(para):
        return []
    if len(para.split()) < 2:
        return []
    if not _TERMINAL_RE.search(para) and len(para.split()) > _MAX_UNTERMINATED_WORDS:
        return []
    return [para]
