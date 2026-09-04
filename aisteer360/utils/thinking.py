"""Splitting decoded continuations from reasoning models into thinking and answer segments."""
from typing import NamedTuple

DEFAULT_THINK_TAGS: tuple[str, str] = ("<think>", "</think>")


class ThinkingSplit(NamedTuple):
    """The two segments of one decoded continuation.

    Attributes:
        thinking: The reasoning segment, or None when no think tag is present.
        answer: The answer segment; the full text when no think tag is present, and the empty
            string when an opened thinking segment never closes.
    """

    thinking: str | None
    answer: str


def split_thinking(text: str, tags: tuple[str, str] = DEFAULT_THINK_TAGS) -> ThinkingSplit:
    """Split a decoded continuation into its thinking and answer segments.

    Matching is plain substring, case-sensitive. Let `open_tag, close_tag = tags`. The result
    depends on which tags are present:

    - Close tag present (open tag optional): the split is at the last occurrence of `close_tag`.
      `thinking` is everything before it, with one leading `open_tag` removed when the thinking
      segment starts with `open_tag` after leading whitespace. `answer` is everything after,
      left-stripped. The open tag is optional because thinking-mode chat templates commonly end
      the generation prompt with the open tag, so the continuation carries only the closing tag.
    - Open tag present, close tag absent (thinking truncated): `thinking` is everything after the
      first `open_tag` and `answer` is the empty string, since an unclosed thinking segment means
      no final answer was produced.
    - Neither tag present: `thinking` is None and `answer` is the full text, so the split is a
      no-op for non-reasoning models.

    An empty thinking segment yields `thinking == ""` (not None), since a tag was present and the
    model is in the reasoning regime.

    Args:
        text: The decoded continuation to split.
        tags: The `(open_tag, close_tag)` pair. Both entries must be non-empty strings.

    Returns:
        A `ThinkingSplit` with the `thinking` and `answer` segments.

    Raises:
        ValueError: If either tag entry is not a non-empty string.
    """
    open_tag, close_tag = tags
    if not (isinstance(open_tag, str) and open_tag) or not (isinstance(close_tag, str) and close_tag):
        raise ValueError("tags must be a pair of non-empty strings.")

    if close_tag in text:
        before, _, after = text.rpartition(close_tag)
        thinking = before
        if thinking.lstrip().startswith(open_tag):
            thinking = thinking.lstrip()[len(open_tag):]
        return ThinkingSplit(thinking=thinking, answer=after.lstrip())

    if open_tag in text:
        thinking = text.split(open_tag, 1)[1]
        return ThinkingSplit(thinking=thinking, answer="")

    return ThinkingSplit(thinking=None, answer=text)
