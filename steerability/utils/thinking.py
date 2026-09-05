"""Splitting continuations from reasoning models into thinking and answer segments.

The split runs in one of two modes. Text mode is a substring split on the decoded continuation.
Token mode is an id-level split on the continuation ids, decoding each segment afterwards; it exists
because some tokenizers encode the reasoning delimiters as special tokens that `skip_special_tokens=True`
strips before a substring split could see them. `resolve_split_mode` picks between the two for a given
tokenizer and tag pair.
"""
from typing import Literal, NamedTuple, Sequence

from transformers import PreTrainedTokenizerBase

DEFAULT_THINK_TAGS: tuple[str, str] = ("<think>", "</think>")


class ThinkingSplit(NamedTuple):
    """The two segments of one continuation.

    Attributes:
        thinking: The reasoning segment, or None when no think tag is present.
        answer: The answer segment; the full text when no think tag is present, and the empty
            string when an opened thinking segment never closes.
    """

    thinking: str | None
    answer: str


def split_thinking(
    text: str,
    tags: tuple[str, str] = DEFAULT_THINK_TAGS,
    *,
    opened_at_start: bool = False,
) -> ThinkingSplit:
    """Split a decoded continuation into its thinking and answer segments by substring matching.

    Matching is plain substring, case-sensitive. Let `open_tag, close_tag = tags`. The result
    depends on which tags are present and on `opened_at_start`:

    - Close tag present (open tag optional): the split is at the last occurrence of `close_tag`.
      `thinking` is everything before it, with one leading `open_tag` removed when the thinking
      segment starts with `open_tag` after leading whitespace. `answer` is everything after,
      left-stripped. The open tag is optional because thinking-mode chat templates commonly end
      the generation prompt with the open tag, so the continuation carries only the closing tag.
    - Open tag present, close tag absent (thinking truncated): `thinking` is everything after the
      first `open_tag` and `answer` is the empty string, since an unclosed thinking segment means
      no final answer was produced.
    - Neither tag present, `opened_at_start=False`: `thinking` is None and `answer` is the full
      text, so the split is a no-op for non-reasoning models.
    - Neither tag present, `opened_at_start=True`: the generation prompt opened the channel and it
      never closed, so `thinking` is the full text and `answer` is the empty string.

    An empty thinking segment yields `thinking == ""` (not None), since a tag was present or the
    channel was opened by the prompt.

    The split is at the last occurrence of `close_tag`, so any earlier `close_tag` occurrences fold
    into `thinking` and any `open_tag`/`close_tag` occurrences after the split stay in `answer`
    verbatim.

    Args:
        text: The decoded continuation to split.
        tags: The `(open_tag, close_tag)` pair. Both entries must be non-empty strings.
        opened_at_start: Whether the generation prompt already opened the reasoning channel, so a
            continuation carrying neither tag is treated as unclosed reasoning rather than a plain
            answer.

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

    if opened_at_start:
        return ThinkingSplit(thinking=text, answer="")

    return ThinkingSplit(thinking=None, answer=text)


def _encode_tag(tokenizer: PreTrainedTokenizerBase, tag: str) -> list[int]:
    """Encode one tag to its id sequence with `add_special_tokens=False`.

    Raises:
        ValueError: If the tag encodes to an empty id sequence under this tokenizer.
    """
    ids = tokenizer.encode(tag, add_special_tokens=False)
    if not ids:
        raise ValueError(
            f"reasoning tag {tag!r} encodes to an empty id sequence under tokenizer "
            f"{tokenizer.__class__.__name__}; it cannot be matched at the token level."
        )
    return list(ids)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    """Return the first index at or after `start` where `needle` occurs in `haystack`, or -1.

    An empty `needle` never matches (returns -1).
    """
    needle = list(needle)
    span = len(needle)
    if span == 0:
        return -1
    last = len(haystack) - span
    for index in range(start, last + 1):
        if list(haystack[index:index + span]) == needle:
            return index
    return -1


def resolve_split_mode(
    tokenizer: PreTrainedTokenizerBase,
    tags: tuple[str, str] = DEFAULT_THINK_TAGS,
) -> Literal["text", "tokens"]:
    """Resolve `"auto"` to `"text"` or `"tokens"` for a tokenizer and tag pair.

    Each tag is encoded with `add_special_tokens=False` and the result decoded with
    `skip_special_tokens=True`. When both tags round-trip to their exact original strings, a
    substring split on the decoded continuation sees the delimiters, so the mode is `"text"`.
    Otherwise at least one delimiter is stripped by `skip_special_tokens=True` (it is a special
    token) and the split must run on the ids, so the mode is `"tokens"`.

    Args:
        tokenizer: The pipeline tokenizer whose vocabulary decides the delimiter treatment.
        tags: The `(open_tag, close_tag)` pair. Both entries must be non-empty strings.

    Returns:
        `"text"` when both tags survive `skip_special_tokens=True`, `"tokens"` otherwise.

    Raises:
        ValueError: If either tag entry is not a non-empty string, or a tag encodes to an empty id
            sequence (checked so `"auto"` never routes an unmatchable tag into token mode).
    """
    open_tag, close_tag = tags
    if not (isinstance(open_tag, str) and open_tag) or not (isinstance(close_tag, str) and close_tag):
        raise ValueError("tags must be a pair of non-empty strings.")
    open_ids = _encode_tag(tokenizer, open_tag)
    close_ids = _encode_tag(tokenizer, close_tag)
    open_survives = tokenizer.decode(open_ids, skip_special_tokens=True) == open_tag
    close_survives = tokenizer.decode(close_ids, skip_special_tokens=True) == close_tag
    return "text" if open_survives and close_survives else "tokens"


def split_thinking_ids(
    output_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    tags: tuple[str, str] = DEFAULT_THINK_TAGS,
    *,
    opened_at_start: bool = False,
) -> ThinkingSplit:
    """Split a continuation into thinking and answer segments at the token-id level.

    Each tag is the id sequence obtained by encoding it with `add_special_tokens=False`, and
    matching is first-occurrence subsequence search over `output_ids`. The delimiter ids belong to
    neither segment; each resulting segment is decoded with `skip_special_tokens=True`. Let
    `open_ids, close_ids = tags` encoded this way:

    - `opened_at_start=False`, open subsequence found: the reasoning starts after the open
      subsequence. Any ids preceding the open subsequence are prepended to the answer, since output
      emitted before the channel opened is not reasoning.
    - `opened_at_start=False`, no open subsequence but a close subsequence: the reasoning starts
      at position 0. The open tag is optional because thinking-mode chat templates commonly end
      the generation prompt with it, so the continuation carries only the close.
    - `opened_at_start=True`: the continuation begins inside the channel, so the reasoning starts at
      position 0 and no open subsequence is searched.
    - Once inside the channel, the split is at the first close subsequence: `thinking` is the ids
      up to it and `answer` is the ids after it. Close subsequences after the first stay in the
      answer verbatim.
    - Inside the channel with no close subsequence (truncation, or a caller stop string halted
      generation before the close was emitted): `thinking` is the rest of the ids and `answer` is
      the empty string.
    - `opened_at_start=False` and neither subsequence: `thinking` is None and `answer` is the full
      decoded continuation.

    An empty thinking segment yields `thinking == ""` (not None), since the channel was entered.

    A multi-token ordinary-text portion of a composed tag (for example the `thought\\n` following a
    special open token) can tokenize context-dependently, so its id sequence in isolation need not
    be the id sequence it forms inside the continuation. When `opened_at_start=True` the open tag is
    never searched, so this affects the open tag only when the channel is opened within the
    continuation.

    Args:
        output_ids: The continuation token ids (one row, the prompt excluded). Trailing padding or
            eos ids may be present; delimiter matching is unaffected since those ids do not match a
            delimiter, and each segment is decoded with `skip_special_tokens=True`.
        tokenizer: The tokenizer used to encode the tags and decode the segments.
        tags: The `(open_tag, close_tag)` pair. Both entries must be non-empty strings.
        opened_at_start: Whether the generation prompt already opened the reasoning channel. The
            flag matters only for a continuation carrying neither tag, which is classified as
            unclosed reasoning when set and as a plain answer otherwise.

    Returns:
        A `ThinkingSplit` with the decoded `thinking` and `answer` segments.

    Raises:
        ValueError: If either tag entry is not a non-empty string, or a tag encodes to an empty id
            sequence under this tokenizer.
    """
    open_tag, close_tag = tags
    if not (isinstance(open_tag, str) and open_tag) or not (isinstance(close_tag, str) and close_tag):
        raise ValueError("tags must be a pair of non-empty strings.")

    ids = list(output_ids)
    close_ids = _encode_tag(tokenizer, close_tag)

    def decode(segment: list[int]) -> str:
        return tokenizer.decode(segment, skip_special_tokens=True)

    answer_prefix: list[int] = []
    if opened_at_start:
        reasoning_start = 0
    else:
        open_ids = _encode_tag(tokenizer, open_tag)
        open_at = find_subsequence(ids, open_ids)
        if open_at == -1:
            if find_subsequence(ids, close_ids) == -1:
                return ThinkingSplit(thinking=None, answer=decode(ids))
            # the open tag is optional: a generation prompt that opens the channel leaves only the
            # close subsequence in the continuation
            reasoning_start = 0
        else:
            answer_prefix = ids[:open_at]
            reasoning_start = open_at + len(open_ids)

    close_at = find_subsequence(ids, close_ids, start=reasoning_start)
    if close_at == -1:
        return ThinkingSplit(thinking=decode(ids[reasoning_start:]), answer="")

    thinking = decode(ids[reasoning_start:close_at])
    answer = decode(answer_prefix + ids[close_at + len(close_ids):])
    return ThinkingSplit(thinking=thinking, answer=answer)
