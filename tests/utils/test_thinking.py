"""Tests for the reasoning-split utilities: the substring splitter `split_thinking`, the id-level
splitter `split_thinking_ids`, and the `resolve_split_mode` auto-resolver."""
import pytest

from steerability.utils.thinking import (
    DEFAULT_THINK_TAGS,
    ThinkingSplit,
    find_subsequence,
    resolve_split_mode,
    split_thinking,
    split_thinking_ids,
)
from tests.utils.tiny_models import reasoning_tag_tokenizer

TAGS = ("<open>", "<close>")


def encode_continuation(tokenizer, *words: str) -> list[int]:
    """Encode a whitespace-joined sequence of vocabulary words and tags into continuation ids."""
    ids: list[int] = []
    for word in words:
        ids.extend(tokenizer.encode(word, add_special_tokens=False))
    return ids


class TestSplitThinking:
    """Each numbered case in the splitter contract, plus last-occurrence and validation edges."""

    def test_close_tag_present_with_open_tag(self):
        # case 1: both tags present -> thinking is the middle, answer is left-stripped tail
        result = split_thinking("<think>reasoning here</think>Final answer.")
        assert result == ThinkingSplit(thinking="reasoning here", answer="Final answer.")

    def test_close_tag_present_open_tag_absent(self):
        # case 1: close tag only (generation prompt ended with the open tag)
        result = split_thinking("reasoning here</think>Final answer.")
        assert result == ThinkingSplit(thinking="reasoning here", answer="Final answer.")

    def test_open_tag_present_close_tag_absent(self):
        # case 2: truncated thinking -> thinking retained, answer empty
        result = split_thinking("<think>reasoning cut off by length")
        assert result == ThinkingSplit(thinking="reasoning cut off by length", answer="")

    def test_neither_tag_present(self):
        # case 3: strict no-op for non-reasoning models
        result = split_thinking("just a plain answer")
        assert result == ThinkingSplit(thinking=None, answer="just a plain answer")

    def test_empty_thinking_block_yields_empty_string_not_none(self):
        # case 4: a tag was present, so thinking is "" (reasoning regime), not None
        result = split_thinking("<think></think>answer")
        assert result == ThinkingSplit(thinking="", answer="answer")

    def test_empty_thinking_close_only_yields_empty_string(self):
        # case 4 variant: close tag only, nothing before it
        result = split_thinking("</think>answer")
        assert result == ThinkingSplit(thinking="", answer="answer")

    def test_answer_is_left_stripped(self):
        result = split_thinking("<think>r</think>\n   spaced answer")
        assert result.answer == "spaced answer"

    def test_last_occurrence_with_two_close_tags(self):
        # split at the LAST close tag; a repeated block folds into thinking
        result = split_thinking("<think>a</think>b</think>ANSWER")
        assert result == ThinkingSplit(thinking="a</think>b", answer="ANSWER")

    def test_leading_whitespace_then_open_tag_stripped(self):
        # leading whitespace before the open tag is stripped along with the open tag
        result = split_thinking("  \n<think>reason</think>  ANS")
        assert result == ThinkingSplit(thinking="reason", answer="ANS")

    def test_open_tag_with_trailing_newline_preserved_in_thinking(self):
        # only the open tag is removed; interior whitespace stays
        result = split_thinking("<think>\nreason\n</think>\nANS")
        assert result == ThinkingSplit(thinking="\nreason\n", answer="ANS")

    def test_custom_tags(self):
        result = split_thinking("[R]think[/R]answer", tags=("[R]", "[/R]"))
        assert result == ThinkingSplit(thinking="think", answer="answer")

    def test_default_tags_value(self):
        assert DEFAULT_THINK_TAGS == ("<think>", "</think>")

    @pytest.mark.parametrize("tags", [("", "</think>"), ("<think>", ""), ("", "")])
    def test_empty_tag_string_raises(self, tags):
        with pytest.raises(ValueError, match="non-empty strings"):
            split_thinking("x", tags=tags)


class TestSplitThinkingOpenedAtStart:
    """`opened_at_start` in text mode: a tagless continuation is unclosed reasoning, not an answer."""

    def test_no_tags_opened_at_start_is_unclosed_reasoning(self):
        # case (iii): the prompt opened the channel, nothing closed it -> all reasoning, empty answer
        result = split_thinking("still thinking with no tags", opened_at_start=True)
        assert result == ThinkingSplit(thinking="still thinking with no tags", answer="")

    def test_close_present_opened_at_start_splits_normally(self):
        # case (ii): the close tag still drives the split when the channel was opened by the prompt
        result = split_thinking("reasoning</think>answer", opened_at_start=True)
        assert result == ThinkingSplit(thinking="reasoning", answer="answer")

    def test_opened_at_start_false_no_tags_is_plain_answer(self):
        # case (iv): default flag leaves a tagless continuation as a plain answer
        result = split_thinking("plain answer", opened_at_start=False)
        assert result == ThinkingSplit(thinking=None, answer="plain answer")


class TestResolveSplitMode:
    """`resolve_split_mode` routes by whether the delimiters survive `skip_special_tokens=True`."""

    def test_ordinary_tags_resolve_to_text(self):
        tokenizer = reasoning_tag_tokenizer(ordinary_tags=TAGS)
        assert resolve_split_mode(tokenizer, TAGS) == "text"

    def test_special_tags_resolve_to_tokens(self):
        tokenizer = reasoning_tag_tokenizer(special_tags=TAGS)
        assert resolve_split_mode(tokenizer, TAGS) == "tokens"

    def test_one_special_one_ordinary_resolves_to_tokens(self):
        tokenizer = reasoning_tag_tokenizer(special_tags=("<open>",), ordinary_tags=("<close>",))
        assert resolve_split_mode(tokenizer, TAGS) == "tokens"

    def test_empty_tag_string_raises(self):
        tokenizer = reasoning_tag_tokenizer(special_tags=TAGS)
        with pytest.raises(ValueError, match="non-empty strings"):
            resolve_split_mode(tokenizer, ("", "<close>"))

    def test_tag_encoding_to_empty_sequence_raises(self):
        # a tag the tokenizer drops entirely (whitespace under the whitespace pre-tokenizer)
        tokenizer = reasoning_tag_tokenizer(special_tags=("<open>",))
        with pytest.raises(ValueError, match="empty id sequence"):
            resolve_split_mode(tokenizer, ("<open>", "   "))


class TestFindSubsequence:
    def test_first_occurrence_and_start_offset(self):
        assert find_subsequence([9, 1, 2, 3, 1, 2], [1, 2]) == 1
        assert find_subsequence([9, 1, 2, 3, 1, 2], [1, 2], start=2) == 4

    def test_absent_and_empty_needle_return_minus_one(self):
        assert find_subsequence([1, 2, 3], [4]) == -1
        assert find_subsequence([1, 2, 3], []) == -1


class TestSplitThinkingIds:
    """The §2 matrix at the token-id level, with the delimiters as special tokens."""

    @pytest.fixture
    def tokenizer(self):
        return reasoning_tag_tokenizer(special_tags=TAGS)

    def test_case_i_open_reasoning_close_answer(self, tokenizer):
        ids = encode_continuation(tokenizer, "<open>", "R", "<close>", "A")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="R", answer="A")

    def test_case_ii_close_only_default_flag_splits(self, tokenizer):
        # the open subsequence is optional, as in text mode: a template that opens the channel in
        # the generation prompt leaves only the close in the continuation
        ids = encode_continuation(tokenizer, "R", "<close>", "A")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="R", answer="A")

    def test_case_ii_close_only_opened_at_start(self, tokenizer):
        ids = encode_continuation(tokenizer, "R", "<close>", "A")
        result = split_thinking_ids(ids, tokenizer, TAGS, opened_at_start=True)
        assert result == ThinkingSplit(thinking="R", answer="A")

    def test_case_iii_opened_at_start_no_close(self, tokenizer):
        ids = encode_continuation(tokenizer, "R", "plan")
        result = split_thinking_ids(ids, tokenizer, TAGS, opened_at_start=True)
        assert result == ThinkingSplit(thinking="R plan", answer="")

    def test_case_iii_open_present_no_close(self, tokenizer):
        ids = encode_continuation(tokenizer, "<open>", "R", "plan")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="R plan", answer="")

    def test_case_iv_no_tags(self, tokenizer):
        ids = encode_continuation(tokenizer, "plan", "answer")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking=None, answer="plan answer")

    def test_empty_thinking_yields_empty_string_not_none(self, tokenizer):
        ids = encode_continuation(tokenizer, "<open>", "<close>", "A")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="", answer="A")

    def test_no_delimiter_residue_in_answer(self, tokenizer):
        ids = encode_continuation(tokenizer, "<open>", "R", "<close>", "A")
        result = split_thinking_ids(ids, tokenizer, TAGS)
        assert "<open>" not in result.answer and "<close>" not in result.answer

    def test_first_close_wins_later_close_stays_in_answer(self, tokenizer):
        # first-close semantics: the answer begins after the first close; a later close's token
        # decodes back into the answer verbatim
        ids = encode_continuation(tokenizer, "<open>", "R", "<close>", "x", "<close>", "A")
        result = split_thinking_ids(ids, tokenizer, TAGS)
        assert result.thinking == "R"
        assert result.answer.split() == ["x", "A"]

    def test_text_before_open_joins_answer(self, tokenizer):
        # output emitted before the channel opened is answer content, not reasoning
        ids = encode_continuation(tokenizer, "pre", "<open>", "R", "<close>", "A")
        result = split_thinking_ids(ids, tokenizer, TAGS)
        assert result.thinking == "R"
        assert result.answer.split() == ["pre", "A"]

    def test_trailing_pad_ids_do_not_disturb_the_split(self, tokenizer):
        ids = encode_continuation(tokenizer, "<open>", "R", "<close>", "A")
        ids = ids + [tokenizer.pad_token_id, tokenizer.pad_token_id]
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="R", answer="A")

    def test_ordinary_tags_split_at_id_level(self):
        # token mode is agnostic to whether the delimiters are special; ordinary tags split too
        tokenizer = reasoning_tag_tokenizer(ordinary_tags=TAGS)
        ids = encode_continuation(tokenizer, "<open>", "R", "<close>", "A")
        assert split_thinking_ids(ids, tokenizer, TAGS) == ThinkingSplit(thinking="R", answer="A")

    def test_empty_tag_string_raises(self, tokenizer):
        ids = encode_continuation(tokenizer, "R")
        with pytest.raises(ValueError, match="non-empty strings"):
            split_thinking_ids(ids, tokenizer, ("", "<close>"))

    def test_tag_encoding_to_empty_sequence_raises(self, tokenizer):
        ids = encode_continuation(tokenizer, "R")
        with pytest.raises(ValueError, match="empty id sequence"):
            split_thinking_ids(ids, tokenizer, ("<open>", "   "))

    def test_gemma_shaped_close_only_acceptance(self):
        # §8 acceptance: a Gemma-flagged tokenizer, opened_at_start, R <close> A -> R, A, no residue
        tags = ("<|channel>thought\n", "<channel|>")
        tokenizer = reasoning_tag_tokenizer(special_tags=tags)
        ids = (
            encode_continuation(tokenizer, "R")
            + tokenizer.encode("<channel|>", add_special_tokens=False)
            + encode_continuation(tokenizer, "A")
        )
        assert resolve_split_mode(tokenizer, tags) == "tokens"
        result = split_thinking_ids(ids, tokenizer, tags, opened_at_start=True)
        assert result == ThinkingSplit(thinking="R", answer="A")
        assert "<channel|>" not in result.answer
