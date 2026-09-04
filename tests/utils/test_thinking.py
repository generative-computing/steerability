"""Tests for `split_thinking`, one per normative case in the design (§5.1) plus edge cases."""
import pytest

from aisteer360.utils.thinking import DEFAULT_THINK_TAGS, ThinkingSplit, split_thinking


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
