"""Tests for `extract_numeric_answer`: anchored extraction, canonicalization, and fallbacks."""
import pytest

from steerability.utils.answers import extract_numeric_answer


class TestExtractNumericAnswer:
    """Anchor precedence, value forms, canonicalization, and failure modes."""

    def test_answer_label_integer(self):
        assert extract_numeric_answer("Some work.\nAnswer: 5") == "5"

    def test_answer_label_plain_fraction(self):
        assert extract_numeric_answer("Answer: 2/3") == "2/3"

    def test_boxed_integer(self):
        assert extract_numeric_answer(r"The product is \[\boxed{1736}\]") == "1736"

    def test_boxed_latex_fraction(self):
        assert extract_numeric_answer(r"\boxed{\frac{2}{3}}") == "2/3"

    def test_boxed_dfrac(self):
        assert extract_numeric_answer(r"\boxed{\dfrac{4}{6}}") == "2/3"

    def test_equivalent_forms_share_one_key(self):
        forms = ["Answer: 4/6", r"\boxed{\frac{2}{3}}", "Answer: 2/3"]
        assert {extract_numeric_answer(form) for form in forms} == {"2/3"}

    def test_exact_decimal_merges_with_fraction(self):
        assert extract_numeric_answer("Answer: 0.5") == "1/2"

    def test_last_anchored_value_wins(self):
        assert extract_numeric_answer("Answer: 5\nWait, that is wrong.\nAnswer: 7") == "7"

    def test_bold_answer_label(self):
        assert extract_numeric_answer("**Answer:** 5") == "5"

    def test_dollar_wrapped_fraction(self):
        assert extract_numeric_answer(r"Answer: $\frac{1}{2}$") == "1/2"

    def test_lowercase_answer_label(self):
        assert extract_numeric_answer("answer: 12") == "12"

    def test_negative_fraction_reduced(self):
        assert extract_numeric_answer("Answer: -3/6") == "-1/2"

    def test_thousands_separator_ignored(self):
        assert extract_numeric_answer("Answer: 1,736") == "1736"

    def test_fallback_last_number(self):
        assert extract_numeric_answer("She needs 5 more dollars, not 10 dollars.") == "10"

    def test_fallback_fraction(self):
        assert extract_numeric_answer("so the probability is 2/3.") == "2/3"

    def test_anchored_value_beats_later_bare_number(self):
        assert extract_numeric_answer("Answer: 7\nas shown in step 3.") == "7"

    def test_no_number_returns_empty(self):
        assert extract_numeric_answer("no numeric content here") == ""

    def test_zero_denominator_returns_empty(self):
        assert extract_numeric_answer(r"\boxed{\frac{1}{0}}") == ""

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_input_returns_empty(self, text):
        assert extract_numeric_answer(text) == ""
