"""Extracting a canonical numeric answer from a decoded model response."""
import re
from fractions import Fraction

_ANCHORED_VALUE = re.compile(
    r"(?:Answer:|\\boxed\{)[\s$*]*\\?"
    r"(?:d?frac\{(-?\d+)\}\{(-?\d+)\}|(-?\d+)\s*/\s*(-?\d+)|(-?\d+(?:\.\d+)?))",
    re.IGNORECASE,
)
_BARE_VALUE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?")


def extract_numeric_answer(text: str) -> str:
    """Extract the final numeric answer from a response and canonicalize it.

    The last value anchored on an `Answer:` label or a `\\boxed{...}` wrapper wins. Anchored values
    may be integers, decimals, plain fractions (`2/3`), or LaTeX fractions (`\\frac{2}{3}`,
    `\\dfrac{2}{3}`), optionally wrapped in `$` or `*` markers. When no anchored value is present,
    the last number-like token in the text is used instead. Thousands separators are ignored.

    The extracted value is canonicalized through `fractions.Fraction`, so equivalent forms map to
    one string (`4/6`, `\\frac{2}{3}`, and `2/3` all yield `"2/3"`; exact decimals merge with their
    fraction, e.g. `0.5` yields `"1/2"`). This makes the result usable as an agreement key, e.g. as
    the `answer_extractor` of `MajorityVoteScorer`.

    Args:
        text: The decoded response to extract from.

    Returns:
        The canonical answer string, or the empty string when no value is found or the anchored
        value does not parse (e.g. a zero denominator).
    """
    text = text.replace(",", "")

    matches = list(_ANCHORED_VALUE.finditer(text))
    if matches:
        frac_num, frac_den, slash_num, slash_den, plain = matches[-1].groups()
        try:
            if frac_num is not None:
                return str(Fraction(int(frac_num), int(frac_den)))
            if slash_num is not None:
                return str(Fraction(int(slash_num), int(slash_den)))
            return str(Fraction(plain))
        except (ZeroDivisionError, ValueError):
            return ""

    tokens = _BARE_VALUE.findall(text)
    if not tokens:
        return ""
    try:
        return str(Fraction(tokens[-1].replace(" ", "")))
    except (ZeroDivisionError, ValueError):
        return ""
