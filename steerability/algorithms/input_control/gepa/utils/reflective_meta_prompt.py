"""Reflective meta-prompt for GEPA.

`GEPA_DEFAULT` reproduces the reflective instruction-proposal prompt of Agrawal et al., 2025
(Appendix B; matching the open-source gepa-ai/gepa reference wording), adapted to the
`LLMMetaPromptProposer` format-string convention. The placeholders are:

  - `{seed}`: the current instruction for the module being optimized (the parent prompt).
  - `{records}`: rendered per-example reflective records (inputs, outputs, feedback).

The matching parser (`parse_fenced_or_whole`) extracts the content of the first ``` fenced block.
Per-module context (which module is being optimized) can be layered in by a caller-supplied template;
the default omits it to match the reference.
"""
from __future__ import annotations

from typing import Any

GEPA_DEFAULT = (
    "I provided an assistant with the following instructions to perform a task for me:\n"
    "```\n"
    "{seed}\n"
    "```\n\n"
    "The following are examples of different task inputs provided to the assistant along with the "
    "assistant's response for each of them, and some feedback on how the assistant's response could be "
    "better:\n"
    "```\n"
    "{records}\n"
    "```\n\n"
    "Your task is to write a new instruction for the assistant.\n\n"
    "Read the inputs carefully and identify the input format and infer detailed task description about "
    "the task I wish to solve with the assistant.\n\n"
    "Read all the assistant responses and the corresponding feedback. Identify all niche and domain "
    "specific factual information about the task and include it in the instruction, as a lot of it may "
    "not be available to the assistant in the future. The assistant may have utilized a generalizable "
    "strategy to solve the task, if so, include that in the instruction as well.\n\n"
    "Provide the new instructions within ``` blocks."
)


def render_records(records: list[dict[str, Any]]) -> str:
    """Render a per-component list of reflective records into a readable block.

    Each record may carry the keys `Inputs`, `Generated Output`, and `Feedback`. Missing
    fields render with a sensible placeholder. Returns `(no records)` for an empty list.
    """
    if not records:
        return "(no records)"
    lines: list[str] = []
    for record in records:
        inputs = record.get("Inputs", "(no inputs)")
        output = record.get("Generated Output", "(no output)")
        feedback = record.get("Feedback", "(no feedback)")
        lines.append(f"- Inputs: {inputs}")
        lines.append(f"  Generated Output: {output}")
        lines.append(f"  Feedback: {feedback}")
    return "\n".join(lines)
