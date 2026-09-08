"""Self-refinement meta-prompt for CPO.

Follows Appendix EC.3 of Chen et al., 2026 (arXiv:2602.01711), with a single `{seed}` placeholder
so it composes with the common `LLMMetaPromptProposer`.
"""

CPO_DEFAULT = (
    "You are an expert prompt engineer. Refine the prompt below to make it more effective at "
    "the task it describes. Preserve intent; vary wording, structure, and specificity. "
    "Write a single, complete, ready-to-use instruction for the specific task -- not a generic "
    "template. Never include bracketed placeholders (e.g., [topic]); do not explain your changes. "
    "Reply with the refined prompt only, in one or two sentences.\n\n"
    "Current prompt:\n"
    "{seed}\n\n"
    "Refined prompt:"
)
