"""Meta-prompt templates for PRewrite.

Wording matches Appendix B of Kong et al., 2024 (arXiv:2401.08189) up to format-string adaptation for
`LLMMetaPromptProposer`. The `{seed}` placeholder receives the initial instruction.
"""

DEFAULT = (
    "You are a prompt engineer. Rewrite the following instruction to make it clearer, more "
    "specific, and more effective for a downstream language model, while preserving its intent.\n\n"
    "Original instruction:\n"
    "{seed}\n\n"
    "Rewritten instruction:"
)
