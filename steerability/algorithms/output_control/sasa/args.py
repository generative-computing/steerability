import os
from dataclasses import dataclass, field
from typing import Literal

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples


@dataclass
class SASAArgs(BaseArgs):
    """Arguments for `SASA` (subspace-margin guided sampling)."""

    beta: float = field(
        default=0.0,
        metadata={"help": "Scaling coefficient for value redistribution."},
    )
    wv_path: str | None = field(
        default=None,
        metadata={"help": "Path to a saved probe: a probe directory (safetensors plus JSON sidecar), a "
                          "`.probe` JSON file, or a legacy `.pt` tensor checkpoint."},
    )
    gen_wv_data: LabeledExamples | ContrastivePairs | dict | None = field(
        default=None,
        metadata={"help": "In-memory labeled data used to fit the probe. A `{'pos': [...], 'neg': [...]}` dict "
                          "or a `LabeledExamples` gives unpaired classes; a dict with a `'prompts'` key or a "
                          "`ContrastivePairs` gives paired prompt/response data (required for "
                          "prompt_format='chat_completion')."},
    )
    prompt_format: Literal["raw", "chat_completion", "chat_prompt"] = field(
        default="raw",
        metadata={"help": "How fit data is rendered before capture. 'chat_completion' renders each pair as a "
                          "user turn plus the response and requires paired data with prompts."},
    )
    gen_wv_length: int | None = field(
        default=-1,
        metadata={"help": "The maximum number of samples per class used to fit the probe when wv_path is unset."}
    )
    gen_wv_batch_size: int | None = field(
        default=4,
        metadata={"help": "The batch size used to fit the probe when wv_path is unset."}
    )
    candidate_policy: Literal["surviving", "top_p", "top_k"] = field(
        default="surviving",
        metadata={"help": "Which tokens are scored per step. 'surviving' scores every token earlier processors "
                          "left finite; 'top_p' scores the nucleus of the raw logits (the paper's setting); "
                          "'top_k' scores the top-k."},
    )
    top_p: float | None = field(
        default=None,
        metadata={"help": "Nucleus threshold for candidate_policy='top_p' (0 < top_p <= 1)."},
    )
    top_k: int | None = field(
        default=None,
        metadata={"help": "Candidate count for candidate_policy='top_k' (top_k >= 1)."},
    )
    max_candidates: int | None = field(
        default=None,
        metadata={"help": "Optional clamp on the candidate set (top-N by score) to bound the per-step model "
                          "forward. None (default) leaves the policy's candidate set unclamped."}
    )

    # validation
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
        if self.wv_path is not None and not (
            os.path.isdir(self.wv_path) or self.wv_path.endswith((".pt", ".probe"))
        ):
            raise ValueError(
                "wv_path must be a probe directory, a .pt tensor checkpoint, or a .probe JSON file."
            )
        if self.wv_path is None and self.gen_wv_batch_size < 0:
            raise ValueError("'gen_wv_batch_size' must be non-negative.")

        if self.candidate_policy == "top_p":
            if self.top_p is None or not 0.0 < self.top_p <= 1.0:
                raise ValueError("candidate_policy='top_p' requires 0 < top_p <= 1.")
            if self.top_k is not None:
                raise ValueError("candidate_policy='top_p' does not use top_k; leave it unset.")
        elif self.candidate_policy == "top_k":
            if self.top_k is None or self.top_k < 1:
                raise ValueError("candidate_policy='top_k' requires top_k >= 1.")
            if self.top_p is not None:
                raise ValueError("candidate_policy='top_k' does not use top_p; leave it unset.")
        else:  # surviving
            if self.top_p is not None or self.top_k is not None:
                raise ValueError("candidate_policy='surviving' does not use top_p or top_k; leave them unset.")

        if self.prompt_format == "chat_completion" and self.gen_wv_data is not None:
            paired = isinstance(self.gen_wv_data, ContrastivePairs) or (
                isinstance(self.gen_wv_data, dict) and "prompts" in self.gen_wv_data
            )
            if not paired:
                raise ValueError(
                    "prompt_format='chat_completion' requires paired data with prompts: pass a "
                    "ContrastivePairs or a dict carrying a 'prompts' key via gen_wv_data."
                )
