"""Configuration for fitting and searching steering artifacts.

Holds the fit-time vocabulary shared by state control components: `VectorTrainSpec` describes
how direction vectors are extracted, `ConditionSearchSpec` describes how condition points are
searched, and the comparator vocabulary (`Comparator`, `CompMode`) carries the gate-comparison
semantics (`"ge"` opens when score >= threshold, `"le"` when score <= threshold). These specs
describe how artifacts are produced; the intervention IR in `specs.py` describes how bound
artifacts are applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from steerability.algorithms.core.internals.capture import HiddenStateLocation
from steerability.utils.rendering import PromptFormat

Comparator = Literal["ge", "le"]
CompMode = Literal["mean", "last"]


@dataclass(frozen=True)
class VectorTrainSpec:
    """Configuration for how to train/extract direction vectors.

    Attributes:
        method: Extraction algorithm.
            "pca_pairwise" uses PCA on paired differences of hidden states.
            "pca_center" uses PCA on all positive/negative hidden states centered
                by their grand mean (the CAST extraction from the paper).
            "mean_diff" uses the mean difference of hidden states (CAA method).
        accumulate: How to select hidden state spans for aggregation.
            "all" uses the full sequence.
            "suffix-only" uses only the portion after the shared prompt.
            "last_token" uses only the final non-pad token position.
        batch_size: Batch size for hidden state extraction forward passes.
        prompt_format: How to render contrastive examples into model-ready text
            (via `render_for_model`); the rendered string is tokenized with
            `add_special_tokens=False`.
            "chat_completion" renders `prompts` as user turns and appends
            positives/negatives as completions (prompt+answer pairs, e.g. CAA);
            falls back to "raw" when no `prompts` are provided.
            "chat_prompt" renders each positive/negative as a standalone user turn
            (standalone-prompt contrasts, e.g. the CAST condition); matches the
            inference rendering exactly.
            "raw" concatenates `prompts` + text verbatim with no chat template
            (base-model methods and standalone statements).
        location: Residual-stream boundary each layer key maps to. `outputs.hidden_states` is a
            tuple of `num_layers + 1` tensors: index 0 is the embedding output (the input to layer
            0) and index `i` is the output of layer `i - 1`.
            "layer_output" (default): key `l` maps to the output of layer `l`
            (`hidden_states[l + 1]`), the boundary hooked by controls that intervene on the layer
            output.
            "layer_input": key `l` maps to the input of layer `l`, i.e. the output of layer `l - 1`
            (`hidden_states[l]`), the boundary observed by layer pre-hooks.
            A vector fit at one boundary is a distinct artifact from one fit at the other, so fit it
            at the boundary where the consuming control scores or applies it.
    """

    method: Literal["pca_pairwise", "pca_center", "mean_diff"] = "pca_pairwise"
    accumulate: Literal["all", "suffix-only", "last_token"] = "all"
    batch_size: int = 8
    prompt_format: PromptFormat = "chat_completion"
    location: HiddenStateLocation = "layer_output"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.prompt_format not in ("raw", "chat_completion", "chat_prompt"):
            raise ValueError(
                f"prompt_format must be one of raw/chat_completion/chat_prompt, got {self.prompt_format!r}."
            )
        if self.location not in ("layer_output", "layer_input"):
            raise ValueError(
                f"location must be 'layer_output' or 'layer_input', got {self.location!r}."
            )


@dataclass(frozen=True)
class ConditionSearchSpec:
    """Configuration for automatic condition point search.

    Attributes:
        auto_find: If True, run the search during steer(). If False, the
            user must provide condition_layer_ids and threshold manually.
        candidate_layers: Explicit layer ids to search over. If None, use
            layer_range.
        layer_range: 0-based (start, end) half-open range of layers to consider. Ignored if
            candidate_layers is set. Defaults to all layers.
        threshold_range: (min, max) for the threshold grid search (half-open, step-exact).
        threshold_step: Step size for the threshold grid.
    """

    auto_find: bool = True
    candidate_layers: Sequence[int] | None = None
    layer_range: tuple[int, int] | None = None
    threshold_range: tuple[float, float] = (0.0, 1.0)
    threshold_step: float = 0.01

    def __post_init__(self):
        lo, hi = self.threshold_range
        if lo >= hi:
            raise ValueError(f"threshold_range ({lo}, {hi}): min must be < max.")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be > 0.")
