"""Tools for reading and characterizing model internals (the model's activations and statistics)
without steering them.

The root modules provide the shared signal-reading layer used by detection and by steering
estimation:

- `capture`: hooked/forward capture of hidden states at module boundaries.
- `encoding`: text-to-tensor tokenization entry points.
- `pooling`: token, span, and position aggregation.
- `render`: `ContrastivePairs` to model-ready text.
- `data`: contrastive and labeled-example data containers.
- `stats`: ambient activation statistics and residual-norm characterization.
- `fingerprint`: digests identifying the model a fitted artifact was estimated on.

Layering rules:

1. The root is signal acquisition and characterization; subpackages are capabilities built from
    the readings. Root modules never import from subpackages, and each `__init__.py` re-exports
    its own level only.
2. Modules here may import the `core` foundations (`base_args`, `types`, `core/utils`) and
    `steerability.utils`; they never import the `*_control` category packages at module level.
3. New signals (attention patterns, per-head capture, MLP activations) extend root modules; new
    capabilities become sibling subpackages.
"""
from .capture import HiddenStateLocation, layerwise_tokenwise_hidden
from .data import ContrastivePairs, LabeledExamples, as_contrastive_pairs, as_labeled_examples
from .encoding import tokenize_pairs, tokenize_texts
from .fingerprint import model_fingerprint
from .model_layout import (
    HeadGeometry,
    ModelLayout,
    head_geometry,
    register_layout_detector,
    resolve_model_layout,
    text_config,
)
from .pooling import (
    aggregate_condition_hidden,
    get_last_token_positions,
    masked_mean,
    pool_over_spans,
    select_at_positions,
    select_spans,
)
from .render import RenderedContrastive, render_contrastive
from .stats import ActivationStats, StatsSpec, measure_residual_norms

__all__ = [
    "ActivationStats",
    "ContrastivePairs",
    "HeadGeometry",
    "HiddenStateLocation",
    "LabeledExamples",
    "ModelLayout",
    "RenderedContrastive",
    "StatsSpec",
    "aggregate_condition_hidden",
    "as_contrastive_pairs",
    "as_labeled_examples",
    "get_last_token_positions",
    "head_geometry",
    "layerwise_tokenwise_hidden",
    "masked_mean",
    "measure_residual_norms",
    "model_fingerprint",
    "pool_over_spans",
    "register_layout_detector",
    "render_contrastive",
    "resolve_model_layout",
    "select_at_positions",
    "select_spans",
    "text_config",
    "tokenize_pairs",
    "tokenize_texts",
]
