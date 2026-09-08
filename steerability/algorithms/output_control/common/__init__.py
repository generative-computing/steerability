"""Output control component library.

Factors the output category into reusable components: candidate policies, per-candidate value
functions, full-vocabulary logit sources, sequence scorers, a segment-search driver, a phased
driver, composable stopping criteria, KV-cache utilities, and the `PrefixKeyedProcessor` base
for stateful logits processors.
"""
from steerability.algorithms.core.internals.data import LabeledExamples, as_labeled_examples

from .candidate_forward import CandidateForward
from .candidates import CandidatePolicy, select_candidates
from .criteria import BudgetTokens, StopOnSubstring, StopOnTokens
from .drivers import Fixed, Frontier, Generated, PhasedDriver, SearchDriver, SegmentProposer
from .logit_sources import AuxModelSource, BaseLogitSource, CallableSource, PromptVariantSource
from .processors import (
    ConstraintProcessor,
    ContrastiveMixtureProcessor,
    Normalize,
    PrefixKeyedProcessor,
    ValueGuidedProcessor,
)
from .scorers import MajorityVoteScorer, RewardModelScorer, SampleSequenceScorer, SequenceScorer
from .values import (
    BaseCandidateValue,
    CachedRewardModelValue,
    CallableValue,
    ClassifierValue,
    RewardModelValue,
    StepContext,
    SubspaceMarginValue,
)
