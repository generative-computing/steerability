"""ActAdd (Activation Addition) control implementation."""
from __future__ import annotations

from aisteer360.algorithms.state_control.base import InterventionControl
from aisteer360.algorithms.state_control.common.selectors import FractionalDepthSelector
from aisteer360.algorithms.state_control.common.sources import SinglePairFit, _Precomputed
from aisteer360.algorithms.state_control.common.specs import Intervention, TokenScope
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform
from aisteer360.algorithms.state_control.common.transforms.base import unwrap_modifiers

from .args import ActAddArgs


class ActAdd(InterventionControl):
    """Activation Addition (ActAdd).

    Steers model behavior by adding a positional steering vector, computed from a single contrast
    pair of short prompts, to the residual stream at a single layer. The vector is extracted at
    the layer-input boundary of each layer and injected at the same boundary, so extraction and
    injection read and write the same residual point.

    The control is declarative: `_configure` maps the validated args onto one `Intervention`
    at the layer-input boundary with an `"all"` token scope, since spatial control comes from
    the transform's positional injection rather than the mask. Row `t` of the `[T, H]` vector
    is added at absolute token position `alignment + t`. When the window
    `[alignment, alignment + T)` lies within the prompt, injection happens entirely on the
    prefill pass; when it extends past a shorter prompt, the covered generated positions
    receive their rows once each.

    Reference:

    - "Steering Language Models With Activation Engineering"
    Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid
    [https://arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248)
    """

    Args = ActAddArgs
    supports_batching = False  # ActAdd uses positional alignment which breaks with left-padding

    def _configure(self):
        if self.steering_vector is not None:
            artifact = self.steering_vector.clone()
            if self.normalize_vector:
                for layer_id, direction in artifact.directions.items():
                    norms = direction.norm(dim=-1, keepdim=True)
                    artifact.directions[layer_id] = direction / (norms + 1e-8)
            source = _Precomputed(artifact)
        else:
            source = SinglePairFit(
                positive_prompt=self.positive_prompt,
                negative_prompt=self.negative_prompt,
                normalize=self.normalize_vector,
            )
        transform = AdditiveTransform(source, strength=self.multiplier, alignment=self.alignment, positional=True)
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        self._template = (Intervention(
            # heuristic default: ~20% depth (the paper uses layer 6/48 for GPT-2-XL)
            layers=(self.layer_id,) if self.layer_id is not None
                   else FractionalDepthSelector(fraction=0.2, minimum=1),
            transform=transform,
            scope=TokenScope("all"),
            boundary="layer_input",
        ),)

    @property
    def hook_only_hint(self) -> str:
        if self.layer_id == 0:
            return "layer 0 input edits have no intervention-spec form; run on the huggingface backend"
        return "positional directions have no intervention-spec form; run on the huggingface backend"

    @property
    def _layer_id(self) -> int | None:
        """The resolved behavior layer (None before `steer()`)."""
        return self.interventions[0].layers[0] if self.interventions else None

    @property
    def _steering_vector(self) -> SteeringVector | None:
        """The bound steering artifact as a `SteeringVector` view (None before `steer()`)."""
        if not self.interventions:
            return None
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        if getattr(core, "directions", None) is None:
            return None
        return SteeringVector(
            model_type="unknown",
            directions=core.directions,
            meta=core.artifact_meta or {},
        )
