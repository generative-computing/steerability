# Probes

!!! note
    This document provides a conceptual overview of detection in the toolkit. For the full API, please see the
    reference pages on [internals](../reference/algorithms/core/internals.md) and
    [probes](../reference/algorithms/core/probes.md). For a worked example, please see the notebook on
    [routed decoding](../examples/notebooks/recipes/routed_decoding.ipynb).

Some steering workflows depend on detection, i.e., reading the model's internal state to decide whether a concept is
present in a prompt, e.g., recognizing that a question asks for medical advice so it can be routed to a referral
instead of answered. The toolkit implements detection with probes and keeps a small vocabulary that forms a ladder:
probes measure (hidden states become scores and boolean decisions), gates decide (a binary admit/deny inside a
steered intervention, covered under [state control](controls.md#state-control)), and routers compose named decisions
into a categorical choice of action (inside [routed decoding](controls.md#output-control)). This page covers the
measurement rung.


## Probes and probe sets

A `Probe` is a small linear classifier over the model's hidden states. It is fit on contrastive pairs (prompts where
the concept is present and prompts where it is absent), and at inference it pools a prompt's hidden states, takes a
dot product with its weight vector, adds a bias, and decides `score >= 0`.

Two properties define the artifact:

- **Canonical polarity**: a fitted probe is always oriented so that positives score high, and its operating threshold
  is folded into the bias during calibration. There is no comparator or threshold to configure; the decision is
  always `score >= 0`.
- **Model-free**: a probe holds only weights, a bias, and provenance metadata. It runs no forward passes itself, so
  it can be saved, loaded, and applied to cached activations offline.

Reads over a live model go through a `ProbeSet`, which scores every named probe in one read-only forward and returns
a `ProbeReadings` of per-prompt signed scores and boolean decisions. The read never edits hidden states, so probing
leaves generation untouched.


## Fitting and calibration

Fitting takes two datasets with different jobs. The direction is fit on `data`, discriminative pairs that isolate the
concept, e.g., medical questions against questions from neighboring domains. The operating point is then calibrated
on `calibration_data`, a broader set that also covers the traffic the probe must stay closed on, e.g., general
questions. When no calibration set is given, the fit pairs serve both roles.

Raw directions in activation space make poor detectors because activations share a large common component and a few
outlier coordinates dominate dot products. The default fitting method (`"lda"`) therefore standardizes features with
ambient activation statistics before taking the difference in class means. The statistics (`ActivationStats`) are
estimated once per model from generic texts and can be saved and reused across probes.

```python
from aisteer360.algorithms.core.internals import StatsSpec
from aisteer360.algorithms.core.internals.probes import ProbeSet

stats = StatsSpec(texts=generic_texts).estimate(model, tokenizer)
probes = ProbeSet.fit(
    model,
    tokenizer,
    data={"medical": medical_pairs, "advice": advice_pairs},
    stats=stats,
    calibration_data={"medical": medical_covering_pairs},
)
readout = probes.read(model, input_ids, attention_mask)
```


## From measurement to decisions and routes

A probe's boolean decisions feed the two decision layers above it. For binary gating, `Probe.as_gate()` returns a
steering gate that reproduces the probe's decision, so an intervention (e.g. an
[`ActivationAdapter`](controls.md#state-control)) applies only when the probe fires. For categorical routing, the
[`RoutedDecoding`](controls.md#output-control) driver evaluates a `Router` (ordered routes with predicates over
decision names, first match wins, per row) against a probe set's decisions and executes the matched action (a canned
response, a prefix followed by generation, or plain generation). The routing vocabulary lives with that control; see
the [routed decoding notebook](../examples/notebooks/recipes/routed_decoding.ipynb) for a worked example.


## Detection versus steering

Conditional steering ([CAST](controls.md#state-control)) detects with the steering direction itself, i.e., it steers
when its own direction is present, and that path is unchanged by probes. Concept detection uses probes, and the two
do not share artifacts. The distinction is geometric, i.e., the direction that best detects a concept (the whitened
difference in class means, $\Sigma^{-1}\Delta\mu$) is generally not the raw mean difference ($\Delta\mu$) used to
steer. A direction obtained elsewhere can still be turned into a probe by calibrating a bias with `calibrate_bias`
and constructing a `Probe` directly.


## Provenance

Probes and activation statistics record a fingerprint of the model they were estimated on, and consumers raise on a
mismatch rather than produce miscalibrated decisions (`allow_model_mismatch=True` is the explicit override). For
pipelines whose structural controls produce the final weights inside `steer()`, `ProbeSetFit` defers fitting; it
holds every fitting input except the model, and `RoutedDecoding` fits it at steer time on the model the pipeline
provides.
