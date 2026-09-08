# Probes

!!! note
    This document provides a conceptual overview of detection in the toolkit. For the full API, please see the
    reference pages on [internals](../reference/algorithms/core/internals.md) and
    [probes](../reference/algorithms/core/probes.md). For a worked example, please see the notebook on
    [routed decoding](../examples/notebooks/recipes/routed_decoding/routed_decoding.ipynb).

Some steering workflows depend on detection, i.e., reading the model's internal state to decide whether a concept is
present in a prompt, e.g., recognizing that a question asks for medical advice so it can be routed to a referral
instead of answered. The toolkit implements detection with probes, and builds two kinds of decisions on top of them.
Probes measure, i.e., they turn hidden states into scores and boolean decisions. Gates decide whether a steered
intervention applies (covered under [state control](controls.md#state-control)). Routers combine the decisions of
several probes into a categorical choice of action (inside [routed decoding](controls.md#output-control)). This page
covers probes.


## Probes and probe sets

A `Probe` is a small linear classifier over the model's hidden states. It is fit on contrastive pairs (prompts where
the concept is present and prompts where it is absent), and at inference it pools a prompt's hidden states, takes a
dot product with its weight vector, adds a bias, and decides `score >= 0`.

A fitted probe is always oriented so that positives score high, and its operating threshold is included in the bias
during calibration. There is therefore no comparator or threshold to configure, and the decision is always
`score >= 0`. A probe is also model-free, i.e., it contains only weights, a bias, and provenance metadata. Since it runs
no forward passes itself, it can be saved, loaded, and applied to cached activations offline.

Reads over a loaded model go through a `ProbeSet`, which scores every probe in the set in one read-only forward and returns
a `ProbeReadings` of per-prompt signed scores and boolean decisions. Since the read never edits hidden states,
probing leaves generation untouched.


## Fitting and calibration

Fitting uses two datasets with different roles. The direction is fit on `data`, i.e., discriminative pairs that
isolate the concept, e.g., medical questions against questions from neighboring domains. The operating point is then
calibrated on `calibration_data`, a broader set that also covers the inputs the probe must stay closed on, e.g.,
general questions. When no calibration set is given, the fit pairs serve both roles.

Raw directions in activation space make poor detectors because activations share a large common component and a few
outlier coordinates dominate dot products. The default fitting method (`"lda"`) therefore standardizes features with
ambient activation statistics before taking the difference in class means. The statistics (`ActivationStats`) are
estimated once per model from generic texts and can be saved and reused across probes.

```python
from steerability.algorithms.core.internals import StatsSpec
from steerability.algorithms.core.internals.probes import ProbeSet

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

A probe's boolean decisions feed the two kinds of decisions built on top of them. For binary gating, `Probe.as_gate()`
returns a steering gate that reproduces the probe's decision, such that an intervention (e.g., an
[`ActivationAdapter`](controls.md#state-control)) applies only when the probe fires. For categorical routing, the
[`RoutedDecoding`](controls.md#output-control) driver evaluates a `Router` against a probe set's decisions and executes
the matched action (a canned response, a prefix followed by generation, or plain generation). A router is an ordered
list of routes, each with a predicate over decision names, and the first matching route wins for each row. See the
[routed decoding notebook](../examples/notebooks/recipes/routed_decoding/routed_decoding.ipynb) for a worked example.


## Detection versus steering

Conditional steering ([CAST](controls.md#state-control)) detects with the steering direction itself, i.e., it steers
when its own direction is present, and does not use probes. Concept detection uses probes, and the two
do not share artifacts. The distinction is geometric, i.e., the direction that best detects a concept (the whitened
difference in class means, $\Sigma^{-1}\Delta\mu$) is generally not the raw mean difference ($\Delta\mu$) used to
steer. A direction obtained elsewhere can still be turned into a probe by calibrating a bias with `calibrate_bias`
and constructing a `Probe` directly.

The per-head classifier that [ITI](controls.md#state-control) fits to rank attention heads is also not a `Probe`.
It scores raw per-head activation slices with held-out accuracy at fit time and is discarded once its accuracies
have selected the top-K heads, whereas a `Probe` is a calibrated, model-free detector over pooled residual-stream
features that gates and routers read at generation time.


## Provenance

Probes and activation statistics record a fingerprint of the model they were estimated on, and consumers raise an error on a
mismatch rather than produce miscalibrated decisions (`allow_model_mismatch=True` is the explicit override). For
pipelines whose structural controls produce the final weights inside `steer()`, `ProbeSetFit` defers fitting. It
stores every fitting input except the model, and `RoutedDecoding` fits it at steer time on the model the pipeline
provides.
