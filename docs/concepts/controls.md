# Steering Controls

!!! note
    This document provides the current list of steering controls. To add your own steering control/method, please refer to
    the [tutorial](../tutorials/add_new_steering_method.md). For a better understanding of how steering
    methods can be composed, please see high-level outline on [steering pipelines](steering_pipelines.md).


There are various ways to steer a model. We structure steering methods across four categories of control, loosely
defined as:

- [**input**](#input-control): edits the prompt
- [**structural**](#structural-control): edits the weights/architecture
- [**state**](#state-control): edits the (hidden) states
- [**output**](#output-control): edits the decoding/sampling process

The category of a given steering method is dictated by what aspect of the model the method influences. We define each
category of control below.


## Input control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_\theta(\sigma(x))$

Input control methods describe algorithms that manipulate the input/prompt to guide model behavior. They do not change
the model itself. This is enabled in the toolkit through a prompt adapter $\sigma(x)$ applied to the original prompt
$x$. A pipeline may hold several input controls; they compose in `controls`-list order, each receiving the previous
control's output.

For a control method to be deemed an input control method, it must satisfy the following requirements:

- *Control*: Method only influences the prompt supplied to the model; does not change model's internals (parameters/states/logits)

- *Persistence*: All changes are temporary; removing the prompt adapter $\sigma()$ yields the base model.

- *Access*: Implemented without requiring access to model's internals, e.g., hidden states.

Some examples of input control methods include: few-shot prompting, reasoning guidance (like CoT, ToT, GoT,
self-consistency), automatic prompting methods, and prompt routing. The toolkit implements:

- `FewShot` ([API reference](../reference/algorithms/input_control/few_shot.md), [notebook](../examples/notebooks/algorithms/few_shot.ipynb))
    - *Description*: pool- or runtime-supplied few-shot examples; pluggable selector.
    - *Backends*: HF, vLLM.
- `PRewrite` ([API reference](../reference/algorithms/input_control/prewrite.md), [notebook](../examples/notebooks/algorithms/prewrite.ipynb))
    - *Description*: RL-trained instruction rewriter ([Kong et al. 2024](https://arxiv.org/abs/2401.08189)); supports a greedy "inference" strategy and a best-of-K "search" strategy. The rewriter can optionally be trained with GRPO using a metric-in-the-loop reward (apply the rewrite with the frozen task model over a dev set and score with a `Metric`, the paper's reward).
    - *Backends*: HF, vLLM.
- `CPO` ([API reference](../reference/algorithms/input_control/cpo.md), [notebook](../examples/notebooks/algorithms/cpo.ipynb))
    - *Description*: causal prompt optimization ([Chen et al. 2026](https://arxiv.org/abs/2602.01711)); offline causal reward training (Double ML over PCA-reduced embeddings) plus per-query tree search.
    - *Backends*: HF, vLLM (requires `prompt_lm`; without it the live pipeline model is bound as the proposer, HF-only).
- `GEPA` ([API reference](../reference/algorithms/input_control/gepa.md), [notebook](../examples/notebooks/algorithms/gepa.ipynb))
    - *Description*: reflective genetic prompt evolution ([Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)); single-module variant.
    - *Backends*: HF, vLLM.

The few-shot retriever from [Rubin et al. 2021](https://arxiv.org/abs/2112.08633) (EPR) is shipped as a `BaseSelector`
that slots into `FewShot` rather than as a separate control; see
[`few_shot.selectors.epr`](../reference/algorithms/input_control/few_shot.md).

Reusable building blocks shared across these methods (memory containers, formatters, scorers, proposers, selectors,
Pareto / rollout-budget utilities) live in
[`input_control.common`](../reference/algorithms/input_control/common.md).



## Structural control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta'}(x)$

Structural control methods alter the model’s parameters or architecture to steer its behaviour. These methods usually
allow for more aggressive changes to the model (compared to input control methods). Structural controls are implemented
via fine-tuning, adapter layers, or architectural modifications (e.g., merging) to yield an updated set of weights
$\theta'$. A pipeline may hold several structural controls; `steer()` threads the model through them in
`controls`-list order.

Structural control methods satisfy the following requirements:

- *Control*: Produces a new or modified set of weights $\theta'$ or extends the network with additional modules/layers.

- *Persistence*: Changes are persistent and live inside the checkpoint; reverting requires reloading or undoing the weight edit.

- *Access*: Implementation requires access to parameters and (typically) gradient flows.

Examples of structural control methods include: fine-tuning methods (full, parameter efficient), soft prompting (prefix
tuning, p-tuning), and model merging. Many of the structural control methods in the toolkit are implemented as wrappers
around existing libraries. The toolkit implements:

- `MergeKit` ([API reference](../reference/algorithms/structural_control/mergekit_wrapper.md), [notebook](../examples/notebooks/algorithms/mergekit.ipynb))
    - *Description*: model merging via MergeKit[@goddard-etal-2024-arcees]; combines multiple checkpoints with strategies such as linear interpolation, SLERP, and TIES from a YAML/dict config.
    - *Backends*: HF, vLLM (the merged checkpoint is served).
- `TRL` ([API reference](../reference/algorithms/structural_control/trl_wrapper.md), [notebook](../examples/notebooks/algorithms/trl.ipynb))
    - *Description*: weight-level training via Hugging Face TRL[@vonwerra2022trl]; exposes SFT, DPO, APO, PPO, and GRPO trainers, with optional LoRA/PEFT and a post-training merge.
    - *Backends*: HF, vLLM (serves the steer-time artifact, a checkpoint or LoRA adapter; an output directory must be configured).


## State control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta}^a(x)$

State control methods modify the model's internal/hidden states (e.g., activations, attentions, etc.) at inference time.
These methods are implemented by defining hooks that are inserted/registered into the model to manipulate internal
variables during the forward pass.

State control methods satisfy requirements:

- *Control*: Writes to (augments) model's internal/hidden states; model weights remain fixed.

- *Persistence*: Changes are temporary; behavior reverts to baseline once hooks are removed.

- *Access*: Requires access to internal states (to define hooks).

Some examples of state control methods include: activation addition/steering, attention steering, and representation
patching. The toolkit implements:

- `ActAdd` ([API reference](../reference/algorithms/state_control/act_add.md), [notebook](../examples/notebooks/algorithms/act_add.ipynb))
    - *Description*: activation addition[@turner2023activation]; adds a positional steering vector from a single contrast pair to the residual stream at one layer.
    - *Backends*: HF (positional injection has no intervention-spec form).
- `ActivationAdapter` ([API reference](../reference/algorithms/state_control/activation_adapter.md), [notebook](../examples/notebooks/generics/activation_adapter.ipynb))
    - *Description*: the composable activation-steering atom; wires together the shared `common` components (a transform that carries its own artifact, selector, gate, token scope) so a recipe is assembled without writing a new control class.
    - *Backends*: HF, vLLM (kind-conditional: the configured transform, modifier chain, and gate readout/rule must all have wire forms; a `CallableReadout` gate is HF-only).
- `AngularSteering` ([API reference](../reference/algorithms/state_control/angular_steering.md), [notebook](../examples/notebooks/algorithms/angular_steering.ipynb))
    - *Description*: angular steering[@vu2025angular]; rotates the hidden state within a per-layer 2D plane (feature axis + companion axis) to a target angle, leaving the orthogonal complement untouched. Norm-preserving by construction; vector addition and directional ablation are special cases.
    - *Backends*: HF, vLLM (`intervention_point="layer_output"` only; the default norm-input placement is HF-only).
- `CAA` ([API reference](../reference/algorithms/state_control/caa.md), [notebook](../examples/notebooks/algorithms/caa.ipynb))
    - *Description*: contrastive activation addition[@panickssery2023steering]; adds a learned mean-difference direction to the residual stream at a single layer.
    - *Backends*: HF, vLLM (norm-preserving configurations included).
- `CAST` ([API reference](../reference/algorithms/state_control/cast.md), [notebook](../examples/notebooks/algorithms/cast.ipynb))
    - *Description*: conditional activation steering[@lee2025programming]; applies behavior steering only when a learned condition direction crosses a threshold. The applied behavior transform is pluggable (additive by default; any `BaseTransform` via `behavior_transform`, e.g. directional ablation for conditional abliteration).
    - *Backends*: HF, vLLM (with the default additive behavior transform; a custom `behavior_transform` follows that transform's wire form).
- `DirectionalAblation` ([API reference](../reference/algorithms/state_control/directional_ablation.md), [notebook](../examples/notebooks/algorithms/directional_ablation.ipynb))
    - *Description*: directional ablation / abliteration[@arditi2024refusal]; projects a learned feature direction (or subspace) out of the residual stream at masked positions, with a graded ablation strength.
    - *Backends*: HF, vLLM (single direction at full strength, `K = 1` and `alpha = 1`; graded and subspace ablation are HF-only).
- `ITI` ([API reference](../reference/algorithms/state_control/iti.md), [notebook](../examples/notebooks/algorithms/iti.ipynb))
    - *Description*: inference-time intervention[@li2023inference]; shifts activations at a sparse set of probe-selected attention heads during generation.
    - *Backends*: HF, vLLM (`tensor_parallel_size == 1`; norm-preserving configurations are HF-only; fitting from data runs on the staged model).
- `PASTA` ([API reference](../reference/algorithms/state_control/pasta.md), [notebook](../examples/notebooks/algorithms/pasta.ipynb))
    - *Description*: post-hoc attention steering[@zhang2024tell]; rescales attention to targeted prompt substrings at selected layers and heads.
    - *Backends*: HF with `attn_implementation` `"eager"` or `"sdpa"` (attention-map writes have no engine form).

Reusable building blocks shared across the residual-stream methods (estimators, gating, selectors, transforms,
steering vectors, hook utilities) live in
[`state_control.common`](../reference/algorithms/state_control/common.md).

Positions are read from the `cache_position` kwarg at decoder-layer hook points, so position-scoped and gated state
controls compose exactly with multi-call decoding drivers (segment search, phased splicing) and with step-level
controls that forward the pipeline's own model (SASA-style candidate scoring). Hook points on sub-modules that do not
receive the kwarg assume the plain single-`generate` decode pattern. The variant branch of a CFG-style contrast is a
detached sequence and runs unsteered by design.

A residual-stream state control is a declarative tuple of interventions (layers, a transform, a token scope, an
optional gate), stated once and compiled per backend: to torch hooks on the in-process backend, and to an
intervention spec for engines that host activation edits, so the same steered configuration generates on vLLM. A
configuration either serializes exactly or stays in-process only; the pipeline's `check()` reports which, with a
verdict naming the gap and the fix. The per-control support boundary is recorded on each control's `Backends` line above.

A gate makes an intervention conditional, and it factors into three parts: evidence (which layers are read and how
their hidden states are pooled), a readout (how each pooled state becomes a per-prompt value, e.g. an affine score,
a cosine similarity, or CAST's projected cosine), and a rule (the decision over those values, e.g. a summed score
against a calibrated bias, or per-layer thresholds). The decision is made on the prompt and holds for the
generation, independently per row of a batch. An unconditional intervention simply has no gate.

`ActivationAdapter` is the **composition surface** for these building blocks: each adapter is a single-behavior atom
(one transform chain — which carries its own artifact — one gate, one token scope), and steering with several behaviors
is simply several adapters listed together in a pipeline's `controls`. Because a pipeline accepts
[multiple state controls](steering_pipelines.md) applied in list order, composition across behaviors is owned by that
ordered list — no separate composite abstraction is needed. Joint conditioning across adapters uses one shared gate
instance: a driver carries the gate and feeds it through its condition hooks; followers pass the same instance with
`gate_driven_externally=True` and read its decision. A fitted [`Probe`](probes.md) can also gate an adapter through
`Probe.as_gate()`, which returns a gate reproducing the probe's decision.



## Output control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim d(p_{\theta})(x)$

Output control methods modify model outputs or constrain/transform what leaves the decoder. The base distribution
$p_\theta$ is left intact; only the path through the distribution changes.

Output control methods satisfy:

- *Control*: Replaces or constrains the decoding operator; no prompts, hidden states, or weights are altered.

- *Persistence*: Changes are temporary; behavior is restored once decoding control is removed.

- *Access*: Requires access to logits, token-probabilities, and possibly hidden states (depending on the method).

Examples of output control methods include: sampling/search strategies, weighted decoding, and reward-augmented
decoding. Output controls participate in decoding through one of two modes:

- **Contribute**: a control supplies logits processors and/or stopping criteria (via `get_logits_processors` /
  `get_stopping_criteria`). The pipeline composes them in `controls`-list order, so step-level controls compose with
  each other and with a decoding driver.
- **Drive**: a control subclasses `DecodingDriver` and owns the decode loop (`decode(...)`), applying the composed
  stacks in every forward pass it issues. The loop does not compose, so a pipeline admits at most one enabled driver;
  with none, decoding defaults to the model's own `generate`.

The toolkit implements the following step-level controls:

- `RAD` ([API reference](../reference/algorithms/output_control/rad.md), [notebook](../examples/notebooks/algorithms/rad.ipynb))
    - *Description*: reward-augmented decoding[@deng-raffel-2023-reward]; scores the top-`k` candidate tokens with an `AutoModelForSequenceClassification` reward model and shifts their logits by `beta * reward`. When the reward model is decoder-only and shares the base model's vocabulary it caches the reward-model prefix activations across steps (the paper's efficient path), and otherwise scores each step statelessly.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `SASA` ([API reference](../reference/algorithms/output_control/sasa.md), [notebook](../examples/notebooks/algorithms/sasa.ipynb))
    - *Description*: self-disciplined autoregressive sampling[@ko2025large]; shifts logits toward a learned non-toxic subspace.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `DExperts` ([API reference](../reference/algorithms/output_control/dexperts.md), [notebook](../examples/notebooks/algorithms/dexperts.ipynb))
    - *Description*: decoding-time experts[@liu2021dexperts]; re-weights the base distribution by the log-prob difference between a small expert and anti-expert. Proxy-tuning is the same control with a tuned/untuned small-model pair.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ContrastiveDecoding` ([API reference](../reference/algorithms/output_control/contrastive_decoding.md), [notebook](../examples/notebooks/algorithms/contrastive_decoding.ipynb))
    - *Description*: contrastive decoding[@li2022contrastive]; favors tokens the base (expert) scores higher than a weaker amateur, over an expert-plausibility-masked set.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ConstrainedDecoding` ([API reference](../reference/algorithms/output_control/constrained_decoding.md))
    - *Description*: constrained decoding from one declarative source (JSON schema, regex, EBNF grammar, or a choice set); every logit the grammar forbids is masked at each step.
    - *Backends*: HF (client-side automaton, `aisteer360[guided]`), vLLM (native structured outputs); a control constructed with a live automaton object is HF-only. On vLLM versions predating the structured-outputs surface the engine grammar is not whitespace-compact, so engine and in-process outputs agree structurally rather than byte-for-byte.
- `ValueGuidance` ([API reference](../reference/algorithms/output_control/value_guidance.md), [notebook](../examples/notebooks/generics/value_guidance.ipynb))
    - *Description*: the config-first generic over the step shape (candidates → value → normalize → shift); FUDGE, ARGS, RAD, and SASA are assignments of its config.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ContrastiveGuidance` ([API reference](../reference/algorithms/output_control/contrastive_guidance.md), [notebook](../examples/notebooks/generics/contrastive_guidance.ipynb))
    - *Description*: the config-first generic over the distribution shape (mix weighted log-prob sources); DExperts, contrastive decoding, and proxy-tuning are assignments of its config.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `StoppingRules` ([API reference](../reference/algorithms/output_control/stopping_rules.md), [notebook](../examples/notebooks/generics/stopping_rules.ipynb))
    - *Description*: the config-first generic for stop rules; substring / token / budget stops as pipeline configuration rather than a class. Its stops merge into the call's generation parameters, so rows halted by them report `finish_reason="stop"` and the pipeline truncates decoded text at the stop string.
    - *Backends*: HF, vLLM (stops lower to sampling parameters).

and the following decoding drivers:

- `DeAL` ([API reference](../reference/algorithms/output_control/deal.md), [notebook](../examples/notebooks/algorithms/deal.ipynb))
    - *Description*: decoding-time alignment[@huang2024deal]; iterative lookahead beam search with reward-guided beam selection.
    - *Backends*: HF (beam proposals are in-process only; the sampled-proposal search runs on vLLM as a `SearchDecoding` configuration).
- `BestOfN` ([API reference](../reference/algorithms/output_control/best_of_n.md), [notebook](../examples/notebooks/algorithms/best_of_n.ipynb))
    - *Description*: best-of-N sampling / re-ranking[@nakano2021webgpt]; samples N full continuations and returns the highest-scoring one under a sequence scorer (pairing with a majority-vote scorer recovers self-consistency).
    - *Backends*: HF, vLLM.
- `BudgetForcing` ([API reference](../reference/algorithms/output_control/budget_forcing.md), [notebook](../examples/notebooks/algorithms/budget_forcing.ipynb))
    - *Description*: test-time thinking-length control[@muennighoff2025s1]; caps each thinking segment, optionally appends extensions ("Wait") to prolong reasoning, then forces the closing think tag before answering.
    - *Backends*: HF, vLLM.
- `RoutedDecoding` ([API reference](../reference/algorithms/output_control/routed_decoding.md), [notebook](../examples/notebooks/recipes/routed_decoding.ipynb))
    - *Description*: a decoding driver that routes each row to a response plan via a `Router` over a [`ProbeSet`](probes.md)'s readings, and executes the matched plan (canned response, disclaimer prefix, or plain generation); sits beside `PhasedDecoding` and `SearchDecoding`.
    - *Backends*: HF, offline vLLM (the probe pass needs hidden-state capture, which serve does not return).
- `SearchDecoding` ([API reference](../reference/algorithms/output_control/search_decoding.md), [notebook](../examples/notebooks/generics/search_decoding.ipynb))
    - *Description*: the config-first generic over the segment shape (propose → score → keep → iterate; defaults are best-of-N); best-of-N, self-consistency, blockwise controlled decoding, and DeAL are assignments of its config.
    - *Backends*: HF, vLLM with `propose_mode="sample"` (beam proposals are HF-only).
- `PhasedDecoding` ([API reference](../reference/algorithms/output_control/phased_decoding.md), [notebook](../examples/notebooks/generics/phased_decoding.ipynb))
    - *Description*: the config-first generic over the phase shape (forced / generated segments via a declarative plan grammar); budget forcing, response prefill, and thinking intervention[@wu2025effectively] are assignments of its config.
    - *Backends*: HF, vLLM.

Some decoding strategies are native to Hugging Face's `generate` and need no dedicated control — they flow through the
default driver via `gen_kwargs`, for example DoLa decoding (`gen_kwargs={"dola_layers": ...}`) and watermarking
(`gen_kwargs={"watermarking_config": ...}`).

### Generic controls

The output category's composition surface is a small family of generic, `Args`-configured controls, the output
analogue of state control's [`ActivationAdapter`](#state-control). Where a named method (RAD, SASA, DeAL) is a class, a
generic exposes the `common` component slots through flat, sweepable `Args`, so a method from the literature is an
assignment of a config, not a subclass. Output has two composable mechanisms (logits
processors and stopping criteria) and an exclusive decode loop claimed by type, across four shapes, so the analogue is
not one control but a family, one generic per shape, sharing one idiom: expose the slots through flat `Args`, resolve
component specs (name / instance / callable / dict-with-`kind`) at `steer()` time, derive `supports_batching` /
`include_in_scoring` honestly from the resolved components, and return fresh processors per call.

| generic | mechanism | shape | canonical assignments |
| ------- | --------- | ----- | --------------------- |
| [`ValueGuidance`](../reference/algorithms/output_control/value_guidance.md) | step-level (logits processors) | step | FUDGE, ARGS, RAD-, SASA-equivalents |
| [`ContrastiveGuidance`](../reference/algorithms/output_control/contrastive_guidance.md) | step-level (logits processors) | distribution | DExperts, contrastive decoding, proxy-tuning |
| [`SearchDecoding`](../reference/algorithms/output_control/search_decoding.md) | driver | segment | best-of-N, self-consistency, DeAL-equivalent |
| [`PhasedDecoding`](../reference/algorithms/output_control/phased_decoding.md) | driver | phase | budget forcing, response prefill, thinking intervention |
| [`StoppingRules`](../reference/algorithms/output_control/stopping_rules.md) | sampling-mapped (stop rules) | — | substring / token / budget stops |

The named methods are siblings, not children, of these generics: they sit directly on the same `common` parts and
each keeps the one thing its class adds beyond a config (RAD's cached unidirectional reward path, SASA's probe
fitting, and so on). When a config earns a name through use, promote it with a small preset subclass over the generic.

Reusable building blocks shared across these methods (candidate policies, per-candidate value functions, full-vocabulary
logit sources, sequence scorers, a segment-search driver, a phased driver, composable stopping criteria, and the
`PrefixKeyedProcessor` base) live in
[`output_control.common`](../reference/algorithms/output_control/common.md). Within a `common/<family>/` folder, the
primary class in `<name>.py` is `<Name><FamilySingular>` (for example `values/classifier.py` defines `ClassifierValue`,
`scorers/metric.py` defines `MetricScorer`); the family base lives in `base.py`, and top-level `common/*.py` modules
(such as `candidates.py`, `criteria.py`, `candidate_forward.py`) are collection or helper modules exempt from the
suffix rule.
