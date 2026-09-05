# Steering Controls

!!! note
    This document provides the current list of steering controls. To add your own steering control/method, please refer to
    the [tutorial](../tutorials/add_new_steering_method.md). For a better understanding of how steering
    methods can be composed, please see the high-level outline on [steering pipelines](steering_pipelines.md).


We structure steering methods across four categories of control, loosely defined as:

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
$x$. A pipeline may contain several input controls, which compose in `controls`-list order, each receiving the previous
control's output.

For a control method to be deemed an input control method, it must satisfy the following requirements:

- *Control*: Method only influences the prompt supplied to the model. It does not change the model's internals (parameters/states/logits).

- *Persistence*: All changes are temporary. Removing the prompt adapter $\sigma()$ yields the base model.

- *Access*: Implemented without requiring access to the model's internals, e.g., hidden states.

Some examples of input control methods are few-shot prompting, reasoning guidance (like CoT, ToT, GoT,
self-consistency), automatic prompting methods, and prompt routing. The toolkit implements:

- `FewShot` ([API reference](../reference/algorithms/input_control/few_shot.md), [notebook](../examples/notebooks/algorithms/few_shot.ipynb))
    - *Description*: pool- or runtime-supplied few-shot examples with a pluggable selector.
    - *Backends*: HF, vLLM.
- `PRewrite` ([API reference](../reference/algorithms/input_control/prewrite.md), [notebook](../examples/notebooks/algorithms/prewrite.ipynb))
    - *Description*: RL-trained instruction rewriter ([Kong et al. 2024](https://arxiv.org/abs/2401.08189)) supporting a greedy "inference" strategy and a best-of-K "search" strategy. The rewriter can optionally be trained with GRPO using a scorer-in-the-loop reward (apply the rewrite with the frozen task model over a dev set and score each response with a per-row `SampleScorer`, the paper's reward).
    - *Backends*: HF, vLLM.
- `CPO` ([API reference](../reference/algorithms/input_control/cpo.md), [notebook](../examples/notebooks/algorithms/cpo.ipynb))
    - *Description*: causal prompt optimization ([Chen et al. 2026](https://arxiv.org/abs/2602.01711)), i.e., offline causal reward training (Double ML over PCA-reduced embeddings) plus per-query tree search.
    - *Backends*: HF, vLLM (requires `prompt_lm`). Without `prompt_lm` the pipeline's loaded model is bound as the proposer, which is HF-only.
- `GEPA` ([API reference](../reference/algorithms/input_control/gepa.md), [notebook](../examples/notebooks/algorithms/gepa.ipynb))
    - *Description*: reflective genetic prompt evolution ([Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)), single-module variant.
    - *Backends*: HF, vLLM.
- `SystemPrompt` ([API reference](../reference/algorithms/input_control/system_prompt.md), [notebook](../examples/notebooks/algorithms/system_prompt.ipynb))
    - *Description*: sets or merges the leading system message of a chat, prepending to, appending to, or replacing it (the default is to prepend ahead of an existing system prompt), always producing exactly one system message.
    - *Backends*: HF, vLLM.
- `UserPrefix` ([API reference](../reference/algorithms/input_control/user_prefix.md), used in [notebook](../examples/notebooks/algorithms/user_prefix.ipynb))
    - *Description*: prepends a fixed text marker to a user turn (the last user turn by default, or the first or all user turns), with the token stream as a fallback for non-chat input.
    - *Backends*: HF, vLLM.

The few-shot retriever from [Rubin et al. 2021](https://arxiv.org/abs/2112.08633) (EPR) is provided as a `BaseSelector`
that slots into `FewShot` rather than as a separate control. See
[`few_shot.selectors.epr`](../reference/algorithms/input_control/few_shot.md).

Reusable building blocks shared across these methods (memory containers, formatters, scorers, proposers, selectors,
Pareto / rollout-budget utilities) are located in
[`input_control.common`](../reference/algorithms/input_control/common.md).



## Structural control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta'}(x)$

Structural control methods alter the model's parameters or architecture to steer its behavior. These methods usually
allow for more aggressive changes to the model (compared to input control methods). Structural controls are implemented
via fine-tuning, adapter layers, or architectural modifications (e.g., merging) to yield an updated set of weights
$\theta'$. A pipeline may contain several structural controls, and `steer()` threads the model through them in
`controls`-list order.

Structural control methods satisfy the following requirements:

- *Control*: Produces a new or modified set of weights $\theta'$ or extends the network with additional modules/layers.

- *Persistence*: Changes are persistent and are stored in the checkpoint. Reverting requires reloading or undoing the weight edit.

- *Access*: Implementation requires access to parameters and (typically) gradient flows.

Examples of structural control methods are fine-tuning methods (full, parameter efficient), soft prompting (prefix
tuning, p-tuning), and model merging. Many of the structural control methods in the toolkit are implemented as wrappers
around existing libraries. The toolkit implements:

- `LoadCheckpoint` ([API reference](../reference/algorithms/structural_control/load_checkpoint.md))
    - *Description*: installs a saved full-weights checkpoint as the pipeline model, the frozen form of trained structural controls in a [`.spipe`](spipe.md) bundle.
    - *Backends*: HF, vLLM (the checkpoint is served).
- `LoadLoRA` ([API reference](../reference/algorithms/structural_control/load_lora.md))
    - *Description*: attaches a saved LoRA adapter to the pipeline model (optionally merging it into the base weights), verifying the adapter's recorded base model. This is the frozen form of adapter-producing structural controls in a [`.spipe`](spipe.md) bundle.
    - *Backends*: HF, vLLM (the adapter is served).
- `MergeKit` ([API reference](../reference/algorithms/structural_control/mergekit_wrapper.md), [notebook](../examples/notebooks/algorithms/wrappers/mergekit.ipynb))
    - *Description*: model merging via MergeKit[@goddard-etal-2024-arcees], combining multiple checkpoints with strategies such as linear interpolation, SLERP, and TIES from a YAML/dict config.
    - *Backends*: HF, vLLM (the merged checkpoint is served).
- `TRL` ([API reference](../reference/algorithms/structural_control/trl_wrapper.md), [notebook](../examples/notebooks/algorithms/wrappers/trl.ipynb))
    - *Description*: weight-level training via Hugging Face TRL[@vonwerra2022trl], exposing SFT, DPO, APO, PPO, and GRPO trainers, with optional LoRA/PEFT and a post-training merge. Since `training_args` is forwarded verbatim to the installed TRL config, a key the config does not declare raises an error at control construction.
    - *Backends*: HF, vLLM (serves the steer-time artifact, a checkpoint or LoRA adapter, and requires a configured output directory).


## State control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta}^a(x)$

State control methods modify the model's internal/hidden states (e.g., activations, attentions) at inference time.
These methods are implemented by defining hooks that are inserted/registered into the model to manipulate internal
variables during the forward pass.

State control methods satisfy the following requirements:

- *Control*: Writes to (augments) the model's internal/hidden states. Model weights remain fixed.

- *Persistence*: Changes are temporary. Behavior reverts to baseline once hooks are removed.

- *Access*: Requires access to internal states (to define hooks).

Some examples of state control methods are activation addition/steering, attention steering, and representation
patching. The toolkit implements:

- `ActAdd` ([API reference](../reference/algorithms/state_control/act_add.md), [notebook](../examples/notebooks/algorithms/act_add.ipynb))
    - *Description*: activation addition[@turner2023activation], adding a positional steering vector from a single contrast pair to the residual stream at one layer.
    - *Backends*: HF (positional injection has no intervention-spec form).
- `ActivationAdapter` ([API reference](../reference/algorithms/state_control/activation_adapter.md), [notebook](../examples/notebooks/algorithms/generics/activation_adapter.ipynb))
    - *Description*: the composable activation-steering atom, wiring together the shared `common` components (a transform that contains its own artifact, a selector, a gate, and a token scope) so that a recipe is assembled without writing a new control class.
    - *Backends*: HF, vLLM (kind-conditional, i.e., the configured transform, modifier chain, and gate readout/rule must all have wire forms, and a `CallableReadout` gate is HF-only).
- `AngularSteering` ([API reference](../reference/algorithms/state_control/angular_steering.md), [notebook](../examples/notebooks/algorithms/angular_steering.ipynb))
    - *Description*: angular steering[@vu2025angular], rotating the hidden state within a per-layer 2D plane (feature axis + companion axis) to a target angle while leaving the orthogonal complement untouched. It is norm-preserving by construction, and vector addition and directional ablation are special cases.
    - *Backends*: HF, vLLM (`intervention_point="layer_output"` only, since the default norm-input placement is HF-only).
- `CAA` ([API reference](../reference/algorithms/state_control/caa.md), [notebook](../examples/notebooks/algorithms/caa.ipynb))
    - *Description*: contrastive activation addition[@panickssery2023steering], adding a learned mean-difference direction to the residual stream at a single layer.
    - *Backends*: HF, vLLM (norm-preserving configurations included).
- `CAST` ([API reference](../reference/algorithms/state_control/cast.md), [notebook](../examples/notebooks/algorithms/cast.ipynb))
    - *Description*: conditional activation steering[@lee2025programming], applying behavior steering only when a learned condition direction crosses a threshold. The applied behavior transform is pluggable (additive by default, or any `BaseTransform` via `behavior_transform`, e.g., directional ablation for conditional abliteration).
    - *Backends*: HF, vLLM (with the default additive behavior transform, while a custom `behavior_transform` follows that transform's wire form).
- `DirectionalAblation` ([API reference](../reference/algorithms/state_control/directional_ablation.md), [notebook](../examples/notebooks/algorithms/directional_ablation.ipynb))
    - *Description*: directional ablation / abliteration[@arditi2024refusal], projecting a learned feature direction (or subspace) out of the residual stream at masked positions, with a graded ablation strength.
    - *Backends*: HF, vLLM (single direction at full strength, `K = 1` and `alpha = 1`, while graded and subspace ablation are HF-only).
- `ITI` ([API reference](../reference/algorithms/state_control/iti.md), [notebook](../examples/notebooks/algorithms/iti.ipynb))
    - *Description*: inference-time intervention[@li2023inference], shifting activations at a sparse set of probe-selected attention heads during generation.
    - *Backends*: HF, vLLM (`tensor_parallel_size == 1`, norm-preserving configurations are HF-only, and fitting from data runs on the staged model).
- `PASTA` ([API reference](../reference/algorithms/state_control/pasta.md), [notebook](../examples/notebooks/algorithms/pasta.ipynb))
    - *Description*: post-hoc attention steering[@zhang2024tell], rescaling attention to targeted prompt substrings at selected layers and heads. The `head_config` argument takes a dict or list of layers and heads, or a `HeadProfile` recipe that runs the paper's head-profiling stage as a steer-time fit on the loaded model (scoring each candidate head by its paired lift over an unsteered baseline) and freezes the resolved head map.
    - *Backends*: HF with `attn_implementation` `"eager"` or `"sdpa"` (attention-map writes have no engine form).

Reusable building blocks shared across the residual-stream methods (estimators, gating, selectors, transforms,
steering vectors, hook utilities) are located in
[`state_control.common`](../reference/algorithms/state_control/common.md).

Most state controls in the toolkit are declarative. A control states its edit once, as a tuple of interventions, where
each intervention specifies the layers to edit, a transform (e.g., adding a direction or projecting one out), a token
scope (which positions receive the edit), and optionally a gate. The toolkit compiles this statement for whichever
backend runs it, i.e., to torch hooks in process and to an intervention spec for engines that host activation edits
through the vLLM-Hook plugin. A configuration whose components all have a serialized form therefore generates on vLLM
without any control-specific code, and one that does not stays in process. The pipeline's `check()` reports which,
with a verdict that identifies the gap and the fix.

A gate makes an intervention conditional. It reads hidden states at chosen layers, reduces each pooled state to a
per-prompt value (e.g., an affine score or a cosine similarity against a condition direction), and applies a decision
rule (e.g., a summed score against a calibrated bias, or per-layer thresholds). The decision is made on the prompt,
applies for the whole generation, and is taken independently per row of a batch. An unconditional intervention has no
gate.

`ActivationAdapter` is the general-purpose form of a declarative control. Each adapter steers a single behavior (one
transform chain, one gate, and one token scope), and steering with several behaviors is several adapters listed
together in a pipeline's `controls`, applied in list order. Adapters can share one gate instance for joint
conditioning, and a fitted [`Probe`](probes.md) can gate an adapter through `Probe.as_gate()`. Position-scoped and
gated controls compose with multi-call decoding drivers (e.g., segment search) and with step-level controls that score
candidates through the pipeline's own model.

State controls locate the decoder layers through a model layout, which is resolved automatically for text-only decoder
models (Llama, Mistral, Qwen, and Gemma), for composite multimodal wrappers loaded under `AutoModelForCausalLM` (Gemma
3/4 and Qwen3.5), and for GPT-2. Hybrid architectures that interleave attention layers with another token mixer
(Qwen3.5 and Qwen3-Next) are supported by the residual-stream controls and by hidden-state capture, while controls that
act on attention (`PASTA` and o_proj-site interventions) are restricted to the attention layers, and `ITI` does not
support them. A multimodal checkpoint is steered on its text decoder under text-only prompting, and images and audio
are out of scope. A state control listed after an unmerged LoRA adapter steers the adapted model. For an architecture
not on this list, register a detector with `register_layout_detector` (from `steerability.algorithms.core.internals`).



## Output control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim d(p_{\theta})(x)$

Output control methods modify model outputs or constrain/transform what leaves the decoder. The base distribution
$p_\theta$ is left intact, and only the path through the distribution changes.

Output control methods satisfy the following requirements:

- *Control*: Replaces or constrains the decoding operator. No prompts, hidden states, or weights are altered.

- *Persistence*: Changes are temporary. Behavior is restored once decoding control is removed.

- *Access*: Requires access to logits, token-probabilities, and possibly hidden states (depending on the method).

Examples of output control methods are sampling/search strategies, weighted decoding, and reward-augmented
decoding. Output controls participate in decoding in one of two ways. A step-level control supplies logits processors
and/or stopping criteria (via `get_logits_processors` and `get_stopping_criteria`), which the pipeline composes in
`controls`-list order. Step-level controls therefore compose with each other and with a decoding driver. A decoding
driver subclasses `DecodingDriver` and owns the decode loop (`decode(...)`), applying the composed processors and
stopping criteria in every forward pass it issues. Since the loop does not compose, a pipeline admits at most one
enabled driver, and with none, decoding defaults to the model's own `generate`.

The toolkit implements the following step-level controls:

- `RAD` ([API reference](../reference/algorithms/output_control/rad.md), [notebook](../examples/notebooks/algorithms/rad.ipynb))
    - *Description*: reward-augmented decoding[@deng-raffel-2023-reward], scoring the top-`k` candidate tokens with an `AutoModelForSequenceClassification` reward model and shifting their logits by `beta * reward`. When the reward model is decoder-only and shares the base model's vocabulary it caches the reward-model prefix activations across steps (the paper's efficient path), and otherwise scores each step statelessly. A `value_trace` list passed via `runtime_kwargs` records the per-step candidate scores and rewards for inspection.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `SASA` ([API reference](../reference/algorithms/output_control/sasa.md), [notebook](../examples/notebooks/algorithms/sasa.ipynb))
    - *Description*: self-disciplined autoregressive sampling[@ko2025large], fitting a linear subspace in the model's own final-layer space from labeled examples (unpaired classes or paired prompt/response data) and shifting the candidate-token logits by the softmax-normalized margin to that subspace at each step. The candidate set follows a policy (`surviving`, `top_p`, or `top_k`); the attribute is whatever the labels define.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `DExperts` ([API reference](../reference/algorithms/output_control/dexperts.md), [notebook](../examples/notebooks/algorithms/dexperts.ipynb))
    - *Description*: decoding-time experts[@liu2021dexperts], re-weighting the base distribution by the log-prob difference between a small expert and anti-expert. Proxy-tuning is the same control with a tuned/untuned small-model pair.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ContrastiveDecoding` ([API reference](../reference/algorithms/output_control/contrastive_decoding.md), [notebook](../examples/notebooks/algorithms/contrastive_decoding.ipynb))
    - *Description*: contrastive decoding[@li2022contrastive], favoring tokens the base (expert) scores higher than a weaker amateur, over an expert-plausibility-masked set.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ConstrainedDecoding` ([API reference](../reference/algorithms/output_control/constrained_decoding.md))
    - *Description*: constrained decoding from one declarative source (JSON schema, regex, EBNF grammar, or a choice set). Every logit the grammar forbids is masked at each step.
    - *Backends*: HF (client-side xgrammar automaton), vLLM (native structured outputs). A control constructed with an in-memory automaton object is HF-only.
- `ValueGuidance` ([API reference](../reference/algorithms/output_control/value_guidance.md), [notebook](../examples/notebooks/algorithms/generics/value_guidance.ipynb))
    - *Description*: the config-first generic over the step shape (candidates → value → normalize → shift). FUDGE, ARGS, RAD, and SASA are assignments of its config.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `ContrastiveGuidance` ([API reference](../reference/algorithms/output_control/contrastive_guidance.md), [notebook](../examples/notebooks/algorithms/generics/contrastive_guidance.ipynb))
    - *Description*: the config-first generic over the distribution shape (mix weighted log-prob sources). DExperts, contrastive decoding, and proxy-tuning are assignments of its config.
    - *Backends*: HF (model-backed per-step logit math is in-process only).
- `StoppingRules` ([API reference](../reference/algorithms/output_control/stopping_rules.md), [notebook](../examples/notebooks/algorithms/generics/stopping_rules.ipynb))
    - *Description*: the config-first generic for stop rules, i.e., substring / token / budget stops as pipeline configuration rather than a class. Since its stops merge into the call's generation parameters, rows halted by them report `finish_reason="stop"` and the pipeline truncates decoded text at the stop string.
    - *Backends*: HF, vLLM (stops lower to sampling parameters).

and the following decoding drivers:

- `DeAL` ([API reference](../reference/algorithms/output_control/deal.md), [notebook](../examples/notebooks/algorithms/deal.ipynb))
    - *Description*: decoding-time alignment[@huang2024deal], i.e., iterative lookahead beam search with reward-guided beam selection.
    - *Backends*: HF (beam proposals are in-process only, though the sampled-proposal search runs on vLLM as a `SearchDecoding` configuration).
- `BestOfN` ([API reference](../reference/algorithms/output_control/best_of_n.md), [notebook](../examples/notebooks/algorithms/best_of_n.ipynb))
    - *Description*: best-of-N sampling / re-ranking[@nakano2021webgpt], sampling N full continuations and returning the highest-scoring one under a sequence scorer (pairing with a majority-vote scorer recovers self-consistency).
    - *Backends*: HF, vLLM.
- `BudgetForcing` ([API reference](../reference/algorithms/output_control/budget_forcing.md), [notebook](../examples/notebooks/algorithms/budget_forcing.ipynb))
    - *Description*: test-time thinking-length control[@muennighoff2025s1], capping each thinking segment, optionally appending extensions ("Wait") to prolong reasoning, then forcing the closing think tag before answering. `end_think_token_ids` sets the thinking-phase boundary by token id, for a closing-think delimiter that is a special token.
    - *Backends*: HF, vLLM.
- `RoutedDecoding` ([API reference](../reference/algorithms/output_control/routed_decoding.md), [notebook](../examples/notebooks/recipes/routed_decoding/routed_decoding.ipynb))
    - *Description*: a decoding driver that routes each row to a response plan via a `Router` over a [`ProbeSet`](probes.md)'s readings, and executes the matched plan (canned response, disclaimer prefix, or plain generation). It sits beside `PhasedDecoding` and `SearchDecoding`.
    - *Backends*: HF, offline vLLM (the probe pass needs hidden-state capture, which serve does not return).
- `SearchDecoding` ([API reference](../reference/algorithms/output_control/search_decoding.md), [notebook](../examples/notebooks/algorithms/generics/search_decoding.ipynb))
    - *Description*: the config-first generic over the segment shape (propose → score → keep → iterate, with best-of-N defaults). Best-of-N, self-consistency, blockwise controlled decoding, and DeAL are assignments of its config.
    - *Backends*: HF, vLLM with `propose_mode="sample"` (beam proposals are HF-only).
- `PhasedDecoding` ([API reference](../reference/algorithms/output_control/phased_decoding.md), [notebook](../examples/notebooks/algorithms/generics/phased_decoding.ipynb))
    - *Description*: the config-first generic over the phase shape (forced / generated segments via a declarative plan grammar). Budget forcing, response prefill, and thinking intervention[@wu2025effectively] are assignments of its config. A `generate` phase ends at its `until` substring, any token in `until_token_ids`, or its `budget`, whichever first.
    - *Backends*: HF, vLLM.

Some decoding strategies are native to Hugging Face's `generate` and need no dedicated control. They flow through the
default driver via `gen_kwargs`, for example DoLa decoding (`gen_kwargs={"dola_layers": ...}`) and watermarking
(`gen_kwargs={"watermarking_config": ...}`).

### Generic controls

Alongside the named methods, the output category provides a small family of generic controls, the output analogue
of state control's [`ActivationAdapter`](#state-control). Where a named method (RAD, SASA, DeAL) is a class, a
generic exposes the shared component slots through flat, sweepable `Args`, and a method from the literature becomes an
assignment of a config rather than a subclass. Since output controls act either at the step level or by owning the
decode loop, and do so over a few distinct shapes of computation, there is one generic per shape:

| generic | mechanism | shape | canonical assignments |
| ------- | --------- | ----- | --------------------- |
| [`ValueGuidance`](../reference/algorithms/output_control/value_guidance.md) | step-level (logits processors) | step | FUDGE, ARGS, RAD-, SASA-equivalents |
| [`ContrastiveGuidance`](../reference/algorithms/output_control/contrastive_guidance.md) | step-level (logits processors) | distribution | DExperts, contrastive decoding, proxy-tuning |
| [`SearchDecoding`](../reference/algorithms/output_control/search_decoding.md) | driver | segment | best-of-N, self-consistency, DeAL-equivalent |
| [`PhasedDecoding`](../reference/algorithms/output_control/phased_decoding.md) | driver | phase | budget forcing, response prefill, thinking intervention |
| [`StoppingRules`](../reference/algorithms/output_control/stopping_rules.md) | sampling-mapped (stop rules) | none | substring / token / budget stops |

The named methods are siblings of these generics rather than children. They are built directly on the same `common`
components, and each keeps the one thing its class adds beyond a config (RAD's cached reward path, SASA's subspace
fitting, and so on). When a config earns a name through use, it can be promoted to a small preset subclass over the
generic.

Reusable building blocks shared across these methods (candidate policies, per-candidate value functions, full-vocabulary
logit sources, sequence scorers, a segment-search driver, a phased driver, composable stopping criteria, and the
`PrefixKeyedProcessor` base) are located in
[`output_control.common`](../reference/algorithms/output_control/common.md).
