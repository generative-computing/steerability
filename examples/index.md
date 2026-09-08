# Examples

We have prepared a collection of example notebooks for expressing the toolkit's
functionality.

- `algorithms/` contain demonstrations of the toolkit's built-in algorithms. Its `generics/` subfolder illustrates config-based generic controls and demonstrates how modular controls can be constructed, and its `wrappers/` subfolder covers the wrappers around existing libraries (e.g., `trl`, `mergekit`).
- `recipes/` are worked examples that compose existing toolkit components into something new.
- `studies/` demonstrate more extensive studies that compare methods on a given use case.

## Algorithms

Algorithm notebooks demonstrate how each method (i.e., control) operates. The methods are grouped below by the category of the model or generation process they act on.

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called, for example by rewriting or augmenting it or by supplying in-context examples. Current notebooks cover:

    :octicons-arrow-right-24: [CPO](./notebooks/algorithms/cpo.ipynb)

    :octicons-arrow-right-24: [FewShot](./notebooks/algorithms/few_shot.ipynb)

    :octicons-arrow-right-24: [GEPA](./notebooks/algorithms/gepa.ipynb)

    :octicons-arrow-right-24: [PRewrite](./notebooks/algorithms/prewrite.ipynb)

    :octicons-arrow-right-24: [SystemPrompt](./notebooks/algorithms/system_prompt.ipynb)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights or architecture, such as by fine-tuning or merging checkpoints. These notebooks use our wrappers around established training and merging libraries. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./notebooks/algorithms/wrappers/mergekit.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./notebooks/algorithms/wrappers/trl.ipynb)

-   __State control__

    ---

    State control methods influence the model's internal states (activations, attention, and similar) at inference time. Current notebooks cover:

    :octicons-arrow-right-24: [ActAdd](./notebooks/algorithms/act_add.ipynb)

    :octicons-arrow-right-24: [AngularSteering](./notebooks/algorithms/angular_steering.ipynb)

    :octicons-arrow-right-24: [CAA](./notebooks/algorithms/caa.ipynb)

    :octicons-arrow-right-24: [CAST](./notebooks/algorithms/cast.ipynb)

    :octicons-arrow-right-24: [DirectionalAblation](./notebooks/algorithms/directional_ablation.ipynb)

    :octicons-arrow-right-24: [ITI](./notebooks/algorithms/iti.ipynb)

    :octicons-arrow-right-24: [PASTA](./notebooks/algorithms/pasta.ipynb)

-   __Output control__

    ---

    Output control methods influence the model's behavior at generation time through the `generate()` method, by shifting logits, searching over candidates, or shaping the decoding process. Current notebooks cover:

    :octicons-arrow-right-24: [BestOfN](./notebooks/algorithms/best_of_n.ipynb)

    :octicons-arrow-right-24: [BudgetForcing](./notebooks/algorithms/budget_forcing.ipynb)

    :octicons-arrow-right-24: [ContrastiveDecoding](./notebooks/algorithms/contrastive_decoding.ipynb)

    :octicons-arrow-right-24: [DeAL](./notebooks/algorithms/deal.ipynb)

    :octicons-arrow-right-24: [DExperts](./notebooks/algorithms/dexperts.ipynb)

    :octicons-arrow-right-24: [RAD](./notebooks/algorithms/rad.ipynb)

    :octicons-arrow-right-24: [SASA](./notebooks/algorithms/sasa.ipynb)

</div>


## Generic controls

Several of the methods above are specific settings of a smaller number of generic controls. As part
of the toolkit, we have prepared a collection of such config-based controls, which we call `generics`,
to enable custom construction of (modular) controls.

The notebooks below show how to configure each generic (as well how to use them to build some of the named controls).

<div class="grid cards" markdown>

-   __State control__

    ---

    The composable activation-steering atom; each adapter wires a transform, layer selection, and optionally a gate and token scope into one single-behavior control. Current notebooks cover:

    :octicons-arrow-right-24: [ActivationAdapter](./notebooks/algorithms/generics/activation_adapter.ipynb)

-   __Output control__

    ---

    The output analogues, one generic per shape: per-candidate value shifts, mixed log-prob sources, segment search, phased splicing, and stop rules. Current notebooks cover:

    :octicons-arrow-right-24: [ValueGuidance](./notebooks/algorithms/generics/value_guidance.ipynb)

    :octicons-arrow-right-24: [ContrastiveGuidance](./notebooks/algorithms/generics/contrastive_guidance.ipynb)

    :octicons-arrow-right-24: [SearchDecoding](./notebooks/algorithms/generics/search_decoding.ipynb)

    :octicons-arrow-right-24: [PhasedDecoding](./notebooks/algorithms/generics/phased_decoding.ipynb)

    :octicons-arrow-right-24: [StoppingRules](./notebooks/algorithms/generics/stopping_rules.ipynb)

</div>


## Recipes

Recipes describe useful applications/compositions of the toolkit's functionality. Generally, recipes are where non-trivial combinations of steering methods (beyond the named controls) are demonstrated.

<div class="grid cards" markdown>

-   __Honest-persona prompting__

    ---

    This notebook reproduces some of the honest-only persona prompting from Anthropic's [evaluating honesty post](https://alignment.anthropic.com/2025/honesty-elicitation/) by composing `UserPrefix` (the `|HONEST_ONLY|` control token), `SystemPrompt` (the mode definition), `PhasedDecoding` (the `<honest_only>` tag prefill), and `StoppingRules` (the closing-tag stop). The notebook compares three prompt variants against the (unsteered) baseline on a scenario that pressures the model to misstate a fact.

    [:octicons-arrow-right-24: See the recipe](./notebooks/recipes/honest_persona_prompting.ipynb)

-   __Routed decoding__

    ---

    This notebook fits calibrated probes (`ProbeSet`) on contrastive prompt pools, combines them with boolean routing rules, and routes each query to a response strategy (a canned response, a disclaimer-prefixed answer, or plain generation) via the `RoutedDecoding` driver.

    [:octicons-arrow-right-24: See the recipe](./notebooks/recipes/routed_decoding/routed_decoding.ipynb)

-   __Sharing pipelines (`.spipe`)__

    ---

    This notebook fits a CAA control, freezes the steered pipeline into a portable `.spipe` bundle (the recipe plus the fitted artifacts, content-addressed), and reconstructs the pipeline from the file alone with matching greedy generations.

    [:octicons-arrow-right-24: See the recipe](./notebooks/recipes/working_with_spipes.ipynb)

-   __Serving through a vLLM server__

    ---

    This notebook fits a CAA direction in process, saves the `SteeringVector`, and serves it through a vLLM server running the vLLM-Hook plugin via the `vllm-serve` backend. The served pipeline holds no model, and its generations are compared against an unsteered pipeline on the same server.

    [:octicons-arrow-right-24: See the recipe](./notebooks/recipes/vllm_serve.ipynb)

</div>


## Studies

Studies provide in-depth comparisons of steering methods on a given use case. Note that these notebooks can be computationally heavy.

<div class="grid cards" markdown>

-   :material-list-box-outline:  __Instruction following__

    ---

    This notebook studies the effect of post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)) on a model's ability to follow instructions, on single-instruction prompts from [Split-IFEval](https://huggingface.co/datasets/ibm-research/Split-IFEval). The Inspect task scores each response with the strict IFEval checker and a reward-model quality score, and delivers each prompt's instruction lines to PASTA through per-sample runtime kwargs. We sweep the steering strength and investigate the trade-off between instruction following and response quality.

    [:octicons-arrow-right-24: See the study](./notebooks/studies/instruction_following/instruction_following.ipynb)

-   :material-comment-question-outline:  __Commonsense MCQA__

    ---

    This notebook studies steering methods on the [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)
    dataset, comparing a few-shot sweep against a DPO-trained LoRA adapter and the unsteered
    baseline. The Inspect task measures accuracy and positional bias
    under deterministic choice shuffling; the notebook sweeps the number of few-shot examples and
    composes the figures from the library plotting calls.

    [:octicons-arrow-right-24: See the study](./notebooks/studies/commonsense_mcqa/commonsense_mcqa.ipynb)

-   :material-call-split:  __Routing versus prompting__

    ---

    This notebook compares the probe-based routing from the [routed decoding recipe](./notebooks/recipes/routed_decoding/routed_decoding.ipynb) against two prompting baselines that desribe the same referral policy, i.e., the full policy in a system prompt and a prompted classifier.

    [:octicons-arrow-right-24: See the study](./notebooks/studies/routing_vs_prompting.ipynb)

</div>
