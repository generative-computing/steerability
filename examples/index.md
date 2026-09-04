# Examples

We have prepared a collection of example notebooks for expressing the toolkit's
functionality.

- `algorithms/` contain demonstrations of the toolkit's built-in algorithms, including wrappers around existing libraries (e.g., `trl`, `mergekit`).
- `generics/` illustrate config-based generic controls and demonstrate how modular controls can be constructed.
- `recipes/` are worked examples that compose existing toolkit components into something new.
- `benchmarks/` demonstrate more extensive studies that compare methods on a given use case.

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

-   __Structural control__

    ---

    Structural control methods adapt the model's weights or architecture, such as by fine-tuning or merging checkpoints. These notebooks use our wrappers around established training and merging libraries. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./notebooks/algorithms/mergekit.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./notebooks/algorithms/trl.ipynb)

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

The notebooks below show how to configure each generic and recover named methods from it.

<div class="grid cards" markdown>

-   __State control__

    ---

    The composable activation-steering atom; each adapter wires a transform, layer selection, and optionally a gate and token scope into one single-behavior control. Current notebooks cover:

    :octicons-arrow-right-24: [ActivationAdapter](./notebooks/generics/activation_adapter.ipynb)

-   __Output control__

    ---

    The output analogues, one generic per shape: per-candidate value shifts, mixed log-prob sources, segment search, phased splicing, and stop rules. Current notebooks cover:

    :octicons-arrow-right-24: [ValueGuidance](./notebooks/generics/value_guidance.ipynb)

    :octicons-arrow-right-24: [ContrastiveGuidance](./notebooks/generics/contrastive_guidance.ipynb)

    :octicons-arrow-right-24: [SearchDecoding](./notebooks/generics/search_decoding.ipynb)

    :octicons-arrow-right-24: [PhasedDecoding](./notebooks/generics/phased_decoding.ipynb)

    :octicons-arrow-right-24: [StoppingRules](./notebooks/generics/stopping_rules.ipynb)

</div>


## Recipes

Recipe notebooks compose existing toolkit components into something the toolkit does not ship as a named method.
Where an algorithm notebook demonstrates one control, a recipe builds a new capability out of several.

<div class="grid cards" markdown>

-   __Routed decoding__

    ---

    This notebook fits calibrated probes (`ProbeSet`) on contrastive prompt pools, combines them with boolean routing rules, and routes each query to a response strategy (a canned response, a disclaimer-prefixed answer, or plain generation) via the `RoutedDecoding` driver.

    [:octicons-arrow-right-24: See the recipe](./notebooks/recipes/routed_decoding.ipynb)

</div>


## Benchmarks

<div class="grid cards" markdown>

-   :material-list-box-outline:  __Instruction following__

    ---

    This notebook studies the effect of post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)) on a model's ability to follow instructions. We sweep over the steering strength and investigate the trade-off between a model's instruction following ability and general response quality.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmarks/instruction_following/instruction_following.ipynb)

-   :material-comment-question-outline:  __Commonsense MCQA__

    ---

    This notebook benchmarks steering methods on the [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) dataset, comparing few-shot prompting against a LoRA adapter trained with DPO. We sweep over the number of few-shot examples and study how accuracy scales relative to the fine-tuned baseline across two models.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmarks/commonsense_mcqa/commonsense_mcqa.ipynb)

-   :material-layers-triple-outline:  __Composite steering for truthfulness__

    ---

    One of the primary features of the toolkit is the ability to compose multiple steering methods into one model operation. This notebook composes a state control ([PASTA](https://arxiv.org/abs/2311.02262)) with an output control ([DeAL](https://arxiv.org/abs/2402.06147)) with the goal of improving the model's truthfulness (as measured on [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)) without significantly degrading informativeness. We sweep over the joint parameter space of the controls and study each control's performance (via the tradeoff between truthfulness and informativeness) to that of the composition.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmarks/truthful_qa_composite_steering/truthful_qa_composite_steering.ipynb)

</div>
