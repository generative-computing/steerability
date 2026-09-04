# Welcome!

<p style="font-size: 1.05rem;">
The AI Steerability 360 toolkit is an extensible library for general purpose steering of LLMs.
</p>

The term *steering* describes any deliberate action to change a model's behavior. Building on this term, the concept of
*steerability* has come to describe the ease (and extent) to which a model can be steered to a given behavior.[@miehling2025evaluating; @vafa2025s; @chang2025course]
Quantifying a model's steerability is desirable primarily in that it enables a better understanding of how much a
model's generations can be controlled and, in turn, contributes to a better understanding of the model's general
usability, safety, and alignment.[@sorensen2024roadmap]

The AI Steerability 360 toolkit (AISteer360) provides a structured framework for both steering models and evaluating
their steerability. To help organize the wide range of steering methods (e.g., few-shot learning, activation steering,
attention reweighting, parameter-efficient fine-tuning, reward-driven decoding, etc.), the toolkit structures methods (hereafter referred to as
*controls*) across four categories: **input**, **structural**, **state**, and **output**. Assuming that outputs \( y \)
are generated from a base (unsteered) model as \( y \sim p_\theta(x) \), where \( x \) is the input/prompt,
\( \theta \) is the model's parameters, and \( p_\theta(x) \) is the model's (conditional) distribution over outputs
given \( x \), control for each category is exerted as follows.

<div class="grid cards control-grid" markdown>

- **Input control:** \( y \sim p_\theta(\sigma(x)) \)
    - Methods that manipulate the input/prompt to guide model behavior without modifying the model.
    - Facilitated through a *prompt adapter* \( \sigma(x) \) applied to the original prompt \( x \).

- **Structural control:** \( y \sim p_{\theta'}(x) \)
    - Methods that modify the model's underlying parameters or augment the model's architecture.
    - Facilitated through fine-tuning, adapter layers, or architectural interventions to yield weights \( \theta' \).

- **State control:** \( y \sim p_{\theta}^a(x) \)
    - Methods that modify the model's internal states (e.g., activations, attentions) at inference time.
    - Facilitated through hooks that are inserted into the model to manipulate internal variables during the forward pass.

- **Output control:** \( y \sim d(p_\theta)(x) \)
    - Methods that modify model outputs or constrain/transform what leaves the decoder.
    - Facilitated through decoding-time algorithms/filters that override the `generate` method.

</div>

Given the above structure, AISteer360 enables the composition of various controls into a single operation on a
given model (each exercising control over a different component), in what we term a *steering pipeline*. Steering
pipelines can consist of simply a single control (e.g., activation steering) or a sequence of multiple controls
(e.g., LoRA following by reward-augmented decoding). This flexibility allows users to evaluate the impact of various
steering methods (and combinations thereof) on a given model.

To facilitate a principled comparison, we have developed `UseCase` and `Benchmark` classes. Use cases define tasks for a
(steered) model and specify how performance on that task is measured (via evaluation metrics on the model's generations).
Benchmarks facilitate the comparison of steering pipelines on a given use case. This provides a unified structure for
testing and comparing methods, addressing the current fragmentation in the field where steering algorithms are typically
developed and evaluated within isolated, task-specific environments.[@liang2024controllable]

We encourage the community to use AISteer360 in their steering workflows. We will continue to develop in the open, and
encourage users to suggest any additional features or raise any issues on our [GitHub page](https://github.com/IBM/AISteer360).
