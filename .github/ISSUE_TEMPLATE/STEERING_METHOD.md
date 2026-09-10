---
name: New steering method
about: Propose a new steering method to add to the toolkit
labels: enhancement
---

<!-- If some fields do not apply to your situation, feel free to skip them. -->

## Method
<!-- The name of the method, and a short description of what it does. -->


## Reference
<!-- The paper or other source describing the method, and a link to a reference implementation if one exists. -->

- Title:
- Authors:
- Link:
- Reference implementation (if any):

## Steering category

- [ ] Input control <!-- manipulates the prompt: templating, augmentation, in-context learning -->
- [ ] Structural control <!-- modifies weights or architecture: fine-tuning, DPO/LoRA, merging -->
- [ ] State control <!-- acts on activations at runtime: activation or attention steering -->
- [ ] Output control <!-- guides generation: decoding logic, constrained output, filtering -->

## Preparation
<!-- What happens in steer()? Does the method require training or fitting, and what steering data does it
need? Or is preparation trivial, as it is for few-shot? -->


## Hyperparameters
<!-- The fields that would go in args.py, with their types and any defaults you have in mind. -->


## Runtime arguments
<!-- Does the method need per-example information at inference time (RUNTIME_KWARGS_SCHEMA)? If so, what
names does it consume and what do they mean? For example, PASTA consumes `substrings`. -->

- [ ] This method requires runtime arguments

## Implementation notes
<!-- For state controls, whether it fits the InterventionControl template or needs get_hooks(). For output
controls, whether it is step-level (logits processors, stopping criteria) or owns the decode loop
(DecodingDriver). Any composition constraints or backend limitations worth flagging. -->


## Are you planning to contribute this?

- [ ] Yes, I would like to implement it
- [ ] No, this is a suggestion
