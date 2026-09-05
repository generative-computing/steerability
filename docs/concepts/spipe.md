# Sharing pipelines (`.spipe`)

A steered pipeline normally exists only as in-memory Python objects. The `.spipe` format writes a pipeline down as a
portable bundle that can be saved, version-controlled, and handed to another person or machine.

One format contains two layers of information:

- **The recipe**: the model reference plus the controls exactly as constructed. Every `.spipe` contains the recipe.
  Loading a recipe-only bundle and calling `steer()` re-runs any fits and training. Results may differ across
  machines because of GPU nondeterminism.
- **The frozen resolution**: what the steer step actually produced, i.e., fitted steering vectors, probes, LoRA
  adapters, and optimized prompts, stored content-addressed alongside the recipe. A lock section pins fingerprints of
  the producing model and per-fit digests of the recipe fields each artifact came from. Loading a frozen bundle
  yields controls in precomputed form, where `steer()` still runs but is cheap and model-free.

Freezing is a rewrite of the recipe rather than a second format. A frozen entry is an ordinary control constructed
with precomputed arguments: a CAA fitted from data freezes as a CAA constructed with the fitted vector, and a
fine-tune freezes as a [`LoadLoRA`](../reference/algorithms/structural_control/load_lora.md) or
[`LoadCheckpoint`](../reference/algorithms/structural_control/load_checkpoint.md) pointing at the trained product.
Loading therefore takes the same construction and `steer()` path as any hand-built pipeline.

## Saving and loading

```python
pipeline.steer()
spipe = pipeline.to_spipe()          # frozen by default once steered
spipe.save("formal_tone.spipe")      # a .spipe path writes a zip (any other path writes a directory)
```

```python
from steerability.spipe import SPipe

spipe = SPipe.load("formal_tone.spipe")
pipeline = spipe.pipeline()          # backend, device, and dtype stay the caller's choice
pipeline.steer()                     # installs the frozen artifacts and fits nothing
response = pipeline.generate(...)
```

`to_spipe(freeze=False)` forces a recipe-only bundle from a steered pipeline, and `spipe.thaw()` turns a frozen bundle
back into its recipe. `save(..., artifacts="thin")` writes the manifest without the artifact payloads. A thin bundle
loads against an external store via `SPipe.load(path, artifact_store=...)`.

## Staleness

The lock records, per frozen artifact, a digest of the recipe fields a re-fit would consume. Editing an inert
application parameter in the manifest (say, a CAA multiplier) leaves the pinned vector valid. Editing the training
data does not, and loading then fails with a staleness error that identifies the control and the fix (`thaw()` and re-steer,
or `allow_stale=True`).

## Verification

`spipe.verify()` reports on a bundle without loading a model: format validity, artifact integrity, staleness, version
compatibility, and whether the bundle references code. At `steer()` time, frozen steering artifacts are checked
against the model they are being installed on, under a policy chosen at `pipeline(verify=...)`:

- `"strict"` (default): a wrong architecture or width is an error. A calibrated artifact (a probe, a gate threshold)
  on a model with a different weight fingerprint is an error. A direction artifact on different weights of the same
  architecture is a warning, since a direction can be transferred across fine-tunes deliberately.
- `"warn"`: every mismatch is a warning.
- `"off"`: no checks.

## Trust and `allow_code`

A `.spipe` from someone else is untrusted input. Loading never unpickles by default. Tensors are stored only as
safetensors, archives are extracted behind zip-safety guards, and every artifact is verified against its content
hash. Two things require an explicit `allow_code=True` at load, similar to `trust_remote_code`:

- References to Python callables, e.g., a scorer function a prompt optimizer was configured with. The manifest's
  `code_dependent` flag says up front whether a bundle needs this, and the referenced modules must be on the import
  path.
- Pickle-backed memory payloads (CPO's trained scorer memory), since unpickling executes code.

Frozen prompt-optimization bundles keep their search-only arguments (scorers, budgets) for provenance. A bundle whose
optimizer used a custom scorer is therefore code-dependent even though the frozen memory never calls it.

## Identity

Every bundle contains two digests. `config_id` is the same configuration identity that `SteeringEval` records, which
ties a `.spipe` to evaluation results. `recipe_id` additionally includes the model reference, since a steering
artifact is meaningless without its model.
