# VJP-delta PR review (read-only)

Review scope: diff `f1d8b5fd6d9ed15d506f9445a93d55cb5c5b07df...HEAD@76a5ee2a710f3720b68793d5376c9ac9db1dae83` in
`/home/code/dev/steerability-vjp`. I read every changed production/test/docs/notebook file in full, the surrounding
framework contracts (sources.py, transforms/additive.py, common/specs.py, base.py, spipe freeze/codec, model_layout,
TokenScope/CollectStateEntries paths, tests/utils/tiny_models), AGENTS.md (Developer guide, Testing, DoD, Invariants),
CONTRIBUTING.md, the PR #12/#19/#32 metadata, and the slop/audits + verification logs. I re-ran the focused suites and
several bounded checks against the venv at `/home/code/dev/steerability/.venv`.

Independent checks run (all read-only / no file edits):

- `pytest tests/controls/test_vjp_delta.py -q` → `9 passed in 5.13s`.
- `pytest tests/controls/test_sources.py tests/controls/test_activation_adapter.py tests/controls/test_spipe_freeze_state.py -q` → `83 passed in 23.61s`.
- Full-suite final log tail: `3257 passed, 340 skipped, 166 warnings in 254.56s` (full_pytest_final.log) — consistent with the claimed evidence; I did not rerun the full suite.
- `pre-commit run --files <all 8 changed code/docs files>` → all hooks passed.
- Pipeline-level interactive check: `check()` plan reports `PlannedStep(control='VJPDelta', access=MODULE, venue='live')`, `steer_fits() == (('VJPDeltaFit', 'direction'),)`, `fit_identity` digests the `VJPDeltaFit` dataclass (encodable, init-fields only via codec's `$dc` path), double `steer()` memo-hits and rebinds, bound directions are `[1, H]` unit-norm on the model device.
- CUDA smoke claim: `gpu_smoke_final_status.json` + `run.md` record job 1649 result `{'module': ..., 'master_device': 'cpu', 'bound_device': 'cuda', 'reply': 'the', 'layers': [0, 1]}` — consistent with `fit.py` storing `.detach().cpu()` masters and `TransformContext.resolve` casting the clone (context.py `resolved.to(device, dtype)`).
- Notebook outputs match execution logs (`{'layers': [0, 1], 'reply': "'the'"}`; `{'bundle': ..., 'reply_matches': True}`); `execution_count` 1..3 across cells.

## 1. Findings

### P0 — none

### P1 — none

No functional blocker in the method math, hook lifetime, state restoration, or freeze path. The intended math is
reproduced exactly (traced line by line against the task spec, and the explicit-Jacobian parity test is a genuine
independent contraction of the same formula — including the unequal-class (2 pos / 1 neg) weighting, the cross-token
dependence assertion (`jacobian[:, 0, 1].abs().sum() > 0`), and the exclusion of `skip_first`/final-token positions —
and it passes at `atol=1e-5` on a batch of 2, so batch/padding separation is also validated).

### P2-1 — `VJPDeltaFit.resolve` memo branch is missing the `model is not None` guard

- Location: `steerability/algorithms/state_control/vjp_delta/fit.py`, `resolve` (the memo-hit line).
- Mechanism: the guard is `if self._model_ref is not None and self._model_ref() is model and self._master is not None:`.
  After the memoized model is garbage-collected, `self._model_ref()` returns `None`, so a subsequent `resolve(None, tokenizer)`
  satisfies the identity test (None is None) and silently returns the stale cached master instead of raising.
  `ContrastiveFit.resolve` in `common/sources.py` guards the same branch with `model is not None and ...`, and
  `SinglePairFit.resolve` raises up front for `model=None`; `_fit` itself documents "requires a live model at steer time".
- Repro (run during review): `del model; gc.collect(); source.resolve(None, tokenizer)` → returned a vector
  (layers [0]) instead of raising.
- User impact: the pipeline always passes a live model for MODULE-access sources (the steer plan venues it `live`),
  so this only bites direct API misuse (reusing one fit instance against `model=None` after its model died, or a
  hand-driven `InterventionControl.steer(None, ...)`). Estimated severity: low, but it silently returns a vector
  fitted for a different, dead model — the silent-staleness aspect is the concern.
- Minimal fix: `if model is not None and self._model_ref is not None and self._model_ref() is model and ...` — one
  line, matching the established source pattern.

### P2-2 — `_class_gradients` wraps every RuntimeError as "backward failed ... must support gradients", including real failures

- Location: `fit.py`, `_class_gradients` `except RuntimeError` block.
- Mechanism: any RuntimeError not containing the literal "VJP-delta" (e.g., CUDA OOM, dtype mismatch, a broken
  attention implementation) is replaced by `RuntimeError("VJP-delta backward failed. The configured model must
  support gradients through its decoder layers.")` (`from error` preserves the chain, so the original traceback is
  reachable). OOM is a RuntimeError, so an out-of-memory fit is reported as an unsupported-model failure.
- User impact: diagnostics only; the misleading headline message on large-model fits.
- Minimal fix: re-raise OOM/shape errors verbatim and wrap only autograd-specific failures; or fold the original
  message text into the wrapper.

### P2-3 — 35 workflow-artifact files under `slop/` are added to the branch (base had none)

- Files: `slop/audits/job_1647.md`, `job_1648.md`, `slop/pr_drafts/2026-09-17_vjp_delta.md`,
  `slop/verification/2026-09-17_vjp-delta/*` (pre-commit/docs/full-suite/notebook/CUDA logs, `gpu_smoke.py`, queue
  status JSON). `git ls-files f1d8b5f | grep ^slop/` → 0; `git ls-files HEAD | grep ^slop/` → 35.
- Mechanism: the diff adds them; the draft file itself ends "draft for editorial review, not publication." Logs embed
  local machine paths and queue metadata.
- User impact: repository hygiene / merge decision, not a code defect. If the maintainers do not want
  agent-workflow artifacts in the tree, these should be excluded/ignored before merge.
- Minimal fix: add `slop/` (or the specific audit/draft/verification subtrees) to `.gitignore` and drop the files
  from the branch, or explicitly confirm they are intended to ship.

### P2-4 — editorial: provenance/license wording and agent-signature comments (flagged by the PR draft itself)

- `control.py` docstring and notebook cell state: "This is a repo-native implementation from the documented
  mathematics and Steerability contracts. It does not copy or assert a license for that source repository." This is
  honest attribution with a pinned commit (`cb03382ebd0cc9cad615d169f42e68e8ae3e12a7`), makes no relicensing claim,
  and matches the "unlicensed provenance repo" fact. The phrase "assert a license" is slightly awkward; the PR body
  already carries the clearer disclaimer. Not a blocker; the draft requested editorial review, which I echo.
- `<!-- PI[...]: ... -->` HTML comments appear in `docs/reference/algorithms/state_control/vjp_delta.md` and
  `examples/notebooks/algorithms/vjp_delta.ipynb` — grep across the other reference pages (`caa.md`, `iti.md`, ...)
  and other algorithm notebooks shows no other file carries them, so they are not an established repo convention;
  strip before merge if agent signatures are unwanted.
- Cosmetic, non-blocking.

### P3 notes (not findings)

- fp16/bf16 fit precision: `grad_outputs = torch.zeros_like(target)` and `cotangent.to(dtype=target.dtype)` compute
  the VJP in the model's reduced precision, upcasting only after the per-prompt average. Functional on CUDA (job
  1649 passed), but large fp16 models will get reduced-precision extraction. Consider an fp32 fit pass if direction
  quality matters; not a defect.
- `cleanup()` drops `interventions` but the template's `VJPDeltaFit` source keeps the CPU master vectors (and the
  weakref to the model remains). GPU copies are freed; the docstring "Drop fitted intervention tensors" is slightly
  stronger than the effect. The weakref test passes (`ref() is None` after `del model; gc.collect()`).
- Gradient-checkpointed models re-fire the fit's forward hooks during `autograd.grad` recomputation; captured lists
  grow with unused entries and `source_states` entries are overwritten after autograd.grad already consumed the
  originals — benign, but untested.
- No test covers the (unlikely) failure "decoder layer returned a non-tensor" branch or the `requires_grad`-guard
  runtime error path (only `inference_mode` is tested); the maintained suite lacks a regression asserting the
  CPU-master/CUDA-bound device split that the CUDA smoke script checks ad hoc.

## 2. Claims checked and found sound

- **Math** matches the documented spec exactly: per-prompt final-token class means → contrast, cotangent placed at
  all valid target positions, per-prompt mean over valid source positions, independent class averaging (2-vs-1-size
  classes), positive-minus-negative, per-layer unit-norm normalization, `source < target` enforced. Parity with an
  explicit Jacobian contraction is a genuine, non-tautological cross-check and passes.
- **Batch/padding**: right-padded rows, `lengths = mask.sum(1) - 1` final-token indexing, causal-mask row
  independence, per-batch width invariance enforced by per-prompt averaging before aggregation — all sound; the
  short-row guard (`lengths <= skip_first + 1`) raises before any hook registration.
- **State restoration**: `_fit` snapshots `training`, per-parameter `requires_grad`, and `.grad` (detach-clone, None
  preserved) before mutation and restores in `finally`; hooks are removed in `finally` in both `_target_mean` and
  `_class_gradients`; weights are never written. Verified by tests (success and error paths) and by inspection.
- **Weakref/cache lifecycle**: memoization keyed on model identity with a defensive clone per resolve; test
  confirms the model can be collected after steer. (Except the P2-1 None-model gap.)
- **Fit identity / staleness / freeze**: `fit_identity()` digests the `VJPDeltaFit` dataclass (init fields only —
  `_model_ref`/`_master` are `init=False`, so codec excludes them); data edits flip the digest (SpipeStaleError)
  while application-only `strength` edits do not; frozen reload lands on `ActivationAdapter` with `steer_fits() == ()`
  and the monkeypatched-`resolve` test proves no VJP runs on reload.
- **Docs/notebook**: nav entry, controls.md catalog entry with honest Backends line (HF fit; HF + vLLM-Hook additive
  wire form — the broadcast transform's `wire_kind="additive"` and `requirements()` conform to CAA's precedent),
  reference page follows caa.md's mkdocstrings block, `examples/index.md` updated, notebook executed with stored
  outputs.
- **Repo conventions**: three-file layout + `STEERING_METHOD` registry shape, `BaseArgs` promotion to control
  attributes, no new dependency, docstrings end with the reference block, `text_config`/`resolve_model_layout` used
  instead of raw `model.config`, pre-commit clean on changed files.

## 3. Gaps I could not verify

- No network access to `wassname/vjp-steering@cb03382`; the contract review is against the documented intended math
  (which the implementation satisfies), not against the upstream implementation itself.
- No steering-efficacy metric: the only behavior evidence is public generation returning `'the'` on tiny models
  (notebook, CUDA smoke, tests).
- I did not rerun the full 3257-test suite (relied on `full_pytest_final.log`); the pre-existing strict-docs-blocking
  warnings and the `rad.ipynb`/`instruction_following.ipynb` high-entropy pre-commit failures are unrelated to this
  PR (verified the failing files are not in the diff).
- The vLLM-Hook wire path was not executed (no engine in this worktree); only the declarative wire-form contract
  was checked.
- Large-model CUDA memory behavior during the fit (full-graph retention from embedding to `num_layers-3` with
  batch_size=8 / max_length=384 defaults) is untested outside tiny fixtures.
- `docs/human_journal.md` is an untracked supervisor session artifact in the worktree (contains session metadata);
  it is not part of the diff but should likely be gitignored.

## 4. Verdict

**Ready after listed fixes** — all listed fixes are P2 (one-line guard in `resolve`, error-wrap selectivity,
`slop/` exclusion decision, optional comment/license wording cleanup); none block the method's function, and the
mathematics, lifecycle restoration, freeze/reload behavior, and framework integration are correct as implemented.

PI[deepseek-v4-flash]
