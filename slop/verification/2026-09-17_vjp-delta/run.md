# VJP-delta verification log

- base: f1d8b5fd6d9ed15d506f9445a93d55cb5c5b07df
- branch: feat/vjp-delta
- assignment: exact parent `01a09cdb-9985-75d5-9f07-68b80024a10b`

## Initial import

- `/home/code/dev/steerability/.venv/bin/python -m compileall -q steerability/algorithms/state_control/vjp_delta`: passed.
- Registry import reported `VJPDelta` with `ModelAccess.MODULE`.
- `uv run --no-sync` in the new worktree did not have dependencies and was not used for tests. The existing repository environment is used read-only for local verification.

## Focused verification

| command | result | evidence |
| --- | --- | --- |
| `pytest tests/controls/test_vjp_delta.py -q` | 9 passed | [test_vjp_delta_final.log](test_vjp_delta_final.log) |
| `pytest tests/controls/test_vjp_delta.py tests/controls/test_sources.py tests/controls/test_activation_adapter.py tests/controls/test_spipe_freeze_state.py -q` | 91 passed | [focused_tests.log](focused_tests.log) |
| `pre-commit run --files ...` | passed | [precommit_final.log](precommit_final.log) |
| `mkdocs build` | passed with existing warnings | [docs_build_nonstrict.log](docs_build_nonstrict.log) |
| `mkdocs build --strict` | failed on 23 existing unrelated warnings | [docs_build.log](docs_build.log) |
| executed `vjp_delta.ipynb` | all three code cells completed | [notebook_execution_retry.log](notebook_execution_retry.log) |

## CUDA smoke

GPU queue job `1649` used the default one-worker group and completed in 9 seconds. Its complete result was:

> `{'module': '/home/code/dev/steerability-vjp/steerability/__init__.py', 'master_device': 'cpu', 'bound_device': 'cuda', 'reply': 'the', 'layers': [0, 1]}`

The two failed harness attempts are audited in [job_1647.md](../../audits/job_1647.md) and [job_1648.md](../../audits/job_1648.md). They failed a post-bind device assertion and an output-only private-property lookup, respectively; neither was interpreted as a method failure.

## Full suite

The first full-suite run completed with `3256 passed, 340 skipped, 166 warnings in 262.72s`; see [full_pytest.log](full_pytest.log). A final full-suite run after the remaining regression updates is active at this log path: [full_pytest_final.log](full_pytest_final.log).

## Review fixes

- `VJPDeltaFit.resolve(None, tokenizer)` can no longer match a dead weakref cache entry; it runs the live-model guard and raises.
- Model-forward failures now propagate unchanged. `torch.autograd.grad` failures retain their original message in the `VJP-delta autograd.grad failed: ...` headline.
- Both parent review outputs are copied byte-for-byte to `slop/reviews/`; [review_copy_hashes.log](review_copy_hashes.log) records matching SHA-256 values.
- Lifecycle-focused tests: `93 passed` in [review_fix_lifecycle_tests.log](review_fix_lifecycle_tests.log). The post-review full suite: `3258 passed, 340 skipped, 166 warnings in 283.24s` in [review_fix_full_pytest.log](review_fix_full_pytest.log).
- Changed production/test files passed pre-commit in [review_fix_precommit_code.log](review_fix_precommit_code.log). The exact copied oracle review has no terminal newline; the end-of-file hook would alter that source artifact, so it was excluded from formatter execution to preserve the required byte identity.

## Residual risks

- The VJP path supports standard differentiable torch decoder execution. Inference mode and unsupported backward paths raise rather than changing method behavior.
- Frozen `.spipe` resolution stores its historical additive transform. Editing application strength in a historical recipe does not alter that already-resolved transform, matching the existing `ActivationAdapter` freeze form.
- The external provenance repository is unlicensed. This branch records attribution and makes no relicensing claim.

<!-- PI[gpt-5.6-terra]: verification record. -->
