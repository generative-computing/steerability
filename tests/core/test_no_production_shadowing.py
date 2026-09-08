"""
Meta-test enforcing the no-shadowing invariant for the test suite.

Flags, anywhere under `tests/`:

- `class` definitions whose name matches a production class name
- `def` definitions whose name matches a production function name
- imports of production names from a conftest module

The module also runs standalone against an arbitrary directory
(`python3 tests/core/test_no_production_shadowing.py <directory>`), printing findings and
exiting nonzero when any are present.
"""
import ast
import sys
from pathlib import Path

PRODUCTION_CLASSES = {
    "BaseArgs", "BaseControl",
    "InputControl", "StructuralControl", "StateControl", "OutputControl",
    "DecodingDriver",
    "InterventionControl", "HookControl", "SteeredSession",
    "SteeringPipeline", "ControlSpec", "Output",
    "SampleScorer", "SampleSequenceScorer", "TaskEvaluationScorer",
    "ConfigPoint", "PipelineFactory",
    "ProviderOptions", "SteeringPipelineModelAPI", "LockLeaderCollator", "BatchRequest",
    "InspectSuite", "SteeringEval",
    "Backend", "BackendSpec", "BackendCapabilities", "Capability",
    "InterventionKinds", "ProcessorKinds", "CaptureKinds",
    "Requirements", "SpecConstraint", "SupportReport", "SupportFailure",
    "ConstrainedDecoding",
    "ConstraintSource",
    "ConstraintKinds",
    "ConstraintEntry",
    "InterventionSpec",
    "InterventionEntry",
    "HFBackend", "ExclusiveSession", "SteeringSession", "ModelLayout", "ModelFacts",
    "PreparedPrompt", "GenerationParams", "GenerationItem", "ScoringItem",
    "ItemResult", "CaptureResult", "HookEntry", "StackEntry",
    "VLLMBackend", "VLLMServeBackend", "VLLMOfflineSession", "VLLMServeSession",
    "PartialBatchError", "TransportError",
    "Artifact", "ArtifactProvenance", "ModelArtifact", "CheckpointArtifact", "LoRAArtifact",
}

PRODUCTION_FUNCTIONS = {
    "merge_controls", "ensure_pad_token", "warn_if_adapt_messages_bypassed",
    "infer_attention_mask_from_ids", "to_left_pad", "warn_if_duplicate_bos",
    "derive_item_seed", "run_bounded", "with_transport_retries",
    "render_vllm_sampling_args", "truncate_at_stop_strings", "merge_lowered_params",
    "runtime_kwargs_schema", "expand_configurations", "preflight",
    "as_inspect_model", "sample_scorer_from_inspect", "runtime_kwargs_solver",
}


def check_file(path: Path) -> list[tuple[Path, int, str]]:
    """Return `(path, lineno, message)` findings for one module."""
    findings = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in PRODUCTION_CLASSES:
            findings.append((path, node.lineno, f"class `{node.name}` shadows a production class"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PRODUCTION_FUNCTIONS:
            findings.append((path, node.lineno, f"def `{node.name}` shadows a production function"))
        if isinstance(node, ast.ImportFrom) and node.module and "conftest" in node.module:
            bad = sorted({alias.name for alias in node.names} & (PRODUCTION_CLASSES | PRODUCTION_FUNCTIONS))
            if bad:
                findings.append((path, node.lineno, f"imports production names from conftest: {', '.join(bad)}"))
    return findings


def check_tree(root: Path) -> list[tuple[Path, int, str]]:
    """Return findings for every `.py` file under `root`."""
    findings = []
    for path in sorted(root.rglob("*.py")):
        findings.extend(check_file(path))
    return findings


def test_no_production_shadowing_in_tests():
    tests_root = Path(__file__).resolve().parents[1]
    findings = check_tree(tests_root)
    assert not findings, "\n".join(f"{path}:{lineno}: {message}" for path, lineno, message in findings)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests")
    results = check_tree(target)
    for finding_path, lineno, message in results:
        print(f"{finding_path}:{lineno}: {message}")
    if results:
        print(f"\n{len(results)} finding(s).")
        sys.exit(1)
    print("OK: no production symbols are shadowed in test modules.")
