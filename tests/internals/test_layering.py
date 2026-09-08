"""Machine checks of the `core/internals` dependency DAG.

A subprocess imports every `core/internals` module and asserts that no `*_control` category
package loads; a second case shows `Probe.as_gate()` is the single edge that pulls one in.
The registry assertion keeps `core/internals` structurally outside the steering-method crawl.
"""
import json
import subprocess
import sys

INTERNALS_MODULES = [
    "steerability.algorithms.core.internals",
    "steerability.algorithms.core.internals.capture",
    "steerability.algorithms.core.internals.data",
    "steerability.algorithms.core.internals.encoding",
    "steerability.algorithms.core.internals.fingerprint",
    "steerability.algorithms.core.internals.pooling",
    "steerability.algorithms.core.internals.render",
    "steerability.algorithms.core.internals.stats",
    "steerability.algorithms.core.internals.probes",
    "steerability.algorithms.core.internals.probes.probe",
    "steerability.algorithms.core.internals.probes.fitting",
    "steerability.algorithms.core.internals.probes.probe_set",
]

_CATEGORY_SCAN = """
def category_modules(modules):
    bad = []
    for name in modules:
        if not name.startswith("steerability.algorithms."):
            continue
        segments = name.split(".")
        if len(segments) > 2 and segments[2].endswith("_control"):
            bad.append(name)
    return sorted(bad)
"""


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_internals_imports_load_no_category_package():
    code = _CATEGORY_SCAN + f"""
import importlib
import json
import sys

for module in {INTERNALS_MODULES!r}:
    importlib.import_module(module)

print(json.dumps(category_modules(sys.modules)))
"""
    assert json.loads(_run(code)) == []


def test_as_gate_is_the_single_category_edge():
    code = _CATEGORY_SCAN + """
import json
import sys

import torch

from steerability.algorithms.core.internals.probes import Probe

before = category_modules(sys.modules)

probe = Probe(
    model_type="llama", location="layer_input", pooling="mean",
    layer_ids=[0], weights={0: torch.ones(4)}, bias=0.0,
)
probe.as_gate()

after = category_modules(sys.modules)
print(json.dumps({"before": before, "loaded_state_control": any(
    name.startswith("steerability.algorithms.state_control") for name in after
)}))
"""
    result = json.loads(_run(code))
    assert result["before"] == []
    assert result["loaded_state_control"] is True


def test_registry_crawl_excludes_core_internals():
    code = """
import json
from pathlib import Path

import steerability.algorithms.core.registry as registry

crawled_dirs = [
    d.name for d in sorted(registry.ROOT.iterdir())
    if d.is_dir() and d.name not in {"core", "__pycache__"}
]
registered_modules = [
    method.control_cls.__module__
    for bucket in registry.REGISTRY.values()
    for method in bucket.values()
]
print(json.dumps({"crawled_dirs": crawled_dirs, "registered_modules": registered_modules}))
"""
    result = json.loads(_run(code))
    assert "core" not in result["crawled_dirs"]
    assert all("core.internals" not in module for module in result["registered_modules"])


def test_orchestration_modules_do_not_import_internals():
    code = """
import json
import sys

import steerability.algorithms.core.steering_pipeline
import steerability.algorithms.core.specs

print(json.dumps(sorted(
    name for name in sys.modules if name.startswith("steerability.algorithms.core.internals")
)))
"""
    assert json.loads(_run(code)) == []
