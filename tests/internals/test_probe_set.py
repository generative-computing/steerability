"""ProbeSet and ProbeSetFit tests: construction guards, read parity, hook hygiene, coexistence.

Runs hub-free on a tiny randomly-initialized Llama. Behavioral read assertions use hand-built
probes whose large-magnitude biases force known outcomes; parity assertions compare `read()`
against offline capture plus `score_hidden` on the same prompts.
"""
import warnings

import pytest
import torch

from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
from steerability.algorithms.core.internals.probes.probe import Probe
from steerability.algorithms.core.internals.probes.probe_set import ProbeReadings, ProbeSet, ProbeSetFit
from steerability.algorithms.core.internals.stats import StatsSpec
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

DATA = {
    "topic": ContrastivePairs(
        positives=["the cat sat on mat", "the cat ran"],
        negatives=["dog ran fast", "the dog ran"],
    ),
}


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _probe(layer_ids, bias=0.0, seed=7, model_type="llama", location="layer_input",
           pooling="mean", meta=None):
    return Probe(
        model_type=model_type,
        location=location,
        pooling=pooling,
        layer_ids=list(layer_ids),
        weights={lid: _unit_vector(seed + lid) for lid in layer_ids},
        bias=bias,
        meta=meta or {},
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one probe"):
            ProbeSet({})

    def test_mixed_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            ProbeSet({"a": _probe([1]), "b": _probe([2], model_type="gpt2")})

    def test_mixed_location_raises(self):
        with pytest.raises(ValueError, match="location"):
            ProbeSet({"a": _probe([1]), "b": _probe([2], location="layer_output")})

    def test_unequal_recorded_fingerprints_raise(self):
        with pytest.raises(ValueError, match="fingerprints"):
            ProbeSet({
                "a": _probe([1], meta={"model_fingerprint": "aaaa"}),
                "b": _probe([2], meta={"model_fingerprint": "bbbb"}),
            })

    def test_empty_meta_probes_mix_freely(self):
        probes = ProbeSet({
            "a": _probe([1], meta={"model_fingerprint": "aaaa"}),
            "b": _probe([2]),  # no recorded fingerprint
        })
        assert probes.names == ("a", "b")

    def test_layer_union_sorted(self):
        probes = ProbeSet({"a": _probe([3, 1]), "b": _probe([2, 1])})
        assert probes.layer_ids == [1, 2, 3]


class TestRead:
    def _forced_set(self):
        return ProbeSet({
            "always": _probe([1], bias=1e9),
            "never": _probe([2], bias=-1e9),
        })

    def test_forced_decisions_per_row(self, model):
        readout = self._forced_set().read(model, torch.tensor([[3, 4, 5], [6, 7, 8]]))
        assert isinstance(readout, ProbeReadings)
        assert readout.decisions["always"].tolist() == [True, True]
        assert readout.decisions["never"].tolist() == [False, False]
        assert readout.scores["always"].shape == (2,)

    def test_latest_stashed(self, model):
        probes = self._forced_set()
        assert probes.latest is None
        readout = probes.read(model, torch.tensor([[3, 4, 5]]))
        assert probes.latest is readout
        second = probes.read(model, torch.tensor([[3, 4], [5, 6]]))
        assert probes.latest is second
        assert readout.decisions["always"].numel() == 1  # earlier readout untouched

    def test_parity_with_offline_capture(self, model, tokenizer):
        probe = _probe([1, 2], bias=0.1, pooling="last")
        probes = ProbeSet({"p": probe})

        enc = tokenizer(["the cat sat on mat", "dog ran"], return_tensors="pt", padding=True)
        readout = probes.read(model, enc["input_ids"], enc["attention_mask"])

        hidden = layerwise_tokenwise_hidden(model, dict(enc), location="layer_input")
        expected = probe.score_hidden(
            {lid: hidden[lid].to(torch.float32) for lid in probe.layer_ids},
            prompt_mask=enc["attention_mask"],
        )
        assert torch.allclose(readout.scores["p"], expected, atol=1e-4)

    def test_padded_row_scores_as_if_unpadded(self, model):
        probe = _probe([1], pooling="mean")
        probes = ProbeSet({"p": probe})

        ids = torch.tensor([[3, 4, 5, 2], [6, 7, 8, 9]])  # row 0 right-padded
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        padded = probes.read(model, ids, mask)
        unpadded = probes.read(model, torch.tensor([[3, 4, 5]]))
        assert torch.allclose(padded.scores["p"][0], unpadded.scores["p"][0], atol=1e-4)

    def test_model_type_guard(self, model):
        probes = ProbeSet({"p": _probe([1], model_type="gpt2")})
        with pytest.raises(ValueError, match="model_type"):
            probes.read(model, torch.tensor([[3, 4, 5]]))

    def test_layer_out_of_range_raises(self, model):
        probes = ProbeSet({"p": _probe([99])})
        with pytest.raises(ValueError, match="out of range"):
            probes.read(model, torch.tensor([[3, 4, 5]]))

    def test_layer_output_location_raises(self, model):
        probes = ProbeSet({"p": _probe([1], location="layer_output")})
        with pytest.raises(ValueError, match="layer_input"):
            probes.read(model, torch.tensor([[3, 4, 5]]))


class TestHookHygiene:
    def _hook_counts(self, model):
        return [
            len(model.get_submodule(f"model.layers.{i}")._forward_pre_hooks)
            for i in range(LAYERS)
        ]

    def test_hooks_removed_after_read(self, model):
        probes = ProbeSet({"p": _probe([1, 2])})
        before = self._hook_counts(model)
        probes.read(model, torch.tensor([[3, 4, 5]]))
        assert self._hook_counts(model) == before

    def test_hooks_removed_when_forward_raises(self, model):
        probes = ProbeSet({"p": _probe([1, 2])})
        before = self._hook_counts(model)
        original_forward = model.forward

        def boom(*args, **kwargs):
            raise RuntimeError("forward failure")

        model.forward = boom
        try:
            with pytest.raises(RuntimeError, match="forward failure"):
                probes.read(model, torch.tensor([[3, 4, 5]]))
        finally:
            model.forward = original_forward
        assert self._hook_counts(model) == before


class TestCoexistence:
    """`read()` is an aligned auxiliary pass: co-resident condition scoring ignores it while
    `"all"`-scoped behavior transforms apply to it."""

    def test_read_skips_condition_scoring_and_applies_behavior(self, model):
        from steerability.algorithms.state_control.common.gating import CallableReadout, Evidence, Gate
        from steerability.algorithms.state_control.common.runtime import TransformHookRuntime
        from steerability.algorithms.state_control.common.token_scope import compute_prompt_lens
        from tests.utils.runtime_helpers import NeverCompleteRule, RecordingTransform

        ids = torch.tensor([[3, 4, 5, 6]])
        probes = ProbeSet({"p": _probe([2])})
        baseline = probes.read(model, ids).scores["p"].clone()

        runtime = TransformHookRuntime(hook_point="layer_output")
        runtime.reset(compute_prompt_lens(ids, None))
        readout_calls: list[tuple] = []

        def readout(pooled, layer_id):
            readout_calls.append(tuple(pooled.shape))
            return torch.zeros(pooled.size(0))

        gate = Gate(Evidence((0,), CallableReadout(readout)), NeverCompleteRule(open=True))
        gate.reset(1)

        transform = RecordingTransform(value=0.5)
        condition_hook = runtime.build_condition_hook(
            layer_id=0, gate=gate, is_pass_opener=True
        )
        behavior_hook = runtime.build_behavior_hook(
            layer_id=0, transform=transform, gate=gate, token_scope="all",
            is_pass_opener=False,
        )
        layer0 = model.get_submodule("model.layers.0")
        handles = [
            layer0.register_forward_hook(condition_hook, with_kwargs=True),
            layer0.register_forward_hook(behavior_hook, with_kwargs=True),
        ]
        try:
            steered = probes.read(model, ids).scores["p"].clone()
        finally:
            for handle in handles:
                handle.remove()

        assert readout_calls == []  # condition scoring ignored the auxiliary pass
        assert transform.masks  # the "all"-scoped transform applied during the read
        assert not torch.allclose(steered, baseline)  # scores measure the stream as deployed

    def test_read_leaves_live_cast_counters_and_gates_untouched(self, model, tokenizer, monkeypatch):
        from steerability.algorithms.state_control.cast.control import CAST
        from steerability.algorithms.state_control.common.steering_vector import SteeringVector

        def steering_vector(seed, layers):
            return SteeringVector(
                model_type="llama",
                directions={lid: _unit_vector(seed + lid).unsqueeze(0) for lid in layers},
            )

        cast = CAST(
            behavior_vector=steering_vector(100, [0, 1]),
            behavior_layer_ids=[0, 1],
            behavior_vector_strength=1.0,
            condition_vector=steering_vector(200, [1]),
            condition_layer_ids=[1],
            condition_vector_threshold=0.5,
            condition_comparator_threshold_is="ge",
        )
        cast.steer(model, tokenizer)

        from tests.utils.runtime_helpers import capture_built_runtimes

        capture = capture_built_runtimes(monkeypatch)
        ids = torch.tensor([[3, 4, 5, 6]])
        hooks = cast.get_hooks(ids, None)
        runtime = capture.last
        handles = []
        for phase, register in (("pre", "register_forward_pre_hook"), ("forward", "register_forward_hook")):
            for spec in hooks[phase]:
                module = model.get_submodule(spec["module"])
                handles.append(getattr(module, register)(spec["hook_func"], with_kwargs=True))
        try:
            assert not cast._gate.is_ready()
            assert cast._gate.evidence_values() == {}
            offset_before = runtime._offset
            prefill_before = runtime._prefill_seen

            ProbeSet({"p": _probe([2])}).read(model, ids)

            assert not cast._gate.is_ready()  # no condition evidence from the auxiliary pass
            assert cast._gate.evidence_values() == {}
            assert runtime._offset == offset_before
            assert runtime._prefill_seen == prefill_before
        finally:
            for handle in handles:
                handle.remove()


class TestProbeSetFit:
    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="at least one probe name"):
            ProbeSetFit(data={})

    def test_stats_requiring_method_without_stats_raises(self):
        with pytest.raises(ValueError, match="requires ambient activation"):
            ProbeSetFit(data=DATA, spec=ProbeFitSpec(method="lda"))

    def test_per_name_spec_must_cover_names(self):
        with pytest.raises(ValueError, match="missing entries"):
            ProbeSetFit(data=DATA, spec={"other": ProbeFitSpec(method="mean_diff")})

    def test_unknown_calibration_name_raises(self):
        with pytest.raises(ValueError, match="unknown probe"):
            ProbeSetFit(
                data=DATA,
                spec=ProbeFitSpec(method="mean_diff"),
                calibration_data={"ghost": DATA["topic"]},
            )

    def test_names_available_before_fitting(self):
        recipe = ProbeSetFit(data=DATA, spec=ProbeFitSpec(method="mean_diff"))
        assert recipe.names == ("topic",)

    def test_fit_resolves_statsspec_on_given_model(self, model, tokenizer):
        recipe = ProbeSetFit(
            data=DATA,
            spec=ProbeFitSpec(method="lda", candidate_layers=[1]),
            stats=StatsSpec(texts=["the cat sat on mat", "dog ran fast"], location="layer_input"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # low-sample stats warning
            probes = recipe.fit(model, tokenizer)
        assert isinstance(probes, ProbeSet)
        assert probes.names == ("topic",)
        fingerprint = model_fingerprint(model)
        for probe in probes.probes.values():
            assert probe.meta["model_fingerprint"] == fingerprint

    def test_eager_fit_smoke(self, model, tokenizer):
        probes = ProbeSet.fit(
            model, tokenizer, data=DATA, spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1])
        )
        readout = probes.read(model, torch.tensor([[3, 4, 5]]))
        assert set(readout.decisions) == {"topic"}

    def test_fit_and_read_on_composite_wrapper(self, tokenizer):
        """ProbeSet.fit and read succeed on a composite multimodal wrapper (nested decoder root)."""
        from steerability.algorithms.core.internals.model_layout import resolve_model_layout
        from tests.utils.tiny_models import tiny_gemma3_conditional

        model = tiny_gemma3_conditional(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        probes = ProbeSet.fit(
            model, tokenizer, data=DATA, spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1])
        )
        assert resolve_model_layout(model).layer_names == [
            f"model.language_model.layers.{i}" for i in range(LAYERS)
        ]
        readout = probes.read(model, torch.tensor([[3, 4, 5]]))
        assert set(readout.decisions) == {"topic"}
