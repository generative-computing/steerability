"""Conditional-gating tests for CAST: per-row gates, padding-aware scoring, config, diagnostics.

Runs hub-free on a tiny randomly-initialized Llama. Behavioral assertions target the per-row gate
plumbing (each logical row is gated independently) rather than natural-language output.
"""
import warnings

import pytest
import torch

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.cast.args import CASTArgs
from aisteer360.algorithms.state_control.cast.control import CAST
from aisteer360.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _steering_vector(seed: int, layers) -> SteeringVector:
    return SteeringVector(
        model_type="llama",
        directions={l: _unit_vector(seed + l).unsqueeze(0) for l in layers},
        explained_variances={l: 0.5 for l in layers},
    )


class _RecordingTransform:
    """Captures the token_mask passed to the wrapped transform on each behavior application."""

    def __init__(self, inner):
        self._inner = inner
        self.masks: list[torch.Tensor] = []

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        self.masks.append(token_mask.detach().clone())
        return self._inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs)


def _build_cast(condition_threshold, comparator="ge", comparison_mode="mean"):
    behavior_vec = _steering_vector(seed=100, layers=[0, 1])
    condition_vec = _steering_vector(seed=200, layers=[1])
    return CAST(
        behavior_vector=behavior_vec,
        behavior_layer_ids=[0, 1],
        behavior_vector_strength=1.0,
        condition_vector=condition_vec,
        condition_layer_ids=[1],
        condition_vector_threshold=condition_threshold,
        condition_comparator_threshold_is=comparator,
        condition_threshold_comparison_mode=comparison_mode,
    )


def _steer_pipeline(control, seed: int = 0):
    torch.manual_seed(seed)  # fixed so a probe run and the graded run share the same model weights
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestConfigMatrix:
    def _base(self, **overrides):
        kwargs = dict(behavior_vector=_steering_vector(1, [0]))
        kwargs.update(overrides)
        return kwargs

    def test_layers_without_threshold_raises(self):
        with pytest.raises(ValueError):
            CASTArgs(**self._base(
                condition_vector=_steering_vector(2, [1]),
                condition_layer_ids=[1],
                condition_vector_threshold=None,
                search=ConditionSearchSpec(auto_find=False),
            ))

    def test_threshold_without_layers_raises(self):
        with pytest.raises(ValueError):
            CASTArgs(**self._base(
                condition_vector=_steering_vector(2, [1]),
                condition_layer_ids=None,
                condition_vector_threshold=0.1,
                search=ConditionSearchSpec(auto_find=False),
            ))

    def test_condition_data_without_autofind_or_point_raises(self):
        with pytest.raises(ValueError):
            CASTArgs(**self._base(
                condition_data={"positives": ["a"], "negatives": ["b"]},
                search=ConditionSearchSpec(auto_find=False),
            ))

    def test_valid_manual_point(self):
        args = CASTArgs(**self._base(
            condition_vector=_steering_vector(2, [1]),
            condition_layer_ids=[1],
            condition_vector_threshold=0.1,
            search=ConditionSearchSpec(auto_find=False),
        ))
        assert args.condition_vector_threshold == 0.1

    def test_valid_unconditional(self):
        args = CASTArgs(behavior_vector=_steering_vector(1, [0]))
        assert args.condition_vector is None


class TestDiagnosticsLifecycle:
    def test_latest_decision_none_before_generation(self):
        control = _build_cast(condition_threshold=0.0)
        _steer_pipeline(control)
        assert control.latest_decision is None

    def test_decision_populated_and_reset(self):
        control = _build_cast(condition_threshold=0.0)
        pipeline, _, _ = _steer_pipeline(control)
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=4)

        decision = control.latest_decision
        assert decision is not None
        assert set(decision.scores.keys()) == {1}
        assert all(isinstance(v, float) for v in decision.scores.values())
        assert len(decision.open_per_row) == 1

        # a new generation's hook build resets the gate; the decision clears
        control.get_hooks(torch.tensor([[3, 4, 5]]), None)
        assert control.latest_decision is None
        assert not control._gate.is_ready()  # evidence cleared; gate awaits the next prefill


class TestPerRowGating:
    def _generate_capture(self, control, input_ids, max_new_tokens=3):
        pipeline, _, _ = _steer_pipeline(control)
        recorder = _RecordingTransform(control._transform)
        control._transform = recorder
        pipeline.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        return control, recorder

    @staticmethod
    def _decode_mask(recorder):
        # the last recorded behavior mask is a decode pass (gate fully cached), reflecting per-row gating
        return recorder.masks[-1]

    def test_gate_mask_matches_open_rows(self):
        control = _build_cast(condition_threshold=0.0, comparator="ge")
        input_ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        control, recorder = self._generate_capture(control, input_ids)

        decision = control.latest_decision
        assert decision is not None
        assert len(decision.open_per_row) == 2

        mask = self._decode_mask(recorder)  # [B, T] on a decode pass
        assert mask.size(0) == 2
        for row, is_open in enumerate(decision.open_per_row):
            assert bool(mask[row].any()) == is_open

    def _separating_threshold(self, control, input_ids):
        # run once with an all-open threshold to read the runtime per-row scores, then bisect them
        pipeline, _, _ = _steer_pipeline(control)
        pipeline.generate(input_ids=input_ids, max_new_tokens=1)
        layer_id = next(iter(control._cond_config.layer_ids))
        row_scores = control.latest_decision.scores_per_row[layer_id]
        lo, hi = sorted(row_scores)
        return lo, hi

    def test_rows_gated_independently(self):
        input_ids = torch.tensor([[3, 4, 5, 6], [11, 12, 13, 14]])
        probe = _build_cast(condition_threshold=-1.0, comparator="ge")
        lo, hi = self._separating_threshold(probe, input_ids)
        if hi - lo < 1e-5:
            pytest.skip("tiny-model condition scores not separable for this seed")
        sep = (lo + hi) / 2

        control = _build_cast(condition_threshold=sep, comparator="ge")
        control, recorder = self._generate_capture(control, input_ids)
        decision = control.latest_decision

        assert sum(decision.open_per_row) == 1  # exactly one row opens
        mask = self._decode_mask(recorder)
        steered_rows = [row for row in range(2) if bool(mask[row].any())]
        assert steered_rows == [row for row, is_open in enumerate(decision.open_per_row) if is_open]

    def test_row_zero_closed_row_one_open(self):
        # regression guard: scoring only row 0 (the old behavior) would gate the whole batch on row 0
        input_ids = torch.tensor([[3, 4, 5, 6], [11, 12, 13, 14]])
        probe = _build_cast(condition_threshold=-1.0, comparator="ge")
        lo, hi = self._separating_threshold(probe, input_ids)
        if hi - lo < 1e-5:
            pytest.skip("tiny-model condition scores not separable for this seed")
        sep = (lo + hi) / 2

        control = _build_cast(condition_threshold=sep, comparator="ge")
        control, recorder = self._generate_capture(control, input_ids)
        row_scores = control.latest_decision.scores_per_row[next(iter(control._cond_config.layer_ids))]

        # whichever row scored higher is the one steered; the other is not
        higher_row = 0 if row_scores[0] > row_scores[1] else 1
        mask = self._decode_mask(recorder)
        assert bool(mask[higher_row].any())
        assert not bool(mask[1 - higher_row].any())


class TestPaddingAwareMean:
    def test_mean_ignores_pads(self):
        # aggregate_condition_hidden mean must ignore poisoned pad positions
        torch.manual_seed(0)
        hidden = torch.randn(2, 6, HIDDEN)
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[0, 4:] = 0  # right padding on row 0
        mask[1, :2] = 0  # left padding on row 1
        clean = aggregate_condition_hidden(hidden, "mean", mask)
        poisoned = hidden.clone()
        poisoned[mask == 0] = 1e6
        assert torch.allclose(clean, aggregate_condition_hidden(poisoned, "mean", mask))


class TestConditionMaskThreading:
    """WS2: `get_hooks` hands the pipeline-supplied attention mask to `build_hooks`; falls back to
    leading/trailing-only inference (interior pad==eos preserved) when omitted."""

    def _steered_control(self):
        control = _build_cast(condition_threshold=0.0)
        _steer_pipeline(control)  # attaches tokenizer + resolves the condition config
        return control

    def _built_prompt_mask(self, control, ids, attention_mask, monkeypatch):
        import aisteer360.algorithms.state_control.common.runtime as runtime_module

        captured = {}
        original = runtime_module.build_hooks

        def capture(interventions, layout, prompt_lens, prompt_mask=None, model=None):
            captured["prompt_mask"] = prompt_mask
            return original(interventions, layout, prompt_lens, prompt_mask, model=model)

        monkeypatch.setattr(runtime_module, "build_hooks", capture)
        control.get_hooks(ids, runtime_kwargs=None, attention_mask=attention_mask)
        return captured["prompt_mask"]

    def test_supplied_attention_mask_used_verbatim(self, monkeypatch):
        control = self._steered_control()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        attention_mask = torch.tensor([[1, 1, 0, 1, 0]])  # arbitrary, includes an interior zero
        prompt_mask = self._built_prompt_mask(control, ids, attention_mask, monkeypatch)
        assert prompt_mask is not None
        assert prompt_mask.dtype == torch.bool
        assert prompt_mask.tolist() == [[True, True, False, True, False]]

    def test_supplied_1d_mask_unsqueezed(self, monkeypatch):
        control = self._steered_control()
        ids = torch.tensor([[3, 4, 5]])
        prompt_mask = self._built_prompt_mask(control, ids, torch.tensor([1, 0, 1]), monkeypatch)
        assert prompt_mask.tolist() == [[True, False, True]]

    def test_omitted_mask_preserves_interior_pad(self, monkeypatch):
        # pad == eos: an interior pad-id token must remain unmasked when no mask is supplied
        control = self._steered_control()
        control.tokenizer.pad_token = control.tokenizer.eos_token
        control.tokenizer.pad_token_id = control.tokenizer.eos_token_id
        pad = control.tokenizer.pad_token_id
        ids = torch.tensor([[3, pad, 4, pad, 5]])  # interior pad at pos 1 and 3
        prompt_mask = self._built_prompt_mask(control, ids, None, monkeypatch)
        assert prompt_mask.tolist() == [[True, True, True, True, True]]

    def test_omitted_mask_masks_trailing_pad(self, monkeypatch):
        control = self._steered_control()
        control.tokenizer.pad_token = control.tokenizer.eos_token
        control.tokenizer.pad_token_id = control.tokenizer.eos_token_id
        pad = control.tokenizer.pad_token_id
        ids = torch.tensor([[3, 4, 5, pad, pad]])
        prompt_mask = self._built_prompt_mask(control, ids, None, monkeypatch)
        assert prompt_mask.tolist() == [[True, True, True, False, False]]


class TestComparatorVocabulary:
    """Comparators are `ge`/`le`; the retired aliases and names are rejected."""

    @pytest.mark.parametrize("stale", ["larger", "smaller", "score_above", "score_below", "bogus"])
    def test_castargs_rejects_non_canonical_comparators(self, stale):
        with pytest.raises(ValueError, match="'ge' or 'le'"):
            CASTArgs(
                behavior_vector=_steering_vector(1, [0]),
                condition_vector=_steering_vector(2, [1]),
                condition_layer_ids=[1],
                condition_vector_threshold=0.1,
                condition_comparator_threshold_is=stale,
                search=ConditionSearchSpec(auto_find=False),
            )

    def test_le_round_trips_to_below_threshold_gate(self):
        # a CAST configured with "le" fires when the runtime score is at or below threshold
        control = _build_cast(condition_threshold=1e9, comparator="le")
        pipeline, _, _ = _steer_pipeline(control)
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)
        decision = control.latest_decision
        assert decision is not None
        assert decision.comparator == "le"
        # every realistic cosine score is < 1e9, so the gate opens
        assert all(decision.open_per_row)


def _condition_pairs():
    return {
        "positives": ["yes indeed", "affirmative reply", "sure absolutely", "of course yes"],
        "negatives": ["no thanks", "negative reply", "never decline", "of course not"],
    }


class TestConditionalBehaviorTransform:
    """Conditional semantics with a non-additive behavior transform (directional ablation).

    Reuses the per-row gating harness: a probe run reads runtime condition scores to pick a
    separating threshold, then an ablation-configured CAST is gated at that threshold. Gate-open
    rows have the direction projected out; gate-closed rows are untouched.
    """

    DIRECTION_SEED = 500

    def _direction(self, layer_id):
        return _unit_vector(self.DIRECTION_SEED + layer_id)

    def _build_ablation_cast(self, condition_threshold, comparator="ge"):
        from aisteer360.algorithms.state_control.common.transforms import ProjectionTransform

        directions = {l: self._direction(l).unsqueeze(0) for l in (0, 1)}
        condition_vec = _steering_vector(seed=200, layers=[1])
        return CAST(
            behavior_transform=ProjectionTransform(directions, alpha=1.0),
            behavior_layer_ids=[0, 1],
            condition_vector=condition_vec,
            condition_layer_ids=[1],
            condition_vector_threshold=condition_threshold,
            condition_comparator_threshold_is=comparator,
        )

    def _separating_threshold(self, input_ids):
        probe = self._build_ablation_cast(condition_threshold=-1.0)
        pipeline, _, _ = _steer_pipeline(probe)
        pipeline.generate(input_ids=input_ids, max_new_tokens=1)
        layer_id = next(iter(probe._cond_config.layer_ids))
        row_scores = probe.latest_decision.scores_per_row[layer_id]
        lo, hi = sorted(row_scores)
        return lo, hi

    def test_gate_open_ablates_gate_closed_untouched(self):
        input_ids = torch.tensor([[3, 4, 5, 6], [11, 12, 13, 14]])
        lo, hi = self._separating_threshold(input_ids)
        if hi - lo < 1e-5:
            pytest.skip("tiny-model condition scores not separable for this seed")
        sep = (lo + hi) / 2

        control = self._build_ablation_cast(condition_threshold=sep, comparator="ge")
        pipeline, _, _ = _steer_pipeline(control)
        recorder = _RecordingTransform(control._transform)
        control._transform = recorder
        pipeline.generate(input_ids=input_ids, max_new_tokens=3)

        decision = control.latest_decision
        assert sum(decision.open_per_row) == 1  # exactly one row opens

        # the last recorded behavior mask is a decode pass reflecting per-row gating
        mask = recorder.masks[-1]
        steered_rows = [row for row in range(2) if bool(mask[row].any())]
        open_rows = [row for row, is_open in enumerate(decision.open_per_row) if is_open]
        assert steered_rows == open_rows  # only gate-open rows are ablated; closed rows untouched

    def test_unconditional_ablation_applies_to_all_rows(self):
        # no condition -> gate always open -> ablation applied everywhere it is masked
        from aisteer360.algorithms.state_control.common.transforms import ProjectionTransform

        directions = {l: self._direction(l).unsqueeze(0) for l in (0, 1)}
        control = CAST(
            behavior_transform=ProjectionTransform(directions, alpha=1.0),
            behavior_layer_ids=[0, 1],
        )
        pipeline, _, _ = _steer_pipeline(control)

        captured = {"pre": [], "post": []}

        class _Probe:
            def __init__(self, inner, direction):
                self._inner = inner
                self._unit = direction

            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                out = self._inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs)
                if layer_id == 0 and token_mask.any():
                    unit = self._unit.to(out.device, out.dtype)
                    captured["pre"].append(float((hidden_states[token_mask] @ unit).abs().max()))
                    captured["post"].append(float((out[token_mask] @ unit).abs().max()))
                return out

            @property
            def covered_layer_ids(self):
                return self._inner.covered_layer_ids

        control._transform = _Probe(control._transform, self._direction(0))
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5, 6]]), max_new_tokens=2)
        assert captured["post"], "transform never applied at a masked position"
        # unconditional gate is always open: the component is removed at every masked position
        for pre, post in zip(captured["pre"], captured["post"]):
            assert post < 0.02 * pre + 1e-6


class TestFitLocationDefaults:
    def test_condition_fit_defaults_to_layer_input(self):
        args = CASTArgs(behavior_vector=_steering_vector(1, [0]))
        assert args.condition_fit.location == "layer_input"

    def test_behavior_fit_defaults_to_layer_output(self):
        args = CASTArgs(behavior_vector=_steering_vector(1, [0]))
        assert args.behavior_fit.location == "layer_output"


class TestConditionPointProperty:
    def test_unconditional_returns_none(self):
        control = CAST(behavior_vector=_steering_vector(1, [0]), behavior_layer_ids=[0])
        _steer_pipeline(control)
        assert control.condition_point is None

    def test_conditional_returns_resolved_dict(self):
        control = _build_cast(condition_threshold=0.25, comparator="ge", comparison_mode="last")
        _steer_pipeline(control)
        point = control.condition_point
        assert point == {
            "layer_ids": [1],
            "threshold": 0.25,
            "comparator": "ge",
            "comparison_mode": "last",
        }


class TestConditionalAutoFindFitLocation:
    def _run(self, condition_fit):
        behavior_vec = _steering_vector(seed=100, layers=[0, 1])
        kwargs = dict(
            behavior_vector=behavior_vec,
            behavior_layer_ids=[0, 1],
            condition_data=_condition_pairs(),
            search=ConditionSearchSpec(auto_find=True),
        )
        if condition_fit is not None:
            kwargs["condition_fit"] = condition_fit  # else fall back to CASTArgs default
        control = CAST(**kwargs)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            pipeline, _, _ = _steer_pipeline(control)
            pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)
        mismatch = [w for w in record if "layer-input" in str(w.message)]
        return control, mismatch

    def test_default_condition_fit_no_fit_location_warning(self):
        # default CASTArgs condition_fit uses location="layer_input"
        control, mismatch = self._run(condition_fit=None)
        assert control._cond_config is not None
        assert not mismatch

    def test_user_condition_fit_without_location_warns(self):
        user_fit = VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="chat_prompt")
        control, mismatch = self._run(condition_fit=user_fit)
        assert control._cond_config is not None
        assert mismatch
