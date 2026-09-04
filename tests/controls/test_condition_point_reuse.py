"""Reuse of a searched CAST condition point as a single object.

Covers `ConditionPoint.comparison_mode` / `flipped()`, `CASTArgs.condition_point` expansion (object
and dict shapes, comparator validation, conflict errors), and the precedence of a supplied point over
`search.auto_find`. Hub-free on a tiny Llama.
"""
import pytest
import torch

from aisteer360.algorithms.core.internals.data import ContrastivePairs
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.cast.control import CAST
from aisteer360.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from aisteer360.algorithms.state_control.common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control.common.selectors.condition_point import ConditionPoint
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4


def _unit_vector(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(HIDDEN, generator=g)
    return v / v.norm()


def _steering_vector(seed: int, layers) -> SteeringVector:
    return SteeringVector(
        model_type="llama",
        directions={l: _unit_vector(seed + l).unsqueeze(0) for l in layers},
        explained_variances={l: 0.5 for l in layers},
    )


def _steer(control, seed: int = 0):
    torch.manual_seed(seed)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestConditionPointObject:
    def test_flipped_inverts_only_comparator(self):
        cp = ConditionPoint(layer_id=2, threshold=0.3, comparator="ge", f1=0.8, margin=0.05,
                            comparison_mode="last")
        flipped = cp.flipped()
        assert flipped.comparator == "le"
        assert flipped.layer_id == 2
        assert flipped.threshold == 0.3
        assert flipped.f1 == 0.8 and flipped.margin == 0.05  # search stats carried over unchanged
        assert flipped.comparison_mode == "last"
        assert flipped.flipped().comparator == "ge"  # round trip

    def test_selector_populates_comparison_mode(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        tokenizer = wordlevel_tokenizer()
        data = ContrastivePairs(
            positives=["the cat sat", "the dog ran"],
            negatives=["mat on fast", "span attention"],
        )
        condition_vec = _steering_vector(200, layers=list(range(LAYERS)))
        result = ConditionPointSelector().select(
            model=model,
            tokenizer=tokenizer,
            condition_directions=condition_vec.directions,
            data=data,
            fit_spec=VectorTrainSpec(method="pca_center", accumulate="all",
                                     prompt_format="chat_prompt", location="layer_input"),
            search_spec=ConditionSearchSpec(auto_find=True),
            comparison_mode="last",
        )
        assert result.comparison_mode == "last"


class TestConditionPointExpansion:
    def _cast(self, condition_point):
        return CAST(
            behavior_vector=_steering_vector(100, [0, 1]),
            behavior_layer_ids=[0, 1],
            condition_vector=_steering_vector(200, [1]),
            condition_point=condition_point,
        )

    def test_object_matches_manual_triple(self):
        cp = ConditionPoint(layer_id=1, threshold=0.25, comparator="ge", f1=0.9,
                            comparison_mode="mean")
        from_point, _, _ = _steer(self._cast(cp))
        manual = CAST(
            behavior_vector=_steering_vector(100, [0, 1]),
            behavior_layer_ids=[0, 1],
            condition_vector=_steering_vector(200, [1]),
            condition_layer_ids=[1],
            condition_vector_threshold=0.25,
            condition_comparator_threshold_is="ge",
            condition_threshold_comparison_mode="mean",
            search=ConditionSearchSpec(auto_find=False),
        )
        from_manual, _, _ = _steer(manual)
        cfg_p = from_point.controls[0]._cond_config
        cfg_m = from_manual.controls[0]._cond_config
        assert cfg_p.layer_ids == cfg_m.layer_ids
        assert cfg_p.threshold == cfg_m.threshold
        assert cfg_p.comparator == cfg_m.comparator
        assert cfg_p.comparison_mode == cfg_m.comparison_mode
        assert cfg_p.enabled == cfg_m.enabled is True

    def test_dict_roundtrip_reproduces_gate_decisions(self):
        """CAST(condition_point=other.condition_point) reproduces identical gate decisions."""
        search_control = CAST(
            behavior_vector=_steering_vector(100, [0, 1]),
            behavior_layer_ids=[0, 1],
            condition_vector=_steering_vector(200, list(range(LAYERS))),
            condition_data=ContrastivePairs(
                positives=["the cat sat", "the dog ran"],
                negatives=["mat on fast", "span attention"],
            ),
            search=ConditionSearchSpec(auto_find=True),
        )
        pipe_a, model, tokenizer = _steer(search_control)
        point = dict(search_control.condition_point)  # dict from the property

        reuse_control = CAST(
            behavior_vector=_steering_vector(100, [0, 1]),
            behavior_layer_ids=[0, 1],
            condition_vector=_steering_vector(200, list(range(LAYERS))),
            condition_point=point,
        )
        # steer the reuse control on the SAME model/tokenizer so scores are comparable
        pipe_b = SteeringPipeline(controls=[reuse_control], model=model, tokenizer=tokenizer)
        pipe_b.steer()

        cfg_a = search_control._cond_config
        cfg_b = reuse_control._cond_config
        assert cfg_a.layer_ids == cfg_b.layer_ids
        assert cfg_a.threshold == cfg_b.threshold
        assert cfg_a.comparator == cfg_b.comparator
        assert cfg_a.comparison_mode == cfg_b.comparison_mode

        # gate decisions match on the same prompts (plain-text input; no chat template needed here)
        for prompt in ("the cat sat", "span attention fast"):
            pipe_a.generate(prompt, max_new_tokens=2, do_sample=False)
            pipe_b.generate(prompt, max_new_tokens=2, do_sample=False)
            assert search_control.latest_decision.open_per_row == reuse_control.latest_decision.open_per_row

    @pytest.mark.parametrize("stale", ["score_below", "larger", "smaller"])
    def test_non_canonical_comparator_raises(self, stale):
        point = {"layer_ids": [1], "threshold": 0.3, "comparator": stale}
        with pytest.raises(ValueError, match="'ge' or 'le'"):
            self._cast(point)

    def test_conflict_with_condition_layer_ids_raises(self):
        cp = ConditionPoint(layer_id=1, threshold=0.25, comparator="ge", f1=0.9)
        with pytest.raises(ValueError, match="drop"):
            CAST(
                behavior_vector=_steering_vector(100, [0, 1]),
                condition_vector=_steering_vector(200, [1]),
                condition_point=cp,
                condition_layer_ids=[1],
            )

    def test_conflict_with_threshold_raises(self):
        cp = ConditionPoint(layer_id=1, threshold=0.25, comparator="ge", f1=0.9)
        with pytest.raises(ValueError, match="drop"):
            CAST(
                behavior_vector=_steering_vector(100, [0, 1]),
                condition_vector=_steering_vector(200, [1]),
                condition_point=cp,
                condition_vector_threshold=0.5,
            )

    def test_missing_dict_key_raises(self):
        with pytest.raises(ValueError, match="must have keys"):
            self._cast({"layer_ids": [1], "threshold": 0.3})  # no comparator

    def test_empty_layer_ids_raises(self):
        """An empty layer list must raise rather than silently degrade to unconditional steering."""
        with pytest.raises(ValueError, match="no condition layers"):
            self._cast({"layer_ids": [], "threshold": 0.3, "comparator": "ge"})

    def test_bad_comparison_mode_raises_at_construction(self):
        """A typo'd comparison_mode is rejected at construction, not deferred to generation."""
        with pytest.raises(ValueError, match="comparison_mode must be"):
            self._cast({"layer_ids": [1], "threshold": 0.3, "comparator": "ge",
                        "comparison_mode": "LAST_TYPO"})


class TestConditionPointSupersedesSearch:
    def test_selector_not_invoked(self, monkeypatch):
        """With condition_point provided and search.auto_find at its default, the selector is never
        invoked."""
        def _boom(*args, **kwargs):
            raise AssertionError("ConditionPointSelector.select must not run when condition_point is given.")

        monkeypatch.setattr(ConditionPointSelector, "select", _boom)

        cp = ConditionPoint(layer_id=1, threshold=0.2, comparator="ge", f1=0.9,
                            comparison_mode="mean")
        control = CAST(
            behavior_vector=_steering_vector(100, [2, 3]),
            behavior_layer_ids=[2, 3],
            condition_vector=_steering_vector(200, list(range(LAYERS))),
            condition_data=ContrastivePairs(positives=["the cat"], negatives=["mat fast"]),
            condition_point=cp,  # supersedes the auto-search
        )
        _steer(control)  # must not raise
        assert control.condition_point["layer_ids"] == [1]
        assert control.condition_point["threshold"] == 0.2
