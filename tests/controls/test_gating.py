"""Gating component tests: readout math, rule decisions, the `Gate` lifecycle, and probe
equivalence.

Runs hub-free on deterministic tensors. The sign-erasure tests build their cluster geometry
analytically (an on-axis component plus a unit vector orthogonalized against the direction), so
the characterization of the projected score as `|cos(h, d)|` and its polarity inversion hold by
construction rather than by model behavior.
"""
import pytest
import torch
import torch.nn.functional as F

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden, masked_mean
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.state_control.common.gating import (
    AffineReadout,
    CallableReadout,
    CosineReadout,
    Evidence,
    Gate,
    PerKeyThreshold,
    ProjectedCosineReadout,
    SumThreshold,
    gate_from_probe,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)

HIDDEN = 32


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _orthonormal_rows(direction: torch.Tensor, num_rows: int, seed: int) -> torch.Tensor:
    """Unit rows orthogonal to `direction`, via Gram-Schmidt on seeded noise."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(num_rows, direction.numel(), generator=g)
    projected = noise - (noise @ direction).unsqueeze(-1) * direction
    return projected / projected.norm(dim=-1, keepdim=True)


def _probe(layer_ids=(1, 2), bias=0.0, seed=7, pooling="mean", meta=None) -> Probe:
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling=pooling,
        layer_ids=list(layer_ids),
        weights={lid: _unit_vector(seed + lid) for lid in layer_ids},
        bias=bias,
        meta=meta or {},
    )


class TestSignErasure:
    """The unsigned projected score erases direction sign; the signed cosine keeps it.

    Clusters mimic a mean-difference domain gate at the failing layer: positives moderately
    aligned with the direction (`0.3 * d`), negatives strongly anti-aligned (`-0.6 * d`), and
    unrelated content nearly orthogonal (`0.02 * d`), each plus a unit orthogonal component.
    The unrelated cluster keeps a small on-axis component on purpose: at exact orthogonality
    (`d @ h == 0`) the projected score is a 0/0 guarded only by the production epsilon and
    returns amplified float noise rather than 0, so the `|cos|` characterization below holds
    only away from exact orthogonality.
    """

    def setup_method(self):
        self.direction = _unit_vector(seed=7)
        self.positives = 0.3 * self.direction + _orthonormal_rows(self.direction, 4, seed=11)
        self.negatives = -0.6 * self.direction + _orthonormal_rows(self.direction, 4, seed=22)
        self.unrelated = 0.02 * self.direction + _orthonormal_rows(self.direction, 4, seed=33)
        self.projector = rank_one_projector(self.direction)

    def _projected(self, rows: torch.Tensor) -> torch.Tensor:
        return projected_cosine_similarity_tensor(rows, self.projector)

    def _signed(self, rows: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(rows, self.direction.unsqueeze(0), dim=-1)

    def test_projected_score_is_absolute_cosine(self):
        # tanh distortion is tiny at these magnitudes, so the projected score matches |cos|
        for rows in (self.positives, self.negatives, self.unrelated):
            assert torch.allclose(self._projected(rows), self._signed(rows).abs(), atol=0.005)

    def test_unsigned_score_inverts_polarity_and_opens_on_unrelated(self):
        proj_pos = self._projected(self.positives)
        proj_neg = self._projected(self.negatives)
        proj_unrel = self._projected(self.unrelated)
        # anti-aligned negatives outscore positives, so only "le" separates the classes
        assert proj_pos.max() < proj_neg.min()
        # every unrelated point lies below any separating threshold, i.e. opens the gate
        assert proj_unrel.max() < proj_pos.max() < proj_neg.min()

    def test_signed_score_fails_closed_on_unrelated(self):
        signed_pos = self._signed(self.positives)
        signed_neg = self._signed(self.negatives)
        signed_unrel = self._signed(self.unrelated)
        assert signed_pos.min() > 0 > signed_neg.max()
        assert signed_unrel.abs().max() < 0.05
        assert signed_pos.min() > signed_unrel.max()
        assert signed_pos.min() > signed_neg.max()


class TestReadouts:
    def setup_method(self):
        self.direction = _unit_vector(seed=5)
        g = torch.Generator().manual_seed(41)
        self.pooled = torch.randn(3, HIDDEN, generator=g)

    def test_affine_matches_hand_dot_product(self):
        weights = _unit_vector(seed=9)
        readout = AffineReadout({1: weights})
        values = readout(self.pooled, 1)
        assert torch.allclose(values, self.pooled @ weights, atol=1e-6)

    def test_affine_missing_layer_returns_zeros(self):
        readout = AffineReadout({1: _unit_vector(seed=9)})
        assert torch.equal(readout(self.pooled, 3), torch.zeros(3))

    def test_cosine_matches_hand_value(self):
        readout = CosineReadout({1: self.direction})
        values = readout(self.pooled, 1)
        expected = F.cosine_similarity(self.pooled, self.direction.unsqueeze(0), dim=-1)
        assert torch.allclose(values, expected, atol=1e-6)

    def test_cosine_antiparallel_row_scores_negative_one(self):
        readout = CosineReadout({1: self.direction})
        values = readout((-self.direction).unsqueeze(0), 1)
        assert torch.allclose(values, torch.tensor([-1.0]), atol=1e-5)

    def test_cosine_missing_layer_returns_zeros(self):
        readout = CosineReadout({1: self.direction})
        assert torch.equal(readout(self.pooled, 3), torch.zeros(3))

    def test_projected_cosine_matches_reference_function(self):
        readout = ProjectedCosineReadout({1: self.direction})
        values = readout(self.pooled, 1)
        expected = projected_cosine_similarity_tensor(self.pooled, rank_one_projector(self.direction))
        assert torch.allclose(values, expected, atol=1e-6)

    def test_projected_cosine_missing_layer_returns_zeros(self):
        readout = ProjectedCosineReadout({1: self.direction})
        assert torch.equal(readout(self.pooled, 3), torch.zeros(3))

    @pytest.mark.parametrize("readout_cls", [CosineReadout, ProjectedCosineReadout])
    def test_multi_row_artifact_uses_row_zero(self, readout_cls):
        stacked = torch.stack([self.direction, _unit_vector(seed=13)])  # [K, H], K=2
        from_stacked = readout_cls({1: stacked})(self.pooled, 1)
        from_row_zero = readout_cls({1: self.direction})(self.pooled, 1)
        assert torch.allclose(from_stacked, from_row_zero, atol=1e-6)

    def test_affine_multi_row_artifact_uses_row_zero(self):
        weights = _unit_vector(seed=9)
        stacked = torch.stack([weights, _unit_vector(seed=13)])
        assert torch.allclose(
            AffineReadout({1: stacked})(self.pooled, 1),
            AffineReadout({1: weights})(self.pooled, 1),
            atol=1e-6,
        )

    def test_callable_readout_wraps_fn_and_never_lowers(self):
        readout = CallableReadout(lambda pooled, layer_id: pooled.mean(dim=-1))
        assert torch.allclose(readout(self.pooled, 1), self.pooled.mean(dim=-1))
        assert type(readout).wire_kind is None
        assert readout.export((1,)) is None

    def test_junk_artifact_raises(self):
        from aisteer360.algorithms.state_control.common.sources import ContrastiveFit

        with pytest.raises(TypeError, match="concrete SteeringVector or Mapping"):
            CosineReadout(ContrastiveFit(data={"positives": ["a"], "negatives": ["b"]}))

    def test_export_stacks_rows_aligned_with_layer_order(self):
        weights = {1: _unit_vector(seed=1), 2: _unit_vector(seed=2)}
        form = AffineReadout(weights).export((2, 1))
        assert form.kind == "affine"
        assert torch.equal(form.tensors["weights"][0], weights[2])
        assert torch.equal(form.tensors["weights"][1], weights[1])

    def test_export_missing_layer_returns_none(self):
        assert AffineReadout({1: _unit_vector(seed=1)}).export((1, 3)) is None


class TestRules:
    def test_sum_threshold_ties_open(self):
        rule = SumThreshold(bias=-1.0)
        values = {1: torch.tensor([0.5, 0.2]), 2: torch.tensor([0.5, 0.2])}
        assert rule.decide(values, 2).tolist() == [True, False]  # 1.0 - 1.0 == 0 opens

    def test_sum_threshold_empty_values_all_closed(self):
        assert SumThreshold().decide({}, 3).tolist() == [False, False, False]

    @pytest.mark.parametrize("comparator,expected", [("ge", [True, False]), ("le", [False, True])])
    def test_per_key_comparators(self, comparator, expected):
        rule = PerKeyThreshold(threshold=0.5, comparator=comparator)
        assert rule.decide({1: torch.tensor([0.9, 0.1])}, 2).tolist() == expected

    def test_per_key_ge_ties_open(self):
        rule = PerKeyThreshold(threshold=0.5, comparator="ge")
        assert rule.decide({1: torch.tensor([0.5])}, 1).tolist() == [True]

    def test_per_key_any_vs_all(self):
        values = {1: torch.tensor([0.9, 0.9]), 2: torch.tensor([0.9, 0.1])}
        assert PerKeyThreshold(0.5, "ge", aggregate="any").decide(values, 2).tolist() == [True, True]
        assert PerKeyThreshold(0.5, "ge", aggregate="all").decide(values, 2).tolist() == [True, False]

    def test_per_key_rejects_unknown_comparator(self):
        with pytest.raises(ValueError, match="'ge' or 'le'"):
            PerKeyThreshold(0.5, comparator="larger")

    def test_is_complete_requires_every_expected_layer(self):
        rule = SumThreshold()
        assert not rule.is_complete(frozenset({1}), frozenset({1, 2}))
        assert rule.is_complete(frozenset({1, 2}), frozenset({1, 2}))

    def test_exports_inline_params(self):
        assert SumThreshold(bias=-3.2).export().params == {"bias": -3.2}
        form = PerKeyThreshold(0.4, "le", aggregate="all").export()
        assert form.params == {"threshold": 0.4, "comparator": "le", "aggregate": "all"}


class TestGateLifecycle:
    def _gate(self, layer_ids=(1, 2), bias=-1.0):
        weights = {lid: _unit_vector(seed=7 + lid) for lid in layer_ids}
        return Gate(Evidence(layer_ids, AffineReadout(weights)), SumThreshold(bias=bias))

    def test_all_closed_before_any_evidence(self):
        gate = self._gate()
        gate.reset(3)
        assert gate.open_rows().tolist() == [False, False, False]
        assert not gate.is_ready()

    def test_freezes_when_every_layer_reports(self):
        gate = self._gate(bias=-1.0)
        gate.reset(2)
        gate.update(torch.tensor([0.6, 0.1]), key=1)
        assert not gate.is_ready()
        gate.update(torch.tensor([0.5, 0.2]), key=2)
        assert gate.is_ready()
        # row 0: 1.1 - 1.0 >= 0 opens; row 1: 0.3 - 1.0 stays closed
        assert gate.open_rows().tolist() == [True, False]

    def test_updates_after_freeze_are_ignored(self):
        gate = self._gate(layer_ids=(1,), bias=0.0)
        gate.reset(1)
        gate.update(torch.tensor([1.0]), key=1)
        assert gate.is_ready() and gate.is_open()
        gate.update(torch.tensor([-100.0]), key=1)
        assert gate.open_rows().tolist() == [True]

    def test_ties_open(self):
        gate = self._gate(layer_ids=(1,), bias=-1.0)
        gate.reset(1)
        gate.update(torch.tensor([1.0]), key=1)
        assert gate.open_rows().tolist() == [True]

    def test_reset_clears_evidence_and_decision(self):
        gate = self._gate()
        gate.reset(1)
        gate.update(torch.tensor([5.0]), key=1)
        gate.update(torch.tensor([5.0]), key=2)
        assert gate.is_ready()
        gate.reset(2)
        assert not gate.is_ready()
        assert gate.open_rows().tolist() == [False, False]
        assert gate.evidence_values() == {}

    def test_reset_is_idempotent(self):
        single = self._gate()
        single.reset(3)
        double = self._gate()
        double.reset(3)
        double.reset(3)
        values = torch.tensor([0.3, 0.05, 0.2])
        single.update(values, key=1)
        double.update(values, key=1)
        assert torch.equal(single.open_rows(), double.open_rows())
        assert single.is_ready() == double.is_ready()
        assert single.num_rows == double.num_rows == 3

    def test_scalar_value_allowed_single_row_only(self):
        gate = self._gate(layer_ids=(1,), bias=0.0)
        gate.reset(1)
        gate.update(0.5, key=1)
        assert gate.open_rows().tolist() == [True]
        gate.reset(2)
        with pytest.raises(ValueError, match="scalar"):
            gate.update(0.5, key=1)

    def test_wrong_row_count_raises(self):
        gate = self._gate(layer_ids=(1,))
        gate.reset(2)
        with pytest.raises(ValueError, match="2 rows but received 3"):
            gate.update(torch.tensor([0.1, 0.2, 0.3]), key=1)

    def test_per_row_independence(self):
        gate = Gate(
            Evidence((1,), AffineReadout({1: _unit_vector(seed=8)})),
            PerKeyThreshold(threshold=0.5, comparator="ge"),
        )
        gate.reset(3)
        gate.update(torch.tensor([0.9, 0.1, 0.5]), key=1)
        assert gate.open_rows().tolist() == [True, False, True]

    def test_evidence_values_snapshot_survives_freeze(self):
        gate = self._gate()
        gate.reset(1)
        gate.update(torch.tensor([0.6]), key=1)
        gate.update(torch.tensor([0.5]), key=2)
        snapshot = gate.evidence_values()
        assert set(snapshot) == {1, 2}
        assert snapshot[1].tolist() == [pytest.approx(0.6)]

    def test_partial_evidence_decides_live_until_complete(self):
        gate = Gate(
            Evidence((1, 2), AffineReadout({lid: _unit_vector(seed=lid) for lid in (1, 2)})),
            PerKeyThreshold(threshold=0.5, comparator="ge", aggregate="any"),
        )
        gate.reset(1)
        gate.update(torch.tensor([0.9]), key=1)
        assert not gate.is_ready()
        assert gate.open_rows().tolist() == [True]  # live partial decision under "any"

    def test_wire_kinds_pairs_readout_and_rule(self):
        gate = self._gate()
        readouts, rules = gate.wire_kinds()
        assert readouts == frozenset({"affine"})
        assert rules == frozenset({"sum_threshold"})

    def test_callable_readout_has_no_wire_kinds(self):
        gate = Gate(
            Evidence((1,), CallableReadout(lambda pooled, lid: pooled.mean(-1))),
            SumThreshold(),
        )
        assert gate.wire_kinds() is None

    def test_shared_instance_follower_reads_driver_decision(self):
        # the driver updates the shared instance; a follower only reads open_rows()
        gate = self._gate(layer_ids=(1,), bias=0.0)
        gate.reset(2)
        gate.reset(2)  # follower hook build re-resets the shared instance harmlessly
        gate.update(torch.tensor([1.0, -1.0]), key=1)
        assert gate.open_rows().tolist() == [True, False]  # both interventions read this

    def test_evidence_requires_layers_and_valid_pooling(self):
        readout = AffineReadout({1: _unit_vector(seed=1)})
        with pytest.raises(ValueError, match="at least one condition layer"):
            Evidence((), readout)
        with pytest.raises(ValueError, match="pooling"):
            Evidence((1,), readout, pooling="max")


class TestGateFromProbe:
    def test_equivalent_to_hand_built_gate(self):
        probe = _probe(layer_ids=(1, 2), bias=-0.25)
        assembled = gate_from_probe(probe)
        hand_built = Gate(
            Evidence(tuple(probe.layer_ids), AffineReadout(dict(probe.weights)), pooling=probe.pooling),
            SumThreshold(bias=probe.bias),
        )
        g = torch.Generator().manual_seed(51)
        pooled = {lid: torch.randn(3, HIDDEN, generator=g) for lid in (1, 2)}
        for gate in (assembled, hand_built):
            gate.reset(3)
            for lid in (1, 2):
                gate.update(gate.evidence.readout(pooled[lid], lid), key=lid)
        assert torch.equal(assembled.open_rows(), hand_built.open_rows())
        assert assembled.evidence.pooling == probe.pooling
        assert isinstance(assembled.rule, SumThreshold)
        assert assembled.rule.bias == probe.bias

    def test_carries_probe_validation_metadata(self):
        probe = _probe(meta={"model_fingerprint": "abcd"})
        gate = gate_from_probe(probe)
        assert gate.evidence.readout.location == "layer_input"
        assert gate.evidence.readout.model_fingerprint == "abcd"

    def test_allow_model_mismatch_disarms_fingerprint(self):
        probe = _probe(meta={"model_fingerprint": "abcd"})
        gate = gate_from_probe(probe, allow_model_mismatch=True)
        assert gate.evidence.readout.model_fingerprint is None

    def test_matches_probe_predict_over_identical_hidden_states(self):
        """The probe's pooled decision and the gating path agree per row, ties included."""
        g = torch.Generator().manual_seed(61)
        hidden = {lid: torch.randn(4, 6, HIDDEN, generator=g) for lid in (1, 2)}
        mask = torch.ones(4, 6)
        raw = _probe(layer_ids=(1, 2), bias=0.0)

        # bias placing row 0 exactly on the boundary (ties open) and splitting the rest
        scores = raw.score_hidden(hidden, prompt_mask=mask)
        probe = _probe(layer_ids=(1, 2), bias=-float(scores[0]))
        expected = probe.score_hidden(hidden, prompt_mask=mask) >= 0
        assert bool(expected[0])  # the boundary row opens (ties open)

        gate = gate_from_probe(probe)
        gate.reset(4)
        for lid in (1, 2):
            pooled = aggregate_condition_hidden(hidden[lid], probe.pooling, attention_mask=mask)
            gate.update(gate.evidence.readout(pooled, lid), key=lid)
        assert gate.is_ready()
        assert torch.equal(gate.open_rows(), expected)

    def test_divergent_rows_split(self):
        g = torch.Generator().manual_seed(71)
        hidden = {1: torch.randn(4, 5, HIDDEN, generator=g)}
        probe_raw = _probe(layer_ids=(1,), bias=0.0)
        scores = probe_raw.score_hidden(hidden)
        ordered = scores.sort().values
        midpoint = float((ordered[1] + ordered[2]) / 2)
        probe = _probe(layer_ids=(1,), bias=-midpoint)
        expected = probe.score_hidden(hidden) >= 0
        assert expected.any() and not expected.all()

        gate = gate_from_probe(probe)
        gate.reset(4)
        pooled = aggregate_condition_hidden(hidden[1], probe.pooling)
        gate.update(gate.evidence.readout(pooled, 1), key=1)
        assert torch.equal(gate.open_rows(), expected)


class TestPoolingModes:
    def test_last_pooling_selects_last_real_token(self):
        g = torch.Generator().manual_seed(81)
        hidden = torch.randn(2, 4, HIDDEN, generator=g)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()
        direction = _unit_vector(seed=5)
        pooled = aggregate_condition_hidden(hidden, "last", attention_mask=mask)
        values = CosineReadout({1: direction})(pooled, 1)
        last = torch.stack([hidden[0, 2], hidden[1, 3]])
        expected = F.cosine_similarity(last, direction.unsqueeze(0), dim=-1)
        assert torch.allclose(values, expected, atol=1e-6)

    def test_mean_pooling_matches_masked_mean(self):
        g = torch.Generator().manual_seed(82)
        hidden = torch.randn(2, 4, HIDDEN, generator=g)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()
        direction = _unit_vector(seed=5)
        pooled = aggregate_condition_hidden(hidden, "mean", attention_mask=mask)
        values = CosineReadout({1: direction})(pooled, 1)
        expected = F.cosine_similarity(masked_mean(hidden, mask), direction.unsqueeze(0), dim=-1)
        assert torch.allclose(values, expected, atol=1e-6)
