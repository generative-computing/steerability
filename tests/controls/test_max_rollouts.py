"""The declared per-query rollout bound `DecodingDriver.max_rollouts_per_query()`.

Construction-only, hub-free: the bound is a static declaration read from a driver's configuration
before any generation runs.
"""
from steerability.algorithms.output_control.base import DecodingDriver
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.output_control.budget_forcing.control import BudgetForcing
from steerability.algorithms.output_control.common.drivers.phased import PhasedDriver
from steerability.algorithms.output_control.deal.control import DeAL
from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding
from steerability.algorithms.output_control.search_decoding.control import SearchDecoding


def _zero_scorer(prompt, continuations, params):
    return [0.0] * len(continuations)


class TestSearchDrivers:

    def test_best_of_n_returns_n(self):
        assert BestOfN(n=8, scorer=_zero_scorer).max_rollouts_per_query() == 8

    def test_search_decoding_iterated_beam(self):
        driver = SearchDecoding(
            scorer=_zero_scorer, num_candidates=4, keep_k=2, max_iterations=3, propose_mode="sample",
        )
        # 4 * (1 + (3 - 1) * 2) = 4 * 5 = 20
        assert driver.max_rollouts_per_query() == 20

    def test_search_decoding_default_is_best_of_n(self):
        driver = SearchDecoding(scorer=_zero_scorer, num_candidates=6)
        assert driver.max_rollouts_per_query() == 6

    def test_deal_inherits_the_search_bound(self):
        driver = DeAL(
            reward_func=_zero_scorer, lookahead=8, init_beams=5, topk=2, max_iterations=4,
        )
        # 5 * (1 + (4 - 1) * 2) = 5 * 7 = 35
        assert driver.max_rollouts_per_query() == 35

    def test_bound_tracks_num_candidates(self):
        base = SearchDecoding(scorer=_zero_scorer, num_candidates=3, keep_k=2, max_iterations=2)
        doubled = SearchDecoding(scorer=_zero_scorer, num_candidates=6, keep_k=2, max_iterations=2)
        assert doubled.max_rollouts_per_query() == 2 * base.max_rollouts_per_query()


class TestPhasedDrivers:

    def test_phased_decoding_counts_generate_phases(self):
        driver = PhasedDecoding(plan=[
            {"generate": {"until": "</think>"}},
            {"fixed": "<answer>"},
            {"generate": {}},
        ])
        assert driver.max_rollouts_per_query() == 2

    def test_phased_decoding_single_generate(self):
        driver = PhasedDecoding(plan=[{"fixed": "Sure: "}, {"generate": {}}])
        assert driver.max_rollouts_per_query() == 1

    def test_budget_forcing_num_extensions_plus_two(self):
        assert BudgetForcing(num_extensions=2).max_rollouts_per_query() == 4
        assert BudgetForcing(num_extensions=0).max_rollouts_per_query() == 2


class TestDefaults:

    def test_decoding_driver_subclass_without_override_returns_none(self):
        class _Custom(DecodingDriver):
            def decode(self, *args, **kwargs):
                raise NotImplementedError

        assert _Custom().max_rollouts_per_query() is None

    def test_phased_driver_subclass_with_per_example_plan_returns_none(self):
        class _PerExample(PhasedDriver):
            def plan(self, prompt_text, params):
                return []

        assert _PerExample().max_rollouts_per_query() is None
