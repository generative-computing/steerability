"""Pure-logic tests for routing predicates and routes (no model required)."""
import pytest
import torch

from steerability.algorithms.output_control.routed_decoding.routing import P, Predicate, Route, Router


def _bools(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


class TestPredicateTruthTables:
    DECISIONS = {
        "a": _bools(True, True, False, False),
        "b": _bools(True, False, True, False),
    }

    def test_leaf(self):
        assert P("a").evaluate(self.DECISIONS).tolist() == [True, True, False, False]

    def test_and(self):
        assert (P("a") & P("b")).evaluate(self.DECISIONS).tolist() == [True, False, False, False]

    def test_or(self):
        assert (P("a") | P("b")).evaluate(self.DECISIONS).tolist() == [True, True, True, False]

    def test_not(self):
        assert (~P("a")).evaluate(self.DECISIONS).tolist() == [False, False, True, True]

    def test_nesting(self):
        pred = (P("a") & ~P("b")) | (~P("a") & P("b"))  # xor
        assert pred.evaluate(self.DECISIONS).tolist() == [False, True, True, False]

    def test_decision_names(self):
        pred = (P("a") & ~P("b")) | P("c")
        assert pred.decision_names() == {"a", "b", "c"}

    def test_repr_infix(self):
        assert repr(P("legal") & ~P("advice")) == "(legal & ~advice)"

    def test_operators_reject_non_predicates(self):
        with pytest.raises(TypeError):
            P("a") & "b"

    def test_result_is_predicate(self):
        assert isinstance(~(P("a") | P("b")), Predicate)


class TestDecisionValidation:
    def test_unknown_decision_name_raises_keyerror_naming_available(self):
        with pytest.raises(KeyError, match=r"Unknown decision name 'missing'.*'a'.*'b'"):
            P("missing").evaluate({"a": _bools(True), "b": _bools(False)})

    def test_scalar_bool_accepted_single_row(self):
        assert (P("a") & P("b")).evaluate({"a": True, "b": True}).tolist() == [True]

    def test_scalar_bool_rejected_multi_row(self):
        with pytest.raises(ValueError, match="bare bool"):
            P("a").evaluate({"a": True, "b": _bools(True, False)})

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same logical batch"):
            P("a").evaluate({"a": _bools(True, False), "b": _bools(True, False, True)})

    def test_non_bool_dtype_raises(self):
        with pytest.raises(ValueError, match="bool"):
            P("a").evaluate({"a": torch.tensor([1, 0])})

    def test_2d_tensor_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            P("a").evaluate({"a": torch.ones(2, 2, dtype=torch.bool)})

    def test_empty_decisions_raise(self):
        with pytest.raises(ValueError, match="empty"):
            P("a").evaluate({})


class TestRouter:
    def _router(self):
        return Router(
            routes=[
                Route("both", when=P("a") & P("b"), action="both_action"),
                Route("just_a", when=P("a"), action="a_action"),
                Route("just_b", when=P("b"), action="b_action"),
            ],
            default_action="default_action",
        )

    def test_first_match_wins(self):
        routes = self._router().route({"a": _bools(True), "b": _bools(True)})
        assert routes[0].name == "both"  # not "just_a", despite also matching

    def test_default_on_no_match(self):
        routes = self._router().route({"a": _bools(False), "b": _bools(False)})
        assert routes == [None]

    def test_per_row_independence_mixed_batch(self):
        decisions = {
            "a": _bools(True, True, False, False),
            "b": _bools(True, False, True, False),
        }
        routes = self._router().route(decisions)
        assert [r.name if r else None for r in routes] == ["both", "just_a", "just_b", None]

    def test_route_length_matches_rows(self):
        routes = self._router().route({"a": _bools(True, False), "b": _bools(False, False)})
        assert len(routes) == 2

    def test_decision_names_union(self):
        assert self._router().decision_names() == {"a", "b"}

    def test_validate_names_passes(self):
        self._router().validate_names({"a", "b", "c"})

    def test_validate_names_raises_naming_missing(self):
        with pytest.raises(ValueError, match=r"\['b'\]"):
            self._router().validate_names({"a"})

    def test_duplicate_route_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate route name 'dup'"):
            Router(routes=[
                Route("dup", when=P("a"), action=None),
                Route("dup", when=P("b"), action=None),
            ])

    def test_non_route_entry_raises(self):
        with pytest.raises(TypeError, match="Route instances"):
            Router(routes=["not a route"])

    def test_route_requires_predicate(self):
        with pytest.raises(TypeError, match="Predicate"):
            Route("bad", when="a", action=None)

    def test_route_requires_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            Route("", when=P("a"), action=None)

    def test_empty_routes_fall_to_default(self):
        router = Router(routes=[], default_action="fallback")
        assert router.route({"a": _bools(True, False)}) == [None, None]

    def test_unknown_decision_in_route_raises_at_route(self):
        router = Router(routes=[Route("r", when=P("ghost"), action=None)])
        with pytest.raises(KeyError, match="ghost"):
            router.route({"a": _bools(True)})


class TestDescribe:
    def test_contains_routes_in_order_and_default(self):
        text = self._router_text()
        first = text.index("legal_deferral")
        second = text.index("medical_note")
        assert first < second
        assert "default" in text
        assert text.splitlines()[0] == "Router"

    def test_arrow_alignment(self):
        lines = self._router_text().splitlines()[1:]
        arrow_columns = {line.index("->") for line in lines}
        assert len(arrow_columns) == 1

    def test_action_label_fallback_is_type_name(self):
        router = Router(routes=[Route("raw", when=P("a"), action=["some", "payload"])])
        text = router.describe()
        assert "-> list" in text
        assert "['some', 'payload']" not in text

    def test_describe_snapshot(self):
        text = self._router_text()
        assert text.splitlines() == [
            "Router",
            "├─ 1. legal_deferral   if (legal & advice)     -> respond",
            "├─ 2. medical_note     if (medical & advice)   -> prefix",
            "└─ default                                     -> generate",
        ]

    @staticmethod
    def _router_text() -> str:
        router = Router(
            routes=[
                Route("legal_deferral", when=P("legal") & P("advice"), action="respond"),
                Route("medical_note", when=P("medical") & P("advice"), action="prefix"),
            ],
            default_action="generate",
        )
        return router.describe()
