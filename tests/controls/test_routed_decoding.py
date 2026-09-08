"""RoutedDecoding tests: canned/prefix/pass-through execution, routing, overrides, validation.

Runs hub-free on a tiny randomly-initialized Llama with a WordLevel tokenizer. Hand-built probes
with large-magnitude biases force deterministic routing (a huge positive bias always opens, a
huge negative one never does); the mixed-batch test derives per-row separating biases from a
probe run. Empty probe `meta` keeps the fingerprint checks dormant except where a test arms them.
"""
import pytest
import torch

from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.fingerprint import model_fingerprint
from steerability.algorithms.core.internals.probes import Probe, ProbeFitSpec, ProbeSet, ProbeSetFit
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.utils.auxiliary_pass import current_auxiliary_pass
from steerability.algorithms.output_control.common.drivers.phased import Fixed
from steerability.algorithms.output_control.routed_decoding import (
    P,
    Route,
    RoutedDecoding,
    Router,
    generate,
    prefix,
    respond,
)
from steerability.algorithms.structural_control.base import StructuralControl
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _probe(layer_ids, bias=0.0, seed=200, meta=None):
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling="mean",
        layer_ids=list(layer_ids),
        weights={lid: _unit_vector(seed + lid) for lid in layer_ids},
        bias=bias,
        meta=meta or {},
    )


def _forced_probes():
    """'always' fires on every row; 'never' fires on none."""
    return ProbeSet({
        "always": _probe([1], bias=1e9, seed=200),
        "never": _probe([2], bias=-1e9, seed=300),
    })


def _make_pipeline(probes, rules, seed: int = 0):
    torch.manual_seed(seed)  # fixed so parity runs share the same model weights
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    router = RoutedDecoding(probes=probes, rules=rules)
    pipeline = SteeringPipeline(controls=[router], model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, router, model, tokenizer


def _text_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class _GenerateSpy:
    """Counts generate calls while delegating to the model's own generate."""

    def __init__(self, model):
        self._model = model
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self._model.generate(**kwargs)


class TestRespond:
    def test_canned_tokens_exact_and_no_decode_steps(self, monkeypatch):
        rules = Router(
            routes=[Route("canned", when=P("always"), action=respond("the cat sat"))],
            default_action=generate(),
        )
        pipeline, router, model, tokenizer = _make_pipeline(_forced_probes(), rules)
        spy = _GenerateSpy(model)
        script_session_generate(monkeypatch, spy)

        out = pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            runtime_kwargs={},
            max_new_tokens=4,
        )

        assert out[0].tolist() == _text_ids(tokenizer, "the cat sat")
        assert spy.calls == 0
        assert router.latest_routes == ["canned"]


class TestPrefix:
    def test_prefix_tokens_then_generated_tail(self, monkeypatch):
        rules = Router(
            routes=[Route("note", when=P("always"), action=prefix("the dog ran"))],
            default_action=generate(),
        )
        pipeline, router, model, tokenizer = _make_pipeline(_forced_probes(), rules)
        spy = _GenerateSpy(model)
        script_session_generate(monkeypatch, spy)

        out = pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            runtime_kwargs={},
            max_new_tokens=4,
        )

        prefix_ids = _text_ids(tokenizer, "the dog ran")
        assert out[0][: len(prefix_ids)].tolist() == prefix_ids
        assert out.size(1) > len(prefix_ids)  # generated tokens follow the prefix
        assert spy.calls == 1
        assert router.latest_routes == ["note"]


class TestDefaultParity:
    def test_default_route_matches_plain_pipeline(self):
        rules = Router(
            routes=[Route("unreached", when=P("never"), action=respond("the mat"))],
            default_action=generate(),
        )
        pipeline, router, model, tokenizer = _make_pipeline(_forced_probes(), rules)

        plain = SteeringPipeline(controls=[], model=model, tokenizer=tokenizer)
        plain.steer()

        prompt = torch.tensor([[3, 4, 5, 6]])
        routed_out = pipeline.generate(input_ids=prompt, max_new_tokens=6, do_sample=False)
        plain_out = plain.generate(input_ids=prompt, max_new_tokens=6, do_sample=False)

        assert router.latest_routes == ["default"]
        assert torch.equal(routed_out, plain_out)


class TestMixedBatch:
    PROMPTS = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 3, 5]])

    def _row_scores(self):
        """Per-row scores for the separating probe, read from a forced probe run."""
        probes = ProbeSet({"sep": _probe([1], bias=0.0, seed=400)})
        rules = Router(routes=[], default_action=generate())
        pipeline, router, _, _ = _make_pipeline(probes, rules)
        pipeline.generate(input_ids=self.PROMPTS, max_new_tokens=1)
        return router.probes.latest.scores["sep"].tolist()

    def test_three_routes_in_one_call(self):
        scores = self._row_scores()
        order = sorted(range(3), key=lambda i: scores[i])  # row indices, ascending score
        s = sorted(scores)
        if s[1] - s[0] < 1e-5 or s[2] - s[1] < 1e-5:
            pytest.skip("tiny-model probe scores not separable for this seed")
        thr_mid = (s[0] + s[1]) / 2  # opens for the top two rows
        thr_hi = (s[1] + s[2]) / 2  # opens only for the top row

        probes = ProbeSet({
            "hi": _probe([1], bias=-thr_hi, seed=400),
            "mid": _probe([1], bias=-thr_mid, seed=400),
        })
        rules = Router(
            routes=[
                Route("respond_route", when=P("hi"), action=respond("the mat")),
                Route("prefix_route", when=P("mid"), action=prefix("the dog")),
            ],
            default_action=generate(),
        )
        pipeline, router, _, tokenizer = _make_pipeline(probes, rules)
        out = pipeline.generate(input_ids=self.PROMPTS, max_new_tokens=3)

        expected = {order[2]: "respond_route", order[1]: "prefix_route", order[0]: "default"}
        assert router.latest_routes == [expected[i] for i in range(3)]

        # the respond row carries exactly the canned tokens (plus batch padding)
        canned_ids = _text_ids(tokenizer, "the mat")
        respond_row = out[order[2]].tolist()
        assert respond_row[: len(canned_ids)] == canned_ids
        assert all(t == tokenizer.pad_token_id for t in respond_row[len(canned_ids):])

        # the prefix row starts with the prefix tokens
        prefix_ids = _text_ids(tokenizer, "the dog")
        assert out[order[1]][: len(prefix_ids)].tolist() == prefix_ids


class TestCannedOverrides:
    def _rules(self):
        return Router(
            routes=[Route("canned", when=P("always"), action=respond("the cat sat"))],
            default_action=generate(),
        )

    def test_override_replaces_text(self):
        pipeline, _, _, tokenizer = _make_pipeline(_forced_probes(), self._rules())
        out = pipeline.generate(
            input_ids=torch.tensor([[3, 4, 5]]),
            runtime_kwargs={"canned_responses": {"canned": "the dog ran fast"}},
            max_new_tokens=4,
        )
        assert out[0].tolist() == _text_ids(tokenizer, "the dog ran fast")

    def test_unknown_override_key_warns_and_is_ignored(self):
        pipeline, _, _, tokenizer = _make_pipeline(_forced_probes(), self._rules())
        with pytest.warns(UserWarning, match="ghost"):
            out = pipeline.generate(
                input_ids=torch.tensor([[3, 4, 5]]),
                runtime_kwargs={"canned_responses": {"ghost": "the dog"}},
                max_new_tokens=4,
            )
        assert out[0].tolist() == _text_ids(tokenizer, "the cat sat")


class TestPaddedBatches:
    def test_mixed_length_batch_left_padding_tokenizer(self):
        rules = Router(
            routes=[Route("canned", when=P("always"), action=respond("the cat sat"))],
            default_action=generate(),
        )
        pipeline, router, model, tokenizer = _make_pipeline(_forced_probes(), rules)
        tokenizer.padding_side = "left"  # the driver's output padding must not depend on this

        ids = torch.tensor([[3, 4, 5, 2], [6, 7, 8, 9]])  # row 0 right-padded prompt
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        out = pipeline.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6)

        assert router.latest_routes == ["canned", "canned"]
        canned_ids = _text_ids(tokenizer, "the cat sat")
        for row in out:
            row = row.tolist()
            assert row[: len(canned_ids)] == canned_ids
            assert all(t == tokenizer.pad_token_id for t in row[len(canned_ids):])

    def test_mixed_routes_mixed_lengths_continuations_exact(self):
        rules = Router(
            routes=[Route("unreached", when=P("never"), action=respond("the mat"))],
            default_action=generate(),
        )
        pipeline, router, model, tokenizer = _make_pipeline(_forced_probes(), rules)

        single = pipeline.generate(input_ids=torch.tensor([[6, 7, 8, 9]]), max_new_tokens=6)

        ids = torch.tensor([[3, 4, 5, 2], [6, 7, 8, 9]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        batched = pipeline.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6)

        assert router.latest_routes == ["default", "default"]
        assert batched[1].tolist() == single[0].tolist()  # pads in a sibling row change nothing


class TestRawPhasePlans:
    def test_list_of_phases_accepted_as_action(self):
        rules = Router(
            routes=[
                Route("raw", when=P("always"),
                     action=[Fixed("the cat", add_special_tokens=False)]),
            ],
            default_action=generate(),
        )
        pipeline, _, _, tokenizer = _make_pipeline(_forced_probes(), rules)
        out = pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)
        assert out[0].tolist() == _text_ids(tokenizer, "the cat")

    def test_unloadable_action_raises(self):
        rules = Router(
            routes=[Route("bad", when=P("always"), action=123)],
            default_action=generate(),
        )
        pipeline, _, _, _ = _make_pipeline(_forced_probes(), rules)
        with pytest.raises(TypeError, match="Cannot lower action"):
            pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)

    def test_replacing_fixed_phase_rejected(self):
        rules = Router(
            routes=[
                Route("rewrite", when=P("always"),
                     action=[Fixed("the dog", replace=True, add_special_tokens=False)]),
            ],
            default_action=generate(),
        )
        pipeline, _, _, _ = _make_pipeline(_forced_probes(), rules)
        with pytest.raises(ValueError, match="replace=True"):
            pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)


class _SwapModelControl(StructuralControl):
    """Stub structural control that replaces the pipeline's model with a prepared one."""

    Args = None

    def __init__(self, replacement):
        super().__init__()
        self.replacement = replacement

    def steer(self, model, tokenizer=None, **kwargs):
        return self.replacement


class TestValidation:
    DATA = {
        "topic": ContrastivePairs(
            positives=["the cat sat on mat", "the cat ran"],
            negatives=["dog ran fast", "the dog ran"],
        ),
    }

    def test_bad_rule_name_fails_at_construction_probe_set(self):
        rules = Router(
            routes=[Route("r", when=P("ghost"), action=generate())],
            default_action=generate(),
        )
        with pytest.raises(ValueError, match="ghost"):
            RoutedDecoding(probes=_forced_probes(), rules=rules)

    def test_bad_rule_name_fails_at_construction_probe_set_fit(self):
        recipe = ProbeSetFit(data=self.DATA, spec=ProbeFitSpec(method="mean_diff"))
        rules = Router(
            routes=[Route("r", when=P("ghost"), action=generate())],
            default_action=generate(),
        )
        with pytest.raises(ValueError, match="ghost"):
            RoutedDecoding(probes=recipe, rules=rules)

    def test_deferred_recipe_fitted_at_steer(self):
        recipe = ProbeSetFit(
            data=self.DATA,
            spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1]),
        )
        rules = Router(
            routes=[Route("r", when=P("topic"), action=respond("the mat"))],
            default_action=generate(),
        )
        pipeline, router, model, _ = _make_pipeline(recipe, rules)
        assert isinstance(router.probes, ProbeSet)
        assert router.probes.names == ("topic",)
        assert router.probes.probes["topic"].meta["model_fingerprint"] == model_fingerprint(model)
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=1)
        assert router.latest_routes[0] in ("r", "default")

    def test_deferred_fit_runs_on_the_steered_model(self):
        torch.manual_seed(0)
        model_a = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        torch.manual_seed(1)
        model_b = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        tokenizer = wordlevel_tokenizer()

        recipe = ProbeSetFit(
            data=self.DATA,
            spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1]),
        )
        rules = Router(
            routes=[Route("r", when=P("topic"), action=respond("the mat"))],
            default_action=generate(),
        )
        router = RoutedDecoding(probes=recipe, rules=rules)
        pipeline = SteeringPipeline(
            controls=[_SwapModelControl(model_b), router]
        )
        pipeline.model = model_a
        pipeline.tokenizer = tokenizer
        pipeline.steer()

        fitted = router.probes.probes["topic"].meta["model_fingerprint"]
        assert fitted == model_fingerprint(model_b)
        assert fitted != model_fingerprint(model_a)

    def test_eager_set_passes_through_steer_unchanged(self):
        probes = _forced_probes()
        rules = Router(
            routes=[Route("r", when=P("always"), action=respond("the mat"))],
            default_action=generate(),
        )
        _, router, _, _ = _make_pipeline(probes, rules)
        assert router.probes is probes

    def test_fitted_set_from_other_model_raises_and_escape_works(self):
        torch.manual_seed(2)
        other = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        probes = ProbeSet({
            "always": _probe([1], bias=1e9, meta={"model_fingerprint": model_fingerprint(other)}),
        })
        rules = Router(
            routes=[Route("r", when=P("always"), action=respond("the mat"))],
            default_action=generate(),
        )
        with pytest.raises(ValueError, match="different model than this pipeline produced"):
            _make_pipeline(probes, rules)

        torch.manual_seed(0)
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
        tokenizer = wordlevel_tokenizer()
        router = RoutedDecoding(probes=probes, rules=rules, allow_model_mismatch=True)
        pipeline = SteeringPipeline(controls=[router], model=model, tokenizer=tokenizer)
        pipeline.steer()
        out = pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=2)
        assert out[0].tolist() == _text_ids(tokenizer, "the mat")

    def test_probe_pass_is_auxiliary(self):
        rules = Router(
            routes=[Route("note", when=P("always"), action=prefix("the dog"))],
            default_action=generate(),
        )
        pipeline, _, model, _ = _make_pipeline(_forced_probes(), rules)

        recorded = []
        original_forward = model.forward

        def recording_forward(*args, **kwargs):
            recorded.append(current_auxiliary_pass())
            return original_forward(*args, **kwargs)

        model.forward = recording_forward
        pipeline.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=3)

        assert len(recorded) >= 2  # the probe pass plus at least one decode forward
        assert recorded[0] is not None and recorded[0].aligned
        assert all(info is None for info in recorded[1:])

    def test_args_require_probes_instance(self):
        rules = Router(routes=[], default_action=generate())
        with pytest.raises(TypeError, match="ProbeSet"):
            RoutedDecoding(probes="not probes", rules=rules)

    def test_args_require_routing_rules(self):
        with pytest.raises(TypeError, match="Router"):
            RoutedDecoding(
                probes=_forced_probes(),
                rules=[Route("r", when=P("always"), action=generate())],
            )
