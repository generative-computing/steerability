"""Tests for the generic output controls (config-first steering: the output-category analogue of
`ActivationAdapter`).

Covers the shared spec resolver and the five generics (`ValueGuidance`, `ContrastiveGuidance`,
`SearchDecoding`, `PhasedDecoding`, `StoppingRules`), including equivalence with the named methods
they generalize (RAD, SASA, DeAL).

Hub-free: tiny classifier / aux LMs are built via config classes saved to `tmp_path`, on the shared
`tests/utils/tiny_models.py` fixtures.
"""
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.common.logit_sources import AuxModelSource, CallableSource
from steerability.algorithms.output_control.common.processors.contrastive_mixture import ContrastiveMixtureProcessor
from steerability.algorithms.output_control.common.processors.value_guided import ValueGuidedProcessor
from steerability.algorithms.output_control.common.resolve import resolve_scorer, resolve_source, resolve_value
from steerability.algorithms.output_control.common.scorers.majority_vote import MajorityVoteScorer
from steerability.algorithms.output_control.common.scorers.reward_model import RewardModelScorer
from steerability.algorithms.output_control.common.values.base import BaseCandidateValue, StepContext
from steerability.algorithms.output_control.common.values.callable import CallableValue
from steerability.algorithms.output_control.common.values.classifier import ClassifierValue
from steerability.algorithms.output_control.common.values.reward_model import RewardModelValue
from steerability.algorithms.output_control.common.values.subspace_margin import (
    SubspaceMarginValue,
    load_single_file_probe,
)
from steerability.algorithms.output_control.contrastive_guidance.control import ContrastiveGuidance
from steerability.algorithms.output_control.deal.control import DeAL
from steerability.algorithms.output_control.phased_decoding.control import PhasedDecoding
from steerability.algorithms.output_control.rad.control import RAD
from steerability.algorithms.output_control.sasa.control import SASA
from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
from steerability.algorithms.output_control.stopping_rules.control import StoppingRules
from steerability.algorithms.output_control.value_guidance.control import ValueGuidance
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _write_probe_json(path, hidden=16):
    """Write a `.probe` JSON checkpoint (direction and midpoint lists); returns the tensors."""
    direction = torch.randn(hidden)
    midpoint = torch.randn(hidden)
    with open(path, "w") as f:
        json.dump({"direction": direction.tolist(), "midpoint": midpoint.tolist()}, f)
    return direction, midpoint


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


def _make_tiny_classifier(tmp_path):
    """A hub-free tiny sequence classifier + wordlevel tokenizer saved to tmp_path."""
    cfg = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=VOCAB,
        num_labels=2, pad_token_id=2,
    )
    clf = LlamaForSequenceClassification(cfg).eval()
    wordlevel_tokenizer().save_pretrained(str(tmp_path))
    clf.save_pretrained(str(tmp_path))
    return str(tmp_path)


def _save_aux(tmp_path, name):
    """Save a tiny causal LM + wordlevel tokenizer to a subdir; return its path (shared vocab)."""
    path = tmp_path / name
    tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB).save_pretrained(str(path))
    wordlevel_tokenizer().save_pretrained(str(path))
    return str(path)


class _RecordingValue(BaseCandidateValue):
    """A value stub that records whether `prepare` / `cleanup` were called."""

    supports_batching = True
    scoring_cost = "cheap"

    def __init__(self):
        self.prepared = False
        self.cleaned = False

    def score(self, ctx):
        return torch.zeros(ctx.candidate_ids.shape)

    def prepare(self, model=None, tokenizer=None, **kwargs):
        self.prepared = True

    def cleanup(self):
        self.cleaned = True


# resolver
class TestResolveValue:
    def test_instance_calls_prepare(self):
        v = _RecordingValue()
        out = resolve_value(v, model=None, tokenizer=None, device="cpu")
        assert out is v
        assert v.prepared is True

    def test_callable_wrapped(self):
        out = resolve_value(lambda ctx: torch.zeros(1, 3), model=None, tokenizer=None, device="cpu")
        assert isinstance(out, CallableValue)

    def test_reward_model_dict(self, tmp_path):
        path = _make_tiny_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_value({"kind": "reward_model", "model_id": path},
                            model=model, tokenizer=wordlevel_tokenizer(), device="cpu")
        assert isinstance(out, RewardModelValue)

    def test_reward_model_dict_score_transform(self, tmp_path):
        path = _make_tiny_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_value({"kind": "reward_model", "model_id": path, "score_index": 1,
                             "score_transform": "softmax"},
                            model=model, tokenizer=wordlevel_tokenizer(), device="cpu")
        assert isinstance(out, RewardModelValue)
        assert out.score_index == 1
        assert out.score_transform == "softmax"

    def test_classifier_dict_from_model(self, tmp_path):
        path = _make_tiny_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_value({"kind": "classifier", "model_id": path, "label_index": 0},
                            model=model, tokenizer=wordlevel_tokenizer(), device="cpu")
        assert isinstance(out, ClassifierValue)
        assert out.label_index == 0

    def test_classifier_dict_from_fn(self):
        out = resolve_value({"kind": "classifier", "fn": lambda texts: torch.zeros(len(texts))},
                            model=None, tokenizer=None, device="cpu")
        assert isinstance(out, ClassifierValue)

    def test_subspace_margin_from_probe_path(self, tmp_path):
        probe_path = str(tmp_path / "p.probe")
        _write_probe_json(probe_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_value({"kind": "subspace_margin", "probe_path": probe_path},
                            model=model, tokenizer=wordlevel_tokenizer(), device="cpu")
        assert isinstance(out, SubspaceMarginValue)

    def test_subspace_margin_from_data(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        data = {"positives": ["the cat sat", "the dog ran"], "negatives": ["mat fast", "span on"]}
        out = resolve_value({"kind": "subspace_margin", "data": data},
                            model=model, tokenizer=wordlevel_tokenizer(), device="cpu")
        assert isinstance(out, SubspaceMarginValue)

    def test_callable_dict_flags(self):
        out = resolve_value({"kind": "callable", "fn": lambda ctx: torch.zeros(1, 2),
                             "supports_batching": True, "scoring_cost": "aux_forward"},
                            model=None, tokenizer=None, device="cpu")
        assert isinstance(out, CallableValue)
        assert out.supports_batching is True
        assert out.scoring_cost == "aux_forward"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="accepted kinds"):
            resolve_value({"kind": "nope"}, model=None, tokenizer=None, device="cpu")

    def test_malformed_missing_key_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            resolve_value({"kind": "reward_model"}, model=None, tokenizer=None, device="cpu")


class TestResolveSource:
    def test_str_shorthand(self, tmp_path):
        path = _save_aux(tmp_path, "aux")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_source(path, model=model, tokenizer=wordlevel_tokenizer())
        assert isinstance(out, AuxModelSource)
        assert out.model is not None

    def test_callable_wrapped(self):
        out = resolve_source(lambda ids: torch.zeros(1, VOCAB), model=None, tokenizer=None)
        assert isinstance(out, CallableSource)

    def test_aux_model_dict(self, tmp_path):
        path = _save_aux(tmp_path, "aux")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        out = resolve_source({"kind": "aux_model", "name_or_path": path},
                             model=model, tokenizer=wordlevel_tokenizer())
        assert isinstance(out, AuxModelSource)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="accepted kinds"):
            resolve_source({"kind": "nope"}, model=None, tokenizer=None)


class TestResolveScorer:
    def test_callable_passthrough(self):
        f = lambda p, c, params: [0.0] * len(c)
        assert resolve_scorer(f, device="cpu") is f

    def test_majority_vote_dict(self):
        out = resolve_scorer({"kind": "majority_vote"}, device="cpu")
        assert isinstance(out, MajorityVoteScorer)

    def test_reward_model_dict(self, tmp_path):
        path = _make_tiny_classifier(tmp_path)
        out = resolve_scorer({"kind": "reward_model", "model_id": path}, device="cpu")
        assert isinstance(out, RewardModelScorer)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="accepted kinds"):
            resolve_scorer({"kind": "nope"}, device="cpu")


# ValueGuidance
class TestValueGuidanceValidation:
    def test_missing_value(self):
        with pytest.raises(ValueError, match="value"):
            ValueGuidance()

    def test_bad_policy(self):
        with pytest.raises(ValueError, match="policy"):
            ValueGuidance(value=lambda ctx: ctx, policy="nope")

    def test_bad_normalize(self):
        with pytest.raises(ValueError, match="normalize"):
            ValueGuidance(value=lambda ctx: ctx, normalize="nope")

    def test_top_p_requires_p(self):
        with pytest.raises(ValueError, match="top_p"):
            ValueGuidance(value=lambda ctx: ctx, policy="top_p", p=None)

    def test_top_k_requires_positive_k(self):
        with pytest.raises(ValueError, match="top_k"):
            ValueGuidance(value=lambda ctx: ctx, policy="top_k", k=0)


class TestValueGuidanceEquivalence:
    def test_rad_equivalence_fixed_scores(self, tmp_path):
        """RAD and the RAD-equivalent ValueGuidance config produce the same shift on fixed scores.

        Both sides use the stateless shared-vocab reward value (`RAD(efficient=False)` and a
        `RewardModelValue(shared_vocab=True)` instance), so the comparison isolates the processor
        shift math (top-k, clamp, mask).
        """
        rm_path = _make_tiny_classifier(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        rad = RAD(beta=7.0, reward_model_id=rm_path, efficient=False)
        _pipeline([rad], model=model, tokenizer=tokenizer)

        vg = ValueGuidance(
            value=rad._value,
            policy="top_k", k=20, beta=7.0, normalize="clamp", mask_non_candidates=True,
        )
        _pipeline([vg], model=model, tokenizer=tokenizer)

        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        rad_proc = rad.get_logits_processors(prefix, {})[0]  # default top-20
        vg_proc = vg.get_logits_processors(prefix, {})[0]
        assert torch.allclose(
            rad_proc(prefix, scores.clone()), vg_proc(prefix, scores.clone()),
            atol=1e-4, equal_nan=True,
        )

    def test_sasa_equivalence_fixed_scores(self, tmp_path):
        """SASA and the SASA-equivalent ValueGuidance config produce the same shift on fixed scores."""
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        probe_path = str(tmp_path / "sasa.probe")
        _write_probe_json(probe_path)
        probe = load_single_file_probe(probe_path, layer_id=1)

        sasa = SASA(beta=3.0)
        sasa.model = model
        sasa.tokenizer = tokenizer
        sasa.probe = probe

        vg = ValueGuidance(
            value={"kind": "subspace_margin", "probe_path": probe_path},
            policy="surviving", beta=3.0, normalize="softmax", mask_non_candidates=False,
            include_in_scoring=False,
        )
        _pipeline([vg], model=model, tokenizer=tokenizer)

        prefix = torch.tensor([[0, 3, 4]])
        attention_mask = torch.ones_like(prefix)
        scores = torch.randn(1, VOCAB)
        scores[0, 10:] = float("-inf")

        sasa_proc = sasa.get_logits_processors(prefix, {}, attention_mask=attention_mask)[0]
        vg_proc = vg.get_logits_processors(prefix, {}, attention_mask=attention_mask)[0]
        assert torch.allclose(
            sasa_proc(prefix, scores.clone()), vg_proc(prefix, scores.clone()),
            atol=1e-4, equal_nan=True,
        )


class TestValueGuidanceBehavior:
    def test_supports_batching_top_k_batchable(self, tmp_path):
        rm_path = _make_tiny_classifier(tmp_path)
        vg = ValueGuidance(value={"kind": "reward_model", "model_id": rm_path}, policy="top_k")
        _pipeline([vg])
        assert vg.supports_batching is True  # RewardModelValue batchable + top_k

    def test_supports_batching_surviving_false(self, tmp_path):
        rm_path = _make_tiny_classifier(tmp_path)
        vg = ValueGuidance(value={"kind": "reward_model", "model_id": rm_path},
                           policy="surviving", mask_non_candidates=False, normalize="softmax")
        _pipeline([vg])
        assert vg.supports_batching is False  # surviving policy is batch-1

    def test_supports_batching_unbatchable_value_false(self):
        vg = ValueGuidance(value=lambda ctx: torch.zeros(ctx.candidate_ids.shape), policy="top_k")
        _pipeline([vg])
        assert vg.supports_batching is False  # CallableValue defaults supports_batching=False

    def test_model_forward_scoring_warns(self, tmp_path):
        probe_path = str(tmp_path / "p.probe")
        _write_probe_json(probe_path)
        vg = ValueGuidance(
            value={"kind": "subspace_margin", "probe_path": probe_path},
            policy="surviving", mask_non_candidates=False, normalize="softmax",
            include_in_scoring=True,
        )
        with pytest.warns(UserWarning, match="include_in_scoring"):
            _pipeline([vg])

    def test_fresh_processor_per_call(self, tmp_path):
        rm_path = _make_tiny_classifier(tmp_path)
        vg = ValueGuidance(value={"kind": "reward_model", "model_id": rm_path})
        _pipeline([vg])
        p1 = vg.get_logits_processors(torch.tensor([[0, 3]]), {})[0]
        p2 = vg.get_logits_processors(torch.tensor([[0, 3]]), {})[0]
        assert p1 is not p2
        assert p1.value is p2.value  # resolved value persists across calls

    def test_cleanup_nulls_value(self):
        v = _RecordingValue()
        vg = ValueGuidance(value=v, policy="top_k")
        _pipeline([vg])
        vg.cleanup()
        assert v.cleaned is True
        assert vg._value is None

    def test_unsteered_raises(self):
        vg = ValueGuidance(value=lambda ctx: ctx, policy="top_k")
        with pytest.raises(RuntimeError, match="steer"):
            vg.get_logits_processors(torch.tensor([[0, 3, 4]]), {})

    def test_end_to_end_forces_token(self):
        """A scripted callable value that dominates forces the first continuation token."""
        target = 5

        def force(ctx):
            return (ctx.candidate_ids == target).float() * 1000.0

        vg = ValueGuidance(value=force, policy="top_k", k=VOCAB, beta=1.0,
                           normalize="none", mask_non_candidates=True)
        pipeline, model, tokenizer = _pipeline([vg])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=4, do_sample=False, eos_token_id=None)
        # with the full vocabulary as candidates and a dominating value, every step picks the target
        assert torch.all(out[0] == target)


# ContrastiveGuidance
class TestContrastiveGuidanceValidation:
    def test_lengths_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            ContrastiveGuidance(sources=["a", "b"], weights=[1.0])

    def test_empty_sources(self):
        with pytest.raises(ValueError, match="non-empty"):
            ContrastiveGuidance(sources=[], weights=[])

    def test_alpha_range(self):
        with pytest.raises(ValueError, match="alpha"):
            ContrastiveGuidance(sources=["a"], weights=[1.0], alpha=2.0)


class TestContrastiveGuidanceMath:
    def test_mixture_matches_reference(self, tmp_path):
        expert = _save_aux(tmp_path, "expert")
        anti = _save_aux(tmp_path, "anti")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        cg = ContrastiveGuidance(sources=[expert, anti], weights=[2.0, -2.0])
        _pipeline([cg], model=model)

        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        proc = cg.get_logits_processors(prefix, {})[0]
        assert isinstance(proc, ContrastiveMixtureProcessor)
        out = proc(prefix, scores.clone())

        base_lp = torch.log_softmax(scores, dim=-1)
        expert_lp = cg._sources[0].logprobs(prefix)
        anti_lp = cg._sources[1].logprobs(prefix)
        expected = base_lp + 2.0 * expert_lp - 2.0 * anti_lp
        assert torch.allclose(out, expected, atol=1e-4)

    def test_alpha_mask_keep_set(self, tmp_path):
        expert = _save_aux(tmp_path, "expert")
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        cg = ContrastiveGuidance(sources=[expert], weights=[1.0], alpha=0.5)
        _pipeline([cg], model=model)
        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        out = cg.get_logits_processors(prefix, {})[0](prefix, scores.clone())
        base_probs = torch.log_softmax(scores, dim=-1).exp()
        threshold = 0.5 * base_probs.max()
        keep = base_probs >= threshold
        assert torch.all(torch.isinf(out[~keep]))  # masked-out tokens are -inf
        assert not torch.any(torch.isinf(out[keep]))

    def test_cleanup_releases_sources(self, tmp_path):
        expert = _save_aux(tmp_path, "expert")
        anti = _save_aux(tmp_path, "anti")
        cg = ContrastiveGuidance(sources=[expert, anti], weights=[1.0, -1.0])
        _pipeline([cg])
        sources = cg._sources
        assert all(s.model is not None for s in sources)
        cg.cleanup()
        assert cg._sources is None
        assert all(s.model is None for s in sources)  # §3.3.2: cleanup drops aux models

    def test_unsteered_raises(self):
        cg = ContrastiveGuidance(sources=["x"], weights=[1.0])
        with pytest.raises(RuntimeError, match="steer"):
            cg.get_logits_processors(torch.tensor([[0, 3]]), {})


# SearchDecoding
class TestSearchDecoding:
    def test_defaults_are_best_of_n(self):
        seen = {"segment_lens": [], "iters": 0}

        def scorer(prompt, conts, params):
            seen["segment_lens"].append(params.get("segment_len"))
            seen["iters"] += 1
            return [float(i) for i in range(len(conts))]

        sd = SearchDecoding(scorer=scorer, num_candidates=4)
        pipeline, model, tokenizer = _pipeline([sd])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=6, do_sample=True, eos_token_id=None)
        assert out.ndim == 2
        assert seen["segment_lens"] == [6]  # segment_len=None uses the global budget
        assert seen["iters"] == 1  # one iteration (best-of-N)

    def test_no_budget_raises(self):
        sd = SearchDecoding(scorer=lambda p, c, params: [0.0] * len(c))
        pipeline, model, tokenizer = _pipeline([sd])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        with pytest.raises(ValueError, match="segment length"):
            pipeline.generate(input_ids=prompt, do_sample=True, eos_token_id=None)

    def test_deal_trajectory_equivalence(self):
        """DeAL and the DeAL-equivalent SearchDecoding config produce identical ids (seed-pinned)."""
        def scorer(prompt, continuations, params):
            return [float(c.count("mat")) for c in continuations]

        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        prompt = tokenizer("the cat", return_tensors="pt").input_ids

        deal = DeAL(reward_func=scorer, lookahead=4, init_beams=4, topk=2, max_iterations=3)
        p_deal, _, _ = _pipeline([deal], model=model, tokenizer=tokenizer)
        torch.manual_seed(0)
        out_deal = p_deal.generate(input_ids=prompt, max_new_tokens=12)

        sd = SearchDecoding(scorer=scorer, segment_len=4, num_candidates=4, keep_k=2,
                            max_iterations=3, propose_mode="beam")
        p_sd, _, _ = _pipeline([sd], model=model, tokenizer=tokenizer)
        torch.manual_seed(0)
        out_sd = p_sd.generate(input_ids=prompt, max_new_tokens=12)

        assert out_deal.shape == out_sd.shape
        assert torch.equal(out_deal, out_sd)

    def test_scorer_dict_resolved_in_steer(self, tmp_path):
        rm_path = _make_tiny_classifier(tmp_path)
        sd = SearchDecoding(scorer={"kind": "reward_model", "model_id": rm_path})
        assert isinstance(sd.scorer, dict)  # unresolved before steer
        _pipeline([sd])
        assert isinstance(sd.scorer, RewardModelScorer)  # resolved during steer

    def test_keep_k_exceeds_candidates_rejected(self):
        with pytest.raises(ValueError, match="keep_k"):
            SearchDecoding(scorer=lambda p, c, params: [], num_candidates=2, keep_k=4)


# PhasedDecoding
class TestPhasedDecodingValidation:
    def test_two_keys_in_one_entry(self):
        with pytest.raises(ValueError, match="exactly one"):
            PhasedDecoding(plan=[{"fixed": "x", "generate": {}}])

    def test_unknown_subkey(self):
        with pytest.raises(ValueError, match="Unknown subkey"):
            PhasedDecoding(plan=[{"generate": {"nope": 1}}])

    def test_empty_plan(self):
        with pytest.raises(ValueError, match="non-empty"):
            PhasedDecoding(plan=[])

    def test_no_generated_warns(self):
        with pytest.warns(UserWarning, match="no 'generate' phase"):
            PhasedDecoding(plan=[{"fixed": "x"}])


class TestPhasedDecodingBehavior:
    def test_budget_forcing_respects_phase_budget(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        plan = [
            {"generate": {"budget": 3}},
            {"fixed": " sat"},
            {"generate": {"budget": 2}},
        ]
        pd = PhasedDecoding(plan=plan)
        pipeline, model, tokenizer = _pipeline([pd], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=50, do_sample=False, eos_token_id=None)
        # prompt(2 or 3) + <=3 generated + 1 fixed ("sat") + <=2 generated: bounded well under 50
        assert out.size(1) <= prompt.size(1) + 3 + 2 + 2

    def test_replacing_fixed_with_tail_extraction(self):
        """The thinking-intervention config: a replacing Fixed rewrite + Generated, with
        extract_after stripping the reasoning span (Wu et al., 2025, arXiv:2503.24370)."""
        def intervention(prompt, params):
            return f"the dog </think> {prompt}"

        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        prompt = tokenizer("the cat", return_tensors="pt").input_ids

        pd = PhasedDecoding(
            plan=[{"fixed": intervention, "replace": True, "add_special_tokens": True}, {"generate": {}}],
            extract_after="</think>",
        )
        pipeline, model, tokenizer = _pipeline([pd], model=model, tokenizer=tokenizer)
        torch.manual_seed(0)
        out = pipeline.generate(input_ids=prompt, max_new_tokens=6, do_sample=False, eos_token_id=None)

        decoded = tokenizer.decode(out[0])
        assert "</think>" not in decoded  # reasoning span stripped by the tail rule
        assert "the cat" in decoded       # extract_after keeps the post-marker remainder

    def test_serializable_plan_round_trips(self):
        plan = [
            {"fixed": "Answer: "},
            {"generate": {"until": "</think>", "budget": 8}},
            {"generate": {}},
        ]
        restored = json.loads(json.dumps(plan))
        pd_a = PhasedDecoding(plan=plan)
        pd_b = PhasedDecoding(plan=restored)
        a = pd_a.plan("p", {})
        b = pd_b.plan("p", {})
        assert [type(x).__name__ for x in a] == [type(x).__name__ for x in b]
        assert a[0].text == b[0].text
        assert (a[1].until, a[1].budget) == (b[1].until, b[1].budget)


# StoppingRules
class TestStoppingRules:
    def test_validation_nothing_configured(self):
        with pytest.raises(ValueError, match="at least one"):
            StoppingRules()

    def test_stop_texts_without_tokenizer_raises(self):
        sr = StoppingRules(stop_texts=["\n"])
        pipeline = SteeringPipeline(
            controls=[sr], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB), tokenizer=None,
        )
        with pytest.raises(RuntimeError, match="tokenizer"):
            pipeline.steer()

    def test_budget_halts(self):
        sr = StoppingRules(budget=3)
        pipeline, model, tokenizer = _pipeline([sr])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=20, do_sample=False, eos_token_id=None)
        assert out.size(1) <= 3  # continuation-only, budget respected

    def test_stop_on_token_halts(self):
        sr = StoppingRules(stop_token_ids=[5])
        pipeline, model, tokenizer = _pipeline([sr])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        # force token 5 via a composed step-level control so the stop fires deterministically
        from steerability.algorithms.output_control.value_guidance.control import ValueGuidance
        vg = ValueGuidance(value=lambda ctx: (ctx.candidate_ids == 5).float() * 1000.0,
                           policy="top_k", k=VOCAB, mask_non_candidates=True)
        pipeline2, model2, tokenizer2 = _pipeline([vg, sr])
        out = pipeline2.generate(input_ids=tokenizer2("the cat", return_tensors="pt").input_ids,
                                 max_new_tokens=20, do_sample=False, eos_token_id=None)
        assert out[0, 0].item() == 5
        assert out.size(1) <= 20

    def test_prompt_anchoring_across_calls(self):
        """Fresh criteria per call anchor at each prompt's own length."""
        sr = StoppingRules(budget=4)
        _pipeline([sr])
        c1 = sr.get_stopping_criteria(torch.tensor([[0, 3, 4]]), {})
        c2 = sr.get_stopping_criteria(torch.tensor([[0, 3, 4, 5, 6]]), {})
        assert c1[0]._prompt_len == 3
        assert c2[0]._prompt_len == 5
        assert c1[0] is not c2[0]

    def test_no_logits_processors(self):
        sr = StoppingRules(budget=4)
        _pipeline([sr])
        assert sr.get_logits_processors(torch.tensor([[0, 3]]), {}) == []


# registry
class TestRegistry:
    def test_all_five_discoverable(self):
        import steerability.algorithms.core.registry as r
        names = r.REGISTRY["output_control"]
        for name in ("value_guidance", "contrastive_guidance", "search_decoding",
                     "phased_decoding", "stopping_rules"):
            assert name in names
