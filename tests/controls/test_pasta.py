import logging
from types import SimpleNamespace

import pytest
import torch

from steerability.algorithms.core.execution.payloads import ItemResult
from steerability.algorithms.core.output import Output
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.pasta.control import PASTA
from steerability.algorithms.state_control.pasta.profiling import HeadProfile, HeadProfileResult
from tests.utils.sweep import build_param_grid
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PROMPT_TEXT = (
    "Answer truthfully. Therefore, when you respond: "
    "First, present your main point. "
    "Second, support it with evidence. "
    "Finally, conclude succinctly."
)

PASTA_GRID = {
    "substrings": [
        ["Therefore"],
        ["First,", "Second,", "Finally,"]
    ],
    "alpha": [0.25, 0.75],
    "scale_position": ["include", "exclude", "generation"],
    "head_config": [
        [0],
        [0, 1]
    ],
}


@pytest.mark.parametrize("conf", build_param_grid(PASTA_GRID))
def test_pasta(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that PASTA steers and generates on every model/device/param combo.
    """

    # move model to target device
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build pipeline with PASTA control
    pasta = PASTA(
        head_config=conf["head_config"],
        alpha=conf["alpha"],
        scale_position=conf["scale_position"]
    )
    pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
    pipeline.steer()

    # prepare prompt & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = {"substrings": conf["substrings"]}

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"


class TestFindTokenRangeMissingSubstring:
    def test_absent_substring_returns_sentinel_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            start, end = PASTA._find_token_range("hello world", "absent", [(0, 5), (5, 11)])
        assert (start, end) == (0, 0)
        assert any("not found" in record.message for record in caplog.records)
        # the full input text is not dumped into the log record
        assert not any("hello world" in record.getMessage() for record in caplog.records)


class TestAttentionPreHookMissingRangeNoOp:
    """Include-mode: an item whose only range is the (0, 0) sentinel must be left untouched, while a
    sibling item with a valid range is steered as before."""

    def _make_pasta(self, num_heads: int = 2) -> PASTA:
        pasta = PASTA.__new__(PASTA)
        pasta.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=num_heads))
        pasta.scale_position = "include"
        return pasta

    def test_empty_range_item_unchanged(self):
        pasta = self._make_pasta(num_heads=2)
        head_idx = [0, 1]
        batch_size, num_heads, seq_len = 2, 2, 6
        input_len = seq_len

        # item 0 has a valid range [1, 3]; item 1 has only the (0, 0) sentinel
        token_ranges = [
            torch.tensor([[1, 3]]),
            torch.tensor([[0, 0]]),
        ]

        attention_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        input_kwargs = {"attention_mask": attention_mask.clone()}
        hidden_states = torch.zeros(batch_size, seq_len, 4)

        original = attention_mask.clone()
        _, out_kwargs = pasta._attention_pre_hook(
            module=None,
            input_args=(hidden_states,),
            input_kwargs=input_kwargs,
            head_idx=head_idx,
            token_ranges=token_ranges,
            input_len=input_len,
            scale_constant=torch.tensor([2.0]).log(),
        )
        result = out_kwargs["attention_mask"]

        # item 1 (empty range only) is a true no-op
        assert torch.equal(result[1], original[1])
        # item 0 is modified relative to the untouched baseline
        assert not torch.equal(result[0], original[0])


class TestSubstringsAcceptedForms:
    """The `substrings` runtime kwarg accepts a str, a nested per-row list, and a flat list only
    at batch size 1."""

    def test_str_broadcasts_to_every_row(self):
        assert PASTA._normalize_substrings("First,", 3) == [["First,"], ["First,"], ["First,"]]

    def test_nested_groups_of_batch_length(self):
        groups = PASTA._normalize_substrings([["First,"], ["Second,", "Finally,"]], 2)
        assert groups == [["First,"], ["Second,", "Finally,"]]

    def test_flat_list_accepted_at_batch_size_one(self):
        assert PASTA._normalize_substrings(["First,", "Second,"], 1) == [["First,", "Second,"]]

    def test_flat_list_at_batch_size_above_one_raises_naming_workaround(self):
        with pytest.raises(ValueError, match=r"\[\[\.\.\.\]\] \* batch_size"):
            PASTA._normalize_substrings(["First,", "Second,"], 2)

    def test_str_group_raises(self):
        with pytest.raises(ValueError, match="non-str sequences of str"):
            PASTA._normalize_substrings(["First,", ["Second,"]], 2)

    def test_non_str_group_element_raises(self):
        with pytest.raises(ValueError, match="only str elements"):
            PASTA._normalize_substrings([[1], ["Second,"]], 2)

    def test_group_count_must_match_batch(self):
        with pytest.raises(ValueError, match="Need 3 substring groups"):
            PASTA._normalize_substrings([["First,"], ["Second,"]], 3)

    def test_local_copy_never_mutates_caller_groups(self):
        caller = [["First,"], ["Second,"]]
        groups = PASTA._normalize_substrings(caller, 2)
        groups[0].append("x")
        assert caller == [["First,"], ["Second,"]]


# head profiling (HeadProfile / HeadProfileResult)

def contains_target(response: str, row: dict) -> float:
    """Module-level toy scorer: 1.0 when the response contains `row['target']`, else 0.0."""
    return 1.0 if row.get("target", "\0") in response else 0.0


class _FakeSession:
    """A session stub whose `generate` returns a controlled score per candidate.

    The response text is the string `"steered"` or `"base"`; scoring is driven by `lift_map`,
    read by the paired scorer through this session, so a candidate's lift is exactly its map
    entry (the baseline scores 0.0). Records the number of generation items it is handed so a
    test can assert against `budget()`.
    """

    def __init__(self, tokenizer, lift_map: dict[tuple[int, int], float]):
        self.tokenizer = _DecodeToText(tokenizer)
        self.lift_map = lift_map
        self.current_score = 0.0
        self.item_count = 0

    def generate(self, items, params):
        entries = items[0].state_entries
        if entries:
            keywords = entries[0].hooks["pre"][0]["hook_func"].keywords
            candidate = (int(keywords["layer_idx"]), int(keywords["head_idx"][0]))
            self.current_score = float(self.lift_map.get(candidate, 0.0))
            token, text = 1, "steered"
        else:
            self.current_score = 0.0
            token, text = 0, "base"
        self.item_count += len(items)
        return [
            ItemResult(
                index=index,
                output=Output(output_ids=torch.tensor([[token]]), adapted_input_ids=None, finish_reason="stop"),
            )
            for index in range(len(items))
        ]


class _DecodeToText:
    """Wrap a tokenizer so `decode` returns a fixed marker for the fake session's token ids."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def decode(self, ids, skip_special_tokens=True):
        value = int(ids.reshape(-1)[0])
        return "steered" if value == 1 else "base"


def _steered_dict_pasta(num_layers: int = 2, heads: int = 3):
    """A dict-form PASTA steered on a tiny Llama, for a resolved control table and a fake session."""
    model = tiny_llama(num_layers=num_layers, hidden=heads * 4, heads=heads)
    tokenizer = wordlevel_tokenizer()
    pasta = PASTA(head_config={0: [0]}, alpha=100.0, scale_position="include")
    pasta.steer(model, tokenizer)
    return pasta, model, tokenizer


def _score_by_current(session: "_FakeSession"):
    """A scorer that reads the fake session's current per-candidate score."""
    return lambda response, row: session.current_score


class TestHeadProfileResolution:
    """Ranking, screening, eligibility, and selection over a fake session with controlled lifts."""

    def _rows(self, count: int = 4, groups=None):
        rows = [{"input": "the cat sat on mat", "substrings": ["cat sat"]} for _ in range(count)]
        if groups is not None:
            for row, group in zip(rows, groups):
                row["group"] = group
        return rows

    def test_head_profile_resolves_and_ranks_deterministically(self):
        pasta, model, tokenizer = _steered_dict_pasta(num_layers=2, heads=3)
        lift_map = {(0, 0): 0.5, (0, 1): 0.1, (1, 2): 0.3}  # others default to 0.0
        session = _FakeSession(tokenizer, lift_map)
        profile = HeadProfile(
            rows=self._rows(4), scorer=_score_by_current(session), alpha=100.0, num_heads=3,
            min_lift=-1.0, gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        result = profile.resolve(pasta, model, tokenizer, session=session)

        candidates = [(layer, head) for layer in (0, 1) for head in range(3)]
        for layer, head in candidates:
            assert not torch.isnan(result.lift[layer, head])
            assert int(result.n[layer, head]) == 4
        flat = [(layer, head) for layer, heads in result.head_config.items() for head in heads]
        assert len(flat) == 3

        expected = sorted(
            candidates, key=lambda lh: (-result.lift[lh[0], lh[1]].item(), lh[0], lh[1]),
        )[:3]
        assert sorted(result.selected) == sorted(expected)

        # a fresh recipe on the same model reproduces the selection exactly
        session2 = _FakeSession(tokenizer, lift_map)
        profile2 = HeadProfile(
            rows=self._rows(4), scorer=_score_by_current(session2), alpha=100.0, num_heads=3,
            min_lift=-1.0, gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        assert profile2.resolve(pasta, model, tokenizer, session=session2).selected == result.selected

    def test_head_profile_screen_stages(self):
        pasta, model, tokenizer = _steered_dict_pasta(num_layers=2, heads=3)
        lift_map = {(0, 0): 0.9, (1, 1): 0.8}  # the two clear winners survive the screen
        session = _FakeSession(tokenizer, lift_map)
        rows = self._rows(4)
        profile = HeadProfile(
            rows=rows, scorer=_score_by_current(session), alpha=100.0, num_heads=2,
            screen_rows=1, screen_keep=2, min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False}, seed=0,
        )
        result = profile.resolve(pasta, model, tokenizer, session=session)

        stage_two = [(l, h) for l in (0, 1) for h in range(3) if int(result.stage[l, h]) == 2]
        assert len(stage_two) == 2
        for layer, head in stage_two:
            assert int(result.n[layer, head]) == len(rows)
        for layer in (0, 1):
            for head in range(3):
                if (layer, head) not in stage_two:
                    assert int(result.n[layer, head]) == 1

        budget = profile.budget(2, 3)
        assert session.item_count == budget["total"]

    def test_head_profile_min_lift(self):
        pasta, model, tokenizer = _steered_dict_pasta(num_layers=2, heads=3)
        session = _FakeSession(tokenizer, lift_map={})  # every steered score equals the baseline
        rows = self._rows(4)

        raise_profile = HeadProfile(
            rows=rows, scorer=_score_by_current(session), alpha=100.0, num_heads=2, min_lift=0.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        with pytest.raises(ValueError, match="no candidate beat the baseline"):
            raise_profile.resolve(pasta, model, tokenizer, session=session)

        select_profile = HeadProfile(
            rows=rows, scorer=_score_by_current(session), alpha=100.0, num_heads=2, min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        assert len(select_profile.resolve(pasta, model, tokenizer, session=session).selected) == 2

        # a partially eligible set warns and selects the eligible subset
        partial_session = _FakeSession(tokenizer, lift_map={(0, 0): 0.5, (1, 1): 0.4})
        partial_profile = HeadProfile(
            rows=rows, scorer=_score_by_current(partial_session), alpha=100.0, num_heads=5, min_lift=0.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        with pytest.warns(UserWarning, match="only 2 candidates were eligible"):
            partial = partial_profile.resolve(pasta, model, tokenizer, session=partial_session)
        assert sorted(partial.selected) == [(0, 0), (1, 1)]

    def test_head_profile_intersection(self):
        pasta, model, tokenizer = _steered_dict_pasta(num_layers=2, heads=3)

        no_group = HeadProfile(
            rows=self._rows(4), scorer=lambda r, row: 0.0, alpha=100.0, num_heads=2,
            selection="intersection", gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        with pytest.raises(ValueError, match="requires a 'group'"):
            no_group.resolve(pasta, model, tokenizer, session=_FakeSession(tokenizer, {}))

        # two groups; (0,0) and (1,1) top both groups, so their intersection reaches num_heads=2
        rows = self._rows(4, groups=["a", "a", "b", "b"])
        session = _FakeSession(tokenizer, lift_map={(0, 0): 0.9, (1, 1): 0.8, (0, 1): 0.2})
        profile = HeadProfile(
            rows=rows, scorer=_score_by_current(session), alpha=100.0, num_heads=2,
            selection="intersection", min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        with pytest.warns(UserWarning, match="robust down to 200 samples"):
            result = profile.resolve(pasta, model, tokenizer, session=session)
        assert sorted(result.selected) == [(0, 0), (1, 1)]

    def test_head_profile_memoizes_per_model(self):
        pasta, model, tokenizer = _steered_dict_pasta(num_layers=2, heads=3)
        calls = {"n": 0}

        def counting_scorer(response, row):
            calls["n"] += 1
            return session.current_score

        session = _FakeSession(tokenizer, lift_map={(0, 0): 0.5})
        profile = HeadProfile(
            rows=self._rows(4), scorer=counting_scorer, alpha=100.0, num_heads=2, min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        # two controls sharing one recipe on one model: the second reuses the memoized result.
        # differing only in the control's own alpha does not change the fit.
        pasta_a = PASTA(head_config=profile, alpha=25.0, scale_position="include")
        pasta_a.steer(model, tokenizer, session=session)
        after_first = calls["n"]
        pasta_b = PASTA(head_config=profile, alpha=200.0, scale_position="include")
        pasta_b.steer(model, tokenizer, session=session)
        assert calls["n"] == after_first  # no additional scorer calls on the memo hit
        assert pasta_a.head_profile is pasta_b.head_profile

        # a different model instance refits
        other = tiny_llama(num_layers=2, hidden=12, heads=3)
        session_other = _FakeSession(tokenizer, lift_map={(0, 0): 0.5})
        profile.scorer = lambda r, row: session_other.current_score
        pasta_c = PASTA(head_config=profile, alpha=100.0, scale_position="include")
        pasta_c.steer(other, tokenizer, session=session_other)
        assert pasta_c.head_profile is not pasta_a.head_profile


class TestHeadProfileResultRoundTrip:
    def test_head_profile_result_round_trip(self, tmp_path):
        lift = torch.tensor([[float("nan"), 0.1], [0.2, float("nan")]])
        result = HeadProfileResult(
            head_config={0: [1], 1: [0]}, selected=[(0, 1), (1, 0)],
            lift=lift, se=torch.tensor([[float("nan"), 0.01], [0.02, float("nan")]]),
            n=torch.tensor([[0, 4], [4, 0]]), stage=torch.tensor([[0, 2], [2, 0]]),
            group_lift=None, groups=None, group_rows=None,
            baseline=0.25, num_rows=4, tie_at_cutoff=1, alpha=100.0, scale_position="include",
        )
        path = tmp_path / "profile.json"
        result.save(path)
        loaded = HeadProfileResult.load(path)

        assert loaded.head_config == result.head_config
        assert loaded.selected == result.selected
        assert torch.equal(loaded.lift.isnan(), result.lift.isnan())
        assert torch.equal(loaded.lift.nan_to_num(), result.lift.nan_to_num())
        assert loaded.group_lift is None and loaded.groups is None and loaded.group_rows is None
        assert loaded.to_frame().equals(result.to_frame())


class TestProfiledPASTAThroughPipeline:
    def test_pasta_profiled_through_pipeline(self):
        model = tiny_llama(num_layers=2, hidden=12, heads=3)
        tokenizer = wordlevel_tokenizer()
        profile = HeadProfile(
            rows=[{"input": "the cat sat on mat", "substrings": ["cat sat"], "target": "cat"} for _ in range(3)],
            scorer=contains_target, alpha=100.0, num_heads=2, layers=[0, 1], min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        pasta = PASTA(head_config=profile, alpha=100.0, scale_position="include")
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)

        plan = pipeline.check().plan
        assert [(fit.control, fit.artifact, fit.artifact_class, fit.venue) for fit in plan.fits] == [
            ("PASTA", "HeadProfile", "direction", "live"),
        ]

        pipeline.steer()
        assert pasta.head_profile is not None
        assert isinstance(pasta.head_config, HeadProfile)  # the recipe is retained on the control
        flat = [(layer, head) for layer, heads in pasta.head_profile.head_config.items() for head in heads]
        assert len(flat) == 2

        out = pipeline.generate(
            input_ids=tokenizer("the cat sat on mat", return_tensors="pt").input_ids,
            runtime_kwargs={"substrings": ["cat sat"]},
            max_new_tokens=4,
        )
        assert out.size(1) >= 1


class TestHeadProfileBatchSize:
    """`batch_size` changes dispatch granularity only: the rollout accounting and the resolved
    profile are the same, and the number of `model.generate` calls scales down with the batch."""

    def _resolve(self, model, tokenizer, rows, batch_size):
        from unittest.mock import patch

        profile = HeadProfile(
            rows=rows, scorer=contains_target, alpha=100.0, num_heads=2, layers=[0, 1],
            min_lift=-1.0, gen_kwargs={"max_new_tokens": 4, "do_sample": False},
            batch_size=batch_size,
        )
        pasta = PASTA(head_config=profile, alpha=100.0, scale_position="include")
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
        with patch.object(model, "generate", wraps=model.generate) as spy:
            pipeline.steer()
        return spy.call_count, pasta.head_profile, profile.budget(2, 3)

    def test_batch_size_changes_dispatch_only(self):
        model = tiny_llama(num_layers=2, hidden=12, heads=3)
        tokenizer = wordlevel_tokenizer()
        # identical prompts, so every chunk is padding-free and batching is numerically inert
        rows = [
            {"input": "the cat sat on mat", "substrings": ["cat sat"], "target": "cat"}
            for _ in range(6)
        ]

        serial_calls, serial, budget = self._resolve(model, tokenizer, rows, batch_size=1)
        batched_calls, batched, _ = self._resolve(model, tokenizer, rows, batch_size=6)

        # one generate call per rollout when serial; one per (baseline or candidate) when batched
        assert serial_calls == budget["total"]                      # 6 baseline + 6 candidates * 6 rows = 42
        assert batched_calls == 1 + budget["candidates"]            # 7

        assert torch.equal(serial.n, batched.n)
        assert torch.equal(serial.stage, batched.stage)
        torch.testing.assert_close(serial.lift, batched.lift, equal_nan=True)
        assert serial.selected == batched.selected
        assert serial.head_config == batched.head_config


class TestBuildHooksMatchesGetHooks:
    def test_build_hooks_matches_get_hooks(self):
        model = tiny_llama(num_layers=2, hidden=12, heads=3)
        tokenizer = wordlevel_tokenizer()
        pasta = PASTA(head_config={0: [1]}, alpha=3.0, scale_position="include")
        pasta.steer(model, tokenizer)

        input_ids = tokenizer("the cat sat on mat", return_tensors="pt").input_ids
        hooks_public = pasta.get_hooks(input_ids, {"substrings": ["cat sat"]})
        token_ranges, input_len = pasta.locate_spans(input_ids, ["cat sat"])
        hooks_built = pasta.build_hooks_for(token_ranges, input_len, {0: [1]}, 3.0)

        assert hooks_public["pre"][0]["module"] == hooks_built["pre"][0]["module"]
        kw_public = hooks_public["pre"][0]["hook_func"].keywords
        kw_built = hooks_built["pre"][0]["hook_func"].keywords
        assert kw_public["head_idx"] == kw_built["head_idx"] == [1]
        assert kw_public["input_len"] == kw_built["input_len"]
        assert torch.equal(kw_public["token_ranges"][0], kw_built["token_ranges"][0])
        torch.testing.assert_close(kw_public["scale_constant"], kw_built["scale_constant"])


class TestIncludeModeSingleTouch:
    """Include mode edits each prompt column once: span columns net to zero, overlapping too."""

    def _apply(self, ranges, scale_constant, seq_len=6, num_heads=2):
        pasta = PASTA.__new__(PASTA)
        pasta.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=num_heads))
        pasta.scale_position = "include"
        attention_mask = torch.zeros(1, num_heads, seq_len, seq_len, dtype=torch.bfloat16)
        _, out = pasta._attention_pre_hook(
            module=None,
            input_args=(torch.zeros(1, seq_len, 4),),
            input_kwargs={"attention_mask": attention_mask.clone()},
            head_idx=[0, 1],
            token_ranges=[torch.tensor(ranges)],
            input_len=seq_len,
            scale_constant=scale_constant,
        )
        return out["attention_mask"]

    def test_span_columns_net_to_zero_in_bfloat16(self):
        scale_constant = torch.tensor([100.0]).log()
        mask = self._apply([[2, 4]], scale_constant)
        row = mask[0, 0, -1]
        # span columns (2, 3) are untouched (exactly zero), non-span prompt columns are lowered
        assert torch.equal(row[2], torch.zeros((), dtype=row.dtype))
        assert torch.equal(row[3], torch.zeros((), dtype=row.dtype))
        assert (row[0] < 0) and (row[1] < 0) and (row[4] < 0) and (row[5] < 0)

    def test_overlapping_spans_net_to_zero(self):
        scale_constant = torch.tensor([100.0]).log()
        mask = self._apply([[2, 4], [3, 5]], scale_constant)  # overlap on column 3
        row = mask[0, 0, -1]
        for column in (2, 3, 4):
            assert torch.equal(row[column], torch.zeros((), dtype=row.dtype))


class TestProfiledPASTAFreezes:
    def test_profiled_pasta_freezes_and_loads(self, tmp_path):
        from steerability.spipe import SPipe
        from steerability.spipe.errors import SpipeStaleError

        model = tiny_llama(num_layers=2, hidden=12, heads=3)
        tokenizer = wordlevel_tokenizer()
        rows = [{"input": "the cat sat on mat", "substrings": ["cat sat"], "target": "cat"} for _ in range(3)]
        profile = HeadProfile(
            rows=rows, scorer=contains_target, alpha=100.0, num_heads=2, layers=[0, 1], min_lift=-1.0,
            gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        pasta = PASTA(head_config=profile, alpha=100.0, scale_position="include")
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
        pipeline.steer()
        resolved = pasta.head_profile.head_config

        path = tmp_path / "profiled.spipe"
        pipeline.to_spipe(model_ref="tiny-llama").save(path)

        loaded = SPipe.load(path, allow_code=True).pipeline()
        frozen = loaded.state_controls[0]
        # the frozen control carries the resolved dict head map and makes no fit
        assert frozen.head_config == resolved
        assert frozen.steer_fits() == ()
        # steering the frozen control takes the dict branch: no HeadProfile.resolve, no rollouts
        frozen.steer(model, tokenizer)
        assert frozen.head_profile is None
        assert frozen._head_map == {int(layer): list(heads) for layer, heads in resolved.items()}

        # editing a fit-relevant recipe field invalidates the frozen artifact
        def mutate(manifest):
            entry = manifest["controls"][0]
            head_config_field = entry["args"]["head_config"]["fields"]
            head_config_field["num_heads"] = 3

        _edit_spipe(path, mutate)
        with pytest.raises(SpipeStaleError):
            SPipe.load(path, allow_code=True)
        SPipe.load(path, allow_code=True, allow_stale=True)  # bypass

    def test_frozen_load_when_scorer_module_absent(self, tmp_path):
        # the profiling scorer is loaded from a file into a module name that is not on the import
        # path, mirroring a notebook that loads its task module via spec_from_file_location; the
        # staleness check on the allow_code=True load must digest the scorer's $ref without
        # importing that module, so the frozen resolved head map loads
        import importlib.util
        import sys

        from steerability.spipe import SPipe

        scorer_source = (
            "from typing import Mapping\n"
            "def strict_follow(response: str, row: Mapping) -> float:\n"
            "    return 1.0 if row.get('target', '\\0') in response else 0.0\n"
        )
        scorer_file = tmp_path / "notebook_task.py"
        scorer_file.write_text(scorer_source)
        module_name = "notebook_task_absent_from_path"
        spec = importlib.util.spec_from_file_location(module_name, scorer_file)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        assert module_name not in sys.modules

        model = tiny_llama(num_layers=2, hidden=12, heads=3)
        tokenizer = wordlevel_tokenizer()
        rows = [{"input": "the cat sat on mat", "substrings": ["cat sat"], "target": "cat"} for _ in range(3)]
        profile = HeadProfile(
            rows=rows, scorer=task_module.strict_follow, alpha=100.0, num_heads=2, layers=[0, 1],
            min_lift=-1.0, gen_kwargs={"max_new_tokens": 4, "do_sample": False},
        )
        pasta = PASTA(head_config=profile, alpha=100.0, scale_position="include")
        pipeline = SteeringPipeline(controls=[pasta], model=model, tokenizer=tokenizer)
        pipeline.steer()
        resolved = pasta.head_profile.head_config

        path = tmp_path / "profiled.spipe"
        pipeline.to_spipe(model_ref="tiny-llama").save(path)

        loaded = SPipe.load(path, allow_code=True).pipeline()
        assert loaded.state_controls[0].head_config == resolved


def _edit_spipe(path, mutate) -> None:
    """Rewrite the manifest of a `.spipe` zip in place after applying `mutate`."""
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        blobs = {name: archive.read(name) for name in names}
    manifest = json.loads(blobs["spipe.json"])
    mutate(manifest)
    blobs["spipe.json"] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr(name, blob)
