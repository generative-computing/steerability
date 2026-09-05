"""Rollout-scored attention-head profiling for PASTA, resolved at steer time.

`HeadProfile` is a third accepted form of `PASTAArgs.head_config`, beside the dict and list
forms. It is a declarative recipe that `PASTA.steer()` resolves through the pipeline's session:
each candidate `(layer, head)` is steered on its own on a task-agnostic set of profiling
prompts, scored by a `SampleScorer`, and ranked by the paired lift of its score over an
unsteered baseline. The selected heads become the control's dict-form head map, and the
resolution `HeadProfileResult` records the per-head lift, standard error, and selection so the
run can be inspected and rebuilt.

The recipe follows the toolkit's source idiom (`ContrastiveFit`, `ConditionPointSearch`):
it declares `access` and `artifact_class` for the steer plan, memoizes its resolution per model,
and its resolved head map freezes into a `.spipe` as a same-class `state_control/pasta` entry
with the fit digest on the lift grid.

Reference:

    - "Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs"
      Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao
      [https://arxiv.org/abs/2311.02262](https://arxiv.org/abs/2311.02262)
"""
from __future__ import annotations

import json
import logging
import math
import random
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, Mapping, Sequence

import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.params import GenerationParams
from steerability.algorithms.core.execution.payloads import GenerationItem, HookEntry, PreparedPrompt
from steerability.algorithms.core.scoring import SampleScorer
from steerability.algorithms.core.utils.generation import (
    PromptWarnings,
    prepare_inputs,
    resolve_messages_prompt,
    resolve_text_prompt,
)
from steerability.utils.rendering import has_chat_template

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession
    from steerability.algorithms.state_control.pasta.control import PASTA

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProfileBatch:
    """One prepared batch of profiling rows with its PASTA spans located once.

    Attributes:
        input_ids: Left-padded prompt token ids on the model device, shape `[rows, seq_len]`.
        attention_mask: Attention mask matching `input_ids`.
        token_ranges: One `[G, 2]` tensor of `(start, end)` spans per row, in the padded
            coordinates of `input_ids`.
        input_len: The padded sequence length (the attention mask's key axis).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_ranges: list[torch.Tensor]
    input_len: int


@dataclass
class HeadProfile:
    """Rollout-scored head profiling, resolved at `PASTA.steer()`.

    A recipe passed as `PASTA(head_config=HeadProfile(...))`. It scores each candidate
    `(layer, head)` by steering that head alone at the profiling strength `alpha` and measuring
    the mean gain in `scorer` over an unsteered baseline on `rows`, ranks the candidates
    deterministically, filters to those that beat the baseline, and returns the top `num_heads`
    as a dict head map. Resolution runs through the session the pipeline hands to `steer()`, so
    the rollouts execute on the steering backend.

    The paired lift and its standard error are the statistic; ranking by lift and ranking by the
    raw follow rate order candidates identically (the baseline is constant across candidates),
    but the pairing gives a standard error the raw rate does not. A two-stage screen bounds cost:
    stage 1 scores every candidate on a fixed subset of `screen_rows` rows, and stage 2 rescores
    only the top `screen_keep` candidates on every row.

    The profile uses the control's `scale_position`; it is not a recipe field, since a profile is
    only meaningful for the mechanism it is deployed with.

    Attributes:
        rows: One mapping per profiling prompt. `"input"` is the user turn (the `SampleScorer`
            row convention); `"substrings"` is the row's PASTA runtime kwarg in its per-row form
            (`list[str]`). An optional `"group"` enables the per-group statistics and the
            `"intersection"` selection. Every other key passes through to the scorer. The rows
            encode into a frozen `.spipe` inline, so the set must stay under the codec's
            per-entry inline limit (1 MB, about a few thousand short prompts).
        scorer: `(response, row) -> float`, higher is better. For a strict pass/fail checker this
            is 0.0 or 1.0; a loose checker or a reward score are drop-in alternatives with finer
            granularity.
        alpha: The profiling strength. Required, with no default. The toolkit parameterizes
            `alpha` as the reciprocal of the paper's coefficient, so `alpha=100.0` is the paper's
            operating point (coefficient 0.01). Stating it here keeps every point of an `alpha`
            sweep the same fit, so the profile is resolved once per run.
        num_heads: Number of heads to select.
        layers: Candidate layers. None means every attention layer of the resolved layout;
            non-attention layers of a hybrid stack are never candidates.
        selection: `"pooled"` selects the first `num_heads` eligible candidates in ranking order.
            `"intersection"` is the paper's rule and requires a `"group"` on every row.
        min_lift: A candidate is eligible when its pooled lift exceeds this value. The default
            `0.0` keeps only heads that beat the baseline.
        screen_rows: Rows scored in stage 1 of the two-stage screen. Set together with
            `screen_keep`, or leave both unset for a single-stage profile over every row.
        screen_keep: Candidates rescored on every row in stage 2. At least `num_heads`;
            `2 * num_heads` or more is recommended, since the stage-1 standard error is large
            enough to push a true positive out of a stage-2 pool sized exactly to the target.
        gen_kwargs: Profiling generation parameters, normalized through
            `GenerationParams.from_gen_kwargs`. Greedy by default.
        batch_size: Number of rows generated per session call. Rows in a batch share one hook entry
            and generate in one batched pass, so larger values trade memory for wall-clock; excluded
            from the fit digest.
        seed: Fixes the stage-1 subsample.
        progress_callback: Optional callback invoked once per scored candidate with a small dict
            (`stage`, `candidate`, `completed`, `total`, `lift`). Opt-in; no-op when None.

    Reference:

        - "Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs"
          Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao
          [https://arxiv.org/abs/2311.02262](https://arxiv.org/abs/2311.02262)
    """

    access: ClassVar[ModelAccess] = ModelAccess.MODULE
    artifact_class: ClassVar[str] = "direction"

    rows: Sequence[Mapping]
    scorer: SampleScorer
    alpha: float
    num_heads: int = 48
    layers: Sequence[int] | None = None
    selection: Literal["pooled", "intersection"] = "pooled"
    min_lift: float = 0.0
    screen_rows: int | None = None
    screen_keep: int | None = None
    gen_kwargs: dict = field(default_factory=lambda: {"max_new_tokens": 128, "do_sample": False})
    batch_size: int = 8
    seed: int = 0
    progress_callback: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        # the memo is a plain instance attribute rather than a dataclass field: identity
        # canonicalization walks every dataclasses.fields() entry, so a fitted memo stored as a
        # field would change a sweep's config_id after steer()
        self._model_ref: "weakref.ref | None" = None
        self._result: "HeadProfileResult | None" = None
        self._result_scale_position: str | None = None

        if not callable(self.scorer):
            raise ValueError("HeadProfile.scorer must be callable (response, row) -> float.")
        if not (isinstance(self.alpha, (int, float)) and self.alpha > 0):
            raise ValueError("HeadProfile.alpha must be a positive number.")
        if int(self.num_heads) < 1:
            raise ValueError("HeadProfile.num_heads must be at least 1.")
        self.num_heads = int(self.num_heads)
        if self.selection not in ("pooled", "intersection"):
            raise ValueError(f"HeadProfile.selection must be 'pooled' or 'intersection'; got {self.selection!r}.")
        if not math.isfinite(float(self.min_lift)):
            raise ValueError("HeadProfile.min_lift must be finite.")
        rows = list(self.rows)
        if not rows:
            raise ValueError("HeadProfile.rows must be non-empty.")
        for index, row in enumerate(rows):
            if "input" not in row or "substrings" not in row:
                raise ValueError(
                    f"HeadProfile.rows[{index}] must carry 'input' and 'substrings'; got keys {sorted(row)}."
                )
        self.rows = rows

        screen_set = self.screen_rows is not None
        keep_set = self.screen_keep is not None
        if screen_set != keep_set:
            raise ValueError("HeadProfile.screen_rows and screen_keep must both be set or both be unset.")
        if screen_set:
            if not 0 < int(self.screen_rows) < len(rows):
                raise ValueError(
                    f"HeadProfile.screen_rows must be in (0, {len(rows)}); got {self.screen_rows}."
                )
            if int(self.screen_keep) < self.num_heads:
                raise ValueError(
                    f"HeadProfile.screen_keep ({self.screen_keep}) must be at least num_heads ({self.num_heads})."
                )
            self.screen_rows = int(self.screen_rows)
            self.screen_keep = int(self.screen_keep)

    def budget(self, num_layers: int, num_heads: int) -> dict[str, int]:
        """Rollout counts for a `(num_layers, num_heads)` grid, without loading a model.

        A rollout is one prompt generated once. The candidate count is restricted to `layers`
        when set. With a screen, stage 1 scores every candidate on `screen_rows` rows and stage 2
        rescores `screen_keep` candidates on every row; without one, every candidate is scored on
        every row in stage 1.

        Args:
            num_layers: Number of attention layers in the model.
            num_heads: Number of attention heads per layer.

        Returns:
            A mapping with keys `candidates`, `baseline`, `stage_1`, `stage_2`, and `total`.
        """
        if self.layers is None:
            candidates = int(num_layers) * int(num_heads)
        else:
            candidates = len([layer for layer in self.layers if 0 <= int(layer) < int(num_layers)]) * int(num_heads)
        num_rows = len(self.rows)
        baseline = num_rows
        if self.screen_rows is not None:
            stage_1 = candidates * self.screen_rows
            stage_2 = min(self.screen_keep, candidates) * num_rows
        else:
            stage_1 = candidates * num_rows
            stage_2 = 0
        return {
            "candidates": candidates,
            "baseline": baseline,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "total": baseline + stage_1 + stage_2,
        }

    def fit_ingredients(self) -> dict:
        """The fit-relevant recipe inputs, digested for staleness detection.

        Covers `rows`, `scorer`, `alpha`, `num_heads`, `layers`, `selection`, `min_lift`,
        `screen_rows`, `screen_keep`, `gen_kwargs`, and `seed`. `batch_size` and
        `progress_callback` are execution details and are excluded; the control's own `alpha` is
        an application parameter and is excluded here (it lives on the control, not the recipe).
        The scorer digests as its qualified name, so renaming the scorer invalidates the frozen
        profile.
        """
        return {
            "rows": list(self.rows),
            "scorer": self.scorer,
            "alpha": float(self.alpha),
            "num_heads": self.num_heads,
            "layers": None if self.layers is None else [int(layer) for layer in self.layers],
            "selection": self.selection,
            "min_lift": float(self.min_lift),
            "screen_rows": self.screen_rows,
            "screen_keep": self.screen_keep,
            "gen_kwargs": dict(self.gen_kwargs),
            "seed": int(self.seed),
        }

    def resolve(
        self,
        control: "PASTA",
        model,
        tokenizer,
        *,
        session: "SteeringSession",
    ) -> "HeadProfileResult":
        """Resolve the profile against `model`, fitting once and memoizing.

        Reruns the profiling loop through `session` and returns the ranked, filtered
        `HeadProfileResult`. The result is memoized per model (a weakref slot) and the control's
        `scale_position`, so a second control sharing this recipe on the same model reuses it and
        a different model refits.

        Args:
            control: The steering PASTA, consulted for `scale_position` and its hook builder.
            model: The live model to profile against.
            tokenizer: The tokenizer for prompt preparation and decoding.
            session: The `ScopedSession` the pipeline hands to `steer()`, scoped to
                `ModelAccess.MODULE`, through which the rollouts run.

        Returns:
            The `HeadProfileResult`.

        Raises:
            ValueError: If no candidate beats the baseline (the message names the fixes).
        """
        scale_position = control.scale_position
        if (
            model is not None
            and self._model_ref is not None
            and self._model_ref() is model
            and self._result is not None
            and self._result_scale_position == scale_position
        ):
            return self._result

        result = self._run(control, model, tokenizer, session=session)
        if model is not None:
            self._model_ref = weakref.ref(model)
            self._result = result
            self._result_scale_position = scale_position
        return result

    def _run(
        self,
        control: "PASTA",
        model,
        tokenizer,
        *,
        session: "SteeringSession",
    ) -> "HeadProfileResult":
        """Score every candidate, rank, filter, and select (no caching)."""
        device = next(model.parameters()).device
        params = GenerationParams.from_gen_kwargs(**self.gen_kwargs)
        rows = self.rows
        num_rows = len(rows)

        groups = [row.get("group") for row in rows]
        has_groups = all(group is not None for group in groups)
        if self.selection == "intersection" and not has_groups:
            raise ValueError("HeadProfile.selection='intersection' requires a 'group' on every row.")
        group_names = sorted({str(group) for group in groups}) if has_groups else None

        # prepare each batch of rows once (left-padded on the model device, spans located in the
        # padded coordinates) and reuse the tensors for the baseline and every candidate
        batches = self._prepare_batches(control, tokenizer, device)

        # baseline: unsteered scores per row, in row order
        s_base = self._score_rows(control, session, batches, params, head_map=None, rows=rows)

        candidates = self._enumerate_candidates(control)
        screen_rows_idx = self._screen_subset(groups) if self.screen_rows is not None else None
        if screen_rows_idx is not None:
            # prepare the screen subset once (spans located once), reused across every stage-1
            # candidate, and take its baseline slice in row order
            screen_batches = self._prepare_batches(control, tokenizer, device, row_indices=screen_rows_idx)
            screen_rows = [rows[i] for i in screen_rows_idx]
            screen_base = torch.tensor([s_base[i].item() for i in screen_rows_idx], dtype=torch.float32)
            screen_groups = [str(groups[i]) for i in screen_rows_idx] if has_groups else None

        num_candidate_layers = max((layer for layer, _ in candidates), default=-1) + 1
        max_heads = max((head for _, head in candidates), default=-1) + 1
        lift = torch.full((num_candidate_layers, max_heads), float("nan"))
        se = torch.full((num_candidate_layers, max_heads), float("nan"))
        n = torch.zeros((num_candidate_layers, max_heads), dtype=torch.long)
        stage = torch.zeros((num_candidate_layers, max_heads), dtype=torch.long)
        group_lift = None
        group_rows = None
        if has_groups:
            group_lift = torch.full((len(group_names), num_candidate_layers, max_heads), float("nan"))
            group_index = {name: i for i, name in enumerate(group_names)}
            group_rows = [sum(1 for group in groups if str(group) == name) for name in group_names]

        total = len(candidates)
        completed = 0

        def score_candidate(layer, head, scored_batches, scored_rows, base, scored_groups, stage_value):
            nonlocal completed
            steered = self._score_rows(
                control, session, scored_batches, params, head_map={layer: [head]}, rows=scored_rows,
            )
            diff = steered - base
            lift[layer, head] = diff.mean()
            se[layer, head] = self._std_err(diff)
            n[layer, head] = diff.numel()
            stage[layer, head] = stage_value
            if has_groups:
                for name in group_names:
                    mask = [i for i, g in enumerate(scored_groups) if g == name]
                    if mask:
                        group_lift[group_index[name], layer, head] = diff[mask].mean()
            completed += 1
            if self.progress_callback is not None:
                self.progress_callback({
                    "stage": stage_value,
                    "candidate": (layer, head),
                    "completed": completed,
                    "total": total,
                    "lift": float(diff.mean()),
                })

        all_groups = [str(group) for group in groups] if has_groups else None
        if screen_rows_idx is None:
            for layer, head in candidates:
                score_candidate(layer, head, batches, rows, s_base, all_groups, 2)
            scored = candidates
        else:
            for layer, head in candidates:
                score_candidate(layer, head, screen_batches, screen_rows, screen_base, screen_groups, 1)
            screened = sorted(
                candidates, key=lambda lh: (-_nan_low(lift[lh[0], lh[1]].item()), lh[0], lh[1]),
            )[: self.screen_keep]
            total = len(candidates) + len(screened)
            for layer, head in screened:
                score_candidate(layer, head, batches, rows, s_base, all_groups, 2)
            scored = screened

        baseline_value = float(s_base.mean())
        selected, tie_at_cutoff = self._select(scored, lift, group_lift, group_names, group_rows)
        head_config = {}
        for layer, head in selected:
            head_config.setdefault(layer, []).append(head)
        head_config = {layer: sorted(heads) for layer, heads in sorted(head_config.items())}

        return HeadProfileResult(
            head_config=head_config,
            selected=selected,
            lift=lift,
            se=se,
            n=n,
            stage=stage,
            group_lift=group_lift,
            groups=group_names,
            group_rows=group_rows,
            baseline=baseline_value,
            num_rows=num_rows,
            tie_at_cutoff=tie_at_cutoff,
            alpha=float(self.alpha),
            scale_position=control.scale_position,
        )

    def _prepare_batches(
        self, control: "PASTA", tokenizer, device, *, row_indices: Sequence[int] | None = None,
    ) -> list["_ProfileBatch"]:
        """Left-padded batches for the selected rows, with PASTA spans located once per batch.

        Reproduces the pipeline's prompt path: rows render as chat messages when the tokenizer
        has a chat template and as text otherwise, then tokenize with an empty input-control
        chain. Batching by `batch_size` matches the session's batched generate path, which
        re-stacks and left-packs the rows before the forward pass, an identity on a batch already
        left-padded to a common length. The `substrings` spans are located in these padded
        coordinates once and reused for every candidate, so a candidate only reassembles the hook
        dict rather than re-decoding and re-tokenizing.
        """
        rows = self.rows if row_indices is None else [self.rows[i] for i in row_indices]
        chat = has_chat_template(tokenizer)
        warnings_state = PromptWarnings()
        batches: list[_ProfileBatch] = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            inputs = [row["input"] for row in chunk]
            if chat:
                input_ids, attention_mask, message_handled, _ = resolve_messages_prompt(
                    [[{"role": "user", "content": text}] for text in inputs],
                    runtime_kwargs={},
                    input_controls=[],
                    tokenizer=tokenizer,
                )
            else:
                input_ids, attention_mask, _ = resolve_text_prompt(
                    inputs, input_controls=[], tokenizer=tokenizer, warnings_state=warnings_state,
                )
                message_handled = frozenset()
            input_ids, attention_mask = prepare_inputs(
                input_ids, attention_mask,
                input_controls=[], tokenizer=tokenizer, device=device,
                runtime_kwargs={}, message_handled=message_handled, warnings_state=warnings_state,
            )
            substrings = [list(row["substrings"]) for row in chunk]
            token_ranges, input_len = control.locate_spans(input_ids, substrings)
            batches.append(_ProfileBatch(input_ids, attention_mask, token_ranges, input_len))
        return batches

    def _score_rows(
        self,
        control: "PASTA",
        session: "SteeringSession",
        batches: Sequence["_ProfileBatch"],
        params: GenerationParams,
        *,
        head_map: dict[int, list[int]] | None,
        rows: Sequence[Mapping],
    ) -> torch.Tensor:
        """Generate every batch through the session and score each response, in row order.

        With `head_map` None the batches generate unsteered (the baseline). Otherwise one
        `HookEntry` per batch is built from the batch's located spans and shared across the
        batch's items, so the session takes its batched generate path (identical entries).
        """
        scores: list[float] = []
        row_iter = iter(rows)
        for batch in batches:
            if head_map is None:
                state_entries: tuple = ()
            else:
                hooks = control.build_hooks_for(batch.token_ranges, batch.input_len, head_map, self.alpha)
                state_entries = (HookEntry(hooks=hooks),)
            items = [
                GenerationItem(
                    prompt=PreparedPrompt.from_token_ids(
                        batch.input_ids[i:i + 1], batch.attention_mask[i:i + 1],
                    ),
                    state_entries=state_entries,
                )
                for i in range(batch.input_ids.size(0))
            ]
            results = session.generate(items, params)
            for result in results:
                row = next(row_iter)
                text = session.tokenizer.decode(result.output.output_ids[0], skip_special_tokens=True)
                scores.append(float(self.scorer(text, row)))
        return torch.tensor(scores, dtype=torch.float32)

    def _enumerate_candidates(self, control: "PASTA") -> list[tuple[int, int]]:
        """Every `(layer, head)` candidate in `(layer, head)` order."""
        layers = control.attention_layers() if self.layers is None else [int(layer) for layer in self.layers]
        candidates: list[tuple[int, int]] = []
        for layer in sorted(set(layers)):
            for head in range(control.num_heads_of_layer(layer)):
                candidates.append((layer, head))
        return candidates

    def _screen_subset(self, groups: Sequence) -> list[int]:
        """A fixed subset of `screen_rows` row indices, stratified by group when present."""
        rng = random.Random(self.seed)
        num_rows = len(self.rows)
        indices = list(range(num_rows))
        if all(group is not None for group in groups):
            by_group: dict[str, list[int]] = {}
            for index in indices:
                by_group.setdefault(str(groups[index]), []).append(index)
            names = sorted(by_group)
            per_group = max(1, self.screen_rows // len(names))
            chosen: list[int] = []
            for name in names:
                pool = by_group[name]
                chosen.extend(rng.sample(pool, min(per_group, len(pool))))
            if len(chosen) < self.screen_rows:
                remaining = [index for index in indices if index not in set(chosen)]
                chosen.extend(rng.sample(remaining, min(self.screen_rows - len(chosen), len(remaining))))
            return sorted(chosen[: self.screen_rows])
        return sorted(rng.sample(indices, self.screen_rows))

    @staticmethod
    def _std_err(diff: torch.Tensor) -> float:
        """Standard error of the paired differences (sample std over sqrt n)."""
        count = diff.numel()
        if count < 2:
            return float("nan")
        return float(diff.std(unbiased=True) / math.sqrt(count))

    def _select(
        self,
        scored: Sequence[tuple[int, int]],
        lift: torch.Tensor,
        group_lift: torch.Tensor | None,
        group_names: Sequence[str] | None,
        group_rows: Sequence[int] | None,
    ) -> tuple[list[tuple[int, int]], int]:
        """Rank the scored candidates, filter by eligibility, and select per the mode."""
        eligible = [(layer, head) for layer, head in scored if lift[layer, head].item() > self.min_lift]
        if not eligible:
            raise ValueError(
                "HeadProfile selected no heads: no candidate beat the baseline "
                f"(min_lift={self.min_lift}). Lower min_lift, add rows, or restrict layers."
            )

        ranked = sorted(eligible, key=lambda lh: (-lift[lh[0], lh[1]].item(), lh[0], lh[1]))

        if self.selection == "pooled":
            selected = ranked[: self.num_heads]
        else:
            selected = self._intersection_select(eligible, group_lift, group_names, group_rows, ranked)

        if len(selected) < self.num_heads:
            warnings.warn(
                f"HeadProfile requested {self.num_heads} heads but only {len(selected)} candidates were "
                f"eligible (lift > {self.min_lift}); selecting the eligible set.",
                UserWarning,
            )

        cutoff_lift = lift[selected[-1][0], selected[-1][1]].item()
        tie_at_cutoff = sum(1 for layer, head in ranked if lift[layer, head].item() == cutoff_lift)
        return selected, tie_at_cutoff

    def _intersection_select(
        self,
        eligible: Sequence[tuple[int, int]],
        group_lift: torch.Tensor,
        group_names: Sequence[str],
        group_rows: Sequence[int],
        pooled_ranked: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """The paper's rule: intersection of per-group top-k sets at the smallest reaching k."""
        smallest = min(group_rows)
        if smallest < 200:
            smallest_name = group_names[group_rows.index(smallest)]
            warnings.warn(
                f"HeadProfile selection='intersection' uses group {smallest_name!r} with only {smallest} rows; "
                "the paper reports the per-group ranking as robust down to 200 samples per group.",
                UserWarning,
            )
        per_group_rank: list[list[tuple[int, int]]] = []
        for group_index in range(len(group_names)):
            ranked = sorted(
                eligible,
                key=lambda lh: (-_nan_low(group_lift[group_index, lh[0], lh[1]].item()), lh[0], lh[1]),
            )
            per_group_rank.append(ranked)

        pooled_order = {lh: i for i, lh in enumerate(pooled_ranked)}
        for k in range(1, len(eligible) + 1):
            top_sets = [set(ranked[:k]) for ranked in per_group_rank]
            intersection = set.intersection(*top_sets) if top_sets else set()
            if len(intersection) >= self.num_heads:
                ordered = sorted(intersection, key=lambda lh: pooled_order[lh])
                return ordered[: self.num_heads]
        # no k reaches num_heads: return the full final intersection, pooled-ordered
        final = set.intersection(*[set(ranked) for ranked in per_group_rank]) if per_group_rank else set()
        return sorted(final, key=lambda lh: pooled_order[lh])


def _nan_low(value: float) -> float:
    """`value`, or negative infinity when NaN, so unscored candidates sort last."""
    return float("-inf") if math.isnan(value) else value


@dataclass
class HeadProfileResult:
    """The resolution of a `HeadProfile`: the selected head map plus per-candidate statistics.

    `lift`, `se`, `n`, and `stage` are `[num_layers, max_heads]` tensors with `NaN` (0 for `n`
    and `stage`) at positions that were not candidates. `stage` is 1 for a candidate scored only
    on the screen subset and 2 for a candidate scored on every row. `group_lift` is
    `[num_groups, num_layers, max_heads]` and `group_rows` the per-group row counts, both None
    when the rows carry no `"group"`.

    Attributes:
        head_config: The selected head map, layer index to sorted head indices.
        selected: The selected candidates, in selection order.
        lift: Per-candidate mean gain over the baseline.
        se: Per-candidate standard error of the paired differences.
        n: Per-candidate number of rows scored.
        stage: Per-candidate screen stage (1 or 2), 0 for non-candidates.
        group_lift: Per-group lift, or None.
        groups: The group names, or None.
        group_rows: The per-group row counts, or None.
        baseline: The unsteered mean score over all rows.
        num_rows: The number of profiling rows.
        tie_at_cutoff: Candidates whose lift equals the last selected candidate's.
        alpha: The profiling strength.
        scale_position: The control's scale position the profile was scored under.
    """

    head_config: dict[int, list[int]]
    selected: list[tuple[int, int]]
    lift: torch.Tensor
    se: torch.Tensor
    n: torch.Tensor
    stage: torch.Tensor
    group_lift: torch.Tensor | None
    groups: list[str] | None
    group_rows: list[int] | None
    baseline: float
    num_rows: int
    tie_at_cutoff: int
    alpha: float
    scale_position: str

    def to_frame(self) -> "pandas.DataFrame":  # noqa: F821
        """One row per candidate, columns `layer`, `head`, `lift`, `se`, `n`, `stage`,
        `selected`, `rank`, plus one `lift_<group>` column per group when groups are present."""
        import pandas as pd

        selected_set = {tuple(pair) for pair in self.selected}
        rank_of = {tuple(pair): i for i, pair in enumerate(self.selected)}
        records: list[dict] = []
        num_layers, max_heads = self.lift.shape
        for layer in range(num_layers):
            for head in range(max_heads):
                if int(self.stage[layer, head]) == 0:
                    continue
                record = {
                    "layer": layer,
                    "head": head,
                    "lift": float(self.lift[layer, head]),
                    "se": float(self.se[layer, head]),
                    "n": int(self.n[layer, head]),
                    "stage": int(self.stage[layer, head]),
                    "selected": (layer, head) in selected_set,
                    "rank": rank_of.get((layer, head)),
                }
                if self.groups is not None:
                    for group_index, name in enumerate(self.groups):
                        record[f"lift_{name}"] = float(self.group_lift[group_index, layer, head])
                records.append(record)
        return pd.DataFrame(records)

    def save(self, path: str | Path) -> None:
        """Write the complete result to one JSON file (tensors as nested lists, NaN as null)."""
        payload = {
            "head_config": {str(layer): list(heads) for layer, heads in self.head_config.items()},
            "selected": [list(pair) for pair in self.selected],
            "lift": _tensor_to_json(self.lift),
            "se": _tensor_to_json(self.se),
            "n": _tensor_to_json(self.n),
            "stage": _tensor_to_json(self.stage),
            "group_lift": None if self.group_lift is None else _tensor_to_json(self.group_lift),
            "groups": self.groups,
            "group_rows": self.group_rows,
            "baseline": self.baseline,
            "num_rows": self.num_rows,
            "tie_at_cutoff": self.tie_at_cutoff,
            "alpha": self.alpha,
            "scale_position": self.scale_position,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HeadProfileResult":
        """Read a result written by `save`, reproducing every field exactly."""
        payload = json.loads(Path(path).read_text())
        return cls(
            head_config={int(layer): list(heads) for layer, heads in payload["head_config"].items()},
            selected=[tuple(pair) for pair in payload["selected"]],
            lift=_json_to_tensor(payload["lift"], torch.float32),
            se=_json_to_tensor(payload["se"], torch.float32),
            n=_json_to_tensor(payload["n"], torch.long),
            stage=_json_to_tensor(payload["stage"], torch.long),
            group_lift=None if payload["group_lift"] is None else _json_to_tensor(payload["group_lift"], torch.float32),
            groups=payload["groups"],
            group_rows=payload["group_rows"],
            baseline=payload["baseline"],
            num_rows=payload["num_rows"],
            tie_at_cutoff=payload["tie_at_cutoff"],
            alpha=payload["alpha"],
            scale_position=payload["scale_position"],
        )


def _tensor_to_json(tensor: torch.Tensor) -> list:
    """A tensor as nested lists, with NaN rendered as null."""
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        return None if isinstance(value, float) and math.isnan(value) else value

    return convert(tensor.tolist())


def _json_to_tensor(data: list, dtype: torch.dtype) -> torch.Tensor:
    """Nested lists back to a tensor, with null rendered as NaN."""
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        return float("nan") if value is None else value

    return torch.tensor(convert(data), dtype=dtype)
