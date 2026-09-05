from __future__ import annotations

import gc
import logging
import os

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.internals.data import ContrastivePairs, LabeledExamples
from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec, fit_probe
from steerability.algorithms.core.internals.probes.probe import Probe
from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.common.processors.value_guided import ValueGuidedProcessor
from steerability.algorithms.output_control.common.values.subspace_margin import (
    SubspaceMarginValue,
    load_single_file_probe,
)
from steerability.algorithms.output_control.sasa.args import SASAArgs
from steerability.utils.tokenization import ensure_pad_token

logger = logging.getLogger(__name__)


def _validate_probe_space(probe: Probe, final_layer: int) -> None:
    """Raise unless the probe is fitted in the space the margins are evaluated in.

    Margins are evaluated on last-token hidden states at the raw output boundary of the final
    decoder layer, so the probe must record `location="layer_output"`, `pooling="last"`, and
    exactly the final decoder layer.
    """
    if probe.location != "layer_output":
        raise ValueError(
            f"SASA requires a probe fitted at location 'layer_output', got {probe.location!r}; "
            "margins are evaluated at the raw output boundary of the final decoder layer."
        )
    if probe.pooling != "last":
        raise ValueError(
            f"SASA requires a probe with pooling 'last', got {probe.pooling!r}; margins are "
            "evaluated at the candidate token position."
        )
    if list(probe.layer_ids) != [final_layer]:
        raise ValueError(
            f"SASA requires a probe over exactly the final decoder layer [{final_layer}], got "
            f"layer_ids {list(probe.layer_ids)}."
        )


class SASA(OutputControl):
    """Implementation of SASA (Self-disciplined autoregressive sampling) from Ko et al., 2024.

    SASA steers generation toward a target attribute defined by labeled examples. It works in two phases:

    1. **Subspace learning**: From labeled positive (desired) and negative (undesired) examples, it fits a linear
    classifier in the model's own final-layer space. The classifier's weight vector defines a subspace separating
    the two attribute classes. Data is passed via `gen_wv_data` as unpaired classes (a `{'pos', 'neg'}` dict or a
    `LabeledExamples`) or as paired prompt/response data (a dict carrying a `'prompts'` key or a `ContrastivePairs`),
    and `prompt_format` selects how each example is rendered before capture. `prompt_format='chat_completion'`
    renders each pair as a user turn plus the response and requires the paired form (the paper's
    `{prompt, response, annotation}` format); `'raw'` and `'chat_prompt'` accept unpaired classes.

    2. **Controlled decoding**: At every decoding step the candidate-token logits are shifted by `beta * margin`,
    where `margin` is the classifier distance of the updated context from the undesired side of the subspace.
    Sampling from the softmax of the adjusted logits nudges generation toward the desired attribute while staying
    close to the original distribution.

    Any binary-labeled attribute works, since the attribute is defined by the labels on `gen_wv_data`. A previously
    fitted probe is loaded via `wv_path`.

    SASA is a step-level control. `steer()` fits (or loads) a `Probe`, and `get_logits_processors()` returns a
    `ValueGuidedProcessor` whose per-candidate value is the subspace margin, obtained via a single same-model
    forward per step. `candidate_policy` selects which tokens are scored: `'surviving'` (every token earlier
    processors left finite), `'top_p'` (the nucleus of the raw logits, the paper's setting), or `'top_k'`. The
    margins are softmax-normalized over the candidate set and added (scaled by `beta`) to the candidate logits with
    no non-candidate masking. `max_candidates` clamps the set on top of any policy to bound the per-step forward. As
    a step-level control, SASA composes with other output controls and with a decoding driver.

    A `value_trace` list passed via `runtime_kwargs` receives one `ValueStepRecord` per scored step (the candidate
    ids, their pre-shift scores, the raw margins, and the softmax-normalized shift), for inspecting the per-step
    redistribution.

    `include_in_scoring` defaults to False, since scoring under SASA costs a K-candidate model forward per
    reference position; opt in explicitly if needed.

    Reference:

    - "Large Language Models can Become Strong Self-Detoxifiers"
      Ching-Yun Ko, Pin-Yu Chen, Payel Das, Youssef Mroueh, Soham Dan, Georgios Kollias, Subhajit Chaudhury,
      Tejaswini Pedapati, Luca Daniel
      [https://arxiv.org/abs/2410.03818](https://arxiv.org/abs/2410.03818)
    """
    Args = SASAArgs

    RUNTIME_KWARGS_SCHEMA: list[dict] = [
        {
            "name": "value_trace",
            "type": "list",
            "scope": "call",
            "help": "A caller-owned list that receives one ValueStepRecord per scored step of this call.",
        },
    ]

    include_in_scoring: bool = False
    same_model_forwards: bool = True

    # placeholders (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    probe: Probe | None = None

    beta: float

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the probe fits on the live model, which is retained for the
        per-step value forwards (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def export_state(self) -> dict:
        """The fitted value-subspace probe under the `"probe"` key (after `steer()`)."""
        return {"probe": self.probe} if self.probe is not None else {}

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        """A same-class frozen form: `wv_path` points at the exported probe directory and the
        fit-only `gen_wv_*` fields are dropped. The decode-time settings (`candidate_policy`,
        `top_p`, `top_k`, `max_candidates`) are carried forward."""
        from steerability.spipe.codec import AsPath

        return "output_control/sasa", {
            "beta": self.beta,
            "wv_path": AsPath(state["probe"]),
            "gen_wv_data": None,
            "candidate_policy": self.candidate_policy,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_candidates": self.max_candidates,
        }

    def fit_identity(self):
        """The probe-fitting inputs (`gen_wv_*` fields and `prompt_format`), or None when a saved
        probe is loaded."""
        if getattr(self, "wv_path", None):
            return None
        return {
            "gen_wv_data": self.gen_wv_data,
            "prompt_format": self.prompt_format,
            "gen_wv_length": self.gen_wv_length,
            "gen_wv_batch_size": self.gen_wv_batch_size,
        }

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase | None = None,
            **__,
    ) -> PreTrainedModel:
        """Load or fit the linear probe defining the attribute subspace.

        A `wv_path` naming a directory loads a saved `Probe` artifact; a `.probe` JSON file or a
        legacy `{'wv', 'mu_mu'}` tensor checkpoint is adapted into a `Probe` over the final
        decoder layer. Without `wv_path`, a probe is fitted on `gen_wv_data` (fisher direction over
        last-token features at the raw final-layer boundary, rendered per `prompt_format`, midpoint
        calibration). The probe's recorded space is validated against the boundary the margins are
        evaluated at.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizerBase | None): Tokenizer for the base model.

        Returns:
            PreTrainedModel: The input model (unchanged).

        Raises:
            ValueError: If neither `wv_path` nor `gen_wv_data` is set, or a loaded probe's
                `location`, `pooling`, or layer ids do not match last-token features at the raw
                output boundary of the final decoder layer.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer = ensure_pad_token(self.tokenizer)
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        final_layer = resolve_model_layout(model).num_layers - 1
        if getattr(self, "wv_path", None):
            logger.info("Loading SASA probe.")
            if os.path.isdir(self.wv_path):
                self.probe = Probe.load(self.wv_path)
            else:
                self.probe = load_single_file_probe(self.wv_path, layer_id=final_layer)
        else:
            if self.gen_wv_data is None:
                raise ValueError("SASA.steer() requires either gen_wv_data (to fit a probe) or wv_path (to load one).")
            logger.info("Fitting SASA probe.")
            data = self._resolve_fit_data()
            spec = ProbeFitSpec(
                method="fisher",
                pooling="last",
                location="layer_output",
                prompt_format=self.prompt_format,
                candidate_layers=[final_layer],
                calibration="midpoint",
            )
            self.probe = fit_probe(
                model,
                self.tokenizer,
                data=data,
                spec=spec,
                batch_size=self.gen_wv_batch_size,
                max_length=1024,
            )
        _validate_probe_space(self.probe, final_layer)
        return model

    def _resolve_fit_data(self) -> LabeledExamples | ContrastivePairs:
        """Normalize `gen_wv_data` into a fit-data container and truncate each class to `gen_wv_length`.

        A `{'pos', 'neg'}` dict becomes `LabeledExamples`; a dict carrying a `'prompts'` key becomes
        `ContrastivePairs`; a `LabeledExamples` or `ContrastivePairs` instance passes through. When
        `0 < gen_wv_length`, each class (and the aligned prompts for paired data) is truncated to at
        most `gen_wv_length` entries.
        """
        data = self.gen_wv_data
        if isinstance(data, dict):
            if "prompts" in data:
                data = ContrastivePairs(
                    positives=data["pos"], negatives=data["neg"], prompts=data["prompts"]
                )
            else:
                data = LabeledExamples(positives=data["pos"], negatives=data["neg"])

        limit = self.gen_wv_length
        if limit is None or limit <= 0:
            return data

        if isinstance(data, ContrastivePairs):
            prompts = None if data.prompts is None else list(data.prompts[:limit])
            return ContrastivePairs(
                positives=list(data.positives[:limit]),
                negatives=list(data.negatives[:limit]),
                prompts=prompts,
            )
        return LabeledExamples(
            positives=list(data.positives[:limit]), negatives=list(data.negatives[:limit])
        )

    def get_logits_processors(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs) -> list:
        """Return a fresh `ValueGuidedProcessor` implementing SASA's margin-based shift.

        The candidate set follows `candidate_policy`: `surviving` (every token left finite by earlier
        processors), `top_p` (the nucleus of the raw logits, the paper's setting), or `top_k`. Margins
        are softmax-normalized over the candidate set and added (scaled by `beta`) with no
        non-candidate masking. `max_candidates` clamps the set on top of the policy to bound the
        per-step model forward. A `value_trace` list in `runtime_kwargs` receives one `ValueStepRecord`
        per step.
        """
        if self.probe is None:
            raise RuntimeError("SASA.steer() must run before generation (probe not fitted/loaded).")
        return [
            ValueGuidedProcessor(
                SubspaceMarginValue(self.probe),
                policy=self.candidate_policy,
                p=self.top_p,
                k=self.top_k,
                beta=self.beta,
                normalize="softmax",
                mask_non_candidates=False,
                max_candidates=self.max_candidates,
                lm_tokenizer=self.tokenizer,
                model=self.model,
                attention_mask=attention_mask,
                trace=(runtime_kwargs or {}).get("value_trace"),
            )
        ]

    def cleanup(self) -> None:
        """Release the probe and model references."""
        self.probe = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("SASA cleanup completed")
