from __future__ import annotations

import gc
import logging
import os

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.core.internals.data import LabeledExamples
from aisteer360.algorithms.core.internals.probes.fitting import ProbeFitSpec, fit_probe
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.common.processors.value_guided import ValueGuidedProcessor
from aisteer360.algorithms.output_control.common.values.subspace_margin import (
    SubspaceMarginValue,
    load_single_file_probe,
)
from aisteer360.algorithms.output_control.sasa.args import SASAArgs
from aisteer360.utils.tokenization import ensure_pad_token

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

    1. **Subspace learning**: From a corpus of labeled positive (desired) and negative (undesired) examples, it fits
    a linear classifier in the model's own sentence-embedding space. The classifier's weight vector defines a
    subspace separating the two attribute classes.

    2. **Controlled decoding**: At every decoding step the candidate-token logits are shifted by `beta * margin`,
    where `margin` is the classifier distance of the updated context from the undesired side of the subspace.
    Sampling from the softmax of the adjusted logits (optionally with nucleus sampling) nudges generation toward
    the desired attribute while staying close to the original distribution.

    The reference paper applies SASA to detoxification; the default data loader reads the Jigsaw toxicity corpus
    from `gen_wv_data_path`. Any binary-labeled attribute works: pass in-memory positives/negatives via
    `gen_wv_data`, or a previously fitted probe via `wv_path`.

    SASA is a step-level control. `steer()` fits (or loads) a `Probe`, and `get_logits_processors()` returns
    a `ValueGuidedProcessor` over the `surviving` candidate policy whose per-candidate value is the subspace margin,
    obtained via a single same-model forward per step. The margins are softmax-normalized over the surviving set and
    added (scaled by `beta`) to the surviving logits. As a step-level control, SASA composes with other output
    controls and with a decoding driver.

    `include_in_scoring` defaults to False, since scoring under SASA costs a K-candidate model forward per
    reference position; opt in explicitly if needed.

    Args:
        beta (float): Scaling coefficient for value redistribution. Defaults to 0.0.
        wv_path (str, optional): Path to a saved probe. Defaults to None.
        gen_wv_data_path (str, optional): Path to the labeled attribute dataset used to fit the probe (defaults to
            the Jigsaw toxicity corpus layout).
        gen_wv_length (int, optional): The maximum number of samples used for preparing SASA steering if wv_path does not exist. Defaults to -1 (use all).
        gen_wv_batch_size (int, optional): The batch size used for preparing SASA steering if wv_path does not exist. Defaults to 4.

    Reference:

    - "Large Language Models can Become Strong Self-Detoxifiers"
      Ching-Yun Ko, Pin-Yu Chen, Payel Das, Youssef Mroueh, Soham Dan, Georgios Kollias, Subhajit Chaudhury,
      Tejaswini Pedapati, Luca Daniel
      [https://arxiv.org/abs/2410.03818](https://arxiv.org/abs/2410.03818)
    """
    Args = SASAArgs

    include_in_scoring: bool = False
    same_model_forwards: bool = True

    # placeholders (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    probe: Probe | None = None

    beta: float

    def steer_access(self) -> ModelAccess:
        """`ModelAccess.MODULE`; the probe fits on the live model, which is retained for the
        per-step value forwards (the generate phase is in-process)."""
        return ModelAccess.MODULE

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **__,
    ) -> PreTrainedModel:
        """Load or fit the linear probe defining the attribute subspace.

        A `wv_path` naming a directory loads a saved `Probe` artifact; a `.probe` JSON file or a
        legacy `{'wv', 'mu_mu'}` tensor checkpoint is adapted into a `Probe` over the final
        decoder layer. Without `wv_path`, a probe is fitted on the labeled data (fisher direction
        over last-token features at the raw final-layer boundary, midpoint calibration). The
        probe's recorded space is validated against the boundary the margins are evaluated at.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for the base model.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model (unchanged).

        Raises:
            ValueError: If a loaded probe's `location`, `pooling`, or layer ids do not match
                last-token features at the raw output boundary of the final decoder layer.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer = ensure_pad_token(self.tokenizer)
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        final_layer = int(model.config.num_hidden_layers) - 1
        if getattr(self, "wv_path", None):
            logger.info("Loading SASA probe.")
            if os.path.isdir(self.wv_path):
                self.probe = Probe.load(self.wv_path)
            else:
                self.probe = load_single_file_probe(self.wv_path, layer_id=final_layer)
        else:
            logger.info("Fitting SASA probe.")
            data = self._resolve_labeled_examples()
            spec = ProbeFitSpec(
                method="fisher",
                pooling="last",
                location="layer_output",
                prompt_format="raw",
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

    def _resolve_labeled_examples(self) -> LabeledExamples:
        """Resolve labeled positives/negatives from the configured data source (SASA's loader)."""
        if self.gen_wv_data is not None:
            logger.debug("Data provided in-memory.")
            return LabeledExamples(positives=self.gen_wv_data["pos"], negatives=self.gen_wv_data["neg"])

        os.makedirs(self.gen_wv_data_path, exist_ok=True)
        csv_path = os.path.join(self.gen_wv_data_path, "all_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"""
                    Jigsaw dataset not found at: {self.gen_wv_data_path}
                    To use jigsaw_unintended_bias you have to download it manually from Kaggle:
                    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
                    Extract all files into one folder and load with:
                    dataset = pd.read_csv('/tmp/Jigsaw_data/all_data.csv')
                    """
            )
        dataset = pd.read_csv(csv_path, low_memory=False)  # jigsaw csv has mixed-dtype columns
        pos = [row for i, row in dataset["comment_text"].items()
               if isinstance(row, str) and dataset["toxicity"][i] == 0]
        neg = [row for i, row in dataset["comment_text"].items()
               if isinstance(row, str) and dataset["toxicity"][i] > 0]

        num = len(pos) + len(neg)
        if 0 < self.gen_wv_length < num:
            num_pos = int(self.gen_wv_length / num * len(pos))
            num_neg = self.gen_wv_length - num_pos
            pos = pos[:num_pos]
            neg = neg[:num_neg]
        return LabeledExamples(positives=pos, negatives=neg)

    def get_logits_processors(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs) -> list:
        """Return a fresh `ValueGuidedProcessor` implementing SASA's margin-based shift.

        The candidate policy is `surviving` (every token left finite by earlier processors), matching
        SASA's use of the merged stack; margins are softmax-normalized over the surviving set and added
        (scaled by `beta`) with no non-candidate masking. The surviving set is whatever earlier
        processors leave finite; bound it with sampler kwargs (`top_k`/`top_p`) or `max_candidates` to
        cap the per-step model forward.
        """
        if self.probe is None:
            raise RuntimeError("SASA.steer() must run before generation (probe not fitted/loaded).")
        return [
            ValueGuidedProcessor(
                SubspaceMarginValue(self.probe),
                policy="surviving",
                beta=self.beta,
                normalize="softmax",
                mask_non_candidates=False,
                max_candidates=self.max_candidates,
                lm_tokenizer=self.tokenizer,
                model=self.model,
                attention_mask=attention_mask,
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
