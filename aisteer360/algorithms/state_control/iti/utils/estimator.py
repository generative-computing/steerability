"""Probe-based mass mean shift estimator for ITI."""
import logging

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.internals.data import LabeledExamples
from aisteer360.algorithms.core.internals.encoding import tokenize_texts
from aisteer360.algorithms.core.internals.pooling import get_last_token_positions, select_at_positions
from aisteer360.algorithms.state_control.common.estimators.base import BaseEstimator
from aisteer360.algorithms.state_control.common.fit_specs import VectorTrainSpec
from aisteer360.algorithms.state_control.common.model_layout import resolve_model_layout
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)


class ProbeMassShiftEstimator(BaseEstimator[SteeringVector]):
    """Learns per-head direction vectors using probe-based mass mean shift.

    For each (layer, head) pair:

    1. Extract pre-o_proj attention head outputs (individual head activations before the output projection).
    2. Train a binary logistic regression probe to classify positive vs negative samples.
    3. Record the probe's validation accuracy (80/20 split) for later head selection.
    4. Compute the mass mean shift: direction = mean(activations_true) - mean(activations_false).

    Returns a unified SteeringVector with directions shaped [num_heads, head_dim]
    per layer and probe_accuracies populated for all (layer, head) pairs.

    This estimator captures attention outputs using temporary forward pre-hooks on o_proj
    modules. The hooks are registered only during fit() and removed after extraction.

    Reference:

        - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
          Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
          [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: LabeledExamples,
        spec: VectorTrainSpec,
    ) -> SteeringVector:
        """Extract head steering vectors using probe-based mass mean shift.

        Args:
            model: Model to extract attention outputs from.
            tokenizer: Tokenizer for encoding the labeled examples.
            data: Independent positive/negative texts (true/false statements for ITI).
                Unlike ContrastivePairs, these do not need to be equal length.
            spec: Training configuration (accumulate mode and batch_size).

        Returns:
            SteeringVector with directions shaped [num_heads, head_dim] per layer,
            num_heads/head_dim metadata, and probe_accuracies for all heads.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        # some architectures define an independent head_dim (not hidden_size // num_heads)
        head_dim = getattr(model.config, "head_dim", None) or hidden_size // num_heads

        # sanity-check the per-head slicing matches the o_proj input width (else the reshape corrupts)
        layout = resolve_model_layout(model)
        o_proj = model.get_submodule(layout.oproj_names[0])
        in_features = getattr(o_proj, "in_features", None)
        if in_features is not None and num_heads * head_dim != in_features:
            raise ValueError(
                f"ITI head slicing mismatch: num_heads ({num_heads}) * head_dim ({head_dim}) = "
                f"{num_heads * head_dim} != o_proj.in_features ({in_features}). "
                f"Set model.config.head_dim to the correct per-head dimension."
            )

        pos_texts = list(data.positives)
        neg_texts = list(data.negatives)
        n_pos = len(pos_texts)
        n_neg = len(neg_texts)

        logger.debug("Tokenizing %d positive and %d negative examples", n_pos, n_neg)

        # tokenize independently (no interleaving needed for ITI)
        enc_pos = tokenize_texts(tokenizer, pos_texts, device)
        enc_neg = tokenize_texts(tokenizer, neg_texts, device)

        # extract attention outputs via temporary hooks
        logger.debug("Extracting attention outputs with batch_size=%d", spec.batch_size)
        attn_out_pos = self._extract_attention_outputs(model, enc_pos, spec.batch_size)
        attn_out_neg = self._extract_attention_outputs(model, enc_neg, spec.batch_size)

        num_layers = len(attn_out_pos)
        logger.debug("Computing probe-based directions for %d layers x %d heads", num_layers, num_heads)

        # get attention masks for position selection
        attn_mask_pos = enc_pos.get("attention_mask")
        attn_mask_neg = enc_neg.get("attention_mask")
        if attn_mask_pos is not None:
            attn_mask_pos = attn_mask_pos.cpu()
        if attn_mask_neg is not None:
            attn_mask_neg = attn_mask_neg.cpu()

        directions: dict[int, torch.Tensor] = {}
        probe_accuracies: dict[tuple[int, int], float] = {}

        for layer_id in range(num_layers):
            ap = attn_out_pos[layer_id]  # [n_pos, T_pos, H]
            an = attn_out_neg[layer_id]  # [n_neg, T_neg, H]

            # aggregate based on accumulate mode
            if spec.accumulate == "last_token":
                pos_positions = get_last_token_positions(attn_mask_pos, ap.size(1), n_pos)
                neg_positions = get_last_token_positions(attn_mask_neg, an.size(1), n_neg)
                ap_agg = select_at_positions(ap, pos_positions)  # [n_pos, H]
                an_agg = select_at_positions(an, neg_positions)  # [n_neg, H]
            elif spec.accumulate == "all":
                ap_agg = ap.mean(dim=1)  # [n_pos, H]
                an_agg = an.mean(dim=1)  # [n_neg, H]
            else:
                raise ValueError(f"ProbeMassShiftEstimator does not support accumulate='{spec.accumulate}'")

            # reshape to [N, num_heads, head_dim]
            ap_heads = ap_agg.view(n_pos, num_heads, head_dim)
            an_heads = an_agg.view(n_neg, num_heads, head_dim)

            layer_directions = []
            for head_id in range(num_heads):
                hp = ap_heads[:, head_id, :].float().numpy()  # [n_pos, head_dim]
                hn = an_heads[:, head_id, :].float().numpy()  # [n_neg, head_dim]

                # concatenate for probe training: positive=1, negative=0
                X = torch.cat([ap_heads[:, head_id, :], an_heads[:, head_id, :]], dim=0).float().numpy()
                y = [1] * n_pos + [0] * n_neg

                # train logistic regression probe with 80/20 train/val split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y,
                )
                probe = LogisticRegression(max_iter=1000, solver="lbfgs")
                probe.fit(X_train, y_train)
                accuracy = probe.score(X_val, y_val)

                # compute mass mean shift (raw direction)
                raw_direction = torch.from_numpy(hp.mean(axis=0) - hn.mean(axis=0)).to(dtype=torch.float32)

                # L2-normalize to get unit direction theta_hat
                norm = raw_direction.norm()
                if norm > 0:
                    theta_hat = raw_direction / norm
                else:
                    theta_hat = raw_direction

                # compute sigma: std of ALL activations projected onto theta_hat
                all_activations = torch.cat([ap_heads[:, head_id, :], an_heads[:, head_id, :]], dim=0).float()
                proj_vals = all_activations @ theta_hat
                sigma = proj_vals.std()

                # store sigma * theta_hat so that downstream transform applies alpha * sigma * theta_hat
                direction = theta_hat * sigma
                layer_directions.append(direction)

                probe_accuracies[(layer_id, head_id)] = accuracy

            # stack all head directions into [num_heads, head_dim]
            directions[layer_id] = torch.stack(layer_directions, dim=0)

        logger.debug("Finished fitting probe-based head directions")
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            num_heads=num_heads,
            head_dim=head_dim,
            probe_accuracies=probe_accuracies,
        )

    @torch.no_grad()
    def _extract_attention_outputs(
        self,
        model: PreTrainedModel,
        enc: dict[str, torch.Tensor],
        batch_size: int,
    ) -> dict[int, torch.Tensor]:
        """Extract pre-o_proj attention head outputs from all layers using temporary hooks.

        Registers forward pre-hooks on each layer's output projection (o_proj / c_proj)
        to capture the concatenated per-head attention outputs BEFORE the output projection
        mixes them. This matches the probing point described in the ITI paper (Section 3.1):
        activations x^h_l after Att and before Q^h_l.

        Args:
            model: The model to extract from.
            enc: Tokenized input with input_ids and attention_mask.
            batch_size: Batch size for forward passes.

        Returns:
            Dict mapping layer_id to tensor of shape [N, T, num_heads * head_dim].
        """
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        N = input_ids.size(0)

        layout = resolve_model_layout(model)
        oproj_names = layout.oproj_names
        num_layers = layout.num_layers

        storage: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_pre_hook(layer_id: int):
            def hook(_module, args, kwargs):
                # o_proj input is the first positional arg: [B, T, num_heads * head_dim]
                oproj_input = args[0] if args else kwargs.get("input")
                storage[layer_id].append(oproj_input.detach().cpu())
            return hook

        try:
            for layer_id, oproj_name in enumerate(oproj_names):
                oproj_module = model.get_submodule(oproj_name)
                handle = oproj_module.register_forward_pre_hook(make_pre_hook(layer_id), with_kwargs=True)
                handles.append(handle)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_ids = input_ids[start:end]
                batch_mask = attention_mask[start:end] if attention_mask is not None else None

                model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        result = {}
        for layer_id, tensors in storage.items():
            result[layer_id] = torch.cat(tensors, dim=0)

        return result
