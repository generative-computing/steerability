"""Train a prefix-scored reward model, the recipe RAD's cached path needs (Deng and Raffel, 2023, §2.1).

RAD scores the reward model at every prefix during decoding, so the reward model must be trained to
predict the sequence-level attribute from any prefix, not only the complete text. This module fits a
scalar-head sequence classifier on a causal-LM backbone with the paper's cumulative prefix loss: the
squared error between the per-prefix prediction and the sequence label, weighted by prefix length and
averaged so a full-length sequence and a short one contribute comparably.

The label is the dataset's native attribute (for detoxification the `civil_comments` toxicity score in
`[0, 1]`), so the trained head predicts toxicity; RAD's `invert=True` turns the prediction into a
reward that steers away from it. Training is a plain PyTorch loop under bf16 autocast, with no
`Trainer` or TRL reward trainer (those implement pairwise preference losses, not this regression).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from steerability.algorithms.output_control.common.granite_heads import sequence_classifier_class

logger = logging.getLogger(__name__)

# the fresh scalar head is scaled down so its initial logits sit near zero; without this the head's
# default init produces large-magnitude logits over the backbone's hidden states, the sigmoid saturates,
# and its gradient vanishes before training moves the head off its starting point.
HEAD_INIT_SCALE = 0.01


@dataclass
class PrefixRewardTrainSpec:
    """Training settings for a prefix-scored reward model.

    Attributes:
        max_length: Maximum token length per training example (right-padded).
        batch_size: Examples per optimizer step.
        epochs: Passes over the data.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        seed: Seed for shuffling and initialization of the fresh scalar head.
        log_every: Log the running loss every this many optimizer steps.
    """

    max_length: int = 64
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    seed: int = 0
    log_every: int = 50


def prefix_reward_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """The cumulative squared-error loss over prefixes, averaged over the batch.

    For one sequence of length `l` with prediction `r_t` at every prefix `t` (1-indexed) and label
    `r`, the loss is `sum_t t * (r_t - r)^2 / S_l` with `S_l = l * (l + 1) / 2`. Positions with
    `attention_mask == 0` are excluded and `l` counts the included positions, so a fully masked row
    contributes nothing. The loss is computed in float32 regardless of the input dtypes.

    Args:
        predictions: Per-position predictions `[B, T]`.
        labels: Per-sequence labels `[B]`.
        attention_mask: `[B, T]` mask; `0` positions are excluded.

    Returns:
        A float32 scalar tensor, the mean per-sequence prefix loss over the non-empty rows.
    """
    predictions = predictions.float()
    mask = attention_mask.to(torch.float32)  # [B, T]
    positions = torch.cumsum(mask, dim=1) * mask  # 1-indexed prefix length at each kept position
    lengths = mask.sum(dim=1)  # [B]
    normalizer = lengths * (lengths + 1) / 2  # S_l per row

    squared_error = (predictions - labels.unsqueeze(1).float()) ** 2  # [B, T]
    per_row = (positions * squared_error).sum(dim=1) / normalizer.clamp_min(1.0)  # [B]

    nonempty = lengths > 0
    if not torch.any(nonempty):
        return per_row.sum() * 0.0
    return per_row[nonempty].mean()


def prefix_rewards(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-position rewards `[B, T]` from a sequence-classification model with a scalar head.

    Runs the backbone (`getattr(model, model.base_model_prefix)`) once, applies `model.score` at every
    position rather than only the pooled last token, and returns the sigmoid. Training and the
    prefix-tracking diagnostics use this instead of the classifier's own `forward`, which pools the
    last token only.

    Args:
        model: An `AutoModelForSequenceClassification` with a single-logit `score` head.
        input_ids: Token ids `[B, T]`.
        attention_mask: `[B, T]` mask forwarded to the backbone.

    Returns:
        Per-position rewards `[B, T]` in `[0, 1]`.
    """
    backbone = getattr(model, model.base_model_prefix)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, H]
    logits = model.score(hidden).squeeze(-1)  # [B, T]
    return torch.sigmoid(logits)


def train_prefix_reward_model(
    backbone_name_or_path: str,
    texts: Sequence[str],
    labels: Sequence[float],
    output_dir: str | Path,
    *,
    spec: PrefixRewardTrainSpec | None = None,
    device: str | torch.device | None = None,
) -> Path:
    """Fine-tune a scalar-head sequence classifier on `texts` with `prefix_reward_loss` and save it.

    Builds the classifier from the causal-LM backbone with `num_labels=1` (the head class is selected
    by `sequence_classifier_class`, so a Granite backbone resolves), scales the fresh scalar head down
    by `HEAD_INIT_SCALE` so its initial sigmoid output is unsaturated, right-pads with the backbone
    tokenizer (pad falls back to eos), trains with AdamW under bf16 autocast and a linear decay
    schedule, and writes the model and tokenizer to `output_dir`. The saved `config.json` records
    `num_labels: 1` and the head's architecture name, so `load_sequence_classifier(output_dir)`
    reloads it.

    Labels are the dataset's native attribute (for `civil_comments`, toxicity in `[0, 1]`); the head
    predicts the attribute, and RAD's `invert=True` turns the prediction into a reward.

    Args:
        backbone_name_or_path: HF id or local path of the causal-LM backbone.
        texts: Training texts.
        labels: Per-text labels in `[0, 1]`, aligned with `texts`.
        output_dir: Directory the trained model and tokenizer are written to.
        spec: Training settings; defaults to `PrefixRewardTrainSpec()`.
        device: Device to train on; defaults to CUDA when available, else CPU.

    Returns:
        `output_dir` as a `Path`.
    """
    spec = spec or PrefixRewardTrainSpec()
    output_dir = Path(output_dir)
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec.seed)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {"num_labels": 1, "dtype": torch.float32}
    model = sequence_classifier_class(backbone_name_or_path, **kwargs).from_pretrained(backbone_name_or_path, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    with torch.no_grad():
        model.score.weight.mul_(HEAD_INIT_SCALE)
    model = model.to(device)
    model.train()

    texts = list(texts)
    labels = torch.tensor(list(labels), dtype=torch.float32)
    num_examples = len(texts)
    steps_per_epoch = (num_examples + spec.batch_size - 1) // spec.batch_size
    total_steps = steps_per_epoch * spec.epochs

    optimizer = AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(total_steps, 1))
    use_bf16 = device.type == "cuda"

    generator = torch.Generator().manual_seed(spec.seed)
    step = 0
    for epoch in range(spec.epochs):
        order = torch.randperm(num_examples, generator=generator)
        for start in range(0, num_examples, spec.batch_size):
            batch_index = order[start:start + spec.batch_size]
            batch_texts = [texts[i] for i in batch_index.tolist()]
            batch_labels = labels[batch_index].to(device)
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=spec.max_length,
            ).to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                predictions = prefix_rewards(model, encoded["input_ids"], encoded["attention_mask"])
                loss = prefix_reward_loss(predictions, batch_labels, encoded["attention_mask"])
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % spec.log_every == 0:
                logger.info("prefix-reward training epoch %d step %d/%d loss %.4f",
                            epoch, step, total_steps, loss.item())

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("saved prefix-reward model to %s", output_dir)
    return output_dir
