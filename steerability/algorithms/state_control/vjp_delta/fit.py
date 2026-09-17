"""VJP-delta steering-vector extraction."""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.internals.data import LabeledExamples, as_labeled_examples
from steerability.algorithms.core.internals.model_layout import resolve_model_layout, text_config
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession


def _hidden(output) -> torch.Tensor:
    """Return the residual tensor from a decoder-layer output."""
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Decoder layer returned {type(hidden).__name__}, not a tensor or tuple headed by a tensor.")
    return hidden


def _token_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize raw texts into a right-padded batch without changing tokenizer state."""
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    rows = encoded["input_ids"]
    if not rows or any(len(row) == 0 for row in rows):
        raise ValueError("VJP-delta fitting requires every example to contain at least one token after tokenization.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("VJP-delta fitting needs tokenizer.pad_token_id or tokenizer.eos_token_id for batching.")
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, :len(row)] = 1
    return input_ids, attention_mask


def _valid_token_mask(attention_mask: torch.Tensor, skip_first: int) -> torch.BoolTensor:
    """Select valid positions, excluding the prefix and each row's final real token."""
    mask = attention_mask.to(torch.bool).clone()
    lengths = mask.sum(dim=1)
    if torch.any(lengths <= skip_first + 1):
        short = torch.nonzero(lengths <= skip_first + 1, as_tuple=False).flatten().tolist()
        raise ValueError(
            "VJP-delta fitting needs a valid token span after skip_first and before the final real token; "
            f"rows {short} have lengths {lengths[short].tolist()} with skip_first={skip_first}."
        )
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    mask &= positions >= skip_first
    mask[torch.arange(mask.size(0), device=mask.device), lengths - 1] = False
    return mask


@dataclass
class VJPDeltaFit:
    """Fit normalized VJP-delta directions from independent labeled prompt pools.

    The fit reads the final unpadded state at `target_layer` to form the target contrast. It then
    applies that contrast as a cotangent at every valid target position and averages each prompt's
    valid source-position gradients before class averaging. The positive-minus-negative class mean
    is L2-normalized independently for every source layer.

    Args:
        data: Independent positive and negative raw prompts. Class sizes may differ.
        target_layer: Target residual layer. None selects `num_layers - 3`.
        source_layer_ids: Source residual layers. None selects every layer before the target.
        skip_first: Prefix positions excluded from target and source token spans.
        max_length: Maximum tokenized prompt length.
        batch_size: Number of prompts per differentiable forward.
    """

    produces_positional = False
    access = ModelAccess.MODULE
    artifact_class = "direction"

    data: LabeledExamples | dict
    target_layer: int | None = None
    source_layer_ids: Sequence[int] | None = None
    skip_first: int = 16
    max_length: int = 384
    batch_size: int = 8

    _model_ref: weakref.ref | None = field(default=None, init=False, repr=False, compare=False)
    _master: SteeringVector | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, LabeledExamples):
            self.data = as_labeled_examples(self.data)
        if self.skip_first < 0:
            raise ValueError(f"skip_first must be >= 0, got {self.skip_first}.")
        if self.max_length < 2:
            raise ValueError(f"max_length must be >= 2, got {self.max_length}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.target_layer is not None and self.target_layer < 0:
            raise ValueError(f"target_layer must be >= 0, got {self.target_layer}.")
        if self.source_layer_ids is not None:
            source_ids = tuple(int(layer_id) for layer_id in self.source_layer_ids)
            if not source_ids:
                raise ValueError("source_layer_ids must not be empty.")
            if len(set(source_ids)) != len(source_ids):
                raise ValueError("source_layer_ids must not contain duplicates.")
            if min(source_ids) < 0:
                raise ValueError("source_layer_ids must all be >= 0.")
            self.source_layer_ids = source_ids

    def _layers(self, model: PreTrainedModel) -> tuple[int, tuple[int, ...]]:
        layout = resolve_model_layout(model)
        target = layout.num_layers - 3 if self.target_layer is None else self.target_layer
        if not 0 <= target < layout.num_layers:
            raise ValueError(f"target_layer {target} is out of range for {layout.num_layers} decoder layers.")
        source_ids = tuple(range(target)) if self.source_layer_ids is None else tuple(self.source_layer_ids)
        if not source_ids:
            raise ValueError("VJP-delta fitting needs at least one source layer before target_layer.")
        invalid = [layer_id for layer_id in source_ids if not 0 <= layer_id < layout.num_layers]
        if invalid:
            raise ValueError(f"source_layer_ids {invalid} are out of range for {layout.num_layers} decoder layers.")
        after_target = [layer_id for layer_id in source_ids if layer_id >= target]
        if after_target:
            raise ValueError(
                f"VJP-delta source layers must precede target_layer {target}; got {after_target}."
            )
        return target, source_ids

    def _target_mean(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        *,
        target_layer: int,
        target_module: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the mean target-layer state at each row's final unpadded token."""
        total: torch.Tensor | None = None
        count = 0
        for start in range(0, len(texts), self.batch_size):
            input_ids, attention_mask = _token_batch(
                tokenizer, texts[start:start + self.batch_size], max_length=self.max_length, device=device,
            )
            captured: list[torch.Tensor] = []
            handle = target_module.register_forward_hook(lambda _m, _a, output: captured.append(_hidden(output)))
            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError(f"VJP-delta target hook fired {len(captured)} times; expected one forward output.")
            lengths = attention_mask.sum(dim=1) - 1
            rows = captured[0][torch.arange(input_ids.size(0), device=device), lengths].float()
            total = rows.sum(dim=0) if total is None else total + rows.sum(dim=0)
            count += rows.size(0)
        if total is None or count == 0:
            raise ValueError("VJP-delta fitting needs at least one prompt in each class.")
        return total / count

    def _class_gradients(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        *,
        target_module: torch.nn.Module,
        source_modules: dict[int, torch.nn.Module],
        cotangent: torch.Tensor,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Return independent per-class means of per-prompt source-token VJPs."""
        totals: dict[int, torch.Tensor] = {}
        count = 0
        for start in range(0, len(texts), self.batch_size):
            input_ids, attention_mask = _token_batch(
                tokenizer, texts[start:start + self.batch_size], max_length=self.max_length, device=device,
            )
            valid = _valid_token_mask(attention_mask, self.skip_first)
            source_states: dict[int, torch.Tensor] = {}
            target_states: list[torch.Tensor] = []
            def source_capture(layer_id: int):
                def hook(_module, _args, output):
                    source_states[layer_id] = _hidden(output)
                    return None

                return hook

            def target_capture(_module, _args, output):
                target_states.append(_hidden(output))
                return None

            handles = [
                module.register_forward_hook(source_capture(layer_id))
                for layer_id, module in source_modules.items()
            ]
            handles.append(target_module.register_forward_hook(target_capture))
            try:
                model(input_ids=input_ids, attention_mask=attention_mask)
                if len(target_states) != 1 or set(source_states) != set(source_modules):
                    raise RuntimeError("VJP-delta extraction hooks did not capture one target and every source layer.")
                target = target_states[0]
                if not target.requires_grad or any(not state.requires_grad for state in source_states.values()):
                    raise RuntimeError(
                        "VJP-delta fitting needs differentiable decoder-layer outputs. "
                        "Use a standard torch attention implementation without inference_mode or no-grad wrappers."
                    )
                grad_outputs = torch.zeros_like(target)
                grad_outputs[valid] = cotangent.to(dtype=target.dtype, device=target.device)
                gradients = torch.autograd.grad(
                    outputs=target,
                    inputs=tuple(source_states[layer_id] for layer_id in source_modules),
                    grad_outputs=grad_outputs,
                    allow_unused=False,
                )
            except RuntimeError as error:
                if "VJP-delta" in str(error):
                    raise
                raise RuntimeError(
                    "VJP-delta backward failed. The configured model must support gradients through its decoder layers."
                ) from error
            finally:
                for handle in handles:
                    handle.remove()
            for layer_id, gradient in zip(source_modules, gradients, strict=True):
                per_prompt = (
                    gradient.float() * valid.unsqueeze(-1).to(dtype=gradient.dtype)
                ).sum(dim=1) / valid.sum(dim=1, keepdim=True)
                summed = per_prompt.sum(dim=0)
                totals[layer_id] = summed if layer_id not in totals else totals[layer_id] + summed
            count += input_ids.size(0)
        return {layer_id: total / count for layer_id, total in totals.items()}

    def _fit(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringVector:
        if model is None:
            raise ValueError("VJP-delta fitting requires a live model at steer time.")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("VJP-delta fitting cannot run while torch.inference_mode() is enabled.")
        target_layer, source_ids = self._layers(model)
        layout = resolve_model_layout(model)
        target_module = model.get_submodule(layout.layer_names[target_layer])
        source_modules = {layer_id: model.get_submodule(layout.layer_names[layer_id]) for layer_id in source_ids}
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("VJP-delta fitting requires a model with parameters.") from error
        if device.type == "meta":
            raise ValueError("VJP-delta fitting requires materialized model parameters, not meta tensors.")

        parameters = tuple(model.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        parameter_grads = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ]
        was_training = model.training
        try:
            model.eval()
            for parameter in parameters:
                parameter.requires_grad_(True)
            with torch.enable_grad():
                positive_target = self._target_mean(
                    model, tokenizer, self.data.positives, target_layer=target_layer,
                    target_module=target_module, device=device,
                )
                negative_target = self._target_mean(
                    model, tokenizer, self.data.negatives, target_layer=target_layer,
                    target_module=target_module, device=device,
                )
                cotangent = positive_target - negative_target
                if not torch.isfinite(cotangent).all() or cotangent.norm() == 0:
                    raise ValueError("VJP-delta target contrast must have a finite, nonzero norm.")
                positive = self._class_gradients(
                    model, tokenizer, self.data.positives, target_module=target_module,
                    source_modules=source_modules, cotangent=cotangent, device=device,
                )
                negative = self._class_gradients(
                    model, tokenizer, self.data.negatives, target_module=target_module,
                    source_modules=source_modules, cotangent=cotangent, device=device,
                )
        finally:
            model.train(was_training)
            for parameter, flag, grad in zip(parameters, requires_grad, parameter_grads, strict=True):
                parameter.requires_grad_(flag)
                parameter.grad = grad

        directions: dict[int, torch.Tensor] = {}
        for layer_id in source_ids:
            direction = positive[layer_id] - negative[layer_id]
            norm = direction.norm()
            if not torch.isfinite(norm) or norm == 0:
                raise ValueError(f"VJP-delta direction at source layer {layer_id} must have a finite, nonzero norm.")
            directions[layer_id] = (direction / norm).detach().cpu().unsqueeze(0)
        return SteeringVector(
            model_type=text_config(model).model_type,
            directions=directions,
            meta={
                "method": "vjp_delta",
                "target_layer": target_layer,
                "source_layer_ids": list(source_ids),
                "skip_first": self.skip_first,
                "max_length": self.max_length,
            },
        )

    def resolve(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        session: "SteeringSession | None" = None,
    ) -> SteeringVector:
        """Fit once per model identity and return a defensive clone."""
        del session
        if self._model_ref is not None and self._model_ref() is model and self._master is not None:
            return self._master.clone()
        master = self._fit(model, tokenizer)
        self._model_ref = weakref.ref(model)
        self._master = master
        return master.clone()
