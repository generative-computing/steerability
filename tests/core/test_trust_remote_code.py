"""Tests that `trust_remote_code` is opt-in (default False) across the loaders that expose it.

Each test mocks `from_pretrained` so no weights are fetched, then asserts the `trust_remote_code`
kwarg passed to the loader: False by default, True when the user opts in. CPO is the documented
exception (defaults to True for its paper-default encoder) and is covered in `tests/controls/test_cpo.py`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.prewrite import PRewrite
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer import SFT


def _mock_tokenizer() -> MagicMock:
    """A tokenizer stub whose pad token is already set, so `ensure_pad_token` is a no-op."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.device = "cpu"
    return model


class TestSteeringPipelineTrustRemoteCode:
    def _build(self, **kwargs) -> tuple[MagicMock, MagicMock]:
        """Construct a real SteeringPipeline with both loaders patched; return the two mocks."""
        with (
            patch(
                "steerability.algorithms.core.steering_pipeline.AutoModelForCausalLM"
            ) as model_cls,
            patch(
                "steerability.algorithms.core.steering_pipeline.AutoTokenizer"
            ) as tokenizer_cls,
        ):
            model_cls.from_pretrained.return_value = _mock_model()
            tokenizer_cls.from_pretrained.return_value = _mock_tokenizer()
            SteeringPipeline(model_name_or_path="some/model", **kwargs).steer()
        return model_cls, tokenizer_cls

    def test_default_false(self):
        _, tokenizer_cls = self._build()
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is False

    def test_opt_in_true(self):
        _, tokenizer_cls = self._build(trust_remote_code=True)
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True

    def test_model_load_untouched_by_flag(self):
        """The flag routes to the tokenizer only; the model opts in via `hf_model_kwargs`."""
        model_cls, _ = self._build(trust_remote_code=True)
        assert "trust_remote_code" not in model_cls.from_pretrained.call_args.kwargs


class TestPRewriteTrustRemoteCode:
    def _resolve(self, **kwargs) -> tuple[MagicMock, MagicMock]:
        """Run `_resolve_rewriter` with both loaders patched; return the two mocks."""
        prewrite = PRewrite(
            initial_instruction="be helpful",
            strategy="inference",
            rewriter_model_name_or_path="some/rewriter",
            **kwargs,
        )
        with (
            patch(
                "steerability.algorithms.input_control.prewrite.control.AutoModelForCausalLM"
            ) as model_cls,
            patch(
                "steerability.algorithms.input_control.prewrite.control.AutoTokenizer"
            ) as tokenizer_cls,
        ):
            model_cls.from_pretrained.return_value = _mock_model()
            tokenizer_cls.from_pretrained.return_value = _mock_tokenizer()
            prewrite._resolve_rewriter(task_lm=None, tokenizer=None)
        return model_cls, tokenizer_cls

    def test_default_false(self):
        model_cls, tokenizer_cls = self._resolve()
        assert model_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is False
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is False

    def test_opt_in_true(self):
        model_cls, tokenizer_cls = self._resolve(trust_remote_code=True)
        assert model_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True


class TestTRLWrapperTrustRemoteCode:
    def _resolve(self, **kwargs) -> tuple[MagicMock, MagicMock]:
        """Run `TRLMixin._resolve_model_tokenizer` (via SFT) with both loaders patched."""
        sft = SFT(base_model_name_or_path="some/base", **kwargs)
        with (
            patch(
                "steerability.algorithms.structural_control.wrappers.trl.base_mixin.AutoModelForCausalLM"
            ) as model_cls,
            patch(
                "steerability.algorithms.structural_control.wrappers.trl.base_mixin.AutoTokenizer"
            ) as tokenizer_cls,
        ):
            model = _mock_model()
            model.parameters.return_value = iter([MagicMock(device="cpu")])
            model_cls.from_pretrained.return_value = model
            tokenizer_cls.from_pretrained.return_value = _mock_tokenizer()
            sft._resolve_model_tokenizer(None, None)
        return model_cls, tokenizer_cls

    def test_default_false(self):
        model_cls, tokenizer_cls = self._resolve()
        assert model_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is False
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is False

    def test_opt_in_true(self):
        model_cls, tokenizer_cls = self._resolve(trust_remote_code=True)
        assert model_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True
        assert tokenizer_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True
