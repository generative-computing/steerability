"""Construction semantics of `SteeringPipeline`.

Construction is unconditionally cheap: acquisition happens in `steer()`, preloaded objects
may be injected via `model=`/`tokenizer=`, misconfiguration fails fast at construction, and
pipelines are weak-referenceable.
"""
import weakref

import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class TestCheapConstruction:
    def test_construction_does_not_load(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
        assert pipeline.model is None
        assert pipeline.tokenizer is None
        pipeline.steer()
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None

    def test_lazy_init_is_inert(self):
        with_flag = SteeringPipeline(model_name_or_path=TINY_MODEL, lazy_init=True)
        without_flag = SteeringPipeline(model_name_or_path=TINY_MODEL)
        assert with_flag.model is None and without_flag.model is None


class TestInjection:
    def test_preloaded_objects_are_reused(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        tokenizer = wordlevel_tokenizer()
        pipeline = SteeringPipeline(model=model, tokenizer=tokenizer)
        assert pipeline.device == model.device
        pipeline.steer()
        assert pipeline.model is model
        assert pipeline.tokenizer is tokenizer


class TestMisconfiguration:
    def test_no_model_source_raises_at_construction(self):
        with pytest.raises(ValueError, match="model_name_or_path"):
            SteeringPipeline()


class TestWeakReference:
    def test_pipeline_is_weak_referenceable(self):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
        assert weakref.ref(pipeline)() is pipeline
