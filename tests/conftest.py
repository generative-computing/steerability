"""
Shared fixtures and mock components for tests.

This module provides:

- Device and model fixtures for integration tests
- Recording mock controls for each category, subclassing the package base classes
- Mock metrics and a mock use case, subclassing the package base classes
- Mock model and tokenizer factories for isolating tests from Hugging Face loading
- Common evaluation-data fixtures

Mock controls record the calls the pipeline makes into them (call counts, received
`runtime_kwargs`) while inheriting construction, validation, and lifecycle behavior from the
real base classes. Only the model and tokenizer boundaries are replaced with `MagicMock`s.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.execution.access import ModelAccess
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.evaluation.metrics.base import Metric
from aisteer360.evaluation.use_cases.base import UseCase
from tests.utils.load_ci_models import get_models

# Real Model/Device Fixtures (for integration tests)
MODELS = get_models()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    """Parametrized device fixture for testing across CPU/CUDA/MPS."""
    name = request.param
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if name == "mps":
        has_mps = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
        )
        if not has_mps:
            pytest.skip("MPS not available.")
    return torch.device(name)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(repo, id=tag)
        for tag, repo in MODELS.items()
    ],
)
def model_and_tokenizer(request):
    """
    Loads each model once per test session.
    """
    model_id: str = request.param
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load {model_id}: {exc}")

    # ensure padding token exists for batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:  # edge case
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return model, tokenizer


# Mock Controls (recording subclasses of the package base classes)
@dataclass
class MockInputArgs(BaseArgs):
    """Arguments for `MockInputControl`."""
    prefix: str = ""
    suffix: str = ""
    num_examples: int = 0


class MockInputControl(InputControl):
    """Recording input control.

    `adapt` returns `input_ids` unchanged and records the call count and the `runtime_kwargs`
    it received. `steer` stores the model and tokenizer references on the instance.

    Attributes:
        _adapt_call_count: Number of `adapt` invocations since construction.
        _runtime_kwargs_received: The `runtime_kwargs` from the most recent `adapt` call.
    """
    Args = MockInputArgs
    supports_batching: bool = False
    tokenizer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapt_call_count = 0
        self._runtime_kwargs_received = None

    def adapt(self, input_ids, runtime_kwargs=None):
        self._adapt_call_count += 1
        self._runtime_kwargs_received = runtime_kwargs
        return input_ids

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model=None, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer


@dataclass
class MockStructuralArgs(BaseArgs):
    """Arguments for `MockStructuralControl`."""
    learning_rate: float = 1e-4
    num_epochs: int = 1
    output_dir: str = "./output"


class MockStructuralControl(StructuralControl):
    """Recording structural control.

    `steer` records that it was called, stores the model and tokenizer references, and returns
    the model unchanged.

    Attributes:
        _steer_called: Whether `steer` has been invoked.
    """
    Args = MockStructuralArgs
    supports_batching: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steer_called = False

    def steer(self, model, tokenizer=None, **kwargs):
        self._steer_called = True
        self.model = model
        self.tokenizer = tokenizer
        return model


@dataclass
class MockStateArgs(BaseArgs):
    """Arguments for `MockStateControl`."""
    target_layers: list = field(default_factory=lambda: [0, 1])
    scale_factor: float = 1.0
    mode: str = "add"


class MockStateControl(StateControl):
    """Recording state control.

    `get_hooks` records the call and its `runtime_kwargs`, then returns one no-op forward
    pre-hook per entry in `target_layers`, addressed at `model.layers.<layer>`. The module
    paths resolve on models with a Llama-style layout (and on `MagicMock` models, whose
    `get_submodule` returns a mock).

    Attributes:
        _hooks_created: Whether `get_hooks` has been invoked.
        _runtime_kwargs_received: The `runtime_kwargs` from the most recent `get_hooks` call.
    """
    Args = MockStateArgs
    supports_batching: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hooks_created = False
        self._runtime_kwargs_received = None

    @staticmethod
    def _noop_pre_hook(module, args, kwargs):
        return None

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs):
        self._hooks_created = True
        self._runtime_kwargs_received = runtime_kwargs

        hooks = {"pre": [], "forward": [], "backward": []}
        for layer in self.target_layers:
            hooks["pre"].append({
                "module": f"model.layers.{layer}",
                "hook_func": self._noop_pre_hook,
            })
        return hooks

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = getattr(model, "device", torch.device("cpu"))


@dataclass
class MockOutputArgs(BaseArgs):
    """Arguments for `MockOutputControl`."""
    temperature: float = 1.0
    top_k: int = 50
    constraint_type: str = "none"


class MockOutputControl(OutputControl):
    """Recording step-level output control.

    `get_logits_processors` records the call and the `runtime_kwargs` it received, then
    returns a single identity processor so the decoding driver still produces output.

    Attributes:
        _processors_requested: Whether `get_logits_processors` has been invoked.
        _runtime_kwargs_received: The `runtime_kwargs` from the most recent
            `get_logits_processors` call.
    """
    Args = MockOutputArgs
    supports_batching: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processors_requested = False
        self._runtime_kwargs_received = None

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        self._processors_requested = True
        self._runtime_kwargs_received = runtime_kwargs

        def _identity(prefix_ids, scores):
            return scores

        return [_identity]

    def steer_access(self) -> ModelAccess:
        return ModelAccess.MODULE

    def steer(self, model, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer


# Mock Metrics
class MockAccuracyMetric(Metric):
    """Exact-match accuracy over responses and reference answers."""

    def compute(
            self,
            responses: list[str],
            reference_answers: list[str] = None,
            **kwargs
    ) -> dict[str, float]:
        if reference_answers is None:
            return {"accuracy": 0.0}

        correct = sum(1 for r, ref in zip(responses, reference_answers) if r == ref)
        accuracy = correct / len(responses) if responses else 0.0
        return {"accuracy": accuracy}


class MockScoreMetric(Metric):
    """Metric that returns a fixed score."""

    def __init__(self, fixed_score: float = 0.5, **extras):
        super().__init__(**extras)
        self.fixed_score = fixed_score

    def compute(self, responses: list[str], **kwargs) -> dict[str, float]:
        return {"score": self.fixed_score}


class MockPerSampleMetric(Metric):
    """Metric that returns per-sample scores and their mean."""

    def compute(self, responses: list[str], **kwargs) -> dict[str, Any]:
        scores = [0.5 + 0.1 * i for i in range(len(responses))]
        return {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "scores": scores,
        }


# Mock UseCase
class MockUseCase(UseCase):
    """Recording use case.

    `generate` returns one canned generation per evaluation item and records its call
    arguments; `evaluate` applies each configured metric to the generations; `export` writes
    the profiles to `profiles.json` under `save_dir`.

    Attributes:
        _generate_calls: One entry per `generate` invocation, holding the received arguments.
        _evaluate_calls: One entry per `evaluate` invocation, holding the received generations.
    """

    def __init__(
            self,
            evaluation_data: list[dict],
            evaluation_metrics: list[Metric],
            num_samples: int = -1,
            **kwargs
    ):
        super().__init__(
            evaluation_data=evaluation_data,
            evaluation_metrics=evaluation_metrics,
            num_samples=num_samples,
            **kwargs,
        )
        self._generate_calls = []
        self._evaluate_calls = []

    def generate(
            self,
            model_or_pipeline,
            tokenizer,
            gen_kwargs=None,
            runtime_overrides: dict | None = None,
            **kwargs
    ) -> list[dict[str, Any]]:
        self._generate_calls.append({
            "model_or_pipeline": model_or_pipeline,
            "tokenizer": tokenizer,
            "gen_kwargs": gen_kwargs,
            "runtime_overrides": runtime_overrides,
            "kwargs": kwargs,
        })

        generations = []
        for item in self.evaluation_data:
            generations.append({
                "response": "A",
                "prompt": item.get("question", item.get("prompt", "test prompt")),
                "question_id": item.get("id", "test_id"),
                "reference_answer": item.get("answer", "A"),
            })
        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self._evaluate_calls.append(generations)

        eval_data = {
            "responses": [g["response"] for g in generations],
            "reference_answers": [g["reference_answer"] for g in generations],
            "question_ids": [g["question_id"] for g in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)

        return scores

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        with open(Path(save_dir) / "profiles.json", "w") as f:
            json.dump(profiles, f, indent=4)


# Mock Model/Tokenizer Factories
def create_mock_model(device: str = "cpu") -> MagicMock:
    """Create a mock causal language model.

    The mock exposes `device`, a config with `num_attention_heads`, `num_hidden_layers`,
    `is_encoder_decoder`, and `vocab_size`, a `generate` that appends `max_new_tokens` random
    token ids to the prompt, and a forward call that returns random logits of shape
    `[batch, seq_len, vocab_size]`.
    """
    model = MagicMock()
    model.device = torch.device(device)
    model.config = MagicMock()
    model.config.num_attention_heads = 8
    model.config.num_hidden_layers = 12
    model.config.is_encoder_decoder = False
    model.config.vocab_size = 1000

    def mock_generate(input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        new_tokens = kwargs.get("max_new_tokens", 10)
        return torch.randint(0, 1000, (batch_size, seq_len + new_tokens))

    model.generate = MagicMock(side_effect=mock_generate)

    def mock_forward(*args, input_ids=None, attention_mask=None, **kwargs):
        if args and input_ids is None:
            input_ids = args[0]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = model.config.vocab_size

        outputs = MagicMock()
        outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        return outputs

    model.side_effect = mock_forward

    model.parameters = MagicMock(return_value=iter([torch.tensor([1.0])]))
    return model


def create_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer.

    The mock has pad and eos tokens configured, tokenizes any text batch to random ids of
    shape `[batch, 10]` with an all-ones attention mask, and decodes to the fixed string
    `"decoded text"`.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.padding_side = "left"

    def mock_call(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
        }

    tokenizer.side_effect = mock_call
    tokenizer.batch_decode = MagicMock(return_value=["decoded text"])
    tokenizer.decode = MagicMock(return_value="decoded text")
    return tokenizer


# Common Test Data Fixtures
@pytest.fixture
def sample_evaluation_data() -> list[dict]:
    """Sample evaluation data for testing."""
    return [
        {"id": "q1", "question": "What is 2+2?", "answer": "A", "choices": ["4", "5", "6", "7"]},
        {"id": "q2", "question": "Capital of France?", "answer": "B", "choices": ["London", "Paris", "Berlin"]},
        {"id": "q3", "question": "Closest planet to sun?", "answer": "A", "choices": ["Mercury", "Venus", "Earth"]},
    ]


@pytest.fixture
def large_evaluation_data() -> list[dict]:
    """Larger evaluation dataset for testing."""
    return [
        {"id": f"q{i}", "question": f"Question {i}?", "answer": "A", "choices": ["A", "B", "C", "D"]}
        for i in range(100)
    ]


@pytest.fixture
def evaluation_data_with_metadata() -> list[dict]:
    """Evaluation data with additional metadata fields."""
    return [
        {
            "id": "q1",
            "question": "Test question",
            "answer": "A",
            "instructions": ["instruction1", "instruction2"],
            "context": "Some context",
            "metadata": {"source": "test"},
        },
    ]


@pytest.fixture
def sample_metrics() -> list[Metric]:
    """Sample metrics for testing."""
    return [MockAccuracyMetric(), MockScoreMetric()]


@pytest.fixture
def sample_use_case(sample_evaluation_data, sample_metrics) -> MockUseCase:
    """Sample use case for testing."""
    return MockUseCase(
        evaluation_data=sample_evaluation_data,
        evaluation_metrics=sample_metrics,
    )


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model fixture."""
    return create_mock_model()


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer fixture."""
    return create_mock_tokenizer()


@pytest.fixture
def mock_input_control() -> MockInputControl:
    """Mock input control fixture."""
    return MockInputControl(prefix="test_", num_examples=3)


@pytest.fixture
def mock_structural_control() -> MockStructuralControl:
    """Mock structural control fixture."""
    return MockStructuralControl(learning_rate=1e-5, num_epochs=2)


@pytest.fixture
def mock_state_control() -> MockStateControl:
    """Mock state control fixture."""
    return MockStateControl(target_layers=[0, 1, 2], scale_factor=0.5)


@pytest.fixture
def mock_output_control() -> MockOutputControl:
    """Mock output control fixture."""
    return MockOutputControl(temperature=0.7, top_k=40)
