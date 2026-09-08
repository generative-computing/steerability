"""Integration tests: real Inspect evals over real steered pipelines on tiny hub-free models.

Tasks are defined in-test over in-memory datasets; no `inspect_evals` datasets and no network.
"""
import pytest
import torch

pytest.importorskip("inspect_ai")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.state_control.pasta.control import PASTA
from steerability.evaluation.provider import ProviderOptions, as_inspect_model
from steerability.evaluation.solvers import runtime_kwargs_solver
from steerability.evaluation.suite import InspectSuite
from tests.evaluation.conftest import CHAT_TEMPLATE
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _RecordingMessageControl(InputControl):
    """Message-level input control recording every conversation it adapts."""

    Args = None
    supports_batching = True

    def __init__(self):
        super().__init__()
        self.dispatch_sizes: list[int] = []
        self.seen_contents: list[str] = []

    def adapt_messages(self, messages, runtime_kwargs=None):
        self.dispatch_sizes.append(len(messages))
        for chat in messages:
            self.seen_contents.append(chat[-1]["content"])
        return [[{"role": "system", "content": "attention"}] + list(chat) for chat in messages]

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


class _RecordingPasta(PASTA):
    """PASTA subclass recording the `substrings` values `get_hooks` receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_substrings: list = []

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        self.received_substrings.append(runtime_kwargs.get("substrings"))
        return super().get_hooks(input_ids, runtime_kwargs, **kwargs)


def _chat_pipeline(controls=()) -> SteeringPipeline:
    tokenizer = wordlevel_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    pipeline = SteeringPipeline(controls=list(controls), model=tiny_llama(), tokenizer=tokenizer)
    pipeline.steer()
    return pipeline


class TestMessagesPathIntegration:
    def test_batched_eval_fires_adapt_messages_once_per_sample(self, tmp_path):
        control = _RecordingMessageControl()
        pipeline = _chat_pipeline([control])
        model = as_inspect_model(pipeline, options=ProviderOptions(max_batch_size=4, default_max_tokens=4))
        assert model.api.prompt_path == "messages"

        prompts = [f"the cat sat {i}" for i in range(8)]
        task = Task(
            dataset=MemoryDataset([Sample(input=prompt, target="the") for prompt in prompts]),
            solver=[generate()],
            scorer=includes(),
        )
        logs = inspect_eval(
            task, model=model, display="none", log_dir=str(tmp_path), temperature=0,
            max_connections=4,
        )
        (log,) = logs
        assert log.status == "success"
        assert log.results.completed_samples == 8
        assert any(score.name == "includes" for score in log.results.scores)

        assert sorted(control.seen_contents) == sorted(prompts)  # once per sample
        assert sum(control.dispatch_sizes) == 8
        assert max(control.dispatch_sizes) > 1  # at least one dispatch carried more than one row

    def test_runtime_kwargs_solver_delivers_row_aligned_substrings(self, tmp_path):
        pasta = _RecordingPasta(head_config=[0], alpha=1.5, scale_position="include")
        pipeline = _chat_pipeline([pasta])
        model = as_inspect_model(pipeline, options=ProviderOptions(max_batch_size=4, default_max_tokens=4))

        words = ["cat", "dog", "mat", "sat", "ran", "fast"]
        samples = [
            Sample(
                input=f"the {word} on",
                target="the",
                metadata={"runtime_kwargs": {"substrings": [word]}},
            )
            for word in words
        ]
        task = Task(
            dataset=MemoryDataset(samples),
            solver=[runtime_kwargs_solver()],
            scorer=includes(),
        )
        logs = inspect_eval(
            task, model=model, display="none", log_dir=str(tmp_path), temperature=0,
            max_connections=4,
        )
        assert logs[0].status == "success"

        rows = [group for received in pasta.received_substrings for group in received]
        assert sorted(rows) == sorted([[word] for word in words])  # per-row nested groups
        for received in pasta.received_substrings:
            assert isinstance(received, list)
            assert all(isinstance(group, list) for group in received)

    def test_suite_run_over_eval_set(self, tmp_path):
        pipeline = _chat_pipeline()
        suite = InspectSuite(
            name="target",
            tasks=(Task(
                dataset=MemoryDataset([Sample(input="the cat", target="the") for _ in range(3)]),
                solver=[generate()],
                scorer=includes(),
                name="tiny_includes",
            ),),
            generate_overrides={"temperature": 0, "max_tokens": 4},
        )
        results = suite.run(pipeline, log_dir=tmp_path / "logs", model_name="cfg-a")
        (task_result,) = results.values()
        assert "includes/accuracy" in task_result["metrics"]
        assert "includes/stderr" in task_result["metrics"]
        assert task_result["n"] == 3
        assert (tmp_path / "logs" / task_result["log"]).exists()


class TestTextPathIntegration:
    def test_template_less_tokenizer_evaluates_through_text(self, tmp_path):
        pipeline = SteeringPipeline(model=tiny_llama(), tokenizer=wordlevel_tokenizer())
        pipeline.steer()
        with pytest.warns(UserWarning, match="adapt_messages"):
            model = as_inspect_model(
                pipeline, options=ProviderOptions(max_batch_size=2, default_max_tokens=4),
            )
        assert model.api.prompt_path == "text"

        task = Task(
            dataset=MemoryDataset([Sample(input="the cat sat", target="the") for _ in range(4)]),
            solver=[generate()],
            scorer=includes(),
        )
        logs = inspect_eval(
            task, model=model, display="none", log_dir=str(tmp_path), temperature=0,
            max_connections=2,
        )
        assert logs[0].status == "success"
        assert logs[0].results.completed_samples == 4
