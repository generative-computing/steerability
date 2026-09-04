"""Tests for `ConstrainedDecoding`: declarative source validation, requirements per arm, the
in-process xgrammar-compiled automaton, and the automaton-object configuration."""
import pytest
import torch

from aisteer360.algorithms.core.execution import BackendSpec, Capability, ConstraintSource
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control.constrained_decoding import ConstrainedDecoding
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=2, hidden=16, heads=2)


@pytest.fixture(scope="module")
def tokenizer():
    return wordlevel_tokenizer()


class TestArgs:

    def test_convenience_fields_build_the_source(self):
        control = ConstrainedDecoding(choice=["cat", "dog"])
        assert control.source == ConstraintSource(kind="choice", value=("cat", "dog"))
        control = ConstrainedDecoding(regex="cat|dog")
        assert control.source.kind == "regex"

    def test_exactly_one_constraint_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            ConstrainedDecoding()
        with pytest.raises(ValueError, match="exactly one"):
            ConstrainedDecoding(regex="a", choice=["b"])

    def test_source_mapping_coerces(self):
        control = ConstrainedDecoding(source={"kind": "regex", "value": "cat"})
        assert isinstance(control.source, ConstraintSource)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown constraint kind"):
            ConstraintSource(kind="template", value="x")


class TestRequirements:

    def test_declarative_source_is_portable(self):
        control = ConstrainedDecoding(json_schema='{"type": "object"}', include_in_scoring=False)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.supported("generate")

    def test_automaton_object_is_in_process_only(self):
        class _NullAutomaton:
            def reset(self, prefix_ids):
                pass

            def allowed(self, prefix_ids):
                return torch.tensor([0])

        control = ConstrainedDecoding(automaton=_NullAutomaton(), include_in_scoring=False)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        (failure,) = report.failures_for("generate")
        assert failure.message == (
            "ConstrainedDecoding is unsupported at generate on backend kind 'vllm': missing "
            "IN_PROCESS_TORCH; a live automaton object has no declarative form; construct the "
            "control with a ConstraintSource (or json_schema/regex/grammar/choice) or run this "
            "pipeline on the huggingface backend."
        )

    def test_scoring_participation_requires_in_process(self):
        control = ConstrainedDecoding(regex="cat", include_in_scoring=True)
        pipeline = SteeringPipeline(model_name_or_path="m", controls=[control])
        report = pipeline.check(backend=BackendSpec(kind="vllm", model="m"))
        assert report.supported("generate")
        assert not report.supported("score")
        opted_out = SteeringPipeline(
            model_name_or_path="m",
            controls=[ConstrainedDecoding(regex="cat", include_in_scoring=False)],
        ).check(backend=BackendSpec(kind="vllm", model="m"))
        assert opted_out.supported("score")

    def test_stale_engine_range_names_the_kind(self):
        from aisteer360.algorithms.core.execution import BackendCapabilities, ConstraintKinds, evaluate_support

        control = ConstrainedDecoding(grammar='root ::= "a"', include_in_scoring=False)
        stale = BackendCapabilities(
            atoms=frozenset({Capability.GUIDED_DECODING}),
            constraint_kinds=ConstraintKinds(constraints=frozenset({"json_schema"})),
        )
        spec = BackendSpec(kind="vllm", model="m")
        report = evaluate_support([control], spec, stale)
        (failure,) = report.failures_for("generate")
        assert "ConstraintKinds(grammar)" in failure.message


class TestInProcessArm:

    def test_choice_constraint_masks_generation(self, model, tokenizer):
        pytest.importorskip("xgrammar")
        control = ConstrainedDecoding(choice=["cat", "dog"], include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        text = pipeline.generate(text="the mat sat on the", max_new_tokens=4, do_sample=False)
        assert text.strip() in ("cat", "dog")

    def test_live_automaton_drives_the_processor(self, model, tokenizer):
        forced = tokenizer.convert_tokens_to_ids("mat")

        class _ForcedAutomaton:
            def reset(self, prefix_ids):
                pass

            def allowed(self, prefix_ids):
                return torch.tensor([forced])

        control = ConstrainedDecoding(automaton=_ForcedAutomaton(), include_in_scoring=False)
        pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=tokenizer)
        pipeline.steer()
        output = pipeline.generate(
            text="the cat sat", max_new_tokens=3, do_sample=False, return_output=True,
        )
        assert output.output_ids[0].tolist() == [forced] * 3

    def test_export_constraint_returns_the_source(self):
        control = ConstrainedDecoding(regex="cat|dog")
        assert control.export_constraint() == ConstraintSource(kind="regex", value="cat|dog")
        control = ConstrainedDecoding(automaton=object())
        assert control.export_constraint() is None
