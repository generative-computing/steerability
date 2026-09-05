"""Tests for state control common components.

Tests cover:
- SteeringVector save/load round-trip
- ContrastivePairs validation
- as_contrastive_pairs helper
- make_token_mask for all scopes
- get_model_layer_list against test model fixtures
- projected_cosine_similarity
- AdditiveTransform
- NormPreservingTransform
"""
from pathlib import Path

import pytest
import torch

from steerability.algorithms.state_control.common import (
    ContrastivePairs,
    SteeringVector,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from steerability.algorithms.state_control.common.hook_utils import (
    extract_hidden_states,
    get_model_layer_list,
    replace_hidden_states,
)
from steerability.algorithms.state_control.common.token_scope import compute_prompt_lens, make_token_mask


class TestSteeringVector:
    """Tests for SteeringVector dataclass."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves all data with torch.Tensor directions."""
        directions = {
            0: torch.randn(768),
            1: torch.randn(768),
            5: torch.randn(768),
        }
        variances = {0: 0.85, 1: 0.72, 5: 0.91}

        original = SteeringVector(
            model_type="llama",
            directions=directions,
            explained_variances=variances,
        )

        save_path = str(tmp_path / "test_vector")
        original.save(save_path)

        loaded = SteeringVector.load(save_path)

        assert loaded.model_type == original.model_type
        assert set(loaded.directions.keys()) == set(original.directions.keys())
        assert set(loaded.explained_variances.keys()) == set(original.explained_variances.keys())

        for k in original.directions:
            # loaded directions are at least 2D [T, H]; compare squeezed for 1D originals
            expected = original.directions[k].float()
            if expected.ndim == 1:
                expected = expected.unsqueeze(0)
            torch.testing.assert_close(
                loaded.directions[k],
                expected,
                rtol=1e-5,
                atol=1e-5,
            )

        for k in original.explained_variances:
            assert abs(loaded.explained_variances[k] - original.explained_variances[k]) < 1e-6

    def test_save_adds_svec_extension(self, tmp_path):
        """Test that .svec extension is added if not present."""
        vec = SteeringVector(
            model_type="gpt2",
            directions={0: torch.zeros(128)},
            explained_variances={0: 0.5},
        )

        save_path = str(tmp_path / "no_extension")
        vec.save(save_path)

        assert Path(save_path + ".svec").exists()

    def test_to_moves_tensors(self):
        """Test that to() moves all direction tensors."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(64), 1: torch.randn(64)},
            explained_variances={0: 0.5, 1: 0.5},
        )

        # move to float16
        vec.to("cpu", torch.float16)

        for d in vec.directions.values():
            assert d.dtype == torch.float16

    def test_validate_empty_model_type_raises(self):
        """Test that empty model_type raises ValueError."""
        vec = SteeringVector(
            model_type="",
            directions={0: torch.zeros(64)},
            explained_variances={0: 0.5},
        )
        with pytest.raises(ValueError, match="model_type must be provided"):
            vec.validate()

    def test_validate_empty_directions_raises(self):
        """Test that empty directions raises ValueError."""
        vec = SteeringVector(
            model_type="llama",
            directions={},
            explained_variances={0: 0.5},
        )
        with pytest.raises(ValueError, match="directions must not be empty"):
            vec.validate()

    def test_num_tokens_and_is_positional_1d(self):
        """Test num_tokens and is_positional with 1D (non-positional) directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(1, 64)},  # T=1, H=64
            explained_variances={0: 1.0},
        )
        assert vec.num_tokens == 1
        assert vec.is_positional is False

    def test_num_tokens_and_is_positional_2d(self):
        """Test num_tokens and is_positional with 2D (positional) directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(5, 64)},  # T=5, H=64
            explained_variances={0: 1.0},
        )
        assert vec.num_tokens == 5
        assert vec.is_positional is True

    def test_num_tokens_empty_directions(self):
        """Test num_tokens returns 0 for empty directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={},
            explained_variances={},
        )
        assert vec.num_tokens == 0
        assert vec.is_positional is False

    def test_normalized_unit_norm_clone(self):
        """normalized() returns a unit-norm deep clone; the original is untouched."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.tensor([3.0, 4.0]), 1: torch.zeros(2)},
        )
        original = {k: v.clone() for k, v in vec.directions.items()}
        norm_vec = vec.normalized()

        assert norm_vec is not vec
        assert float(norm_vec.directions[0].norm()) == pytest.approx(1.0, abs=1e-6)
        assert torch.equal(norm_vec.directions[1], vec.directions[1])  # zero-norm row left as-is
        for k, d in original.items():
            assert torch.equal(vec.directions[k], d)  # original unchanged

    def test_scaled_to_norms_sets_target_norms(self):
        """Each covered direction is rescaled to scale * target_norms[layer]."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.tensor([3.0, 4.0]), 1: torch.tensor([0.0, 2.0])},
        )
        out = vec.scaled_to_norms({0: 10.0, 1: 4.0}, scale=0.5)
        assert float(out.directions[0].norm()) == pytest.approx(5.0, abs=1e-5)
        assert float(out.directions[1].norm()) == pytest.approx(2.0, abs=1e-5)
        # orientation preserved
        torch.testing.assert_close(
            out.directions[0] / out.directions[0].norm(),
            (vec.directions[0] / vec.directions[0].norm()).unsqueeze(0),
        )

    def test_scaled_to_norms_original_untouched(self):
        """The source vector is not mutated (ownership contract, mirrors normalized())."""
        vec = SteeringVector(model_type="llama", directions={0: torch.tensor([3.0, 4.0])})
        snapshot = vec.directions[0].clone()
        _ = vec.scaled_to_norms({0: 100.0})
        assert torch.equal(vec.directions[0], snapshot)

    def test_scaled_to_norms_stores_row_vector(self):
        """Both [H] and [1, H] inputs yield [1, H] storage."""
        vec_1d = SteeringVector(model_type="llama", directions={0: torch.tensor([3.0, 4.0])})
        vec_2d = SteeringVector(model_type="llama", directions={0: torch.tensor([[3.0, 4.0]])})
        assert vec_1d.scaled_to_norms({0: 1.0}).directions[0].shape == (1, 2)
        assert vec_2d.scaled_to_norms({0: 1.0}).directions[0].shape == (1, 2)

    def test_scaled_to_norms_covers_intersection_only(self):
        """The result covers exactly the intersection of target_norms and directions keys."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0]), 2: torch.tensor([1.0, 1.0])},
        )
        out = vec.scaled_to_norms({1: 2.0, 2: 3.0, 99: 1.0})
        assert set(out.directions.keys()) == {1, 2}

    def test_scaled_to_norms_empty_intersection_raises(self):
        vec = SteeringVector(model_type="llama", directions={0: torch.tensor([1.0, 1.0])})
        with pytest.raises(ValueError, match="no overlap"):
            vec.scaled_to_norms({5: 1.0})

    def test_scaled_to_norms_positional_raises(self):
        """K > 1 (positional) directions are rejected."""
        vec = SteeringVector(model_type="llama", directions={0: torch.randn(3, 4)})
        with pytest.raises(ValueError, match="broadcast directions"):
            vec.scaled_to_norms({0: 1.0})

    def test_scaled_to_norms_bad_scale_raises(self):
        vec = SteeringVector(model_type="llama", directions={0: torch.tensor([1.0, 1.0])})
        with pytest.raises(ValueError, match="scale must be positive"):
            vec.scaled_to_norms({0: 1.0}, scale=0.0)

    def test_scaled_to_norms_nonpositive_target_raises(self):
        vec = SteeringVector(model_type="llama", directions={0: torch.tensor([1.0, 1.0])})
        with pytest.raises(ValueError, match="target norm for layer"):
            vec.scaled_to_norms({0: 0.0})

    def test_scaled_to_norms_zero_source_raises(self):
        """Unlike normalized(), a zero-norm source is unreachable and raises."""
        vec = SteeringVector(model_type="llama", directions={0: torch.zeros(2)})
        with pytest.raises(ValueError, match="zero-norm source"):
            vec.scaled_to_norms({0: 1.0})

    def test_scaled_to_norms_preserves_metadata(self):
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.tensor([3.0, 4.0])},
            num_heads=8,
            head_dim=16,
            explained_variances={0: 0.9},
        )
        out = vec.scaled_to_norms({0: 1.0})
        assert out.model_type == "llama"
        assert out.num_heads == 8 and out.head_dim == 16
        assert out.explained_variances == {0: 0.9}


class TestContrastivePairs:
    """Tests for ContrastivePairs dataclass."""

    def test_valid_pairs(self):
        """Test creation with valid data."""
        pairs = ContrastivePairs(
            positives=["positive example 1", "positive example 2"],
            negatives=["negative example 1", "negative example 2"],
        )
        assert len(pairs.positives) == 2
        assert len(pairs.negatives) == 2
        assert pairs.prompts is None

    def test_valid_pairs_with_prompts(self):
        """Test creation with prompts."""
        pairs = ContrastivePairs(
            positives=["yes", "yes"],
            negatives=["no", "no"],
            prompts=["Is this good? ", "Is this bad? "],
        )
        assert len(pairs.prompts) == 2

    def test_empty_positives_raises(self):
        """Test that empty positives raises ValueError."""
        with pytest.raises(ValueError, match="at least one entry"):
            ContrastivePairs(positives=[], negatives=["neg"])

    def test_empty_negatives_raises(self):
        """Test that empty negatives raises ValueError."""
        with pytest.raises(ValueError, match="at least one entry"):
            ContrastivePairs(positives=["pos"], negatives=[])

    def test_unequal_lengths_raises(self):
        """Test that unequal positive/negative lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have equal length"):
            ContrastivePairs(
                positives=["a", "b"],
                negatives=["c"],
            )

    def test_prompts_length_mismatch_raises(self):
        """Test that prompts with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="prompts must have the same length"):
            ContrastivePairs(
                positives=["a", "b"],
                negatives=["c", "d"],
                prompts=["prompt1"],  # wrong length
            )


class TestAsContrastivePairs:
    """Tests for as_contrastive_pairs helper function."""

    def test_passthrough_instance(self):
        """Test that existing instance is returned as-is."""
        pairs = ContrastivePairs(positives=["a"], negatives=["b"])
        result = as_contrastive_pairs(pairs)
        assert result is pairs

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "positives": ["pos1", "pos2"],
            "negatives": ["neg1", "neg2"],
        }
        result = as_contrastive_pairs(data)
        assert isinstance(result, ContrastivePairs)
        assert result.positives == tuple(data["positives"]) or list(result.positives) == data["positives"]

    def test_from_dict_with_prompts(self):
        """Test creation from dict with prompts."""
        data = {
            "positives": ["p1"],
            "negatives": ["n1"],
            "prompts": ["prompt "],
        }
        result = as_contrastive_pairs(data)
        assert result.prompts is not None

    def test_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected ContrastivePairs or dict"):
            as_contrastive_pairs("invalid")

        with pytest.raises(TypeError, match="Expected ContrastivePairs or dict"):
            as_contrastive_pairs(123)


class TestVectorTrainSpec:
    """Tests for VectorTrainSpec dataclass."""

    def test_defaults(self):
        """Test default values."""
        spec = VectorTrainSpec()
        assert spec.method == "pca_pairwise"
        assert spec.accumulate == "all"
        assert spec.batch_size == 8

    def test_invalid_batch_size_raises(self):
        """Test that batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            VectorTrainSpec(batch_size=0)


class TestComputePromptLens:
    """Tests for compute_prompt_lens function.

    compute_prompt_lens returns seq_len for all items, regardless of padding.
    This ensures that with "after_prompt" scope, generation starts after all
    input positions (including pads), which is correct for KV-cached generation.
    """

    def test_no_padding(self):
        """Test prompt lens with no padding."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        lens = compute_prompt_lens(input_ids, pad_token_id=None)
        assert lens.tolist() == [5]

    def test_with_padding(self):
        """Test prompt lens with left padding returns seq_len for all items."""
        input_ids = torch.tensor([
            [0, 0, 1, 2, 3],  # 3 non-pad tokens but seq_len=5
            [0, 1, 2, 3, 4],  # 4 non-pad tokens but seq_len=5
        ])
        lens = compute_prompt_lens(input_ids, pad_token_id=0)
        # returns seq_len (5) for all items regardless of pad count
        assert lens.tolist() == [5, 5]

    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        input_ids = torch.tensor([1, 2, 3, 4])
        lens = compute_prompt_lens(input_ids, pad_token_id=None)
        assert lens.tolist() == [4]


class TestMakeTokenMask:
    """Tests for make_token_mask function."""

    def test_scope_all(self):
        """Test 'all' scope returns all True."""
        prompt_lens = torch.tensor([3, 4])
        mask = make_token_mask("all", seq_len=5, prompt_lens=prompt_lens)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_scope_after_prompt(self):
        """Test 'after_prompt' scope returns True only after prompt."""
        prompt_lens = torch.tensor([3, 4])
        mask = make_token_mask("after_prompt", seq_len=6, prompt_lens=prompt_lens)

        # batch 0: prompt_len=3, so positions 3,4,5 should be True
        assert mask[0].tolist() == [False, False, False, True, True, True]
        # batch 1: prompt_len=4, so positions 4,5 should be True
        assert mask[1].tolist() == [False, False, False, False, True, True]

    def test_scope_after_prompt_with_position_offset(self):
        """Test 'after_prompt' with position_offset for KV-cached generation."""
        prompt_lens = torch.tensor([10])

        # simulate KV-cached generation: seq_len=1, but we're at position 10 (first generated token)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=10)
        assert mask[0].tolist() == [True]  # position 10 >= prompt_len 10

        # simulate being at position 9 (still in prompt)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=9)
        assert mask[0].tolist() == [False]  # position 9 < prompt_len 10

        # simulate being at position 15 (well into generation)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=15)
        assert mask[0].tolist() == [True]  # position 15 >= prompt_len 10

    def test_scope_after_prompt_initial_pass_all_prompt(self):
        """Test 'after_prompt' on initial pass where seq_len == prompt_len."""
        prompt_lens = torch.tensor([5])

        # initial forward pass: seq_len == prompt_len, all positions are prompt
        mask = make_token_mask("after_prompt", seq_len=5, prompt_lens=prompt_lens, position_offset=0)
        assert mask[0].tolist() == [False, False, False, False, False]

    def test_scope_last_k(self):
        """Test 'last_k' scope returns True only for last k tokens."""
        prompt_lens = torch.tensor([3])
        mask = make_token_mask("last_k", seq_len=5, prompt_lens=prompt_lens, last_k=2)

        # last 2 positions (3, 4) should be True
        assert mask[0].tolist() == [False, False, False, True, True]

    def test_scope_last_k_ignores_position_offset(self):
        """Test that 'last_k' uses local positions, ignoring position_offset."""
        prompt_lens = torch.tensor([3])

        # last_k should work on local seq_len regardless of position_offset
        mask = make_token_mask("last_k", seq_len=5, prompt_lens=prompt_lens, last_k=2, position_offset=100)
        assert mask[0].tolist() == [False, False, False, True, True]

        # with seq_len=1, last_k=1 should always be True
        mask = make_token_mask("last_k", seq_len=1, prompt_lens=prompt_lens, last_k=1, position_offset=100)
        assert mask[0].tolist() == [True]

    def test_last_k_none_raises(self):
        """Test that last_k=None with 'last_k' scope raises ValueError."""
        with pytest.raises(ValueError, match="last_k must be >= 1"):
            make_token_mask("last_k", seq_len=5, prompt_lens=torch.tensor([3]), last_k=None)

    def test_unknown_scope_raises(self):
        """Test that unknown scope raises ValueError."""
        with pytest.raises(ValueError, match="Unknown token scope"):
            make_token_mask("invalid", seq_len=5, prompt_lens=torch.tensor([3]))


class TestExtractHiddenStates:
    """Tests for extract_hidden_states function."""

    def test_from_positional_args(self):
        """Test extraction from positional args."""
        hidden = torch.randn(2, 10, 64)
        result = extract_hidden_states((hidden, "other"), {})
        assert result is hidden

    def test_from_kwargs(self):
        """Test extraction from kwargs."""
        hidden = torch.randn(2, 10, 64)
        result = extract_hidden_states((), {"hidden_states": hidden})
        assert result is hidden

    def test_not_found(self):
        """Test that None is returned if not found."""
        result = extract_hidden_states((), {"other_key": "value"})
        assert result is None


class TestReplaceHiddenStates:
    """Tests for replace_hidden_states function."""

    def test_replace_in_positional_args(self):
        """Test replacement in positional args."""
        old_hidden = torch.randn(2, 10, 64)
        new_hidden = torch.randn(2, 10, 64)
        other_arg = "other"

        new_args, new_kwargs = replace_hidden_states(
            (old_hidden, other_arg), {}, new_hidden
        )

        assert new_args[0] is new_hidden
        assert new_args[1] == other_arg
        assert new_kwargs == {}

    def test_replace_in_kwargs(self):
        """Test replacement in kwargs."""
        old_hidden = torch.randn(2, 10, 64)
        new_hidden = torch.randn(2, 10, 64)
        original_kwargs = {"hidden_states": old_hidden, "other": "value"}

        new_args, new_kwargs = replace_hidden_states(
            (), original_kwargs, new_hidden
        )

        assert new_args == ()
        assert new_kwargs["hidden_states"] is new_hidden
        assert new_kwargs["other"] == "value"


class TestGetModelLayerList:
    """Tests for get_model_layer_list function."""

    def test_llama_style_model(self, model_and_tokenizer):
        """Test layer extraction from llama-style model."""
        from steerability.algorithms.core.internals.model_layout import resolve_model_layout

        model, _ = model_and_tokenizer
        model_type = model.config.model_type

        # skip if the resolver does not recognize the architecture
        try:
            layout = resolve_model_layout(model)
        except ValueError:
            pytest.skip(f"Model {model_type} has unknown architecture")

        modules, names = get_model_layer_list(model)

        assert len(modules) > 0
        assert len(names) == len(modules)
        assert all(n.startswith(layout.layer_prefix + ".") for n in names)


class TestProjectedCosineSimilarity:
    """Tests for projected_cosine_similarity function."""

    def test_known_values(self):
        """Test against known values."""
        from steerability.algorithms.state_control.common.gating import projected_cosine_similarity

        # create a simple case
        hidden = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])

        # projector = outer(d, d) / dot(d, d) = [[1,0,0],[0,0,0],[0,0,0]]
        projector = torch.outer(direction, direction) / (direction @ direction + 1e-8)

        # projection = tanh(P @ h) = tanh([1,0,0]) = [tanh(1), 0, 0]
        # sim = dot(h, proj) / (norm(h) * norm(proj))
        #     = tanh(1) / (1 * tanh(1)) = 1.0

        sim = projected_cosine_similarity(hidden, projector)
        assert abs(sim - 1.0) < 0.01  # should be close to 1

    def test_orthogonal_vectors(self):
        """Test with orthogonal vectors."""
        from steerability.algorithms.state_control.common.gating import projected_cosine_similarity

        hidden = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([0.0, 1.0, 0.0])
        projector = torch.outer(direction, direction) / (direction @ direction + 1e-8)

        # P @ h = [[0,0,0],[0,1,0],[0,0,0]] @ [1,0,0] = [0,0,0]
        # projection is zero, so tanh(0) = 0
        # sim with zero vector -> handle division by zero gracefully

        sim = projected_cosine_similarity(hidden, projector)
        # should be 0 or very small due to the 1e-8 epsilon
        assert abs(sim) < 0.1


class TestAdditiveTransform:
    """Tests for AdditiveTransform."""

    def test_applies_direction_with_mask(self):
        """Test that direction is added only where mask is True."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 4, 8)  # [B=1, T=4, H=8]
        directions = {0: torch.ones(8)}  # layer 0: all ones
        transform = AdditiveTransform(directions, strength=2.0)

        # mask only positions 1 and 3
        mask = torch.tensor([[False, True, False, True]])

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # positions 0, 2 should be zeros
        assert result[0, 0, :].sum().item() == 0
        assert result[0, 2, :].sum().item() == 0

        # positions 1, 3 should be 2.0 * ones = 2.0 per element, 8 elements
        assert result[0, 1, :].sum().item() == 16.0
        assert result[0, 3, :].sum().item() == 16.0

    def test_no_direction_returns_unchanged(self):
        """Test that missing layer direction returns hidden unchanged."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.randn(2, 5, 16)
        transform = AdditiveTransform({0: torch.randn(16)}, strength=1.0)
        mask = torch.ones(2, 5, dtype=torch.bool)

        # layer 99 not in directions
        result = transform.apply(hidden, layer_id=99, token_mask=mask)
        assert torch.equal(result, hidden)

    def test_strength_scaling(self):
        """Test that strength parameter scales correctly."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 1, 4)
        directions = {0: torch.tensor([1.0, 2.0, 3.0, 4.0])}
        transform = AdditiveTransform(directions, strength=0.5)
        mask = torch.ones(1, 1, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)
        expected = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]])
        torch.testing.assert_close(result, expected)

    def test_positional_mode_with_alignment(self):
        """Test positional mode with alignment parameter."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 6, 4)  # [B=1, T=6, H=4]
        # positional steering vector with T=3 tokens
        directions = {0: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 2.0, 0.0, 0.0],  # token 1
            [0.0, 0.0, 3.0, 0.0],  # token 2
        ])}
        # inject starting at position 2
        transform = AdditiveTransform(directions, strength=1.0, alignment=2, positional=True)
        mask = torch.ones(1, 6, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # positions 0, 1 should be unchanged (before alignment)
        assert result[0, 0, :].sum().item() == 0
        assert result[0, 1, :].sum().item() == 0
        # positions 2, 3, 4 should get the steering vectors
        torch.testing.assert_close(result[0, 2, :], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 3, :], torch.tensor([0.0, 2.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 4, :], torch.tensor([0.0, 0.0, 3.0, 0.0]))
        # position 5 should be unchanged (after steering vector ends)
        assert result[0, 5, :].sum().item() == 0

    def test_positional_mode_clips_at_seq_end(self):
        """Test that positional mode clips steering vectors at sequence end."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 4, 4)  # [B=1, T=4, H=4]
        # steering vector with T=3, but aligned at position 2 so only 2 fit
        directions = {0: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],  # won't fit
        ])}
        transform = AdditiveTransform(directions, strength=1.0, alignment=2, positional=True)
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # only positions 2, 3 should get steering (position 4 is out of bounds)
        torch.testing.assert_close(result[0, 2, :], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 3, :], torch.tensor([0.0, 2.0, 0.0, 0.0]))

    def test_positional_mode_skips_when_out_of_range(self):
        """Test that positional mode returns unchanged when alignment is beyond seq_len."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 3, 4)  # [B=1, T=3, H=4]
        directions = {0: torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])}
        # alignment at position 5, but seq_len is only 3
        transform = AdditiveTransform(directions, strength=1.0, alignment=5, positional=True)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # should return unchanged since alignment is beyond sequence
        assert result.sum().item() == 0


class TestNormPreservingTransform:
    """Tests for NormPreservingTransform."""

    def test_preserves_norm_when_increased(self):
        """Test that norm is preserved when it would increase."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

        # start with unit norm vectors
        hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # norm = 1
        directions = {0: torch.tensor([0.0, 2.0, 0.0, 0.0])}  # would add norm
        inner = AdditiveTransform(directions, strength=1.0)
        transform = NormPreservingTransform(inner)

        mask = torch.ones(1, 1, dtype=torch.bool)
        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # original norm = 1, after addition norm would be sqrt(1 + 4) = sqrt(5)
        # should be scaled back to norm = 1
        result_norm = result.norm(dim=-1)
        torch.testing.assert_close(result_norm, torch.tensor([[1.0]]), rtol=1e-5, atol=1e-5)

    def test_does_not_scale_when_norm_decreases(self):
        """Test that scaling doesn't happen when norm decreases."""
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

        # large initial norm
        hidden = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]])  # norm = 3
        directions = {0: torch.tensor([-2.0, 0.0, 0.0, 0.0])}  # subtracts
        inner = AdditiveTransform(directions, strength=1.0)
        transform = NormPreservingTransform(inner)

        mask = torch.ones(1, 1, dtype=torch.bool)
        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # after addition: [1, 0, 0, 0], norm = 1 < 3
        # should NOT be scaled (norm decreased)
        expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        torch.testing.assert_close(result, expected)

    def test_raises_on_nan(self):
        """Test that NaN detection raises ValueError."""
        from steerability.algorithms.state_control.common.transforms import NormPreservingTransform
        from steerability.algorithms.state_control.common.transforms.base import BaseTransform

        class NaNTransform(BaseTransform):
            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                return torch.tensor([[[float("nan")]]])

        transform = NormPreservingTransform(NaNTransform())
        mask = torch.ones(1, 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="NaN or Inf detected"):
            transform.apply(torch.ones(1, 1, 1), layer_id=0, token_mask=mask)


class TestTransformBinding:
    """Binding protocol across the six transforms (is_bound / bind / covered_layer_ids / guards)."""

    HIDDEN = 8

    def _sv(self, k=1):
        return SteeringVector(model_type="x", directions={0: torch.randn(k, self.HIDDEN), 1: torch.randn(k, self.HIDDEN)})

    def _stub_source(self, sv):
        from steerability.algorithms.state_control.common.sources import _Precomputed
        return _Precomputed(sv)

    def _ctx(self, resolve_result=None):
        """A minimal TransformContext whose resolve returns a fixed vector (or coerces its input)."""
        from steerability.algorithms.state_control.common.sources import _as_artifact_source
        from steerability.algorithms.state_control.common.transforms.context import TransformContext

        def resolve(artifact):
            if resolve_result is not None:
                return resolve_result.clone()
            return _as_artifact_source(artifact).resolve(None, None)

        return TransformContext(
            layer_ids=[0, 1], num_layers=4, hidden_size=self.HIDDEN, num_heads=2, head_dim=4,
            dtype=torch.float32, device=torch.device("cpu"), resolve=resolve,
        )

    def test_additive_bound_from_dict_and_sv(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        sv = self._sv()
        assert AdditiveTransform(sv).is_bound is True
        assert AdditiveTransform(sv).covered_layer_ids == {0, 1}
        assert AdditiveTransform({0: torch.randn(1, self.HIDDEN)}).covered_layer_ids == {0}

    def test_additive_bound_bind_returns_self(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        t = AdditiveTransform(self._sv(), strength=2.0)
        assert t.bind(self._ctx()) is t

    def test_additive_source_binds_functionally(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        sv = self._sv()
        src = self._stub_source(sv)
        t = AdditiveTransform(src, strength=2.0)
        assert t.is_bound is False and t.covered_layer_ids is None
        bound = t.bind(self._ctx())
        assert bound is not t and bound.is_bound is True
        assert bound.strength == 2.0 and bound.covered_layer_ids == {0, 1}
        assert t.is_bound is False  # template untouched

    def test_unbound_apply_raises(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        t = AdditiveTransform(self._stub_source(self._sv()))
        with pytest.raises(RuntimeError, match="unbound"):
            t.apply(torch.randn(1, 3, self.HIDDEN), layer_id=0, token_mask=torch.ones(1, 3, dtype=torch.bool))

    def test_directional_ablation_junk_positional(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform
        with pytest.raises(TypeError, match="alpha"):
            ProjectionTransform(0.5)

    def test_additive_junk_positional(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform
        with pytest.raises(TypeError, match="strength"):
            AdditiveTransform(2.0)

    def test_fresh_caches_per_bound_instance(self):
        """One template bound against two ctxs with different directions -> independent bases."""
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform
        src = self._stub_source(self._sv())
        template = ProjectionTransform(src, alpha=1.0)

        sv_a = SteeringVector(model_type="x", directions={0: torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])})
        sv_b = SteeringVector(model_type="x", directions={0: torch.tensor([[0, 1.0, 0, 0, 0, 0, 0, 0]])})
        a = template.bind(self._ctx(resolve_result=sv_a))
        b = template.bind(self._ctx(resolve_result=sv_b))
        assert a is not b and a._basis_cache is not b._basis_cache

        hidden = torch.ones(1, 1, self.HIDDEN)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out_a = a.apply(hidden, layer_id=0, token_mask=mask)
        out_b = b.apply(hidden, layer_id=0, token_mask=mask)
        assert not torch.allclose(out_a, out_b)  # different ablated components

    def test_rotation_deferred_validation(self):
        """A [1, H] (non-basis-pair) resolve errors at bind, matching the concrete __init__ error."""
        from steerability.algorithms.state_control.common.transforms import RotationTransform
        bad = SteeringVector(model_type="x", directions={0: torch.randn(1, self.HIDDEN)})
        # concrete bad shape errors at __init__
        with pytest.raises(ValueError, match=r"\[2, H\]"):
            RotationTransform(bad)
        # deferred: source resolving to a bad shape errors at bind
        t = RotationTransform(self._stub_source(bad))
        assert t.is_bound is False
        with pytest.raises(ValueError, match=r"\[2, H\]"):
            t.bind(self._ctx())

    def test_head_additive_rejects_bare_mapping(self):
        from steerability.algorithms.state_control.common.transforms import HeadAdditiveTransform
        with pytest.raises(ValueError, match="num_heads and head_dim"):
            HeadAdditiveTransform({0: torch.randn(2, 4)}, active_heads={0: {0}})

    def test_norm_preserving_delegates_binding(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform
        inner = AdditiveTransform(self._stub_source(self._sv()))
        wrapper = NormPreservingTransform(inner)
        assert wrapper.is_bound is False and wrapper.covered_layer_ids is None
        bound = wrapper.bind(self._ctx())
        assert bound is not wrapper and bound.is_bound is True
        assert bound.covered_layer_ids == {0, 1}

    def test_alignment_adaptive_two_part_binding(self):
        from steerability.algorithms.state_control.common.transforms import (
            AdditiveTransform,
            AlignmentAdaptiveTransform,
        )
        sv = self._sv()
        # own concrete, inner unbound -> not bound (inner unbound)
        inner_unbound = AdditiveTransform(self._stub_source(sv))
        t1 = AlignmentAdaptiveTransform(inner_unbound, sv)
        assert t1.is_bound is False
        # own source, inner bound -> not bound (own unbound)
        t2 = AlignmentAdaptiveTransform(AdditiveTransform(sv), self._stub_source(sv))
        assert t2.is_bound is False
        # both concrete -> bound
        t3 = AlignmentAdaptiveTransform(AdditiveTransform(sv), sv)
        assert t3.is_bound is True
        # binding resolves both, delegates coverage to inner
        bound = t1.bind(self._ctx())
        assert bound.is_bound is True and bound.covered_layer_ids == {0, 1}


class TestLayerHeuristics:
    """Tests for layer heuristics functions."""

    def test_late_third(self):
        """Test late_third returns correct layer range."""
        from steerability.algorithms.state_control.common.selectors import late_third

        # 12 layers -> last third is layers 8-11
        result = late_third(12)
        assert result == [8, 9, 10, 11]

        # 24 layers -> last third is layers 16-23
        result = late_third(24)
        assert result == list(range(16, 24))

        # 3 layers -> last third is layer 2
        result = late_third(3)
        assert result == [2]


class TestResolveTransformSlot:
    """Unit tests for the shared `resolve_transform_slot` helper (hub-free tiny Llama)."""

    HIDDEN = 32
    LAYERS = 4
    HEADS = 4

    def _model(self):
        from tests.utils.tiny_models import tiny_llama

        return tiny_llama(num_layers=self.LAYERS, hidden=self.HIDDEN, heads=self.HEADS)

    def _sv(self, layers=(0, 1), k=1):
        return SteeringVector(
            model_type="llama",
            directions={l: torch.randn(k, self.HIDDEN) for l in layers},
        )

    def _stub_source(self, sv):
        from steerability.algorithms.state_control.common.sources import _Precomputed

        return _Precomputed(sv)

    def test_bound_instance_passes_through(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, resolve_transform_slot

        transform = AdditiveTransform(self._sv(layers=(0, 1)), strength=1.5)
        built = resolve_transform_slot(transform, self._model(), None, [0, 1])
        assert built is transform  # already bound -> used as-is

    def test_source_carrying_instance_comes_back_bound(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform, resolve_transform_slot

        template = ProjectionTransform(self._stub_source(self._sv(layers=(0, 1))), alpha=0.7)
        assert template.is_bound is False
        built = resolve_transform_slot(template, self._model(), None, [0, 1])
        assert built is not template
        assert built.is_bound is True
        assert built.alpha == 0.7
        assert template.is_bound is False  # template untouched

    def test_factory_returning_bound_transform(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, resolve_transform_slot

        sv = self._sv(layers=(0, 1))
        built = resolve_transform_slot(
            lambda ctx: AdditiveTransform(ctx.resolve(sv), strength=2.0),
            self._model(), None, [0, 1],
        )
        assert isinstance(built, AdditiveTransform)
        assert built.is_bound is True and built.strength == 2.0

    def test_factory_returning_source_carrying_transform_is_bound(self):
        # an unbound factory result is bound here
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform, resolve_transform_slot

        source = self._stub_source(self._sv(layers=(0, 1)))
        built = resolve_transform_slot(
            lambda ctx: ProjectionTransform(source, alpha=1.0),
            self._model(), None, [0, 1],
        )
        assert isinstance(built, ProjectionTransform)
        assert built.is_bound is True

    def test_factory_returning_non_transform_raises(self):
        from steerability.algorithms.state_control.common.transforms import resolve_transform_slot

        with pytest.raises(TypeError, match="must return a BaseTransform"):
            resolve_transform_slot(lambda ctx: object(), self._model(), None, [0, 1])

    def test_coverage_passes_when_layers_covered(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform, resolve_transform_slot

        transform = ProjectionTransform(self._sv(layers=(0, 1, 2)))
        built = resolve_transform_slot(transform, self._model(), None, [0, 1])
        assert built is transform

    def test_coverage_raises_when_layer_missing(self):
        from steerability.algorithms.state_control.common.transforms import ProjectionTransform, resolve_transform_slot

        transform = ProjectionTransform(self._sv(layers=(0,)))
        with pytest.raises(ValueError, match="no direction for layer"):
            resolve_transform_slot(transform, self._model(), None, [0, 1])

    def test_coverage_opts_out_when_none(self):
        # a transform reporting covered_layer_ids=None is not coverage-checked
        from steerability.algorithms.state_control.common.transforms import resolve_transform_slot
        from steerability.algorithms.state_control.common.transforms.base import BaseTransform

        class _NoCoverage(BaseTransform):
            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                return hidden_states

        transform = _NoCoverage()
        assert transform.covered_layer_ids is None
        built = resolve_transform_slot(transform, self._model(), None, [0, 1])
        assert built is transform

    def test_context_exposes_resolved_layers_and_working_resolve(self):
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform, resolve_transform_slot

        seen = {}

        def _factory(ctx):
            seen["ctx"] = ctx
            sv = self._sv(layers=tuple(ctx.layer_ids))
            resolved = ctx.resolve(sv)  # round-trip a vector through resolve
            assert set(resolved.directions.keys()) == set(ctx.layer_ids)
            for d in resolved.directions.values():
                assert d.device == ctx.device and d.dtype == ctx.dtype
            return AdditiveTransform(resolved, strength=1.0)

        model = self._model()
        resolve_transform_slot(_factory, model, None, [1, 2])
        ctx = seen["ctx"]
        assert ctx.layer_ids == [1, 2]
        assert ctx.num_layers == self.LAYERS
        assert ctx.hidden_size == self.HIDDEN
        assert ctx.num_heads == self.HEADS
        assert ctx.head_dim == self.HIDDEN // self.HEADS
        assert ctx.device == next(model.parameters()).device
        assert ctx.dtype == model.dtype
