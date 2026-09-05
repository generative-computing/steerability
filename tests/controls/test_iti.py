"""ITI fit path: head geometry, pooled feature extraction, probe partition, and control wiring.

Pins that `head_geometry` reads the per-layer attention geometry off the module tree (matching the
config on uniform-head models, including a LoRA wrapper), that the ITI estimator fails loudly on a
model with heterogeneous head geometry before running any forward pass, and that the estimator's
per-head directions and probe accuracies match an in-test oracle over the same captured features.
"""
import warnings

import numpy as np
import pytest
import torch
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.internals.data import LabeledExamples, as_labeled_examples
from steerability.algorithms.core.internals.encoding import tokenize_texts
from steerability.algorithms.core.internals.model_layout import head_geometry, resolve_model_layout
from steerability.algorithms.core.internals.pooling import get_last_token_positions, masked_mean, select_at_positions
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.transforms.base import unwrap_modifiers
from steerability.algorithms.state_control.iti.utils.estimator import (
    _SPLIT_SEED,
    _VAL_FRACTION,
    ProbeMassShiftEstimator,
    _probe_partition,
)
from tests.utils.tiny_models import heterogeneous_head_stub, tiny_gpt2, tiny_llama, tiny_lora

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

LAYERS = 3
HIDDEN = 32
HEADS = 4


@pytest.mark.parametrize("factory", [tiny_llama, tiny_gpt2, tiny_lora])
def test_head_geometry_matches_config_on_uniform_models(factory):
    model = factory() if factory is tiny_lora else factory(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    layout = resolve_model_layout(model)
    for layer_id in range(layout.num_layers):
        geometry = head_geometry(model, layout, layer_id)
        assert geometry.num_heads * geometry.head_dim == HIDDEN


def test_iti_raises_on_heterogeneous_geometry_before_forward():
    """The estimator raises the heterogeneous-geometry error naming layers, before any forward."""
    stub = heterogeneous_head_stub(num_layers=4, hidden=HIDDEN)
    data = LabeledExamples(positives=["a", "b"], negatives=["c", "d"])
    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token")
    with pytest.raises(ValueError, match="uniform attention head geometry") as excinfo:
        ProbeMassShiftEstimator().fit(stub, tokenizer=None, data=data, spec=spec)
    message = str(excinfo.value)
    assert "num_heads" in message and "head_dim" in message


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(LLAMA)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


@pytest.fixture
def data():
    return LabeledExamples(
        positives=[f"true statement number {i}" for i in range(8)],
        negatives=[f"false claim item {i}" for i in range(8)],
    )


@pytest.fixture
def grouped_data():
    positives = [f"positive answer {i}" for i in range(8)]
    negatives = [f"negative answer {i}" for i in range(8)]
    # four groups of 2 positives + 2 negatives
    positive_groups = [i // 2 for i in range(8)]
    negative_groups = [i // 2 for i in range(8)]
    return LabeledExamples(
        positives=positives, negatives=negatives,
        positive_groups=positive_groups, negative_groups=negative_groups,
    )


def _oracle(model, tokenizer, data, accumulate):
    """Recompute per-head directions and accuracies from one padded batch per class.

    Captures each `o_proj` input with a plain forward pre-hook, pools it with the same helpers the
    estimator uses, and recomputes theta_hat, sigma, sigma * theta_hat, and the held-out accuracy
    with the identical `train_test_split` / `LogisticRegression` calls and constants. Returns
    `(directions, accuracies)` matching the estimator's output shapes.
    """
    layout = resolve_model_layout(model)
    geometry = head_geometry(model, layout, 0)
    num_heads, head_dim = geometry.num_heads, geometry.head_dim

    pos = list(data.positives)
    neg = list(data.negatives)
    n_pos, n_neg = len(pos), len(neg)

    device = next(model.parameters()).device
    enc_pos = tokenize_texts(tokenizer, pos, device)
    enc_neg = tokenize_texts(tokenizer, neg, device)

    def capture(enc):
        storage = {i: None for i in range(layout.num_layers)}
        handles = []

        def make_hook(layer_id):
            def hook(_module, args, kwargs):
                x = args[0] if args else kwargs.get("input")
                storage[layer_id] = x.detach()
            return hook

        try:
            for layer_id, name in enumerate(layout.oproj_names):
                handles.append(
                    model.get_submodule(name).register_forward_pre_hook(make_hook(layer_id), with_kwargs=True)
                )
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return storage

    mask_pos = enc_pos.get("attention_mask")
    mask_neg = enc_neg.get("attention_mask")
    raw_pos = capture(enc_pos)
    raw_neg = capture(enc_neg)

    labels = np.array([1] * n_pos + [0] * n_neg)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=_VAL_FRACTION, random_state=_SPLIT_SEED, stratify=labels
    )

    directions = {}
    accuracies = {}
    for layer_id in range(layout.num_layers):
        xp = raw_pos[layer_id]
        xn = raw_neg[layer_id]
        if accumulate == "last_token":
            pooled_pos = select_at_positions(xp, get_last_token_positions(mask_pos, xp.size(1), n_pos))
            pooled_neg = select_at_positions(xn, get_last_token_positions(mask_neg, xn.size(1), n_neg))
        else:
            pooled_pos = masked_mean(xp, mask_pos)
            pooled_neg = masked_mean(xn, mask_neg)
        pos_heads = pooled_pos.cpu().view(n_pos, num_heads, head_dim)
        neg_heads = pooled_neg.cpu().view(n_neg, num_heads, head_dim)

        layer_dirs = []
        for head_id in range(num_heads):
            hp = pos_heads[:, head_id, :].float()
            hn = neg_heads[:, head_id, :].float()
            x = torch.cat([hp, hn], dim=0)
            probe = LogisticRegression(max_iter=1000, solver="lbfgs")
            probe.fit(x.numpy()[train_idx], labels[train_idx])
            accuracies[(layer_id, head_id)] = float(probe.score(x.numpy()[val_idx], labels[val_idx]))

            raw = hp.mean(dim=0) - hn.mean(dim=0)
            norm = raw.norm()
            theta_hat = raw / norm if norm > 0 else raw
            sigma = (x @ theta_hat).std()
            layer_dirs.append((sigma * theta_hat).to(dtype=torch.float32))
        directions[layer_id] = torch.stack(layer_dirs, dim=0)

    return directions, accuracies


def test_shapes_and_counts(model_and_tokenizer, data):
    model, tokenizer = model_and_tokenizer
    layout = resolve_model_layout(model)
    geometry = head_geometry(model, layout, 0)

    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token", batch_size=3)
    sv = ProbeMassShiftEstimator().fit(model, tokenizer, data=data, spec=spec)

    assert sv.num_heads == geometry.num_heads
    assert sv.head_dim == geometry.head_dim
    for layer_id in range(layout.num_layers):
        assert sv.directions[layer_id].shape == (geometry.num_heads, geometry.head_dim)
        assert sv.directions[layer_id].dtype == torch.float32
    assert len(sv.probe_accuracies) == layout.num_layers * geometry.num_heads
    assert all(0.0 <= acc <= 1.0 for acc in sv.probe_accuracies.values())


@pytest.mark.parametrize("accumulate", ["last_token", "all"])
def test_matches_oracle(model_and_tokenizer, data, accumulate):
    model, tokenizer = model_and_tokenizer
    spec = VectorTrainSpec(method="mean_diff", accumulate=accumulate, batch_size=3)
    sv = ProbeMassShiftEstimator().fit(model, tokenizer, data=data, spec=spec)
    directions, accuracies = _oracle(model, tokenizer, data, accumulate)

    for layer_id in sv.directions:
        assert torch.allclose(sv.directions[layer_id], directions[layer_id], atol=1e-5, rtol=1e-4)
    assert sv.probe_accuracies == accuracies


def test_chunk_invariance(model_and_tokenizer, data):
    model, tokenizer = model_and_tokenizer
    small = ProbeMassShiftEstimator().fit(
        model, tokenizer, data=data, spec=VectorTrainSpec(method="mean_diff", accumulate="last_token", batch_size=2)
    )
    large = ProbeMassShiftEstimator().fit(
        model, tokenizer, data=data, spec=VectorTrainSpec(method="mean_diff", accumulate="last_token", batch_size=16)
    )
    for layer_id in small.directions:
        assert torch.allclose(small.directions[layer_id], large.directions[layer_id], atol=1e-5, rtol=1e-4)
    assert small.probe_accuracies == large.probe_accuracies


def test_head_probe_fits_emit_no_sklearn_warnings(model_and_tokenizer, data):
    # "always" disables the once-per-location registry, so every emission is recorded
    model, tokenizer = model_and_tokenizer
    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token", batch_size=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sv = ProbeMassShiftEstimator().fit(model, tokenizer, data=data, spec=spec)
    leaked = [w for w in caught if issubclass(w.category, (OptimizeWarning, ConvergenceWarning))]
    assert not leaked, [str(w.message) for w in leaked]
    assert len(sv.probe_accuracies) == len(sv.directions) * sv.num_heads


def test_probe_partition_grouped(grouped_data):
    labels = np.array([1] * 8 + [0] * 8)
    groups = list(grouped_data.positive_groups) + list(grouped_data.negative_groups)
    train_idx, val_idx = _probe_partition(labels, groups)

    groups_arr = np.asarray(groups)
    train_groups = set(groups_arr[train_idx].tolist())
    val_groups = set(groups_arr[val_idx].tolist())
    assert train_groups.isdisjoint(val_groups)
    assert len(set(labels[train_idx].tolist())) == 2
    assert len(set(labels[val_idx].tolist())) == 2


def test_probe_partition_single_group_raises():
    labels = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError, match="two distinct groups"):
        _probe_partition(labels, ["q", "q", "q", "q"])


def test_probe_partition_stranded_class_raises():
    # two groups, but one group is all-positive and the other all-negative: any group split
    # strands a class
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    with pytest.raises(ValueError, match="both classes"):
        _probe_partition(labels, groups)


def test_probe_partition_ungrouped_matches_train_test_split():
    labels = np.array([1] * 8 + [0] * 8)
    train_idx, val_idx = _probe_partition(labels, None)
    indices = np.arange(len(labels))
    exp_train, exp_val = train_test_split(
        indices, test_size=_VAL_FRACTION, random_state=_SPLIT_SEED, stratify=labels
    )
    assert np.array_equal(train_idx, exp_train)
    assert np.array_equal(val_idx, exp_val)


def test_split_mode_in_meta(model_and_tokenizer, data, grouped_data):
    model, tokenizer = model_and_tokenizer
    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token", batch_size=3)

    ungrouped = ProbeMassShiftEstimator().fit(model, tokenizer, data=data, spec=spec)
    assert ungrouped.meta["probe_split"] == "statement"
    assert "model_fingerprint" in ungrouped.meta

    grouped = ProbeMassShiftEstimator().fit(model, tokenizer, data=grouped_data, spec=spec)
    assert grouped.meta["probe_split"] == "group"
    assert "model_fingerprint" in grouped.meta


def test_labeled_examples_validation():
    with pytest.raises(ValueError):
        LabeledExamples(positives=["a"], negatives=["b"], positive_groups=["g"])
    with pytest.raises(ValueError):
        LabeledExamples(positives=["a", "b"], negatives=["c"], positive_groups=["g"], negative_groups=["g"])
    with pytest.raises(ValueError):
        LabeledExamples(positives=["a"], negatives=["b"], positive_groups=[1.5], negative_groups=[2.5])

    restored = as_labeled_examples(
        {"positives": ["a"], "negatives": ["b"], "positive_groups": ["q"], "negative_groups": ["r"]}
    )
    assert list(restored.positive_groups) == ["q"]
    assert list(restored.negative_groups) == ["r"]


def test_control_selects_heads_and_populates_accuracies(model_and_tokenizer, grouped_data):
    from steerability.algorithms.state_control.iti.control import ITI

    model, tokenizer = model_and_tokenizer
    iti = ITI(data=grouped_data, num_heads=5, alpha=2.0)
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[iti], model_name_or_path=LLAMA)
    pipeline.steer()

    core, _ = unwrap_modifiers(iti.interventions[0].transform)
    assert sum(len(heads) for heads in core.active_heads.values()) == 5
    assert iti.export_state()["steering_vector"].probe_accuracies
