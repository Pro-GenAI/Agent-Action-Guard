"""Tests for the research-only Mixture of Classifiers prototype."""

import pytest
import torch

from experiments.moc import MixtureOfClassifiers, make_piecewise_dataset


def test_moc_forward_shapes_and_normalized_routing():
    model = MixtureOfClassifiers(
        in_dim=4, hidden_dim=8, num_classes=3, num_experts=4, top_k=2
    )
    inputs = torch.randn(7, 4)

    logits, weights, expert_logits = model(inputs)

    assert logits.shape == (7, 3)
    assert weights.shape == (7, 4)
    assert expert_logits.shape == (7, 4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6)
    assert torch.all((weights > 0).sum(dim=-1) == 2)


def test_dense_routing_uses_all_experts():
    model = MixtureOfClassifiers(
        in_dim=4, hidden_dim=8, num_classes=2, num_experts=3, top_k=None
    )
    weights = model.routing_weights(torch.randn(5, 4))

    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)


@pytest.mark.parametrize("top_k", [0, 4])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        MixtureOfClassifiers(
            in_dim=4, hidden_dim=8, num_classes=2, num_experts=3, top_k=top_k
        )


def test_load_balance_loss_is_zero_for_uniform_utilization():
    weights = torch.full((10, 4), 0.25)
    loss = MixtureOfClassifiers.load_balance_loss(weights)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_piecewise_dataset_split_and_labels():
    data = make_piecewise_dataset(n_samples=100, seed=7, test_fraction=0.2)

    assert data.x_train.shape == (80, 4)
    assert data.x_test.shape == (20, 4)
    assert set(data.y_train.tolist()).issubset({0, 1})
    assert set(data.region_train.tolist()).issubset({0, 1})
