"""Mixture of Classifiers (MoC) research prototype.

MoC mirrors the routing idea from Mixture of Experts, but each expert is a
classifier that maps the same feature vector to class logits. A learned router
assigns each sample to one or more classifier experts, and their logits are
combined using the router probabilities.

This module intentionally lives outside the agent_action_guard package. It is
an experiment, not a public/runtime API.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ClassifierExpert(nn.Module):
    """Small MLP classifier used as one MoC expert."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfClassifiers(nn.Module):
    """Classifier experts combined by a learned per-example router."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_experts: int = 4,
        router_hidden_dim: int = 16,
        top_k: int | None = 2,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            ClassifierExpert(in_dim, hidden_dim, num_classes)
            for _ in range(num_experts)
        )
        self.router = nn.Sequential(
            nn.Linear(in_dim, router_hidden_dim),
            nn.Tanh(),
            nn.Linear(router_hidden_dim, num_experts),
        )

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized dense or top-k sparse routing probabilities."""
        weights = torch.softmax(self.router(x), dim=-1)
        if self.top_k is None or self.top_k == self.num_experts:
            return weights

        top_values, top_indices = torch.topk(weights, self.top_k, dim=-1)
        sparse = torch.zeros_like(weights).scatter(-1, top_indices, top_values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixed logits, routing weights, and per-expert logits."""
        weights = self.routing_weights(x)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed_logits = torch.sum(expert_logits * weights.unsqueeze(-1), dim=1)
        return mixed_logits, weights, expert_logits

    @staticmethod
    def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
        """Penalize batches whose mean router utilization is far from uniform."""
        num_experts = weights.shape[-1]
        utilization = weights.mean(dim=0)
        target = torch.full_like(utilization, 1.0 / num_experts)
        return num_experts * torch.mean((utilization - target) ** 2)


class SingleClassifier(nn.Module):
    """Single-MLP baseline with the same basic expert architecture."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = ClassifierExpert(in_dim, hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ExperimentData:
    """Train/test tensors plus the latent regime used for diagnostics."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    region_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    region_test: torch.Tensor


def make_piecewise_dataset(
    n_samples: int = 6000, seed: int = 42, test_fraction: float = 0.25
) -> ExperimentData:
    """Create a task where different input regions obey different rules."""
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 4, generator=generator)
    region = (x[:, 0] >= 0).long()

    left_score = 1.5 * x[:, 1] - 0.45 * x[:, 2] + 0.20 * x[:, 3]
    right_score = (
        1.15 * x[:, 1] * x[:, 2]
        + 0.65 * x[:, 3]
        - 0.35 * x[:, 1]
        + 0.15
    )
    score = torch.where(region.bool(), right_score, left_score)
    noise = 0.18 * torch.randn(n_samples, generator=generator)
    y = (score + noise > 0).long()

    permutation = torch.randperm(n_samples, generator=generator)
    test_size = int(n_samples * test_fraction)
    test_idx = permutation[:test_size]
    train_idx = permutation[test_size:]
    return ExperimentData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        region_train=region[train_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        region_test=region[test_idx],
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute standard multiclass accuracy."""
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train_single(
    data: ExperimentData,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> SingleClassifier:
    """Train the single-classifier baseline."""
    model = SingleClassifier(data.x_train.shape[1], hidden_dim, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def train_moc(
    data: ExperimentData,
    hidden_dim: int,
    num_experts: int,
    top_k: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    balance_weight: float,
) -> MixtureOfClassifiers:
    """Train MoC with classification and router load-balancing losses."""
    model = MixtureOfClassifiers(
        in_dim=data.x_train.shape[1],
        hidden_dim=hidden_dim,
        num_classes=2,
        num_experts=num_experts,
        top_k=top_k,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits, weights, _ = model(xb)
            task_loss = nn.functional.cross_entropy(logits, yb)
            loss = task_loss + balance_weight * model.load_balance_loss(weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_moc(
    model: MixtureOfClassifiers, data: ExperimentData
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return accuracy, overall utilization, and utilization by latent region."""
    model.eval()
    with torch.inference_mode():
        logits, weights, _ = model(data.x_test)
    overall = weights.mean(dim=0)
    by_region = torch.stack(
        [weights[data.region_test == region].mean(dim=0) for region in (0, 1)]
    )
    return accuracy(logits, data.y_test), overall, by_region


def run_experiment(args: argparse.Namespace) -> None:
    """Train both models and print a compact MoC diagnostic report."""
    set_seed(args.seed)
    data = make_piecewise_dataset(args.samples, args.seed)

    single = train_single(
        data,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    single.eval()
    with torch.inference_mode():
        single_accuracy = accuracy(single(data.x_test), data.y_test)

    set_seed(args.seed)
    moc = train_moc(
        data,
        hidden_dim=args.hidden,
        num_experts=args.experts,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        balance_weight=args.balance_weight,
    )
    moc_accuracy, utilization, by_region = evaluate_moc(moc, data)

    print("Mixture of Classifiers experiment")
    print(f"single classifier accuracy: {single_accuracy:.4f}")
    print(f"MoC accuracy:               {moc_accuracy:.4f}")
    print(f"delta:                      {moc_accuracy - single_accuracy:+.4f}")
    print(
        "router mean utilization:    "
        + np.array2string(utilization.numpy(), precision=3)
    )
    print(
        "router utilization, x0<0:  "
        + np.array2string(by_region[0].numpy(), precision=3)
    )
    print(
        "router utilization, x0>=0: "
        + np.array2string(by_region[1].numpy(), precision=3)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--balance-weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
