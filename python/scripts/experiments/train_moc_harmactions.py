"""Train the research-only MoC model on the packaged HarmActions dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent_action_guard._runtime_utils import (
    ALL_CLASSES,
    EmbeddingModel,
    flatten_action_to_text,
)
from experiments.moc import MixtureOfClassifiers, set_seed


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "agent_action_guard" / "harmactions_dataset.json"
DEFAULT_MODEL_PATH = Path(__file__).with_name("moc_harmactions.pt")


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/cpu/cuda/cuda:N consistently."""
    if device_name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device_name}")
    return device


def load_dataset():
    """Load flattened action texts and three-way labels."""
    with open(DATA_PATH, encoding="utf-8") as handle:
        rows = json.load(handle)
    texts = [flatten_action_to_text(row["action"]) for row in rows]
    labels = [ALL_CLASSES.index(row["classification"].lower()) for row in rows]
    classes = [row["classification"].lower() for row in rows]
    return texts, labels, classes


def embed_texts(
    model: EmbeddingModel,
    texts,
    batch_size: int,
    description: str,
) -> np.ndarray:
    """Embed texts in batches with a visible progress bar."""
    chunks = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task(description, total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            chunk = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(chunk, dtype=np.float32))
            progress.advance(task, len(batch))
    return np.concatenate(chunks, axis=0)


def binary_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Treat harmful and unethical as one blocked class."""
    return float(((labels != 0) == (predictions != 0)).float().mean().item())


def train(args: argparse.Namespace) -> None:
    """Train MoC on the same dataset/split convention as the regular trainer."""
    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"Training MoC on: {device}")

    texts, labels, classes = load_dataset()
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=classes,
    )

    # Use the same embedding-backend selection as normal Agent Action Guard.
    embed_model = EmbeddingModel()
    print(f"Embedding backend: {embed_model.backend}", flush=True)
    x_train_np = embed_texts(
        embed_model, x_train_text, args.embedding_batch_size, "Embedding train set"
    )
    x_test_np = embed_texts(
        embed_model, x_test_text, args.embedding_batch_size, "Embedding test set"
    )

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test_np, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    # Match the regular trainer: derive class weights from the full packaged dataset.
    counts = Counter(labels)
    total = len(labels)
    class_weights = torch.tensor(
        [total / counts[index] for index in range(len(ALL_CLASSES))],
        dtype=torch.float32,
        device=device,
    )

    model = MixtureOfClassifiers(
        in_dim=x_train.shape[1],
        hidden_dim=args.hidden,
        num_classes=len(ALL_CLASSES),
        num_experts=args.experts,
        router_hidden_dim=args.router_hidden,
        top_k=args.top_k,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(
        TensorDataset(x_train, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task("Training MoC", total=args.epochs * len(loader))
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_index, (xb, yb) in enumerate(loader, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                logits, weights, _ = model(xb)
                task_loss = loss_fn(logits, yb)
                balance_loss = model.load_balance_loss(weights)
                loss = task_loss + args.balance_weight * balance_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                progress.update(
                    task,
                    description=(
                        f"Training MoC epoch {epoch}/{args.epochs} "
                        f"loss={running_loss / batch_index:.4f}"
                    ),
                )
                progress.advance(task)

    model.eval()
    with torch.inference_mode():
        logits, weights, _ = model(x_test)
        predictions = logits.argmax(dim=-1)
        three_way = float((predictions == y_test_tensor).float().mean().item())
        blocked = binary_accuracy(y_test_tensor, predictions)
        utilization = weights.mean(dim=0).detach().cpu().numpy()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_dim": int(x_train.shape[1]),
        "num_classes": len(ALL_CLASSES),
        "config": {
            "hidden": args.hidden,
            "num_experts": args.experts,
            "router_hidden_dim": args.router_hidden,
            "top_k": args.top_k,
            "balance_weight": args.balance_weight,
            "lr": args.lr,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "embedding_model": embed_model.model_name,
        "embedding_backend": embed_model.backend,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    print(f"Validation 3-way accuracy:  {three_way:.4f}")
    print(f"Validation blocked accuracy: {blocked:.4f}")
    print("Router utilization:          " + np.array2string(utilization, precision=3))
    print(f"Saved MoC checkpoint to: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--router-hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--balance-weight", type=float, default=0.03)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.experts < 1:
        raise SystemExit("--experts must be >= 1")
    if args.top_k < 1 or args.top_k > args.experts:
        raise SystemExit("--top-k must be between 1 and --experts")
    if args.embedding_batch_size < 1 or args.batch_size < 1:
        raise SystemExit("batch sizes must be >= 1")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
