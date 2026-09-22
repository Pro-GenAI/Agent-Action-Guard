"""Compare regular Action Guard and MoC on the same HarmActionsEval rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn

from agent_action_guard._runtime_utils import (
    ALL_CLASSES,
    EmbeddingModel,
    flatten_action_to_text,
)
from experiments.moc import MixtureOfClassifiers


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "agent_action_guard" / "harmactions_dataset.json"
DEFAULT_REGULAR_MODEL = ROOT / "agent_action_guard" / "action_classifier_model.pt"
DEFAULT_MOC_MODEL = Path(__file__).with_name("moc_harmactions.pt")


class RegularActionNet(nn.Module):
    """Architecture used by training.train_nn.ActionClassNet."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, len(ALL_CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/cpu/cuda/cuda:N consistently for both classifiers."""
    if device_name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device_name}")
    return device


def load_rows(include_safe: bool):
    with open(DATA_PATH, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not include_safe:
        rows = [row for row in rows if row["classification"].lower() != "safe"]
    return rows


def embed_rows(rows, batch_size: int) -> np.ndarray:
    """Compute embeddings once so both classifiers receive identical features."""
    model = EmbeddingModel()
    print(f"Embedding backend: {model.backend}", flush=True)
    texts = [flatten_action_to_text(row["action"]) for row in rows]
    chunks = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task("Embedding shared benchmark", total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            values = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(values, dtype=np.float32))
            progress.advance(task, len(batch))
    return np.concatenate(chunks, axis=0)


def load_regular(path: Path, device: torch.device) -> RegularActionNet:
    checkpoint = torch.load(path, map_location=device)
    hidden = int(checkpoint.get("config", {}).get("hidden", 64))
    model = RegularActionNet(int(checkpoint["in_dim"]), hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_moc(path: Path, device: torch.device) -> MixtureOfClassifiers:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = MixtureOfClassifiers(
        in_dim=int(checkpoint["in_dim"]),
        hidden_dim=int(config["hidden"]),
        num_classes=int(checkpoint.get("num_classes", len(ALL_CLASSES))),
        num_experts=int(config["num_experts"]),
        router_hidden_dim=int(config["router_hidden_dim"]),
        top_k=int(config["top_k"]) if config["top_k"] is not None else None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_regular(
    model: RegularActionNet,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with torch.inference_mode(), progress:
        task = progress.add_task("Regular model HarmActionsEval", total=len(embeddings))
        for start in range(0, len(embeddings), batch_size):
            values = torch.as_tensor(
                embeddings[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(model(values).argmax(dim=-1).cpu().numpy())
            progress.advance(task, len(values))
    return np.concatenate(predictions)


def infer_moc(
    model: MixtureOfClassifiers,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with torch.inference_mode(), progress:
        task = progress.add_task("MoC HarmActionsEval", total=len(embeddings))
        for start in range(0, len(embeddings), batch_size):
            values = torch.as_tensor(
                embeddings[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits, _, _ = model(values)
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            progress.advance(task, len(values))
    return np.concatenate(predictions)


def metrics(rows, predictions: np.ndarray):
    labels = np.asarray(
        [ALL_CLASSES.index(row["classification"].lower()) for row in rows],
        dtype=np.int64,
    )
    true_harmful = labels != 0
    predicted_harmful = predictions != 0
    return {
        "total": int(len(labels)),
        "harm_actions_score": float(
            predicted_harmful[true_harmful].mean() * 100 if true_harmful.any() else 0.0
        ),
        "binary_accuracy": float((predicted_harmful == true_harmful).mean() * 100),
        "three_way_accuracy": float((predictions == labels).mean() * 100),
        "predicted_harmful": int(predicted_harmful.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--regular-model", type=Path, default=DEFAULT_REGULAR_MODEL)
    parser.add_argument("--moc-model", type=Path, default=DEFAULT_MOC_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--include-safe",
        action="store_true",
        help="Include safe rows; default matches HarmActionsEval harmful/unethical-only.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.embedding_batch_size < 1:
        parser.error("batch sizes must be >= 1")

    device = resolve_device(args.device)
    print(f"Benchmark device for BOTH models: {device}")

    rows = load_rows(args.include_safe)
    embeddings = embed_rows(rows, args.embedding_batch_size)

    regular = load_regular(args.regular_model, device)
    moc = load_moc(args.moc_model, device)

    regular_predictions = infer_regular(regular, embeddings, device, args.batch_size)
    moc_predictions = infer_moc(moc, embeddings, device, args.batch_size)

    result = {
        "device": str(device),
        "include_safe": bool(args.include_safe),
        "regular": metrics(rows, regular_predictions),
        "moc": metrics(rows, moc_predictions),
    }

    print("\nHarmActionsEval comparison")
    print(
        f"{'model':<10} {'HarmActions':>12} {'binary acc':>12} "
        f"{'3-way acc':>12} {'harmful':>10}"
    )
    for name in ("regular", "moc"):
        item = result[name]
        print(
            f"{name:<10} {item['harm_actions_score']:>11.2f}% "
            f"{item['binary_accuracy']:>11.2f}% "
            f"{item['three_way_accuracy']:>11.2f}% "
            f"{item['predicted_harmful']:>10d}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved results to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
