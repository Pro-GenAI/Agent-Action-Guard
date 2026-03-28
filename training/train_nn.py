"""
Train a neural network on embeddings built from action metadata.
- Trains a PyTorch MLP classifier.
- Saves the trained PyTorch model and the embedding model name used.
"""

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent_action_guard._runtime_utils import (
    ALL_CLASSES,
    EMBED_MODEL_NAME,
    ONNX_MODEL_PATH,
    EmbeddingModel,
    flatten_action_to_text,
)

MODEL_PATH = Path(__file__).parent.parent / "agent_action_guard" / "action_classifier_model.pt"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ActionClassNet(nn.Module):
    """Simple MLP for multi-class classification."""

    def __init__(self, in_dim, hidden=256):
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
            nn.Linear(hidden // 2, 3),
        )
        self.net.to(DEVICE)

    def forward(self, x):
        return self.net(x)


def _check_match(actual_label, predicted_label):
    """Check if predicted label matches actual label."""
    # The agent's action gets blocked if label is 1 or 2 (harmful/unethical),
    # which means both are the same.
    if actual_label and predicted_label:
        return True
    if not actual_label and not predicted_label:
        return True
    # If one is harmful/unethical and the other is safe, no match.
    return False


def _accuracy_score(y_true, y_pred):
    """Lightweight accuracy scorer used when sklearn is not available.

    Returns fraction of matching labels between y_true and y_pred.
    Works for iterables of hashable items (strings, ints).
    """
    # Convert to lists to allow multiple iterator types
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) == 0:
        return 0.0
    # Only compare up to the shorter length to avoid throwing
    n = min(len(yt), len(yp))
    matches = sum(1 for i in range(n) if _check_match(yt[i], yp[i]))
    return matches / n


# ------------------ Configuration (disable parser) ------------------

_SEED = 42


def _set_seed():
    """Make runs reproducible across torch/numpy/random where possible."""
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    # Deterministic behaviour for cuDNN (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_set_seed()

# Best hyperparameters found via tuning
_HIDDEN = 64
_LR = 0.0010
_EPOCHS = 3
_WEIGHT_DECAY = 0.0
_BATCH_SIZE = 8

_ROOT = Path(__file__).parent.parent
_DATA_PATH = _ROOT / "agent_action_guard" / "harmactions_dataset.json"

_embed_model = EmbeddingModel(EMBED_MODEL_NAME)


def _get_class_label(entry):
    """Map a dataset row to the classifier label index."""
    classification = entry["classification"].lower()
    if classification not in ALL_CLASSES:
        raise ValueError(f"Unknown classification label: {classification}")

    return ALL_CLASSES.index(classification)


with open(_DATA_PATH, encoding="utf-8") as f:
    dataset: list[dict] = json.load(f)


def load_texts_and_labels():
    """Load texts and labels from dataset."""
    texts = []
    labels = []
    classes = []
    for row in dataset:
        texts.append(flatten_action_to_text(row["action"]))
        lbl = _get_class_label(row)
        labels.append(lbl)
        cls = row["classification"]
        classes.append(cls)

    # Calculate class weights for imbalanced dataset
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    weights = torch.tensor(
        [class_weights[0], class_weights[1], class_weights[2]], dtype=torch.float32
    )

    return texts, labels, classes, weights


def make_embeddings(texts):
    """Generate embeddings for texts."""
    # Sentence-transformers returns numpy arrays
    embs = _embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs)


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden: int,
    lr: float,
    epochs: int,
    weight_decay: float = 0,
    batch_size: int = _BATCH_SIZE,
    class_weights: torch.Tensor | None = None,
):
    """Train the model for one configuration.

    Returns the trained model, classification accuracy (based on mapping scores
    back to class labels), and the raw predicted scores for the test set.
    """
    _set_seed()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ActionClassNet(in_dim=X_train.shape[1], hidden=hidden)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    with torch.inference_mode():
        model.eval()
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        yte_np = np.array(y_test)
        cls_acc = _accuracy_score(yte_np, preds)

    return model, cls_acc, preds


def _train_model():
    """Train the model with best hyperparameters."""
    # Ensure reproducible split and training
    _set_seed()
    texts, labels, classes, class_weights = load_texts_and_labels()
    Xtr_texts, Xte_texts, ytr, yte = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=_SEED,
        stratify=classes,
    )

    print("Generating embeddings...")
    # Embeddings should be deterministic given seed and model
    _set_seed()
    Xtr_embs = make_embeddings(Xtr_texts)
    Xte_embs = make_embeddings(Xte_texts)

    # Use best values discovered interactively
    print("Training...")
    cfg = {"hidden": _HIDDEN, "lr": _LR, "epochs": _EPOCHS}
    model, acc, _ = train_one(
        Xtr_embs,
        ytr,
        Xte_embs,
        yte,
        hidden=_HIDDEN,
        lr=_LR,
        epochs=_EPOCHS,
        weight_decay=_WEIGHT_DECAY,
        class_weights=class_weights,
    )
    print(f"-> acc: {acc:.4f}")

    # Save best model and embedding info
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": Xtr_embs.shape[1],
            "config": cfg,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    # Also export to ONNX for runtime use
    try:
        in_dim = Xtr_embs.shape[1]
        export_model_to_onnx(model, in_dim)
    except Exception as e:
        print("ONNX export failed during training save:", e)


def export_model_to_onnx(model: torch.nn.Module, in_dim: int, onnx_path: Path = ONNX_MODEL_PATH, opset_version: int = 13):
    """Export a PyTorch model to ONNX at `onnx_path`.

    The function places the model on CPU and uses a dummy input with shape
    `(1, in_dim)` so it can be consumed by ONNX runtimes.
    """
    try:
        model_to_export = model.eval().cpu()

        dummy_input = torch.randn(1, in_dim, dtype=torch.float32)

        torch.onnx.export(
            model_to_export,
            dummy_input,  # type: ignore
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        print(f"Saved model in ONNX format to {onnx_path}")
    except Exception as e:
        print("Failed saving ONNX model:", e)


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    _train_model()
