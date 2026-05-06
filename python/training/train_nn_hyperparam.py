"""
Hyperparameter tuning script for the neural network model.
"""

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from train_nn import (
    _accuracy_score,
    export_model_to_onnx,
    load_texts_and_labels,
    make_embeddings,
    train_one,
)

from agent_action_guard._runtime_utils import ONNX_MODEL_PATH

# Hyperparameter grid
_PARAM_GRID = {
    # Updated search space: focus on mid/large networks and slightly larger lr
    "hidden": [32, 64, 128, 256, 512],
    "lr": [5e-4, 1e-3, 2e-3],
    "epochs": [2, 3, 4, 6, 8],
    "weight_decay": [0],  # , 1e-4
    "batch_size": [4, 8, 12, 16],  # , 32
}


def hyperparam_tuning():
    """
    Perform hyperparameter tuning using grid search on the neural network model.
    """
    texts, labels, classes, class_weights = load_texts_and_labels()
    # Split texts, labels and classes together so we keep class strings for RL
    xtr_texts, xte_texts, ytr, yte, _, _ = train_test_split(
        texts, labels, classes, test_size=0.25, random_state=42, stratify=classes
    )

    print("Generating embeddings...")
    xtr_embs = make_embeddings(xtr_texts)
    xte_embs = make_embeddings(xte_texts)

    # Simple grid search
    best: dict[str, Any] = {"acc": -1.0, "config": None, "model": None, "preds": None}
    results = []
    keys, values = zip(*_PARAM_GRID.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        print(f"Trying config: {cfg}")
        model, acc, preds = train_one(
            xtr_embs,
            ytr,
            xte_embs,
            yte,
            hidden=cfg["hidden"],
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            class_weights=class_weights,
        )
        print(f"-> acc: {acc:.4f}")
        # Store trial result
        results.append(
            {
                "hidden": cfg["hidden"],
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "weight_decay": cfg["weight_decay"],
                "batch_size": cfg["batch_size"],
                "acc": float(acc),
            }
        )
        if acc > best["acc"]:
            best.update({"acc": acc, "config": cfg, "model": model, "preds": preds})

    print("Best config:", best["config"], "acc:", best["acc"])
    # Show all results sorted by accuracy
    sorted_results = sorted(
        results,
        key=lambda row: (
            -row["acc"],
            row["hidden"],
            row["lr"],
            row["epochs"],
            row["weight_decay"],
            row["batch_size"],
        ),
    )
    print("\nAll trials sorted by acc:")
    columns = [
        "hidden",
        "lr",
        "epochs",
        "weight_decay",
        "batch_size",
        "acc",
    ]
    widths = {
        col: max(len(col), *(len(f"{row[col]}") for row in sorted_results))
        for col in columns
    }
    header = " ".join(f"{col:{widths[col]}}" for col in columns)
    print(header)
    print(" ".join("-" * widths[col] for col in columns))
    for row in sorted_results:
        line = " ".join(f"{row[col]:{widths[col]}}" for col in columns)
        print(line)
    # Export best model to ONNX using shared helper
    try:
        if best.get("model") is None:
            raise ValueError("No trained model available for ONNX export")

        in_dim = xtr_embs.shape[1]
        export_model_to_onnx(best["model"], in_dim)
    except Exception as e:
        print("ONNX export failed:", e)


def evaluate_onnx_model(
    onnx_path: str | Path, X_test: np.ndarray, y_test, batch_size: int = 32
):
    """Run inference with an ONNX model using onnxruntime and return coarse accuracy.

    The returned accuracy uses the same coarse matching logic as the PyTorch
    training code (`_accuracy_score`) which considers labels 1 and 2 as
    "harmful" (truthy) and 0 as "safe" (falsy).
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    preds = []
    n = X_test.shape[0]
    for i in range(0, n, batch_size):
        batch = X_test[i : i + batch_size].astype(np.float32)
        out = sess.run([out_name], {in_name: batch})[0]
        batch_preds = np.argmax(out, axis=1)
        preds.extend(batch_preds.tolist())

    preds = np.array(preds, dtype=int)[:n]
    acc = _accuracy_score(y_test, preds)
    return float(acc), preds


def evaluate_saved_onnx():
    """Load the test split, compute embeddings, and evaluate the saved ONNX model."""
    if not ONNX_MODEL_PATH.exists():
        print(f"No ONNX model found at {ONNX_MODEL_PATH}; skipping ONNX evaluation.")
        return

    texts, labels, classes, _ = load_texts_and_labels()
    # Keep the same split seed/stratify to match training evaluation
    xtr_texts, xte_texts, ytr, yte, _, _ = train_test_split(
        texts, labels, classes, test_size=0.25, random_state=42, stratify=classes
    )

    print("Generating embeddings for evaluation...")
    xte_embs = make_embeddings(xte_texts)

    print(f"Evaluating ONNX model at {ONNX_MODEL_PATH} on {len(xte_embs)} samples...")
    try:
        acc, _ = evaluate_onnx_model(ONNX_MODEL_PATH, xte_embs, yte)
        print(f"ONNX model accuracy (coarse safe/harm match): {acc:.4f}")
    except Exception as e:
        print("ONNX evaluation failed:", e)


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    hyperparam_tuning()
    # Evaluate the saved ONNX model on the test set
    evaluate_saved_onnx()
