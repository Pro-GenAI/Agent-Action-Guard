"""
Hyperparameter tuning script for the neural network model.
"""

import itertools

from sklearn.model_selection import train_test_split
import pandas as pd

from train_nn import (
    load_texts_and_labels,
    make_embeddings,
    train_one,
    MODEL_PATH,
    rl_fine_tune,
)
import torch

# Hyperparameter grid
PARAM_GRID = {
    # updated search space: focus on mid/large networks and slightly larger lr
    "hidden": [64, 128, 256, 512],
    "lr": [5e-4, 1e-3, 2e-3],
    "epochs": [2, 3, 4, 6, 8],
    "weight_decay": [0, 1e-4],
}


def hyperparam_tuning():
    """
    Perform hyperparameter tuning using grid search on the neural network model.
    """
    texts, labels, classes, class_weights = load_texts_and_labels()
    # split texts, labels and classes together so we keep class strings for RL
    xtr_texts, xte_texts, ytr, yte, ctr, cte = train_test_split(
        texts, labels, classes, test_size=0.25, random_state=42, stratify=classes
    )

    print("Generating embeddings...")
    xtr_embs = make_embeddings(xtr_texts)
    xte_embs = make_embeddings(xte_texts)

    # simple grid search
    best = {"acc": -1, "config": None, "model": None, "preds": None}
    results = []
    keys, values = zip(*PARAM_GRID.items())
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
            class_weights=class_weights,
        )
        print(f"-> acc: {acc:.4f}")
        # store trial result
        results.append(
            {
                "hidden": cfg["hidden"],
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "weight_decay": cfg["weight_decay"],
                "acc": float(acc),
            }
        )
        if acc > best["acc"]:
            best.update({"acc": acc, "config": cfg, "model": model, "preds": preds})

    print("Best config:", best["config"], "acc:", best["acc"])
    # show all results sorted by accuracy
    if pd is not None:
        df = pd.DataFrame(results)
        df = df.sort_values(
            by=["acc", "hidden", "lr", "epochs", "weight_decay"], ascending=[False, True, True, True, True]
        )
        print("\nAll trials sorted by acc:")
        print(df.to_string(index=False))
    else:
        # fallback: pretty-print JSON sorted
        results_sorted = sorted(
            results, key=lambda x: (-x["acc"], x["hidden"], x["lr"], x["epochs"])
        )
        print("\nAll trials sorted by acc:")
        for r in results_sorted:
            print(r)
    # save best supervised/pretrained model for RL initialization
    try:
        if best.get("model") is not None:
            torch.save(
                {"model_state_dict": best["model"].state_dict(), "config": best["config"],
                    "in_dim": xtr_embs.shape[1]},
                MODEL_PATH
            )
            print(f"Saved best pretrained model to {MODEL_PATH}")
    except Exception as e:
        print("Failed saving pretrained model:", e)

    # # Run RL fine-tuning on top of embeddings using class strings
    # try:
    #     print("Running RL fine-tune on best config (safe defaults)...")
    #     # use a safer, smaller LR and fewer epochs for RL to avoid degrading pretrained
    #     cfg = best["config"]
    #     m = best.get("model")
    #     pre_state = m.state_dict() if m is not None else None

    #     rl_lr = float(cfg.get("lr", 0.001)) * 0.1
    #     rl_epochs = max(2, int(cfg.get("epochs", 6) // 2))
    #     print(f"RL params: hidden={cfg['hidden']} lr={rl_lr:.6f} epochs={rl_epochs}")

    #     clf, rl_acc = rl_fine_tune(
    #         xtr_embs,
    #         xte_embs,
    #         ctr,
    #         cte,
    #         hidden=cfg["hidden"],
    #         lr=rl_lr,
    #         epochs=rl_epochs,
    #         batch_size=32,
    #         pretrained_state_dict=pre_state,
    #     )
    #     print(f"-> RL acc: {rl_acc:.4f} (supervised: {best['acc']:.4f})")

    #     # Save whichever is better: supervised pretrained or RL head
    #     if rl_acc > float(best["acc"]):
    #         try:
    #             torch.save({"model_state_dict": clf.state_dict(), "config": cfg},
    #                        MODEL_PATH.with_name(MODEL_PATH.stem + "_rl.pt"))
    #             print(f"RL improved. Saved RL model to {MODEL_PATH.with_name(MODEL_PATH.stem + '_rl.pt')}")
    #         except Exception:
    #             print("RL improved but failed to save RL model file.")
    #     else:
    #         print("RL did not improve over supervised model. Keeping pretrained supervised model.")
    # except Exception as e:
    #     print("RL fine-tune failed:", e)


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    hyperparam_tuning()
