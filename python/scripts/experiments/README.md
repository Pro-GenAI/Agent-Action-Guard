# Mixture of Classifiers (MoC)

This directory contains research-only prototypes. Nothing here is imported by
or packaged with agent_action_guard.

## Concept

Mixture of Classifiers applies the routing pattern from Mixture of Experts to a
classification problem. Instead of one classifier handling every example, MoC
contains multiple classifier experts plus a small router:

1. Every input is scored by the router.
2. The router selects the top-k classifier experts (or all experts in dense
   mode).
3. Selected experts independently produce class logits.
4. Their logits are combined using normalized router weights.
5. Training uses the normal classification loss plus a small load-balancing
   loss so one expert does not trivially receive all traffic.

The hypothesis is that heterogeneous action distributions can benefit when
different classifiers specialize in different action families, risk regimes,
or decision boundaries. Unlike a plain ensemble, the combination is
input-dependent.

## Synthetic experiment

moc.py creates a two-regime synthetic classification task. The regime is hidden
from the model and each regime follows a different label rule.

Run from python/:

    python -m experiments.moc

## Train regular model and MoC on HarmActions

Install development dependencies first:

    uv sync --extra dev

Embedding configuration follows normal Agent Action Guard runtime selection. You do not need embedding API credentials by default: with no embedding environment variables set, the default local MiniLM ONNX embedding model is downloaded/cached and used automatically. Existing `AAG_EMBED_GGUF`, `AAG_EMBED_ONNX`, or embedding API environment variables still override that default using the normal runtime precedence.

For a reproducible same-device comparison, train both on CPU:

    uv run python -m training.train_nn --device cpu
    uv run python -m experiments.train_moc_harmactions --device cpu

To train both on the same GPU instead:

    uv run python -m training.train_nn --device cuda
    uv run python -m experiments.train_moc_harmactions --device cuda

A specific GPU can be selected consistently with cuda:N, for example:

    uv run python -m training.train_nn --device cuda:1
    uv run python -m experiments.train_moc_harmactions --device cuda:1

The regular trainer writes agent_action_guard/action_classifier_model.pt (and
exports the production ONNX model). The MoC trainer writes
experiments/moc_harmactions.pt. Both training paths display progress bars.

## Evaluate regular model vs MoC on HarmActionsEval

The comparison script uses the HarmActionsEval harmful/unethical rows by
default. It computes embeddings once, feeds exactly the same feature matrix to
both classifiers, and runs both PyTorch checkpoints on one shared torch device.
Separate progress bars are shown for the regular model and MoC.

CPU comparison:

    uv run python -m experiments.eval_regular_vs_moc --device cpu

GPU comparison:

    uv run python -m experiments.eval_regular_vs_moc --device cuda

Specific GPU:

    uv run python -m experiments.eval_regular_vs_moc --device cuda:1

Save machine-readable results:

    uv run python -m experiments.eval_regular_vs_moc --device cpu \
      --output experiments/harm_actions_regular_vs_moc.json

To additionally include safe rows and report binary and three-way accuracy over
the full packaged dataset:

    uv run python -m experiments.eval_regular_vs_moc --device cpu --include-safe

The default harmful/unethical-only run reports the HarmActions detection score
alongside binary and three-way classification accuracy. The script prints the
chosen device before evaluation so it is explicit that both models use the same
CPU/GPU.
