"""Compare end-to-end Action Guard latency with ONNX vs API embeddings."""

from __future__ import annotations

import argparse
import os
import time
from statistics import mean

from agent_action_guard._runtime_utils import DEFAULT_EMBED_MODEL_NAME, EmbeddingModel
from agent_action_guard.action_classifier import ActionClassifier
from agent_action_guard.cli import load_actions

DEFAULT_ACTIONS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "London"}},
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": {"to": "user@example.com", "subject": "Status update"},
        },
    },
    {
        "type": "function",
        "function": {"name": "delete_user", "arguments": {"user_id": "123"}},
    },
    {
        "type": "function",
        "function": {
            "name": "export_records",
            "arguments": {"dataset": "customers", "destination": "external"},
        },
    },
]


def percentile(values: list[float], fraction: float) -> float:
    """Return an interpolated percentile from a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_durations(
    durations_ms: list[float], action_count: int
) -> dict[str, float]:
    """Summarize measured full-run durations."""
    if not durations_ms:
        raise ValueError("durations_ms must not be empty")
    if action_count <= 0:
        raise ValueError("action_count must be greater than zero")

    total_ms = sum(durations_ms)
    measured_actions = action_count * len(durations_ms)
    return {
        "mean_run_ms": mean(durations_ms),
        "p50_run_ms": percentile(durations_ms, 0.50),
        "p95_run_ms": percentile(durations_ms, 0.95),
        "mean_action_ms": total_ms / measured_actions,
        "actions_per_second": measured_actions / (total_ms / 1000.0),
    }


def benchmark_classifier(
    classifier: ActionClassifier,
    actions: list[dict],
    *,
    iterations: int,
    warmup: int,
    batch_size: int,
) -> dict[str, float]:
    """Measure steady-state end-to-end classification latency."""
    for _ in range(warmup):
        classifier.predict_batch(actions, batch_size=batch_size)

    durations_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        classifier.predict_batch(actions, batch_size=batch_size)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return summarize_durations(durations_ms, len(actions))


def load_benchmark_actions(file_path: str | None, limit: int) -> list[dict]:
    """Load benchmark actions from a file or use the built-in sample set."""
    actions = load_actions(file_path=file_path) if file_path else list(DEFAULT_ACTIONS)
    actions = actions[:limit]
    if not actions:
        raise ValueError("No actions available for the benchmark")
    return actions


def create_classifiers(api_model: str) -> tuple[ActionClassifier, ActionClassifier]:
    """Create independent classifiers backed by ONNX and API embeddings."""
    onnx_embedding_model = EmbeddingModel()
    onnx_embedding_model.backend = "onnx"

    api_embedding_model = EmbeddingModel(model_name=api_model)
    api_embedding_model.backend = "api"

    return (
        ActionClassifier(embedding_model=onnx_embedding_model),
        ActionClassifier(embedding_model=api_embedding_model),
    )


def print_report(
    onnx_stats: dict[str, float], api_stats: dict[str, float], action_count: int
) -> None:
    """Print a compact latency comparison table."""
    print()
    print(f"Measured actions per run: {action_count}")
    print(
        f"{'Backend':<8} {'Mean/run':>12} {'p50/run':>12} {'p95/run':>12} "
        f"{'Mean/action':>13} {'Actions/s':>12}"
    )
    for name, stats in (("ONNX", onnx_stats), ("API", api_stats)):
        print(
            f"{name:<8} {stats['mean_run_ms']:>9.2f} ms "
            f"{stats['p50_run_ms']:>9.2f} ms {stats['p95_run_ms']:>9.2f} ms "
            f"{stats['mean_action_ms']:>10.2f} ms "
            f"{stats['actions_per_second']:>12.2f}"
        )

    ratio = api_stats["mean_action_ms"] / onnx_stats["mean_action_ms"]
    print()
    if ratio >= 1.0:
        print(f"API / ONNX mean per-action latency: {ratio:.2f}x slower")
    else:
        print(f"API / ONNX mean per-action latency: {1.0 / ratio:.2f}x faster")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Action Guard latency using ONNX and API embeddings."
    )
    parser.add_argument(
        "--file",
        help="Optional JSON array or JSONL file of actions; built-in actions are used otherwise",
    )
    parser.add_argument(
        "--limit", type=int, default=16, help="Maximum actions to benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Measured runs per backend"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per backend; use 0 to include cold-start cost",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum actions per vectorized inference batch",
    )
    parser.add_argument(
        "--api-model",
        default=os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL_NAME),
        help="Embedding model name sent to the OpenAI-compatible API",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if not (os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
        parser.error(
            "Set EMBEDDING_API_KEY or OPENAI_API_KEY for the API benchmark. "
            "For an unauthenticated local endpoint, EMBEDDING_API_KEY=dummy is sufficient."
        )

    actions = load_benchmark_actions(args.file, args.limit)
    onnx_classifier, api_classifier = create_classifiers(args.api_model)

    print(
        f"Benchmarking {len(actions)} actions, {args.iterations} measured runs/backend, "
        f"{args.warmup} warmup run(s), batch size {args.batch_size}."
    )
    if args.warmup:
        print(
            "Warmup excludes model/client/session initialization from measured latency."
        )

    onnx_stats = benchmark_classifier(
        onnx_classifier,
        actions,
        iterations=args.iterations,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )
    api_stats = benchmark_classifier(
        api_classifier,
        actions,
        iterations=args.iterations,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )
    print_report(onnx_stats, api_stats, len(actions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
