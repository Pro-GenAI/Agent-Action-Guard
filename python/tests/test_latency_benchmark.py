import importlib.util
from pathlib import Path

import pytest

_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "scripts"
    / "compare_embedding_backend_latency.py"
)
_SPEC = importlib.util.spec_from_file_location("latency_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_summarize_durations_reports_per_action_latency_and_throughput():
    stats = benchmark.summarize_durations([100.0, 200.0], action_count=10)

    assert stats["mean_run_ms"] == pytest.approx(150.0)
    assert stats["p50_run_ms"] == pytest.approx(150.0)
    assert stats["p95_run_ms"] == pytest.approx(195.0)
    assert stats["mean_action_ms"] == pytest.approx(15.0)
    assert stats["actions_per_second"] == pytest.approx(1000.0 / 15.0)


def test_benchmark_classifier_runs_warmups_and_measured_iterations(monkeypatch):
    calls = []
    clock = iter([1.0, 1.1, 2.0, 2.2])

    class FakeClassifier:
        def predict_batch(self, actions, batch_size=None):
            calls.append((list(actions), batch_size))
            return [("safe", 1.0)] * len(actions)

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    actions = [{"id": 1}, {"id": 2}]

    stats = benchmark.benchmark_classifier(
        FakeClassifier(),
        actions,
        iterations=2,
        warmup=1,
        batch_size=2,
    )

    assert len(calls) == 3
    assert all(batch_size == 2 for _actions, batch_size in calls)
    assert stats["mean_run_ms"] == pytest.approx(150.0)
    assert stats["mean_action_ms"] == pytest.approx(75.0)


def test_load_benchmark_actions_uses_built_in_actions_and_limit():
    actions = benchmark.load_benchmark_actions(None, 2)

    assert len(actions) == 2
    assert actions == benchmark.DEFAULT_ACTIONS[:2]
