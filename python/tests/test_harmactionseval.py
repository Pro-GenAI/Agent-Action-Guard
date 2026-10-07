import importlib
import json
import os

import pytest

_previous_api_key = os.environ.get("OPENAI_API_KEY")
_previous_model = os.environ.get("OPENAI_MODEL")
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["OPENAI_MODEL"] = "test-model"

harmactionseval = importlib.import_module("agent_action_guard.harmactionseval")

if _previous_api_key is None:
    os.environ.pop("OPENAI_API_KEY", None)
else:
    os.environ["OPENAI_API_KEY"] = _previous_api_key

if _previous_model is None:
    os.environ.pop("OPENAI_MODEL", None)
else:
    os.environ["OPENAI_MODEL"] = _previous_model


def _action(action_id, prompt):
    return {
        "action_id": action_id,
        "classification": "harmful",
        "prompt": prompt,
        "manipulated_prompt": prompt,
        "action": {
            "type": "function",
            "function": {"name": f"tool_{action_id}", "arguments": {}},
        },
    }


@pytest.fixture
def configured_eval(tmp_path, monkeypatch):
    cache_path = tmp_path / "harmactionseval_cache.json"
    monkeypatch.setattr(harmactionseval, "model_key", "test-model")
    monkeypatch.setattr(harmactionseval, "cache_path", cache_path)
    monkeypatch.setattr(harmactionseval, "WAIT_INTERVAL", 0)
    return cache_path


def test_load_cache_adds_k_for_new_and_legacy_caches(tmp_path):
    new_cache = harmactionseval._load_cache_from_path(tmp_path / "missing.json", "m")
    assert new_cache == {"m": {"k": 0, "results": {}, "last_attempt": {}}}

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps({"m": {"results": {"1": False, "2": 1}}}), encoding="utf-8"
    )
    legacy_cache = harmactionseval._load_cache_from_path(legacy_path, "m")
    assert legacy_cache["m"]["k"] == 1
    assert legacy_cache["m"]["results"] == {"1": False, "2": 1}
    assert legacy_cache["m"]["last_attempt"] == {"1": 1}


def test_load_cache_migrates_result_k_to_last_attempt(tmp_path):
    cache_path = tmp_path / "result-k.json"
    cache_path.write_text(
        json.dumps(
            {
                "m": {
                    "k": 3,
                    "results": {"1": False, "2": 1},
                    "result_k": {"1": 2, "2": 3},
                }
            }
        ),
        encoding="utf-8",
    )

    migrated_cache = harmactionseval._load_cache_from_path(cache_path, "m")

    assert migrated_cache["m"]["last_attempt"] == {"1": 2}
    assert "result_k" not in migrated_cache["m"]


def test_increasing_k_only_extends_cached_false_results(configured_eval, monkeypatch):
    harmactionseval.cache = {
        "test-model": {
            "k": 1,
            "results": {"-1": 1, "1": False, "2": 1, "3": False},
            "last_attempt": {"1": 1, "3": 1},
        }
    }
    monkeypatch.setattr(
        harmactionseval,
        "load_dataset",
        lambda: [_action(1, "one"), _action(2, "two"), _action(3, "three")],
    )
    calls = []

    def fake_attempt(prompt, attempt=0, tool=None):
        calls.append((prompt, attempt))
        if prompt == "one" and attempt == 3:
            return {"tool_calls": [tool["function"]["name"]]}
        return {"tool_calls": []}

    monkeypatch.setattr(harmactionseval, "attempt_tool_call", fake_attempt)

    summary = harmactionseval.evaluate(k=3)

    assert calls == [("one", 2), ("one", 3), ("three", 2), ("three", 3)]
    assert harmactionseval.cache["test-model"] == {
        "k": 3,
        "results": {"-1": 1, "1": 3, "2": 1, "3": False},
        "last_attempt": {"3": 3},
    }
    assert summary["predicted_harmful"] == 2
    assert summary["predicted_safe"] == 1

    persisted = json.loads(configured_eval.read_text(encoding="utf-8"))
    assert persisted["test-model"]["k"] == 3
    assert persisted["test-model"]["results"]["3"] is False


def test_same_or_lower_k_reuses_cache_without_llm_calls(configured_eval, monkeypatch):
    harmactionseval.cache = {
        "test-model": {
            "k": 3,
            "results": {"-1": 1, "1": False, "2": 2},
            "last_attempt": {"1": 3},
        }
    }
    monkeypatch.setattr(
        harmactionseval,
        "load_dataset",
        lambda: [_action(1, "one"), _action(2, "two")],
    )
    monkeypatch.setattr(
        harmactionseval,
        "attempt_tool_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached results should not call the LLM")
        ),
    )

    same_k = harmactionseval.evaluate(k=3)
    lower_k = harmactionseval.evaluate(k=1)

    assert same_k["predicted_harmful"] == 1
    assert lower_k["predicted_harmful"] == 0
    assert harmactionseval.cache["test-model"]["k"] == 3


def test_new_result_starts_at_attempt_one_even_when_requested_k_is_larger(
    configured_eval, monkeypatch
):
    harmactionseval.cache = {"test-model": {"k": 0, "results": {}, "last_attempt": {}}}
    monkeypatch.setattr(harmactionseval, "load_dataset", lambda: [_action(1, "target")])
    calls = []

    def fake_attempt(prompt, attempt=0, tool=None):
        calls.append((prompt, attempt))
        if prompt.startswith("Calculate the sum") or attempt == 2:
            return {"tool_calls": [tool["function"]["name"]]}
        return {"tool_calls": []}

    monkeypatch.setattr(harmactionseval, "attempt_tool_call", fake_attempt)

    harmactionseval.evaluate(k=3)

    target_attempts = [attempt for prompt, attempt in calls if prompt == "target"]
    assert target_attempts == [1, 2]
    assert harmactionseval.cache["test-model"]["k"] == 3
    assert harmactionseval.cache["test-model"]["results"]["1"] == 2


def test_legacy_false_result_resumes_after_implicit_k_one(configured_eval, monkeypatch):
    configured_eval.write_text(
        json.dumps(
            {
                "test-model": {
                    "results": {"-1": 1, "1": False},
                }
            }
        ),
        encoding="utf-8",
    )
    harmactionseval.load_cache(configured_eval)
    monkeypatch.setattr(harmactionseval, "load_dataset", lambda: [_action(1, "target")])
    calls = []

    def fake_attempt(prompt, attempt=0, tool=None):
        calls.append((prompt, attempt))
        return {"tool_calls": []}

    monkeypatch.setattr(harmactionseval, "attempt_tool_call", fake_attempt)

    harmactionseval.evaluate(k=3)

    assert calls == [("target", 2), ("target", 3)]
    assert harmactionseval.cache["test-model"]["k"] == 3


def test_chunked_k_increase_tracks_false_coverage_per_action(
    configured_eval, monkeypatch
):
    harmactionseval.cache = {
        "test-model": {
            "k": 1,
            "results": {"-1": 1, "1": False, "2": False},
            "last_attempt": {"1": 1, "2": 1},
        }
    }
    monkeypatch.setattr(
        harmactionseval,
        "load_dataset",
        lambda: [_action(1, "one"), _action(2, "two")],
    )
    calls = []

    def fake_attempt(prompt, attempt=0, tool=None):
        calls.append((prompt, attempt))
        return {"tool_calls": []}

    monkeypatch.setattr(harmactionseval, "attempt_tool_call", fake_attempt)

    harmactionseval.evaluate(k=3, offset=0, limit=1)
    assert calls == [("one", 2), ("one", 3)]
    assert harmactionseval.cache["test-model"]["k"] == 3
    assert harmactionseval.cache["test-model"]["last_attempt"] == {"1": 3, "2": 1}

    calls.clear()
    harmactionseval.evaluate(k=3, offset=1, limit=1)

    assert calls == [("two", 2), ("two", 3)]
    assert harmactionseval.cache["test-model"]["last_attempt"] == {"1": 3, "2": 3}


def test_parser_k_default_and_help_are_consistent():
    parser = harmactionseval.build_parser()
    args = parser.parse_args([])

    assert args.k == 1
    assert "default: 1" in parser.format_help()
