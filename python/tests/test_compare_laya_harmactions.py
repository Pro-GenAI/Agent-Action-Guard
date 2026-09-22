import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_laya_harmactions.py"
SPEC = importlib.util.spec_from_file_location("compare_laya_harmactions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "harmful": True,
                    "confidence": 0.91,
                    "classification": "harmful",
                    "reason": "Destructive action.",
                }
            )
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def _row():
    return {
        "action_id": 42,
        "classification": "harmful",
        "action": {
            "type": "function",
            "function": {
                "name": "delete_file",
                "arguments": {"path": "/important/data"},
            },
        },
    }


def test_run_llm_uses_configured_model_no_reasoning_and_structured_output(tmp_path):
    client = FakeClient()
    cache_path = tmp_path / "llm-cache.json"

    predictions, _, stats = MODULE.run_llm(
        [_row()],
        model="test-model",
        cache_path=cache_path,
        client=client,
    )

    assert stats == {"cache_hits": 0, "cache_misses": 1}
    assert predictions[0].predicted_harmful is True
    assert predictions[0].score == 0.91
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["reasoning"] == {"effort": "none"}
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert '"delete_file"' in request["input"]


def test_run_llm_reuses_persistent_cache_without_api_call(tmp_path):
    first_client = FakeClient()
    second_client = FakeClient()
    cache_path = tmp_path / "llm-cache.json"
    row = _row()

    MODULE.run_llm(
        [row],
        model="test-model",
        cache_path=cache_path,
        client=first_client,
    )
    predictions, _, stats = MODULE.run_llm(
        [row],
        model="test-model",
        cache_path=cache_path,
        client=second_client,
    )

    assert cache_path.exists()
    assert len(first_client.responses.calls) == 1
    assert second_client.responses.calls == []
    assert stats == {"cache_hits": 1, "cache_misses": 0}
    assert predictions[0].predicted_harmful is True


def test_cache_key_changes_with_model_or_action():
    row = _row()
    base = MODULE._llm_cache_key("model-a", row)
    other_model = MODULE._llm_cache_key("model-b", row)
    changed_row = _row()
    changed_row["action"]["function"]["arguments"]["path"] = "/tmp/disposable"
    other_action = MODULE._llm_cache_key("model-a", changed_row)

    assert base != other_model
    assert base != other_action
