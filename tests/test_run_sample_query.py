from agent_action_classifier.run_sample_query import TOOL_METADATA


def test_tool_metadata_has_entry():
    # Ensure the sample tool metadata exists for the demo tool
    assert "send_threat" in TOOL_METADATA


def test_execute_tool_simulation(monkeypatch):
    # Simulate is_action_harmful behavior by patching
    monkeypatch.setattr("run_sample_query.is_action_harmful", lambda md: (True, 0.9))
    # When is_action_harmful returns True, the function printing/behavior is in main;
    # here we just assert metadata
    assert TOOL_METADATA.get("send_threat") is not None
