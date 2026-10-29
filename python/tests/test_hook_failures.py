"""Each native harness must receive its blocking protocol on bridge failures."""

import io
import json

import pytest

from agent_action_guard import hooks


@pytest.mark.parametrize("target", hooks.TARGETS)
@pytest.mark.parametrize("error", [RuntimeError, OSError, ImportError, ValueError])
def test_hook_failure_blocks_without_disclosing_details(
    target, error, monkeypatch, capsys
):
    def fail(*args, **kwargs):
        raise error("PRIVATE_PROVIDER_OR_ACTION_DETAILS")

    monkeypatch.setattr(hooks.sys, "stdin", io.StringIO('{"tool_name":"audit"}'))
    monkeypatch.setattr(hooks, "run_hook", fail)
    code = hooks.main(["run", "--target", target])
    out, err = capsys.readouterr()
    assert "PRIVATE_PROVIDER" not in out + err
    assert "Traceback" not in out + err
    if target in {"kiro", "opencode", "copilot"}:
        assert code == 2
        assert "blocked" in err
    else:
        assert code == 0
        result = json.loads(out)
        if target in {"codex", "claude-code"}:
            assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        elif target == "cursor":
            assert result["permission"] == "deny"
        elif target == "agy":
            assert result["decision"] == "deny"
        elif target == "openclaw":
            assert result["block"] is True
        else:
            assert result["action"] == "block"


def test_claude_install_upgrades_existing_guard_and_preserves_unrelated_hooks(tmp_path):
    path = tmp_path / ".claude/settings.json"
    path.parent.mkdir()
    unrelated = {"type": "command", "command": "other-check", "onFailure": "continue"}
    guard = {
        "type": "command",
        "command": "agent-action-guard hooks run --target claude-code",
    }
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [{"matcher": ".*", "hooks": [guard, unrelated]}]
                },
                "unrelated": True,
            }
        )
    )
    hooks.install_hook("claude-code", tmp_path)
    first = path.read_bytes()
    hooks.install_hook("claude-code", tmp_path)
    assert path.read_bytes() == first
    data = json.loads(first)
    assert data["unrelated"] is True
    entries = data["hooks"]["PreToolUse"][0]["hooks"]
    assert entries[0]["onFailure"] == "block"
    assert entries[1] == unrelated
