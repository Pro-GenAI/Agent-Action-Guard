import json
from pathlib import Path

import pytest

from agent_action_guard import hooks
from agent_action_guard.harnesses import claude_code, codex, common, cursor, kiro


@pytest.fixture
def classifier(monkeypatch):
    def fake(action):
        arguments = action["function"]["arguments"]
        if arguments.get("dangerous"):
            return "harmful", 0.98
        return None, 0.92

    monkeypatch.setattr(common, "is_action_harmful", fake)


def test_common_normalizes_tool_payload(classifier):
    result = common.classify_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "echo ok"}}
    )
    assert result.allowed
    assert result.label is None


def test_common_blocks_over_threshold(classifier):
    result = common.classify_hook_payload(
        {"tool_name": "Bash", "tool_input": {"dangerous": True}},
        conf_threshold=0.8,
    )
    assert result.blocked
    assert result.label == "harmful"
    assert "Agent Action Guard blocked" in result.reason


def test_common_rejects_malformed_payload():
    with pytest.raises(ValueError, match="missing tool_name"):
        common.normalize_hook_call({"tool_input": {}})
    with pytest.raises(ValueError, match="Invalid hook JSON"):
        common.load_hook_payload("{")


@pytest.mark.parametrize("adapter", [codex, claude_code])
def test_codex_style_adapters_emit_pretooluse_deny(adapter, classifier):
    code, stdout, stderr = adapter.handle(
        {"tool_name": "Bash", "tool_input": {"dangerous": True}}
    )
    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    decision = payload["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"


def test_codex_style_adapter_safe_call_is_silent(classifier):
    assert codex.handle(
        {"tool_name": "Bash", "tool_input": {"command": "pwd"}}
    ) == (0, "", "")


def test_cursor_uses_native_permission_protocol(classifier):
    code, stdout, stderr = cursor.handle(
        {"tool_name": "Shell", "tool_input": {"dangerous": True}}
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["permission"] == "deny"

    code, stdout, stderr = cursor.handle(
        {"tool_name": "Read", "tool_input": {"path": "README.md"}}
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout) == {"permission": "allow"}


def test_kiro_blocks_with_exit_code_two(classifier):
    code, stdout, stderr = kiro.handle(
        {"tool_name": "shell", "tool_input": {"dangerous": True}}
    )
    assert code == 2
    assert stdout == ""
    assert "blocked" in stderr

    assert kiro.handle(
        {"tool_name": "read", "tool_input": {"path": "README.md"}}
    ) == (0, "", "")


@pytest.mark.parametrize(
    ("target", "relative_path"),
    [
        ("codex", ".codex/hooks.json"),
        ("claude-code", ".claude/settings.json"),
        ("cursor", ".cursor/hooks.json"),
        ("kiro", ".kiro/hooks/agent-action-guard.json"),
    ],
)
def test_install_hook_creates_expected_project_config(tmp_path, target, relative_path):
    path = hooks.install_hook(target, tmp_path)
    assert path == tmp_path / relative_path
    data = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(data)
    assert f"hooks run --target {target}" in serialized


@pytest.mark.parametrize("target", ["codex", "claude-code", "cursor"])
def test_install_hook_is_idempotent_and_preserves_existing_config(tmp_path, target):
    path = hooks.install_hook(target, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["custom"] = {"preserved": True}
    path.write_text(json.dumps(data), encoding="utf-8")

    hooks.install_hook(target, tmp_path)
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert merged["custom"] == {"preserved": True}
    assert json.dumps(merged).count(f"hooks run --target {target}") == 1


def test_install_claude_code_preserves_existing_hooks(tmp_path):
    path = tmp_path / ".claude" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "echo audit"}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    hooks.install_hook("claude-code", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "echo audit"
    assert len(data["hooks"]["PreToolUse"]) == 1


def test_kiro_config_uses_current_v1_pretooluse_schema(tmp_path):
    path = hooks.install_hook("kiro", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hook = data["hooks"][0]
    assert data["version"] == "v1"
    assert hook["trigger"] == "PreToolUse"
    assert hook["matcher"] == "*"
    assert hook["action"]["type"] == "command"


def test_run_hook_dispatches_target(classifier):
    payload = {"tool_name": "Shell", "tool_input": {"dangerous": True}}
    assert hooks.run_hook("kiro", payload, 0.5)[0] == 2
    assert json.loads(hooks.run_hook("cursor", payload, 0.5)[1])["permission"] == "deny"


def test_run_hook_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unsupported hook target"):
        hooks.run_hook("unknown", {"tool_name": "x"}, 0.5)


def test_install_hook_rejects_unknown_target(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported hook target"):
        hooks.install_hook("unknown", tmp_path)
