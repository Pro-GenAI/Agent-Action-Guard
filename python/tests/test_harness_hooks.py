import json
from pathlib import Path

import pytest

from agent_action_guard import hooks
from agent_action_guard.harnesses import (
    agy,
    claude_code,
    codex,
    common,
    copilot,
    cursor,
    hermes,
    kiro,
    openclaw,
    opencode,
)


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


def test_common_normalizes_copilot_tool_args_json(classifier):
    result = common.classify_hook_payload(
        {"toolName": "bash", "toolArgs": '{"dangerous": true}'}
    )
    assert result.blocked

    with pytest.raises(ValueError, match="tool arguments are invalid JSON"):
        common.normalize_hook_call({"toolName": "bash", "toolArgs": "{"})


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
    assert codex.handle({"tool_name": "Bash", "tool_input": {"command": "pwd"}}) == (
        0,
        "",
        "",
    )


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

    assert kiro.handle({"tool_name": "read", "tool_input": {"path": "README.md"}}) == (
        0,
        "",
        "",
    )


@pytest.mark.parametrize("adapter", [opencode, copilot])
def test_exit_code_adapters_block_with_exit_code_two(adapter, classifier):
    code, stdout, stderr = adapter.handle(
        {"tool_name": "shell", "tool_input": {"dangerous": True}}
    )
    assert code == 2
    assert stdout == ""
    assert "blocked" in stderr

    assert adapter.handle(
        {"tool_name": "read", "tool_input": {"path": "README.md"}}
    ) == (0, "", "")


def test_openclaw_uses_native_block_protocol(classifier):
    code, stdout, stderr = openclaw.handle(
        {"tool_name": "shell", "tool_input": {"dangerous": True}}
    )
    assert code == 0
    assert stderr == ""
    denied = json.loads(stdout)
    assert denied["block"] is True
    assert "blocked" in denied["blockReason"]

    code, stdout, stderr = openclaw.handle(
        {"tool_name": "read", "tool_input": {"path": "README.md"}}
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout) == {"block": False}


def test_hermes_uses_native_block_protocol(classifier):
    code, stdout, stderr = hermes.handle(
        {"tool_name": "shell", "tool_input": {"dangerous": True}}
    )
    assert code == 0
    assert stderr == ""
    denied = json.loads(stdout)
    assert denied["action"] == "block"
    assert "blocked" in denied["message"]

    code, stdout, stderr = hermes.handle(
        {"tool_name": "read", "tool_input": {"path": "README.md"}}
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout) == {}


def test_agy_uses_gemini_family_decision_protocol(classifier):
    code, stdout, stderr = agy.handle(
        {"tool_name": "shell", "tool_input": {"dangerous": True}}
    )
    assert code == 0
    assert stderr == ""
    denied = json.loads(stdout)
    assert denied["decision"] == "deny"
    assert "blocked" in denied["reason"]

    code, stdout, stderr = agy.handle(
        {"tool_name": "read", "tool_input": {"path": "README.md"}}
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout) == {"decision": "allow"}


@pytest.mark.parametrize(
    ("target", "relative_path"),
    [
        ("codex", ".codex/hooks.json"),
        ("claude-code", ".claude/settings.json"),
        ("cursor", ".cursor/hooks.json"),
        ("kiro", ".kiro/hooks/agent-action-guard.json"),
        ("opencode", ".opencode/plugins/agent-action-guard.js"),
        ("agy", ".agents/hooks.json"),
        ("copilot", ".github/hooks/agent-action-guard.json"),
        ("openclaw", ".openclaw/extensions/agent-action-guard/index.js"),
        ("hermes", ".hermes/plugins/agent-action-guard/__init__.py"),
    ],
)
def test_install_hook_creates_expected_project_config(tmp_path, target, relative_path):
    path = hooks.install_hook(target, tmp_path)
    assert path == tmp_path / relative_path
    serialized = path.read_text(encoding="utf-8")
    assert f"hooks run --target {target}" in serialized


@pytest.mark.parametrize("target", ["codex", "claude-code", "cursor", "agy"])
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


def test_opencode_install_is_idempotent_and_refuses_unrelated_plugin(tmp_path):
    path = hooks.install_hook("opencode", tmp_path)
    first = path.read_text(encoding="utf-8")
    hooks.install_hook("opencode", tmp_path)
    assert path.read_text(encoding="utf-8") == first

    other_root = tmp_path / "other"
    other_path = other_root / ".opencode" / "plugins" / "agent-action-guard.js"
    other_path.parent.mkdir(parents=True)
    other_path.write_text("export const SomethingElse = {};", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        hooks.install_hook("opencode", other_root)


def test_agy_config_uses_before_tool_schema(tmp_path):
    path = hooks.install_hook("agy", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hook = data["hooks"]["BeforeTool"][0]
    assert hook["matcher"] == ".*"
    assert hook["hooks"][0]["name"] == "agent-action-guard"
    assert hook["hooks"][0]["type"] == "command"
    assert hook["hooks"][0]["timeout"] == 10000


def test_copilot_config_uses_v1_pre_tool_use_schema(tmp_path):
    path = hooks.install_hook("copilot", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hook = data["hooks"]["preToolUse"][0]
    assert data["version"] == 1
    assert hook["type"] == "command"
    assert hook["bash"] == hook["powershell"]
    assert hook["timeoutSec"] == 10

    data["custom"] = {"preserved": True}
    path.write_text(json.dumps(data), encoding="utf-8")
    hooks.install_hook("copilot", tmp_path)
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert merged["custom"] == {"preserved": True}
    assert len(merged["hooks"]["preToolUse"]) == 1


def test_openclaw_plugin_uses_before_tool_call_and_trust_gated_workspace_layout(
    tmp_path,
):
    path = hooks.install_hook("openclaw", tmp_path)
    plugin_dir = path.parent
    manifest = json.loads(
        (plugin_dir / "openclaw.plugin.json").read_text(encoding="utf-8")
    )
    package = json.loads((plugin_dir / "package.json").read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert manifest["id"] == "agent-action-guard"
    assert manifest["activation"] == {"onStartup": True}
    assert manifest["configSchema"]["additionalProperties"] is False
    assert package["openclaw"]["extensions"] == ["./index.js"]
    assert 'api.on("before_tool_call"' in source
    assert "blockReason" in source
    assert "Invalid hook response" in source

    first = source
    hooks.install_hook("openclaw", tmp_path)
    assert path.read_text(encoding="utf-8") == first

    other_root = tmp_path / "openclaw-other"
    other_path = (
        other_root / ".openclaw" / "extensions" / "agent-action-guard" / "index.js"
    )
    other_path.parent.mkdir(parents=True)
    other_path.write_text("export default {};", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        hooks.install_hook("openclaw", other_root)


def test_hermes_plugin_uses_pre_tool_call_and_project_plugin_layout(tmp_path):
    path = hooks.install_hook("hermes", tmp_path)
    plugin_dir = path.parent
    manifest = (plugin_dir / "plugin.yaml").read_text(encoding="utf-8")
    source = path.read_text(encoding="utf-8")

    assert "name: agent-action-guard" in manifest
    assert "provides_hooks:\n  - pre_tool_call" in manifest
    assert 'ctx.register_hook("pre_tool_call", before_tool_call)' in source
    assert '"--target", "hermes"' in source
    assert "_BLOCK_FALLBACK" in source

    first = source
    hooks.install_hook("hermes", tmp_path)
    assert path.read_text(encoding="utf-8") == first

    other_root = tmp_path / "hermes-other"
    other_path = (
        other_root / ".hermes" / "plugins" / "agent-action-guard" / "__init__.py"
    )
    other_path.parent.mkdir(parents=True)
    other_path.write_text("def register(ctx): pass", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        hooks.install_hook("hermes", other_root)


def test_run_hook_dispatches_target(classifier):
    payload = {"tool_name": "Shell", "tool_input": {"dangerous": True}}
    assert hooks.run_hook("kiro", payload, 0.5)[0] == 2
    assert json.loads(hooks.run_hook("cursor", payload, 0.5)[1])["permission"] == "deny"
    assert hooks.run_hook("opencode", payload, 0.5)[0] == 2
    assert json.loads(hooks.run_hook("agy", payload, 0.5)[1])["decision"] == "deny"
    assert hooks.run_hook("copilot", payload, 0.5)[0] == 2
    assert json.loads(hooks.run_hook("openclaw", payload, 0.5)[1])["block"] is True
    assert json.loads(hooks.run_hook("hermes", payload, 0.5)[1])["action"] == "block"


def test_run_hook_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unsupported hook target"):
        hooks.run_hook("unknown", {"tool_name": "x"}, 0.5)


def test_install_hook_rejects_unknown_target(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported hook target"):
        hooks.install_hook("unknown", tmp_path)
