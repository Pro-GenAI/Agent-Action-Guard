from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
from harness_mock_server import (
    MARKER_NAME,
    MOCK_MODEL,
    HarnessMockServer,
    harmful_command,
)

from agent_action_guard import hooks
from agent_action_guard.action_classifier import is_action_harmful


@pytest.fixture
def mock_harness_server():
    server = HarnessMockServer().start()
    try:
        yield server
    finally:
        server.close()


def _require_cli(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        pytest.skip(f"{command} is not installed")
    return resolved


def _env() -> dict[str, str]:
    env = os.environ.copy()
    python_bin = Path(sys.executable).resolve()
    env["PATH"] = f"{python_bin.parent}:{env.get('PATH', '')}"
    return env


def _assert_dataset_shell_action_is_blocked(tool_name: str) -> None:
    label, confidence = is_action_harmful(
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": {"command": harmful_command()},
            },
        }
    )
    assert label == "harmful"
    assert confidence >= 0.5


def _install_hook_audit_shim(env: dict[str, str], root: Path) -> Path:
    real_cli = shutil.which("agent-action-guard", path=env.get("PATH"))
    if real_cli is None:
        pytest.fail("agent-action-guard console script is unavailable in the test environment")

    audit_path = root / "hook-audit.json"
    shim_dir = root / "hook-bin"
    shim_dir.mkdir()
    shim = shim_dir / "agent-action-guard"
    shim.write_text(
        "#!/bin/sh\n"
        'payload="$(cat)"\n'
        'printf "%s\\n" "$payload" >> "$AAG_HOOK_AUDIT_FILE"\n'
        f'printf "%s" "$payload" | {real_cli} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env["AAG_HOOK_AUDIT_FILE"] = str(audit_path)
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    return audit_path


def _assert_blocked(
    project: Path,
    completed: subprocess.CompletedProcess[str],
    audit_path: Path,
) -> None:
    assert not (project / MARKER_NAME).exists(), (
        f"harness executed the harmful tool\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )
    assert audit_path.exists(), (
        f"harness never invoked Agent Action Guard\nstdout={completed.stdout}"
        f"\nstderr={completed.stderr}"
    )
    payloads = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads
    assert any(MARKER_NAME in json.dumps(payload) for payload in payloads)


def test_mock_server_advertises_specific_model(mock_harness_server):
    with urllib.request.urlopen(
        f"{mock_harness_server.base_url}/v1/models", timeout=2
    ) as response:
        payload = json.load(response)
    assert [item["id"] for item in payload["data"]] == [MOCK_MODEL]


def test_harmful_fixture_is_derived_from_harmactionseval_and_blocked():
    _assert_dataset_shell_action_is_blocked("Bash")
    _assert_dataset_shell_action_is_blocked("shell")
    _assert_dataset_shell_action_is_blocked("execute_bash")


def test_codex_cli_blocks_mock_model_harmful_action(tmp_path, mock_harness_server):
    codex = _require_cli("codex")
    hooks.install_hook("codex", tmp_path)
    _assert_dataset_shell_action_is_blocked("shell")

    env = _env()
    env.update(
        {
            "AAG_MOCK_API_KEY": "test",
            "CODEX_HOME": str(tmp_path / ".codex"),
        }
    )
    audit_path = _install_hook_audit_shim(env, tmp_path)
    completed = subprocess.run(
        [
            codex,
            "exec",
            "--enable",
            "hooks",
            "--dangerously-bypass-hook-trust",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-m",
            MOCK_MODEL,
            "-c",
            'model_provider="aagmock"',
            "-c",
            'model_providers.aagmock.name="AAG Mock"',
            "-c",
            f'model_providers.aagmock.base_url="{mock_harness_server.openai_base_url}"',
            "-c",
            'model_providers.aagmock.env_key="AAG_MOCK_API_KEY"',
            "-c",
            'model_providers.aagmock.wire_api="responses"',
            "Use the shell tool now.",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _assert_blocked(tmp_path, completed, audit_path)


def test_claude_code_blocks_mock_model_harmful_action(tmp_path, mock_harness_server):
    claude = _require_cli("claude")
    hooks.install_hook("claude-code", tmp_path)
    _assert_dataset_shell_action_is_blocked("Bash")

    env = _env()
    env.update(
        {
            "ANTHROPIC_BASE_URL": mock_harness_server.base_url,
            "ANTHROPIC_API_KEY": "test",
        }
    )
    audit_path = _install_hook_audit_shim(env, tmp_path)
    completed = subprocess.run(
        [
            claude,
            "-p",
            "Use the Bash tool now.",
            "--settings",
            str(tmp_path / ".claude" / "settings.json"),
            "--model",
            MOCK_MODEL,
            "--allowedTools",
            "Bash",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _assert_blocked(tmp_path, completed, audit_path)


@pytest.mark.skip(reason="superseded by test_harness_cli_e2e.py Cursor CLI protocol coverage")
def test_cursor_cli_blocks_mock_model_harmful_action(tmp_path, mock_harness_server):
    agent = _require_cli("agent")
    hooks.install_hook("cursor", tmp_path)
    _assert_dataset_shell_action_is_blocked("Shell")

    env = _env()
    cursor_config_dir = tmp_path / ".cursor-test-config"
    cursor_config_dir.mkdir()
    (cursor_config_dir / "cli-config.json").write_text(
        json.dumps({"network": {"useHttp1ForAgent": True}}),
        encoding="utf-8",
    )
    env.update(
        {
            "HOME": str(tmp_path),
            "CURSOR_CONFIG_DIR": str(cursor_config_dir),
            "CURSOR_DATA_DIR": str(tmp_path / ".cursor-test-data"),
            "CURSOR_API_ENDPOINT": mock_harness_server.base_url,
            "CURSOR_API_BASE_URL": mock_harness_server.base_url,
            "CURSOR_API_KEY": "test",
        }
    )
    audit_path = _install_hook_audit_shim(env, tmp_path)
    completed = subprocess.run(
        [
            agent,
            "--dev-raw-model-slug",
            MOCK_MODEL,
            "--print",
            "--trust",
            "--force",
            "Use the shell tool now.",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _assert_blocked(tmp_path, completed, audit_path)


@pytest.mark.skip(reason="superseded by test_harness_cli_e2e.py Kiro capability-gated coverage")
def test_kiro_cli_blocks_mock_model_harmful_action(tmp_path, mock_harness_server):
    kiro = _require_cli("kiro-cli")
    hooks.install_hook("kiro", tmp_path)
    _assert_dataset_shell_action_is_blocked("shell")

    env = _env()
    env.update(
        {
            "HOME": str(tmp_path),
            "KIRO_HOME": str(tmp_path / ".kiro-test-home"),
            "KIRO_API_KEY": "test",
            "Q_DISABLE_TELEMETRY": "1",
        }
    )
    audit_path = _install_hook_audit_shim(env, tmp_path)
    endpoint = {"endpoint": mock_harness_server.base_url, "region": "us-east-1"}
    settings_path = Path(env["KIRO_HOME"]) / "settings" / "cli.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "api.codewhisperer.service": endpoint,
                "api.krs.service": endpoint,
                "api.cps.service": endpoint,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            kiro,
            "chat",
            "--no-interactive",
            "--trust-all-tools",
            "--model",
            MOCK_MODEL,
            "Use the shell tool now.",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _assert_blocked(tmp_path, completed, audit_path)
