"""Install and run Agent Action Guard harness hooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .harnesses.common import load_hook_payload

TARGETS = (
    "codex",
    "claude-code",
    "cursor",
    "kiro",
    "opencode",
    "agy",
    "copilot",
    "openclaw",
    "hermes",
)
_COMMAND = "agent-action-guard hooks run --target {target}"


def _read_json_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write_json_file(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _contains_command(items: list[Any], command: str) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(item.get(key) == command for key in ("command", "bash", "powershell")):
            return True
        hooks = item.get("hooks")
        if isinstance(hooks, list) and _contains_command(hooks, command):
            return True
    return False


def _install_codex(root: Path) -> Path:
    path = root / ".codex" / "hooks.json"
    data = _read_json_file(path, {})
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError(f"{path}: hooks must be a JSON object")
    entries = hooks.setdefault("PreToolUse", [])
    if not isinstance(entries, list):
        raise TypeError(f"{path}: hooks.PreToolUse must be a JSON array")
    command = _COMMAND.format(target="codex")
    if not _contains_command(entries, command):
        entries.append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": command,
                        "statusMessage": "Checking action safety",
                    }
                ],
            }
        )
    _write_json_file(path, data)
    return path


def _install_claude_code(root: Path) -> Path:
    path = root / ".claude" / "settings.json"
    data = _read_json_file(path, {})
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError(f"{path}: hooks must be a JSON object")
    entries = hooks.setdefault("PreToolUse", [])
    if not isinstance(entries, list):
        raise TypeError(f"{path}: hooks.PreToolUse must be a JSON array")
    command = _COMMAND.format(target="claude-code")
    if not _contains_command(entries, command):
        entries.append(
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "command",
                        "command": command,
                    }
                ],
            }
        )
    _write_json_file(path, data)
    return path


def _install_cursor(root: Path) -> Path:
    path = root / ".cursor" / "hooks.json"
    data = _read_json_file(path, {"version": 1})
    data.setdefault("version", 1)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError(f"{path}: hooks must be a JSON object")
    entries = hooks.setdefault("preToolUse", [])
    if not isinstance(entries, list):
        raise TypeError(f"{path}: hooks.preToolUse must be a JSON array")
    command = _COMMAND.format(target="cursor")
    if not _contains_command(entries, command):
        entries.append({"command": command})
    _write_json_file(path, data)
    return path


def _install_kiro(root: Path) -> Path:
    path = root / ".kiro" / "hooks" / "agent-action-guard.json"
    command = _COMMAND.format(target="kiro")
    data = {
        "version": "v1",
        "hooks": [
            {
                "name": "agent-action-guard",
                "description": "Block harmful tool calls before execution.",
                "trigger": "PreToolUse",
                "matcher": "*",
                "action": {"type": "command", "command": command},
                "enabled": True,
            }
        ],
    }
    _write_json_file(path, data)
    return path


def _install_opencode(root: Path) -> Path:
    path = root / ".opencode" / "plugins" / "agent-action-guard.js"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if "hooks run --target opencode" in text:
            return path
        raise ValueError(f"{path}: refusing to overwrite an existing unrelated plugin")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """import { spawnSync } from "node:child_process";

// agent-action-guard hooks run --target opencode
export const AgentActionGuard = async () => ({
  "tool.execute.before": async (input, output) => {
    const result = spawnSync(
      "agent-action-guard",
      ["hooks", "run", "--target", "opencode"],
      {
        input: JSON.stringify({
          tool_name: input.tool,
          tool_input: output.args,
        }),
        encoding: "utf8",
      },
    );
    if (result.error) throw result.error;
    if (result.status !== 0) {
      throw new Error(
        (result.stderr || result.stdout || "Agent Action Guard blocked this tool call.").trim(),
      );
    }
  },
});
""",
        encoding="utf-8",
    )
    return path


def _install_agy(root: Path) -> Path:
    path = root / ".agents" / "hooks.json"
    data = _read_json_file(path, {})
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError(f"{path}: hooks must be a JSON object")
    entries = hooks.setdefault("BeforeTool", [])
    if not isinstance(entries, list):
        raise TypeError(f"{path}: hooks.BeforeTool must be a JSON array")
    command = _COMMAND.format(target="agy")
    if not _contains_command(entries, command):
        entries.append(
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "name": "agent-action-guard",
                        "type": "command",
                        "command": command,
                        "timeout": 10000,
                    }
                ],
            }
        )
    _write_json_file(path, data)
    return path


def _install_copilot(root: Path) -> Path:
    path = root / ".github" / "hooks" / "agent-action-guard.json"
    data = _read_json_file(path, {"version": 1})
    data.setdefault("version", 1)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise TypeError(f"{path}: hooks must be a JSON object")
    entries = hooks.setdefault("preToolUse", [])
    if not isinstance(entries, list):
        raise TypeError(f"{path}: hooks.preToolUse must be a JSON array")
    command = _COMMAND.format(target="copilot")
    if not _contains_command(entries, command):
        entries.append(
            {
                "type": "command",
                "bash": command,
                "powershell": command,
                "cwd": ".",
                "timeoutSec": 10,
            }
        )
    _write_json_file(path, data)
    return path


def _install_openclaw(root: Path) -> Path:
    plugin_dir = root / ".openclaw" / "extensions" / "agent-action-guard"
    entry_path = plugin_dir / "index.js"
    marker = "hooks run --target openclaw"
    if entry_path.exists():
        text = entry_path.read_text(encoding="utf-8")
        if marker not in text:
            raise ValueError(
                f"{entry_path}: refusing to overwrite an existing unrelated plugin"
            )

    plugin_dir.mkdir(parents=True, exist_ok=True)
    _write_json_file(
        plugin_dir / "openclaw.plugin.json",
        {
            "id": "agent-action-guard",
            "name": "Agent Action Guard",
            "description": "Block harmful tool calls before execution.",
            "activation": {"onStartup": True},
            "configSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    )
    _write_json_file(
        plugin_dir / "package.json",
        {
            "name": "agent-action-guard-openclaw",
            "version": "1.0.0",
            "private": True,
            "type": "module",
            "openclaw": {"extensions": ["./index.js"]},
        },
    )
    entry_path.write_text(
        """import { spawnSync } from "node:child_process";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// agent-action-guard hooks run --target openclaw
const BLOCK_FALLBACK = "Agent Action Guard failed closed before tool execution.";

function runGuard(event) {
  const result = spawnSync(
    "agent-action-guard",
    ["hooks", "run", "--target", "openclaw"],
    {
      input: JSON.stringify({
        tool_name: event.toolName,
        tool_input: event.params,
      }),
      encoding: "utf8",
      timeout: 30000,
      maxBuffer: 1024 * 1024,
    },
  );

  if (result.error) {
    return { block: true, blockReason: `${BLOCK_FALLBACK} ${result.error.message}` };
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    return { block: true, blockReason: detail || BLOCK_FALLBACK };
  }

  try {
    const decision = JSON.parse(result.stdout || "{}");
    if (decision && decision.block === true) {
      return {
        block: true,
        blockReason: String(decision.blockReason || BLOCK_FALLBACK),
      };
    }
    if (decision && decision.block === false) {
      return;
    }
  } catch (error) {
    return { block: true, blockReason: `${BLOCK_FALLBACK} ${error.message}` };
  }

  return { block: true, blockReason: `${BLOCK_FALLBACK} Invalid hook response.` };
}

export default definePluginEntry({
  id: "agent-action-guard",
  name: "Agent Action Guard",
  register(api) {
    api.on("before_tool_call", async (event) => runGuard(event), {
      priority: 1000,
      timeoutMs: 35000,
    });
  },
});
""",
        encoding="utf-8",
    )
    return entry_path


def _install_hermes(root: Path) -> Path:
    plugin_dir = root / ".hermes" / "plugins" / "agent-action-guard"
    entry_path = plugin_dir / "__init__.py"
    marker = "hooks run --target hermes"
    if entry_path.exists():
        text = entry_path.read_text(encoding="utf-8")
        if marker not in text:
            raise ValueError(
                f"{entry_path}: refusing to overwrite an existing unrelated plugin"
            )

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        """name: agent-action-guard
version: "1.0"
description: Block harmful tool calls before execution.
provides_hooks:
  - pre_tool_call
""",
        encoding="utf-8",
    )
    entry_path.write_text(
        '''"""Agent Action Guard project plugin for Hermes Agent."""

from __future__ import annotations

import json
import subprocess

# agent-action-guard hooks run --target hermes
_COMMAND = ["agent-action-guard", "hooks", "run", "--target", "hermes"]
_BLOCK_FALLBACK = "Agent Action Guard failed closed before tool execution."


def _run_guard(tool_name, args):
    try:
        result = subprocess.run(
            _COMMAND,
            input=json.dumps({"tool_name": tool_name, "tool_input": args}),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"action": "block", "message": f"{_BLOCK_FALLBACK} {exc}"}

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return {"action": "block", "message": detail or _BLOCK_FALLBACK}

    try:
        decision = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"action": "block", "message": f"{_BLOCK_FALLBACK} {exc}"}

    if isinstance(decision, dict):
        if decision.get("action") == "block" and decision.get("message"):
            return decision
        if not decision:
            return None

    return {
        "action": "block",
        "message": f"{_BLOCK_FALLBACK} Invalid hook response.",
    }


def register(ctx):
    def before_tool_call(tool_name, args, task_id="", **kwargs):
        del task_id, kwargs
        return _run_guard(tool_name, args)

    ctx.register_hook("pre_tool_call", before_tool_call)
''',
        encoding="utf-8",
    )
    return entry_path


def install_hook(target: str, root: Path) -> Path:
    """Install or merge one project-level harness hook configuration."""
    installers = {
        "codex": _install_codex,
        "claude-code": _install_claude_code,
        "cursor": _install_cursor,
        "kiro": _install_kiro,
        "opencode": _install_opencode,
        "agy": _install_agy,
        "copilot": _install_copilot,
        "openclaw": _install_openclaw,
        "hermes": _install_hermes,
    }
    try:
        installer = installers[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported hook target: {target}") from exc
    return installer(root)


def run_hook(
    target: str, payload: dict[str, Any], conf_threshold: float
) -> tuple[int, str, str]:
    """Run one harness adapter."""
    if target == "codex":
        from .harnesses.codex import handle
    elif target == "claude-code":
        from .harnesses.claude_code import handle
    elif target == "cursor":
        from .harnesses.cursor import handle
    elif target == "kiro":
        from .harnesses.kiro import handle
    elif target == "opencode":
        from .harnesses.opencode import handle
    elif target == "agy":
        from .harnesses.agy import handle
    elif target == "copilot":
        from .harnesses.copilot import handle
    elif target == "openclaw":
        from .harnesses.openclaw import handle
    elif target == "hermes":
        from .harnesses.hermes import handle
    else:
        raise ValueError(f"Unsupported hook target: {target}")
    return handle(payload, conf_threshold=conf_threshold)


def build_hooks_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-action-guard hooks",
        description="Install or run pre-tool-use hooks for agent harnesses.",
    )
    subparsers = parser.add_subparsers(dest="hooks_command", required=True)

    install_parser = subparsers.add_parser(
        "install", help="Install a project-level harness hook"
    )
    install_parser.add_argument("--target", choices=TARGETS, required=True)
    install_parser.add_argument(
        "--project-root",
        default=".",
        help="Project root where harness configuration is written",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run a hook adapter against JSON received on stdin"
    )
    run_parser.add_argument("--target", choices=TARGETS, required=True)
    run_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=float(os.getenv("AAG_HOOK_CONF_THRESHOLD", "0.5")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_hooks_parser()
    args = parser.parse_args(argv)

    if args.hooks_command == "install":
        try:
            path = install_hook(args.target, Path(args.project_root).resolve())
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(path)
        return 0

    try:
        payload = load_hook_payload(sys.stdin.read())
        code, stdout, stderr = run_hook(
            args.target,
            payload,
            conf_threshold=args.conf_threshold,
        )
    except (TypeError, ValueError) as exc:
        print(f"agent-action-guard hook error: {exc}", file=sys.stderr)
        return 2

    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return code


__all__ = [
    "TARGETS",
    "build_hooks_parser",
    "install_hook",
    "main",
    "run_hook",
]
