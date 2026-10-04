"""Install and run Agent Action Guard harness hooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .harnesses.common import load_hook_payload

TARGETS = ("codex", "claude-code", "cursor", "kiro")
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
        if item.get("command") == command:
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


def install_hook(target: str, root: Path) -> Path:
    """Install or merge one project-level harness hook configuration."""
    installers = {
        "codex": _install_codex,
        "claude-code": _install_claude_code,
        "cursor": _install_cursor,
        "kiro": _install_kiro,
    }
    try:
        installer = installers[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported hook target: {target}") from exc
    return installer(root)


def run_hook(target: str, payload: dict[str, Any], conf_threshold: float) -> tuple[int, str, str]:
    """Run one harness adapter."""
    if target == "codex":
        from .harnesses.codex import handle
    elif target == "claude-code":
        from .harnesses.claude_code import handle
    elif target == "cursor":
        from .harnesses.cursor import handle
    elif target == "kiro":
        from .harnesses.kiro import handle
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
