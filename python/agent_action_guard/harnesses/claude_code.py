"""Claude Code PreToolUse hook adapter."""

from __future__ import annotations

import json
from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return exit code/stdout/stderr for a Claude Code PreToolUse hook."""
    result: HookResult = classify_hook_payload(
        payload, conf_threshold=conf_threshold
    )
    if not result.blocked:
        return 0, "", ""

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": result.reason,
        }
    }
    return 0, json.dumps(output), ""


__all__ = ["handle"]
