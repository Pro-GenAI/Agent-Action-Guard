"""Cursor native preToolUse hook adapter."""

from __future__ import annotations

import json
from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return exit code/stdout/stderr for a Cursor preToolUse hook."""
    result: HookResult = classify_hook_payload(
        payload, conf_threshold=conf_threshold
    )
    if result.blocked:
        output = {
            "permission": "deny",
            "user_message": result.reason,
            "agent_message": result.reason,
        }
    else:
        output = {"permission": "allow"}
    return 0, json.dumps(output), ""


__all__ = ["handle"]
