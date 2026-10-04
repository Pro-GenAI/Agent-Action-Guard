"""Antigravity CLI (agy) BeforeTool hook adapter."""

from __future__ import annotations

import json
from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return Gemini-family hook JSON for an Antigravity BeforeTool hook."""
    result: HookResult = classify_hook_payload(payload, conf_threshold=conf_threshold)
    if result.blocked:
        return 0, json.dumps({"decision": "deny", "reason": result.reason}), ""
    return 0, json.dumps({"decision": "allow"}), ""


__all__ = ["handle"]
