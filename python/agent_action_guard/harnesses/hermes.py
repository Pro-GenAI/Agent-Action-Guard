"""Hermes Agent pre_tool_call hook adapter."""

from __future__ import annotations

import json
from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return Hermes pre_tool_call directive JSON."""
    result: HookResult = classify_hook_payload(payload, conf_threshold=conf_threshold)
    if result.blocked:
        return (
            0,
            json.dumps({"action": "block", "message": result.reason}),
            "",
        )
    return 0, "{}", ""


__all__ = ["handle"]
