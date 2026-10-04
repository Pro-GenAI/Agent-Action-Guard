"""OpenClaw before_tool_call hook adapter."""

from __future__ import annotations

import json
from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return OpenClaw before_tool_call decision JSON."""
    result: HookResult = classify_hook_payload(payload, conf_threshold=conf_threshold)
    if result.blocked:
        return (
            0,
            json.dumps({"block": True, "blockReason": result.reason}),
            "",
        )
    return 0, json.dumps({"block": False}), ""


__all__ = ["handle"]
