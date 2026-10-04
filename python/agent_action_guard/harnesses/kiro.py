"""Kiro PreToolUse hook adapter."""

from __future__ import annotations

from typing import Any

from .common import HookResult, classify_hook_payload


def handle(
    payload: dict[str, Any], *, conf_threshold: float = 0.5
) -> tuple[int, str, str]:
    """Return exit code/stdout/stderr for a Kiro PreToolUse hook."""
    result: HookResult = classify_hook_payload(
        payload, conf_threshold=conf_threshold
    )
    if result.blocked:
        return 2, "", result.reason
    return 0, "", ""


__all__ = ["handle"]
