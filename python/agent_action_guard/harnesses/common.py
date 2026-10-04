"""Shared pre-tool-use hook normalization and classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .._framework_utils import build_action
from ..action_classifier import is_action_harmful


@dataclass(frozen=True)
class HookResult:
    """Portable hook result rendered into a harness-specific protocol."""

    blocked: bool
    label: str | None
    confidence: float
    reason: str

    @property
    def allowed(self) -> bool:
        return not self.blocked


def load_hook_payload(text: str) -> dict[str, Any]:
    """Parse one JSON object received from a harness on stdin."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid hook JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Hook payload must be a JSON object")
    return payload


def normalize_hook_call(payload: dict[str, Any]) -> tuple[str, Any]:
    """Extract the tool name and arguments from common PreToolUse payloads."""
    tool_name = payload.get("tool_name") or payload.get("toolName")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("Hook payload is missing tool_name")

    if "tool_input" in payload:
        tool_input = payload["tool_input"]
    elif "toolInput" in payload:
        tool_input = payload["toolInput"]
    elif "input" in payload:
        tool_input = payload["input"]
    elif "toolArgs" in payload:
        tool_input = payload["toolArgs"]
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Hook payload tool arguments are invalid JSON"
                ) from exc
    else:
        tool_input = {}

    return tool_name, tool_input


def classify_hook_payload(
    payload: dict[str, Any],
    *,
    conf_threshold: float = 0.5,
) -> HookResult:
    """Classify a harness tool call without executing it."""
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0.0 and 1.0")

    tool_name, tool_input = normalize_hook_call(payload)
    label, confidence = is_action_harmful(build_action(tool_name, tool_input))
    blocked = bool(label and confidence >= conf_threshold)
    if blocked:
        reason = (
            f"Agent Action Guard blocked '{tool_name}': {label} "
            f"(confidence {confidence:.2f})."
        )
    else:
        reason = (
            f"Agent Action Guard allowed '{tool_name}' (confidence {confidence:.2f})."
        )
    return HookResult(
        blocked=blocked,
        label=label,
        confidence=confidence,
        reason=reason,
    )
