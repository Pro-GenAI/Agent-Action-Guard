"""Shared helpers for optional agent-framework integrations."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Mapping
from typing import Any, Callable

from .action_classifier import HarmfulActionException, is_action_harmful


def normalize_arguments(value: Any) -> dict[str, Any]:
    """Normalize a framework-specific tool input into Action Guard arguments."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"input": value}
        if isinstance(decoded, Mapping):
            return dict(decoded)
        return {"input": decoded}
    return {"input": value}


def build_action(name: str, arguments: Any) -> dict[str, Any]:
    """Build the OpenAI-style function action consumed by Action Guard."""
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": normalize_arguments(arguments),
        },
    }


def check_tool_action(
    name: str,
    arguments: Any,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str | None, float]:
    """Classify a tool call and raise when a harmful result meets the threshold."""
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0.0 and 1.0")

    label, confidence = is_action_harmful(build_action(name, arguments))
    if label and confidence >= conf_threshold:
        raise HarmfulActionException(
            f"Tool action '{name}' classified as harmful ({label}) "
            f"with confidence {confidence:.2f}"
        )
    return label, confidence


def tool_name(tool: Any, default: str = "tool") -> str:
    """Best-effort extraction of a framework tool's public name."""
    name = getattr(tool, "name", None)
    if name:
        return str(name)

    metadata = getattr(tool, "metadata", None)
    metadata_name = getattr(metadata, "name", None)
    if metadata_name:
        return str(metadata_name)
    get_name = getattr(metadata, "get_name", None)
    if callable(get_name):
        try:
            resolved = get_name()
        except TypeError:
            resolved = None
        if resolved:
            return str(resolved)

    func_name = getattr(tool, "__name__", None)
    return str(func_name or default)


async def maybe_await(value: Any) -> Any:
    """Await *value* when needed, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def guard_callable(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    conf_threshold: float = 0.5,
) -> Callable[..., Any]:
    """Wrap a plain callable while preserving sync/async behavior and metadata."""
    action_name = name or getattr(func, "__name__", "tool")

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = kwargs if kwargs else {"args": list(args)}
            check_tool_action(action_name, payload, conf_threshold=conf_threshold)
            return await func(*args, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(action_name, payload, conf_threshold=conf_threshold)
        return func(*args, **kwargs)

    return wrapper
