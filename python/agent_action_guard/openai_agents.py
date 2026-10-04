"""OpenAI Agents SDK integration for Agent Action Guard."""

from __future__ import annotations

import copy
from typing import Any, Callable

from ._framework_utils import (
    check_tool_action,
    guard_callable,
    maybe_await,
    normalize_arguments,
    tool_name,
)


def guard_tool(tool: Any, *, conf_threshold: float = 0.5) -> Any:
    """Return an Action Guard-protected OpenAI Agents function tool."""
    invoke = getattr(tool, "on_invoke_tool", None)
    if invoke is None:
        if callable(tool):
            return guard_callable(tool, conf_threshold=conf_threshold)
        raise TypeError("tool must be callable or expose on_invoke_tool")

    guarded = copy.copy(tool)
    name = tool_name(tool)

    async def on_invoke_tool(context: Any, arguments: Any) -> Any:
        check_tool_action(
            name,
            normalize_arguments(arguments),
            conf_threshold=conf_threshold,
        )
        return await maybe_await(invoke(context, arguments))

    try:
        guarded.on_invoke_tool = on_invoke_tool
    except (AttributeError, TypeError, ValueError):
        object.__setattr__(guarded, "on_invoke_tool", on_invoke_tool)
    return guarded


def guard_tools(tools: list[Any], *, conf_threshold: float = 0.5) -> list[Any]:
    """Guard a list of OpenAI Agents SDK tools."""
    return [guard_tool(tool, conf_threshold=conf_threshold) for tool in tools]


def guard_function(
    func: Callable[..., Any], *, conf_threshold: float = 0.5
) -> Callable[..., Any]:
    """Guard a function before passing it to agents.function_tool."""
    return guard_callable(func, conf_threshold=conf_threshold)


__all__ = ["guard_function", "guard_tool", "guard_tools"]
