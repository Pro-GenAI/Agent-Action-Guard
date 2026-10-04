"""LlamaIndex integration for screening tool calls with Agent Action Guard."""

from __future__ import annotations

from typing import Any

from ._framework_utils import check_tool_action, guard_callable, maybe_await, tool_name


class GuardedTool:
    """Proxy a LlamaIndex tool and guard call/acall execution paths."""

    def __init__(self, tool: Any, *, conf_threshold: float = 0.5) -> None:
        self._tool = tool
        self.conf_threshold = conf_threshold

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        return self._tool.call(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        method = getattr(self._tool, "acall", None)
        if method is None:
            return self._tool.call(*args, **kwargs)
        return await maybe_await(method(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if callable(self._tool):
            payload = kwargs if kwargs else {"args": list(args)}
            check_tool_action(
                tool_name(self._tool), payload, conf_threshold=self.conf_threshold
            )
            return self._tool(*args, **kwargs)
        return self.call(*args, **kwargs)


def guard_tool(tool: Any, *, conf_threshold: float = 0.5) -> GuardedTool:
    """Wrap one LlamaIndex tool."""
    return GuardedTool(tool, conf_threshold=conf_threshold)


def guard_tools(tools: list[Any], *, conf_threshold: float = 0.5) -> list[GuardedTool]:
    """Wrap a list of LlamaIndex tools."""
    return [guard_tool(tool, conf_threshold=conf_threshold) for tool in tools]


def guard_function(func: Any, *, conf_threshold: float = 0.5) -> Any:
    """Guard a function before passing it to LlamaIndex FunctionTool."""
    return guard_callable(func, conf_threshold=conf_threshold)


__all__ = ["GuardedTool", "guard_function", "guard_tool", "guard_tools"]
