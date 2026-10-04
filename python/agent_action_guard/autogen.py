"""AutoGen integration for screening tool execution with Agent Action Guard."""

from __future__ import annotations

from typing import Any, Callable

from ._framework_utils import check_tool_action, guard_callable, maybe_await, tool_name


class GuardedTool:
    """Proxy AutoGen tool execution through Action Guard."""

    def __init__(self, tool: Any, *, conf_threshold: float = 0.5) -> None:
        self._tool = tool
        self.conf_threshold = conf_threshold

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def run_json(self, args: Any, *extra: Any, **kwargs: Any) -> Any:
        check_tool_action(
            tool_name(self._tool), args, conf_threshold=self.conf_threshold
        )
        return await maybe_await(self._tool.run_json(args, *extra, **kwargs))

    async def run(self, args: Any, *extra: Any, **kwargs: Any) -> Any:
        check_tool_action(
            tool_name(self._tool), args, conf_threshold=self.conf_threshold
        )
        return await maybe_await(self._tool.run(args, *extra, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        return self._tool(*args, **kwargs)


def guard_tool(tool: Any, *, conf_threshold: float = 0.5) -> GuardedTool:
    """Wrap an AutoGen tool object."""
    return GuardedTool(tool, conf_threshold=conf_threshold)


def guard_function(
    func: Callable[..., Any], *, conf_threshold: float = 0.5
) -> Callable[..., Any]:
    """Guard a callable before passing it to AutoGen FunctionTool."""
    return guard_callable(func, conf_threshold=conf_threshold)


__all__ = ["GuardedTool", "guard_function", "guard_tool"]
