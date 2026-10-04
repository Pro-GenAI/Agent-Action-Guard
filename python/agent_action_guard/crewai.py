"""CrewAI integration for screening tool execution with Agent Action Guard."""

from __future__ import annotations

from typing import Any, Callable

from ._framework_utils import check_tool_action, guard_callable, maybe_await, tool_name


class GuardedTool:
    """Proxy CrewAI tools while preserving public metadata via delegation."""

    def __init__(self, tool: Any, *, conf_threshold: float = 0.5) -> None:
        self._tool = tool
        self.conf_threshold = conf_threshold

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        return self._tool.run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        return self._tool._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        payload = kwargs if kwargs else {"args": list(args)}
        check_tool_action(
            tool_name(self._tool), payload, conf_threshold=self.conf_threshold
        )
        method = getattr(self._tool, "_arun", None)
        if method is None:
            return self._tool._run(*args, **kwargs)
        return await maybe_await(method(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if callable(self._tool):
            payload = kwargs if kwargs else {"args": list(args)}
            check_tool_action(
                tool_name(self._tool), payload, conf_threshold=self.conf_threshold
            )
            return self._tool(*args, **kwargs)
        return self.run(*args, **kwargs)


def guard_tool(tool: Any, *, conf_threshold: float = 0.5) -> GuardedTool:
    """Wrap a CrewAI tool object."""
    return GuardedTool(tool, conf_threshold=conf_threshold)


def guard_function(
    func: Callable[..., Any], *, conf_threshold: float = 0.5
) -> Callable[..., Any]:
    """Guard a function before decorating/registering it as a CrewAI tool."""
    return guard_callable(func, conf_threshold=conf_threshold)


__all__ = ["GuardedTool", "guard_function", "guard_tool"]
