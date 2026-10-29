"""LangChain integration for screening tool calls with Agent Action Guard."""

# Public keyword names mirror LangChain's SDK.
# pylint: disable=redefined-builtin

from __future__ import annotations

from typing import Any

from ._framework_utils import check_tool_action, guard_callable, maybe_await, tool_name
from ._guard_utils import argument_builder

try:  # pragma: no cover - optional dependency
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler

    BaseCallbackHandler: Any = _BaseCallbackHandler
except ImportError:  # pragma: no cover - optional dependency
    BaseCallbackHandler = object


class ActionGuardCallbackHandler(BaseCallbackHandler):
    """LangChain callback that raises before a harmful tool call executes."""

    raise_error = True
    run_inline = True

    def __init__(self, *, conf_threshold: float = 0.5) -> None:
        self.conf_threshold = conf_threshold

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        name = serialized.get("name") or serialized.get("id") or "tool"
        arguments: Any = inputs if inputs is not None else input_str
        check_tool_action(str(name), arguments, conf_threshold=self.conf_threshold)


class GuardedTool:
    """Transparent LangChain-style tool proxy for direct invocation."""

    def __init__(self, tool: Any, *, conf_threshold: float = 0.5) -> None:
        self._tool = tool
        self.conf_threshold = conf_threshold
        self._call_arguments = argument_builder(tool) if callable(tool) else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def invoke(
        self, input: Any, *args: Any, **kwargs: Any
    ) -> Any:  # pylint: disable=redefined-builtin
        check_tool_action(
            tool_name(self._tool), input, conf_threshold=self.conf_threshold
        )
        return self._tool.invoke(input, *args, **kwargs)

    async def ainvoke(
        self, input: Any, *args: Any, **kwargs: Any
    ) -> Any:  # pylint: disable=redefined-builtin
        check_tool_action(
            tool_name(self._tool), input, conf_threshold=self.conf_threshold
        )
        method = getattr(self._tool, "ainvoke", None)
        if method is None:
            return self._tool.invoke(input, *args, **kwargs)
        return await maybe_await(method(input, *args, **kwargs))

    def __call__(
        self, input: Any, *args: Any, **kwargs: Any
    ) -> Any:  # pylint: disable=redefined-builtin
        if callable(self._tool):
            arguments = (
                self._call_arguments((input, *args), kwargs)
                if (args or kwargs) and self._call_arguments is not None
                else input
            )
            check_tool_action(
                tool_name(self._tool), arguments, conf_threshold=self.conf_threshold
            )
            return self._tool(input, *args, **kwargs)
        return self.invoke(input, *args, **kwargs)


def guard_tool(tool: Any, *, conf_threshold: float = 0.5) -> GuardedTool:
    """Wrap one LangChain tool without requiring LangChain at import time."""
    return GuardedTool(tool, conf_threshold=conf_threshold)


def guard_tools(tools: list[Any], *, conf_threshold: float = 0.5) -> list[GuardedTool]:
    """Wrap a list of LangChain tools."""
    return [guard_tool(tool, conf_threshold=conf_threshold) for tool in tools]


def guard_function(func: Any, *, conf_threshold: float = 0.5) -> Any:
    """Guard a function before passing it to LangChain tool factories/decorators."""
    return guard_callable(func, conf_threshold=conf_threshold)


__all__ = [
    "ActionGuardCallbackHandler",
    "GuardedTool",
    "guard_function",
    "guard_tool",
    "guard_tools",
]
