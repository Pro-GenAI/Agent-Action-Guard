"""Regression coverage for complete, fail-closed callable screening."""

import asyncio
import functools
import inspect
import json

import pytest

from agent_action_guard import _framework_utils as fw
from agent_action_guard import action_classifier as ac
from agent_action_guard import autogen, crewai, langchain, llamaindex, openai_agents


@pytest.fixture
def screening(monkeypatch):
    seen = []

    def classify(action):
        seen.append(action["function"]["arguments"])
        return (
            ("harmful", 0.99)
            if "BLOCK_MARKER" in json.dumps(seen[-1])
            else (None, 0.99)
        )

    monkeypatch.setattr(ac, "is_action_harmful", classify)
    monkeypatch.setattr(fw, "is_action_harmful", classify)
    return seen


GUARDS = [
    ac.action_guarded(),
    *(
        module.guard_function
        for module in [autogen, crewai, langchain, llamaindex, openai_agents]
    ),
]


@pytest.mark.parametrize("guard", GUARDS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_callable_guards_screen_positional_mixed_and_default_inputs(
    guard, asynchronous, screening
):
    executed = []

    def tool(command="BLOCK_MARKER", *, option=False):
        executed.append(command)
        return command

    async def async_tool(command="BLOCK_MARKER", *, option=False):
        return tool(command, option=option)

    wrapped = guard(async_tool if asynchronous else tool)
    assert inspect.iscoroutinefunction(wrapped) is asynchronous
    for args, kwargs in [
        ((), {}),
        (("BLOCK_MARKER",), {}),
        (("BLOCK_MARKER",), {"option": True}),
    ]:
        with pytest.raises(ac.HarmfulActionException):
            result = wrapped(*args, **kwargs)
            if asynchronous:
                asyncio.run(result)
    assert executed == []
    result = wrapped("safe", option=True)
    assert (asyncio.run(result) if asynchronous else result) == "safe"
    assert screening[-1] == {"command": "safe", "option": True}


@pytest.mark.parametrize("guard", GUARDS)
def test_guards_preserve_kwargs_and_variadic_content(guard, screening):
    def tool(*args, **kwargs):
        return args, kwargs

    wrapped = guard(tool)
    assert wrapped(value="safe") == ((), {"value": "safe"})
    assert screening[-1] == {"value": "safe"}
    with pytest.raises(ac.HarmfulActionException):
        wrapped("BLOCK_MARKER", args="safe")


@pytest.mark.parametrize("guard", GUARDS)
def test_guards_support_bound_methods_and_partial_calls(guard, screening):
    class Tool:
        def run(self, command, *, option=True):
            return command

    assert guard(Tool().run)("safe") == "safe"
    assert screening[-1] == {"command": "safe", "option": True}
    with pytest.raises(ac.HarmfulActionException):
        guard(functools.partial(Tool().run, "BLOCK_MARKER"))()


@pytest.mark.parametrize("guard", GUARDS)
def test_decorated_methods_skip_only_the_receiver_and_keep_self_named_inputs(
    guard, screening
):
    class Tool:
        @guard
        def run(self, command):
            return command

    assert Tool().run("safe") == "safe"
    assert screening[-1] == {"command": "safe"}
    with pytest.raises(ac.HarmfulActionException):
        guard(lambda self: self)("BLOCK_MARKER")


def test_signature_is_inspected_once_and_opaque_callables_keep_all_inputs(
    monkeypatch, screening
):
    from agent_action_guard import _guard_utils

    original = _guard_utils.inspect.signature
    inspected = []

    def signature(func):
        inspected.append(func)
        return original(func)

    monkeypatch.setattr(_guard_utils.inspect, "signature", signature)
    guarded = fw.guard_callable(lambda command: command)
    assert guarded("one") == "one"
    assert guarded("two") == "two"
    assert len(inspected) == 1

    def unavailable(_func):
        raise ValueError("No inspectable signature")

    monkeypatch.setattr(_guard_utils.inspect, "signature", unavailable)
    wrapped = fw.guard_callable(lambda *args, **kwargs: None)
    with pytest.raises(ac.HarmfulActionException):
        wrapped("BLOCK_MARKER", args="safe")


def test_async_concurrent_calls_block_independently(screening):
    executed = []

    @ac.action_guarded()
    async def tool(command):
        executed.append(command)
        return command

    async def run():
        return await asyncio.gather(
            tool("one"), tool("BLOCK_MARKER"), tool("two"), return_exceptions=True
        )

    results = asyncio.run(run())
    assert isinstance(results[1], ac.HarmfulActionException)
    assert executed == ["one", "two"]


@pytest.mark.skipif(
    not hasattr(functools, "Placeholder"), reason="Python 3.14 partial placeholders"
)
def test_partial_placeholders_bind_the_executed_inputs(screening):
    placeholder = getattr(functools, "Placeholder")
    wrapped = fw.guard_callable(
        functools.partial(lambda first, second: (first, second), placeholder, "safe")
    )
    assert wrapped("one") == ("one", "safe")
    assert screening[-1] == {"first": "one", "second": "safe"}
    with pytest.raises(ac.HarmfulActionException):
        wrapped("BLOCK_MARKER")
    with pytest.raises(TypeError):
        wrapped()


@pytest.mark.parametrize(
    "threshold", [float("nan"), float("inf"), -1, 1.01, "0.5", None, True]
)
def test_public_decorator_rejects_invalid_thresholds(threshold):
    with pytest.raises((ValueError, TypeError)):
        ac.action_guarded(conf_threshold=threshold)


@pytest.mark.parametrize(
    "module,method",
    [
        (autogen, "__call__"),
        (langchain, "__call__"),
        (llamaindex, "call"),
        (llamaindex, "acall"),
        (llamaindex, "__call__"),
        (crewai, "run"),
        (crewai, "_run"),
        (crewai, "_arun"),
        (crewai, "__call__"),
    ],
)
def test_framework_proxy_keeps_mixed_arguments(module, method, screening):
    class Tool:
        name = "audit_tool"

        def __call__(self, command, *, option=False):
            return command

        call = run = _run = __call__

        async def acall(self, command, *, option=False):
            return command

        _arun = acall

    wrapped = module.guard_tool(Tool())
    with pytest.raises(ac.HarmfulActionException):
        result = getattr(wrapped, method)("BLOCK_MARKER", option=True)
        if inspect.isawaitable(result):
            asyncio.run(fw.maybe_await(result))
