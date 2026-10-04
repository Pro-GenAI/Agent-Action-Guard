import asyncio

import pytest

from agent_action_guard import (
    HarmfulActionException,
    autogen,
    crewai,
    langchain,
    llamaindex,
    openai_agents,
)
from agent_action_guard import _framework_utils as framework_utils


@pytest.fixture
def classifier(monkeypatch):
    calls = []

    def fake(action):
        calls.append(action)
        arguments = action["function"]["arguments"]
        if arguments.get("dangerous"):
            return "harmful", 0.99
        return None, 0.91

    monkeypatch.setattr(framework_utils, "is_action_harmful", fake)
    return calls


def test_normalize_arguments_handles_json_and_scalars():
    assert framework_utils.normalize_arguments('{"city":"Rome"}') == {"city": "Rome"}
    assert framework_utils.normalize_arguments("plain text") == {"input": "plain text"}
    assert framework_utils.normalize_arguments([1, 2]) == {"input": [1, 2]}


def test_threshold_validation(classifier):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        framework_utils.check_tool_action("tool", {}, conf_threshold=1.1)


def test_langchain_callback_blocks_and_accepts_inputs(classifier):
    callback = langchain.ActionGuardCallbackHandler(conf_threshold=0.5)
    callback.on_tool_start({"name": "search"}, "ignored", inputs={"query": "safe"})
    assert classifier[-1]["function"] == {
        "name": "search",
        "arguments": {"query": "safe"},
    }

    with pytest.raises(HarmfulActionException):
        callback.on_tool_start({"name": "shell"}, '{"dangerous": true}')


def test_langchain_guarded_tool_sync_and_async(classifier):
    class Tool:
        name = "lookup"

        def invoke(self, value):
            return {"sync": value}

        async def ainvoke(self, value):
            return {"async": value}

    guarded = langchain.guard_tool(Tool())
    assert guarded.invoke({"x": 1}) == {"sync": {"x": 1}}
    assert asyncio.run(guarded.ainvoke({"x": 2})) == {"async": {"x": 2}}
    with pytest.raises(HarmfulActionException):
        guarded.invoke({"dangerous": True})


def test_llamaindex_guarded_tool_uses_metadata_name(classifier):
    class Metadata:
        name = "weather"

    class Tool:
        metadata = Metadata()

        def call(self, **kwargs):
            return kwargs

        async def acall(self, **kwargs):
            return kwargs

    guarded = llamaindex.guard_tool(Tool())
    assert guarded.call(city="Paris") == {"city": "Paris"}
    assert asyncio.run(guarded.acall(city="Paris")) == {"city": "Paris"}
    assert classifier[-1]["function"]["name"] == "weather"


def test_openai_agents_function_tool_copy_is_guarded(classifier):
    class Tool:
        name = "send_email"

        async def on_invoke_tool(self, context, arguments):
            return context, arguments

    original = Tool()
    guarded = openai_agents.guard_tool(original)
    result = asyncio.run(guarded.on_invoke_tool("ctx", '{"to":"a@example.com"}'))
    assert result == ("ctx", '{"to":"a@example.com"}')
    assert guarded is not original
    assert classifier[-1]["function"]["name"] == "send_email"

    with pytest.raises(HarmfulActionException):
        asyncio.run(guarded.on_invoke_tool("ctx", '{"dangerous": true}'))


def test_openai_agents_plain_function_preserves_sync_behavior(classifier):
    def tool(value=None, dangerous=False):
        return value

    guarded = openai_agents.guard_function(tool)
    assert guarded(value="ok") == "ok"
    with pytest.raises(HarmfulActionException):
        guarded(dangerous=True)


def test_autogen_tool_run_json_and_run_are_guarded(classifier):
    class Tool:
        name = "database"

        async def run_json(self, args, cancellation_token=None):
            return args, cancellation_token

        async def run(self, args, cancellation_token=None):
            return args, cancellation_token

    guarded = autogen.guard_tool(Tool())
    assert asyncio.run(guarded.run_json({"id": 1}, "token")) == ({"id": 1}, "token")
    assert asyncio.run(guarded.run({"id": 2}, "token")) == ({"id": 2}, "token")
    with pytest.raises(HarmfulActionException):
        asyncio.run(guarded.run_json({"dangerous": True}))


def test_crewai_tool_paths_are_guarded(classifier):
    class Tool:
        name = "report"

        def run(self, **kwargs):
            return kwargs

        def _run(self, **kwargs):
            return kwargs

        async def _arun(self, **kwargs):
            return kwargs

    guarded = crewai.guard_tool(Tool())
    assert guarded.run(topic="safe") == {"topic": "safe"}
    assert guarded._run(topic="safe") == {"topic": "safe"}
    assert asyncio.run(guarded._arun(topic="safe")) == {"topic": "safe"}
    with pytest.raises(HarmfulActionException):
        guarded.run(dangerous=True)


def test_async_guard_function(classifier):
    async def tool(*, item=None, dangerous=False):
        return item

    guarded = autogen.guard_function(tool)
    assert asyncio.run(guarded(item="ok")) == "ok"
    with pytest.raises(HarmfulActionException):
        asyncio.run(guarded(dangerous=True))
