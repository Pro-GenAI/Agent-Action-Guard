#!/usr/bin/env python3
"""MCP server entrypoint.

This script provides an MCP server through an HTTP Stream transport server.
The server allows dynamic addition and removal of tools via HTTP endpoints,
allowing seamless evaluation through the HarmActEval framework's Python script.
"""

import os
from typing import Any

from agentor.mcp import LiteMCP
from agentor.mcp.api_router import ToolMetadata
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()
mcp = LiteMCP(name="mcp-server")

MCP_EVAL_SERVER_URL = os.getenv("MCP_EVAL_SERVER_URL", "")
if not MCP_EVAL_SERVER_URL:
    raise ValueError("MCP_EVAL_SERVER_URL environment variable not set.")

parsed_url = urlparse(MCP_EVAL_SERVER_URL)
SERVER_PORT = parsed_url.port or 0
if not SERVER_PORT:
    raise ValueError("MCP_EVAL_SERVER_URL must include a port.")


@mcp.tool(
    name="health-check",
    description="A test tool that logs its calls and returns a log string.",
)
async def health_check() -> str:
    """A test tool that logs its calls and returns a log string."""
    print("health_check called")
    return "The server is working correctly."

@mcp.tool(
    name="report-no-tool",
    description="A tool to report that no tool can be used for the given query.",
)
async def report_no_tool(user_query: str) -> str:
    """A tool to report that no tool can be used for the given query."""
    log_text = f"report_no_tool called with query: {user_query}"
    print("-"*40)
    print(log_text)
    print("-"*40)
    return log_text


execution_counts = {}  # tool_name: count

def call_tool(tool_name: str, *args, **kwargs) -> Any:
    """Wrapper to call a tool and track execution counts."""
    tool = mcp.tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found.")

    if tool_name not in execution_counts:
        execution_counts[tool_name] = 0
    execution_counts[tool_name] += 1

    print(f"Called {tool_name} with args: {args}, kwargs: {kwargs}")
    return f"Successfully called tool {tool_name}."


@mcp.app.post("/add-tool")
async def add_tool(tool_definition: dict) -> dict:
    """Endpoint to dynamically add a new tool to the MCP server."""

    print("Adding tool:", tool_definition)

    # Validate the tool
    tool_name = tool_definition.get("name", "")
    if not tool_name:
        return {"status": "error", "message": "Tool definition must include a 'name' field."}
    tool_description = tool_definition.get("description", "")
    if not tool_description:
        return {"status": "error", "message": "Tool definition must include a 'function' field."}
    schema = tool_definition.get("parameters", {})
    if not schema:
        return {"status": "error", "message": "Tool definition must include a 'parameters' field."}

    # Add the tool to the MCP server
    mcp.tools[tool_name] = ToolMetadata(
        func=lambda *args, **kwargs: call_tool(tool_name, *args, **kwargs),
        name=tool_name,
        description=tool_description,
        input_schema=schema,
        dependencies=None,
    )

    # Check that the tool was added
    if tool_name not in mcp.tools:
        return {"status": "error", "message": f"Failed to add tool {tool_name}."}

    return {"status": "success", "message": f"Tool {tool_definition.get('name')} added."}


@mcp.app.get("/execution-count")
async def get_execution_count(tool_name: str) -> dict:
    """Endpoint to get the execution count of a tool."""
    count = execution_counts.get(tool_name, 0)
    return {"tool_name": tool_name, "execution_count": count}


@mcp.app.post("/remove-tool")
async def remove_tool(tool_definition: dict) -> dict:
    """Endpoint to dynamically remove a tool from the MCP server."""

    # print("Received request to remove tool:", tool_definition)
    tool_name = tool_definition.get("name", "")
    if not tool_name:
        return {"status": "error", "message": "Tool definition must include a 'name' field."}
    print("Removing tool:", tool_name)

    if tool_name not in mcp.tools:
        print(f"Tool {tool_name} not found.")
        return {"status": "error", "message": f"Tool {tool_name} not found."}

    del mcp.tools[tool_name]
    if tool_name in execution_counts:
        del execution_counts[tool_name]
    return {"status": "success", "message": f"Tool {tool_name} removed."}


@mcp.app.get("/")
async def root() -> str:
    """Root endpoint to verify server is running."""
    return "MCP Eval Server is running."


if __name__ == "__main__":
    mcp.run(port=SERVER_PORT)
