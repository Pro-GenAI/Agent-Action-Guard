import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
import openai
from pydantic import BaseModel
import uvicorn


load_dotenv()

model_name = os.getenv("OPENAI_MODEL", "")
if not model_name:
    # allow server to start in test contexts but warn via exception on use
    raise ValueError("OPENAI_MODEL environment variable not set.")

model_name_guarded = model_name + "-GUARDED"

mcp_url_direct = os.getenv("MCP_URL_DIRECT", "")
if not mcp_url_direct:
    raise ValueError("MCP_URL_DIRECT environment variable not set.")

mcp_url_guarded = os.getenv("MCP_URL_GUARDED", "")
if not mcp_url_guarded:
    raise ValueError("MCP_URL_GUARDED environment variable not set.")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "")
if not BACKEND_BASE_URL:
    raise ValueError("BACKEND_BASE_URL environment variable is not set.")

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
if not BACKEND_API_KEY:
    raise ValueError("BACKEND_API_KEY environment variable is not set.")


API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    api_key_header = api_key_header.lstrip("Bearer ").strip()
    if api_key_header == BACKEND_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )


app = FastAPI()
openai_client = openai.OpenAI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


SYSTEM_MESSAGE_TOOL_USE = "You must use tools for every answer. Send the result to the user."

def get_response(messages: list[dict[str, str]], model: str) -> dict[str, Any]:
    """Simple response generator for demo purposes.

    If the model's response contains a JSON object with an `action` key, we will
    route it to the local action classifier and return a short classification
    summary. Otherwise, return the model's response.
    """
    if not model:
        raise ValueError("Model name must be provided to get_response.")
    if model == model_name:
        mcp_url = mcp_url_direct
    elif model == model_name_guarded:
        mcp_url = mcp_url_guarded
    else:
        raise ValueError(f"Unknown model name: {model}")

    messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE_TOOL_USE})
    for attempt in range(1):
        try:
            response = openai_client.chat.completions.create(
                messages=messages,  # type: ignore
                model=model_name,
                tools=[
                    {
                        "type": "mcp",
                        "server_label": "mcp_server",
                        "server_url": mcp_url,
                        "require_approval": "never",
                    }  # type: ignore
                ],
                reasoning_effort="low" if "gpt-oss" in model.lower() else "minimal",  # type: ignore
                tool_choice="auto",
            )
            if not response:
                continue
            message = response.choices[0].message
            if not message.content and not message.tool_calls:
                continue
            return {"content": message.content, "tool_calls": message.tool_calls}
        except openai.BadRequestError as ex:
            if ex.message and ex.message == "tool_use_failed":
                continue
            print(f"Error getting response (attempt {attempt + 1}): {ex}")
            time.sleep(1)
    return {"content": "Error: Unable to get response.", "tool_calls": []}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    # Generate response
    response = get_response(request.messages, request.model)
    print("Tool calls:", response["tool_calls"])

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response["content"],
                            "tool_calls": response["tool_calls"]},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/models")
def list_models(api_key: str = Depends(get_api_key)):
    return {"data": [
        {"id": model_name_guarded, "object": "model"},
        {"id": model_name, "object": "model"},
    ]}

@app.get("/")
def read_root():
    return {"message": "API Server is running."}


def start_server(port=8000):
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # extract port from BACKEND_BASE_URL
    from urllib.parse import urlparse
    parsed_url = urlparse(BACKEND_BASE_URL)
    port = parsed_url.port or 8000

    start_server(port=port)
