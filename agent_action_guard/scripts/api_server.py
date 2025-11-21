import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
import openai
from pydantic import BaseModel
import uvicorn


load_dotenv()

model_name = os.getenv("OPENAI_MODEL", "")
if not model_name:
    # allow server to start in test contexts but warn via exception on use
    raise ValueError("OPENAI_MODEL environment variable not set.")

mcp_url = os.getenv("MCP_URL", "")
if not mcp_url:
    raise ValueError("MCP_URL environment variable not set.")

app = FastAPI()
openai_client = openai.OpenAI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class PredictRequest(BaseModel):
    # model is optional; defaults to server MODEL_ID
    model: Optional[str] = None
    action: dict[str, Any]


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": model_name + "-GUARDED", "object": "model"}]}


SYSTEM_MESSAGE_TOOL_USE = "You must use tools for every answer. Send the result to the user"

def get_response(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Simple response generator for demo purposes.

    If the model's response contains a JSON object with an `action` key, we will
    route it to the local action classifier and return a short classification
    summary. Otherwise, return the model's response.
    """
    messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE_TOOL_USE})
    for attempt in range(1):
        try:
            response = openai_client.chat.completions.create(
                messages=messages,  # type: ignore
                model=model_name,
                tools=[
                    {
                        "type": "mcp",
                        "server_label": "cats",
                        "server_url": mcp_url,
                        "require_approval": "never",
                    }  # type: ignore
                ],
                reasoning_effort="low" if "gpt-oss" in model_name.lower() else "minimal",  # type: ignore
                tool_choice="auto",
            )
            if not response:
                continue
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = message.tool_calls or []
            if not content and not tool_calls:
                continue
            return {"content": content, "tool_calls": tool_calls}
        except openai.BadRequestError as ex:
            if ex.message and ex.message == "tool_use_failed":
                continue
            print(f"Error getting response (attempt {attempt + 1}): {ex}")
            time.sleep(1)
    return {"content": "Error: Unable to get response.", "tool_calls": []}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Generate response
    response = get_response(request.messages)
    content = response.get("content", "")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def start_server(host="0.0.0.0", port=8001):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
