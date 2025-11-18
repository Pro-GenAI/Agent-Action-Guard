import os
import time
from typing import Any, Optional
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel
import uvicorn

from agent_action_classifier import is_action_harmful

load_dotenv()

model_name = os.getenv("OPENAI_MODEL", "")
if not model_name:
    # allow server to start in test contexts but warn via exception on use
    raise ValueError("OPENAI_MODEL environment variable not set.")


app = FastAPI(title="OpenAI-Compatible API", version="1.0")
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
    return {"data": [{"id": model_name, "object": "model"}]}


def get_response(messages: list[dict[str, str]]) -> str:
    """Simple response generator for demo purposes.

    If the model's response contains a JSON object with an `action` key, we will
    route it to the local action classifier and return a short classification
    summary. Otherwise, return the model's response.
    """
    response = openai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
    )
    tool_calls = response.choices[0].message.tool_calls or []

    for tool_call in tool_calls:
        if hasattr(tool_call, "function"):
            func_name = tool_call.function.name  # type: ignore
            args = tool_call.function.arguments  # type: ignore
        elif hasattr(tool_call, "custom"):
            func_name = tool_call.custom.name  # type: ignore
            args = tool_call.custom.input  # type: ignore
        else:
            print("WARNING: unrecognized tool_call format:", tool_call)
            continue

        is_harmful, conf = is_action_harmful({
            "label": "mcp",
            "resource": func_name,
            "action": {
                "server_label": func_name,
                "parameters": args,
            }
        })
        return (
            f"Action classification: is_harmful={is_harmful}, confidence={conf:.4f}"
        )

    return response.choices[0].message.content or ""


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Generate response
    response = get_response(request.messages)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    """Predict whether the provided action (dict) is harmful.

    Request body: {"model": "<model-id>", "action": {...}}
    Response: {"model": "<model-id>", "is_harmful": bool, "confidence": float}
    """
    try:
        is_harmful, confidence = is_action_harmful(req.action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"is_harmful": is_harmful, "confidence": confidence}


def start_server(host="0.0.0.0", port=8001):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
