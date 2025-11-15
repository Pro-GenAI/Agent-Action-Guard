import os
import time
from typing import Any, Optional
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agent_action_classifier import is_action_harmful

load_dotenv()

MODEL_ID = os.getenv("OPENAI_MODEL", "")
if not MODEL_ID:
    # allow server to start in test contexts but warn via exception on use
    raise ValueError("OPENAI_MODEL environment variable not set.")


app = FastAPI(title="OpenAI-Compatible API", version="1.0")


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
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


def _format_chat_response(text: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def get_response(messages: list[dict[str, str]], max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
    """Simple response generator for demo purposes.

    If a message content contains a JSON object with an `action` key, we will
    route it to the local action classifier and return a short classification
    summary. Otherwise, the function concatenates message contents.
    """
    # Find first message with a parsable JSON that includes 'action'
    for m in messages:
        content = m.get("content", "")
        # Try to parse raw JSON content
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "action" in parsed:
                is_harmful, conf = is_action_harmful(parsed["action"])
                return (
                    f"Action classification: is_harmful={is_harmful}, confidence={conf:.4f}"
                )
        except Exception:
            # not JSON — skip
            pass

    # Fallback: join user-visible content
    parts = [m.get("content", "") for m in messages if m.get("content")]
    return "\n".join(parts) if parts else ""


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Generate response
    response = get_response(request.messages, max_new_tokens=request.max_tokens,
                            temperature=request.temperature)

    return _format_chat_response(response)


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
