"""
Runtime helpers for action classification.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import openai

ALL_CLASSES = ["safe", "harmful", "unethical"]
ONNX_MODEL_PATH = Path(__file__).with_name("action_classifier_model.onnx")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
if not EMBED_MODEL_NAME:
    raise ValueError("EMBED_MODEL_NAME environment variable not set.")


class EmbeddingModel:
    """Thin wrapper over the configured embedding endpoint."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBED_MODEL_NAME
        client_kwargs = {}
        self.api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key

        # Instantiate the OpenAI client with optional base_url and api_key.
        self.client = openai.OpenAI(**client_kwargs)

    def encode(self, texts, *_args, **_kwargs):
        if not self.model_name:
            raise ValueError("EMBED_MODEL_NAME environment variable not set.")

        responses = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embs = [data.embedding for data in responses.data]
        return np.array(embs)


def flatten_action_to_text(action_data: Dict[str, str | Dict]) -> str:
    """Flatten action metadata to text."""
    # Example of action_data
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "data_exporter",
    #         "arguments": "{\"dataset\":\"employee_salaries\",\"destination\":\"xyz\"}"
    #     }
    # }
    # Convert to "Call function data_exporter: dataset=employee_salaries, destination=xyz"
    if action_data["type"] != "function":
        return f"Perform action: {action_data}"

    if not isinstance(action_data["function"], dict):
        return f"Call function with data: {action_data['function']}"

    func_name = action_data["function"].get("name", "unknown_function")
    args_raw = action_data["function"].get("arguments", {})
    try:
        if isinstance(args_raw, dict):
            args_dict = args_raw
        elif isinstance(args_raw, str):
            args_dict = json.loads(args_raw)
        else:
            return f"Call function {func_name} with arguments: {args_raw}"

        args_flat = ", ".join(f"{k}={v}" for k, v in args_dict.items())
        return f"Call function {func_name}: {args_flat}"
    except (json.JSONDecodeError, TypeError):
        return f"Call function {func_name} with arguments: {args_raw}"


embed_model = EmbeddingModel()

class ActionGuardDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
