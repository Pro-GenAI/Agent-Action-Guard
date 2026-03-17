"""
Runtime helpers for action classification.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import openai

ALL_CLASSES = ["safe", "harmful", "unethical"]
MODEL_PATH = Path(__file__).with_name("action_classifier_model.pt")
ONNX_MODEL_PATH = Path(__file__).with_name("action_classifier_model.onnx")


class EmbeddingModel:
    """Thin wrapper over the configured embedding endpoint."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBED_MODEL_NAME", "")
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


def flatten_action_to_text(action_data):
    """Flatten action metadata to text."""
    parts = []
    parts.append(action_data.get("label", ""))
    parts.append(action_data.get("resource", ""))

    action_meta = action_data.get("action", {}) or {}
    parts.append(action_meta.get("server_label", ""))
    parts.append(action_meta.get("server_url", ""))
    parts.append(action_meta.get("require_approval", ""))

    action_params = action_meta.get("parameters") or {}
    parts.extend(sorted((str(key).lower() for key in action_params.keys())))

    function = action_data.get("function")
    if function:
        parts.append(function.get("name", ""))
        raw_args = function.get("arguments", "{}")
        action_args = (
            json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        )
        parts.extend(sorted((str(key).lower() for key in action_args.keys())))

    return " ".join([str(part).lower() for part in parts if part])


embed_model = EmbeddingModel()

class ActionGuardDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
