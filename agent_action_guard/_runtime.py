"""
Runtime helpers for action classification.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import openai
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_CLASSES = ["safe", "harmful", "unethical"]
MODEL_PATH = Path(__file__).with_name("action_classifier_model.pt")


class EmbeddingModel:
    """Thin wrapper over the configured embedding endpoint."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBED_MODEL_NAME", "")
        self.client = openai.OpenAI(base_url=os.getenv("EMBEDDING_BASE_URL"))

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


class ActionClassNet(nn.Module):
    """Simple MLP for multi-class classification."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 3),
        )
        self.net.to(DEVICE)

    def forward(self, x):
        return self.net(x)


embed_model = EmbeddingModel()
