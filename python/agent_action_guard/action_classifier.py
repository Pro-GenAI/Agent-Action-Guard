"""
Action Classifier Module (ONNX Runtime)
Lightweight inference without PyTorch dependency.
"""

from __future__ import annotations

import functools

import numpy as np
import onnxruntime as ort

from ._runtime_utils import (
    ALL_CLASSES,
    ONNX_MODEL_PATH,
    embed_model,
    flatten_action_to_text,
)


class ActionClassifier:
    """Classifier for AI agent actions using embeddings and ONNX model."""

    def __init__(self, embedding_model=None):
        self.embedding_model = (
            embed_model if embedding_model is None else embedding_model
        )
        self.session: ort.InferenceSession | None = None
        self.load_model()

    def load_model(self):
        """Load ONNX model."""
        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

        # Create inference session
        self.session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, action_dict: dict) -> tuple[str, float]:
        """Predict the class and confidence of one action."""
        return self.predict_batch([action_dict])[0]

    def predict_batch(
        self, action_dicts: list[dict], batch_size: int | None = None
    ) -> list[tuple[str, float]]:
        """Predict classes for multiple actions using vectorized embedding/inference."""
        if self.session is None:
            raise RuntimeError("ONNX model not loaded")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not action_dicts:
            return []

        chunk_size = batch_size or len(action_dicts)
        predictions: list[tuple[str, float]] = []
        for start in range(0, len(action_dicts), chunk_size):
            chunk = action_dicts[start : start + chunk_size]
            texts = [flatten_action_to_text(action_dict) for action_dict in chunk]
            embedding_model = getattr(self, "embedding_model", embed_model)
            embeddings = embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding_array = np.asarray(embeddings, dtype=np.float32)
            if embedding_array.ndim != 2 or embedding_array.shape[0] != len(chunk):
                raise ValueError("Embedding model returned an invalid batch payload")

            outputs = self.session.run(None, {"input": embedding_array})
            logits = np.asarray(outputs[0])
            if logits.ndim != 2 or logits.shape[0] != len(chunk):
                raise ValueError("Classifier returned an invalid batch logits payload")

            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            pred_class_indices = np.argmax(logits, axis=1)

            predictions.extend(
                (
                    ALL_CLASSES[int(pred_class_idx)],
                    float(probs[row_index, int(pred_class_idx)]),
                )
                for row_index, pred_class_idx in enumerate(pred_class_indices)
            )

        return predictions


# Lazy global classifier accessor
classifier = ActionClassifier()


def is_action_harmful(action_dict) -> tuple[str | None, float]:
    """Convenience function to check if an action is harmful."""
    pred_class, confidence = classifier.predict(action_dict)
    if pred_class == "safe":
        return None, confidence
    return pred_class, confidence


def is_actions_harmful(
    action_dicts: list[dict], batch_size: int | None = None
) -> list[tuple[str | None, float]]:
    """Classify multiple actions, returning ``None`` for safe action labels."""
    return [
        (None if pred_class == "safe" else pred_class, confidence)
        for pred_class, confidence in classifier.predict_batch(
            action_dicts, batch_size=batch_size
        )
    ]


class HarmfulActionException(ValueError):
    """Exception raised when a harmful action is detected."""


def ensure_action_safety(action_dict, raise_exception: bool = False) -> bool:
    """Raise an exception if the action is harmful."""
    is_harmful, confidence = is_action_harmful(action_dict)
    if is_harmful and raise_exception:
        raise HarmfulActionException(
            f"Action classified as harmful ({is_harmful}) with confidence {confidence:.2f}"
        )
    return not is_harmful


def action_guarded(conf_threshold: float = 0.5):
    """
    Decorator to guard a function by treating it as an AI tool.
    Uses func.__name__ as the tool name and kwargs as arguments for safety check.
    If the action is harmful and confidence >= conf_threshold, it raises HarmfulActionException.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Construct action_dict from function name and kwargs
            action_dict = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "arguments": kwargs,
                },
            }

            is_harmful, confidence = is_action_harmful(action_dict)
            if is_harmful and confidence >= conf_threshold:
                raise HarmfulActionException(
                    f"Guarded action '{func.__name__}' classified as harmful "
                    f"({is_harmful}) with confidence {confidence:.2f}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    sample_action = {
        "action_id": 1,
        "prompt": "Retrieve current weather data for New York City.",
        "action": {
            "type": "function",
            "function": {
                "name": "weather_api",
                "arguments": {"location": "New York City", "units": "imperial"},
            },
        },
    }
    example_classification, example_confidence = is_action_harmful(sample_action)
    if example_classification:
        example_classification = "Harmful"
    else:
        example_classification = "Safe"

    print(
        f"{sample_action['label']} Classification: {example_classification}, "
        f"Confidence: {example_confidence:.2f}"
    )

    # Latency test
    import time

    start_time = time.time()
    attempts = 500
    for _ in range(attempts):
        is_action_harmful(sample_action)
    end_time = time.time()
    avg_latency_sec = (end_time - start_time) / attempts
    avg_latency_ms = avg_latency_sec * 1000
    print(f"Average latency over {attempts} runs: {avg_latency_ms:.2f} ms")
