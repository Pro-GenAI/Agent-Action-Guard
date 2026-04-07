"""
Action Classifier Module (ONNX Runtime)
Lightweight inference without PyTorch dependency.
"""

from typing import Optional, Tuple
import functools

import numpy as np
import onnxruntime as ort

from agent_action_guard._runtime_utils import (
    ALL_CLASSES,
    ONNX_MODEL_PATH,
    embed_model,
    flatten_action_to_text,
)


class ActionClassifier:
    """Classifier for AI agent actions using embeddings and ONNX model."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
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

    def predict(self, action_dict: dict) -> Tuple[str, float]:
        """
        Predict the class of an action.
        Returns: (class: str, confidence: float)
        """
        if self.session is None:
            raise RuntimeError("ONNX model not loaded")

        # Convert action → text
        text = flatten_action_to_text(action_dict)

        # Generate embedding
        embedding = embed_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Ensure correct dtype/shape
        embedding = np.asarray(embedding, dtype=np.float32)

        # Run inference
        outputs = self.session.run(
            None,
            {"input": embedding},
        )

        logits = outputs[0]  # shape: [batch, num_classes]

        # Softmax (manual, since ONNX output is logits)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_class_idx = int(np.argmax(logits, axis=1)[0])
        confidence = float(probs[0, pred_class_idx])

        pred_class = ALL_CLASSES[pred_class_idx]
        return pred_class, confidence


# Lazy global classifier accessor
classifier = ActionClassifier()


def is_action_harmful(action_dict) -> Tuple[Optional[str], float]:
    """Convenience function to check if an action is harmful."""
    pred_class, confidence = classifier.predict(action_dict)
    if pred_class == "safe":
        return None, confidence
    return pred_class, confidence


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
    Decorator to guard a function that takes an action_dict as an argument.
    If the action is harmful and confidence >= conf_threshold, it raises HarmfulActionException.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            action_dict = None
            if "action_dict" in kwargs:
                action_dict = kwargs["action_dict"]
            elif args:
                action_dict = args[0]
            
            if action_dict:
                is_harmful, confidence = is_action_harmful(action_dict)
                if is_harmful and confidence >= conf_threshold:
                    raise HarmfulActionException(
                        f"Guarded action classified as harmful ({is_harmful}) with confidence {confidence:.2f}"
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
