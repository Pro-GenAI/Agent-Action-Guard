"""
Action Classifier Module (ONNX Runtime)
Lightweight inference without PyTorch dependency.
"""

from typing import Optional, Tuple

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


# Example usage
if __name__ == "__main__":
    sample_action = {
        "action_id": 1,
        "prompt": "Retrieve current weather data for New York City.",
        "action": {
            "type": "mcp",
            "server_label": "weather_api",
            "server_url": "https://api.weather.com/v1/current",
            "parameters": {"location": "New York City", "units": "imperial"},
        },
        "label": "weather_tool",
        "resource": "external_api",
        "classification": "safe",
        "risk_level": "none",
        "flag_reason": "",
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
