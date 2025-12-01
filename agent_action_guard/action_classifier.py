"""
Action Classifier Module
Loads the trained neural network model to classify agent actions as harmful or safe.
"""

import numpy as np
import torch

from agent_action_guard.train_nn import embed_model, flatten_action_to_text, ActionClassNet, \
    ALL_CLASSES, DEVICE, MODEL_PATH


class ActionClassifier:
    """Classifier for AI agent actions using embeddings and MLP."""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model from file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        in_dim = checkpoint["in_dim"]
        hidden = checkpoint["config"]["hidden"]

        self.model = ActionClassNet(in_dim=in_dim, hidden=hidden)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, action_dict):
        """
        Predict the class of an action.
        Returns: (class: str, confidence: float)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = flatten_action_to_text(action_dict)
        embedding = embed_model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        embedding = np.array(embedding)

        if self.model is None:
            raise RuntimeError(
                "Classifier model not loaded. Create ActionClassifier(load_model=True) "
                "or set model manually for tests."
            )

        with torch.inference_mode():
            logits = self.model(torch.tensor(embedding, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            pred_class_idx = int(torch.argmax(logits, dim=1).item())
            confidence = probs[0, pred_class_idx].item()

        pred_class = ALL_CLASSES[pred_class_idx]
        return pred_class, confidence


# Lazy global classifier accessor
classifier = ActionClassifier()


def is_action_harmful(action_dict):
    """Convenience function to check if an action is harmful."""
    pred_class, confidence = classifier.predict(action_dict)
    if pred_class == "safe":
        return None, confidence
    else:
        return pred_class, confidence


# Example usage
if __name__ == "__main__":
    sample_action = {
        "label": "ping",
        "resource": "Tool",
        "action": {
            "type": "mcp",
            "server_label": "tool",
            "server_url": "",
            "parameters": {},
            "require_approval": "never",
        },
    }
    classification, confidence = is_action_harmful(sample_action)
    print(f"{sample_action['label']} Classification: {classification}, Confidence: {confidence:.2f}")
