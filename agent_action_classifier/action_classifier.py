"""
Action Classifier Module
Loads the trained neural network model to classify agent actions as harmful or safe.
"""

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from agent_action_classifier.train_nn import SimpleMLP, flatten_action_to_text

# Configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FORCE_CPU = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "emb_nn_model.pt"


class ActionClassifier:
    """Classifier for AI agent actions using embeddings and MLP."""

    def __init__(self, load_model: bool = True):
        """If load_model is False, heavy dependencies (embedding/model) are not created.
        Tests can instantiate with load_model=False and then set attributes manually.
        """
        self.embed_model = None
        self.model = None
        self.in_dim = None
        if load_model:
            # create heavy components only on demand
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
            self.load_model()

    def load_model(self):
        """Load the trained model from file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.in_dim = checkpoint["in_dim"]
        config = checkpoint["config"]
        hidden = config["hidden"]

        self.model = SimpleMLP(in_dim=self.in_dim, hidden=hidden)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, action_dict):
        """
        Predict if an action is harmful.
        Returns: (harmful_score: float, unethical_score: float)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = flatten_action_to_text(action_dict)
        if self.embed_model is None:
            raise RuntimeError("Embedding model not initialized.")
        embedding = self.embed_model.encode(
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
            harmful_score = logits[0, 0].item()
            unethical_score = logits[0, 1].item()
        return harmful_score, unethical_score


# Lazy global classifier accessor
classifier = ActionClassifier()


def is_action_harmful(action_dict):
    """Convenience function to check if an action is harmful."""
    harmful_score, unethical_score = classifier.predict(action_dict)
    confidence = max(harmful_score, unethical_score)
    if max(harmful_score, unethical_score) >= 0.5:
        if harmful_score >= unethical_score:
            return "Harmful", confidence
        else:
            return "Unethical", confidence
    return None, confidence


# Example usage
if __name__ == "__main__":
    sample_action = {
        "label": "Delete all files",
        "resource": "File System",
        "action": {
            "type": "mcp",
            "server_label": "weather_api",
            "server_url": "https://api.weather.com/v1/current",
            "parameters": {
                "location": "New York City",
                "units": "imperial",
            },
            "require_approval": "never"
        },
    }

    classification, confidence = is_action_harmful(sample_action)
    print(f"Is harmful: {classification}, Confidence: {confidence:.2f}")
