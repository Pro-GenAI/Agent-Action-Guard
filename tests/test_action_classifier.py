from pathlib import Path

# Create a lightweight test for action_classifier.py by mocking heavy dependencies

ROOT = Path(__file__).parent.parent
AC_PATH = ROOT / "action_classifier" / "action_classifier.py"

# We'll import the module as a module object after temporarily replacing
# SentenceTransformer and torch


def test_flatten_action_to_text_basic():
    from agent_action_classifier import ActionClassifier

    ac = ActionClassifier()
    sample = {
        "label": "test_label",
        "resource": "test_resource",
        "action": {
            "server_label": "srv",
            "server_url": "https://example.com/api/endpoint",
            "require_approval": "always",
            "parameters": {"x": 1, "y": 2},
        },
    }
    txt = ac.flatten_action_to_text(sample)
    assert "test_label" in txt
    assert "example.com" in txt
    assert "x" in txt and "y" in txt


def test_predict_mock_embedding(monkeypatch):
    # Patch the embed model to return a fixed embedding and the MLP to a stub
    import numpy as np

    class DummyEmbed:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.zeros((len(texts), 384))

    class DummyModel:
        def __init__(self):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, x):
            import torch

            # return zeros logits for two classes
            return torch.zeros((x.shape[0], 2))

    from agent_action_classifier import get_classifier

    clf = get_classifier()
    monkeypatch.setattr(clf, "embed_model", DummyEmbed())
    monkeypatch.setattr(clf, "model", DummyModel())

    is_harmful, conf = clf.predict({"label": "abc", "resource": "r"})
    assert isinstance(is_harmful, bool)
    assert isinstance(conf, float)
