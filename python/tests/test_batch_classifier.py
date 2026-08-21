import numpy as np

from agent_action_guard import action_classifier


def test_action_classifier_accepts_embedding_model_injection(monkeypatch):
    custom_embedding_model = object()
    monkeypatch.setattr(
        action_classifier.ActionClassifier, "load_model", lambda self: None
    )

    classifier = action_classifier.ActionClassifier(
        embedding_model=custom_embedding_model
    )

    assert classifier.embedding_model is custom_embedding_model


class FakeSession:
    def __init__(self):
        self.last_input = None

    def run(self, _outputs, feed):
        self.last_input = feed["input"]
        return [np.asarray([[3.0, 0.1, 0.2], [0.1, 2.5, 0.2]], dtype=np.float32)]


def test_predict_batch_vectorizes_embedding_and_inference(monkeypatch):
    classifier = action_classifier.ActionClassifier.__new__(
        action_classifier.ActionClassifier
    )
    classifier.session = FakeSession()
    calls = []

    def fake_encode(texts, **_kwargs):
        calls.append(texts)
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(action_classifier.embed_model, "encode", fake_encode)
    monkeypatch.setattr(
        action_classifier,
        "flatten_action_to_text",
        lambda action: f"action-{action['id']}",
    )

    results = classifier.predict_batch([{"id": 1}, {"id": 2}])

    assert [label for label, _confidence in results] == ["safe", "harmful"]
    assert calls == [["action-1", "action-2"]]
    assert classifier.session.last_input.shape == (2, 4)


def test_is_actions_harmful_maps_safe_labels(monkeypatch):
    class FakeClassifier:
        def predict_batch(self, actions, batch_size=None):
            assert len(actions) == 2
            assert batch_size == 8
            return [("safe", 0.9), ("unethical", 0.8)]

    monkeypatch.setattr(action_classifier, "classifier", FakeClassifier())

    assert action_classifier.is_actions_harmful([{"id": 1}, {"id": 2}], 8) == [
        (None, 0.9),
        ("unethical", 0.8),
    ]
