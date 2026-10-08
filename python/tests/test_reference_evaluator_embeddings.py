"""Regression tests for ONNX Runtime and ONNX ReferenceEvaluator embeddings."""

from types import SimpleNamespace

import numpy as np
import pytest

from agent_action_guard._runtime_utils import EmbeddingModel


class _Tokenizer:
    def encode_batch(self, texts):
        return [SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0]) for _ in texts]


class _Session:
    input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.feed = None

    def run(self, output_names, feed):
        self.feed = feed
        return [np.ones((len(feed["input_ids"]), 2, 3), dtype=np.float32)]


class _OrtSession(_Session):
    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.input_names]


@pytest.mark.parametrize("session_type", [_Session, _OrtSession])
def test_embedding_supports_both_session_input_metadata_interfaces(session_type):
    session = session_type()
    model = EmbeddingModel()
    model._get_onnx_runtime = lambda: (session, _Tokenizer())
    output = model._encode_onnx(["one", "two"])
    assert output.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), [1, 1])
    assert set(session.feed) == {"input_ids", "attention_mask"}


def test_embedding_reference_evaluator_rejects_unknown_model_inputs():
    session = _Session()
    session.input_names = ["unknown_input"]
    model = EmbeddingModel()
    model._get_onnx_runtime = lambda: (session, _Tokenizer())
    with pytest.raises(ValueError, match="Unsupported ONNX embedding model input"):
        model._encode_onnx(["one"])
