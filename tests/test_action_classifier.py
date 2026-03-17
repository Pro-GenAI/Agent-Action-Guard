# mypy: disable-error-code=attr-defined

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_RUNTIME_UTILS_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "agent_action_guard" / "_runtime_utils.py"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _FakeModelPath:
    def __init__(self, exists=True):
        self._exists = exists

    def exists(self):
        return self._exists


class _FakeInferenceSession:
    def __init__(self, model_path, providers=None, outputs=None):
        self.model_path = model_path
        self.providers = providers
        self._outputs = outputs if outputs is not None else [np.array([[0.0, 1.0]])]
        self.last_input = None

    def run(self, _, feed):
        # capture the input embedding for assertions
        self.last_input = feed.get("input")
        return self._outputs


def _runtime_utils_flatten_action_to_text(action_data):
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


def _build_fake_runtime_utils_module():
    runtime_module = types.ModuleType("agent_action_guard._runtime_utils")
    runtime_module.embed_model = types.SimpleNamespace(
        encode=mock.Mock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    )
    runtime_module.flatten_action_to_text = mock.Mock(
        side_effect=_runtime_utils_flatten_action_to_text
    )
    runtime_module.ALL_CLASSES = ["safe", "harmful", "unethical"]
    runtime_module.ONNX_MODEL_PATH = _FakeModelPath(exists=True)
    class ActionGuardDecision:
        pass
    runtime_module.ActionGuardDecision = ActionGuardDecision
    return runtime_module


def _build_fake_onnxruntime_module(fake_session_outputs=None):
    onnx_module = types.ModuleType("onnxruntime")

    def _inference_session(model_path, providers=None):
        return _FakeInferenceSession(model_path=str(model_path), providers=providers, outputs=fake_session_outputs)

    onnx_module.InferenceSession = _inference_session
    return onnx_module


@pytest.fixture
def action_classifier_module(monkeypatch):
    # Ensure a clean import
    for module_name in (
        "agent_action_guard.action_classifier",
        "agent_action_guard",
        "agent_action_guard._runtime_utils",
    ):
        sys.modules.pop(module_name, None)

    # Install fake runtime utils and onnxruntime before importing the classifier
    monkeypatch.setitem(sys.modules, "agent_action_guard._runtime_utils", _build_fake_runtime_utils_module())
    monkeypatch.setitem(sys.modules, "onnxruntime", _build_fake_onnxruntime_module())

    module = importlib.import_module("agent_action_guard.action_classifier")
    yield module

    for module_name in (
        "agent_action_guard.action_classifier",
        "agent_action_guard",
        "agent_action_guard._runtime_utils",
    ):
        sys.modules.pop(module_name, None)


@pytest.fixture
def runtime_module():
    spec = importlib.util.spec_from_file_location("runtime_under_test", _RUNTIME_UTILS_SOURCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ================== Tests for ActionClassifier (ONNX) ==================


def test_global_classifier_is_initialized_and_session_created_on_import(action_classifier_module):
    assert isinstance(action_classifier_module.classifier.session, _FakeInferenceSession)
    assert action_classifier_module.classifier.session.providers == ["CPUExecutionProvider"]


def test_load_model_raises_when_model_file_is_missing(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(action_classifier_module.ActionClassifier)

    with mock.patch.object(action_classifier_module, "ONNX_MODEL_PATH", _FakeModelPath(exists=False)):
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            classifier.load_model()


def test_load_model_creates_inference_session(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(action_classifier_module.ActionClassifier)
    classifier.session = None

    classifier.load_model()

    assert isinstance(classifier.session, _FakeInferenceSession)
    assert "CPUExecutionProvider" in classifier.session.providers


def test_predict_raises_when_session_is_not_loaded_before_embedding(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(action_classifier_module.ActionClassifier)
    classifier.session = None

    with pytest.raises(RuntimeError, match="ONNX model not loaded"):
        classifier.predict({"label": "ping"})


def test_predict_raises_if_session_dropped_during_processing(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(action_classifier_module.ActionClassifier)
    classifier.session = object()

    def _drop_session(_):
        classifier.session = None
        return "flattened"

    with mock.patch.object(action_classifier_module, "flatten_action_to_text", side_effect=_drop_session):
        # embed_model.encode should still return a valid embedding
        with mock.patch.object(action_classifier_module.embed_model, "encode", return_value=[[0.1, 0.2, 0.3, 0.4]]):
            with pytest.raises(AttributeError):
                classifier.predict({"label": "ping"})


def test_predict_returns_predicted_class_and_confidence(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(action_classifier_module.ActionClassifier)
    # prepare logits such that softmax produces probabilities [0.2, 0.1, 0.7]
    probs = np.array([0.2, 0.1, 0.7], dtype=np.float32)
    logits = np.log(probs)[None, :]

    fake_session = _FakeInferenceSession(model_path="m", providers=["CPUExecutionProvider"], outputs=[logits])
    classifier.session = fake_session

    with mock.patch.object(action_classifier_module, "flatten_action_to_text", return_value="flattened action") as flatten_mock:
        with mock.patch.object(action_classifier_module.embed_model, "encode", return_value=[[0.9, 0.8, 0.7, 0.6]]) as encode_mock:
            pred_class, confidence = classifier.predict({"label": "Delete"})

    assert pred_class == "unethical"
    assert confidence == pytest.approx(0.7, rel=1e-6)
    flatten_mock.assert_called_once_with({"label": "Delete"})
    encode_mock.assert_called_once_with(["flattened action"], normalize_embeddings=True, show_progress_bar=False)
    # last_input is the numpy array passed to run
    assert isinstance(fake_session.last_input, np.ndarray)
    assert fake_session.last_input.shape[1] == 4


def test_is_action_harmful_returns_none_for_safe_actions(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("safe", 0.91)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful({"label": "ping"})

    assert label is None
    assert confidence == 0.91


def test_is_action_harmful_returns_label_for_non_safe_actions(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("unethical", 0.87)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful({"label": "delete"})

    assert label == "unethical"
    assert confidence == 0.87


def test_is_action_harmful_propagates_classifier_errors(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.side_effect = ValueError("bad action")

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        with pytest.raises(ValueError, match="bad action"):
            action_classifier_module.is_action_harmful({"label": "delete"})


def test_ensure_action_safety_returns_true_for_safe_actions(action_classifier_module):
    with mock.patch.object(action_classifier_module, "is_action_harmful", return_value=(None, 0.99)):
        is_safe = action_classifier_module.ensure_action_safety({"label": "ping"})

    assert is_safe is True


def test_ensure_action_safety_returns_false_for_harmful_action_without_exception(action_classifier_module):
    with mock.patch.object(action_classifier_module, "is_action_harmful", return_value=("harmful", 0.62)):
        is_safe = action_classifier_module.ensure_action_safety({"label": "drop-database"}, raise_exception=False)

    assert is_safe is False


def test_ensure_action_safety_raises_for_harmful_action_when_requested(action_classifier_module):
    with mock.patch.object(action_classifier_module, "is_action_harmful", return_value=("harmful", 0.625)):
        with pytest.raises(action_classifier_module.HarmfulActionException, match=r"Action classified as harmful \(harmful\) with confidence 0.62"):
            action_classifier_module.ensure_action_safety({"label": "drop-database"}, raise_exception=True)


def test_flatten_action_to_text_returns_empty_string_for_empty_action(runtime_module):
    assert runtime_module.flatten_action_to_text({}) == ""


def test_flatten_action_to_text_handles_none_action_and_parameters(runtime_module):
    flattened = runtime_module.flatten_action_to_text(
        {
            "label": "PING",
            "resource": "Tool",
            "action": None,
        }
    )

    assert flattened == "ping tool"


def test_flatten_action_to_text_sorts_action_parameter_keys_and_lowercases(
    runtime_module,
):
    flattened = runtime_module.flatten_action_to_text(
        {
            "label": "DeleteUser",
            "resource": "MCP",
            "action": {
                "server_label": "AdminServer",
                "server_url": "HTTPS://EXAMPLE.COM",
                "require_approval": "NEVER",
                "parameters": {"Zeta": 1, "alpha": 2, "Beta": 3},
            },
        }
    )

    assert (
        flattened
        == "deleteuser mcp adminserver https://example.com never alpha beta zeta"
    )


def test_flatten_action_to_text_includes_function_name_and_sorted_dict_arguments(
    runtime_module,
):
    flattened = runtime_module.flatten_action_to_text(
        {
            "function": {
                "name": "RunCleanup",
                "arguments": {"z": 1, "a": 2, "m": 3},
            }
        }
    )

    assert flattened == "runcleanup a m z"


def test_flatten_action_to_text_parses_function_arguments_from_json_string(
    runtime_module,
):
    flattened = runtime_module.flatten_action_to_text(
        {
            "function": {
                "name": "SendEmail",
                "arguments": '{"subject": "Hi", "body": "Hello", "cc": []}',
            }
        }
    )

    assert flattened == "sendemail body cc subject"


def test_flatten_action_to_text_ignores_empty_values(runtime_module):
    flattened = runtime_module.flatten_action_to_text(
        {
            "label": "",
            "resource": "",
            "action": {
                "server_label": "",
                "server_url": "",
                "require_approval": "",
                "parameters": {},
            },
            "function": {
                "name": "",
                "arguments": {},
            },
        }
    )

    assert flattened == ""


def test_flatten_action_to_text_raises_for_invalid_json_function_arguments(
    runtime_module,
):
    with pytest.raises(json.JSONDecodeError):
        runtime_module.flatten_action_to_text(
            {
                "function": {
                    "name": "SendEmail",
                    "arguments": '{"subject": "Hi"',
                }
            }
        )
