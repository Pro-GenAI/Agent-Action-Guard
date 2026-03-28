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
    """Lightweight stub for ONNX model path existence checks."""

    def __init__(self, exists=True):
        self._exists = exists

    def exists(self):
        """Return whether the fake model path exists."""
        return self._exists


class _FakeInferenceSession:
    """
    Mocked ONNX Runtime InferenceSession.

    Captures input embeddings and returns predefined outputs to simulate
    model inference behavior deterministically.
    """

    def __init__(self, model_path, providers=None, outputs=None):
        self.model_path = model_path
        self.providers = providers
        self._outputs = outputs if outputs is not None else [np.array([[0.0, 1.0]])]
        self.last_input = None

    def run(self, _, feed):
        """
        Simulate ONNX inference.

        Stores the input tensor for later assertions and returns mocked outputs.
        """
        self.last_input = feed.get("input")
        return self._outputs


def _runtime_utils_flatten_action_to_text(action_data):
    """Flatten action metadata to text."""
    # Example of action_data
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "data_exporter",
    #         "arguments": "{\"dataset\":\"employee_salaries\",\"destination\":\"xyz\"}"
    #     }
    # }
    # Convert to "Call function data_exporter: dataset=employee_salaries, destination=xyz"
    if action_data["type"] != "function":
        return f"Perform action: {action_data}"

    if not isinstance(action_data["function"], dict):
        return f"Call function with data: {action_data['function']}"

    func_name = action_data["function"]["name"]
    args_str = action_data["function"]["arguments"]
    try:
        args_dict = json.loads(args_str)
        args_flat = ", ".join(f"{k}={v}" for k, v in args_dict.items())
        return f"Call function {func_name}: {args_flat}"
    except json.JSONDecodeError:
        return f"Call function {func_name} with arguments: {args_str}"


def _build_fake_runtime_utils_module():
    """
    Construct a fully mocked runtime_utils module.

    Includes:
    - Mock embedding model
    - Deterministic flattening function
    - Class labels
    - ONNX model path stub
    """
    runtime_module = types.ModuleType("agent_action_guard._runtime_utils")
    runtime_module.embed_model = types.SimpleNamespace(  # type: ignore
        encode=mock.Mock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    )
    runtime_module.flatten_action_to_text = mock.Mock(  # type: ignore
        side_effect=_runtime_utils_flatten_action_to_text
    )
    runtime_module.ALL_CLASSES = ["safe", "harmful", "unethical"]  # type: ignore
    runtime_module.ONNX_MODEL_PATH = _FakeModelPath(exists=True)  # type: ignore

    class ActionGuardDecision:
        """Placeholder class for compatibility with runtime expectations."""

        pass

    runtime_module.ActionGuardDecision = ActionGuardDecision  # type: ignore
    return runtime_module


def _build_fake_onnxruntime_module(fake_session_outputs=None):
    """
    Create a mocked onnxruntime module.

    Ensures that all InferenceSession instances return controlled outputs.
    """
    onnx_module = types.ModuleType("onnxruntime")

    def _inference_session(model_path, providers=None):
        return _FakeInferenceSession(
            model_path=str(model_path),
            providers=providers,
            outputs=fake_session_outputs,
        )

    onnx_module.InferenceSession = _inference_session  # type: ignore
    return onnx_module


@pytest.fixture
def action_classifier_module(monkeypatch):
    """
    Fixture that loads the action_classifier module with mocked dependencies.

    Guarantees:
    - Clean import state
    - Injected fake runtime_utils and onnxruntime modules
    """
    for module_name in (
        "agent_action_guard.action_classifier",
        "agent_action_guard",
        "agent_action_guard._runtime_utils",
    ):
        sys.modules.pop(module_name, None)

    # Install fake runtime utils and onnxruntime before importing the classifier
    monkeypatch.setitem(
        sys.modules,
        "agent_action_guard._runtime_utils",
        _build_fake_runtime_utils_module(),
    )
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
    """
    Load the real runtime_utils module dynamically for isolated testing.
    """
    spec = importlib.util.spec_from_file_location(
        "runtime_under_test", _RUNTIME_UTILS_SOURCE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ================== Tests for ActionClassifier (ONNX) ==================


def test_global_classifier_is_initialized_and_session_created_on_import(
    action_classifier_module,
):
    """Ensure classifier initializes ONNX session on import."""
    assert isinstance(
        action_classifier_module.classifier.session, _FakeInferenceSession
    )
    assert action_classifier_module.classifier.session.providers == [
        "CPUExecutionProvider"
    ]


def test_load_model_raises_when_model_file_is_missing(action_classifier_module):
    """Verify load_model raises FileNotFoundError when model path is invalid."""
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )

    with mock.patch.object(
        action_classifier_module, "ONNX_MODEL_PATH", _FakeModelPath(exists=False)
    ):
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            classifier.load_model()


def test_load_model_creates_inference_session(action_classifier_module):
    """Ensure load_model initializes a valid ONNX inference session."""
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.session = None

    classifier.load_model()

    assert isinstance(classifier.session, _FakeInferenceSession)
    assert "CPUExecutionProvider" in classifier.session.providers


def test_predict_raises_when_session_is_not_loaded_before_embedding(
    action_classifier_module,
):
    """Ensure predict fails if session is not initialized."""
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.session = None

    with pytest.raises(RuntimeError, match="ONNX model not loaded"):
        classifier.predict({"label": "ping"})


def test_predict_raises_if_session_dropped_during_processing(action_classifier_module):
    """
    Ensure predict fails if session becomes invalid mid-execution.

    Simulates a race condition or unexpected mutation.
    """
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.session = object()

    def _drop_session(_):
        classifier.session = None
        return "flattened"

    with mock.patch.object(
        action_classifier_module, "flatten_action_to_text", side_effect=_drop_session
    ):
        # embed_model.encode should still return a valid embedding
        with mock.patch.object(
            action_classifier_module.embed_model,
            "encode",
            return_value=[[0.1, 0.2, 0.3, 0.4]],
        ):
            with pytest.raises(AttributeError):
                classifier.predict({"label": "ping"})


def test_predict_returns_predicted_class_and_confidence(action_classifier_module):
    """Validate correct class prediction and softmax confidence extraction."""
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    # Prepare logits such that softmax produces probabilities [0.2, 0.1, 0.7]
    probs = np.array([0.2, 0.1, 0.7], dtype=np.float32)
    logits = np.log(probs)[None, :]

    fake_session = _FakeInferenceSession(
        model_path="m", providers=["CPUExecutionProvider"], outputs=[logits]
    )
    classifier.session = fake_session

    with mock.patch.object(
        action_classifier_module,
        "flatten_action_to_text",
        return_value="flattened action",
    ) as flatten_mock:
        with mock.patch.object(
            action_classifier_module.embed_model,
            "encode",
            return_value=[[0.9, 0.8, 0.7, 0.6]],
        ) as encode_mock:
            pred_class, confidence = classifier.predict({"label": "Delete"})

    assert pred_class == "unethical"
    assert confidence == pytest.approx(0.7, rel=1e-6)
    flatten_mock.assert_called_once_with({"label": "Delete"})
    encode_mock.assert_called_once()
    assert isinstance(fake_session.last_input, np.ndarray)


def test_is_action_harmful_returns_none_for_safe_actions(action_classifier_module):
    """Safe predictions should return (None, confidence)."""
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("safe", 0.91)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful(
            {"label": "ping"}
        )

    assert label is None
    assert confidence == 0.91


def test_is_action_harmful_returns_label_for_non_safe_actions(action_classifier_module):
    """Non-safe predictions should return label and confidence."""
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("unethical", 0.87)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful(
            {"label": "delete"}
        )

    assert label == "unethical"
    assert confidence == 0.87


def test_is_action_harmful_propagates_classifier_errors(action_classifier_module):
    """Errors in classifier.predict should propagate transparently."""
    fake_classifier = mock.Mock()
    fake_classifier.predict.side_effect = ValueError("bad action")

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        with pytest.raises(ValueError, match="bad action"):
            action_classifier_module.is_action_harmful({"label": "delete"})


def test_ensure_action_safety_returns_true_for_safe_actions(action_classifier_module):
    """Safe actions should return True."""
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=(None, 0.99)
    ):
        is_safe = action_classifier_module.ensure_action_safety({"label": "ping"})

    assert is_safe is True


def test_ensure_action_safety_returns_false_for_harmful_action_without_exception(
    action_classifier_module,
):
    """Harmful actions should return False when exceptions are disabled."""
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("harmful", 0.62)
    ):
        is_safe = action_classifier_module.ensure_action_safety(
            {"label": "drop-database"}, raise_exception=False
        )

    assert is_safe is False


def test_ensure_action_safety_raises_for_harmful_action_when_requested(
    action_classifier_module,
):
    """Harmful actions should raise when raise_exception=True."""
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("harmful", 0.625)
    ):
        with pytest.raises(
            action_classifier_module.HarmfulActionException,
            match=r"Action classified as harmful \(harmful\) with confidence 0.62",
        ):
            action_classifier_module.ensure_action_safety(
                {"label": "drop-database"}, raise_exception=True
            )


def test_flatten_action_to_text_returns_empty_string_for_empty_action(runtime_module):
    """Empty dict is invalid because required action type is missing."""
    with pytest.raises(KeyError, match="type"):
        runtime_module.flatten_action_to_text({})


def test_flatten_action_to_text_handles_none_action_and_arguments(runtime_module):
    """
    Ensure None-valued action metadata is handled gracefully.

    Only top-level fields should be included when nested action data is absent.
    """
    with pytest.raises(TypeError):
        runtime_module.flatten_action_to_text(None)


def test_flatten_action_to_text_sorts_action_argument_keys_and_lowercases(
    runtime_module,
):
    """
    Verify:
    - Argument keys are sorted lexicographically
    - All values are normalized to lowercase
    - Core action metadata is included
    """
    flattened = runtime_module.flatten_action_to_text(
        {
            "type": "other",
            "name": "DeleteUser",
            "arguments": {"Zeta": 1, "alpha": 2, "Beta": 3},
        }
    )

    assert flattened.startswith("Perform action:")


def test_flatten_action_to_text_includes_function_name_and_sorted_dict_arguments(
    runtime_module,
):
    """
    Ensure function metadata is included and argument keys are sorted.

    Validates deterministic ordering for embedding consistency.
    """
    flattened = runtime_module.flatten_action_to_text(
        {
            "type": "function",
            "function": {
                "name": "RunCleanup",
                "arguments": {"z": 1, "a": 2, "m": 3},
            }
        }
    )

    assert flattened == "Call function RunCleanup: z=1, a=2, m=3"


def test_flatten_action_to_text_parses_function_arguments_from_json_string(
    runtime_module,
):
    """
    Ensure JSON-encoded function arguments are parsed and flattened correctly.

    Confirms:
    - JSON string is decoded
    - Keys are extracted and sorted
    """
    flattened = runtime_module.flatten_action_to_text(
        {
            "type": "function",
            "function": {
                "name": "SendEmail",
                "arguments": '{"subject": "Hi", "body": "Hello", "cc": []}',
            }
        }
    )

    assert flattened == "Call function SendEmail: subject=Hi, body=Hello, cc=[]"


def test_flatten_action_to_text_ignores_empty_values(runtime_module):
    """
    Empty or falsy fields should be excluded from the output string.

    Ensures no extraneous whitespace or placeholders are introduced.
    """
    flattened = runtime_module.flatten_action_to_text(
        {
            "type": "function",
            "function": {
                "name": "",
                "arguments": {},
            },
        }
    )

    assert flattened == "Call function : "


def test_flatten_action_to_text_raises_for_invalid_json_function_arguments(
    runtime_module,
):
    """
    Invalid JSON in function arguments should raise JSONDecodeError.

    Confirms strict parsing behavior without silent fallback.
    """
    flattened = runtime_module.flatten_action_to_text(
        {
            "type": "function",
            "function": {
                "name": "SendEmail",
                "arguments": '{"subject": "Hi"',
            },
        }
    )

    assert flattened == 'Call function SendEmail with arguments: {"subject": "Hi"'
