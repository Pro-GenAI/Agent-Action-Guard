# mypy: disable-error-code=attr-defined

import contextlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ACTION_CLASSIFIER_MODULE_NAMES = (
    "numpy",
    "torch",
    "agent_action_guard",
    "agent_action_guard.action_classifier",
    "agent_action_guard._runtime",
)
RUNTIME_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "agent_action_guard" / "_runtime.py"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeModelPath:
    def __init__(self, exists=True):
        self._exists = exists

    def exists(self):
        return self._exists


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeIndex:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeTensor:
    def __init__(self, data):
        self.data = data
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeProbabilityTable:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        row_index, col_index = index
        return _FakeScalar(self.rows[row_index][col_index])


class _FakeActionClassNet:
    def __init__(self, in_dim, hidden):
        self.in_dim = in_dim
        self.hidden = hidden
        self.state_dict = None
        self.eval_called = False
        self.last_input = None
        self.output_logits = [[0.1, 0.7, 0.2]]

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def eval(self):
        self.eval_called = True

    def __call__(self, tensor):
        self.last_input = tensor
        return _FakeTensor(self.output_logits)


def _build_fake_numpy_module():
    numpy_module = types.ModuleType("numpy")
    numpy_module.array = lambda value: value
    return numpy_module


def _build_fake_torch_module():
    torch_module = types.ModuleType("torch")

    def _fake_load(*args, **kwargs):
        return {
            "in_dim": 4,
            "config": {"hidden": 8},
            "model_state_dict": {"layer": "weights"},
        }

    @contextlib.contextmanager
    def _inference_mode():
        yield

    def _softmax(logits, dim=1):
        return _FakeProbabilityTable(logits.data)

    def _argmax(logits, dim=1):
        row = logits.data[0]
        best_index = max(range(len(row)), key=row.__getitem__)
        return _FakeIndex(best_index)

    torch_module.load = _fake_load
    torch_module.inference_mode = _inference_mode
    torch_module.tensor = lambda data, dtype=None: _FakeTensor(data)
    torch_module.softmax = _softmax
    torch_module.argmax = _argmax
    torch_module.float32 = "float32"
    return torch_module


def _runtime_flatten_action_to_text(action_data):
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


def _build_fake_runtime_module():
    runtime_module = types.ModuleType("agent_action_guard._runtime")
    runtime_module.embed_model = types.SimpleNamespace(
        encode=mock.Mock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    )
    runtime_module.flatten_action_to_text = mock.Mock(
        side_effect=_runtime_flatten_action_to_text
    )
    runtime_module.ActionClassNet = _FakeActionClassNet
    runtime_module.ALL_CLASSES = ["safe", "harmful", "unethical"]
    runtime_module.DEVICE = "cpu"
    runtime_module.MODEL_PATH = _FakeModelPath(exists=True)
    return runtime_module


def _build_runtime_import_torch_module():
    torch_module = types.ModuleType("torch")

    class _FakeNNModule:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.device = lambda name: name
    torch_module.nn = types.SimpleNamespace(
        Module=_FakeNNModule,
        Sequential=lambda *args, **kwargs: _FakeNNModule(),
        Linear=lambda *args, **kwargs: object(),
        LayerNorm=lambda *args, **kwargs: object(),
        GELU=lambda *args, **kwargs: object(),
        Dropout=lambda *args, **kwargs: object(),
    )
    return torch_module


def _build_runtime_import_openai_module():
    openai_module = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, base_url=None):
            self.base_url = base_url
            self.embeddings = types.SimpleNamespace(create=mock.Mock())

    openai_module.OpenAI = _FakeOpenAI
    return openai_module


@pytest.fixture
def action_classifier_module(monkeypatch):
    original_modules = {
        name: sys.modules.get(name) for name in ACTION_CLASSIFIER_MODULE_NAMES
    }

    for module_name in (
        "agent_action_guard",
        "agent_action_guard.action_classifier",
        "agent_action_guard._runtime",
    ):
        sys.modules.pop(module_name, None)

    monkeypatch.setitem(sys.modules, "numpy", _build_fake_numpy_module())
    monkeypatch.setitem(sys.modules, "torch", _build_fake_torch_module())
    monkeypatch.setitem(
        sys.modules, "agent_action_guard._runtime", _build_fake_runtime_module()
    )

    module = importlib.import_module("agent_action_guard.action_classifier")
    yield module

    for module_name in (
        "agent_action_guard.action_classifier",
        "agent_action_guard",
        "agent_action_guard._runtime",
    ):
        sys.modules.pop(module_name, None)

    for module_name, original_module in original_modules.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


@pytest.fixture
def runtime_module(monkeypatch):
    original_modules = {
        name: sys.modules.get(name) for name in ("numpy", "openai", "torch")
    }
    monkeypatch.setitem(sys.modules, "numpy", _build_fake_numpy_module())
    monkeypatch.setitem(sys.modules, "openai", _build_runtime_import_openai_module())
    monkeypatch.setitem(sys.modules, "torch", _build_runtime_import_torch_module())

    spec = importlib.util.spec_from_file_location(
        "runtime_under_test", RUNTIME_SOURCE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module

    for module_name, original_module in original_modules.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


def test_global_classifier_is_initialized_and_evaluated_on_import(
    action_classifier_module,
):
    assert isinstance(action_classifier_module.classifier.model, _FakeActionClassNet)
    assert action_classifier_module.classifier.model.eval_called is True
    assert action_classifier_module.classifier.model.in_dim == 4
    assert action_classifier_module.classifier.model.hidden == 8


def test_load_model_raises_when_model_file_is_missing(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )

    with mock.patch.object(
        action_classifier_module, "MODEL_PATH", _FakeModelPath(exists=False)
    ):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            classifier.load_model()


def test_load_model_uses_checkpoint_dimensions_and_state(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.model = None

    classifier.load_model()

    assert classifier.model.in_dim == 4
    assert classifier.model.hidden == 8
    assert classifier.model.state_dict == {"layer": "weights"}
    assert classifier.model.eval_called is True


def test_predict_raises_when_model_is_not_loaded_before_embedding(
    action_classifier_module,
):
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.model = None

    with pytest.raises(RuntimeError, match="Model not loaded"):
        classifier.predict({"label": "ping"})


def test_predict_raises_when_model_becomes_none_after_embedding(
    action_classifier_module,
):
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.model = object()

    def _drop_model(_):
        classifier.model = None
        return "flattened"

    with mock.patch.object(
        action_classifier_module, "flatten_action_to_text", side_effect=_drop_model
    ):
        with pytest.raises(RuntimeError, match="Classifier model not loaded"):
            classifier.predict({"label": "ping"})


def test_predict_returns_predicted_class_and_confidence(action_classifier_module):
    classifier = action_classifier_module.ActionClassifier.__new__(
        action_classifier_module.ActionClassifier
    )
    classifier.model = _FakeActionClassNet(in_dim=4, hidden=8)
    classifier.model.output_logits = [[0.2, 0.1, 0.7]]

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
    assert confidence == 0.7
    flatten_mock.assert_called_once_with({"label": "Delete"})
    encode_mock.assert_called_once_with(
        ["flattened action"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert classifier.model.last_input.data == [[0.9, 0.8, 0.7, 0.6]]
    assert classifier.model.last_input.device == "cpu"


def test_is_action_harmful_returns_none_for_safe_actions(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("safe", 0.91)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful(
            {"label": "ping"}
        )

    assert label is None
    assert confidence == 0.91


def test_is_action_harmful_returns_label_for_non_safe_actions(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.return_value = ("unethical", 0.87)

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        label, confidence = action_classifier_module.is_action_harmful(
            {"label": "delete"}
        )

    assert label == "unethical"
    assert confidence == 0.87


def test_is_action_harmful_propagates_classifier_errors(action_classifier_module):
    fake_classifier = mock.Mock()
    fake_classifier.predict.side_effect = ValueError("bad action")

    with mock.patch.object(action_classifier_module, "classifier", fake_classifier):
        with pytest.raises(ValueError, match="bad action"):
            action_classifier_module.is_action_harmful({"label": "delete"})


def test_verify_action_safety_returns_true_for_safe_actions(action_classifier_module):
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=(None, 0.99)
    ):
        is_safe = action_classifier_module.verify_action_safety({"label": "ping"})

    assert is_safe is True


def test_verify_action_safety_returns_false_for_harmful_action_without_exception(
    action_classifier_module,
):
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("harmful", 0.62)
    ):
        is_safe = action_classifier_module.verify_action_safety(
            {"label": "drop-database"}, raise_exception=False
        )

    assert is_safe is False


def test_verify_action_safety_returns_false_for_unethical_action_without_exception(
    action_classifier_module,
):
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("unethical", 0.55)
    ):
        is_safe = action_classifier_module.verify_action_safety(
            {"label": "impersonate-user"}, raise_exception=False
        )

    assert is_safe is False


def test_verify_action_safety_raises_for_harmful_action_when_requested(
    action_classifier_module,
):
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("harmful", 0.625)
    ):
        with pytest.raises(
            action_classifier_module.HarmfulActionException,
            match=r"Action classified as harmful \(harmful\) with confidence 0.62",
        ):
            action_classifier_module.verify_action_safety(
                {"label": "drop-database"}, raise_exception=True
            )


def test_verify_action_safety_raises_for_unethical_action_when_requested(
    action_classifier_module,
):
    with mock.patch.object(
        action_classifier_module, "is_action_harmful", return_value=("unethical", 0.991)
    ):
        with pytest.raises(
            action_classifier_module.HarmfulActionException,
            match=r"Action classified as harmful \(unethical\) with confidence 0.99",
        ):
            action_classifier_module.verify_action_safety(
                {"label": "impersonate-user"}, raise_exception=True
            )


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
