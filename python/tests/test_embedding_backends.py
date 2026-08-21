import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_RUNTIME_UTILS_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "agent_action_guard" / "_runtime_utils.py"
)


def _load_runtime_utils_module():
    spec = importlib.util.spec_from_file_location(
        "embedding_runtime_under_test", _RUNTIME_UTILS_SOURCE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_embedding_env(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AAG_EMBED_ONNX", raising=False)


def test_embedding_model_defaults_to_onnx_without_embedding_env(monkeypatch):
    _clear_embedding_env(monkeypatch)

    runtime_module = _load_runtime_utils_module()
    model = runtime_module.EmbeddingModel()

    assert model.backend == "onnx"
    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedding_model_uses_api_when_api_configuration_exists(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")

    runtime_module = _load_runtime_utils_module()

    assert runtime_module.EmbeddingModel().backend == "api"


def test_embedding_model_uses_api_when_api_key_exists(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "configured")

    runtime_module = _load_runtime_utils_module()

    assert runtime_module.EmbeddingModel().backend == "api"


def test_backend_precedence_model_then_api_then_default_onnx(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBED_MODEL_NAME", "preferred-api-model")
    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    runtime_module = _load_runtime_utils_module()

    model_configured = runtime_module.EmbeddingModel()
    assert model_configured.backend == "api"
    assert model_configured.model_name == "preferred-api-model"

    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_API_KEY", "configured")
    api_configured = runtime_module.EmbeddingModel()
    assert api_configured.backend == "api"
    assert api_configured.model_name == runtime_module.DEFAULT_EMBED_MODEL_NAME

    _clear_embedding_env(monkeypatch)
    default_model = runtime_module.EmbeddingModel()
    assert default_model.backend == "onnx"
    assert default_model.model_name == runtime_module.DEFAULT_EMBED_MODEL_NAME


def test_onnx_env_accepts_model_filepath_or_directory(monkeypatch, tmp_path):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    custom_model = tmp_path / "custom.onnx"

    monkeypatch.setenv("AAG_EMBED_ONNX", str(custom_model))
    model_file, vocab_file = runtime_module._resolve_onnx_model_files()
    assert model_file == custom_model
    assert vocab_file == tmp_path / "tokenizer.json"

    monkeypatch.setenv("AAG_EMBED_ONNX", str(tmp_path))
    model_file, vocab_file = runtime_module._resolve_onnx_model_files()
    assert model_file == tmp_path / "model.onnx"
    assert vocab_file == tmp_path / "tokenizer.json"


def test_default_onnx_assets_use_trained_minilm_model(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    downloads = []
    monkeypatch.setattr(
        runtime_module,
        "_download_file",
        lambda url, destination: downloads.append((url, destination)),
    )

    model_file, vocab_file = runtime_module._resolve_onnx_model_files()

    assert model_file.name == "model.onnx"
    assert vocab_file.name == "tokenizer.json"
    assert downloads == [
        (runtime_module._default_onnx_asset_url("model.onnx"), model_file),
        (runtime_module._default_onnx_asset_url("tokenizer.json"), vocab_file),
    ]


def test_onnx_config_takes_precedence_and_generates_embeddings(monkeypatch, tmp_path):
    _clear_embedding_env(monkeypatch)
    model_file = tmp_path / "custom.onnx"
    vocab_file = tmp_path / "tokenizer.json"
    model_file.touch()
    vocab_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AAG_EMBED_ONNX", str(model_file))
    monkeypatch.setenv("EMBED_MODEL_NAME", "api-embedding-model")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")

    runtime_module = _load_runtime_utils_module()

    class FakeEncoding:
        def __init__(self, ids, attention_mask, type_ids):
            self.ids = ids
            self.attention_mask = attention_mask
            self.type_ids = type_ids

    class FakeTokenizer:
        loaded_file = None

        @classmethod
        def from_file(cls, filename):
            cls.loaded_file = filename
            return cls()

        def enable_truncation(self, max_length):
            assert max_length == 256

        def token_to_id(self, value):
            assert value == "[PAD]"
            return 0

        def enable_padding(self, pad_id, pad_token):
            assert pad_id == 0
            assert pad_token == "[PAD]"

        def encode_batch(self, texts):
            assert texts == ["hello", "world"]
            return [
                FakeEncoding([1, 2, 0], [1, 1, 0], [0, 0, 0]),
                FakeEncoding([3, 4, 0], [1, 1, 0], [0, 0, 0]),
            ]

    class FakeEmbeddingSession:
        def __init__(self, filename, providers=None):
            assert filename == str(model_file)
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                types.SimpleNamespace(name="input_ids"),
                types.SimpleNamespace(name="attention_mask"),
                types.SimpleNamespace(name="token_type_ids"),
            ]

        def run(self, _, feed):
            assert set(feed) == {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            }
            return [
                np.asarray(
                    [
                        [[2.0, 0.0], [2.0, 0.0], [99.0, 99.0]],
                        [[0.0, 3.0], [0.0, 3.0], [99.0, 99.0]],
                    ],
                    dtype=np.float32,
                )
            ]

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.InferenceSession = FakeEmbeddingSession
    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    model = runtime_module.EmbeddingModel()
    embeddings = model.encode(["hello", "world"])

    assert model.backend == "onnx"
    assert FakeTokenizer.loaded_file == str(vocab_file)
    np.testing.assert_allclose(
        embeddings,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("EMBED_MODEL_NAME", "configured-model"),
        ("EMBEDDING_BASE_URL", "http://localhost:1234/v1"),
        ("EMBEDDING_API_KEY", "configured"),
        ("OPENAI_API_KEY", "configured"),
    ],
)
def test_each_api_environment_variable_selects_api_backend(
    monkeypatch, env_name, env_value
):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv(env_name, env_value)

    runtime_module = _load_runtime_utils_module()
    model = runtime_module.EmbeddingModel()

    assert model.backend == "api"


@pytest.mark.parametrize(
    "api_env_name",
    [
        "EMBED_MODEL_NAME",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    ],
)
def test_onnx_environment_always_overrides_api_environment(monkeypatch, api_env_name):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("AAG_EMBED_ONNX", "/tmp/custom.onnx")
    monkeypatch.setenv(api_env_name, "configured")

    runtime_module = _load_runtime_utils_module()

    assert runtime_module.EmbeddingModel().backend == "onnx"


def test_explicit_model_argument_preserves_api_compatibility(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()

    model = runtime_module.EmbeddingModel("explicit-model")

    assert model.backend == "api"
    assert model.model_name == "explicit-model"


@pytest.mark.parametrize("api_key_env", ["EMBEDDING_API_KEY", "OPENAI_API_KEY"])
def test_legacy_model_and_api_key_environment_call_embedding_api(
    monkeypatch, api_key_env
):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBED_MODEL_NAME", "legacy-api-model")
    monkeypatch.setenv(api_key_env, "configured")
    runtime_module = _load_runtime_utils_module()
    client_configs = []
    requests = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            requests.append(kwargs)
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.25, 0.75])]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            client_configs.append(kwargs)
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(runtime_module.openai, "OpenAI", FakeClient)

    model = runtime_module.EmbeddingModel()
    embeddings = model.encode(["hello"])

    assert model.backend == "api"
    assert model.model_name == "legacy-api-model"
    assert client_configs == [{"api_key": "configured"}]
    assert requests == [{"model": "legacy-api-model", "input": ["hello"]}]
    np.testing.assert_allclose(embeddings, [[0.25, 0.75]])


def test_api_encode_preserves_payload_result_shape_and_client_caching(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBED_MODEL_NAME", "api-model")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "primary")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")
    runtime_module = _load_runtime_utils_module()
    created_clients = []
    requests = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            requests.append(kwargs)
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                    types.SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(runtime_module.openai, "OpenAI", FakeClient)
    model = runtime_module.EmbeddingModel()

    first = model.encode(["hello", "world"])
    second_client = model._get_client()

    assert created_clients == [
        {"base_url": "http://localhost:1234/v1", "api_key": "primary"}
    ]
    assert second_client is model.client
    assert requests == [{"model": "api-model", "input": ["hello", "world"]}]
    assert isinstance(first, np.ndarray)
    assert first.shape == (2, 3)
    np.testing.assert_allclose(first[1], [0.4, 0.5, 0.6])


def test_download_file_reuses_existing_cache_without_network(monkeypatch, tmp_path):
    runtime_module = _load_runtime_utils_module()
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"cached")

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("network should not be used for an existing cache file")

    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fail_urlopen)

    runtime_module._download_file("https://example.invalid/model.onnx", destination)

    assert destination.read_bytes() == b"cached"


def test_download_file_writes_atomically_and_cleans_temporary_file(
    monkeypatch, tmp_path
):
    import io

    runtime_module = _load_runtime_utils_module()
    destination = tmp_path / "cache" / "model.onnx"
    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"model-bytes"),
    )

    runtime_module._download_file("https://example.invalid/model.onnx", destination)

    assert destination.read_bytes() == b"model-bytes"
    assert list(destination.parent.iterdir()) == [destination]


def test_missing_local_onnx_model_has_actionable_error(monkeypatch, tmp_path):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("AAG_EMBED_ONNX", str(tmp_path / "missing.onnx"))
    runtime_module = _load_runtime_utils_module()
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = object
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    with pytest.raises(FileNotFoundError, match="ONNX embedding model not found"):
        runtime_module.EmbeddingModel()._get_onnx_runtime()


def test_missing_local_tokenizer_has_actionable_error(monkeypatch, tmp_path):
    _clear_embedding_env(monkeypatch)
    model_file = tmp_path / "model.onnx"
    model_file.touch()
    monkeypatch.setenv("AAG_EMBED_ONNX", str(model_file))
    runtime_module = _load_runtime_utils_module()
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = object
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    with pytest.raises(FileNotFoundError, match="Place tokenizer.json beside"):
        runtime_module.EmbeddingModel()._get_onnx_runtime()


def test_tokenizer_without_pad_token_is_rejected_before_session_creation(
    monkeypatch, tmp_path
):
    _clear_embedding_env(monkeypatch)
    model_file = tmp_path / "model.onnx"
    tokenizer_file = tmp_path / "tokenizer.json"
    model_file.touch()
    tokenizer_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AAG_EMBED_ONNX", str(model_file))
    runtime_module = _load_runtime_utils_module()

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _filename):
            return cls()

        def enable_truncation(self, max_length):
            assert max_length == 256

        def token_to_id(self, token):
            assert token == "[PAD]"

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.InferenceSession = lambda *_args, **_kwargs: pytest.fail(
        "session must not be created for an invalid tokenizer"
    )
    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    with pytest.raises(ValueError, match=r"does not define a \[PAD\] token"):
        runtime_module.EmbeddingModel()._get_onnx_runtime()


def _make_direct_onnx_model(runtime_module, output, input_names=None):
    class FakeEncoding:
        def __init__(self):
            self.ids = [1, 2]
            self.attention_mask = [1, 1]
            self.type_ids = [0, 0]

    class FakeTokenizer:
        def encode_batch(self, texts):
            self.texts = texts
            return [FakeEncoding() for _ in texts]

    class FakeSession:
        def get_inputs(self):
            names = input_names or ["input_ids", "attention_mask", "token_type_ids"]
            return [types.SimpleNamespace(name=name) for name in names]

        def run(self, _output_names, _feed):
            return output

    model = runtime_module.EmbeddingModel()
    model.backend = "onnx"
    model.onnx_session = FakeSession()
    model.onnx_tokenizer = FakeTokenizer()
    return model


def test_onnx_2d_sentence_embeddings_are_normalized_and_zero_safe(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    model = _make_direct_onnx_model(
        runtime_module,
        [np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)],
    )

    result = model.encode(["vector", "zero"])

    np.testing.assert_allclose(result[0], [0.6, 0.8])
    np.testing.assert_allclose(result[1], [0.0, 0.0])
    assert model.onnx_tokenizer.texts == ["vector", "zero"]


def test_onnx_encode_stringifies_non_string_inputs(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    model = _make_direct_onnx_model(
        runtime_module,
        [np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)],
    )

    model.encode([123, None])

    assert model.onnx_tokenizer.texts == ["123", "None"]


def test_onnx_rejects_unknown_model_input(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    model = _make_direct_onnx_model(
        runtime_module,
        [np.asarray([[1.0, 0.0]], dtype=np.float32)],
        input_names=["input_ids", "unsupported_input"],
    )

    with pytest.raises(ValueError, match="Unsupported ONNX embedding model input"):
        model.encode(["hello"])


def test_onnx_rejects_empty_output_list(monkeypatch):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    model = _make_direct_onnx_model(runtime_module, [])

    with pytest.raises(ValueError, match="returned no outputs"):
        model.encode(["hello"])


@pytest.mark.parametrize(
    "output",
    [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
    ],
)
def test_onnx_rejects_unsupported_output_rank(monkeypatch, output):
    _clear_embedding_env(monkeypatch)
    runtime_module = _load_runtime_utils_module()
    model = _make_direct_onnx_model(runtime_module, [output])

    with pytest.raises(ValueError, match="must return a 2D sentence embedding or 3D"):
        model.encode(["hello"])


def test_download_file_supports_concurrent_first_use(monkeypatch, tmp_path):
    import io
    from concurrent.futures import ThreadPoolExecutor

    runtime_module = _load_runtime_utils_module()
    destination = tmp_path / "cache" / "model.onnx"
    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"shared-model"),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: runtime_module._download_file(
                    "https://example.invalid/model.onnx", destination
                ),
                range(2),
            )
        )

    assert results == [None, None]
    assert destination.read_bytes() == b"shared-model"
    assert list(destination.parent.iterdir()) == [destination]


def test_download_failure_leaves_no_partial_cache_file(monkeypatch, tmp_path):
    import io

    runtime_module = _load_runtime_utils_module()
    destination = tmp_path / "cache" / "model.onnx"
    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"partial"),
    )

    def fail_copy(*_args, **_kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(runtime_module.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        runtime_module._download_file("https://example.invalid/model.onnx", destination)

    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
