import importlib.util
from pathlib import Path

_RUNTIME_UTILS_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "agent_action_guard" / "_runtime_utils.py"
)
_REAL_ONNX_BASE_URL = (
    "https://huggingface.co/llmware/bling-tiny-llama-onnx/resolve/main/"
)
_REAL_GGUF_URL = (
    "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/"
    "mmproj-tinygemma3.gguf"
)


def _load_runtime_utils_module():
    spec = importlib.util.spec_from_file_location(
        "embedding_real_url_runtime_under_test", _RUNTIME_UTILS_SOURCE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_onnx_url_download_is_cached():
    runtime_module = _load_runtime_utils_module()
    model_url = f"{_REAL_ONNX_BASE_URL}model.onnx"
    model_file = runtime_module._remote_cache_path(model_url, "onnx", "model.onnx")

    runtime_module._download_file(model_url, model_file)
    first_stat = model_file.stat()
    assert first_stat.st_size > 100_000

    runtime_module._download_file(model_url, model_file)
    second_stat = model_file.stat()
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns


def test_real_gguf_url_download_is_cached():
    runtime_module = _load_runtime_utils_module()
    model_file = runtime_module._remote_cache_path(_REAL_GGUF_URL, "gguf", "model.gguf")

    runtime_module._download_file(_REAL_GGUF_URL, model_file)
    first_stat = model_file.stat()
    assert first_stat.st_size == 1_039_072
    with model_file.open("rb") as model_stream:
        assert model_stream.read(4) == b"GGUF"

    runtime_module._download_file(_REAL_GGUF_URL, model_file)
    second_stat = model_file.stat()
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns
