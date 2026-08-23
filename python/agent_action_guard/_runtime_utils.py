"""
Runtime helpers for action classification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import numpy as np
import openai

ALL_CLASSES = ["safe", "harmful", "unethical"]
ONNX_MODEL_PATH = Path(__file__).with_name("action_classifier_model.onnx")

DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL_NAME)
AAG_EMBED_GGUF_ENV = "AAG_EMBED_GGUF"
AAG_EMBED_ONNX_ENV = "AAG_EMBED_ONNX"
_HF_BASE_URL = "https://huggingface.co/"
_DEFAULT_ONNX_REPO = "onnx-models/all-MiniLM-L6-v2-onnx"
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_onnx_asset_url(filename: str) -> str:
    return f"{_HF_BASE_URL}{_DEFAULT_ONNX_REPO}/resolve/main/{filename}"


def _download_file(url: str, destination: Path) -> None:
    """Download a runtime asset atomically if it is not already cached."""
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with urllib.request.urlopen(
            url, timeout=120
        ) as response, tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=str(destination.parent)
        ) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(response, temp_file)
        temp_path.replace(destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _normalize_source_component(value: str, fallback: str = "source") -> str:
    """Normalize arbitrary source text for safe cross-platform cache names."""
    normalized = unquote(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).replace("-.", ".")
    normalized = normalized.strip(" ._-")
    return normalized or fallback


def _normalized_url_cache_name(url: str) -> str:
    parsed = urlsplit(url)
    readable = _normalize_source_component(
        f"{parsed.scheme}-{parsed.netloc}-{parsed.path}", fallback="remote"
    )[:96]
    canonical = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{fingerprint}"


def _looks_like_hf_repo_id(value: str) -> bool:
    if not _HF_REPO_ID_RE.fullmatch(value):
        return False
    candidate = Path(value).expanduser()
    if candidate.exists() or candidate.parent.exists():
        return False
    return not value.startswith(("/", "./", "../", "~")) and "\\" not in value


def _hf_repo_base_url(repo_id: str) -> str:
    return f"{_HF_BASE_URL}{repo_id}/resolve/main/"


def _normalize_hf_source_url(url: str) -> tuple[str, bool]:
    """Convert Hugging Face browse/repo URLs to download URLs when possible."""
    parsed = urlsplit(url)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return url, parsed.path.endswith("/")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 2:
        base_path = f"/{parts[0]}/{parts[1]}/resolve/main/"
        return (
            urlunsplit((parsed.scheme, parsed.netloc, base_path, parsed.query, "")),
            True,
        )
    if len(parts) >= 4 and parts[2] == "blob":
        path = "/" + "/".join([parts[0], parts[1], "resolve", *parts[3:]])
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")), False
    if len(parts) >= 4 and parts[2] == "tree":
        path = "/" + "/".join([parts[0], parts[1], "resolve", *parts[3:]]) + "/"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")), True
    if len(parts) == 4 and parts[2] == "resolve":
        path = "/" + "/".join(parts) + "/"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, "")), True
    return url, parsed.path.endswith("/")


def _remote_model_url(source: str, default_filename: str) -> str:
    source_url = _hf_repo_base_url(source) if _looks_like_hf_repo_id(source) else source
    source_url, is_base = _normalize_hf_source_url(source_url)
    if is_base:
        parsed = urlsplit(source_url)
        base_path = parsed.path if parsed.path.endswith("/") else f"{parsed.path}/"
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"{base_path}{default_filename}",
                parsed.query,
                "",
            )
        )
    return source_url


def _sibling_url(url: str, filename: str) -> str:
    parsed = urlsplit(url)
    parent = parsed.path.rsplit("/", 1)[0]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{parent}/{filename}", parsed.query, "")
    )


def _remote_cache_path(
    source_url: str,
    kind: str,
    default_filename: str,
    home_dir: Path | None = None,
) -> Path:
    cache_root = (home_dir or Path.home()) / ".cache" / "agent-action-guard" / kind
    parsed = urlsplit(source_url)
    basename = Path(unquote(parsed.path)).name
    normalized_filename = _normalize_source_component(basename, default_filename)
    expected_suffix = Path(default_filename).suffix.lower()
    if expected_suffix and not normalized_filename.endswith(expected_suffix):
        normalized_filename = f"{normalized_filename}{expected_suffix}"
    return cache_root / _normalized_url_cache_name(source_url) / normalized_filename


def _resolve_remote_onnx_model_files(
    source: str, home_dir: Path | None = None
) -> tuple[Path, Path]:
    model_url = _remote_model_url(source, "model.onnx")
    model_path = _remote_cache_path(model_url, "onnx", "model.onnx", home_dir=home_dir)
    vocab_path = model_path.with_name("tokenizer.json")
    _download_file(model_url, model_path)
    _download_file(_sibling_url(model_url, "tokenizer.json"), vocab_path)
    return model_path, vocab_path


def _resolve_onnx_model_files(home_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve the configured ONNX model or the cached default model assets."""
    configured = os.getenv(AAG_EMBED_ONNX_ENV)
    if configured:
        if _is_http_url(configured) or _looks_like_hf_repo_id(configured):
            return _resolve_remote_onnx_model_files(configured, home_dir=home_dir)
        configured_path = Path(configured).expanduser()
        if configured_path.suffix.lower() == ".onnx":
            model_path = configured_path
        else:
            model_path = configured_path / "model.onnx"
        return model_path, model_path.with_name("tokenizer.json")

    cache_dir = (
        (home_dir or Path.home()) / ".cache" / "agent-action-guard" / "all-MiniLM-L6-v2"
    )
    model_path = cache_dir / "model.onnx"
    tokenizer_path = cache_dir / "tokenizer.json"
    _download_file(_default_onnx_asset_url("model.onnx"), model_path)
    _download_file(_default_onnx_asset_url("tokenizer.json"), tokenizer_path)
    return model_path, tokenizer_path


def _resolve_gguf_model_file(home_dir: Path | None = None) -> Path:
    """Resolve the configured local, URL, or Hugging Face GGUF model source."""
    configured = os.getenv(AAG_EMBED_GGUF_ENV)
    if not configured:
        raise ValueError(f"{AAG_EMBED_GGUF_ENV} is not configured")

    if _is_http_url(configured) or _looks_like_hf_repo_id(configured):
        model_url = _remote_model_url(configured, "model.gguf")
        model_path = _remote_cache_path(
            model_url, "gguf", "model.gguf", home_dir=home_dir
        )
        _download_file(model_url, model_path)
        return model_path

    configured_path = Path(configured).expanduser()
    if configured_path.suffix.lower() == ".gguf":
        return configured_path
    return configured_path / "model.gguf"


class EmbeddingModel:
    """Embedding backend with local GGUF/ONNX configuration precedence."""

    def __init__(self, model_name: str | None = None):
        env_model_name = os.getenv("EMBED_MODEL_NAME")
        self.model_name = model_name or env_model_name or DEFAULT_EMBED_MODEL_NAME
        self.client = None
        self.gguf_model = None
        self.onnx_session = None
        self.onnx_tokenizer = None

        local_gguf_configured = bool(os.getenv(AAG_EMBED_GGUF_ENV))
        local_onnx_configured = bool(os.getenv(AAG_EMBED_ONNX_ENV))
        model_configured = bool(model_name is not None or env_model_name)
        api_configured = bool(
            os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        if local_gguf_configured:
            self.backend = "gguf"
        elif local_onnx_configured:
            self.backend = "onnx"
        elif model_configured or api_configured:
            self.backend = "api"
        else:
            self.backend = "onnx"

    def _get_client(self):
        if self.client is not None:
            return self.client

        client_kwargs = {}
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        # Instantiate the OpenAI client only when embeddings are actually requested.
        self.client = openai.OpenAI(**client_kwargs)
        return self.client

    def _get_gguf_runtime(self):
        if self.gguf_model is not None:
            return self.gguf_model

        model_file = _resolve_gguf_model_file()
        if not model_file.exists():
            raise FileNotFoundError(f"GGUF embedding model not found: {model_file}")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "GGUF embeddings require llama-cpp-python. Install it with "
                "`pip install llama-cpp-python`."
            ) from exc

        self.gguf_model = Llama(
            model_path=str(model_file),
            embedding=True,
            verbose=False,
        )
        return self.gguf_model

    def _encode_gguf(self, texts):
        model = self._get_gguf_runtime()
        embeddings = np.asarray(
            model.embed([str(text) for text in texts], normalize=True),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError(
                "GGUF embedding model must return one sentence embedding per input."
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    def _get_onnx_runtime(self):
        if self.onnx_session is not None and self.onnx_tokenizer is not None:
            return self.onnx_session, self.onnx_tokenizer

        import onnxruntime as ort
        from tokenizers import Tokenizer

        files = _resolve_onnx_model_files()
        model_file = files[0]
        vocab_file = files[1]
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX embedding model not found: {model_file}")
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"ONNX embedding tokenizer not found: {vocab_file}. "
                "Place tokenizer.json beside the ONNX model."
            )

        tokenizer = Tokenizer.from_file(str(vocab_file))
        tokenizer.enable_truncation(max_length=256)
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("ONNX embedding tokenizer does not define a [PAD] token.")
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        self.onnx_session = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        self.onnx_tokenizer = tokenizer
        return self.onnx_session, self.onnx_tokenizer

    def _encode_onnx(self, texts):
        runtime = self._get_onnx_runtime()
        session = runtime[0]
        tokenizer = runtime[1]
        items = tokenizer.encode_batch([str(text) for text in texts])
        input_ids = np.asarray([item.ids for item in items], dtype=np.int64)
        attention_mask = np.asarray(
            [item.attention_mask for item in items], dtype=np.int64
        )
        type_ids = np.asarray([item.type_ids for item in items], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }
        feed = {}
        for input_meta in session.get_inputs():
            if input_meta.name not in inputs:
                raise ValueError(
                    f"Unsupported ONNX embedding model input: {input_meta.name}"
                )
            feed[input_meta.name] = inputs[input_meta.name]

        outputs = session.run(None, feed)
        if not outputs:
            raise ValueError("ONNX embedding model returned no outputs.")

        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.ndim == 3:
            mask = attention_mask[..., np.newaxis].astype(np.float32)
            embeddings = (embeddings * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), 1e-9, None
            )
        elif embeddings.ndim != 2:
            raise ValueError(
                "ONNX embedding model must return a 2D sentence embedding or "
                "3D token embeddings."
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    def encode(self, texts, *_args, **_kwargs):
        if self.backend == "gguf":
            return self._encode_gguf(texts)
        if self.backend == "onnx":
            return self._encode_onnx(texts)

        responses = self._get_client().embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embs = [data.embedding for data in responses.data]
        return np.array(embs)


def flatten_action_to_text(action_data: dict[str, str | dict[str, Any]]) -> str:
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

    func_name = action_data["function"].get("name", "unknown_function")
    args_raw = action_data["function"].get("arguments", {})
    try:
        if isinstance(args_raw, dict):
            args_dict = args_raw
        elif isinstance(args_raw, str):
            args_dict = json.loads(args_raw)
        else:
            return f"Call function {func_name} with arguments: {args_raw}"

        args_flat = ", ".join(f"{k}={v}" for k, v in args_dict.items())
        return f"Call function {func_name}: {args_flat}"
    except (json.JSONDecodeError, TypeError):
        return f"Call function {func_name} with arguments: {args_raw}"


embed_model = EmbeddingModel()


class ActionGuardDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
