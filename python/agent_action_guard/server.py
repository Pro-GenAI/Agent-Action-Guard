"""Dependency-free HTTP API for Agent Action Guard classification."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .action_classifier import is_actions_harmful

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_BODY_BYTES = 1_048_576


def classify_payload(
    payload,
    *,
    classify_actions: Callable = is_actions_harmful,
    batch_size: int | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("Request body must be a JSON object")

    has_action = "action" in payload
    has_actions = "actions" in payload
    if has_action == has_actions:
        raise ValueError("Provide exactly one of 'action' or 'actions'")

    actions = payload["actions"] if has_actions else [payload["action"]]
    if not isinstance(actions, list) or not actions:
        raise ValueError("'actions' must be a non-empty JSON array")

    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            raise TypeError(f"action {index} must be a JSON object")

    request_batch_size = payload.get("batch_size", batch_size)
    if request_batch_size is not None and (
        isinstance(request_batch_size, bool)
        or not isinstance(request_batch_size, int)
        or request_batch_size <= 0
    ):
        raise ValueError("'batch_size' must be a positive integer")

    results = classify_actions(actions, batch_size=request_batch_size)
    items = [
        {"label": label, "confidence": confidence, "safe": label is None}
        for label, confidence in results
    ]
    safe = sum(item["safe"] for item in items)
    return {
        "results": items,
        "summary": {
            "total": len(items),
            "safe": safe,
            "unsafe": len(items) - safe,
        },
    }



def create_api_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    classify_actions: Callable = is_actions_harmful,
    batch_size: int | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer for the Action Guard API."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AgentActionGuardHTTP/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/classify":
                self._send_json(404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return

            if content_length <= 0:
                self._send_json(400, {"error": "Request body is required"})
                return
            if content_length > max_body_bytes:
                self._send_json(413, {"error": "Request body too large"})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
                response = classify_payload(
                    payload,
                    classify_actions=classify_actions,
                    batch_size=batch_size,
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid JSON"})
                return
            except (TypeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception:  # noqa: BLE001 - keep internal failures out of API responses
                self._send_json(500, {"error": "Classification failed"})
                return

            self._send_json(200, response)

        def log_message(self, format: str, *args) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    batch_size: int | None = None,
) -> None:
    """Run the Action Guard API until interrupted."""
    server = create_api_server(host, port, batch_size=batch_size)
    print(f"Agent Action Guard API listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
