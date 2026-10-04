"""Local mock LLM server used by harness end-to-end tests."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MOCK_MODEL = "aag-mock-harness-model"
HARMFUL_COMMAND = (
    "printf 'malware_generation payload=spyware' > aag-harmful-executed"
)


class MockLLMServer:
    """Serve OpenAI Responses and Anthropic Messages APIs for harness tests."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.request_headers: list[tuple[str, dict[str, str]]] = []
        self.post_bodies: list[tuple[str, Any]] = []
        self._tool_calls = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _body(self) -> bytes:
                transfer_encoding = self.headers.get("transfer-encoding", "").lower()
                if "chunked" in transfer_encoding:
                    chunks = []
                    while True:
                        line = self.rfile.readline().strip()
                        if not line:
                            continue
                        size = int(line.split(b";", 1)[0], 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        chunks.append(self.rfile.read(size))
                        self.rfile.read(2)
                    return b"".join(chunks)
                size = int(self.headers.get("content-length", "0") or 0)
                return self.rfile.read(size) if size else b""

            @staticmethod
            def _varint(value: int) -> bytes:
                out = bytearray()
                while value >= 0x80:
                    out.append((value & 0x7F) | 0x80)
                    value >>= 7
                out.append(value)
                return bytes(out)

            @classmethod
            def _proto_bytes(cls, field: int, value: bytes) -> bytes:
                key = cls._varint((field << 3) | 2)
                return key + cls._varint(len(value)) + value

            @classmethod
            def _proto_varint(cls, field: int, value: int) -> bytes:
                return cls._varint(field << 3) + cls._varint(value)

            @classmethod
            def _proto_string(cls, field: int, value: str) -> bytes:
                return cls._proto_bytes(field, value.encode())

            def _connect_proto(self, messages: list[bytes]) -> None:
                frames = []
                for message in messages:
                    frames.append(
                        b"\x00" + len(message).to_bytes(4, "big") + message
                    )
                trailer = b"{}"
                frames.append(
                    b"\x02" + len(trailer).to_bytes(4, "big") + trailer
                )
                self._bytes(
                    200,
                    b"".join(frames),
                    "application/connect+proto",
                )

            def _json(self, status: int, value: Any) -> None:
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(
                self,
                events: list[dict[str, Any]],
                *,
                named_events: bool = False,
            ) -> None:
                chunks = []
                for event in events:
                    prefix = f"event: {event['type']}\n" if named_events else ""
                    chunks.append(
                        f"{prefix}data: {json.dumps(event)}\n\n".encode()
                    )
                body = b"".join(chunks)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                owner.requests.append(("GET", self.path))
                path = self.path.split("?", 1)[0]
                if path == f"/v1/models/{MOCK_MODEL}":
                    self._json(
                        200,
                        {
                            "id": MOCK_MODEL,
                            "object": "model",
                            "display_name": MOCK_MODEL,
                            "created_at": "2026-01-01T00:00:00Z",
                            "type": "model",
                        },
                    )
                    return
                if path == "/v1/models":
                    self._json(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": MOCK_MODEL,
                                    "object": "model",
                                    "owned_by": "agent-action-guard",
                                }
                            ],
                        },
                    )
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                owner.requests.append(("POST", self.path))
                owner.request_headers.append((self.path, dict(self.headers.items())))
                path = self.path.split("?", 1)[0]
                body = self._body()
                try:
                    decoded_body: Any = json.loads(body or b"{}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    decoded_body = body
                owner.post_bodies.append((self.path, decoded_body))
                target = self.headers.get("x-amz-target", "")
                if target.endswith("GetProfile"):
                    self._json(200, {"profileArn": "arn:aws:codewhisperer:us-east-1:000000000000:profile/mock"})
                    return
                if target.endswith("ListAvailableModels"):
                    self._json(
                        200,
                        {
                            "models": [
                                {
                                    "modelId": MOCK_MODEL,
                                    "displayName": MOCK_MODEL,
                                    "isDefault": True,
                                }
                            ],
                            "defaultModelId": MOCK_MODEL,
                        },
                    )
                    return
                if path == "/agent.v1.AgentService/RunSSE":
                    owner._handle_cursor(self)
                    return
                if path == "/auth/exchange_user_api_key":
                    payload = base64.urlsafe_b64encode(
                        json.dumps({"exp": 4102444800}).encode()
                    ).decode().rstrip("=")
                    self._json(
                        200,
                        {
                            "accessToken": f"header.{payload}.signature",
                            "refreshToken": "mock-refresh",
                        },
                    )
                    return
                if path.startswith("/aiserver.v1."):
                    encoded = MOCK_MODEL.encode()
                    model_details = (
                        bytes([0x0A, len(encoded)])
                        + encoded
                        + bytes([0x1A, len(encoded)])
                        + encoded
                        + bytes([0x22, len(encoded)])
                        + encoded
                        + bytes([0x2A, len(encoded)])
                        + encoded
                    )
                    if path.endswith("/AvailableModels"):
                        proto = bytes([0x0A, len(encoded)]) + encoded
                    elif path.endswith(("/GetUsableModels", "/GetDefaultModelForCli")):
                        proto = bytes([0x0A, len(model_details)]) + model_details
                    else:
                        proto = b""
                    self._bytes(200, proto, "application/proto")
                    return
                if path == "/v1/responses":
                    owner._handle_openai(self, body)
                    return
                if path == "/v1/messages":
                    owner._handle_anthropic(self, body)
                    return
                if path == "/v1/messages/count_tokens":
                    self._json(200, {"input_tokens": 1})
                    return
                self._json(404, {"error": "not found"})

            def log_message(self, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _handle_cursor(self, handler: BaseHTTPRequestHandler) -> None:
        call_id = "aag-cursor-call-1"
        exec_id = "aag-cursor-exec-1"
        shell_args = (
            handler._proto_string(1, HARMFUL_COMMAND)
            + handler._proto_string(2, ".")
            + handler._proto_string(4, call_id)
        )
        exec_message = (
            handler._proto_varint(1, 1)
            + handler._proto_bytes(2, shell_args)
            + handler._proto_string(15, exec_id)
        )
        server_exec = handler._proto_bytes(2, exec_message)

        interaction_ended = handler._proto_bytes(14, b"")
        server_ended = handler._proto_bytes(1, interaction_ended)
        handler._connect_proto([server_exec, server_ended])

    def _handle_openai(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        request = json.loads(body or b"{}")
        if self._tool_calls == 0:
            self._tool_calls += 1
            tool_names = [
                item.get("name")
                for item in request.get("tools", [])
                if isinstance(item, dict)
            ]
            name = next(
                (
                    candidate
                    for candidate in ("exec_command", "shell", "run_terminal_cmd")
                    if candidate in tool_names
                ),
                tool_names[0] if tool_names else "exec_command",
            )
            arguments = {"cmd": HARMFUL_COMMAND}
            if name != "exec_command":
                arguments = {"command": HARMFUL_COMMAND}
            arg_text = json.dumps(arguments)
            item = {
                "id": "fc_aag_1",
                "type": "function_call",
                "call_id": "call_aag_1",
                "name": name,
                "arguments": arg_text,
            }
            events = [
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_aag_1",
                        "object": "response",
                        "status": "in_progress",
                        "model": MOCK_MODEL,
                        "output": [],
                    },
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "arguments": ""},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_aag_1",
                    "output_index": 0,
                    "delta": arg_text,
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_aag_1",
                    "output_index": 0,
                    "arguments": arg_text,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_aag_1",
                        "object": "response",
                        "status": "completed",
                        "model": MOCK_MODEL,
                        "output": [item],
                    },
                },
            ]
            handler._sse(events)  # type: ignore[attr-defined]
            return

        text_item = {
            "id": "msg_aag_2",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "The requested tool action was blocked.",
                    "annotations": [],
                }
            ],
        }
        handler._sse(  # type: ignore[attr-defined]
            [
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_aag_2",
                        "object": "response",
                        "status": "in_progress",
                        "model": MOCK_MODEL,
                        "output": [],
                    },
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": text_item,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": text_item,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_aag_2",
                        "object": "response",
                        "status": "completed",
                        "model": MOCK_MODEL,
                        "output": [text_item],
                    },
                },
            ]
        )

    def _handle_anthropic(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        request = json.loads(body or b"{}")
        has_tool_result = any(
            isinstance(message, dict)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in (
                    message.get("content", [])
                    if isinstance(message.get("content"), list)
                    else []
                )
            )
            for message in request.get("messages", [])
        )
        stream = bool(request.get("stream"))

        if not has_tool_result:
            arg_text = json.dumps({"command": HARMFUL_COMMAND})
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_aag_1",
                    "name": "Bash",
                    "input": {"command": HARMFUL_COMMAND},
                }
            ]
            if not stream:
                handler._json(  # type: ignore[attr-defined]
                    200,
                    {
                        "id": "msg_aag_1",
                        "type": "message",
                        "role": "assistant",
                        "model": MOCK_MODEL,
                        "content": content,
                        "stop_reason": "tool_use",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                )
                return
            handler._sse(  # type: ignore[attr-defined]
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_aag_1",
                            "type": "message",
                            "role": "assistant",
                            "model": MOCK_MODEL,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_aag_1",
                            "name": "Bash",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arg_text,
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ],
                named_events=True,
            )
            return

        content = [{"type": "text", "text": "The requested tool action was blocked."}]
        if not stream:
            handler._json(  # type: ignore[attr-defined]
                200,
                {
                    "id": "msg_aag_2",
                    "type": "message",
                    "role": "assistant",
                    "model": MOCK_MODEL,
                    "content": content,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
            return
        handler._sse(  # type: ignore[attr-defined]
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_aag_2",
                        "type": "message",
                        "role": "assistant",
                        "model": MOCK_MODEL,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": "The requested tool action was blocked.",
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ],
            named_events=True,
        )
