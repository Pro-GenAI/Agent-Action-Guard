"""Local mock LLM/API server used by harness end-to-end tests."""

from __future__ import annotations

import base64
import binascii
import json
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MOCK_MODEL = "agent-action-guard-harmful-mock"
DATASET_PATH = Path(__file__).parents[1] / "agent_action_guard" / "harmactions_dataset.json"
MARKER_NAME = "aag-harness-tool-executed"


def harmful_dataset_action() -> dict[str, Any]:
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("action_id") == 208:
            return row["action"]
    raise AssertionError("HarmActionsEval action_id=208 is missing")


def harmful_command() -> str:
    action = harmful_dataset_action()["function"]
    arguments = action["arguments"]
    flags = " ".join(f"--{key} {value}" for key, value in arguments.items())
    return f"touch {MARKER_NAME} # {action['name']} {flags}"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _fake_jwt() -> str:
    def encode(value: dict[str, Any]) -> str:
        payload = base64.urlsafe_b64encode(_json_bytes(value)).decode().rstrip("=")
        return payload

    return f"{encode({'alg': 'none'})}.{encode({'exp': 4102444800})}.x"


def _pb_varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _pb_bytes(field_number: int, value: bytes) -> bytes:
    return _pb_varint((field_number << 3) | 2) + _pb_varint(len(value)) + value


def _pb_string(field_number: int, value: str) -> bytes:
    return _pb_bytes(field_number, value.encode())


def _cursor_model_details() -> bytes:
    return b"".join(
        [
            _pb_string(1, MOCK_MODEL),
            _pb_string(3, MOCK_MODEL),
            _pb_string(4, MOCK_MODEL),
            _pb_string(5, MOCK_MODEL),
            _pb_string(6, MOCK_MODEL),
        ]
    )


def _event_header(name: str, value: str) -> bytes:
    name_bytes = name.encode()
    value_bytes = value.encode()
    return bytes([len(name_bytes)]) + name_bytes + bytes([7]) + struct.pack(">H", len(value_bytes)) + value_bytes


def aws_event_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    headers = b"".join(
        [
            _event_header(":message-type", "event"),
            _event_header(":event-type", event_type),
            _event_header(":content-type", "application/json"),
        ]
    )
    body = _json_bytes(payload)
    total_len = 16 + len(headers) + len(body)
    prelude = struct.pack(">II", total_len, len(headers))
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + headers + body
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


class HarnessMockServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._protocol_calls: dict[str, int] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner._handle(self)

            def do_POST(self) -> None:
                owner._handle(self)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def start(self) -> "HarnessMockServer":
        self.thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def _count(self, protocol: str) -> int:
        value = self._protocol_calls.get(protocol, 0)
        self._protocol_calls[protocol] = value + 1
        return value

    def _record(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        self.requests.append(
            {
                "method": handler.command,
                "path": handler.path,
                "target": handler.headers.get("x-amz-target"),
                "body": body,
            }
        )

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        body: bytes,
        *,
        content_type: str = "application/json",
        status: int = 200,
    ) -> None:
        handler.send_response(status)
        handler.send_header("content-type", content_type)
        handler.send_header("content-length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("content-length", "0") or 0)
        body = handler.rfile.read(length) if length else b""
        self._record(handler, body)
        path = handler.path.split("?", 1)[0]
        target = handler.headers.get("x-amz-target", "")

        if path == "/auth/exchange_user_api_key":
            self._send(
                handler,
                _json_bytes({"accessToken": _fake_jwt(), "refreshToken": "test"}),
            )
            return
        if path == "/v1/models":
            self._send(
                handler,
                _json_bytes(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MOCK_MODEL,
                                "object": "model",
                                "created": 0,
                                "owned_by": "agent-action-guard",
                            }
                        ],
                    }
                ),
            )
            return
        if path == f"/v1/models/{MOCK_MODEL}":
            self._send(handler, _json_bytes({"id": MOCK_MODEL, "object": "model", "type": "model"}))
            return
        if path.endswith("/GetUsableModels"):
            self._send(
                handler,
                _pb_bytes(1, _cursor_model_details()),
                content_type="application/proto",
            )
            return
        if path.endswith("/GetDefaultModelForCli"):
            self._send(
                handler,
                _pb_bytes(1, _cursor_model_details()),
                content_type="application/proto",
            )
            return
        if path == "/agent.v1.AgentService/RunSSE":
            self._cursor_stream(handler)
            return
        if path.startswith("/aiserver."):
            self._send(handler, b"", content_type="application/proto")
            return
        if path == "/v1/responses":
            self._openai_responses(handler, body)
            return
        if path == "/v1/messages":
            self._anthropic_messages(handler, body)
            return
        if target.endswith("GetProfile"):
            self._send(
                handler,
                _json_bytes(
                    {
                        "profileArn": "arn:aws:codewhisperer:us-east-1:123456789012:profile/mock",
                        "profileName": "mock",
                    }
                ),
                content_type="application/x-amz-json-1.0",
            )
            return
        if target.endswith("ListAvailableModels"):
            model = {
                "modelId": MOCK_MODEL,
                "modelName": MOCK_MODEL,
                "description": "Agent Action Guard harness mock",
            }
            self._send(
                handler,
                _json_bytes({"models": [model], "defaultModel": model}),
                content_type="application/x-amz-json-1.0",
            )
            return
        if target.endswith("GetUsageLimits"):
            self._send(handler, b"{}", content_type="application/x-amz-json-1.0")
            return
        if target.endswith("GenerateAssistantResponse"):
            self._kiro_stream(handler)
            return

        self._send(handler, _json_bytes({"error": "unsupported mock route"}), status=404)

    def _openai_responses(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        request = json.loads(body or b"{}")
        turn = self._count("openai")
        response_id = f"resp_{uuid.uuid4().hex}"
        if turn == 0:
            tools = [
                item
                for item in request.get("tools", [])
                if isinstance(item, dict) and item.get("name")
            ]
            selected = next(
                (item for item in tools if "shell" in item["name"].lower()),
                next(
                    (item for item in tools if "command" in item["name"].lower()),
                    {"name": "shell", "parameters": {}},
                ),
            )
            tool_name = selected["name"]
            parameters = selected.get("parameters") or {}
            required = parameters.get("required") or []
            argument_name = "cmd" if "cmd" in required else "command"
            call_id = "call_aag_harmful"
            item = {
                "id": "fc_aag_harmful",
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "arguments": json.dumps({argument_name: harmful_command()}),
                "status": "completed",
            }
            events = [
                {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}},
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": 0,
                    "arguments": item["arguments"],
                },
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "output": [item],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                },
            ]
        else:
            events = [
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "output": [
                            {
                                "id": "msg_aag_done",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "The harmful tool action was denied.",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                }
            ]
        payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        payload += "data: [DONE]\n\n"
        self._send(handler, payload.encode(), content_type="text/event-stream")

    def _cursor_stream(self, handler: BaseHTTPRequestHandler) -> None:
        shell_args = b"".join(
            [
                _pb_string(1, harmful_command()),
                _pb_string(4, "aag-harmful"),
            ]
        )
        shell_tool_call = _pb_bytes(1, shell_args)
        tool_call = _pb_bytes(1, shell_tool_call)
        started = b"".join(
            [
                _pb_string(1, "aag-harmful"),
                _pb_bytes(2, tool_call),
                _pb_string(3, "aag-model-call"),
            ]
        )
        tool_started_update = _pb_bytes(2, started)
        tool_started_message = _pb_bytes(1, tool_started_update)
        tool_completed_update = _pb_bytes(3, started)
        tool_completed_message = _pb_bytes(1, tool_completed_update)

        turn_ended = _pb_bytes(14, b"")
        turn_message = _pb_bytes(1, turn_ended)

        def envelope(message: bytes) -> bytes:
            return bytes([0]) + struct.pack(">I", len(message)) + message

        end_payload = b"{}"
        end_envelope = bytes([2]) + struct.pack(">I", len(end_payload)) + end_payload

        # Cursor dispatches tool execution while RunSSE is still open. Flush the
        # proposal first so its native preToolUse hook runs before turn completion.
        handler.send_response(200)
        handler.send_header("content-type", "application/connect+proto")
        handler.send_header("connection", "close")
        handler.end_headers()
        handler.wfile.write(
            envelope(tool_started_message) + envelope(tool_completed_message)
        )
        handler.wfile.flush()
        time.sleep(1.0)
        handler.wfile.write(envelope(turn_message) + end_envelope)
        handler.wfile.flush()

    def _anthropic_messages(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        request = json.loads(body or b"{}")
        turn = self._count("anthropic")
        tool_names = [
            item.get("name")
            for item in request.get("tools", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        tool_name = next(
            (name for name in tool_names if name.lower() in {"bash", "shell", "execute_bash"}),
            next((name for name in tool_names if "bash" in name.lower()), "Bash"),
        )
        if turn == 0:
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_aag_harmful",
                    "name": tool_name,
                    "input": {"command": harmful_command()},
                }
            ]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": "The harmful tool action was denied."}]
            stop_reason = "end_turn"

        response = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": MOCK_MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not request.get("stream"):
            self._send(handler, _json_bytes(response))
            return

        message_start = {
            "type": "message_start",
            "message": {
                **response,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
        if turn == 0:
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_aag_harmful",
                    "name": tool_name,
                    "input": {},
                },
            }
            block_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"command": harmful_command()}),
                },
            }
        else:
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            block_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "The harmful tool action was denied.",
                },
            }
        events = [
            message_start,
            block_start,
            block_delta,
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        payload = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            for event in events
        )
        self._send(handler, payload.encode(), content_type="text/event-stream")

    def _kiro_stream(self, handler: BaseHTTPRequestHandler) -> None:
        turn = self._count("kiro")
        if turn == 0:
            payload = b"".join(
                [
                    aws_event_frame(
                        "toolUseEvent",
                        {
                            "toolUseId": "aag-harmful",
                            "name": "shell",
                            "input": json.dumps({"command": harmful_command()}),
                            "stop": True,
                        },
                    ),
                    aws_event_frame(
                        "messageMetadataEvent",
                        {"conversationId": "aag-conversation", "utteranceId": "aag-utterance"},
                    ),
                ]
            )
        else:
            payload = b"".join(
                [
                    aws_event_frame(
                        "assistantResponseEvent",
                        {"content": "The harmful tool action was denied."},
                    ),
                    aws_event_frame(
                        "messageMetadataEvent",
                        {"conversationId": "aag-conversation", "utteranceId": "aag-utterance-2"},
                    ),
                ]
            )
        self._send(handler, payload, content_type="application/vnd.amazon.eventstream")
