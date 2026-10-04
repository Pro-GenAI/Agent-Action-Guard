import json
import threading
import urllib.error
import urllib.request

import pytest

from agent_action_guard import cli
from agent_action_guard.server import classify_payload, create_api_server


def fake_classify(actions, batch_size=None):
    assert batch_size in (None, 2, 4)
    return [(None, 0.9) if index % 2 == 0 else ("harmful", 0.8) for index, _ in enumerate(actions)]


def test_classify_payload_supports_single_and_batch_actions():
    single = classify_payload({"action": {"id": 1}}, classify_actions=fake_classify)
    assert single["summary"] == {"total": 1, "safe": 1, "unsafe": 0}
    assert single["results"] == [{"label": None, "confidence": 0.9, "safe": True}]

    batch = classify_payload(
        {"actions": [{"id": 1}, {"id": 2}], "batch_size": 2},
        classify_actions=fake_classify,
    )
    assert batch["summary"] == {"total": 2, "safe": 1, "unsafe": 1}


@pytest.mark.parametrize(
    "payload,message",
    [
        ({}, "exactly one"),
        ({"action": {}, "actions": [{}]}, "exactly one"),
        ({"actions": []}, "non-empty"),
        ({"actions": [1]}, "action 1"),
        ({"action": {}, "batch_size": 0}, "positive integer"),
        ({"action": {}, "batch_size": True}, "positive integer"),
    ],
)
def test_classify_payload_rejects_invalid_requests(payload, message):
    with pytest.raises((TypeError, ValueError), match=message):
        classify_payload(payload, classify_actions=fake_classify)


def request_json(url, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_api_health_classification_and_validation():
    server = create_api_server("127.0.0.1", 0, classify_actions=fake_classify)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request_json(f"{base_url}/health") == (200, {"status": "ok"})

        status, body = request_json(
            f"{base_url}/v1/classify",
            method="POST",
            payload={"actions": [{"id": 1}, {"id": 2}], "batch_size": 2},
        )
        assert status == 200
        assert body["summary"] == {"total": 2, "safe": 1, "unsafe": 1}

        status, body = request_json(
            f"{base_url}/v1/classify",
            method="POST",
            payload={"actions": []},
        )
        assert status == 400
        assert "non-empty" in body["error"]

        assert request_json(f"{base_url}/missing")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_command_forwards_options(monkeypatch):
    calls = []

    def fake_run_server(host, port, *, batch_size=None):
        calls.append((host, port, batch_size))

    monkeypatch.setattr(cli, "run_server", fake_run_server)

    assert (
        cli.main(
            [
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )
    assert calls == [("0.0.0.0", 9000, 4)]
