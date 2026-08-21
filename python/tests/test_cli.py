import json

from agent_action_guard import cli


def test_load_actions_accepts_direct_json():
    actions = cli.load_actions(
        json.dumps(
            {
                "type": "function",
                "function": {"name": "ping", "arguments": {}},
            }
        )
    )

    assert len(actions) == 1
    assert actions[0]["function"]["name"] == "ping"


def test_load_actions_reads_json_array_and_jsonl(tmp_path):
    json_path = tmp_path / "actions.json"
    json_path.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")
    jsonl_path = tmp_path / "actions.jsonl"
    jsonl_path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    assert cli.load_actions(file_path=str(json_path)) == [{"id": 1}, {"id": 2}]
    assert cli.load_actions(file_path=str(jsonl_path)) == [{"id": 1}, {"id": 2}]


def test_summarize_results_counts_safe_and_unsafe():
    assert cli.summarize_results(
        [(None, 0.9), ("harmful", 0.8), ("unethical", 0.7)]
    ) == (1, 2)


def test_main_classifies_file_and_prints_counts(tmp_path, monkeypatch, capsys):
    json_path = tmp_path / "actions.json"
    json_path.write_text(
        json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]), encoding="utf-8"
    )
    calls = []

    def fake_classify(actions, batch_size=None):
        calls.append((actions, batch_size))
        return [(None, 0.9), ("harmful", 0.8), (None, 0.7)]

    monkeypatch.setattr(cli, "is_actions_harmful", fake_classify)

    assert cli.main(["--file", str(json_path), "--batch-size", "2"]) == 0
    assert len(calls[0][0]) == 3
    assert calls[0][1] == 2
    assert capsys.readouterr().out.splitlines() == [
        "Total actions: 3",
        "Safe actions: 2",
        "Unsafe actions: 1",
    ]
