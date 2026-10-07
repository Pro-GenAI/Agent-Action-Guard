import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize-harmactionseval-results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_harmactionseval_results",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def test_maximum_experimented_k_uses_only_model_level_k():
    cache = {
        "model-a": {
            "k": 1,
            "results": {"1": 3, "2": False},
            "last_attempt": {"2": 3},
        },
        "model-b": {
            "k": 2,
            "results": {"1": 1},
            "last_attempt": {},
        },
    }

    assert summarizer.model_max_experimented_k(cache["model-a"]) == 1
    assert summarizer.maximum_experimented_k(cache) == 2


def test_summarize_cache_reports_each_k_and_first_incomplete_threshold():
    cache = {
        "model-a": {
            "k": 3,
            "results": {
                "-1": 1,
                "1": 1,
                "2": False,
                "3": 3,
            },
            "last_attempt": {"2": 1},
        }
    }

    summaries = summarizer.summarize_cache(cache, max_k=3, expected_total=3)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["first_incomplete_k"] == 2

    assert summary["by_k"][1]["completed"] == 3
    assert summary["by_k"][1]["total"] == 3
    assert summary["by_k"][1]["score"] == pytest.approx(66.6666667)

    assert summary["by_k"][2]["completed"] == 2
    assert summary["by_k"][2]["total"] == 3
    assert summary["by_k"][2]["score"] == pytest.approx(50.0)

    assert summary["by_k"][3]["completed"] == 2
    assert summary["by_k"][3]["total"] == 3
    assert summary["by_k"][3]["score"] == pytest.approx(0.0)


def test_legacy_cache_without_last_attempt_uses_model_k_for_false_results():
    cache = {
        "legacy-model": {
            "k": 2,
            "results": {"1": False, "2": 1},
        }
    }

    summaries = summarizer.summarize_cache(cache, max_k=2, expected_total=2)

    assert summaries[0]["by_k"][1]["completed"] == 2
    assert summaries[0]["by_k"][2]["completed"] == 2
    assert summaries[0]["first_incomplete_k"] is None


def test_legacy_result_k_is_supported_for_per_k_coverage():
    cache = {
        "legacy-model": {
            "k": 3,
            "results": {"1": False, "2": 1},
            "result_k": {"1": 1},
        }
    }

    summaries = summarizer.summarize_cache(cache, max_k=3, expected_total=2)

    assert summaries[0]["by_k"][1]["completed"] == 2
    assert summaries[0]["by_k"][2]["completed"] == 1
    assert summaries[0]["by_k"][3]["completed"] == 1
    assert summaries[0]["first_incomplete_k"] == 2


def test_print_summary_shows_runs_for_every_k_without_coverage_percentage(capsys):
    summaries = [
        {
            "model": "model-a",
            "by_k": {
                1: {"score": 66.6666667, "completed": 3, "total": 3},
                2: {"score": 50.0, "completed": 2, "total": 3},
                3: {"score": 0.0, "completed": 2, "total": 3},
            },
            "first_incomplete_k": 2,
        }
    ]

    summarizer.print_summary(summaries, max_k=3)

    output = capsys.readouterr().out
    assert "SafeActions@1" in output
    assert "Runs@1" in output
    assert "SafeActions@2" in output
    assert "Runs@2" in output
    assert "SafeActions@3" in output
    assert "Runs@3" in output
    assert "Coverage" not in output
    assert "3/3" in output
    assert output.count("2/3") == 2
    assert "INCOMPLETE (k=2)" in output


def test_main_defaults_to_highest_experimented_k_and_k_caps_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "model-a": {
                    "k": 3,
                    "results": {"1": 1, "2": False},
                    "last_attempt": {"2": 3},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarizer, "expected_action_count", lambda: 2)

    monkeypatch.setattr(sys, "argv", ["summarize", str(cache_path)])
    summarizer.main()
    default_output = capsys.readouterr().out

    assert "SafeActions@1" in default_output
    assert "SafeActions@2" in default_output
    assert "SafeActions@3" in default_output

    monkeypatch.setattr(sys, "argv", ["summarize", str(cache_path), "--k", "2"])
    summarizer.main()
    capped_output = capsys.readouterr().out

    assert "SafeActions@1" in capped_output
    assert "SafeActions@2" in capped_output
    assert "SafeActions@3" not in capped_output
