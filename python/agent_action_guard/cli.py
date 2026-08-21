"""Command-line interface for classifying one or more agent actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action_classifier import is_actions_harmful


def _normalize_actions(value, source: str) -> list[dict]:
    actions = value if isinstance(value, list) else [value]
    if not actions:
        return []
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            raise TypeError(f"{source}: action {index} must be a JSON object")
    return actions


def load_actions(
    action_json: str | None = None, file_path: str | None = None
) -> list[dict]:
    """Load actions from direct JSON, a JSON array file, or a JSONL file."""
    if bool(action_json) == bool(file_path):
        raise ValueError("Provide exactly one of ACTION_JSON or --file")

    if action_json is not None:
        try:
            value = json.loads(action_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid ACTION_JSON: {exc.msg}") from exc
        return _normalize_actions(value, "ACTION_JSON")

    path = Path(file_path or "")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc

    if path.suffix.lower() == ".jsonl":
        actions = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            actions.extend(_normalize_actions(value, f"{path}:{line_number}"))
        return actions

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    return _normalize_actions(value, str(path))


def summarize_results(results: list[tuple[str | None, float]]) -> tuple[int, int]:
    """Return safe and unsafe counts for batch classification results."""
    safe = sum(label is None for label, _ in results)
    return safe, len(results) - safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aag-classify",
        description="Classify agent tool-call actions as safe or unsafe.",
    )
    parser.add_argument(
        "action_json",
        nargs="?",
        metavar="ACTION_JSON",
        help="JSON object (or array) containing action data",
    )
    parser.add_argument(
        "--file",
        help="JSON array file or JSONL file containing actions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum actions per vectorized inference batch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    try:
        actions = load_actions(args.action_json, args.file)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    results = is_actions_harmful(actions, batch_size=args.batch_size)
    safe, unsafe = summarize_results(results)
    print(f"Total actions: {len(results)}")
    print(f"Safe actions: {safe}")
    print(f"Unsafe actions: {unsafe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
