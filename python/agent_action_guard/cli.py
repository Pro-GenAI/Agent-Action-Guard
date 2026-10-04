"""Command-line interface for classifying one or more agent actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .action_classifier import is_actions_harmful
from .server import DEFAULT_HOST, DEFAULT_PORT, run_server


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
        prog="agent-action-guard",
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


def build_serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-action-guard serve",
        description="Serve the Agent Action Guard HTTP classification API.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default maximum actions per vectorized inference batch",
    )
    return parser


def build_harmactionseval_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-action-guard harmactionseval",
        description="Run HarmActionsEval to measure harmful tool-call behavior.",
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-path")
    parser.add_argument("--output")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
    )
    return parser


def _run_harmactionseval(argv: list[str]) -> int:
    try:
        from . import harmactionseval
    except ModuleNotFoundError as exc:
        if exc.name in {"rich", "dotenv"}:
            raise RuntimeError(
                'HarmActionsEval dependencies are not installed. Install '
                '"agent-action-guard[harmactionseval]" first.'
            ) from exc
        raise
    return harmactionseval.main(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hooks":
        from . import hooks

        return hooks.main(argv[1:])

    if argv and argv[0] == "harmactionseval":
        if any(arg in {"-h", "--help"} for arg in argv[1:]):
            build_harmactionseval_help_parser().print_help()
            return 0
        try:
            return _run_harmactionseval(argv[1:])
        except RuntimeError as exc:
            print(f"agent-action-guard: error: {exc}", file=sys.stderr)
            return 2

    if argv and argv[0] == "serve":
        parser = build_serve_parser()
        args = parser.parse_args(argv[1:])
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        if args.batch_size is not None and args.batch_size <= 0:
            parser.error("--batch-size must be greater than zero")
        run_server(args.host, args.port, batch_size=args.batch_size)
        return 0

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
