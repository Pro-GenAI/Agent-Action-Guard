#!/usr/bin/env python3
"""Run a defense-in-depth security audit for the Python package.

No single scanner can detect every vulnerability class. This script combines
dependency auditing, SAST, secret scanning, and repository-specific risky-pattern
checks. It does not install tools.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
REPORT_DIR = ROOT / "security-reports"

SOURCE_PATHS = (
    ROOT / "agent_action_guard",
    ROOT / "examples",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "training",
    ROOT / "sitecustomize.py",
)

EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-venvs",
    ".venv",
    "__pycache__",
    "agent_action_guard.egg-info",
    "build",
    "dist",
    "unused",
    "security-reports",
}

RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dynamic-code", re.compile(r"(?<![.\w])(?:eval|exec)\s*\(")),
    ("unsafe-deserialization", re.compile(r"\b(?:pickle\.loads?|yaml\.load)\s*\(")),
    (
        "shell-execution",
        re.compile(
            r"\b(?:os\.system|subprocess\.(?:Popen|run|call|check_call|check_output))\s*\("
        ),
    ),
    ("shell-true", re.compile(r"\bshell\s*=\s*True\b")),
    ("weak-hash", re.compile(r"\bhashlib\.(?:md5|sha1)\s*\(")),
    (
        "insecure-tls",
        re.compile(r"\bverify\s*=\s*False\b|CERT_NONE|check_hostname\s*=\s*False"),
    ),
    ("tempfile-race", re.compile(r"\btempfile\.mktemp\s*\(")),
    (
        "hardcoded-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"
        ),
    ),
)

BLOCKING_PATTERN_CATEGORIES = {
    "dynamic-code",
    "unsafe-deserialization",
    "shell-true",
    "weak-hash",
    "insecure-tls",
    "tempfile-race",
}


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]
    report: str
    cwd: Path = ROOT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="fail when an external scanner is not installed",
    )
    parser.add_argument(
        "--no-semgrep",
        action="store_true",
        help="skip Semgrep even when installed",
    )
    return parser.parse_args(argv)


def executable_exists(name: str) -> bool:
    return shutil.which(name) is not None


def python_module_exists(name: str) -> bool:
    probe = subprocess.run(
        (sys.executable, "-c", f"import {name}"),
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def run_check(check: Check) -> int:
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / check.report
    with report_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            check.command,
            cwd=check.cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    print(
        f"[{check.name}] exit={result.returncode} "
        f"report={report_path.relative_to(ROOT)}"
    )
    return result.returncode


def run_detect_secrets(check: Check) -> int:
    status = run_check(check)
    if status != 0:
        return status
    try:
        payload = json.loads((REPORT_DIR / check.report).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("[detect-secrets] invalid JSON report")
        return 1
    results = payload.get("results", {})
    finding_count = 0
    ignored_test_keywords = 0
    for filename, entries in results.items():
        is_test_fixture = (
            "/tests/" in filename
            or "/test/" in filename
            or "/runtime-fixtures/" in filename
        )
        for entry in entries:
            if is_test_fixture and entry.get("type") == "Secret Keyword":
                ignored_test_keywords += 1
                continue
            finding_count += 1
    print(
        f"[detect-secrets] findings={finding_count} "
        f"ignored-test-keywords={ignored_test_keywords}"
    )
    return 1 if finding_count else 0


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in SOURCE_PATHS:
        if base.is_file() and base.suffix == ".py":
            files.append(base)
            continue
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            if not any(part in EXCLUDED_PARTS for part in path.parts):
                files.append(path)
    return sorted(set(files))


def run_builtin_pattern_scan() -> int:
    findings: list[dict[str, object]] = []
    for path in iter_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(source.splitlines(), start=1):
            for category, pattern in RISK_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "category": category,
                            "file": str(path.relative_to(ROOT)),
                            "line": line_number,
                            "text": line.strip()[:300],
                        }
                    )

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "python-dangerous-patterns.json"
    report_path.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    blocking = [
        finding
        for finding in findings
        if finding["category"] in BLOCKING_PATTERN_CATEGORIES
    ]
    print(
        f"[dangerous-patterns] findings={len(findings)} blocking={len(blocking)} "
        f"report={report_path.relative_to(ROOT)}"
    )
    return 1 if blocking else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("bash") is None:
        print(
            "Security audit requires Bash (Linux environment recommended). "
            "Bash was not found; aborting before security checks.",
            file=sys.stderr,
        )
        return 1
    failures: list[str] = []
    missing: list[str] = []

    def record(name: str, status: int) -> None:
        if status != 0:
            failures.append(name)

    record("dangerous-patterns", run_builtin_pattern_scan())

    module_checks = (
        (
            "bandit",
            "bandit",
            (
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "agent_action_guard",
                "examples",
                "scripts",
                "training",
                "-ll",
                "-f",
                "json",
            ),
            "bandit.json",
        ),
        (
            "ruff-security",
            "ruff",
            (
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "S",
                "agent_action_guard",
                "examples",
                "scripts",
                "training",
                "sitecustomize.py",
                "--ignore",
                "S101,S106,S603,S607",
                "--output-format",
                "json",
            ),
            "ruff-security.json",
        ),
    )

    for name, module_name, command, report in module_checks:
        if python_module_exists(module_name):
            record(name, run_check(Check(name, command, report)))
        else:
            print(f"[{name}] skipped: Python module '{module_name}' is not installed")
            missing.append(module_name)

    with tempfile.TemporaryDirectory(prefix="aag-security-") as temp_dir:
        requirements = Path(temp_dir) / "requirements.txt"
        if executable_exists("uv"):
            export_command = [
                "uv",
                "export",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
            ]
            if (ROOT / "uv.lock").exists():
                export_command.insert(2, "--frozen")
            export = subprocess.run(
                tuple(export_command),
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if export.returncode != 0:
                print("[pip-audit] dependency export failed")
                failures.append("pip-audit (dependency export)")
            elif python_module_exists("pip_audit"):
                record(
                    "pip-audit",
                    run_check(
                        Check(
                            "pip-audit",
                            (
                                sys.executable,
                                "-m",
                                "pip_audit",
                                "-r",
                                str(requirements),
                                "--progress-spinner",
                                "off",
                                "--format",
                                "json",
                            ),
                            "pip-audit.json",
                        )
                    )
                )
            else:
                print("[pip-audit] skipped: Python module 'pip_audit' is not installed")
                missing.append("pip-audit")
        else:
            print("[pip-audit] skipped: 'uv' is not installed")
            missing.append("uv")

    if executable_exists("detect-secrets"):
        record(
            "detect-secrets",
            run_detect_secrets(
                Check(
                    "detect-secrets",
                    (
                        "detect-secrets",
                        "scan",
                        "--all-files",
                        "--exclude-files",
                        r"(^|/)(\.venv|\.uv-venvs|build|dist|unused|__pycache__|agent_action_guard\.egg-info|security-reports)(/|$)",
                        "python/agent_action_guard",
                        "python/examples",
                        "python/scripts",
                        "python/tests",
                        "python/training",
                        "python/sitecustomize.py",
                    ),
                    "detect-secrets.json",
                    cwd=REPO_ROOT,
                )
            )
        )
    else:
        print("[detect-secrets] skipped: executable not installed")
        missing.append("detect-secrets")

    if not args.no_semgrep:
        if executable_exists("semgrep"):
            record(
                "semgrep",
                run_check(
                    Check(
                        "semgrep",
                        (
                            "semgrep",
                            "scan",
                            "--config",
                            "auto",
                            "--severity",
                            "ERROR",
                            "--error",
                            "--json",
                            "--exclude",
                            ".venv",
                            "--exclude",
                            ".uv-venvs",
                            "--exclude",
                            "build",
                            "--exclude",
                            "dist",
                            "--exclude",
                            "unused",
                            "--exclude",
                            "scripts/security_audit.py",
                            "agent_action_guard",
                            "examples",
                            "scripts",
                            "tests",
                            "training",
                            "sitecustomize.py",
                        ),
                        "semgrep.json",
                    )
                )
            )
        else:
            print("[semgrep] skipped: executable not installed")
            missing.append("semgrep")

    if missing:
        print("Missing scanners/tools: " + ", ".join(sorted(set(missing))))
        if args.strict_tools:
            failures.append("missing required scanners/tools")

    color = sys.stdout.isatty()
    red, reset = ("\033[31m", "\033[0m") if color else ("", "")
    print("\n=== Python security audit summary ===")
    print(f"Failed checks: {len(failures)}")
    print(f"Missing scanners/tools: {', '.join(sorted(set(missing))) if missing else 'none'}")
    if failures:
        for name in failures:
            print(f"{red}FAIL: {name}{reset}")
        print(f"{red}Result: SECURITY AUDIT FAILED{reset}")
    else:
        print("Result: security audit passed" + (" (some scanners skipped)" if missing else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
