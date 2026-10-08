import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "security_audit.py"


def test_security_audit_fails_before_scanning_without_bash():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env={**os.environ, "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Bash was not found" in result.stderr
    assert "dangerous-patterns" not in result.stdout


def test_security_audit_has_named_failures_and_terminal_colors():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "failures: list[str] = []" in source
    assert 'record("dangerous-patterns", run_builtin_pattern_scan())' in source
    assert '"pip-audit",' in source
    assert '"detect-secrets",' in source
    assert '"semgrep",' in source
    assert '"\\033[31m"' in source
    assert "sys.stdout.isatty()" in source
    assert "=== Python security audit summary ===" in source
    assert "Result: SECURITY AUDIT FAILED" in source
