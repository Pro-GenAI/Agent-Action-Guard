import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
MATRIX_SCRIPT = ROOT / "scripts" / "run_tests_uv_matrix.sh"


def test_makefile_exposes_selectable_python_test_matrix():
    assert "PYTHON_TEST_VERSIONS ?=" in MAKEFILE
    assert "test-matrix:" in MAKEFILE
    assert 'PYTHON_TEST_VERSIONS="$(PYTHON_TEST_VERSIONS)"' in MAKEFILE
    assert 'TEST_WORKERS="$(TEST_WORKERS)"' in MAKEFILE
    assert 'TEST_DIST="$(TEST_DIST)"' in MAKEFILE
    assert "bash scripts/run_tests_uv_matrix.sh" in MAKEFILE


def test_python_matrix_script_has_valid_bash_syntax_and_final_summary():
    result = subprocess.run(
        ["bash", "-n", str(MATRIX_SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    script = MATRIX_SCRIPT.read_text(encoding="utf-8")
    assert "PYTHON_TEST_VERSIONS" in script
    assert "=== Python version matrix summary ===" in script
    assert "Succeeded versions:" in script
    assert "Failed versions:" in script
    assert "Result: Python version matrix failed." in script
    assert "Result: all Python version runs succeeded." in script
    # macOS Bash 3.2 with set -u rejects an unguarded empty-array expansion.
    assert '${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}' in script


@pytest.mark.skipif(os.name == "nt", reason="Fake uv shell fixture requires POSIX paths")
@pytest.mark.parametrize(
    ("workers", "distribution"),
    [("logical", "worksteal"), ("2", "load")],
)
def test_matrix_enables_parallel_pytest_with_configurable_workers(
    tmp_path, workers, distribution
):
    """Run the matrix with a fake uv and interpreter to inspect pytest arguments."""
    uv = tmp_path / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "venv" ]]; then
  mkdir -p "$4/bin"
  cat > "$4/bin/python" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$PYTEST_DISABLE_PLUGIN_AUTOLOAD" "$PWD" "$@" > "$TEST_CAPTURE"
EOF
  chmod +x "$4/bin/python"
fi
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    capture = tmp_path / "pytest-args"
    result = subprocess.run(
        ["bash", str(MATRIX_SCRIPT), "3.12"],
        cwd=ROOT,
        env={
            **os.environ,
            "UV": str(uv),
            "VENV_ROOT": str(tmp_path / "venvs"),
            "TEST_CAPTURE": str(capture),
            "TEST_WORKERS": workers,
            "TEST_DIST": distribution,
            # A conflicting caller environment must not auto-register xdist twice.
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
            "PYTEST_ARGS_OVERRIDE": "-q",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "1", str(ROOT), "-m", "pytest", "-p", "xdist.plugin", "-n", workers,
        f"--dist={distribution}", "-q",
    ]
