import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
MATRIX_SCRIPT = ROOT / "scripts" / "run_tests_uv_matrix.sh"


def test_makefile_exposes_selectable_python_test_matrix():
    assert "PYTHON_TEST_VERSIONS ?=" in MAKEFILE
    assert "test-matrix:" in MAKEFILE
    assert 'PYTHON_TEST_VERSIONS="$(PYTHON_TEST_VERSIONS)"' in MAKEFILE
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
