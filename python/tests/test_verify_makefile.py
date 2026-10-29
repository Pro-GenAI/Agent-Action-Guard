import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_STEPS = ["install-dev", "test", "test-matrix", "security"]
ROOT_STEPS = ["-C python verify", "-C typescript verify"]


@pytest.mark.parametrize(
    ("directory", "steps"),
    [
        (ROOT / "python", PACKAGE_STEPS),
        (ROOT / "typescript", PACKAGE_STEPS),
        (ROOT, ROOT_STEPS),
    ],
    ids=["python", "typescript", "root"],
)
@pytest.mark.parametrize("failure_index", [None, 0, 1, 2, 3])
def test_verify_runs_in_order_and_stops_on_failure(
    tmp_path, directory, steps, failure_index
):
    if failure_index is not None and failure_index >= len(steps):
        pytest.skip("This workflow has fewer steps")

    log = tmp_path / "calls"
    fake_make = tmp_path / "fake-make"
    fake_make.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$VERIFY_LOG"\n'
        'if [ "$*" = "$VERIFY_FAIL" ]; then exit 7; fi\n',
        encoding="utf-8",
    )
    fake_make.chmod(0o755)
    env = {
        **os.environ,
        "VERIFY_LOG": str(log),
        "VERIFY_FAIL": steps[failure_index] if failure_index is not None else "",
    }
    result = subprocess.run(
        ["make", "-j4", "verify", f"MAKE={fake_make}"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = steps if failure_index is None else steps[: failure_index + 1]
    assert log.read_text(encoding="utf-8").splitlines() == expected
    assert (result.returncode == 0) == (failure_index is None), result.stderr
