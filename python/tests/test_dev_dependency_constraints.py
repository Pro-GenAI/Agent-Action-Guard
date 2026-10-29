"""Regression checks for wheel-friendly developer dependencies."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cryptography_uses_patched_release_except_the_intel_mac_wheel_fallback():
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_section = config.split("\ndev =", 1)[1]
    dev_section = dev_section.split("\n]", 1)[0]
    assert (
        "\"cryptography>=50.0.2; sys_platform != 'darwin' or platform_machine != 'x86_64'\""
        in dev_section
    )
    assert (
        "\"cryptography>=48.0.1,<49; sys_platform == 'darwin' and platform_machine == 'x86_64'\""
        in dev_section
    )


def test_locked_cryptography_respects_dev_constraint():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    versions = re.findall(
        r'\[\[package\]\]\nname = "cryptography"\nversion = "([^"]+)"', lock
    )
    assert any(int(version.split(".")[0]) >= 50 for version in versions)
    assert any(version.startswith("48.") for version in versions)
