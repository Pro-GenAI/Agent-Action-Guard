"""Regression checks for wheel-friendly developer dependencies."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_cryptography_dev_dependency_stays_below_50():
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_section = config.split("\ndev =", 1)[1]
    dev_section = dev_section.split("\n]", 1)[0]
    assert '"cryptography<49"' in dev_section


def test_locked_cryptography_respects_dev_constraint():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    versions = re.findall(r'\[\[package\]\]\nname = "cryptography"\nversion = "([^"]+)"', lock)
    assert len(versions) == 1
    assert int(versions[0].split(".")[0]) < 49
