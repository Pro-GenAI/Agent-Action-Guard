#!/usr/bin/env python3
"""Test Action Guard across dependency version ranges.

This script creates per-scenario virtual environments, pins dependency versions,
installs the local package, and runs a small smoke import check.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover
    Version = None


PYTHON_VERSION = "3.10"


def _run(cmd, cwd=None, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def _uv_available():
    return shutil.which("uv") is not None


def _parse_versions(pip_output):
    match = re.search(r"Available versions:\s*(.*)", pip_output)
    if not match:
        return []
    raw = match.group(1)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _filter_versions(versions, min_v, max_v):
    if Version is None:
        return versions

    min_ver = Version(min_v)
    max_ver = Version(max_v)
    filtered = []
    for v in versions:
        try:
            parsed = Version(v)
        except Exception:
            continue
        if min_ver <= parsed < max_ver:
            filtered.append(v)
    return filtered


def _sample_versions(versions, count):
    if not versions:
        return []
    if len(versions) <= count:
        return versions

    step = max(1, len(versions) // (count - 1))
    sampled = [versions[0]]

    idx = step
    while len(sampled) < count - 1 and idx < len(versions) - 1:
        sampled.append(versions[idx])
        idx += step

    sampled.append(versions[-1])
    return sampled


def _resolve_versions(python, package, min_v, max_v, samples):
    try:
        output = _run([python, "-m", "pip", "index", "versions", package])
    except Exception:
        return [min_v]

    versions = _parse_versions(output)
    versions = list(reversed(versions))
    versions = _filter_versions(versions, min_v, max_v)
    return _sample_versions(versions, samples) or [min_v]


def _build_scenarios(core_versions, optional_versions, include_optional):
    scenarios = []

    base_min = {name: versions[0] for name, versions in core_versions.items()}
    if include_optional:
        base_min.update(
            {name: versions[0] for name, versions in optional_versions.items()}
        )
    scenarios.append(("base-min", base_min))

    base_max = {name: versions[-1] for name, versions in core_versions.items()}
    if include_optional:
        base_max.update(
            {name: versions[-1] for name, versions in optional_versions.items()}
        )
    scenarios.append(("base-max", base_max))

    for dep_name, versions in core_versions.items():
        for v in versions:
            scenario = base_min.copy()
            scenario[dep_name] = v
            scenarios.append((f"core-{dep_name}-{v}", scenario))

    if include_optional:
        for dep_name, versions in optional_versions.items():
            for v in versions:
                scenario = base_min.copy()
                scenario[dep_name] = v
                scenarios.append((f"opt-{dep_name}-{v}", scenario))

    return scenarios


def _create_venv(venv_path):
    if venv_path.exists():
        shutil.rmtree(venv_path)

    if not _uv_available():
        raise RuntimeError(f"uv is required to ensure Python {PYTHON_VERSION} exactly")

    print(f"Creating venv with uv at {venv_path}")
    _run(["uv", "venv", "--python", PYTHON_VERSION, str(venv_path)])


def _venv_python(venv_path):
    return str(venv_path / "bin" / "python")


def _assert_python_version(python):
    version = _run(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    ).strip()

    if version != PYTHON_VERSION:
        raise RuntimeError(f"Expected Python {PYTHON_VERSION}, got {version}")


def _install_dependencies(python, scenario):
    deps = [f"{name}=={version}" for name, version in scenario.items()]

    print("Installing dependencies:")
    for dep in deps:
        print(f"  - {dep}")

    print("Using uv to install dependencies")
    _run(["uv", "pip", "install", "-p", python, "-U", "pip", "setuptools", "wheel"])
    _run(["uv", "pip", "install", "-p", python, *deps])


def _install_package(python, package_root):
    print("Installing local package (editable) with uv")
    _run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            python,
            "-e",
            ".",
            "--no-deps",
        ],
        cwd=package_root,
    )


def _run_smoke_test(python, include_optional):
    env = os.environ.copy()
    env["EMBED_MODEL_NAME"] = "test-embedding"

    print("Running smoke test")

    snippet_lines = [
        "import agent_action_guard.action_classifier as ac",
        "assert ac.classifier.session is not None",
    ]

    if include_optional:
        snippet_lines += [
            "import dotenv",
            "import requests",
            "import rich",
        ]

    snippet = ";".join(snippet_lines)
    _run([python, "-c", snippet], env=env)


def main():
    parser = argparse.ArgumentParser(description="Test dependency version ranges")
    parser.add_argument(
        "--venv-root",
        default=str(Path.cwd() / ".venv-deps"),
        help="Directory to store created virtual environments",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Versions to sample per dependency",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Skip optional dependency testing",
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    venv_root = Path(args.venv_root)
    venv_root.mkdir(parents=True, exist_ok=True)

    ranges = {
        "numpy": ("1.21.6", "3.0.0"),
        "onnxruntime": ("1.14.0", "2.0.0"),
        "openai": ("1.0.0", "2.0.0"),
    }

    optional_ranges = {
        "python-dotenv": ("0.21.0", "2.0.0"),
        "requests": ("2.28.0", "3.0.0"),
        "rich": ("12.0.0", "15.0.0"),
    }

    resolver_venv_path = venv_root / "_resolver"
    print(f"Setting up resolver venv: {resolver_venv_path}")
    _create_venv(resolver_venv_path)

    resolver_python = _venv_python(resolver_venv_path)
    _assert_python_version(resolver_python)

    print(f"Using resolver python: {resolver_python}")
    _run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            resolver_python,
            "-U",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
        ]
    )

    core_versions = {
        name: _resolve_versions(resolver_python, name, min_v, max_v, args.samples)
        for name, (min_v, max_v) in ranges.items()
    }

    optional_versions = {
        name: _resolve_versions(resolver_python, name, min_v, max_v, args.samples)
        for name, (min_v, max_v) in optional_ranges.items()
    }

    scenarios = _build_scenarios(
        core_versions,
        optional_versions,
        not args.no_optional,
    )

    succeeded = []
    failed_or_skipped = []

    for scenario_name, scenario_deps in scenarios:
        venv_path = venv_root / scenario_name

        print(f"\n   ========= Scenario: {scenario_name} =========")

        try:
            print(f"Setting up venv: {venv_path}")
            _create_venv(venv_path)

            venv_python = _venv_python(venv_path)
            _assert_python_version(venv_python)

            print(f"Using venv python: {venv_python}")

            _install_dependencies(venv_python, scenario_deps)
            _install_package(venv_python, package_root)
            _run_smoke_test(venv_python, include_optional=not args.no_optional)

            print(f"Scenario succeeded: {scenario_name}")
            succeeded.append(scenario_name)

        except subprocess.CalledProcessError as exc:
            print("Scenario failed; skipping to the next scenario.")
            failed_or_skipped.append(scenario_name)
            if exc.stdout:
                print(exc.stdout)
            continue

        except Exception as exc:
            print("Scenario failed or was skipped; continuing to the next scenario.")
            failed_or_skipped.append(scenario_name)
            print(exc)
            continue

    print("\n   ========= Summary =========")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed or skipped: {len(failed_or_skipped)}")

    if failed_or_skipped:
        print("❌ Failed or skipped scenarios:")
        for scenario_name in failed_or_skipped:
            print(f"  - {scenario_name}")
        print("Result: ❌ Some runs failed or were skipped.")
    else:
        print("Result: ✅ All version runs succeeded.")


if __name__ == "__main__":
    main()
