"""Test Action Guard across Python and dependency version ranges.

This script uses uv to provision Python interpreters, creates per-scenario virtual
environments, pins the dependency under test, resolves the remaining dependencies
within their supported ranges, installs the local package, and runs a small smoke
import check.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover
    InvalidVersion = ValueError
    Version = None


DEFAULT_PYTHON_VERSIONS = (
    "3.8",
    "3.9",
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    "3.14",
)

CORE_RANGES = {
    "numpy": ("1.21.6", "3.0.0"),
    "onnxruntime": ("1.14.0", "2.0.0"),
    "openai": ("1.0.0", "4.0.0"),
}

OPTIONAL_RANGES = {
    "python-dotenv": ("0.21.0", "2.0.0"),
    "requests": ("2.28.0", "3.0.0"),
    "rich": ("12.0.0", "15.0.0"),
}

NUMPY_2_ONNXRUNTIME_MIN = "1.19.0"


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


def _normalize_python_versions(values):
    versions = []
    for value in values or DEFAULT_PYTHON_VERSIONS:
        for version in re.split(r"[,\s]+", value.strip()):
            if not version:
                continue
            if not re.fullmatch(r"\d+\.\d+", version):
                raise ValueError(
                    f"Invalid Python version {version!r}; expected MAJOR.MINOR"
                )
            if version not in versions:
                versions.append(version)
    return versions


def _python_version_tuple(version):
    major, minor = version.split(".", maxsplit=1)
    return int(major), int(minor)


def _dependency_ranges_for_python(python_version):
    ranges = dict(CORE_RANGES)
    tokenizers_min = (
        "0.21.0" if _python_version_tuple(python_version) >= (3, 13) else "0.13.3"
    )
    ranges["tokenizers"] = (tokenizers_min, "1.0.0")
    return ranges


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
        except InvalidVersion:
            continue
        if min_ver <= parsed < max_ver:
            filtered.append(v)
    return filtered


def _sample_versions(versions, count):
    if not versions:
        return []
    if count <= 1:
        return [versions[0]]
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


def _version_is_installable(python, package, version):
    try:
        _run(
            [
                "uv",
                "pip",
                "install",
                "-p",
                python,
                "--dry-run",
                "--only-binary",
                ":all:",
                f"{package}=={version}",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _resolve_versions(python, package, min_v, max_v, samples):
    try:
        output = _run([python, "-m", "pip", "index", "versions", package])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to query versions for {package}") from exc

    versions = _filter_versions(list(reversed(_parse_versions(output))), min_v, max_v)
    if not versions:
        raise RuntimeError(
            f"No published {package} versions found in range >={min_v},<{max_v}"
        )

    probe_cache = {}

    def installable(index):
        version = versions[index]
        if version not in probe_cache:
            probe_cache[version] = _version_is_installable(python, package, version)
        return probe_cache[version]

    target_count = min(samples, len(versions))
    if target_count == 1:
        target_indices = [0]
    else:
        target_indices = [
            round(index * (len(versions) - 1) / (target_count - 1))
            for index in range(target_count)
        ]

    selected = []
    for target_index in target_indices:
        for distance in range(len(versions)):
            candidate_indices = [target_index - distance]
            if distance:
                candidate_indices.append(target_index + distance)

            found = None
            for candidate_index in candidate_indices:
                if not 0 <= candidate_index < len(versions):
                    continue
                if installable(candidate_index):
                    found = versions[candidate_index]
                    break

            if found is not None:
                if found not in selected:
                    selected.append(found)
                break

    if len(selected) < target_count:
        for index, version in enumerate(versions):
            if version in selected or not installable(index):
                continue
            selected.append(version)
            if len(selected) >= target_count:
                break

    if not selected:
        raise RuntimeError(
            f"No installable {package} versions found for {python} in range "
            f">={min_v},<{max_v}"
        )

    if Version is not None:
        selected.sort(key=Version)

    return _sample_versions(selected, samples)


def _build_scenarios(
    core_versions, optional_versions, include_optional, bounds_only=False
):
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

    if bounds_only:
        return scenarios

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


def _scenario_target_dependency(scenario_name, dependency_names):
    for dependency_name in sorted(dependency_names, key=len, reverse=True):
        for prefix in ("core-", "opt-"):
            if scenario_name.startswith(f"{prefix}{dependency_name}-"):
                return dependency_name
    return None


def _requirements_for_scenario(
    scenario_name,
    scenario_deps,
    core_ranges,
    optional_ranges,
):
    supported_ranges = dict(core_ranges)
    supported_ranges.update(optional_ranges)
    target_dependency = _scenario_target_dependency(scenario_name, scenario_deps.keys())
    target_version = scenario_deps.get(target_dependency)
    numpy_target_is_v2 = (
        target_dependency == "numpy"
        and target_version is not None
        and int(target_version.split(".", 1)[0]) >= 2
    )

    requirements = []
    for dependency_name, selected_version in scenario_deps.items():
        if dependency_name == target_dependency:
            requirements.append(f"{dependency_name}=={selected_version}")
            continue

        min_version, max_version = supported_ranges[dependency_name]
        if (
            dependency_name == "onnxruntime"
            and numpy_target_is_v2
            and (
                Version is None
                or Version(min_version) < Version(NUMPY_2_ONNXRUNTIME_MIN)
            )
        ):
            min_version = NUMPY_2_ONNXRUNTIME_MIN
        requirements.append(f"{dependency_name}>={min_version},<{max_version}")

    resolution = "highest" if scenario_name == "base-max" else "lowest-direct"
    return requirements, resolution


def _ensure_python(python_version):
    if not _uv_available():
        raise RuntimeError("uv is required to run the dependency matrix")

    print(f"Ensuring Python {python_version} is available with uv")
    _run(["uv", "python", "install", python_version])


def _create_venv(venv_path, python_version):
    if venv_path.exists():
        shutil.rmtree(venv_path)

    if not _uv_available():
        raise RuntimeError("uv is required to run the dependency matrix")

    print(f"Creating Python {python_version} venv with uv at {venv_path}")
    _run(["uv", "venv", "--python", python_version, str(venv_path)])


def _venv_python(venv_path):
    if os.name == "nt":
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def _assert_python_version(python, expected_version):
    version = _run(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    ).strip()

    if version != expected_version:
        raise RuntimeError(f"Expected Python {expected_version}, got {version}")


def _install_dependencies(python, requirements, resolution):
    print("Installing dependencies:")
    for requirement in requirements:
        print(f"  - {requirement}")
    print(f"Resolution strategy: {resolution}")

    print("Using uv to install dependencies")
    _run(["uv", "pip", "install", "-p", python, "-U", "pip", "setuptools", "wheel"])
    _run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            python,
            "--only-binary",
            ":all:",
            "--resolution",
            resolution,
            *requirements,
        ]
    )


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
        "import openai",
        "import tokenizers",
        "assert ac.classifier.session is not None",
        "openai_major = int(openai.__version__.split('.', 1)[0])",
        "client = openai.OpenAI(api_key='test-key') if openai_major >= 3 else None",
        "assert openai_major < 3 or client.embeddings is not None",
    ]

    if include_optional:
        snippet_lines += [
            "import dotenv",
            "import requests",
            "import rich",
        ]

    snippet = ";".join(snippet_lines)
    _run([python, "-c", snippet], env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test dependency versions across multiple Python versions"
    )
    parser.add_argument(
        "--venv-root",
        default=str(Path.cwd() / ".venv-deps"),
        help="Directory to store created virtual environments",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Versions to sample per dependency",
    )
    parser.add_argument(
        "--python-versions",
        nargs="+",
        default=list(DEFAULT_PYTHON_VERSIONS),
        metavar="VERSION",
        help=(
            "Python MAJOR.MINOR versions to test. Values may also be comma-separated. "
            f"Default: {', '.join(DEFAULT_PYTHON_VERSIONS)}"
        ),
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Skip optional dependency testing",
    )
    parser.add_argument(
        "--bounds-only",
        action="store_true",
        help="Run only the oldest and newest compatible dependency profiles",
    )
    args = parser.parse_args(argv)

    if args.samples < 2:
        parser.error("--samples must be at least 2 to exercise dependency diversity")

    try:
        python_versions = _normalize_python_versions(args.python_versions)
    except ValueError as exc:
        parser.error(str(exc))

    if not _uv_available():
        parser.error("uv is required; install the dev extras or install uv separately")

    package_root = Path(__file__).resolve().parents[1]
    venv_root = Path(args.venv_root)
    venv_root.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed_or_skipped = []

    for python_version in python_versions:
        python_root = venv_root / f"py-{python_version}"
        resolver_venv_path = python_root / "_resolver"

        print(f"\n######## Python {python_version} ########")
        _ensure_python(python_version)
        print(f"Setting up resolver venv: {resolver_venv_path}")
        _create_venv(resolver_venv_path, python_version)

        resolver_python = _venv_python(resolver_venv_path)
        _assert_python_version(resolver_python, python_version)

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

        ranges = _dependency_ranges_for_python(python_version)
        core_versions = {
            name: _resolve_versions(resolver_python, name, min_v, max_v, args.samples)
            for name, (min_v, max_v) in ranges.items()
        }
        optional_versions = {
            name: _resolve_versions(resolver_python, name, min_v, max_v, args.samples)
            for name, (min_v, max_v) in OPTIONAL_RANGES.items()
        }

        scenarios = _build_scenarios(
            core_versions,
            optional_versions,
            not args.no_optional,
            bounds_only=args.bounds_only,
        )

        for scenario_name, scenario_deps in scenarios:
            scenario_label = f"py{python_version}:{scenario_name}"
            venv_path = python_root / scenario_name
            scenario_optional_ranges = OPTIONAL_RANGES if not args.no_optional else {}
            requirements, resolution = _requirements_for_scenario(
                scenario_name,
                scenario_deps,
                ranges,
                scenario_optional_ranges,
            )

            print(f"\n   ========= Scenario: {scenario_label} =========")

            try:
                print(f"Setting up venv: {venv_path}")
                _create_venv(venv_path, python_version)

                venv_python = _venv_python(venv_path)
                _assert_python_version(venv_python, python_version)

                print(f"Using venv python: {venv_python}")

                _install_dependencies(venv_python, requirements, resolution)
                _install_package(venv_python, package_root)
                _run_smoke_test(venv_python, include_optional=not args.no_optional)

                print(f"Scenario succeeded: {scenario_label}")
                succeeded.append(scenario_label)

            except subprocess.CalledProcessError as exc:
                print("Scenario failed; continuing to the next scenario.")
                failed_or_skipped.append(scenario_label)
                if exc.stdout:
                    print(exc.stdout)
                continue

            except (OSError, RuntimeError, ValueError) as exc:
                print(
                    "Scenario failed or was skipped; continuing to the next scenario."
                )
                failed_or_skipped.append(scenario_label)
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
        return 1

    print("Result: ✅ All version runs succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
