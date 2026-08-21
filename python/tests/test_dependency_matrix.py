import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "test_dependency_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("dependency_matrix_script", SCRIPT_PATH)
dependency_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dependency_matrix)


def test_default_python_versions_cover_all_declared_supported_minors():
    assert dependency_matrix.DEFAULT_PYTHON_VERSIONS == (
        "3.8",
        "3.9",
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    )


def test_normalize_python_versions_supports_space_comma_and_deduplication():
    assert dependency_matrix._normalize_python_versions(
        ["3.8,3.10", "3.12", "3.10"]
    ) == ["3.8", "3.10", "3.12"]


def test_normalize_python_versions_rejects_patch_versions():
    with pytest.raises(ValueError, match="MAJOR.MINOR"):
        dependency_matrix._normalize_python_versions(["3.10.1"])


@pytest.mark.parametrize(
    ("python_version", "expected_tokenizers_min"),
    [
        ("3.8", "0.13.3"),
        ("3.12", "0.13.3"),
        ("3.13", "0.21.0"),
        ("3.14", "0.21.0"),
    ],
)
def test_dependency_ranges_follow_python_compatibility(
    python_version, expected_tokenizers_min
):
    ranges = dependency_matrix._dependency_ranges_for_python(python_version)

    assert ranges["tokenizers"] == (expected_tokenizers_min, "1.0.0")
    assert ranges["onnxruntime"] == ("1.14.0", "2.0.0")
    assert ranges["openai"] == ("1.0.0", "4.0.0")


def test_bounds_only_matrix_uses_distinct_oldest_and_newest_profiles():
    core_versions = {
        "numpy": ["1.21.6", "1.26.4"],
        "onnxruntime": ["1.14.0", "1.22.0"],
    }

    scenarios = dependency_matrix._build_scenarios(
        core_versions, {}, include_optional=False, bounds_only=True
    )

    assert scenarios == [
        ("base-min", {"numpy": "1.21.6", "onnxruntime": "1.14.0"}),
        ("base-max", {"numpy": "1.26.4", "onnxruntime": "1.22.0"}),
    ]


def test_full_matrix_varies_each_dependency_from_the_minimum_profile():
    core_versions = {
        "numpy": ["1.21.6", "1.24.4", "1.26.4"],
        "onnxruntime": ["1.14.0", "1.17.3"],
    }

    scenarios = dependency_matrix._build_scenarios(
        core_versions, {}, include_optional=False
    )
    scenario_map = dict(scenarios)

    assert scenario_map["core-numpy-1.24.4"] == {
        "numpy": "1.24.4",
        "onnxruntime": "1.14.0",
    }
    assert scenario_map["core-onnxruntime-1.17.3"] == {
        "numpy": "1.21.6",
        "onnxruntime": "1.17.3",
    }


def test_version_installability_probe_uses_uv_dry_run_and_binary_wheels(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dependency_matrix,
        "_run",
        lambda command, **_kwargs: calls.append(command) or "",
    )

    assert dependency_matrix._version_is_installable(
        "/venv/bin/python", "numpy", "1.26.4"
    )
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "-p",
            "/venv/bin/python",
            "--dry-run",
            "--only-binary",
            ":all:",
            "numpy==1.26.4",
        ]
    ]


def test_resolve_versions_skips_advertised_versions_that_uv_cannot_install(monkeypatch):
    pip_output = "Available versions: 1.3.0, 1.2.0, 1.1.0, 1.0.0"
    monkeypatch.setattr(dependency_matrix, "_run", lambda *_args, **_kwargs: pip_output)
    monkeypatch.setattr(
        dependency_matrix,
        "_version_is_installable",
        lambda _python, _package, version: version in {"1.1.0", "1.2.0"},
    )

    assert dependency_matrix._resolve_versions(
        "/venv/bin/python", "example", "1.0.0", "2.0.0", 2
    ) == ["1.1.0", "1.2.0"]


def test_openai_3_3_1_is_selected_as_the_newest_supported_release(monkeypatch):
    pip_output = "Available versions: 3.3.1, 3.3.0, 2.54.0, 1.0.0"
    monkeypatch.setattr(dependency_matrix, "_run", lambda *_args, **_kwargs: pip_output)
    monkeypatch.setattr(
        dependency_matrix,
        "_version_is_installable",
        lambda _python, _package, _version: True,
    )

    assert dependency_matrix._resolve_versions(
        "/venv/bin/python", "openai", "1.0.0", "4.0.0", 2
    ) == ["1.0.0", "3.3.1"]


def test_create_venv_requests_the_selected_python_from_uv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dependency_matrix, "_uv_available", lambda: True)
    monkeypatch.setattr(
        dependency_matrix,
        "_run",
        lambda command, **_kwargs: calls.append(command) or "",
    )

    venv_path = tmp_path / "py312"
    dependency_matrix._create_venv(venv_path, "3.12")

    assert calls == [["uv", "venv", "--python", "3.12", str(venv_path)]]


def test_ensure_python_uses_uv_managed_python_install(monkeypatch):
    calls = []
    monkeypatch.setattr(dependency_matrix, "_uv_available", lambda: True)
    monkeypatch.setattr(
        dependency_matrix,
        "_run",
        lambda command, **_kwargs: calls.append(command) or "",
    )

    dependency_matrix._ensure_python("3.8")

    assert calls == [["uv", "python", "install", "3.8"]]


def test_assert_python_version_compares_against_each_matrix_version(monkeypatch):
    monkeypatch.setattr(dependency_matrix, "_run", lambda *_args, **_kwargs: "3.13\n")

    dependency_matrix._assert_python_version("/venv/bin/python", "3.13")

    with pytest.raises(RuntimeError, match="Expected Python 3.12, got 3.13"):
        dependency_matrix._assert_python_version("/venv/bin/python", "3.12")


def test_base_profiles_resolve_all_dependencies_with_joint_ranges():
    scenario_deps = {
        "numpy": "1.23.2",
        "onnxruntime": "1.15.0",
        "openai": "1.0.0",
        "python-dotenv": "0.21.0",
    }
    core_ranges = {
        "numpy": ("1.21.6", "3.0.0"),
        "onnxruntime": ("1.14.0", "2.0.0"),
        "openai": ("1.0.0", "4.0.0"),
    }
    optional_ranges = {"python-dotenv": ("0.21.0", "2.0.0")}

    requirements, resolution = dependency_matrix._requirements_for_scenario(
        "base-min", scenario_deps, core_ranges, optional_ranges
    )

    assert requirements == [
        "numpy>=1.21.6,<3.0.0",
        "onnxruntime>=1.14.0,<2.0.0",
        "openai>=1.0.0,<4.0.0",
        "python-dotenv>=0.21.0,<2.0.0",
    ]
    assert resolution == "lowest-direct"

    _, resolution = dependency_matrix._requirements_for_scenario(
        "base-max", scenario_deps, core_ranges, optional_ranges
    )
    assert resolution == "highest"


def test_dependency_scenario_pins_only_target_and_resolves_compatible_companions():
    scenario_deps = {
        "numpy": "1.23.2",
        "onnxruntime": "1.15.0",
        "openai": "1.0.0",
    }
    core_ranges = {
        "numpy": ("1.21.6", "3.0.0"),
        "onnxruntime": ("1.14.0", "2.0.0"),
        "openai": ("1.0.0", "4.0.0"),
    }

    requirements, resolution = dependency_matrix._requirements_for_scenario(
        "core-onnxruntime-1.15.0", scenario_deps, core_ranges, {}
    )

    assert requirements == [
        "numpy>=1.21.6,<3.0.0",
        "onnxruntime==1.15.0",
        "openai>=1.0.0,<4.0.0",
    ]
    assert resolution == "lowest-direct"


def test_numpy_2_scenario_requires_numpy_2_compatible_onnxruntime():
    scenario_deps = {
        "numpy": "2.0.2",
        "onnxruntime": "1.17.0",
        "openai": "1.0.0",
    }
    core_ranges = {
        "numpy": ("1.21.6", "3.0.0"),
        "onnxruntime": ("1.14.0", "2.0.0"),
        "openai": ("1.0.0", "4.0.0"),
    }

    requirements, resolution = dependency_matrix._requirements_for_scenario(
        "core-numpy-2.0.2", scenario_deps, core_ranges, {}
    )

    assert requirements == [
        "numpy==2.0.2",
        "onnxruntime>=1.19.0,<2.0.0",
        "openai>=1.0.0,<4.0.0",
    ]
    assert resolution == "lowest-direct"


def test_optional_dependency_name_with_hyphen_is_detected_as_target():
    scenario_deps = {
        "numpy": "1.21.6",
        "python-dotenv": "1.2.3",
    }

    requirements, _ = dependency_matrix._requirements_for_scenario(
        "opt-python-dotenv-1.2.3",
        scenario_deps,
        {"numpy": ("1.21.6", "3.0.0")},
        {"python-dotenv": ("0.21.0", "2.0.0")},
    )

    assert requirements == [
        "numpy>=1.21.6,<3.0.0",
        "python-dotenv==1.2.3",
    ]


def test_install_dependencies_passes_resolution_strategy_to_uv(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dependency_matrix,
        "_run",
        lambda command, **_kwargs: calls.append(command) or "",
    )

    dependency_matrix._install_dependencies(
        "/venv/bin/python",
        ["numpy>=1.21.6,<3.0.0", "onnxruntime==1.15.0"],
        "lowest-direct",
    )

    assert calls[-1] == [
        "uv",
        "pip",
        "install",
        "-p",
        "/venv/bin/python",
        "--only-binary",
        ":all:",
        "--resolution",
        "lowest-direct",
        "numpy>=1.21.6,<3.0.0",
        "onnxruntime==1.15.0",
    ]
