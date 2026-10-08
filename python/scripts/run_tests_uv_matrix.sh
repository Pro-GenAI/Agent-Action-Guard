#!/usr/bin/env bash
set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${VENV_ROOT:-$PYTHON_DIR/.uv-venvs}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
UV="${UV:-uv}"
export UV_VENV_CLEAR="1"

DEFAULT_VERSIONS=(3.10 3.11 3.12 3.13 3.14)
if [[ "$#" -gt 0 ]]; then
  PYTHON_VERSIONS=("$@")
elif [[ -n "${PYTHON_TEST_VERSIONS:-}" ]]; then
  read -r -a PYTHON_VERSIONS <<< "${PYTHON_TEST_VERSIONS}"
else
  PYTHON_VERSIONS=("${DEFAULT_VERSIONS[@]}")
fi

PYTEST_ARGS=()
if [[ -n "${PYTEST_ARGS_OVERRIDE:-}" ]]; then
  read -r -a PYTEST_ARGS <<< "${PYTEST_ARGS_OVERRIDE}"
fi

mkdir -p "$VENV_ROOT"

succeeded=()
failed=()

print_summary() {
  echo
  echo "=== Python version matrix summary ==="
  echo "Succeeded: ${#succeeded[@]}"
  echo "Failed: ${#failed[@]}"
  if [[ "${#succeeded[@]}" -gt 0 ]]; then
    printf 'Succeeded versions: %s\n' "$(IFS=', '; echo "${succeeded[*]}")"
  else
    echo "Succeeded versions: none"
  fi
  if [[ "${#failed[@]}" -gt 0 ]]; then
    printf 'Failed versions: %s\n' "$(IFS=', '; echo "${failed[*]}")"
  else
    echo "Failed versions: none"
  fi
}

record_failure() {
  failed+=("$1")
  if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
    print_summary
    echo "Result: Python version matrix failed."
    exit 1
  fi
}

for version in "${PYTHON_VERSIONS[@]}"; do
  echo "=== Python ${version} ==="

  if ! "$UV" python install "$version"; then
    echo "Python install failed for ${version}."
    record_failure "$version"
    continue
  fi

  venv_dir="$VENV_ROOT/py-${version}"
  if ! "$UV" venv -p "$version" "$venv_dir"; then
    echo "Virtual environment creation failed for ${version}."
    record_failure "$version"
    continue
  fi
  venv_python="$venv_dir/bin/python"

  if ! "$UV" pip install -p "$venv_python" -U pip setuptools wheel; then
    echo "Dependency bootstrap failed for ${version}."
    record_failure "$version"
    continue
  fi

  if ! "$UV" pip install -p "$venv_python" -e "$PYTHON_DIR"; then
    echo "Project install failed for ${version}."
    record_failure "$version"
    continue
  fi

  if ! "$UV" pip install -p "$venv_python" pytest "rich>=12,<15"; then
    echo "Test dependency install failed for ${version}."
    record_failure "$version"
    continue
  fi

  # Run from the Python package root so pytest uses `python/pytest.ini` and
  # doesn't accidentally collect repo-level example/unused scripts.
  pushd "$PYTHON_DIR" >/dev/null
  if ! PATH="$venv_dir/bin:$PATH" "$venv_python" -m pytest ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}; then
    popd >/dev/null
    echo "Tests failed for ${version}."
    record_failure "$version"
    continue
  fi
  popd >/dev/null

  succeeded+=("$version")
done

print_summary
if [[ "${#failed[@]}" -ne 0 ]]; then
  echo "Result: Python version matrix failed."
  exit 1
fi

echo "Result: all Python version runs succeeded."
