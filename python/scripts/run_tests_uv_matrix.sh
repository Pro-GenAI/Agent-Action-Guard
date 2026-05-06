#!/usr/bin/env bash
set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${VENV_ROOT:-$PYTHON_DIR/.uv-venvs}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
export UV_VENV_CLEAR="1"

DEFAULT_VERSIONS=(3.8 3.9 3.10 3.11 3.12 3.13 3.14)
if [[ "$#" -gt 0 ]]; then
  PYTHON_VERSIONS=("$@")
else
  PYTHON_VERSIONS=("${DEFAULT_VERSIONS[@]}")
fi

PYTEST_ARGS=()
if [[ -n "${PYTEST_ARGS_OVERRIDE:-}" ]]; then
  read -r -a PYTEST_ARGS <<< "${PYTEST_ARGS_OVERRIDE}"
fi

mkdir -p "$VENV_ROOT"

failures=0
for version in "${PYTHON_VERSIONS[@]}"; do
  echo "=== Python ${version} ==="

  if ! uv python install "$version"; then
    echo "Skipping ${version}: unable to install via uv."
    failures=1
    if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
      continue
    fi
    exit 1
  fi

  venv_dir="$VENV_ROOT/py-${version}"
  uv venv -p "$version" "$venv_dir"
  venv_python="$venv_dir/bin/python"

  if ! uv pip install -p "$venv_python" -U pip setuptools wheel; then
    echo "Dependency bootstrap failed for ${version}."
    failures=1
    if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
      continue
    fi
    exit 1
  fi

  if ! uv pip install -p "$venv_python" -e "$PYTHON_DIR"; then
    echo "Project install failed for ${version}."
    failures=1
    if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
      continue
    fi
    exit 1
  fi

  if ! uv pip install -p "$venv_python" pytest; then
    echo "Pytest install failed for ${version}."
    failures=1
    if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
      continue
    fi
    exit 1
  fi

  # Run from the Python package root so pytest uses `python/pytest.ini` and
  # doesn't accidentally collect repo-level example/unused scripts.
  pushd "$PYTHON_DIR" >/dev/null
  if ! EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-test-embedding-model}" \
    "$venv_python" -m pytest "${PYTEST_ARGS[@]}"; then
    popd >/dev/null
    echo "Tests failed for ${version}."
    failures=1
    if [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
      continue
    fi
    exit 1
  fi
  popd >/dev/null

done

if [[ "$failures" -ne 0 ]]; then
  echo "Result: ❌ Some runs failed or were skipped."
  exit 1
fi

printf "Result: ✅ All version runs succeeded."
