UV ?= uv
SECURITY_ARGS ?=
TEST_WORKERS ?= logical
TEST_DIST ?= worksteal
PYTHON_TEST_VERSIONS ?=
SOURCE_DIRS := agent_action_guard examples scripts tests training
# Recursively expanded so it is only evaluated by `lint`; avoids Windows' find.exe.
PY_FILES = $(shell $(UV) run --no-project python -c "import pathlib,sys; print(' '.join(sorted(p.as_posix() for d in sys.argv[1:] for p in pathlib.Path(d).rglob('*.py'))))" $(SOURCE_DIRS)) sitecustomize.py

.PHONY: verify install install-dev format lint quality test test-matrix security dependency-matrix dependency-matrix-bounds latency-backends

install:
	$(UV) sync

install-dev:
	$(UV) sync --extra dev

verify:
	$(MAKE) install-dev
	$(MAKE) test
	$(MAKE) test-matrix
	$(MAKE) security

format:
	$(UV) run --with isort isort $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with black black $(SOURCE_DIRS) sitecustomize.py

lint:
	$(UV) run --with ruff ruff check $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with black black --check $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with isort isort --check-only $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --no-project --python 3.10 --with mypy --with numpy --with openai --with pytest mypy $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with flake8 flake8 --jobs=1 $(SOURCE_DIRS) sitecustomize.py
	PYLINTHOME=/tmp/pylint-cache $(UV) run --with pylint pylint $(PY_FILES)

quality: format lint

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(UV) run --with pytest --with pytest-xdist --with rich pytest -p xdist.plugin -n $(TEST_WORKERS) --dist=$(TEST_DIST)

test-matrix:
	UV="$(UV)" PYTHON_TEST_VERSIONS="$(PYTHON_TEST_VERSIONS)" bash scripts/run_tests_uv_matrix.sh

security:
	$(UV) run python scripts/security_audit.py $(SECURITY_ARGS)

dependency-matrix:
	$(UV) run python scripts/test_dependency_matrix.py

dependency-matrix-bounds:
	$(UV) run python scripts/test_dependency_matrix.py --bounds-only --no-optional

latency-backends:
	$(UV) run --with rich python examples/scripts/compare_embedding_backend_latency.py
