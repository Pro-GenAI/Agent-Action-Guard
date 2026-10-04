UV ?= uv
TEST_WORKERS ?= logical
TEST_DIST ?= worksteal
SOURCE_DIRS := agent_action_guard examples scripts tests training
PY_FILES := $(shell find $(SOURCE_DIRS) -type f -name '*.py') sitecustomize.py

.PHONY: install install-dev format lint quality test security dependency-matrix dependency-matrix-bounds latency-backends

install:
	$(UV) sync

install-dev:
	$(UV) sync --extra dev

format:
	$(UV) run --with isort isort $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with black black $(SOURCE_DIRS) sitecustomize.py

lint:
	$(UV) run --with ruff ruff check $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with black black --check $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with isort isort --check-only $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with mypy mypy $(SOURCE_DIRS) sitecustomize.py
	$(UV) run --with flake8 flake8 --jobs=1 $(SOURCE_DIRS) sitecustomize.py
	PYLINTHOME=/tmp/pylint-cache $(UV) run --with pylint pylint $(PY_FILES)

quality: format lint

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(UV) run --with pytest --with pytest-xdist --with rich pytest -p xdist.plugin -n $(TEST_WORKERS) --dist=$(TEST_DIST)

security:
	$(UV) run python scripts/security_audit.py

dependency-matrix:
	$(UV) run python scripts/test_dependency_matrix.py

dependency-matrix-bounds:
	$(UV) run python scripts/test_dependency_matrix.py --bounds-only --no-optional

latency-backends:
	$(UV) run --with rich python examples/scripts/compare_embedding_backend_latency.py
