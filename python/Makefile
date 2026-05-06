PYTHON ?= python3
SOURCE_DIRS := agent_action_guard examples scripts tests training
PY_FILES := $(shell rg --files $(SOURCE_DIRS) -g '*.py') sitecustomize.py

.PHONY: format lint quality

format:
	$(PYTHON) -m isort $(SOURCE_DIRS) sitecustomize.py
	$(PYTHON) -m black $(SOURCE_DIRS) sitecustomize.py

lint:
	$(PYTHON) -m ruff check $(SOURCE_DIRS) sitecustomize.py
	$(PYTHON) -m black --check $(SOURCE_DIRS) sitecustomize.py
	$(PYTHON) -m isort --check-only $(SOURCE_DIRS) sitecustomize.py
	$(PYTHON) -m mypy $(SOURCE_DIRS) sitecustomize.py
	$(PYTHON) -m flake8 --jobs=1 $(SOURCE_DIRS) sitecustomize.py
	PYLINTHOME=/tmp/pylint-cache $(PYTHON) -m pylint $(PY_FILES)

quality: format lint
