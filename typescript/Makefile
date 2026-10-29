SHELL := /bin/sh
UV ?= uv
SECURITY_ARGS ?=
SECURITY_VENV := .security-tools
# Windows venvs place executables in Scripts/ rather than bin/.
ifeq ($(OS),Windows_NT)
SECURITY_BIN := Scripts
SECURITY_PY := python.exe
else
SECURITY_BIN := bin
SECURITY_PY := python
endif
NODE_TEST_VERSIONS ?=

.PHONY: verify install install-dev format lint quality test test-matrix security dependency-matrix dependency-matrix-bounds latency-backends

install:
	npm install --omit=dev

install-dev:
	npm install --include=dev
	rm -rf $(SECURITY_VENV)
	$(UV) venv --python 3.12 $(SECURITY_VENV)
	$(UV) pip install --python $(SECURITY_VENV)/$(SECURITY_BIN)/$(SECURITY_PY) semgrep detect-secrets "cryptography>=50.0.2; sys_platform != 'darwin' or platform_machine != 'x86_64'" "cryptography>=48.0.1,<49; sys_platform == 'darwin' and platform_machine == 'x86_64'" --only-binary cryptography

verify:
	$(MAKE) install-dev
	$(MAKE) test
	$(MAKE) test-matrix
	$(MAKE) security

format:
	npm run format

lint:
	npm run lint

test:
	npm test

test-matrix:
	NODE_TEST_VERSIONS="$(NODE_TEST_VERSIONS)" npm run test:all-versions

security:
	PATH="$(CURDIR)/$(SECURITY_VENV)/$(SECURITY_BIN):$(CURDIR)/node_modules/.bin:$$PATH" npm run security:audit -- $(SECURITY_ARGS)

dependency-matrix:
	npm run test:dependency-matrix

dependency-matrix-bounds:
	npm run test:dependency-matrix-bounds

latency-backends:
	npm run test:latency-backends


quality: format lint
