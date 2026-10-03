SHELL := /bin/sh
UV ?= uv
SECURITY_VENV := .security-tools

.PHONY: install install-dev format lint quality test security dependency-matrix dependency-matrix-bounds latency-backends

install:
	npm install --omit=dev

install-dev:
	npm install --include=dev
	rm -rf $(SECURITY_VENV)
	$(UV) venv --python 3.12 $(SECURITY_VENV)
	$(UV) pip install --python $(SECURITY_VENV)/bin/python semgrep detect-secrets "cryptography<50" --only-binary cryptography

format:
	npm run format

lint:
	npm run lint

test:
	npm test

security:
	PATH="$(CURDIR)/$(SECURITY_VENV)/bin:$(CURDIR)/node_modules/.bin:$$PATH" npm run security:audit

dependency-matrix:
	npm run test:dependency-matrix

dependency-matrix-bounds:
	npm run test:dependency-matrix-bounds

latency-backends:
	npm run test:latency-backends


quality: format lint
