SHELL := /bin/sh

.PHONY: format lint quality test dependency-matrix dependency-matrix-bounds latency-backends

format:
	npm run format

lint:
	npm run lint

test:
	npm run test:all-versions

dependency-matrix:
	npm run test:dependency-matrix

dependency-matrix-bounds:
	npm run test:dependency-matrix-bounds

latency-backends:
	npm run test:latency-backends

quality: format lint
