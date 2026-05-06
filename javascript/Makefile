SHELL := /bin/sh

.PHONY: format lint quality test

format:
	npm run format

lint:
	npm run lint

test:
	npm run test:all-versions

quality: format lint
