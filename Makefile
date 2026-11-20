.PHONY: verify test test-python test-typescript

# Run package tests concurrently without changing verify's ordered checks.
test:
	$(MAKE) -j2 test-python test-typescript

test-python:
	$(MAKE) -C python test

test-typescript:
	$(MAKE) -C typescript test

verify:
	$(MAKE) -C python verify
	$(MAKE) -C typescript verify
