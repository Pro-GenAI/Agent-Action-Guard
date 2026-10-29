.PHONY: verify

verify:
	$(MAKE) -C python verify
	$(MAKE) -C typescript verify
