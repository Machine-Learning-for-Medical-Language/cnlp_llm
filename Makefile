.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff check --fix
	ruff format
	pyright

.PHONY: test
test:
	pytest
