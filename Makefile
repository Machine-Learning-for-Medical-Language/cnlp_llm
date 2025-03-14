.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff check --fix
	ruff format
	pyright
	pre-commit run --all-files

.PHONY: test
test:
	pytest
