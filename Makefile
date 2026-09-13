.PHONY: install test test-watch lint coverage

install:
	uv sync --extra dev
	uv run pre-commit install

test:
	uv run pytest

test-watch:
	uv run ptw --now --clear

lint:
	uv run ruff check src tests scripts

coverage:
	uv run pytest --cov=pomdp_breast_cancer --cov=scripts --cov-report=term-missing
