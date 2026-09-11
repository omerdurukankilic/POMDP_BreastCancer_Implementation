.PHONY: install test test-watch lint coverage

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-watch:
	ptw --now --clear

lint:
	ruff check src tests

coverage:
	pytest --cov=pomdp_breast_cancer --cov-report=term-missing
