PYTHON := python3.7
LINT_TARGET := setup.py src/ tests/
MYPY_TARGET := src/

.PHONY: clean
# target: clean - Remove intermediate and generated files
clean:
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf *.egg-info
	@rm -f VERSION

.PHONY: lint
# target: lint - Check source code with linters
lint: lint-isort lint-black lint-flake8 lint-mypy lint-pylint

.PHONY: lint-black
lint-black:
	@${PYTHON} -m black  ${LINT_TARGET}

.PHONY: lint-flake8
lint-flake8:
	@${PYTHON} -m flake8 --statistics ${LINT_TARGET}

.PHONY: lint-isort
lint-isort:
	@${PYTHON} -m isort.main -df -c -rc ${LINT_TARGET}

.PHONY: lint-mypy
lint-mypy:
	@${PYTHON} -m mypy ${MYPY_TARGET}

.PHONY: lint-pylint
lint-pylint:
	@${PYTHON} -m pylint --errors-only ${LINT_TARGET}

# `venv` target is intentionally not PHONY.
# target: venv - Creates virtual environment
venv:
	@${PYTHON} -m venv venv

.PHONY: format
# target: format - Format the code according to the coding styles
format: format-black format-isort

.PHONY: format-black
format-black:
	@black ${LINT_TARGET}

.PHONY: format-isort
format-isort:
	@isort -rc ${LINT_TARGET}

.PHONY: test
test:
	pytest tests
