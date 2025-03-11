MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# allow overriding the name of the venv directory
VENV_DIR ?= .venv

# use activated venv if any
export UV_PROJECT_ENVIRONMENT=$(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),$(VENV_DIR))

# allow overriding which extras are installed
VENV_EXTRAS ?= --extra gui --extra dev --extra jax --extra riscv --extra docs --extra bench

# default lit options
LIT_OPTIONS ?= -v --order=smart

# make tasks run all commands in a single shell
.ONESHELL:

# use bash as the shell
SHELL := /bin/bash

.PHONY: uv-installed
uv-installed:
	@command -v uv &> /dev/null ||\
		(echo "UV doesn't seem to be installed, try the following instructions:" &&\
		echo "https://docs.astral.sh/uv/getting-started/installation/" && false)

# set up the venv with all dependencies for development
.PHONY: ${VENV_DIR}/
${VENV_DIR}/: uv-installed
	uv sync ${VENV_EXTRAS}
	@if [ ! -z "$(XDSL_MLIR_OPT_PATH)" ]; then \
		ln -sf $(XDSL_MLIR_OPT_PATH) ${VENV_DIR}/bin/mlir-opt; \
	fi

# make sure `make venv` also works correctly
.PHONY: venv
venv: ${VENV_DIR}/

# remove all caches
.PHONY: clean-caches
clean-caches: coverage-clean asv-clean
	rm -rf .pytest_cache *.egg-info

# remove all caches and the venv
.PHONY: clean
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
.PHONY: filecheck
filecheck: uv-installed
	uv run lit $(LIT_OPTIONS) tests/filecheck

# run pytest tests
.PHONY: pytest
pytest: uv-installed
	uv run pytest tests -W error -vv

# run pytest on notebooks
.PHONY: pytest-nb
pytest-nb: uv-installed
	uv run pytest -W error --nbval -vv docs \
		--ignore=docs/mlir_interoperation.ipynb \
		--ignore=docs/Toy \
		--nbval-current-env

# run tests for Toy tutorial
.PHONY: filecheck-toy
filecheck-toy: uv-installed
	uv run lit $(LIT_OPTIONS) docs/Toy/examples

.PHONY: pytest-toy
pytest-toy: uv-installed
	uv run pytest docs/Toy/toy/tests

.PHONY: pytest-toy-nb
pytest-toy-nb:
	@if uv run python -c "import riscemu" > /dev/null 2>&1; then \
		uv run pytest -W error --nbval -vv docs/Toy --nbval-current-env; \
	else \
		echo "riscemu is not installed, skipping tests."; \
	fi

.PHONY: tests-toy
tests-toy: filecheck-toy pytest-toy pytest-toy-nb


.PHONY: tests-marimo
tests-marimo: uv-installed
	@ERROR_LOG=$(shell mktemp)
	@declare -a FAILED_MARIMO_TESTS
	@for file in docs/marimo/*.py; do \
		echo "Running $$file"; \
		uv run python3 "$$file" 2>> "$${ERROR_LOG}"; \
		if [ $$? -ne 0 ]; then \
			FAILED_MARIMO_TESTS+=($$file); \
		fi; \
	done;
	@if [[ ! -z $${FAILED_MARIMO_TESTS[@]} ]]; then \
		cat "$${ERROR_LOG}"; \
		echo -e "\n\nThe following marimo tests failed: $${FAILED_MARIMO_TESTS[@]}"; \
		exit 1; \ 
	else \
		echo -e "\n\nAll marimo tests passed successfully."; \
	fi


# run all tests
.PHONY: tests-functional
tests-functional: pytest tests-toy filecheck pytest-nb tests-marimo
	@echo All functional tests done.

# run all tests
.PHONY: tests
tests: tests-functional pyright
	@echo All tests done.

# re-generate the output from all jupyter notebooks in the docs directory
.PHONY: rerun-notebooks
rerun-notebooks: uv-installed
	uv run jupyter nbconvert \
		--ClearMetadataPreprocessor.enabled=True \
		--inplace \
		--to notebook \
		--execute docs/*.ipynb docs/Toy/*.ipynb

# set up all precommit hooks
.PHONY: precommit-install
precommit-install: uv-installed
	uv run pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit: uv-installed
	uv run pre-commit run --all

# run pyright on all files in the current git commit
# make sure to generate the python typing stubs before running pyright
.PHONY: pyright
pyright: uv-installed
	uv run xdsl-stubgen
	uv run pyright $(shell git diff --staged --name-only  -- '*.py')

# run coverage over all tests and combine data files
.PHONY: coverage
coverage: coverage-tests coverage-filecheck-tests
	uv run coverage combine --append

# use different coverage data file per coverage run, otherwise combine complains
.PHONY: coverage-tests
coverage-tests: uv-installed
	COVERAGE_FILE="${COVERAGE_FILE}.$@" uv run pytest -W error --cov

# run coverage over filecheck tests
.PHONY: coverage-filecheck-tests
coverage-filecheck-tests: uv-installed
	uv run lit $(LIT_OPTIONS) tests/filecheck/ -DCOVERAGE

# generate html coverage report
.PHONY: coverage-report-html
coverage-report-html: uv-installed
	uv run coverage html

# generate coverage report
.PHONY: coverage-report
coverage-report: uv-installed
	uv run coverage report

.PHONY: coverage-clean
coverage-clean: uv-installed
	uv run coverage erase

# generate asv benchmark regression website
.PHONY: asv
asv: uv-installed
	uv run asv run

.PHONY: asv-html
asv-html: uv-installed
	uv run asv publish

.PHONY: asv-preview
asv-preview: uv-installed .asv/html
	uv run asv preview

.PHONY: asv-clean
asv-clean:
	rm -rf .asv/

# docs
.PHONY: docs-serve
docs-serve: uv-installed
	uv run mkdocs serve

.PHONY: docs-build
docs-build: uv-installed
	uv run mkdocs build
