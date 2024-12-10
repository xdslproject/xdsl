MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# default lit options
LIT_OPTIONS ?= -v --order=smart

# make tasks run all commands in a single shell
.ONESHELL:

# set up the venv with all dependencies for development
.PHONY: ${VENV_DIR}/
${VENV_DIR}/: requirements.txt
	python3 -m venv ${VENV_DIR}
	. ${VENV_DIR}/bin/activate
	python3 -m pip --require-virtualenv install -r requirements.txt

# make sure `make venv` always works no matter what $VENV_DIR is
venv: ${VENV_DIR}/

# remove all caches
.PHONY: clean-caches
clean-caches: coverage-clean
	rm -rf .pytest_cache *.egg-info

# remove all caches and the venv
.PHONY: clean
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
.PHONY: filecheck
filecheck:
	lit $(LIT_OPTIONS) tests/filecheck

# run pytest tests
.PHONY: pytest
pytest:
	pytest tests -W error -vv

# run pytest on notebooks
.PHONY: pytest-nb
pytest-nb:
	pytest -W error --nbval -vv docs \
		--ignore=docs/mlir_interoperation.ipynb \
		--ignore=docs/Toy \
		--nbval-current-env

# run tests for Toy tutorial
.PHONY: filecheck-toy
filecheck-toy:
	lit $(LIT_OPTIONS) docs/Toy/examples

.PHONY: pytest-toy
pytest-toy:
	pytest docs/Toy/toy/tests

.PHONY: pytest-toy-nb
pytest-toy-nb:
	@if python -c "import riscemu" > /dev/null 2>&1; then \
		pytest -W error --nbval -vv docs/Toy --nbval-current-env; \
	else \
		echo "riscemu is not installed, skipping tests."; \
	fi

.PHONY: tests-toy
tests-toy: filecheck-toy pytest-toy pytest-toy-nb

.PHONY: tests-marimo
tests-marimo:
	@for file in docs/marimo/*.py; do \
		echo "Running $$file"; \
		error_message=$$(python3 "$$file" 2>&1) || { \
			echo "Error running $$file"; \
			echo "$$error_message"; \
			exit 1; \
		}; \
	done
	@echo "All marimo tests passed successfully."

.PHONY: tests-marimo-onnx
tests-marimo-onnx:
	@if python -c "import onnx" > /dev/null 2>&1; then \
		echo "onnx is installed, running tests."; \
		if ! command -v mlir-opt > /dev/null 2>&1; then \
			echo "MLIR is not installed, skipping tests."; \
			exit 0; \
		fi; \
		for file in docs/marimo/onnx/*.py; do \
			echo "Running $$file"; \
			error_message=$$(python3 "$$file" 2>&1) || { \
				echo "Error running $$file"; \
				echo "$$error_message"; \
				exit 1; \
			}; \
		done; \
		echo "All marimo onnx tests passed successfully."; \
	else \
		echo "onnx is not installed, skipping tests."; \
	fi

# run all tests
.PHONY: tests-functional
tests-functional: pytest tests-toy filecheck pytest-nb tests-marimo tests-marimo-onnx
	@echo All functional tests done.

# run all tests
.PHONY: tests
tests: tests-functional pyright
	@echo All tests done.

# re-generate the output from all jupyter notebooks in the docs directory
.PHONY: rerun-notebooks
rerun-notebooks:
	jupyter nbconvert \
		--ClearMetadataPreprocessor.enabled=True \
		--inplace \
		--to notebook \
		--execute docs/*.ipynb docs/Toy/*.ipynb

# set up all precommit hooks
.PHONY: precommit-install
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
# make sure to generate the python typing stubs before running pyright
.PHONY: pyright
pyright:
	xdsl-stubgen
	pyright $(shell git diff --staged --name-only  -- '*.py')

# run coverage over all tests and combine data files
.PHONY: coverage
coverage: coverage-tests coverage-filecheck-tests
	coverage combine --append

# run coverage over tests
# use different coverage data file per coverage run, otherwise combine complains
.PHONY: coverage-tests
coverage-tests:
	COVERAGE_FILE="${COVERAGE_FILE}.$@" pytest -W error --cov --cov-config=.coveragerc

# run coverage over filecheck tests
.PHONY: coverage-filecheck-tests
coverage-filecheck-tests:
	lit $(LIT_OPTIONS) tests/filecheck/ -DCOVERAGE

# generate html coverage report
.PHONY: coverage-report-html
coverage-report-html:
	coverage html

# generate coverage report
.PHONY: coverage-report
coverage-report:
	coverage report

.PHONY: coverage-clean
coverage-clean:
	coverage erase
