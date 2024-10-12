MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# allow overriding which extras are installed
VENV_EXTRAS ?= --all-extras

# use different coverage data file per coverage run, otherwise combine complains
TESTS_COVERAGE_FILE = ${COVERAGE_FILE}.tests

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: venv clean clean-caches filecheck pytest pytest-nb tests-toy tests
.PHONY: rerun-notebooks precommit-install precommit pyright tests-marimo
.PHONY: tests-marimo-mlir coverage coverage-tests coverage-filecheck-tests
.PHONY: coverage-report-html coverage-report-md

# set up the venv with all dependencies for development
${VENV_DIR}/:
	uv sync ${VENV_EXTRAS}

# make sure `make venv` always works no matter what $VENV_DIR is
venv: ${VENV_DIR}/

# remove all caches
clean-caches:
	rm -rf .pytest_cache *.egg-info .coverage.*
	find . -type f -name "*.cover" -delete

# remove all caches and the venv
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
filecheck:
	lit -vv tests/filecheck --order=smart --timeout=20

# run pytest tests
pytest:
	pytest tests -W error -vv

# run pytest on notebooks
pytest-nb:
	pytest -W error --nbval -vv docs \
		--ignore=docs/mlir_interoperation.ipynb \
		--nbval-current-env

# run tests for Toy tutorial
filecheck-toy:
	lit -v docs/Toy/examples --order=smart

pytest-toy:
	pytest docs/Toy/toy/tests

tests-toy: filecheck-toy pytest-toy

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

tests-marimo-mlir:
	@if ! command -v mlir-opt > /dev/null 2>&1; then \
		echo "MLIR is not installed, skipping tests."; \
		exit 0; \
	fi
	@echo "MLIR is installed, running tests."
	@for file in docs/marimo/mlir/*.py; do \
		echo "Running $$file"; \
		error_message=$$(python3 "$$file" 2>&1) || { \
			echo "Error running $$file"; \
			echo "$$error_message"; \
			exit 1; \
		}; \
	done
	@echo "All marimo mlir tests passed successfully."

# run all tests
tests: pytest tests-toy filecheck pytest-nb tests-marimo tests-marimo-mlir pyright
	@echo All tests done.

# re-generate the output from all jupyter notebooks in the docs directory
rerun-notebooks:
	jupyter nbconvert \
		--ClearMetadataPreprocessor.enabled=True \
		--inplace \
		--to notebook \
		--execute docs/*.ipynb docs/Toy/*.ipynb

# set up all precommit hooks
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
pyright:
    # We make sure to generate the python typing stubs before running pyright
	xdsl-stubgen
	pyright $(shell git diff --staged --name-only  -- '*.py')

# run coverage over all tests and combine data files
coverage: coverage-tests coverage-filecheck-tests
	coverage combine --append

# run coverage over tests
coverage-tests:
	COVERAGE_FILE=${TESTS_COVERAGE_FILE} pytest -W error --cov --cov-config=.coveragerc

# run coverage over filecheck tests
coverage-filecheck-tests:
	lit -v tests/filecheck/ -DCOVERAGE

# generate html coverage report
coverage-report-html:
	coverage html

# generate markdown coverage report
coverage-report-md:
	coverage report --format=markdown
