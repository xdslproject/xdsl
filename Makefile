MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# allow overriding the name of the venv directory
VENV_DIR ?= .venv

# use activated venv if any
export UV_PROJECT_ENVIRONMENT=$(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),$(VENV_DIR))

# allow overriding which extras are installed
VENV_EXTRAS ?= --all-extras
VENV_GROUPS ?= --all-groups


# default lit options
LIT_OPTIONS ?= -v --order=smart

# make tasks run all commands in a single shell
.ONESHELL:

.PHONY: uv-installed
uv-installed:
	@command -v uv &> /dev/null ||\
		(echo "UV doesn't seem to be installed, try the following instructions:" &&\
		echo "https://docs.astral.sh/uv/getting-started/installation/" && false)

# set up the venv with all dependencies for development
.PHONY: ${VENV_DIR}/
${VENV_DIR}/: uv-installed
	uv sync ${VENV_EXTRAS} ${VENV_GROUPS}
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

# run tests for Toy tutorial
.PHONY: filecheck-toy
filecheck-toy: uv-installed
	uv run lit $(LIT_OPTIONS) docs/Toy/examples

.PHONY: pytest-toy-nb
pytest-toy-nb:
	@if uv run python -c "import riscemu" > /dev/null 2>&1; then \
		uv run pytest -W error --nbval -vv docs/Toy --nbval-current-env; \
	else \
		echo "riscemu is not installed, skipping tests."; \
	fi

.PHONY: tests-toy
tests-toy: filecheck-toy pytest-toy-nb


.PHONY: tests-marimo
tests-marimo: uv-installed
	@bash -c '\
		error_log="/tmp/marimo_test_$$$$.log"; \
		failed_tests=""; \
		files_requiring_mlir_opt=("docs/marimo/mlir_interoperation.py"); \
		for file in docs/marimo/*.py; do \
			if [[ " $${files_requiring_mlir_opt[@]} " =~ " $$file " ]]; then \
				if ! command -v mlir-opt &> /dev/null; then \
					echo "Skipping $$file (mlir-opt is not available)"; \
					continue; \
			  fi; \
			fi; \
			echo "Running $$file"; \
			if ! output=$$(uv run python -W error "$$file" 2>&1); then \
				echo "$$output" >> "$$error_log"; \
				failed_tests="$$failed_tests $$file"; \
			fi; \
		done; \
		if [ ! -z "$$failed_tests" ]; then \
			cat "$$error_log"; \
			echo -e "\n\nThe following marimo tests failed: $$failed_tests"; \
			rm -f "$$error_log"; \
			exit 1; \
		else \
			rm -f "$$error_log"; \
			echo -e "\n\nAll marimo tests passed successfully."; \
		fi'


# run all tests
.PHONY: tests-functional
tests-functional: pytest tests-toy filecheck tests-marimo
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
	uv run pyright --createstub llvmlite
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

# Set to 1 to skip the generation of "API Reference".
SKIP_GEN_PAGES ?= 0
# Set to 1 to skip the building of xDSL wheel
SKIP_BUILD_WHEEL ?= 0


.PHONY: docs-serve
docs-serve: uv-installed
	uv run mkdocs serve

.PHONY: docs-serve-fast
docs-serve-fast: uv-installed
	SKIP_GEN_PAGES=1 uv run mkdocs serve

.PHONY: docs-build
docs-build: uv-installed
	uv run mkdocs build
