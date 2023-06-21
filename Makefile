MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: clean filecheck pytest tests rerun-notebooks precommit-install precommit black pyright

# remove all caches and the venv
clean:
	rm -rf ${VENV_DIR} .pytest_cache *.egg-info .coverage.*

# run filecheck tests
filecheck:
	lit -vv tests/filecheck
	lit -vv docs/Toy/examples

# run pytest tests
pytest:
	pytest tests -W error -vv
	pytest -W error --nbval -vv docs --ignore=docs/mlir_interoperation.ipynb

# run all tests
tests: pytest filecheck pyright
	echo test

# re-generate the output from all jupyter notebooks in the docs directory
rerun-notebooks:
	jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace --to notebook --execute docs/*.ipynb

# set up all precommit hooks
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
pyright:
	pyright $(shell git diff --staged --name-only)

# run black on all files currently staged
black:
	black $(shell git diff --staged --name-only)

# set up the venv with all dependencies for development
venv: requirements-optional.txt requirements.txt
	python3 -m venv ${VENV_DIR}
	source ${VENV_DIR}/bin/activate
	pip install -r requirements-optional.txt -r requirements.txt
	pip install -e ".[extras]"
