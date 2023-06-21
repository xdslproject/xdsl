MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

xdsl_sources = $(shell find xdsl -type f)
test_sources = $(shell find tests -type f)
filecheck_sources = $(shell find tests -iname '*.mlir' -type f)


.PHONY: clean filecheck pytest tests rerun-notebooks precommit-install precommit black pyright

# remove all caches and the venv
clean:
	rm -rf venv .pytest_cache *.egg-info .coverage.*

# run filecheck tests
filecheck:
	lit -vv tests/filecheck

# run pytest tests
pytest:
	pytest tests -W error -vv

# run all tests
tests: pytest filecheck
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
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements-optional.txt requirements.txt
	pip install -e ".[extras]"
