# define the name of the virtual environment directory
VENV := venv

# default target, when make executed without arguments
all: venv

install: FORCE
	pip install --upgrade pip
	pip install -r requirements.txt --require-venv
	pip install -r requirements-optional.txt --require-venv
	pip install --editable . --require-venv

requirements.txt:
	pip install -r requirements.txt --require-venv

# update is a shortcut target
update: requirements.txt

test: update
	# Executes pytests which are located in tests/
	pytest
	# Executes filecheck tests
	lit tests/filecheck

format:
	yapf -ir xdsl

lint:
	pyright

lint-ci: pyright-ci.json
	pyright -p pyright-ci.json

clean:
	rm -r $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: all install update venv test clean

FORCE: ;
