# Best practices for makefiles
MAKEFLAGS += --warn-undefined-variables
SHELL := bash

# ===== #
#  ASV  #
# ===== #

.PHONY: asv
asv:
	cd .. && uv run asv run

.PHONY: html
html:
	cd .. && uv run asv publish

.PHONY: preview
preview: ../.asv/html
	cd .. && uv run asv preview

.PHONY: clean-local-asv
clean-local-asv: ../.asv
	rm -rf ../.asv/html ../.asv/results/$(shell hostname)

.PHONY: clean-asv
clean-asv: clean-local-asv
	rm -rf ../.asv/results/github-action

# ========= #
# Developer #
# ========= #

.PHONY: lint
lint:
	uv run ruff check . --fix --show-fixes --exit-non-zero-on-fix

.PHONY: format
format:
	uv run ruff format .

.PHONY: types
types:
	uv run pyright .

.PHONY: check
check: lint format types
