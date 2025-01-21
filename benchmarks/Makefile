# Best practices for makefiles
MAKEFLAGS += --warn-undefined-variables
SHELL := bash

# ========= #
# Profiling #
# ========= #

## Generate profile data from benchmarks
profiles:
	mkdir -p profiles

.PHONY: clean-profiles
clean-profiles:
	rm -rf profiles

.PHONY: timeit_end_to_end
timeit_end_to_end:
	uv run python3 end_to_end.py timeit

.PHONY: snakeviz_end_to_end
snakeviz_end_to_end:
	uv run python3 end_to_end.py -t time_end_to_end_opt__constant_folding snakeviz

.PHONY: flameprof_end_to_end
flameprof_end_to_end:
	uv run python3 end_to_end.py -t time_end_to_end_opt__constant_folding flameprof

.PHONY: viztracer_end_to_end
viztracer_end_to_end:
	uv run python3 end_to_end.py -t time_end_to_end_opt__constant_folding viztracer

## Profile command line directly
.PHONY: viztracer
viztracer_xdsl_opt: .venv xdsl/.venv profiles
	uv run viztracer -o profiles/empty_program.json \
		xdsl-opt ../tests/xdsl_opt/empty_program.mlir
	uv run vizviewer profiles/empty_program.json

# ========= #
# Developer #
# ========= #

.PHONY: lint
lint: .venv
	uv run ruff check . --fix --show-fixes --exit-non-zero-on-fix

.PHONY: format
format:
	uv run ruff format .

.PHONY: types
types:
	uv run mypy . --scripts-are-modules

.PHONY: check
check: lint format types
