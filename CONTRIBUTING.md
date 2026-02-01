# Contributing to xDSL

## Contents

- [Developer Setup](#developer-setup)
  - [Installation](#installation)
  - [Alternative Installations](#alternative-installations)
  - [Custom MLIR Build](#custom-mlir-build)
- [Code Style](#code-style)
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [FileCheck Tests](#filecheck-tests)
  - [Benchmarks](#benchmarks)
- [Linting and Formatting](#linting-and-formatting)
  - [Pre-commit Hooks](#pre-commit-hooks)

## Developer Setup

### Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management of xDSL.
See uv's [getting started page](https://docs.astral.sh/uv/getting-started/) for more
details.

To locally set up your development repository:

```sh
# Ensure uv is installed
uv --version

# Clone repo
git clone https://github.com/xdslproject/xdsl.git
cd xdsl

# Set up local environment with all optional and dev dependencies
# Creates a virtual environment called `.venv`
make venv
# Set up pre-commit hook for automatic formatting
make precommit-install
# Run all tests to verify installation was successful
make tests
```

Please take a look at the [Makefile](https://github.com/xdslproject/xdsl/blob/main/Makefile)
for the available commands such as running specific tests,
running the documentation website locally, and others.

### Alternative Installations

For some use-cases, such as running xDSL with [PyPy](https://pypy.org/),
it may be preferable to install a minimal set of dependencies instead.
This can be done with `uv sync`. Note that Pyright will then complain
about missing dependencies, so run `make tests-functional` instead of
`make tests` to test the functionality of xDSL.

### Custom MLIR Build

To make a custom mlir-opt available in the virtual environment, set the
`XDSL_MLIR_OPT_PATH` variable when running `make venv`:

```bash
XDSL_MLIR_OPT_PATH=/PATH/TO/LLVM/BUILD/bin/mlir-opt make venv
```

## Code Style

We aim to follow these rules for all changes in this repository:

- Match existing code style and architectural patterns.
- Zero Technical Debt: Fix issues immediately. Never rely on future refactoring.
- Keep it simple: No code > Obvious code > Clever code. Do not abstract prematurely.
- Locality over DRY: Prioritize code locality. Keep related logic close together even if
  it results in slight duplication. Prefer lambdas/inline logic over tiny single-use
  functions (<5 LoC). Minimize variable scope.
- Self-Describing Code: Minimize comments. Use descriptive variable names and constant
  intermediary variables to explain where possible.
- Guard-First Logic: Handle edge cases, invalid inputs and errors at the start of
  functions. Return early to keep the "happy path" at the lowest indentation level.
- Flat Structure: Keep if/else blocks small. Avoid nesting beyond two levels if possible.
- Centralize Control Flow: Branching logic belongs in parents. Leaf functions should be
  pure logic.
- Fail Fast: Detect unexpected conditions immediately. Raise Exceptions rather than
  corrupt state.
- [Ask for forgiveness not permission](https://docs.python.org/3/glossary.html#term-eafp):
  Assume valid keys or attributes exist and catch exceptions if the assumption proves
  false. Use try-except blocks:

  ```python
  # Good
  try:
      return mapping[key]
  except KeyError:
      return default_value

  # Bad
  if key in mapping:
      return mapping[key]
  return default_value
  ```

## Testing

The xDSL project uses pytest unit tests and LLVM-style filecheck tests.

### Unit Tests

Python tests in `tests/` (excluding `tests/filecheck`) for testing APIs and logic:

```bash
# Run unit tests
uv run pytest

# or via makefile
make pytest

# Check coverage
make coverage && make coverage-report
```

### FileCheck Tests

File-based tests in `tests/filecheck` using `filecheck` (a Python reimplementation of
LLVM's FileCheck) to verify tool output. These tests rely on the textual format to
represent and construct IR. They are used to test that custom format implementations
print and parse in the expected way, and to verify transformations such as pattern
rewrites or passes:

```bash
uv run lit tests/filecheck

# or via makefile
make filecheck
```

### Benchmarks

Benchmarks for the project are tracked in the
<https://github.com/xdslproject/xdsl-bench> repository.
These run automatically every day on the main branch, reporting their results to <https://xdsl.dev/xdsl-bench/>.
However, they can also be ran manually by cloning the repository and pointing the
submodule at your feature branch to benchmark.

## Linting and Formatting

Configuration for linting and formatting is found in `pyproject.toml`.

[Ruff](https://github.com/astral-sh/ruff) is used for linting and formatting.
Configured in `[tool.ruff]`.

[Pyright](https://github.com/microsoft/pyright) is used for static type checking.
Configured in `[tool.pyright]`.

```bash
# Format code
uv run ruff format

# Type check code
uv run pyright
# or via makefile
make pyright
```

> [!IMPORTANT]
>
> ### Experimental Pyright Features
>
> xDSL currently relies on an experimental feature of Pyright called TypeForm.
> TypeForm is [in discussion](https://discuss.python.org/t/pep-747-typeexpr-type-hint-for-a-type-expression/55984)
> and will likely land in some future version of Python.
>
> For xDSL to type check correctly using Pyright, please add this to your `pyproject.toml`:
>
> ```toml
> [tool.pyright]
> enableExperimentalFeatures = true
> ```

### Pre-commit Hooks

To automate the formatting and type checking, we use pre-commit hooks from the
[pre-commit](https://pypi.org/project/pre-commit/) package:

```bash
# Install the pre-commit on your `.git` folder
make precommit-install
# Run the hooks
make precommit
```
