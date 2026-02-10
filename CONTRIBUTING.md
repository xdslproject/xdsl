# Contributing to xDSL

This document describes the steps needed to clone, install, and test the repository,
as well as a description of our coding and contributing conventions.

## Contents

- [Developer Installation](#developer-installation)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Formatting and Typechecking](#formatting-and-typechecking)
- [Code Style](#code-style)
- [Discussion](#discussion)

## Developer Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management of xDSL.
See uv's [getting started page](https://docs.astral.sh/uv/getting-started/) for more
details.

```sh
# Ensure uv is installed
uv --version
```

Then, here are the commands to locally set up your development repository:

```sh
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

Please take a look at the [Makefile](./Makefile)
for the available commands such as running specific tests,
running the documentation website locally, and others.

To make a custom mlir-opt available in the virtual environment, set the
`XDSL_MLIR_OPT_PATH` variable when running `make venv`, like so:

``` bash
XDSL_MLIR_OPT_PATH=/PATH/TO/LLVM/BUILD/bin/mlir-opt make venv
```

### Alternative installations

For some use-cases, such as running xDSL with [PyPy](https://pypy.org/),
it may be preferable to install a minimal set of dependencies instead.
This can be done with `uv sync`. Note that Pyright will then complain
about missing dependencies, so run `make tests-functional` instead of
`make tests` to test the functionality of xDSL.

## Testing and benchmarking

The xDSL project uses pytest unit tests, LLVM-style filecheck tests and performance
benchmarks. They can be executed from the root directory with `make tests` (which runs
everything except benchmarks and also runs pyright for type checking).

### Unit Tests

Python tests in `tests/` (excluding `tests/filecheck`) for testing APIs and logic:

```bash
# Run unit tests
uv run pytest
# or via makefile
make pytest
```

### FileCheck Tests

File-based tests in `tests/filecheck` using `filecheck` (a Python reimplementation of
LLVM's FileCheck) to verify tool output. These tests rely on the textual format to
represent and construct IR. They are used to test that custom format implementations
print and parse in the expected way, and to verify transformations such as pattern
rewrites or passes:

```bash
# Run filecheck tests
uv run lit tests/filecheck
# or via makefile
make filecheck
```

### Coverage Tests

To generate coverage reports for both unit tests and filecheck tests:

```bash
# Run coverage for unit tests and filecheck tests, then combine data files
make coverage
# Generate a coverage report
make coverage-report
# Or generate an HTML coverage report
make coverage-report-html
```

You can also run coverage for each test type individually:

```bash
# Run coverage for unit tests only
make coverage-tests
# Run coverage for filecheck tests only
make coverage-filecheck-tests
```

### Benchmarks

Benchmarks for the project are tracked in the
<https://github.com/xdslproject/xdsl-bench> repository.
These run automatically every day on the main branch, reporting their results to <https://xdsl.dev/xdsl-bench/>.
However, they can also be ran manually by cloning the repository and pointing the
submodule at your feature branch to benchmark.

## Formatting and Typechecking

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
[prek](https://github.com/j178/prek) package, a drop-in replacement for
[pre-commit](https://pypi.org/project/pre-commit/).

```bash
# Install the pre-commit on your `.git` folder
make precommit-install
# Run the hooks
make precommit
```

## Code Style

We aim to follow these rules for all changes in this repository:

- We aim for consistency in the code style and architectural patterns throughout the
  codebase in order to make it as easy as possible to understand and modify any part of
  xDSL.
- We fix issues immediately rather than relying on future refactoring, as technical debt
  tends to accumulate and become harder to address over time.
- We prefer simplicity: no code is better than obvious code, which is better than clever
  code. Premature abstraction often adds complexity without clear benefit.
- We prioritize code locality over DRY (Don't Repeat Yourself). Keeping related logic close
  together - even if it results in slight duplication - makes it easier to understand
  code in isolation. We prefer lambdas and inline logic over tiny single-use functions
  (<5 LoC), and we minimize variable scope.
- We write self-describing code by using descriptive variable names and constant
  intermediary variables rather than relying heavily on comments.
- We use guard-first logic, handling edge cases, invalid inputs and errors at the start
  of functions. Returning early keeps the "happy path" at the lowest indentation level,
  making the main logic easier to follow.
- We keep if/else blocks small and avoid nesting beyond two levels when possible, as
  flat structures are easier to read and reason about.
- We centralize control flow in parent functions, keeping leaf functions as pure logic.
  This separation makes the codebase more predictable and testable.
- We fail fast by detecting unexpected conditions immediately and raising exceptions
  rather than corrupting state, as this makes debugging easier.
- We follow the Python philosophy of
  "[ask for forgiveness not permission](https://docs.python.org/3/glossary.html#term-eafp)":
  we assume valid keys or attributes exist and catch exceptions if the assumption proves
  false. This leads to cleaner, more Pythonic code:

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

## Discussion

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com),
kindly supported by community hosting from [Zulip](https://zulip.com/).
