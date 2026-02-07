# Contributing to xDSL

## Contents

- [Developer Installation](#developer-installation)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Formatting and Typechecking](#formatting-and-typechecking)

## xDSL Developer Setup

To contribute to the development of xDSL follow the subsequent steps.

### Developer Installation

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

### Testing and benchmarking

The xDSL project uses pytest unit tests and LLVM-style filecheck tests. They can
be executed from the root directory:

```bash
# Executes pytests which are located in tests/
uv run pytest

# Executes filecheck tests
uv run lit tests/filecheck

# run all tests using makefile
make tests
```

Benchmarks for the project are tracked in the
<https://github.com/xdslproject/xdsl-bench> repository.
These run automatically every day on the main branch, reporting their results to <https://xdsl.dev/xdsl-bench/>.
However, they can also be ran manually by cloning the repository and pointing the
submodule at your feature branch to benchmark.

### Formatting and Typechecking

All Python code used in xDSL uses [ruff](https://docs.astral.sh/ruff/formatter/) to
format the code in a uniform manner.

To automate the formatting, we use pre-commit hooks from
[prek](https://github.com/j178/prek), a drop-in replacement for
[pre-commit](https://pypi.org/project/pre-commit/).

```bash
# Install the pre-commit on your `.git` folder
make precommit-install
# to run the hooks:
make precommit
# alternatively, run ruff directly:
uv run ruff format
```

When commiting to xDSL, try to pass all Python code through
[pyright](https://github.com/microsoft/pyright) without errors.
Pyright checks all staged files through the
makefile using `make pyright`.

> [!IMPORTANT]
>
> #### Experimental Pyright Features
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

### Discussion

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com),
kindly supported by community hosting from [Zulip](https://zulip.com/).
