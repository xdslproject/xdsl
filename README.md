<!-- markdownlint-disable-next-line MD041 -->
[![Build Status for the Core backend](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml/badge.svg)](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml?query=workflow%3A%22CI+-+Python+application%22++)
[![PyPI version](https://badge.fury.io/py/xdsl.svg)](https://badge.fury.io/py/xdsl)
[![Downloads](https://static.pepy.tech/badge/xdsl)](https://www.pepy.tech/projects/xdsl)
[![Downloads](https://static.pepy.tech/badge/xdsl/week)](https://pepy.tech/project/xdsl)
[![Code Coverage](https://codecov.io/gh/xdslproject/xdsl/main/graph/badge.svg)](https://codecov.io/gh/xdslproject/xdsl)
[![Zulip Status](https://img.shields.io/badge/chat-on%20zulip-%2336C5F0)](https://xdsl.zulipchat.com)

# xDSL: A Python-native SSA Compiler Framework

[xDSL](http://www.xdsl.dev) is a Python-native framework for
building compiler infrastructure. It provides *[SSA-based intermediate
representations (IRs)](https://en.wikipedia.org/wiki/Static_single-assignment_form)*
and Pythonic APIs to define, assemble, and optimize custom IRs—all with seamless
compatibility with [MLIR](https://mlir.llvm.org/) from the LLVM project.

Inspired by MLIR, xDSL enables smooth translation of programs and abstractions
between frameworks. This lets users prototype compilers entirely in Python,
while still accessing MLIR’s powerful optimization and code generation pipeline.
All IRs in xDSL employ a unified SSA-based data structure, with regions and basic blocks,
making it easy to write generic analyses and transformation passes.

xDSL supports assembling compilers from predefined or custom IRs, and organizing
transformations across a multi-level IR stack. This layered approach enables
abstraction-specific optimization passes, similar to the architecture of projects
like [Devito](https://github.com/devitocodes/devito), [PSyclone](https://github.com/stfc/PSyclone),
and [Firedrake](https://github.com/firedrakeproject/firedrake).

In short, xDSL makes it possible to:

- Prototype compilers quickly in Python
- Build DSLs with custom IRs
- Run analyses and transformations with simple scripts
- Interoperate smoothly with MLIR and benefit from LLVM's backend

## Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [xDSL Developer Setup](#xdsl-developer-setup)
  - [Developer Installation](#developer-installation)
  - [Testing and benchmarking](#testing-and-benchmarking)
  - [Formatting and Typechecking](#formatting-and-typechecking)

## Installation

To use xDSL as part of a larger project for developing your own compiler,
just install [xDSL via pip](https://pypi.org/project/xdsl/):

```bash
pip install xdsl
```

To quickly install xDSL for development and contribution purposes, use:

``` bash
pip install xdsl[dev]
```

This may be useful for projects wanting to replicate the xDSL testing setup.
For a more comprehensive experience, follow: [xDSL Developer Setup](#xdsl-developer-setup)

*Note:* This version of xDSL is validated against a specific MLIR version,
interoperability with other versions is not guaranteed. The supported
MLIR version is 21.1.1.

### Subprojects With Extra Dependencies

xDSL has a number of subprojects, some of which require extra dependencies.
To keep the set of dependencies to a minimum, these extra dependencies have to
be specified explicitly, e.g. by using:

``` bash
pip install xdsl[gui] # or [jax], [riscv]
```

## Getting Started

Check out the dedicated [Getting Started guide](https://xdsl.readthedocs.io/stable/)
for a comprehensive tutorial.

To get familiar with xDSL, we recommend starting with our Jupyter notebooks. The
notebooks provide hands-on examples and documentation of xDSL's core concepts: data
structures, the Python-embedded abstraction definition language, and end-to-end custom
compilers construction, like a database compiler.
There also exists a small documentation showing how to connect xDSL with MLIR
for users interested in that use case.

- [A Database example](https://xdsl.dev/xdsl/lab/index.html?path=database_example.ipynb)
- [A simple introduction](https://xdsl.dev/xdsl/lab/index.html?path=tutorial.ipynb)
- [A DSL for defining new IRs](https://xdsl.dev/xdsl/lab/index.html?path=irdl.ipynb)
- [Connecting xDSL with MLIR](docs/guides/mlir_interoperation.md)

We provide a [Makefile](https://github.com/xdslproject/xdsl/blob/main/Makefile)
containing a lot of common tasks, which might provide an overview of common actions.

## xDSL Developer Setup

To contribute to the development of xDSL follow the subsequent steps.

### Developer Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management of xDSL.
See uv's [getting started page](https://docs.astral.sh/uv/getting-started/) for more
details.

```sh
# Ensure uv is installed
uv -v
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

Please take a look at the [Makefile](https://github.com/xdslproject/xdsl/blob/main/Makefile)
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

To automate the formatting, we use pre-commit hooks from the
[pre-commit](https://pypi.org/project/pre-commit/) package.

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
