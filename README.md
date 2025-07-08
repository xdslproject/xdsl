<!-- markdownlint-disable-next-line MD041 -->
[![Build Status for the Core backend](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml/badge.svg)](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml?query=workflow%3A%22CI+-+Python+application%22++)
[![PyPI version](https://badge.fury.io/py/xdsl.svg)](https://badge.fury.io/py/xdsl)
[![Downloads](https://static.pepy.tech/badge/xdsl)](https://www.pepy.tech/projects/xdsl)
[![Downloads](https://static.pepy.tech/badge/xdsl/week)](https://pepy.tech/project/xdsl)
[![Code Coverage](https://codecov.io/gh/xdslproject/xdsl/main/graph/badge.svg)](https://codecov.io/gh/xdslproject/xdsl)
[![Zulip Status](https://img.shields.io/badge/chat-on%20zulip-%2336C5F0)](https://xdsl.zulipchat.com)

# xDSL: A Python-native SSA Compiler Framework

[xDSL](http://www.xdsl.dev) is a Python-native compiler framework built around
SSA-based intermediate representations (IRs). Users of xDSL build a compiler by
assembling predefined domain-specific IRs and, optionally, defining their own custom IRs. xDSL uses multi-level IRs, meaning
that during the compilation process, a program will be lowered through several
of these IRs. This allows the implementation of abstraction-specific
optimization passes, similar to the structure of common DSL compilers (such as
Devito, Psyclone, and Firedrake). To simplify the writing of these passes, xDSL
uses a uniform data structure based on SSA, basic blocks, and regions, which
additionally enables the writing of generic passes.

The design of xDSL is influenced by [MLIR](https://mlir.llvm.org/), a compiler
framework developed in C++, that is part of the LLVM project. An inherent
advantage of a close design is the easy interaction between the two frameworks,
making it possible to translate abstractions and programs back and forth. This
results in one big SSA-based abstraction ecosystem that can be worked with
through Python, making analysis through simple scripting languages possible.
Additionally, xDSL can leverage MLIR's code generation and low-level
optimization capabilities.

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

*Note:* This version of xDSL is validated against a specific MLIR version,
interoperability with other versions may result in problems. The supported
MLIR version is 20.1.7.

### Subprojects With Extra Dependencies

xDSL has a number of subprojects, some of which require extra dependencies.
In order to keep the set of dependencies to a minimum, these extra dependencies have to be
specified explicitly. To install these, use:

``` bash
pip install xdsl[gui,jax,riscv]
```

To install the testing/development dependencies, use:

``` bash
pip install xdsl[dev]
```

These may be useful for projects wanting to replicate the xDSL testing setup.

## Getting Started

Check out the dedicated [Getting Started guide](GETTING_STARTED.md) for a comprehensive tutorial.

To get familiar with xDSL, we recommend starting with our Jupyter notebooks. The
notebooks consist of examples and documentation concerning the core xDSL data
structures and the xDSL's Python-embedded abstraction definition language, as
well as examples of implementing custom compilers, like a database compiler.
There also exists a small documentation showing how to connect xDSL with MLIR
for users interested in that use case.

- [A Database example](https://xdsl.dev/xdsl/lab/index.html?path=database_example.ipynb)
- [A simple introduction](https://xdsl.dev/xdsl/lab/index.html?path=tutorial.ipynb)
- [A DSL for defining new IRs](https://xdsl.dev/xdsl/lab/index.html?path=irdl.ipynb)
- [Connecting xDSL with MLIR](docs/guides/mlir_interoperation.md)

We provide a Makefile containing a lot of common tasks, which might provide
an overview of common actions.

## xDSL Developer Setup

To contribute to the development of xDSL follow the subsequent steps.

### Developer Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management of xDSL.
Getting started documentation can be found [here](https://docs.astral.sh/uv/getting-started/),
and is also printed by the `make uv-installed` and `make venv` targets if it
is not already installed on your system.

```bash
git clone https://github.com/xdslproject/xdsl.git
cd xdsl
# set up the venv and install everything
make venv
```

To make a custom mlir-opt available in the virtual environment, set the `XDSL_MLIR_OPT_PATH` variable when running `make venv`, like so:

``` bash
XDSL_MLIR_OPT_PATH=/PATH/TO/LLVM/BUILD/bin/mlir-opt make venv
```

#### If you can't use `uv`

For some systems and workflows, changing to a new dependency management system
may be inconvenient, impractical, or impossible. If this is the case for you,
xDSL can still be installed using `pip`.

To create the required virtual environment (the equivalent of `make venv`):

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

The following commands can then be run using this virtual environment rather
than `uv` by running `source venv/bin/activate` when starting a new shell, then
eliding the `uv run` prefix from the commands. For example, to run the commands
in the following testing section:

```bash
source venv/bin/activate
pytest
lit tests/filecheck
```

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

Benchmarks for the project are tracked in the <https://github.com/xdslproject/xdsl-bench>
repository. These run automatically every day on the main branch, reporting
their results to <https://xdsl.dev/xdsl-bench/>. However, they can also be run
manually by cloning the repository and pointing the submodule at your
feature branch to benchmark.

### Formatting and Typechecking

All python code used in xDSL uses [ruff](https://docs.astral.sh/ruff/formatter/) to
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

Furthermore, all python code must run through [pyright](https://github.com/microsoft/pyright)
without errors. Pyright can be run on all staged files through the
makefile using `make pyright`.

> [!IMPORTANT]
>
> #### Experimental Pyright Features
>
> xDSL currently relies on an experimental feature of Pyright called TypeForm
> TypeForm is [in discussion](https://discuss.python.org/t/pep-747-typeexpr-type-hint-for-a-type-expression/55984) and will likely land in some future version of Python.
>
> For xDSL to type check correctly using Pyright, please add this to your `pyproject.toml`:
>
> ```toml
> [tool.pyright]
> enableExperimentalFeatures = true
> ```

### Discussion

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com), kindly supported by community hosting from [Zulip](https://zulip.com/).
