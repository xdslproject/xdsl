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
  - [Testing](#testing)
  - [Formatting](#formatting)

## Installation

To use xDSL as part of a larger project for developing your own compiler,
just install [xDSL via pip](https://pypi.org/project/xdsl/):

```bash
pip install xdsl
```

*Note:* This version of xDSL is validated against a specific MLIR version,
interoperability with other versions may result in problems. The supported
MLIR version is commit `98e674c9f16d677d95c67bc130e267fae331e43c`.

### Subprojects With Extra Dependencies

xDSL has a number of subprojects, some of which require extra dependencies.
In order to keep the set of dependencies ot a minimum, these extra dependencies have to be
specified explicitly. To install these, use:

``` bash
pip install xdsl[gui,jax,riscv,wgpu,onnx]
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
- [Connecting xDSL with MLIR](docs/mlir_interoperation.md)

We provide a Makefile containing a lot of common tasks, which might provide
an overview of common actions.

## xDSL Developer Setup

To contribute to the development of xDSL follow the subsequent steps.

### Developer Installation

```bash
git clone https://github.com/xdslproject/xdsl.git
cd xdsl
# set up the venv and install everything
make venv
# activate the venv
source venv/bin/activate
```

### Testing

The xDSL project uses pytest unit tests and LLVM-style filecheck tests. They can
be executed from the root directory:

```bash
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck

# run all tests using makefile
make tests
```

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
ruff format
```

Furthermore, all python code must run through [pyright](https://github.com/microsoft/pyright)
without errors. Pyright can be run on all staged files through the
makefile using `make pyright`.

### Discussion

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com), kindly supported by community hosting from [Zulip](https://zulip.com/).
