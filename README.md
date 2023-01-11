[![Build Status for the Core backend](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml/badge.svg)](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml?query=workflow%3A%22CI+-+Python+application%22++)
[![PyPI version](https://badge.fury.io/py/xdsl.svg)](https://badge.fury.io/py/xdsl)
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
- [Using xDSL](#using-xdsl)
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

## Using xDSL

To use xDSL we recommend following the [xDSL Developer Setup](#xdsl-developer-setup)
to clone the repository including the notebooks.
The following tutorials will present xDSL basic concepts, how to
use its irdl dialect to define new dialects in a user-friendly way, and how to
work with both xDSL and MLIR.

- [A simple introduction](https://xdsl.dev/xdsl/retro/notebooks/?path=docs/tutorial.ipynb)
- [A DSL for defining new IRs](https://xdsl.dev/xdsl/retro/notebooks/?path=docs/irdl.ipynb)
- [Connecting xDSL with MLIR](docs/mlir_interoperation.md)

## xDSL Developer Setup

To contribute to the development of xDSL follow the subsequent steps.

### Developer Installation

```bash
git clone https://github.com/xdslproject/xdsl.git
pip install --editable .
# Optional installation of extra requirements
pip install --requirement requirements-optional.txt
```

### Testing

The xDSL project uses pytest unit tests and LLVM-style filecheck tests. They can
be executed from the root directory:

```bash
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck
```

### Formatting

All python code used in xDSL uses [yapf](https://github.com/google/yapf) to
format the code in a uniform manner.

To automate the formatting within vim, one can use
https://github.com/vim-autoformat/vim-autoformat and trigger a `:Autoformat` on
save.
