# xDSL: A common ecosystem for DSL development

[![Build Status for the Core backend](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml/badge.svg)](https://github.com/xdslproject/xdsl/actions/workflows/ci-core.yml?query=workflow%3A%22CI+-+Python+application%22++)
[![PyPI version](https://badge.fury.io/py/xdsl.svg)](https://badge.fury.io/py/xdsl)
[![Code Coverage](https://codecov.io/gh/xdslproject/xdsl/main/graph/badge.svg)](https://codecov.io/gh/xdslproject/xdsl)

[![Zulip Status](https://img.shields.io/badge/chat-on%20zulip-%2336C5F0)](https://xdsl.zulipchat.com)

[xDSL](http://www.xdsl.dev) is a compiler toolkit natively developed in Python.
It allows users to reuse existing domain-specific intermediate representations
(IRs), or to define their own custom IRs. xDSL uses multi-level IRs, meaning
that during the compilation process, a program will be lowered through several
of these IRs. This allows the implementation of abstraction-specific
optimization passes, similar to the structure of common DSL compilers (such as
Devito, Psyclone, and Firedrake). To simplify the writing of these passes, xDSL
uses a uniform data structure based on SSA, basic blocks, and regions, which
additionally enables the writing of generic passes.

This design is influenced by [MLIR](https://mlir.llvm.org/), a compiler
framework developed in C++, that is part of the LLVM project. An inherent
advantage of a close design is the easy interaction between the two frameworks,
making it possible to translate abstractions and programs back and forth. This
results in one big SSA-based abstraction ecosystem that can be worked with
through Python, making analysis through simple scripting languages possible.
Additionally, xDSL can leverage MLIR's code generation and low-level
optimization capabilities.

## Prerequisites

To install xDSL you can either clone the GitHub repository and install the requirements by following:

### Installation for Users

```bash
pip install xdsl
```

### Installation for Developers
```bash
git clone https://github.com/xdslproject/xdsl.git
pip install -e .
pip install -e .[extras]
```

## Testing

This project includes pytest unit test and LLVM-style filecheck tests. They can
be executed using to following commands from within the root directory of the
project:

```bash
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck
```

## Formatting

All python code used in xDSL uses [yapf](https://github.com/google/yapf) to
format the code in a uniform manner.

To automate the formatting within vim, one can use
https://github.com/vim-autoformat/vim-autoformat and trigger a `:Autoformat` on
save.

Please refer to the [MLIR
Interoperation](https://github.com/xdslproject/xdsl/docs/mlir_interoperation.md)
markdown file for further instructions on that.
