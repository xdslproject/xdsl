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
while still accessing MLIR's powerful optimization and code generation pipeline.
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
- [Contributing](#contributing)

## Installation

To use xDSL as part of a larger project for developing your own compiler,
just install [xDSL via pip](https://pypi.org/project/xdsl/):

```bash
pip install xdsl
```

*Note:* This version of xDSL is validated against a specific MLIR version,
interoperability with other versions is not guaranteed. The supported
MLIR version is 21.1.1.

### Subprojects With Extra Dependencies

xDSL has a number of subprojects, some of which require extra dependencies.
To keep the set of dependencies to a minimum, these extra dependencies have to
be specified explicitly:

```bash
pip install xdsl[gui]   # or [jax], [riscv], [dev]
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

## Contributing

For developer setup, code style guidelines, testing, and other contribution
information, see [CONTRIBUTING.md](CONTRIBUTING.md).

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com),
kindly supported by community hosting from [Zulip](https://zulip.com/).
