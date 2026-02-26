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
- [Discussion](#discussion)

## Installation

To contribute to xDSL, follow the [xDSL Developer Setup Guide](CONTRIBUTING.md).

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

*Note:* This version of xDSL is validated against a specific MLIR version,
interoperability with other versions is not guaranteed. The supported
MLIR version is 21.1.1.

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

### Subprojects With Extra Dependencies

xDSL has a number of subprojects, some of which require extra dependencies.
To keep the set of dependencies to a minimum, these extra dependencies have to
be specified explicitly, e.g. by using:

``` bash
pip install xdsl[gui] # or [jax], [riscv]
```

## Getting Started

Check out the dedicated [Getting Started guide](https://docs.xdsl.dev)
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

## Discussion

You can also join the discussion at our [Zulip chat room](https://xdsl.zulipchat.com),
kindly supported by community hosting from [Zulip](https://zulip.com/).

### Citing xDSL

xDSL itself has been published as
[xDSL: Sidekick Compilation for SSA-Based Compilers](https://2025.cgo.org/details/cgo-2025-papers/32/xDSL-Sidekick-Compilation-for-SSA-Based-Compilers)
at CGO 2025, which is the paper that should be cited when referencing xDSL the
tool:

```bibtex
@inproceedings{fehr_xdsl_2025,
  address = {Las Vegas NV USA},
  title = {{xDSL}: {Sidekick} {Compilation} for {SSA}-{Based} {Compilers}},
  isbn = {979-8-4007-1275-3},
  shorttitle = {{xDSL}},
  url = {https://dl.acm.org/doi/10.1145/3696443.3708945},
  doi = {10.1145/3696443.3708945},
  language = {en},
  urldate = {2026-01-26},
  booktitle = {Proceedings of the 23rd {ACM}/{IEEE} {International} {Symposium} on {Code} {Generation} and {Optimization}},
  publisher = {ACM},
  author = {Fehr, Mathieu and Weber, Michel and Ulmann, Christian and Lopoukhine, Alexandre and Lücke, Martin Paul and Degioanni, Théo and Vasiladiotis, Christos and Steuwer, Michel and Grosser, Tobias},
  month = mar,
  year = {2025},
  pages = {179--192},
}
```

Other published works that use xDSL internally (feel free to open a PR to extend
this list):

- [Stencil-HMLS: A multi-layered approach to the automatic optimisation of stencil codes on FPGA](https://dl.acm.org/doi/abs/10.1145/3624062.3624543)
- [Fortran performance optimisation and auto-parallelisation by leveraging MLIR-based domain specific abstractions in Flang](https://dl.acm.org/doi/abs/10.1145/3624062.3624167)
- [A shared compilation stack for distributed-memory parallelism in stencil DSLs](https://dl.acm.org/doi/abs/10.1145/3620666.3651344)
- [A Multi-level Compiler Backend for Accelerated Micro-kernels Targeting RISC-V ISA Extensions](https://dl.acm.org/doi/abs/10.1145/3696443.3708952)
- [The Configuration Wall: Characterization and Elimination of Accelerator Configuration Overhead](https://dl.acm.org/doi/abs/10.1145/3760250.3762225)
- [First-Class Verification Dialects for MLIR](https://dl.acm.org/doi/abs/10.1145/3729309)
- [Integrating Quantum Software Tools with(in) MLIR](https://arxiv.org/abs/2601.02062)
- [eqsat: An Equality Saturation Dialect for Non-destructive Rewriting](https://arxiv.org/abs/2505.09363)
- [An MLIR pipeline for offloading Fortran to FPGAs via OpenMP](https://dl.acm.org/doi/full/10.1145/3731599.3767485)
- [Development of a Custom Compilation Workflow With MLIR Leveraging OpenFPM to Accelerate Particle Simulations](https://esim-project.eu/files/Images/people/chair-cc/theses/2508_Cornel_MA.pdf)
- [Quantum Oracle Synthesis from HDL Designs via Multi Level Intermediate Representation](https://re.public.polimi.it/handle/11311/1301614)
