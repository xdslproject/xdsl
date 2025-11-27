---
title: Getting Started
hide:
  - navigation
  - toc
---

## Opening your first Pull Request

If you'd like to get familiar with xDSL, and are not sure how to get started, this
document will walk you through opening your first pull request in the repository.
When a new member joins the team, we ask them to perform the following series of steps
on the first day, and it takes complete novices at most a couple of hours from starting
this task to opening a PR.
One of the ways to make the time as short as possible is being in the same room as
someone familiar with the project.
The next best thing to asking questions in person is reaching out on [our Zulip chatroom](https://xdsl.zulipchat.com).
Please don't hesitate to post to the Beginners channel with any questions about the framework.

## Pre-Requisites

Before starting the coding part of this tutorial please make sure that xDSL is correctly
installed.
The instructions on setting it up locally are in the [README](https://github.com/xdslproject/xdsl).
In particular, please follow the steps in **Developer Installation**, and
**Formatting and Typechecking**.
Please run `make tests` in the terminal.
If they pass, you're ready to start!
(If they don't, and you're not sure why, please reach out on the Zulip.)

## The Mission

The aim of this tutorial is to add an optimisation to our RISC-V representation.
Here are some examples of existing optimisations at a high level:

``` txt
x + 0 -> x
2 + 2 -> 4
x * 1 -> x
```

In xDSL, this class of optimisation is called "canonicalization".
This kind of optimisation is common in compilers, and can broadly be described as a
transformation that reduces the complexity of the IR, making it easier for further
analysis to improve the program.
For further information about canonicalization, please take a look at [MLIR's documentation](https://mlir.llvm.org/docs/Canonicalization/).

## A Short Guide to xDSL

We have a short series of notebooks describing the APIs necessary to build
representations of code, and transformations on it.
The notebooks in [docs/Toy](https://github.com/xdslproject/xdsl/tree/main/docs/Toy) are
the best place to start, we recommend first looking at them, and then at the
implementation of the Toy compiler itself: [Dialect](https://github.com/xdslproject/xdsl/tree/main/docs/Toy/toy/dialects/toy.py),
[Canonicalization](https://github.com/xdslproject/xdsl/tree/main/docs/Toy/toy/rewrites/optimise_toy.py).

## Adding Your Own Canonicalization Pattern

Here are some examples of first PRs from existing contributors:
[#1566](https://github.com/xdslproject/xdsl/pull/1566),
[#1567](https://github.com/xdslproject/xdsl/pull/1567).

The first step is finding a missing optimisation pattern.
You're welcome to come up with your own, or do one of the following:

- `ori %r, i -> li (%r or i)` when `%r` is defined by an immediate
- `xori %r, i -> li (%r xor i)` when `%r` is defined by an immediate

The patterns are defined in
[xdsl/transforms/canonicalization_patterns/riscv.py](https://github.com/xdslproject/xdsl/blob/main/xdsl/transforms/canonicalization_patterns/riscv.py).

We try to put the patterns roughly in the order of the operations that they operate on
in the `riscv` dialect definition file:
[xdsl/dialects/riscv.py](https://github.com/xdslproject/xdsl/blob/main/xdsl/dialects/riscv.py).

xDSL uses two ways to test its code,
[pytest](https://docs.pytest.org/en/8.0.x/) and
[lit](https://llvm.org/docs/CommandGuide/lit.html).

We prefer `lit` tests in general to test everything that isn't the Python API itself,
and the `riscv` canonnicalization tests are also a lit test, in
[tests/filecheck/backend/riscv/canonicalize.mlir](https://github.com/xdslproject/xdsl/blob/main/tests/filecheck/backend/riscv/canonicalize.mlir).

All `lit` tests have a `RUN:` comment that includes the command to run in the terminal
to exercise it.
You may find it useful when developing to copy/paste the command to see the output
directly, instead of running the test with `lit`.
In this case, the command is `xdsl-opt --split-input-file -p canonicalize tests/filecheck/backend/riscv/canonicalize.mlir`.

Please follow the existing test structure to add a test for your rewrite.

Once you're satisfied with the result, please open a PR!
