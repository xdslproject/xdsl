---
title: Overview
layout: page
---

These are introductory notebooks for the basics of compiler development in xDSL.
We currently provide two series of notebooks on manipulating IR and working with
assembly-level dialects.

We use [Marimo](https://marimo.app/), a modern Python notebook that runs in your browser.
It keeps a dependency graph between cells in the notebook, re-running all dependencies
of a cell that you change.
Note that each notebook automatically installs xDSL in an environment in your browser
tab when you open it, so you may need to wait a little for installation to complete and
re-run all the cells.

## Manipulating IR

This series introduces MLIR IR, and xDSL APIs to traverse it, define your own IR
constructs, and implement your rewrites.

1. [MLIR IR](mlir_ir.py)
1. [Traversing IR](traversing_ir.py)
1. [Builders](builders.py)
1. [IR Generation](ir_gen.py)
1. [Rewrites](pattern_rewrites.py)
1. [Rewrite Exercises](rewrite_exercises.py)
1. [Defining Dialects](defining_dialects.py)
1. [Pattern Description Language (PDL)](pdl.py)

## RISC-V Dialects

This series of notebooks introduces the RISC-V dialects, which let us represent
assembly-level concepts as MLIR IR.

1. [RISC-V Dialects](riscv_dialects.py)
1. [Compiling `linalg` to Snitch](linalg_snitch.py)

## Other Projects

1. [Equality Saturation](eqsat.py)
