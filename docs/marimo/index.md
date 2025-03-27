---
title: Overview
layout: page
---

These are introductory notebooks for the basics of compiler development in xDSL.
We currently provide two series of notebooks on manipulating IR and working with assembly-level dialects.

We use [Marimo](https://marimo.app/), a modern Python notebook that runs in your browser.
It keeps a dependency graph between cells in the notebook, re-running all dependencies of a cell that you change.
Note that each notebook automatically installs xDSL in an environment in your browser tab when you open it, so you may need to wait a little for installation to complete and re-run all the cells.

## Manipulating IR

This series introduces MLIR IR, and xDSL APIs to traverse it, define your own IR constructs, and implement your rewrites.

## RISC-V Dialects

This series of notebooks introduces the RISC-V dialects, which let us represent assembly-level concepts as MLIR IR.
