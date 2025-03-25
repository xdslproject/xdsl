---
title: Overview
layout: page
---

This is a series of introductory notebooks for the basics of compiler development in xDSL.

It uses [Marimo](https://marimo.app/), a modern Python notebook that runs in your browser.
It keeps a dependency graph between cells in the notebook, re-running all dependencies of a cell that you change.
Note that the notebook automatically installs xDSL in an environment in your tab when you open a notebook, so you may need to wait a little for it to do that and re-run all the cells.

## Manipulating IR

This series introduces MLIR IR, and xDSL APIs to traverse it, define your own IR constructs, and implement your rewrites.

## RISC-V Dialects

This series of notebooks introduces the RISC-V dialects, which let us represent assembly-level concepts as MLIR IR.
