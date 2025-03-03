#!/usr/bin/env python3
"""Helper functions for benchmarking xDSL."""

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.parser import Parser as XdslParser
from xdsl.printer import Printer


def get_context() -> Context:
    """Get an xDSL context."""
    return Context(allow_unregistered=True)


def parse_module(context: Context, contents: str) -> ModuleOp:
    """Parse a MLIR file as a module."""
    parser = XdslParser(context, contents)
    return parser.parse_module()


def print_op(op: Operation) -> None:
    """Print and xDSL operation"""
    Printer().print_op(op)
