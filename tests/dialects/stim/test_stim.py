from __future__ import annotations

from io import StringIO

import pytest

from xdsl.context import MLContext
from xdsl.dialects import stim
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.dialects.stim.ops import QubitAttr, QubitMappingAttr
from xdsl.dialects.test import Test
from xdsl.ir import Attribute, Operation
from xdsl.parser import Parser
from xdsl.printer import Printer

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_roundtrip(program: str):
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    """Check that the given program roundtrips exactly (including whitespaces)."""
    parser = Parser(ctx, program)
    ops: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops.append(op)

    res_io = StringIO()
    printer = Printer(stream=res_io)
    for op in ops[:-1]:
        printer.print_op(op)
        printer.print("\n")
    printer.print_op(ops[-1])

    assert program == res_io.getvalue()


def check_equivalence(program1: str, program2: str):
    """Check that the given programs are structurally equivalent."""
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    parser = Parser(ctx, program1)
    ops1: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops1.append(op)

    parser = Parser(ctx, program2)
    ops2: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops2.append(op)

    mod1 = ModuleOp(ops1)
    mod2 = ModuleOp(ops2)

    assert mod1.is_structurally_equivalent(mod2), str(mod1) + "\n!=\n" + str(mod2)


def check_attribute(attr: Attribute, program: str):
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    ctx.load_dialect(Test)
    parser = Parser(ctx, program)
    assert str(attr) == str(parser.parse_attribute())


def test_qubit_attribute():
    attr = QubitAttr(0)
    program = "!stim.qubit<0>"
    check_attribute(attr, program)


def test_qubit_coord_attribute():
    coords = ArrayAttr([IntAttr(0), IntAttr(0)])
    qubitname = QubitAttr(IntAttr(0))
    attr = QubitMappingAttr(coords, qubitname)
    program = "#stim.qubit_coord<(0,0), !stim.qubit<0>>"
    check_attribute(attr, program)

