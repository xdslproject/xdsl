from __future__ import annotations
from typing import Annotated

from xdsl.ir import MLContext
from xdsl.irdl import AnyAttr, VarOpResult, VarOperand, irdl_op_definition, Operation
from xdsl.parser import Parser, ParserError
from pytest import raises


@irdl_op_definition
class UnkownOp(Operation):
    name = "unknown"
    ops: Annotated[VarOperand, AnyAttr()]
    res: Annotated[VarOpResult, AnyAttr()]


def check_error(prog: str, line: int, column: int, message: str):
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    parser = Parser(ctx, prog)
    with raises(ParserError) as e:
        parser.parse_op()

    assert e.value.pos
    assert e.value.pos.line is line
    assert e.value.pos.column is column
    assert e.value.message == message


def test_parser_missing_equal():
    """Test a missing equal sign error."""
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    prog = \
"""
unknown() {
  %0 : !i32 unknown()
}
"""
    check_error(prog, 3, 13, "'=' expected, got u")


def test_parser_redefined_value():
    """Test an SSA value redefinition error."""
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    prog = \
"""
unknown() {
  %val : !i32 = unknown()
  %val : !i32 = unknown()
}
"""
    check_error(prog, 4, 3, "SSA value val is already defined")


def test_parser_missing_operation_name():
    """Test a missing operation name error."""
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    prog = \
"""
unknown() {
  %val : !i32 = 
}
"""
    check_error(prog, 4, 1, "operation name expected")


def test_parser_missing_attribute():
    """Test a missing attribute error."""
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    prog = \
"""
unknown() {
  %val : i32 = unknown()
}
"""
    check_error(prog, 3, 10, "attribute expected")
