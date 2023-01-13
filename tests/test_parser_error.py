from __future__ import annotations
from typing import Annotated

from xdsl.ir import MLContext, OpResult, SSAValue
from xdsl.irdl import AnyAttr, VarOperandDef, VarResultDef, irdl_op_definition, Operation
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
    with raises(ParseError) as e:
        parser.must_parse_operation()

    assert e.value.span
    assert e.value.span.get_line_col() == (line, column)
    assert any(message in ex.error.msg for ex in e.value.history.iterate())


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
    check_error(prog, 3, 13, "Operation definitions expect an `=` after op-result-list!")


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
    check_error(prog, 4, 2, "SSA value %val is already defined")


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
    check_error(prog, 3, 13, "Expected an operation name here")


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
