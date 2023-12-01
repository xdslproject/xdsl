import pytest

from xdsl.ir import MLContext
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError


@irdl_op_definition
class UnkownOp(IRDLOperation):
    name = "unknown"
    ops: VarOperand = var_operand_def(AnyAttr())
    res: VarOpResult = var_result_def(AnyAttr())


def check_error(prog: str, line: int, column: int, message: str):
    ctx = MLContext()
    ctx.load_op(UnkownOp)

    parser = Parser(ctx, prog)
    with pytest.raises(ParseError, match=message) as e:
        parser.parse_operation()

    assert e.value.span.get_line_col() == (line, column)


def test_parser_missing_equal():
    """Test a missing equal sign error."""
    ctx = MLContext()
    ctx.load_op(UnkownOp)

    prog = """
"unknown"() ({
  %0 "unknown"() : () -> !i32
}) : () -> ()
"""
    check_error(prog, 3, 5, "Expected '=' after operation result list")


def test_parser_redefined_value():
    """Test an SSA value redefinition error."""
    ctx = MLContext()
    ctx.load_op(UnkownOp)

    prog = """
"unknown"() ({
  %val = "unknown"() : () -> i32
  %val = "unknown"() : () -> i32
}) : () -> ()
"""
    check_error(prog, 4, 2, "SSA value %val is already defined")


def test_parser_missing_operation_name():
    """Test a missing operation name error."""
    ctx = MLContext()
    ctx.load_op(UnkownOp)

    prog = """
"unknown"() ({
  %val =
}) : () -> ()
"""
    check_error(prog, 4, 0, "operation name expected")
