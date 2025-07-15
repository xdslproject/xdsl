import pytest

from xdsl.context import Context
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError


@irdl_op_definition
class UnknownOp(IRDLOperation):
    name = "test.unknown"
    ops = var_operand_def()
    res = var_result_def()


def check_error(prog: str, line: int, column: int, message: str):
    ctx = Context()
    ctx.load_op(UnknownOp)

    parser = Parser(ctx, prog)
    with pytest.raises(ParseError, match=message) as e:
        parser.parse_operation()

    _, l, c = e.value.span.get_location()

    assert (l, c) == (line, column)


def test_parser_missing_equal():
    """Test a missing equal sign error."""
    ctx = Context()
    ctx.load_op(UnknownOp)

    prog = """
"test.unknown"() ({
  %0 "test.unknown"() : () -> !i32
}) : () -> ()
"""
    check_error(prog, 3, 6, "Expected '=' after operation result list")


def test_parser_redefined_value():
    """Test an SSA value redefinition error."""
    ctx = Context()
    ctx.load_op(UnknownOp)

    prog = """
"test.unknown"() ({
  %val = "test.unknown"() : () -> i32
  %val = "test.unknown"() : () -> i32
}) : () -> ()
"""
    check_error(prog, 4, 3, "SSA value %val is already defined")


def test_parser_missing_operation_name():
    """Test a missing operation name error."""
    ctx = Context()
    ctx.load_op(UnknownOp)

    prog = """
"test.unknown"() ({
  %val =
}) : () -> ()
"""
    check_error(prog, 4, 1, "operation name expected")
