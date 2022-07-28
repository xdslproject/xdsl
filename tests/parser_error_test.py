from __future__ import annotations

from xdsl.ir import MLContext
from xdsl.irdl import AnyAttr, VarOperandDef, VarResultDef, irdl_op_definition, Operation
from xdsl.parser import Parser, ParserError
from pytest import raises


@irdl_op_definition
class UnkownOp(Operation):
    name = "unkown"
    ops = VarOperandDef(AnyAttr())
    res = VarResultDef(AnyAttr())


def test_parser_error():
    """Test the error in case of a parsing error."""
    ctx = MLContext()
    ctx.register_op(UnkownOp)

    prog = \
"""
unkown() {
  %0 : !i32 unknown()
}
"""
    parser = Parser(ctx, prog)
    with raises(ParserError) as e:
        parser.parse_op()

    assert e.value.pos is not None
    assert e.value.pos.line == 3
    assert e.value.pos.column == 13

    assert e.value.message == "'=' expected"
